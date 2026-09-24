"""One-token KV-cached decoder matching ammon/llama3.cuda forward()."""

from __future__ import annotations

import math
import time

from .checkpoint import Checkpoint, Config

ROPE_THETA = 10000.0
RMS_EPS = 1e-5

_TORCH = None


def _torch():
    global _TORCH
    if _TORCH is None:
        try:
            import torch
            import torch.nn.functional as F
        except ImportError as exc:
            raise ImportError("PyTorch is required for the decoder; pip install torch") from exc
        _TORCH = (torch, F)
    return _TORCH


def configure_precision(device: str) -> list[str]:
    """Strict FP32: no TF32 anywhere. Fails loudly if CUDA was asked for but is missing."""
    torch, _ = _torch()
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"--device {device} requested but torch {torch.__version__} has no usable CUDA "
            f"(torch.version.cuda={torch.version.cuda}); use the CUDA venv or --device cpu"
        )
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return []


def _rmsnorm(x, weight):
    torch, _ = _torch()
    ss = torch.rsqrt(x.square().mean() + RMS_EPS)
    return weight * (ss * x)


def _rope_tables(cfg: Config, device):
    """cos/sin of pos * theta^(-(i % head_size)/head_size) for every pos, i even; built once."""
    torch, _ = _torch()
    i = torch.arange(0, cfg.dim, 2, device=device, dtype=torch.int64)
    head_dim = (i % cfg.head_size).to(torch.float32)
    freq = 1.0 / torch.pow(
        torch.tensor(ROPE_THETA, dtype=torch.float32, device=device), head_dim / float(cfg.head_size)
    )
    pos = torch.arange(cfg.max_seq_len, device=device, dtype=torch.float32)
    val = pos[:, None] * freq[None, :]
    return torch.cos(val).contiguous(), torch.sin(val).contiguous()


class Decoder:
    """FP32 one-token decoder. Weights, RoPE tables, and KV live on `device`."""

    def __init__(self, checkpoint: Checkpoint, device: str = "cuda"):
        torch, _ = _torch()
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"device {device} requested but CUDA is unavailable")
        self.device = torch.device(device)
        self.config: Config = checkpoint.config
        self._load_weights(checkpoint)
        cfg = self.config
        # One buffer per layer: under torch.compile, a row write into a single [L, seq, kv]
        # tensor is functionalized into a full-cache select_scatter copy per layer.
        shape = (cfg.max_seq_len, cfg.kv_dim)
        f32 = torch.float32
        self.k_cache = [torch.zeros(shape, dtype=f32, device=self.device) for _ in range(cfg.n_layers)]
        self.v_cache = [torch.zeros(shape, dtype=f32, device=self.device) for _ in range(cfg.n_layers)]
        self.rope_cos, self.rope_sin = _rope_tables(cfg, self.device)
        # c10::complex mul is (q0*c - q1*s, q0*s + q1*c): the llama3.cuda RoPE formula.
        self.rope_complex = torch.complex(self.rope_cos, self.rope_sin)
        self._compiled = None

    def _load_weights(self, checkpoint: Checkpoint) -> None:
        torch, _ = _torch()
        cfg = checkpoint.config
        layout = checkpoint.layout
        host = torch.frombuffer(checkpoint.weights, dtype=torch.float32)
        blob = host.to(self.device) if self.device.type != "cpu" else host.clone()

        def view(off: int, shape: tuple[int, ...]):
            n = math.prod(shape)
            return blob[off : off + n].view(*shape)

        L, dim, hidden, kv = cfg.n_layers, cfg.dim, cfg.hidden_dim, cfg.kv_dim
        self.tok_emb = view(layout.token_embedding, (cfg.vocab_size, dim))
        self.rms_final = view(layout.rms_final_weight, (dim,))
        self.wcls = view(layout.wcls, (cfg.vocab_size, dim))
        # llama2.c stores each weight kind contiguously across layers: [L, out, in].
        self.rms_att_all = view(layout.rms_att_weight, (L, dim))
        self.wq_all = view(layout.wq, (L, dim, dim))
        self.wk_all = view(layout.wk, (L, kv, dim))
        self.wv_all = view(layout.wv, (L, kv, dim))
        self.wo_all = view(layout.wo, (L, dim, dim))
        self.rms_ffn_all = view(layout.rms_ffn_weight, (L, dim))
        self.w1_all = view(layout.w1, (L, hidden, dim))
        self.w2_all = view(layout.w2, (L, dim, hidden))
        self.w3_all = view(layout.w3, (L, hidden, dim))
        self.rms_att = list(self.rms_att_all.unbind(0))
        self.wq = list(self.wq_all.unbind(0))
        self.wk = list(self.wk_all.unbind(0))
        self.wv = list(self.wv_all.unbind(0))
        self.wo = list(self.wo_all.unbind(0))
        self.rms_ffn = list(self.rms_ffn_all.unbind(0))
        self.w1 = list(self.w1_all.unbind(0))
        self.w2 = list(self.w2_all.unbind(0))
        self.w3 = list(self.w3_all.unbind(0))

    def _attention(self, q, k, v, t: int):
        """q [dim], k/v [t, kv_dim] -> [dim]; softmax(q.k / sqrt(head_size)) @ v per head."""
        torch, _ = _torch()
        cfg = self.config
        qh = q.view(cfg.n_kv_heads, cfg.kv_mul, cfg.head_size)
        kh = k.view(t, cfg.n_kv_heads, cfg.head_size)
        vh = v.view(t, cfg.n_kv_heads, cfg.head_size)
        scores = torch.matmul(qh, kh.permute(1, 2, 0)) / math.sqrt(cfg.head_size)
        att = torch.softmax(scores, dim=-1)
        return torch.matmul(att, vh.permute(1, 0, 2)).reshape(cfg.dim)

    def _forward_eager(self, token: int, pos: int):
        """Eager forward: Python-int token/pos (views, no syncs), K/V written in place."""
        torch, F = _torch()
        cfg = self.config
        half_kv = cfg.kv_dim // 2
        rope = self.rope_complex[pos]
        rope_k = rope[:half_kv]
        t = pos + 1
        x = self.tok_emb[token]
        for layer in range(cfg.n_layers):
            xb = _rmsnorm(x, self.rms_att[layer])
            q = torch.mv(self.wq[layer], xb)
            q = torch.view_as_real(torch.view_as_complex(q.view(-1, 2)) * rope).view(cfg.dim)
            k = torch.view_as_complex(torch.mv(self.wk[layer], xb).view(-1, 2))
            torch.mul(k, rope_k, out=torch.view_as_complex(self.k_cache[layer][pos].view(-1, 2)))
            torch.mv(self.wv[layer], xb, out=self.v_cache[layer][pos])
            xb = self._attention(q, self.k_cache[layer][:t], self.v_cache[layer][:t], t)
            x = x + torch.mv(self.wo[layer], xb)
            xb = _rmsnorm(x, self.rms_ffn[layer])
            hb = F.silu(torch.mv(self.w1[layer], xb)) * torch.mv(self.w3[layer], xb)
            x = x + torch.mv(self.w2[layer], hb)
        x = _rmsnorm(x, self.rms_final)
        return torch.mv(self.wcls, x)

    def _forward_compiled(self, token: int, pos: int):
        """Real-arithmetic RoPE forward for torch.compile (token/pos become SymInts)."""
        torch, F = _torch()
        cfg = self.config
        half = cfg.dim // 2
        half_kv = cfg.kv_dim // 2
        c = self.rope_cos[pos]
        s = self.rope_sin[pos]
        ck = c[:half_kv]
        sk = s[:half_kv]
        t = pos + 1
        x = self.tok_emb[token]
        for layer in range(cfg.n_layers):
            xb = _rmsnorm(x, self.rms_att_all[layer])
            q = torch.mv(self.wq_all[layer], xb).view(half, 2)
            k = torch.mv(self.wk_all[layer], xb).view(half_kv, 2)
            v = torch.mv(self.wv_all[layer], xb)
            q0, q1 = q[:, 0], q[:, 1]
            k0, k1 = k[:, 0], k[:, 1]
            q = torch.stack((q0 * c - q1 * s, q0 * s + q1 * c), dim=-1).view(cfg.dim)
            k = torch.stack((k0 * ck - k1 * sk, k0 * sk + k1 * ck), dim=-1).view(cfg.kv_dim)
            # select().copy_() keeps pos symbolic; `cache[layer, pos] = k` guards pos == N.
            self.k_cache[layer].select(0, pos).copy_(k)
            self.v_cache[layer].select(0, pos).copy_(v)
            xb = self._attention(q, self.k_cache[layer][:t], self.v_cache[layer][:t], t)
            x = x + torch.mv(self.wo_all[layer], xb)
            xb = _rmsnorm(x, self.rms_ffn_all[layer])
            hb = F.silu(torch.mv(self.w1_all[layer], xb)) * torch.mv(self.w3_all[layer], xb)
            x = x + torch.mv(self.w2_all[layer], hb)
        x = _rmsnorm(x, self.rms_final)
        return torch.mv(self.wcls, x)

    def compile_forward(self, warm_positions: int = 4) -> tuple[float, str | None]:
        """torch.compile (default mode, no CUDA graphs) the decode step; returns (seconds, error).

        Automatic dynamic shapes: weights stay static, and the first call with a new
        token/pos value turns those ints into SymInts. Tracing a few positions here builds
        the specialized and the dynamic graph before timing, so later positions never
        recompile. (dynamic=True would also symbolize every weight dim.)
        """
        torch, _ = _torch()
        start = time.perf_counter()
        try:
            compiled = torch.compile(self._forward_compiled, fullgraph=True)
            with torch.inference_mode():
                for pos in range(min(warm_positions, self.config.max_seq_len)):
                    compiled(2 + pos if pos else 1, pos)
            self.synchronize()
        except Exception as exc:  # noqa: BLE001 — compile is optional
            self._compiled = None
            return time.perf_counter() - start, f"{type(exc).__name__}: {exc}".splitlines()[0][:500]
        self._compiled = compiled
        return time.perf_counter() - start, None

    @property
    def compiled(self) -> bool:
        return self._compiled is not None

    def synchronize(self) -> None:
        torch, _ = _torch()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def forward_device(self, token: int, pos: int):
        """Forward only; returns the device-resident logits (no DtoH, no sync)."""
        torch, _ = _torch()
        fwd = self._compiled or self._forward_eager
        with torch.inference_mode():
            return fwd(int(token), int(pos))

    def step(self, token: int, pos: int):
        """Forward + full-logit DtoH; returns 1-D CPU float32 logits once the copy has landed.

        The DtoH (into pinned memory) plus one stream sync is the only per-step host sync.
        """
        torch, _ = _torch()
        logits = self.forward_device(token, pos)
        if self.device.type != "cuda":
            return logits
        host = torch.empty(logits.shape, dtype=logits.dtype, pin_memory=True)
        host.copy_(logits, non_blocking=True)
        torch.cuda.current_stream(self.device).synchronize()
        return host

    def tensor_slice(self, name: str, index: int = 0, count: int = 8) -> list[float]:
        table = {
            "tok_emb": self.tok_emb[index, :count],
            "rms_final": self.rms_final[:count],
            "wq0": self.wq[0].reshape(-1)[:count],
        }
        return [float(x) for x in table[name].detach().to("cpu")]


def disable_tf32() -> None:
    configure_precision("cpu")
