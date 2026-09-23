"""One-token KV-cached decoder matching ammon/llama3.cuda forward()."""

from __future__ import annotations

import math

from .checkpoint import Checkpoint, Config

ROPE_THETA = 10000.0
RMS_EPS = 1e-5


def _torch():
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError("PyTorch is required for the decoder; pip install torch") from exc
    return torch, F


def configure_precision(device: str) -> list[str]:
    notes: list[str] = []
    torch, _ = _torch()
    torch.set_default_dtype(torch.float32)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("highest")
    elif device.startswith("cuda"):
        notes.append("cuda requested but unavailable; using cpu")
    return notes


def _rmsnorm(x, weight):
    ss = x.pow(2).mean() + RMS_EPS
    return weight * (x * torch_rsqrt(ss))


def torch_rsqrt(ss):
    torch, _ = _torch()
    return torch.rsqrt(ss)


def _rope(q, k, pos: int, head_size: int):
    torch, _ = _torch()
    dim = q.numel()
    kv_dim = k.numel()
    idx = torch.arange(0, dim, 2, device=q.device, dtype=torch.int64)
    head_dim = (idx % head_size).to(dtype=q.dtype)
    freq = 1.0 / (ROPE_THETA ** (head_dim / float(head_size)))
    val = float(pos) * freq
    fcr = torch.cos(val)
    fci = torch.sin(val)
    q0 = q[0::2].clone()
    q1 = q[1::2].clone()
    q[0::2] = q0 * fcr - q1 * fci
    q[1::2] = q0 * fci + q1 * fcr
    n_k = kv_dim // 2
    k0 = k[0::2].clone()
    k1 = k[1::2].clone()
    k[0::2] = k0 * fcr[:n_k] - k1 * fci[:n_k]
    k[1::2] = k0 * fci[:n_k] + k1 * fcr[:n_k]


def _attention(q, k_cache, v_cache, pos: int, n_heads: int, kv_mul: int, head_size: int):
    torch, _ = _torch()
    t = pos + 1
    qh = q.view(n_heads, head_size)
    k = k_cache[:t].view(t, -1, head_size)
    v = v_cache[:t].view(t, -1, head_size)
    if kv_mul != 1:
        k = k.repeat_interleave(kv_mul, dim=1)
        v = v.repeat_interleave(kv_mul, dim=1)
    scale = 1.0 / math.sqrt(head_size)
    scores = torch.einsum("hd,thd->ht", qh, k) * scale
    att = torch.softmax(scores, dim=-1)
    return torch.einsum("ht,thd->hd", att, v).reshape(-1)


class Decoder:
    """FP32 one-token decoder. Weights and KV live on `device`."""

    def __init__(self, checkpoint: Checkpoint, device: str = "cuda"):
        torch, _ = _torch()
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.config: Config = checkpoint.config
        self._load_weights(checkpoint)
        seq = self.config.max_seq_len
        kv = self.config.kv_dim
        self.k_cache = torch.zeros(
            self.config.n_layers, seq, kv, dtype=torch.float32, device=self.device
        )
        self.v_cache = torch.zeros_like(self.k_cache)
        self._compiled = None

    def _load_weights(self, checkpoint: Checkpoint) -> None:
        torch, _ = _torch()
        cfg = checkpoint.config
        layout = checkpoint.layout
        blob = torch.tensor(checkpoint.weights, dtype=torch.float32, device=self.device)

        def view(off: int, shape: tuple[int, ...]):
            n = 1
            for d in shape:
                n *= d
            return blob[off : off + n].view(*shape)

        self.tok_emb = view(layout.token_embedding, (cfg.vocab_size, cfg.dim))
        self.rms_final = view(layout.rms_final_weight, (cfg.dim,))
        self.wcls = view(layout.wcls, (cfg.vocab_size, cfg.dim))
        self.rms_att = []
        self.wq = []
        self.wk = []
        self.wv = []
        self.wo = []
        self.rms_ffn = []
        self.w1 = []
        self.w2 = []
        self.w3 = []
        for layer in range(cfg.n_layers):
            off = layout.layer_offsets(layer, cfg)
            self.rms_att.append(view(off["rms_att"], (cfg.dim,)))
            self.wq.append(view(off["wq"], (cfg.dim, cfg.dim)))
            self.wk.append(view(off["wk"], (cfg.kv_dim, cfg.dim)))
            self.wv.append(view(off["wv"], (cfg.kv_dim, cfg.dim)))
            self.wo.append(view(off["wo"], (cfg.dim, cfg.dim)))
            self.rms_ffn.append(view(off["rms_ffn"], (cfg.dim,)))
            self.w1.append(view(off["w1"], (cfg.hidden_dim, cfg.dim)))
            self.w2.append(view(off["w2"], (cfg.dim, cfg.hidden_dim)))
            self.w3.append(view(off["w3"], (cfg.hidden_dim, cfg.dim)))

    def _step_gpu(self, token: int, pos: int):
        torch, F = _torch()
        cfg = self.config
        x = self.tok_emb[token].clone()
        for layer in range(cfg.n_layers):
            xb = _rmsnorm(x, self.rms_att[layer])
            q = F.linear(xb, self.wq[layer])
            k = F.linear(xb, self.wk[layer])
            v = F.linear(xb, self.wv[layer])
            _rope(q, k, pos, cfg.head_size)
            self.k_cache[layer, pos].copy_(k)
            self.v_cache[layer, pos].copy_(v)
            xb = _attention(
                q,
                self.k_cache[layer],
                self.v_cache[layer],
                pos,
                cfg.n_heads,
                cfg.kv_mul,
                cfg.head_size,
            )
            x = x + F.linear(xb, self.wo[layer])
            xb = _rmsnorm(x, self.rms_ffn[layer])
            hb = F.linear(xb, self.w1[layer])
            hb2 = F.linear(xb, self.w3[layer])
            hb = F.silu(hb) * hb2
            x = x + F.linear(hb, self.w2[layer])
        x = _rmsnorm(x, self.rms_final)
        return F.linear(x, self.wcls)

    def compile_forward(self) -> tuple[float, str | None]:
        """torch.compile the GPU forward. Time is warmup-only."""
        torch, _ = _torch()
        import time

        start = time.perf_counter()
        try:
            with torch.inference_mode():
                self._compiled = torch.compile(self._step_gpu, dynamic=True)
                _ = self._compiled(1, 0)
                self.synchronize()
        except Exception as exc:  # noqa: BLE001 — compile is optional
            self._compiled = None
            return 0.0, str(exc)
        return time.perf_counter() - start, None

    def synchronize(self) -> None:
        torch, _ = _torch()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def step(self, token: int, pos: int):
        """GPU forward + full logit DtoH. Returns 1-D CPU float32 logits."""
        torch, _ = _torch()
        fwd = self._compiled or self._step_gpu
        with torch.inference_mode():
            logits = fwd(int(token), int(pos))
            return logits.detach().to("cpu", dtype=logits.dtype)

    def tensor_slice(self, name: str, index: int = 0, count: int = 8) -> list[float]:
        table = {
            "tok_emb": self.tok_emb[index, :count],
            "rms_final": self.rms_final[:count],
            "wq0": self.wq[0].reshape(-1)[:count],
        }
        return [float(x) for x in table[name].detach().to("cpu")]


def disable_tf32() -> None:
    configure_precision("cuda")
