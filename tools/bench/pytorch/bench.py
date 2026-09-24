"""Direct FP32 PyTorch TinyStories benchmark (eager primary, compile secondary)."""

from __future__ import annotations

import argparse
import sys
import pathlib
from typing import Any

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1].parent))  # tools/
sys.path.insert(0, str(HERE.parents[1]))  # tools/bench

from bench import common  # noqa: E402
from bench.schema import base_result  # noqa: E402
from bench.pytorch.checkpoint import Checkpoint  # noqa: E402
from bench.pytorch.env import process_affinity_mask  # noqa: E402
from bench.pytorch.model import Decoder, configure_precision  # noqa: E402
from bench.pytorch.tokenizer import (  # noqa: E402
    BOS_ID,
    Tokenizer,
    apply_dream_prompt_patch,
    printable_piece,
    sample_argmax,
)


def _sync(decoder: Decoder) -> None:
    decoder.synchronize()


def _now() -> float:
    return common.monotonic_s()


def pad_or_trim(tokens: list[int], length: int) -> list[int]:
    if length <= 0:
        raise ValueError("context length must be positive")
    if len(tokens) >= length:
        out = tokens[:length]
        return out
    filler = next((t for t in reversed(tokens) if t != BOS_ID), 13)
    return tokens + [filler] * (length - len(tokens))


def encode_prompt(tokenizer: Tokenizer, prompt: str, *, patch: bool) -> list[int]:
    tokens = tokenizer.encode(prompt, bos=True, eos=False)
    if patch:
        apply_dream_prompt_patch(tokens)
    return tokens


def generate_loop(
    decoder: Decoder,
    prompt_tokens: list[int],
    max_new_tokens: int,
    *,
    stop_on_bos: bool,
) -> dict[str, Any]:
    if not prompt_tokens:
        raise ValueError("expected at least one prompt token")
    n_prompt = len(prompt_tokens)
    token = prompt_tokens[0]
    pos = 0
    generated: list[int] = []
    prompt_s = 0.0
    decode_step_s: list[float] = []
    ttft_s = 0.0
    first_forward_done = False
    compat_start = 0.0
    run_start = 0.0

    # decoder.step() ends with the full-logit DtoH and a stream sync, so every forward is
    # bracketed by syncs without extra per-step device-wide synchronize() calls.
    _sync(decoder)
    run_start = _now()
    while pos < max_new_tokens - 1:
        t0 = _now()
        logits = decoder.step(token, pos)
        t1 = _now()
        if pos < n_prompt - 1:
            nxt = prompt_tokens[pos + 1]
            prompt_s += t1 - t0
        else:
            nxt = sample_argmax(logits)
            decode_step_s.append(t1 - t0)
            if len(decode_step_s) == 1:
                ttft_s = t1 - run_start
        pos += 1
        if not first_forward_done:
            compat_start = t1
            first_forward_done = True
        if stop_on_bos and nxt == BOS_ID:
            break
        generated.append(int(nxt))
        token = int(nxt)

    compat_elapsed = (_now() - compat_start) if first_forward_done else 0.0
    return {
        "generated": generated,
        "pos": pos,
        "prompt_s": prompt_s,
        "ttft_s": ttft_s,
        "decode_step_s": decode_step_s,
        "compat_elapsed_s": compat_elapsed,
    }


def decode_text(tokenizer: Tokenizer, prompt_tokens: list[int], generated: list[int]) -> str:
    text = []
    prev = prompt_tokens[0]
    for tok in generated:
        text.append(printable_piece(tokenizer.decode(prev, tok)))
        prev = tok
    return "".join(text)


def rates(prompt_tokens: int, prompt_s: float, decode_n: int, decode_s: float, pos: int, elapsed: float) -> dict[str, float]:
    prompt_tok_s = ((prompt_tokens - 1) / prompt_s) if prompt_s > 0 else 0.0
    decode_tok_s = (decode_n / decode_s) if decode_s > 0 else 0.0
    legacy = ((pos - 1) / elapsed) if elapsed > 0 else 0.0
    return {
        "prompt_tok_s": prompt_tok_s,
        "decode_tok_s": decode_tok_s,
        "legacy_compat_tok_s": legacy,
    }


def run_once(
    decoder: Decoder,
    tokenizer: Tokenizer,
    *,
    engine: str,
    ckpt: Checkpoint,
    ckpt_path: pathlib.Path,
    sha: str,
    workload: dict[str, Any],
    prompt_tokens: list[int],
    max_new_tokens: int,
    load_s: float,
    warmup_s: float,
    notes: list[str],
    require_match: bool,
    build: dict[str, Any],
) -> dict[str, Any]:
    out = generate_loop(
        decoder,
        prompt_tokens,
        max_new_tokens,
        stop_on_bos=workload["tier"] == "compatibility",
    )
    text = decode_text(tokenizer, prompt_tokens, out["generated"])
    match = text == common.DREAM_STORY if workload["tier"] == "compatibility" else True
    if require_match and workload["tier"] == "compatibility" and not match:
        raise SystemExit(f"token mismatch\n--- got ---\n{text}\n--- expected ---\n{common.DREAM_STORY}")
    decode_s = sum(out["decode_step_s"])
    metrics = rates(
        len(prompt_tokens),
        out["prompt_s"],
        len(out["decode_step_s"]),
        decode_s,
        out["pos"],
        out["compat_elapsed_s"],
    )
    return base_result(
        engine=engine,
        execution="native",
        checkpoint={
            "path": str(ckpt_path),
            "sha256": sha,
            "config": ckpt.config.as_dict(),
        },
        workload=workload,
        tokens={
            "prompt": prompt_tokens,
            "generated": out["generated"],
            "text": text,
            "match_expected": match,
        },
        phases={
            "load_s": load_s,
            "warmup_s": warmup_s,
            "prompt_s": out["prompt_s"],
            "ttft_s": out["ttft_s"],
            "decode_step_s": out["decode_step_s"],
            "compat_elapsed_s": out["compat_elapsed_s"],
        },
        metrics=metrics,
        engine_native_notes=notes,
        build=build,
        replay_stats=None,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=str(common.MODELS / "stories15M.bin"))
    p.add_argument("--tokenizer", default=str(common.MODELS / "tokenizer.bin"))
    p.add_argument("--mode", choices=("compatibility", "scaling"), default="compatibility")
    p.add_argument("--prompt", default=common.DREAM_PROMPT)
    p.add_argument("--context", type=int, default=None, help="scaling context length")
    p.add_argument("--decode-steps", type=int, default=common.SCALING_DECODE_STEPS)
    p.add_argument("-n", "--total-positions", type=int, default=common.COMPAT_TOTAL_POSITIONS)
    p.add_argument(
        "--device",
        default="cuda",
        help="cuda[:N] (fails if CUDA is unavailable) or cpu",
    )
    p.add_argument("--compile", action="store_true")
    p.add_argument(
        "--require-compile",
        action="store_true",
        help="exit non-zero instead of falling back to eager when torch.compile fails",
    )
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--no-dream-patch", action="store_true")
    args = p.parse_args(argv)

    ckpt_path = pathlib.Path(args.checkpoint)
    tok_path = pathlib.Path(args.tokenizer)
    sha = common.sha256_file(ckpt_path)
    expected = common.CHECKPOINT_SHA256.get(ckpt_path.name)
    notes: list[str] = []
    if expected and sha != expected:
        raise SystemExit(f"checkpoint hash {sha} != pinned {expected}")

    try:
        notes.extend(configure_precision(args.device))
    except RuntimeError as exc:
        raise SystemExit(f"pytorch bench: {exc}") from exc
    import torch

    t0 = _now()
    ckpt = Checkpoint.read_path(ckpt_path)
    decoder = Decoder(ckpt, device=args.device)
    _sync(decoder)
    load_s = _now() - t0
    tokenizer = Tokenizer.from_path(tok_path, ckpt.config.vocab_size)

    build: dict[str, Any] = {
        "device": str(decoder.device),
        "compile": False,
        "compile_requested": bool(args.compile),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python": sys.version.split()[0],
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "logits_dtoh": "pinned non_blocking copy + stream sync" if decoder.device.type == "cuda" else "none (cpu)",
        "cpu_affinity_mask": process_affinity_mask(),
    }
    if decoder.device.type == "cuda":
        build["device_name"] = torch.cuda.get_device_name(decoder.device)
        cap = torch.cuda.get_device_capability(decoder.device)
        build["device_capability"] = f"sm_{cap[0]}{cap[1]}"

    prompt_tokens = encode_prompt(tokenizer, args.prompt, patch=not args.no_dream_patch)
    if args.mode == "compatibility":
        context_len = args.total_positions
        max_new = min(args.total_positions, ckpt.config.max_seq_len)
        decode_steps = max_new - len(prompt_tokens)
        workload_prompt = args.prompt
    else:
        context_len = min(args.context or common.SCALING_CONTEXT_LENGTHS[0], ckpt.config.max_seq_len)
        prompt_tokens = pad_or_trim(prompt_tokens, context_len)
        max_new = min(context_len + args.decode_steps, ckpt.config.max_seq_len + 1)
        decode_steps = args.decode_steps
        workload_prompt = args.prompt
        notes.append("scaling pads/trims prompt tokens to context_len; filler is last non-BOS id")

    workload = {
        "tier": args.mode,
        "prompt": workload_prompt,
        "batch": 1,
        "context_len": context_len,
        "total_positions": max_new,
        "decode_steps": decode_steps,
        "sampling": "greedy",
    }

    stop_on_bos = args.mode == "compatibility"
    warmup_s = 0.0
    eager_tokens: list[int] | None = None
    if args.compile:
        # Eager reference tokens for this exact workload; compiled runs must reproduce them.
        t_w = _now()
        eager_tokens = generate_loop(decoder, prompt_tokens, max_new, stop_on_bos=stop_on_bos)["generated"]
        warmup_s += _now() - t_w
        compile_s, compile_err = decoder.compile_forward()
        warmup_s += compile_s
        if compile_err:
            if args.require_compile:
                raise SystemExit(f"torch.compile failed: {compile_err}")
            notes.append(f"torch.compile unavailable, ran eager instead: {compile_err}")
        else:
            build.update(_compile_build_info())
            notes.append(
                "torch.compile(fullgraph=True, mode=default, automatic dynamic token/pos, "
                "no CUDA graphs); compilation happens in warmup"
            )
    engine = "pytorch-compile" if decoder.compiled else "pytorch-eager"
    build["engine"] = engine
    build["compile"] = decoder.compiled

    for _ in range(max(0, args.warmups)):
        t_w = _now()
        warm = generate_loop(decoder, prompt_tokens, max_new, stop_on_bos=stop_on_bos)
        _sync(decoder)
        warmup_s += _now() - t_w
        if eager_tokens is not None and decoder.compiled and warm["generated"] != eager_tokens:
            raise SystemExit(
                f"torch.compile tokens diverge from eager\n  eager:    {eager_tokens}\n"
                f"  compiled: {warm['generated']}"
            )
    graphs_after_warmup = _dynamo_graphs() if decoder.compiled else None

    require_match = args.mode == "compatibility"
    for _ in range(max(1, args.reps)):
        result = run_once(
            decoder,
            tokenizer,
            engine=engine,
            ckpt=ckpt,
            ckpt_path=ckpt_path,
            sha=sha,
            workload=workload,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new,
            load_s=load_s,
            warmup_s=warmup_s,
            notes=notes,
            require_match=require_match,
            build=build,
        )
        if decoder.compiled:
            if eager_tokens is not None and result["tokens"]["generated"] != eager_tokens:
                raise SystemExit("torch.compile tokens diverge from eager in a timed repetition")
            graphs = _dynamo_graphs()
            if graphs != graphs_after_warmup:
                raise SystemExit(
                    f"torch.compile recompiled during timing ({graphs_after_warmup} -> {graphs} graphs)"
                )
            result["build"] = {**build, "dynamo_graphs": graphs, "recompiles_during_timing": 0}
        print(common.dumps(result), flush=True)
    return 0


def _dynamo_graphs() -> int:
    from torch._dynamo.utils import counters

    return int(counters["stats"]["unique_graphs"])


def _compile_build_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "compile_backend": "inductor",
        "compile_mode": "default",
        "compile_dynamic": "automatic (token/pos SymInts, static weights)",
        "cuda_graphs": False,
    }
    try:
        import triton

        info["triton"] = triton.__version__
    except Exception as exc:  # noqa: BLE001
        info["triton"] = f"unavailable: {exc}"
    return info


if __name__ == "__main__":
    sys.exit(main())
