"""Run llama-completion (phased) and llama-bench (engine-native) without mixing metrics."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
TOOLS = HERE.parents[1].parent
sys.path.insert(0, str(TOOLS))

from bench import common  # noqa: E402
from bench.adapters.llama_cpp import build as cpp_build  # noqa: E402
from bench.adapters.llama_cpp import convert as cpp_convert  # noqa: E402
from bench.adapters.util import run  # noqa: E402
from bench.schema import base_result, validate_result  # noqa: E402

LOAD_RE = re.compile(r"load time =\s*([0-9.]+)\s*ms")
PROMPT_RE = re.compile(
    r"prompt eval time =\s*([0-9.]+)\s*ms /\s*([0-9]+)\s*tokens"
)
EVAL_RE = re.compile(r"eval time =\s*([0-9.]+)\s*ms /\s*([0-9]+)\s*runs")

NATIVE_NOTES = [
    "native Windows/Linux llama.cpp Release CUDA build; refs/ is unmodified",
    "F32 GGUF via llama-convert-llama2c-to-ggml (GGUF metadata n_ctx defaults to 128; runtime --ctx-size overrides)",
    "full GPU offload -ngl all, K/V f32, flash attention off",
    "llama.cpp -n is generated tokens, not llama3.cuda total positions",
    "llama.cpp tokenizer/BOS/EOS differ from llama3.cuda; compatibility text is recorded, not rewritten",
    "llama-completion decode_step_s is eval_time/n_eval repeated (no per-step clock)",
]


def parse_perf(text: str) -> dict[str, float]:
    load_ms = 0.0
    prompt_ms = 0.0
    prompt_n = 0
    eval_ms = 0.0
    eval_n = 0
    m = LOAD_RE.search(text)
    if m:
        load_ms = float(m.group(1))
    m = PROMPT_RE.search(text)
    if m:
        prompt_ms = float(m.group(1))
        prompt_n = int(m.group(2))
    m = EVAL_RE.search(text)
    if m:
        eval_ms = float(m.group(1))
        eval_n = int(m.group(2))
    return {
        "load_s": load_ms / 1000.0,
        "prompt_s": prompt_ms / 1000.0,
        "prompt_n": float(prompt_n),
        "eval_s": eval_ms / 1000.0,
        "eval_n": float(eval_n),
    }


def completion_cmd(
    exe: pathlib.Path,
    gguf: pathlib.Path,
    *,
    prompt: str,
    n_predict: int,
    ctx: int,
    warmup: bool,
) -> list[str]:
    cmd = [
        str(exe),
        "-m",
        str(gguf),
        "-p",
        prompt,
        "-n",
        str(n_predict),
        "-c",
        str(ctx),
        "-ngl",
        "all",
        "-ctk",
        "f32",
        "-ctv",
        "f32",
        "-fa",
        "off",
        "-b",
        "1",
        "-ub",
        "1",
        "--temp",
        "0",
        "--top-k",
        "1",
        "--top-p",
        "1.0",
        "--min-p",
        "0",
        "--repeat-penalty",
        "1.0",
        "-no-cnv",
        "--simple-io",
        "--no-display-prompt",
        "--perf",
        "--seed",
        "0",
    ]
    if not warmup:
        cmd.append("--no-warmup")
    return cmd


def bench_cmd(
    exe: pathlib.Path,
    gguf: pathlib.Path,
    *,
    n_prompt: int,
    n_gen: int,
    depth: int,
    reps: int,
) -> list[str]:
    return [
        str(exe),
        "-m",
        str(gguf),
        "-ngl",
        "99",
        "-ctk",
        "f32",
        "-ctv",
        "f32",
        "-fa",
        "off",
        "-p",
        str(n_prompt),
        "-n",
        str(n_gen),
        "-d",
        str(depth),
        "-r",
        str(reps),
        "-o",
        "jsonl",
    ]


def wrap_completion(
    *,
    gguf: pathlib.Path,
    checkpoint: pathlib.Path,
    sha: str,
    config: dict,
    prompt: str,
    mode: str,
    n_predict: int,
    ctx: int,
    stdout: str,
    stderr: str,
    warmup_s: float,
) -> dict:
    perf = parse_perf(stderr + "\n" + stdout)
    eval_n = int(perf["eval_n"])
    step = (perf["eval_s"] / eval_n) if eval_n else 0.0
    decode_step_s = [step] * eval_n
    ttft_s = perf["prompt_s"] + (decode_step_s[0] if decode_step_s else 0.0)
    text = stdout.strip()
    match = text.startswith(common.DREAM_STORY) or text == common.DREAM_STORY
    notes = list(NATIVE_NOTES)
    if mode == "compatibility" and not match:
        notes.append("generated text does not match llama3.cuda DREAM_STORY (engine-native tokenizer)")
    prompt_n = max(int(perf["prompt_n"]), 1)
    return base_result(
        engine="llama.cpp",
        execution="native",
        checkpoint={"path": str(checkpoint), "sha256": sha, "config": config},
        workload={
            "tier": mode,
            "prompt": prompt,
            "batch": 1,
            "context_len": ctx,
            "total_positions": n_predict + prompt_n,
            "decode_steps": eval_n,
            "sampling": "greedy",
        },
        tokens={
            "prompt": [],
            "generated": [],
            "text": text,
            "match_expected": bool(match) if mode == "compatibility" else True,
        },
        phases={
            "load_s": perf["load_s"],
            "warmup_s": warmup_s,
            "prompt_s": perf["prompt_s"],
            "ttft_s": ttft_s,
            "decode_step_s": decode_step_s,
            "compat_elapsed_s": perf["prompt_s"] + perf["eval_s"],
        },
        metrics={
            "prompt_tok_s": (perf["prompt_n"] / perf["prompt_s"]) if perf["prompt_s"] else 0.0,
            "decode_tok_s": (perf["eval_n"] / perf["eval_s"]) if perf["eval_s"] else 0.0,
            "legacy_compat_tok_s": 0.0,
        },
        engine_native_notes=notes + ["legacy_compat_tok_s omitted: llama.cpp is not (pos-1)/elapsed"],
        build={"backend": "cuda", "gguf": str(gguf), "commit": common.LLAMA_CPP_COMMIT},
        replay_stats=None,
    )


def wrap_bench_line(raw: dict, *, checkpoint: pathlib.Path, sha: str, config: dict, gguf: pathlib.Path) -> dict:
    n_prompt = int(raw.get("n_prompt") or 0)
    n_gen = int(raw.get("n_gen") or 0)
    avg_ns = float(raw.get("avg_ns") or 0.0)
    elapsed = avg_ns / 1e9
    notes = list(NATIVE_NOTES) + [
        "llama.cpp-bench uses random tokens; not the compatibility headline",
        f"llama-bench test n_prompt={n_prompt} n_gen={n_gen}",
    ]
    decode_n = n_gen
    prompt_s = elapsed if n_gen == 0 else 0.0
    eval_s = elapsed if n_gen else 0.0
    steps = [eval_s / decode_n] * decode_n if decode_n and eval_s else []
    return base_result(
        engine="llama.cpp-bench",
        execution="native",
        checkpoint={"path": str(checkpoint), "sha256": sha, "config": config},
        workload={
            "tier": "scaling" if n_gen or n_prompt else "compatibility",
            "prompt": "",
            "batch": 1,
            "context_len": int(raw.get("n_depth") or n_prompt or 0),
            "total_positions": n_prompt + n_gen,
            "decode_steps": n_gen,
            "sampling": "engine-native",
        },
        tokens={"prompt": [], "generated": [], "text": "", "match_expected": True},
        phases={
            "load_s": 0.0,
            "warmup_s": 0.0,
            "prompt_s": prompt_s,
            "ttft_s": prompt_s,
            "decode_step_s": steps,
            "compat_elapsed_s": 0.0,
        },
        metrics={
            "prompt_tok_s": (n_prompt / prompt_s) if prompt_s and n_prompt else 0.0,
            "decode_tok_s": float(raw.get("avg_ts") or 0.0) if n_gen else 0.0,
            "legacy_compat_tok_s": 0.0,
        },
        engine_native_notes=notes,
        build={"backend": "cuda", "gguf": str(gguf), "llama_bench": raw},
        replay_stats=None,
    )


def _header_config(checkpoint: pathlib.Path) -> dict:
    import struct

    raw = checkpoint.read_bytes()[:28]
    dim, hidden, layers, heads, kv, vocab, seq = struct.unpack("<7i", raw)
    return {
        "dim": dim,
        "hidden_dim": hidden,
        "n_layers": layers,
        "n_heads": heads,
        "n_kv_heads": kv,
        "vocab_size": abs(vocab),
        "max_seq_len": seq,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=str(common.MODELS / "stories15M.bin"))
    p.add_argument("--tokenizer", default=str(common.MODELS / "tokenizer.bin"))
    p.add_argument("--mode", choices=("compatibility", "scaling", "bench"), default="compatibility")
    p.add_argument("--prompt", default=common.DREAM_PROMPT)
    p.add_argument("--context", type=int, default=None)
    p.add_argument("--decode-steps", type=int, default=common.SCALING_DECODE_STEPS)
    p.add_argument("-n", type=int, default=common.COMPAT_TOTAL_POSITIONS)
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--print-cmd", action="store_true")
    args = p.parse_args(argv)

    ckpt = pathlib.Path(args.checkpoint).resolve()
    tok = pathlib.Path(args.tokenizer).resolve()
    sha = common.sha256_file(ckpt) if ckpt.exists() else ""
    config = _header_config(ckpt) if ckpt.exists() else {
        "dim": 288, "hidden_dim": 768, "n_layers": 6, "n_heads": 6,
        "n_kv_heads": 6, "vocab_size": 32000, "max_seq_len": 256,
    }
    ctx = args.context or (args.n if args.mode == "compatibility" else common.SCALING_CONTEXT_LENGTHS[0])
    ctx = min(ctx, config["max_seq_len"])
    n_predict = args.n if args.mode == "compatibility" else args.decode_steps
    gguf = cpp_convert.gguf_path_for(ckpt)

    if args.print_cmd:
        completion = pathlib.Path("llama-completion")
        print(" ".join(completion_cmd(completion, gguf, prompt=args.prompt, n_predict=n_predict, ctx=ctx, warmup=False)))
        print(" ".join(bench_cmd(pathlib.Path("llama-bench"), gguf, n_prompt=0, n_gen=n_predict, depth=ctx, reps=args.reps)))
        return 0

    tools = cpp_build.build()
    gguf = cpp_convert.convert(ckpt, tok, gguf)

    if args.mode == "bench":
        cmd = bench_cmd(
            tools["llama-bench"],
            gguf,
            n_prompt=0,
            n_gen=n_predict,
            depth=ctx,
            reps=args.reps,
        )
        proc = run(cmd, capture=True, check=True)
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            raw = json.loads(line)
            obj = wrap_bench_line(raw, checkpoint=ckpt, sha=sha, config=config, gguf=gguf)
            print(common.dumps(obj), flush=True)
        return 0

    warmup_s = 0.0
    for _ in range(max(0, args.warmups)):
        cmd = completion_cmd(
            tools["llama-completion"], gguf, prompt=args.prompt, n_predict=n_predict, ctx=ctx, warmup=True
        )
        t0 = common.monotonic_s()
        run(cmd, capture=True, check=True)
        warmup_s += common.monotonic_s() - t0

    for _ in range(max(1, args.reps)):
        cmd = completion_cmd(
            tools["llama-completion"], gguf, prompt=args.prompt, n_predict=n_predict, ctx=ctx, warmup=False
        )
        proc = run(cmd, capture=True, check=True)
        obj = wrap_completion(
            gguf=gguf,
            checkpoint=ckpt,
            sha=sha,
            config=config,
            prompt=args.prompt,
            mode=args.mode,
            n_predict=n_predict,
            ctx=ctx,
            stdout=proc.stdout,
            stderr=proc.stderr,
            warmup_s=warmup_s,
        )
        validate_result(obj)
        print(common.dumps(obj), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
