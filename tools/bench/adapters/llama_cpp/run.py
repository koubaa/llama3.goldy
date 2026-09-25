"""Run llama-completion (phased) and llama-bench (engine-native) without mixing metrics."""

from __future__ import annotations

import argparse
import ast
import json
import os
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
TOOLS = HERE.parents[1].parent
sys.path.insert(0, str(TOOLS))

from bench import common  # noqa: E402
from bench.adapters.llama_cpp import build as cpp_build  # noqa: E402
from bench.adapters.llama_cpp import convert as cpp_convert  # noqa: E402
from bench.schema import base_result, validate_result  # noqa: E402

# Perf lines are "<prefix>_print: <name> time = ..."; the prefix changed from
# llama_perf_context_print to common_perf_print across llama.cpp versions.
LOAD_RE = re.compile(r"_print:\s+load time =\s*([0-9.]+)\s*ms")
PROMPT_RE = re.compile(r"_print:\s+prompt eval time =\s*([0-9.]+)\s*ms /\s*([0-9]+)\s*tokens")
EVAL_RE = re.compile(r"_print:\s+eval time =\s*([0-9.]+)\s*ms /\s*([0-9]+)\s*runs")
N_PROMPT_TOKENS_RE = re.compile(r"number of tokens in prompt = (\d+)")
PROMPT_TOKEN_RE = re.compile(r"^(?:[0-9.]+ \w )?\s*(\d+) -> '", re.M)
OFFLOAD_RE = re.compile(r"offloaded (\d+)/(\d+) layers to GPU")
DEVICE_RE = re.compile(r"using device (\S+) \(([^)]*)\)")
GPU_BUFFER_RE = re.compile(
    r"(CUDA\d+|Metal\d+|MTL\d+)(?:_Mapped)? (model|KV) buffer size =\s*([0-9.]+) MiB"
)
KV_TYPES_RE = re.compile(r"K \((\w+)\):.*V \((\w+)\):")
FLASH_ATTN_RE = re.compile(r"flash_attn\s+= (\w+)")
# common/log.cpp timestamps: minutes.seconds.milliseconds.microseconds
LOG_TS = r"^(\d+)\.(\d{2})\.(\d{3})\.(\d{3}) \w "
LOAD_START_RE = re.compile(LOG_TS + r".*load the model", re.M)
LOAD_END_RE = re.compile(LOG_TS + r"sched_reserve: reserve took", re.M)

NATIVE_NOTES = [
    (
        "native llama.cpp Release Metal build (Ninja); refs/ is unmodified"
        if sys.platform == "darwin"
        else "native llama.cpp Release CUDA build (Ninja + MSVC on Windows); refs/ is unmodified"
    ),
    "F32 GGUF via llama-convert-llama2c-to-ggml; GGUF n_ctx_train is 128 and llama.cpp pads the runtime context to 256 cells",
    "full GPU offload; token_embd stays in a CPU_Mapped buffer (llama.cpp input-layer default, get_rows only)",
]
COMPLETION_NOTES = [
    "-ngl all, K/V f32, flash attention off, -b 1 -ub 1 serial prompt forwards",
    "llama.cpp SPM tokenizer encodes ' I' (306); llama3.cuda patches it to 76, so compatibility text is recorded, not rewritten",
    "each repetition is a fresh llama-completion process (weights reloaded per repetition)",
    "load_s is log time from model load start to scheduler reservation (weights + GPU upload + context/KV); excludes process start and CUDA init",
    "prompt_s is llama.cpp prompt eval over all n_prompt forwards, including the one that yields the first token; ttft_s = prompt_s",
    "decode_step_s is eval_time/n_eval repeated n_eval times (no per-step clock); n_eval = generated tokens - 1",
    "tokens.generated is empty: llama-completion does not expose sampled ids",
    "legacy_compat_tok_s omitted: llama.cpp is not (pos-1)/elapsed",
]
BENCH_NOTES = [
    "llama-bench random tokens; not the compatibility headline",
    "llama-bench at this commit rejects -ctk/-ctv f32 (accepts f16/bf16/q*), so K/V are f16",
    "-ngl 99, flash attention off, llama-bench default batch sizes; depth prefill is untimed",
    "one record per llama-bench sample (samples_ns); decode_step_s is sample_ns/n_gen repeated",
]


def _perf_value(regex: re.Pattern[str], text: str) -> tuple[float, int]:
    m = regex.search(text)
    if not m:
        return 0.0, 0
    return float(m.group(1)), int(m.group(2)) if m.lastindex and m.lastindex >= 2 else 0


def _log_seconds(m: re.Match[str]) -> float:
    minutes, seconds, ms, us = (int(g) for g in m.groups()[:4])
    return minutes * 60 + seconds + ms / 1e3 + us / 1e6


def parse_load_s(text: str) -> float | None:
    """Model read + GPU upload + context/KV setup, from verbose log timestamps."""
    start = LOAD_START_RE.search(text)
    end = LOAD_END_RE.search(text, start.end()) if start else None
    if not (start and end):
        return None
    return _log_seconds(end) - _log_seconds(start)


def parse_perf(text: str) -> dict[str, float]:
    # llama_perf "load time" is context start -> first eval, i.e. it overlaps prompt eval.
    load_ms, _ = _perf_value(LOAD_RE, text)
    load_s = parse_load_s(text)
    prompt_ms, prompt_n = _perf_value(PROMPT_RE, text)
    eval_ms, eval_n = _perf_value(EVAL_RE, text)
    return {
        "load_s": load_s if load_s is not None else load_ms / 1000.0,
        "prompt_s": prompt_ms / 1000.0,
        "prompt_n": float(prompt_n),
        "eval_s": eval_ms / 1000.0,
        "eval_n": float(eval_n),
    }


def parse_prompt_tokens(text: str) -> list[int]:
    """Prompt ids printed by llama-completion --verbose-prompt."""
    m = N_PROMPT_TOKENS_RE.search(text)
    if not m:
        return []
    n = int(m.group(1))
    return [int(t) for t in PROMPT_TOKEN_RE.findall(text, m.end())[:n]]


def parse_gpu_offload(text: str) -> dict:
    info: dict = {}
    m = DEVICE_RE.search(text)
    if m:
        info["device"], info["device_name"] = m.group(1), m.group(2)
    m = OFFLOAD_RE.search(text)
    if m:
        info["offloaded_layers"], info["total_layers"] = int(m.group(1)), int(m.group(2))
    for dev, kind, mib in GPU_BUFFER_RE.findall(text):
        info.setdefault(f"gpu_{kind.lower()}_buffer_mib", float(mib))
        info.setdefault("buffer_device", dev)
    m = KV_TYPES_RE.search(text)
    if m:
        info["kv_types"] = [m.group(1), m.group(2)]
    m = FLASH_ATTN_RE.search(text)
    if m:
        info["flash_attn"] = m.group(1)
    return info


def require_gpu_offload(info: dict, *, kv_type: str | None) -> None:
    problems = []
    total = info.get("total_layers", 0)
    if not total or info.get("offloaded_layers") != total:
        problems.append(f"offloaded {info.get('offloaded_layers')}/{total} layers")
    if "gpu_model_buffer_mib" not in info:
        problems.append("no GPU model buffer")
    if "gpu_kv_buffer_mib" not in info:
        problems.append("no GPU KV buffer")
    if kv_type and info.get("kv_types") != [kv_type, kv_type]:
        problems.append(f"K/V types {info.get('kv_types')} != {kv_type}")
    if info.get("flash_attn") not in (None, "disabled"):
        problems.append(f"flash_attn={info['flash_attn']}")
    if problems:
        raise SystemExit("llama.cpp did not run fully on the GPU: " + "; ".join(problems))


def completion_cmd(
    exe: pathlib.Path,
    gguf: pathlib.Path,
    *,
    prompt: str,
    n_predict: int,
    ctx: int,
    warmup: bool,
    ignore_eos: bool = False,
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
        "-fit",
        "off",
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
        "--verbose-prompt",
        "--perf",
        "--seed",
        "0",
        "-lv",
        "4",
    ]
    if ignore_eos:
        cmd.append("--ignore-eos")
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
    warmup: bool = True,
) -> list[str]:
    cmd = [
        str(exe),
        "-m",
        str(gguf),
        "-ngl",
        "99",
        "-ctk",
        "f16",
        "-ctv",
        "f16",
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
        "-v",
    ]
    if not warmup:
        cmd.append("--no-warmup")
    return cmd


def tokenize_cmd(exe: pathlib.Path, gguf: pathlib.Path, text: str) -> list[str]:
    return [str(exe), "-m", str(gguf), "-p", text, "--ids", "--log-disable"]


def _exec(cmd: list[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        cmd,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-4000:])
        raise SystemExit(f"{pathlib.Path(cmd[0]).name} exited {proc.returncode}")
    return proc


def tokenize(exe: pathlib.Path, gguf: pathlib.Path, text: str) -> list[int]:
    out = _exec(tokenize_cmd(exe, gguf, text)).stdout
    lines = [ln.strip() for ln in out.splitlines() if ln.strip().startswith("[")]
    if not lines:
        raise SystemExit(f"llama-tokenize printed no id list: {out!r}")
    return [int(t) for t in ast.literal_eval(lines[-1])]


def scaling_prompt(prompt: str, base_ids: list[int], ctx: int, tokenize_fn) -> tuple[str, list[int]]:
    """Pad the prompt with its last word so it encodes to ctx ids (pytorch pad_or_trim filler rule)."""
    if ctx < len(base_ids):
        raise SystemExit(f"context {ctx} < prompt length {len(base_ids)}; llama.cpp text prompts cannot be trimmed by id")
    filler = " " + prompt.split()[-1]
    text = prompt + filler * (ctx - len(base_ids))
    ids = tokenize_fn(text)
    expected = base_ids + [base_ids[-1]] * (ctx - len(base_ids))
    if ids != expected:
        raise SystemExit(f"padded scaling prompt tokenized to {len(ids)} ids, not the expected {ctx}")
    return text, ids


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
    notes: list[str],
    build: dict,
) -> dict:
    perf = parse_perf(stderr + "\n" + stdout)
    eval_n = int(perf["eval_n"])
    prompt_n = int(perf["prompt_n"])
    step = (perf["eval_s"] / eval_n) if eval_n else 0.0
    generated = stdout.rstrip("\r\n")
    text = prompt + generated if mode == "compatibility" else generated
    match = text.startswith(common.DREAM_STORY)
    notes = list(notes)
    if mode == "compatibility" and not match:
        notes.append("generated text does not match llama3.cuda DREAM_STORY (engine-native tokenizer)")
    return base_result(
        engine="llama.cpp",
        execution="native",
        checkpoint={"path": str(checkpoint), "sha256": sha, "config": config},
        workload={
            "tier": mode,
            "prompt": prompt,
            "batch": 1,
            "context_len": ctx,
            "total_positions": prompt_n + n_predict,
            "decode_steps": n_predict,
            "sampling": "greedy",
        },
        tokens={
            "prompt": parse_prompt_tokens(stderr),
            "generated": [],
            "text": text,
            "match_expected": bool(match) if mode == "compatibility" else True,
        },
        phases={
            "load_s": perf["load_s"],
            "warmup_s": warmup_s,
            "prompt_s": perf["prompt_s"],
            "ttft_s": perf["prompt_s"],
            "decode_step_s": [step] * eval_n,
            "compat_elapsed_s": perf["prompt_s"] + perf["eval_s"],
        },
        metrics={
            "prompt_tok_s": (prompt_n / perf["prompt_s"]) if perf["prompt_s"] else 0.0,
            "decode_tok_s": (eval_n / perf["eval_s"]) if perf["eval_s"] else 0.0,
            "legacy_compat_tok_s": 0.0,
        },
        engine_native_notes=notes,
        build=build,
        replay_stats=None,
    )


def wrap_bench_line(
    raw: dict,
    *,
    checkpoint: pathlib.Path,
    sha: str,
    config: dict,
    gguf: pathlib.Path,
    sample_ns: float | None = None,
    build: dict | None = None,
) -> dict:
    n_prompt = int(raw.get("n_prompt") or 0)
    n_gen = int(raw.get("n_gen") or 0)
    n_depth = int(raw.get("n_depth") or 0)
    elapsed = float(sample_ns if sample_ns is not None else raw.get("avg_ns") or 0.0) / 1e9
    notes = list(NATIVE_NOTES) + BENCH_NOTES + [
        f"llama-bench test n_prompt={n_prompt} n_gen={n_gen} n_depth={n_depth}",
    ]
    prompt_s = elapsed if n_gen == 0 else 0.0
    eval_s = elapsed if n_gen else 0.0
    steps = [eval_s / n_gen] * n_gen if n_gen and eval_s else []
    raw_summary = {k: v for k, v in raw.items() if k not in ("samples_ns", "samples_ts")}
    return base_result(
        engine="llama.cpp-bench",
        execution="native",
        checkpoint={"path": str(checkpoint), "sha256": sha, "config": config},
        workload={
            "tier": "scaling",
            "prompt": "",
            "batch": 1,
            "context_len": n_depth or n_prompt,
            "total_positions": n_depth + n_prompt + n_gen,
            "decode_steps": n_gen,
            "sampling": "engine-native",
        },
        precision={"weights": "fp32", "activations": "fp32", "kv": "fp16", "tf32": False},
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
            "decode_tok_s": (n_gen / eval_s) if n_gen and eval_s else 0.0,
            "legacy_compat_tok_s": 0.0,
        },
        engine_native_notes=notes,
        build={**(build or {}), "gguf": str(gguf), "llama_bench": raw_summary},
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


def _build_info(gpu: dict) -> dict:
    info = {
        "backend": "cuda",
        "commit": common.LLAMA_CPP_COMMIT,
        "cuda_architectures": cpp_build.cuda_arch(),
        "gpu": gpu,
    }
    if os.name == "nt":
        info["cuda_flags"] = list(cpp_build.WINDOWS_CUDA_FLAGS)
    return info


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=str(common.MODELS / "stories15M.bin"))
    p.add_argument("--tokenizer", default=str(common.MODELS / "tokenizer.bin"))
    p.add_argument("--mode", choices=("compatibility", "scaling", "bench"), default="compatibility")
    p.add_argument("--prompt", default=common.DREAM_PROMPT)
    p.add_argument("--context", type=int, default=None)
    p.add_argument("--decode-steps", type=int, default=common.SCALING_DECODE_STEPS)
    p.add_argument("-n", type=int, default=common.COMPAT_TOTAL_POSITIONS, help="compatibility total positions")
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
    if args.mode == "compatibility":
        ctx = min(args.context or args.n, config["max_seq_len"])
    else:
        ctx = min(args.context or common.SCALING_CONTEXT_LENGTHS[0], config["max_seq_len"])
    gguf = cpp_convert.gguf_path_for(ckpt)
    warmup = args.warmups > 0

    if args.print_cmd:
        completion = pathlib.Path("llama-completion")
        print(subprocess.list2cmdline(completion_cmd(
            completion, gguf, prompt=args.prompt, n_predict=args.decode_steps, ctx=ctx, warmup=warmup
        )))
        print(subprocess.list2cmdline(bench_cmd(
            pathlib.Path("llama-bench"), gguf, n_prompt=0, n_gen=args.decode_steps, depth=ctx, reps=args.reps,
            warmup=warmup,
        )))
        return 0

    tools = cpp_build.ensure_built()
    gguf = cpp_convert.convert(ckpt, tok, gguf)

    if args.mode == "bench":
        cmd = bench_cmd(
            tools["llama-bench"],
            gguf,
            n_prompt=0,
            n_gen=args.decode_steps,
            depth=ctx,
            reps=args.reps,
            warmup=warmup,
        )
        proc = _exec(cmd)
        gpu = parse_gpu_offload(proc.stderr)
        require_gpu_offload(gpu, kv_type="f16")
        build = _build_info(gpu)
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            raw = json.loads(line)
            for sample_ns in raw.get("samples_ns") or [raw.get("avg_ns")]:
                obj = wrap_bench_line(
                    raw, checkpoint=ckpt, sha=sha, config=config, gguf=gguf, sample_ns=sample_ns, build=build
                )
                validate_result(obj)
                print(common.dumps(obj), flush=True)
        return 0

    base_ids = tokenize(tools["llama-tokenize"], gguf, args.prompt)
    notes = list(NATIVE_NOTES) + COMPLETION_NOTES
    if args.mode == "compatibility":
        prompt_text, prompt_ids = args.prompt, base_ids
        n_predict = ctx - len(base_ids)
        ignore_eos = False
        notes.append("EOS stops generation (llama3.cuda stops on BOS)")
    else:
        prompt_text, prompt_ids = scaling_prompt(
            args.prompt, base_ids, ctx, lambda text: tokenize(tools["llama-tokenize"], gguf, text)
        )
        n_predict = args.decode_steps
        ignore_eos = True
        notes.append("scaling prompt pads the text prompt with its last word so it encodes to context_len ids")
        notes.append("--ignore-eos so exactly decode_steps tokens are sampled")
    if not warmup:
        notes.append("--no-warmup: timed run includes llama.cpp first-use CUDA kernel/graph setup")
    kv_ctx = len(prompt_ids) + n_predict

    def cmd_for(with_warmup: bool) -> list[str]:
        return completion_cmd(
            tools["llama-completion"],
            gguf,
            prompt=prompt_text,
            n_predict=n_predict,
            ctx=kv_ctx,
            warmup=with_warmup,
            ignore_eos=ignore_eos,
        )

    warmup_s = 0.0
    for _ in range(max(0, args.warmups)):
        t0 = common.monotonic_s()
        _exec(cmd_for(True))
        warmup_s += common.monotonic_s() - t0

    for _ in range(max(1, args.reps)):
        proc = _exec(cmd_for(warmup))
        gpu = parse_gpu_offload(proc.stderr)
        require_gpu_offload(gpu, kv_type="f32")
        seen_ids = parse_prompt_tokens(proc.stderr)
        if seen_ids != prompt_ids:
            raise SystemExit(f"llama-completion prompt ids {seen_ids} != planned {prompt_ids}")
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
            notes=notes,
            build={**_build_info(gpu), "gguf": str(gguf)},
        )
        validate_result(obj)
        print(common.dumps(obj), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
