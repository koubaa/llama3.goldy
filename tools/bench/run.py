"""Run the TinyStories engine matrix and write JSON Lines plus a report."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Iterable

HERE = pathlib.Path(__file__).resolve().parent
TOOLS = HERE.parent
ROOT = TOOLS.parent
sys.path.insert(0, str(TOOLS))

from bench import common  # noqa: E402
from bench.adapters.llama3_cuda import build as llama3_cuda_build  # noqa: E402
from bench.adapters.llama_cpp import build as llama_cpp_build  # noqa: E402
from bench.pytorch import env as torch_env  # noqa: E402
from bench.report import write_report  # noqa: E402
from bench.schema import validate_result  # noqa: E402

MODELS = {
    "15m": "stories15M.bin",
    "42m": "stories42M.bin",
    "110m": "stories110M.bin",
}

ALL_ENGINES = list(common.ENGINES)


def _which(name: str) -> str | None:
    return shutil.which(name)


def _run_capture(cmd: list[str]) -> str:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    out = (proc.stdout or "") + (proc.stderr or "")
    return out.strip()


def capture_metadata() -> dict[str, Any]:
    gpu = _run_capture(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    rustc = _run_capture(["rustc", "--version"])
    cargo = _run_capture(["cargo", "--version"])
    nvcc = _run_capture(["nvcc", "--version"])
    info = torch_env.probe()
    if "error" in info:
        torch_ver = info["error"]
    else:
        torch_ver = f"{info['torch']} cuda={info.get('cuda')} triton={info.get('triton')} ({info['python']})"
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "os": f"{platform.system()} {platform.release()} {platform.version()}",
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "gpu": gpu,
        "rustc": rustc,
        "cargo": cargo,
        "nvcc": nvcc,
        "torch": torch_ver,
        "cwd": str(ROOT),
        "llama3_cuda_commit": common.LLAMA3_CUDA_COMMIT,
        "llama_cpp_commit": common.LLAMA_CPP_COMMIT,
    }


def required_assets(models: Iterable[str]) -> list[pathlib.Path]:
    paths = [common.MODELS / "tokenizer.bin"]
    for key in models:
        paths.append(common.MODELS / MODELS[key])
    return paths


def verify_assets(models: Iterable[str]) -> None:
    fetch = TOOLS / "fetch_assets.py"
    if not (common.MODELS / "tokenizer.bin").exists():
        raise SystemExit("missing models/tokenizer.bin; run python tools/fetch_assets.py")
    subprocess.run([sys.executable, str(fetch), "--verify-only", "--only", "tokenizer," + ",".join(models)], check=True)


def engine_ready(engine: str) -> tuple[bool, str]:
    if engine == "goldy":
        if not _which("cargo"):
            return False, "cargo not on PATH"
        return True, ""
    if engine.startswith("pytorch"):
        return torch_env.torch_ready(compile=engine == "pytorch-compile")
    if engine == "llama3.cuda":
        src = common.REFS / "llama3.cuda" / "llama3.cu"
        if not src.exists():
            return False, f"missing {src}"
        if llama3_cuda_build.detect_execution() is None:
            return False, "no WSL nvcc/g++ toolchain and no native nvcc + vcvars64"
        return True, ""
    if engine.startswith("llama.cpp"):
        src = common.REFS / "llama.cpp"
        if not src.exists():
            return False, f"missing {src}"
        try:
            for target in llama_cpp_build.TARGETS:
                llama_cpp_build.find_exe(target)
            return True, ""
        except FileNotFoundError:
            pass
        if os.name == "nt":
            try:
                llama_cpp_build.find_vs()
            except SystemExit as exc:
                return False, str(exc)
        elif not _which("cmake"):
            return False, "cmake not on PATH"
        return True, ""
    return False, f"unknown engine {engine}"


def run_correctness_gates() -> None:
    subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", str(HERE / "tests"), "-v"],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(["cargo", "test", "--offline", "--manifest-path", str(ROOT / "Cargo.toml")], cwd=ROOT, check=True)


def invoke_engine(
    engine: str,
    *,
    checkpoint: pathlib.Path,
    tokenizer: pathlib.Path,
    mode: str,
    prompt: str,
    context: int | None,
    decode_steps: int,
    total_positions: int,
    warmups: int,
    reps: int,
) -> list[dict[str, Any]]:
    py = sys.executable
    env = None
    if engine == "goldy":
        script = HERE / "adapters" / "goldy" / "run.py"
        cmd = [py, str(script)]
        if getattr(invoke_engine, "_goldy_built", False):
            cmd.append("--skip-build")
    elif engine.startswith("pytorch"):
        compile = engine == "pytorch-compile"
        cmd = torch_env.bench_command("--device", torch_env.bench_device(), compile=compile)
        if compile:
            cmd.append("--require-compile")
        env = torch_env.bench_env(compile=compile)
    elif engine == "llama3.cuda":
        script = HERE / "adapters" / "llama3_cuda" / "run.py"
        cmd = [py, str(script)]
        if mode != "compatibility":
            cmd.append("--skip-unpatched-verify")
    elif engine == "llama.cpp":
        script = HERE / "adapters" / "llama_cpp" / "run.py"
        cmd = [py, str(script)]
    elif engine == "llama.cpp-bench":
        script = HERE / "adapters" / "llama_cpp" / "run.py"
        cmd = [py, str(script), "--mode", "bench"]
    else:
        raise SystemExit(f"unknown engine {engine}")

    if engine != "llama.cpp-bench":
        cmd.extend(["--mode", mode])
    cmd.extend(
        [
            "--checkpoint",
            str(checkpoint),
            "--tokenizer",
            str(tokenizer),
            "--prompt",
            prompt,
            "--decode-steps",
            str(decode_steps),
            "-n",
            str(total_positions),
            "--warmups",
            str(warmups),
            "--reps",
            str(reps),
        ]
    )
    if context is not None:
        cmd.extend(["--context", str(context)])

    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="" if proc.stderr.endswith("\n") else "\n")
    if proc.returncode != 0:
        raise SystemExit(
            f"{engine} failed ({proc.returncode})\ncmd: {' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    rows: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        obj = json.loads(line)
        validate_result(obj)
        rows.append(obj)
    if not rows:
        raise SystemExit(f"{engine} produced no JSON Lines\nstdout:\n{proc.stdout}")
    if engine == "goldy":
        invoke_engine._goldy_built = True
    return rows


def matrix(
    *,
    engines: list[str],
    models: list[str],
    contexts: list[int],
    smoke: bool,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    if "15m" in models:
        for engine in engines:
            jobs.append(
                {
                    "engine": engine,
                    "model": "15m",
                    "mode": "compatibility" if engine != "llama.cpp-bench" else "bench",
                    "context": None,
                    "checkpoint": MODELS["15m"],
                }
            )
    if not smoke:
        for model in models:
            for ctx in contexts:
                for engine in engines:
                    jobs.append(
                        {
                            "engine": engine,
                            "model": model,
                            "mode": "scaling" if engine != "llama.cpp-bench" else "bench",
                            "context": ctx,
                            "checkpoint": MODELS[model],
                        }
                    )
    else:
        for engine in engines:
            jobs.append(
                {
                    "engine": engine,
                    "model": "15m",
                    "mode": "scaling" if engine != "llama.cpp-bench" else "bench",
                    "context": contexts[0] if contexts else 8,
                    "checkpoint": MODELS["15m"],
                }
            )
    return jobs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engines", default=",".join(ALL_ENGINES), help="comma-separated engine list")
    p.add_argument("--models", default="15m,42m,110m", help="comma-separated: 15m,42m,110m")
    p.add_argument(
        "--contexts",
        default=",".join(str(c) for c in common.SCALING_CONTEXT_LENGTHS),
    )
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--decode-steps", type=int, default=common.SCALING_DECODE_STEPS)
    p.add_argument("--out-dir", default=str(ROOT / "benchmarks"))
    p.add_argument("--skip-gates", action="store_true")
    p.add_argument("--skip-missing", action="store_true", default=True)
    p.add_argument("--require-all", action="store_true")
    p.add_argument("--smoke", action="store_true", help="15M compatibility + one scaling context")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    contexts = [int(c) for c in args.contexts.split(",") if c.strip()]
    unknown = [e for e in engines if e not in ALL_ENGINES]
    if unknown:
        raise SystemExit(f"unknown engines {unknown}; choose from {ALL_ENGINES}")
    unknown_m = [m for m in models if m not in MODELS]
    if unknown_m:
        raise SystemExit(f"unknown models {unknown_m}")

    out_dir = pathlib.Path(args.out_dir)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_gates:
        run_correctness_gates()
    verify_assets(models if not args.smoke else ["15m"])

    meta = capture_metadata()
    (out_dir / "machine.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(common.dumps({"metadata": meta}), flush=True)

    selected: list[str] = []
    skipped: list[str] = []
    seen_skip: set[str] = set()
    for engine in engines:
        ok, reason = engine_ready(engine)
        if ok:
            selected.append(engine)
        else:
            if f"{engine}: {reason}" not in seen_skip:
                skipped.append(f"{engine}: {reason}")
                seen_skip.add(f"{engine}: {reason}")
            if args.require_all or not args.skip_missing:
                raise SystemExit(f"engine {engine} not ready: {reason}")
            print(f"skip {engine}: {reason}", file=sys.stderr)

    jobs = matrix(engines=selected, models=["15m"] if args.smoke else models, contexts=contexts, smoke=args.smoke)
    if args.dry_run:
        for job in jobs:
            print(common.dumps(job), flush=True)
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_path = raw_dir / f"{stamp}.jsonl"
    results: list[dict[str, Any]] = []
    tokenizer = common.MODELS / "tokenizer.bin"
    with raw_path.open("w", encoding="utf-8") as raw:
        for job in jobs:
            ckpt = common.MODELS / job["checkpoint"]
            print(f"run {job['engine']} {job['mode']} {job['checkpoint']} ctx={job['context']}", file=sys.stderr)
            try:
                rows = invoke_engine(
                    job["engine"],
                    checkpoint=ckpt,
                    tokenizer=tokenizer,
                    mode="compatibility" if job["mode"] == "compatibility" else "scaling",
                    prompt=common.DREAM_PROMPT,
                    context=job["context"],
                    decode_steps=args.decode_steps,
                    total_positions=common.COMPAT_TOTAL_POSITIONS,
                    warmups=args.warmups,
                    reps=args.reps,
                )
            except SystemExit as exc:
                print(f"error {job}: {exc}", file=sys.stderr)
                if args.require_all:
                    raise
                continue
            for row in rows:
                raw.write(common.dumps(row) + "\n")
                raw.flush()
                results.append(row)

    report_md, summary = write_report(
        results,
        metadata=meta,
        skipped=skipped,
        out_md=out_dir / "baseline.md",
        out_json=out_dir / "baseline.json",
    )
    print(f"wrote {raw_path}", file=sys.stderr)
    print(f"wrote {report_md}", file=sys.stderr)
    print(common.dumps({"jobs": len(jobs), "rows": len(results), "skipped": skipped, "summary": summary.get("headline")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
