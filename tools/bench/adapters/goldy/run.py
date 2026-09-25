"""Build and run the Goldy JSON benchmark binary."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
TOOLS = HERE.parents[1].parent
ROOT = TOOLS.parent
sys.path.insert(0, str(TOOLS))

from bench import common  # noqa: E402
from bench.adapters.util import run  # noqa: E402
from bench.schema import validate_result  # noqa: E402


def exe_name() -> str:
    return "llama3-goldy-bench.exe" if os.name == "nt" else "llama3-goldy-bench"


def target_dir() -> pathlib.Path:
    env = os.environ.get("CARGO_TARGET_DIR")
    if env:
        return pathlib.Path(env)
    proc = run(
        ["cargo", "metadata", "--format-version", "1", "--no-deps", "--manifest-path", str(ROOT / "Cargo.toml")],
        cwd=ROOT,
        capture=True,
    )
    data = json.loads(proc.stdout)
    return pathlib.Path(data["target_directory"])


def exe_path() -> pathlib.Path:
    return target_dir() / "release" / exe_name()


def features() -> list[str]:
    if os.environ.get("KOBA_BENCH_GOLDY_FEATURES"):
        return os.environ["KOBA_BENCH_GOLDY_FEATURES"].split(",")
    if sys.platform == "darwin":
        return ["metal"]
    return ["cuda"]


def build() -> pathlib.Path:
    feat = features()
    cmd = [
        "cargo",
        "build",
        "--release",
        "--manifest-path",
        str(ROOT / "Cargo.toml"),
        "--bin",
        "llama3-goldy-bench",
        "--features",
        ",".join(feat),
    ]
    run(cmd, cwd=ROOT)
    path = exe_path()
    if not path.exists():
        raise SystemExit(f"missing {path} after cargo build")
    return path


def bench_cmd(
    exe: pathlib.Path,
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
) -> list[str]:
    cmd = [
        str(exe),
        "--checkpoint",
        str(checkpoint),
        "--tokenizer",
        str(tokenizer),
        "--mode",
        mode,
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
    if context is not None:
        cmd.extend(["--context", str(context)])
    return cmd


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=str(common.MODELS / "stories15M.bin"))
    p.add_argument("--tokenizer", default=str(common.MODELS / "tokenizer.bin"))
    p.add_argument("--mode", choices=("compatibility", "scaling"), default="compatibility")
    p.add_argument("--prompt", default=common.DREAM_PROMPT)
    p.add_argument("--context", type=int, default=None)
    p.add_argument("--decode-steps", type=int, default=common.SCALING_DECODE_STEPS)
    p.add_argument("-n", type=int, default=common.COMPAT_TOTAL_POSITIONS)
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--skip-build", action="store_true")
    args = p.parse_args(argv)

    exe = exe_path() if args.skip_build and exe_path().exists() else build()
    cmd = bench_cmd(
        exe,
        checkpoint=pathlib.Path(args.checkpoint).resolve(),
        tokenizer=pathlib.Path(args.tokenizer).resolve(),
        mode=args.mode,
        prompt=args.prompt,
        context=args.context,
        decode_steps=args.decode_steps,
        total_positions=args.n,
        warmups=args.warmups,
        reps=args.reps,
    )
    proc = run(cmd, cwd=ROOT, capture=True, check=False)
    n = 0
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="" if proc.stderr.endswith("\n") else "\n")
    if proc.returncode != 0:
        raise SystemExit(f"goldy bench exited {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        validate_result(obj)
        print(common.dumps(obj), flush=True)
        n += 1
    if n == 0:
        raise SystemExit(f"goldy bench produced no JSON\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
