"""Verify unpatched llama3.cuda, then run the patched JSON bench in WSL."""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent
TOOLS = HERE.parents[1].parent
sys.path.insert(0, str(TOOLS))

from bench import common  # noqa: E402
from bench.adapters import util  # noqa: E402
from bench.adapters.llama3_cuda import build as cuda_build  # noqa: E402
from bench.schema import validate_result  # noqa: E402


def _quote(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"


def _wsl_run(work: pathlib.Path, args: list[str]) -> tuple[str, str]:
    wsl_dir = util.to_wsl_path(work)
    cmd = " ".join(_quote(a) for a in args)
    script = f"set -euo pipefail; cd {wsl_dir} && ./runcuda {cmd}"
    proc = util.wsl_bash(script, capture=True)
    return proc.stdout, proc.stderr


def verify_unpatched(checkpoint: pathlib.Path, tokenizer: pathlib.Path) -> None:
    exe = cuda_build.build("unpatched")
    work = exe.parent
    dest_ckpt = work / "stories15M.bin"
    dest_tok = work / "tokenizer.bin"
    shutil.copy2(checkpoint, dest_ckpt)
    shutil.copy2(tokenizer, dest_tok)
    stdout, stderr = _wsl_run(work, ["I have a dream"])
    text = stdout.replace("\r\n", "\n").strip("\n")
    if text != common.DREAM_STORY:
        raise SystemExit(f"unpatched text mismatch:\n{text!r}\nexpected:\n{common.DREAM_STORY!r}")
    if "Token count: 50" not in stderr:
        raise SystemExit(f"unpatched missing Token count: 50 in stderr:\n{stderr}")
    if "tokens/s" not in stderr:
        raise SystemExit(f"unpatched missing legacy tokens/s metric:\n{stderr}")
    print("ok unpatched llama3.cuda 50-token output and legacy metric", file=sys.stderr)


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
    p.add_argument("--skip-unpatched-verify", action="store_true")
    args = p.parse_args(argv)

    ckpt = pathlib.Path(args.checkpoint).resolve()
    tok = pathlib.Path(args.tokenizer).resolve()
    sha = common.sha256_file(ckpt)

    if not args.skip_unpatched_verify and args.mode == "compatibility":
        verify_unpatched(ckpt, tok)

    exe = cuda_build.build("patched")
    work = exe.parent
    cmd = [
        "--checkpoint",
        util.to_wsl_path(ckpt),
        "--tokenizer",
        util.to_wsl_path(tok),
        "--mode",
        args.mode,
        "--json",
        "--sha256",
        sha,
        "--warmups",
        str(args.warmups),
        "--reps",
        str(args.reps),
        "-n",
        str(args.n),
        "--decode-steps",
        str(args.decode_steps),
        args.prompt,
    ]
    if args.context is not None:
        cmd.extend(["--context", str(args.context)])
    stdout, stderr = _wsl_run(work, cmd)
    if stderr.strip():
        print(stderr, file=sys.stderr, end="" if stderr.endswith("\n") else "\n")
    n = 0
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        validate_result(obj)
        print(common.dumps(obj), flush=True)
        n += 1
    if n == 0:
        raise SystemExit(f"patched llama3.cuda produced no JSON\nstdout:\n{stdout}\nstderr:\n{stderr}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
