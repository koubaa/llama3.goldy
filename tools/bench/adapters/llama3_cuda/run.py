"""Verify unpatched llama3.cuda, then run the patched JSON bench (WSL or native Windows)."""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
TOOLS = HERE.parents[1].parent
sys.path.insert(0, str(TOOLS))

from bench import common  # noqa: E402
from bench.adapters import util  # noqa: E402
from bench.adapters.llama3_cuda import build as cuda_build  # noqa: E402
from bench.schema import validate_result  # noqa: E402

RUN_TIMEOUT_S = 600.0


def _quote(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"


def host_path(path: pathlib.Path) -> str:
    """Path as the llama3.cuda binary sees it."""
    return util.to_wsl_path(path) if cuda_build.execution() == "wsl" else str(path)


def run_binary(exe: pathlib.Path, args: list[str]) -> subprocess.CompletedProcess:
    work = exe.parent
    if cuda_build.execution() == "wsl":
        cmd = " ".join(_quote(a) for a in args)
        script = f"set -euo pipefail; cd {_quote(util.to_wsl_path(work))} && ./{exe.name} {cmd}"
        return util.wsl_bash(script, check=False, capture=True, timeout=RUN_TIMEOUT_S)
    return subprocess.run(
        [str(exe), *args],
        cwd=work,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=RUN_TIMEOUT_S,
    )


def _checked(proc: subprocess.CompletedProcess, what: str) -> tuple[str, str]:
    if proc.returncode != 0:
        raise SystemExit(
            f"{what} exited with {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc.stdout, proc.stderr


def verify_unpatched(checkpoint: pathlib.Path, tokenizer: pathlib.Path) -> None:
    exe = cuda_build.build("unpatched")
    work = exe.parent
    # Upstream main() hardcodes ./stories15M.bin and ./tokenizer.bin.
    shutil.copy2(checkpoint, work / "stories15M.bin")
    shutil.copy2(tokenizer, work / "tokenizer.bin")
    stdout, stderr = _checked(run_binary(exe, [common.DREAM_PROMPT]), "unpatched llama3.cuda")
    text = stdout.replace("\r\n", "\n").strip("\n")
    if text != common.DREAM_STORY:
        raise SystemExit(f"unpatched text mismatch:\n{text!r}\nexpected:\n{common.DREAM_STORY!r}")
    if "Token count: 50" not in stderr:
        raise SystemExit(f"unpatched missing Token count: 50 in stderr:\n{stderr}")
    if "tokens/s" not in stderr:
        raise SystemExit(f"unpatched missing legacy tokens/s metric:\n{stderr}")
    print(f"ok unpatched llama3.cuda 50-token output and legacy metric: {stderr.strip()}", file=sys.stderr)


def patched_args(args: argparse.Namespace, ckpt: pathlib.Path, tok: pathlib.Path, sha: str) -> list[str]:
    cmd = [
        "--checkpoint",
        host_path(ckpt),
        "--tokenizer",
        host_path(tok),
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
    ]
    if args.context is not None:
        cmd.extend(["--context", str(args.context)])
    cmd.append(args.prompt)
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
    p.add_argument("--skip-unpatched-verify", action="store_true")
    args = p.parse_args(argv)

    ckpt = pathlib.Path(args.checkpoint).resolve()
    tok = pathlib.Path(args.tokenizer).resolve()
    sha = common.sha256_file(ckpt)
    expected = common.CHECKPOINT_SHA256.get(ckpt.name)
    if expected and sha != expected:
        raise SystemExit(f"checkpoint hash {sha} != pinned {expected}")

    if not args.skip_unpatched_verify and args.mode == "compatibility":
        verify_unpatched(ckpt, tok)

    exe = cuda_build.build("patched")
    build = cuda_build.build_info("patched")
    proc = run_binary(exe, patched_args(args, ckpt, tok, sha))
    if proc.stderr.strip():
        print(proc.stderr, file=sys.stderr, end="" if proc.stderr.endswith("\n") else "\n")
    stdout, _ = _checked(proc, "patched llama3.cuda")
    n = 0
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        obj["checkpoint"]["path"] = str(ckpt)
        obj["build"] = {**obj.get("build", {}), **build}
        validate_result(obj)
        if obj["execution"] != build["execution"]:
            raise SystemExit(f"binary reports execution {obj['execution']!r}, adapter built {build['execution']!r}")
        print(common.dumps(obj), flush=True)
        n += 1
    if n == 0:
        raise SystemExit(f"patched llama3.cuda produced no JSON\nstdout:\n{stdout}\nstderr:\n{proc.stderr}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
