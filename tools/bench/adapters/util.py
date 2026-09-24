"""Shared adapter helpers. Never writes into refs/."""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys
from typing import Sequence

TOOLS = pathlib.Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(TOOLS))
from bench import common  # noqa: E402

CACHE = common.CACHE
REFS_LLAMA3_CUDA = common.REFS / "llama3.cuda"
REFS_LLAMA_CPP = common.REFS / "llama.cpp"
SPLICE_MARKER = "// utilities: time"
DREAM_STORY = common.DREAM_STORY


def to_wsl_path(path: pathlib.Path | str) -> str:
    p = pathlib.Path(path).resolve()
    s = str(p)
    if len(s) >= 2 and s[1] == ":":
        return "/mnt/" + s[0].lower() + s[2:].replace("\\", "/")
    return s.replace("\\", "/")


def which(name: str) -> str | None:
    return shutil.which(name)


def run(
    cmd: Sequence[str],
    *,
    cwd: pathlib.Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
    capture: bool = False,
) -> subprocess.CompletedProcess:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        env=merged,
        check=check,
        text=True,
        capture_output=capture,
    )


WSL_DISTRO_ENV = "KOBA_BENCH_WSL_DISTRO"
DEFAULT_WSL_DISTRO = "Ubuntu"
# Login shells in a fresh distro do not necessarily have the CUDA toolkit on PATH.
WSL_CUDA_PATH = "export PATH=/usr/local/cuda/bin:$PATH; "


def wsl_available() -> bool:
    return which("wsl") is not None


def wsl_distro() -> str:
    return os.environ.get(WSL_DISTRO_ENV) or DEFAULT_WSL_DISTRO


def wsl_bash_cmd(script: str, distro: str | None = None) -> list[str]:
    # The default distro may be docker-desktop, which has no bash; always name one.
    return ["wsl", "-d", distro or wsl_distro(), "-e", "bash", "-lc", WSL_CUDA_PATH + script]


def wsl_bash(
    script: str,
    *,
    check: bool = True,
    capture: bool = False,
    timeout: float | None = None,
) -> subprocess.CompletedProcess:
    if not wsl_available():
        raise SystemExit("WSL is required for the llama3.cuda WSL build")
    return subprocess.run(
        wsl_bash_cmd(script),
        check=check,
        capture_output=capture,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def wsl_has_cuda_toolchain(timeout: float = 60.0) -> bool:
    if not wsl_available():
        return False
    try:
        proc = subprocess.run(
            wsl_bash_cmd("command -v nvcc >/dev/null && command -v g++ >/dev/null"),
            capture_output=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def verify_ref_commit(repo: pathlib.Path, expected: str) -> None:
    got = run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture=True).stdout.strip()
    if got != expected:
        raise SystemExit(f"{repo} HEAD {got} != pinned {expected}")
