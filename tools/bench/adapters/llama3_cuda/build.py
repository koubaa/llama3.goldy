"""Copy pinned llama3.cuda, splice the JSON bench driver, build in WSL. Does not edit refs/."""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent
TOOLS = HERE.parents[1].parent
sys.path.insert(0, str(TOOLS))

from bench import common  # noqa: E402
from bench.adapters.util import (  # noqa: E402
    CACHE,
    REFS_LLAMA3_CUDA,
    SPLICE_MARKER,
    to_wsl_path,
    verify_ref_commit,
    wsl_bash,
)

UNPATCHED_DIR = CACHE / "llama3.cuda-unpatched"
PATCHED_DIR = CACHE / "llama3.cuda-patched"
TAIL = HERE / "bench_tail.cu"
NVCC_UNPATCHED = "nvcc -DUSE_CUBLAS=1 -g -o runcuda llama3.cu -lm -lcublas"
NVCC_PATCHED = "nvcc -O3 -DNDEBUG -DUSE_CUBLAS=1 -o runcuda llama3.cu -lm -lcublas"


def splice(src: pathlib.Path, dest: pathlib.Path) -> None:
    text = src.read_text(encoding="utf-8", errors="replace")
    idx = text.find(SPLICE_MARKER)
    if idx < 0:
        raise SystemExit(f"{src} missing splice marker {SPLICE_MARKER!r}")
    line_start = text.rfind("\n", 0, idx)
    prefix = text[: line_start + 1] if line_start >= 0 else ""
    tail = TAIL.read_text(encoding="utf-8")
    dest.write_text(prefix + tail, encoding="utf-8", newline="\n")


def write_audit_patch(original: pathlib.Path, patched: pathlib.Path, out: pathlib.Path) -> None:
    import difflib

    a = original.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    b = patched.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    diff = difflib.unified_diff(a, b, fromfile="llama3.cu", tofile="llama3.cu", n=3)
    out.write_text("".join(diff), encoding="utf-8")


def prepare(kind: str) -> pathlib.Path:
    verify_ref_commit(REFS_LLAMA3_CUDA, common.LLAMA3_CUDA_COMMIT)
    src = REFS_LLAMA3_CUDA / "llama3.cu"
    if kind == "unpatched":
        dest_dir = UNPATCHED_DIR
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest_dir / "llama3.cu")
        return dest_dir
    dest_dir = PATCHED_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest_dir / "llama3.cu.orig")
    splice(src, dest_dir / "llama3.cu")
    write_audit_patch(dest_dir / "llama3.cu.orig", dest_dir / "llama3.cu", dest_dir / "llama3.cuda.bench.patch")
    shutil.copy2(dest_dir / "llama3.cuda.bench.patch", HERE / "llama3.cuda.bench.patch")
    return dest_dir


def build(kind: str) -> pathlib.Path:
    dest_dir = prepare(kind)
    nvcc = NVCC_UNPATCHED if kind == "unpatched" else NVCC_PATCHED
    wsl_dir = to_wsl_path(dest_dir)
    wsl_bash(f"set -euo pipefail; cd {wsl_dir} && {nvcc}")
    return dest_dir / "runcuda"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=("unpatched", "patched", "splice-only"))
    args = p.parse_args(argv)
    if args.kind == "splice-only":
        prepare("patched")
        print(PATCHED_DIR / "llama3.cu")
        return 0
    exe = build(args.kind)
    print(exe)
    return 0


if __name__ == "__main__":
    sys.exit(main())
