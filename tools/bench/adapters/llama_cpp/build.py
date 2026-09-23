"""Native Release CUDA build of pinned llama.cpp. Does not edit refs/."""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
TOOLS = HERE.parents[1].parent
sys.path.insert(0, str(TOOLS))

from bench import common  # noqa: E402
from bench.adapters.util import CACHE, REFS_LLAMA_CPP, run, verify_ref_commit  # noqa: E402

BUILD_DIR = CACHE / "llama.cpp-build"
TARGETS = (
    "llama-convert-llama2c-to-ggml",
    "llama-completion",
    "llama-bench",
)


def exe_name(stem: str) -> str:
    return stem + (".exe" if os.name == "nt" else "")


def find_exe(stem: str) -> pathlib.Path:
    name = exe_name(stem)
    candidates = [
        BUILD_DIR / "bin" / "Release" / name,
        BUILD_DIR / "bin" / name,
        BUILD_DIR / "Release" / "bin" / name,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"{name} not found under {BUILD_DIR}")


def configure() -> None:
    verify_ref_commit(REFS_LLAMA_CPP, common.LLAMA_CPP_COMMIT)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    cmake = [
        "cmake",
        "-S",
        str(REFS_LLAMA_CPP),
        "-B",
        str(BUILD_DIR),
        "-DGGML_CUDA=ON",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLLAMA_BUILD_EXAMPLES=ON",
        "-DLLAMA_BUILD_TOOLS=ON",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_SERVER=OFF",
    ]
    run(cmake)


def build() -> dict[str, pathlib.Path]:
    configure()
    cmd = [
        "cmake",
        "--build",
        str(BUILD_DIR),
        "--config",
        "Release",
        "--target",
        *TARGETS,
        "-j",
    ]
    run(cmd)
    return {stem: find_exe(stem) for stem in TARGETS}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--configure-only", action="store_true")
    args = p.parse_args(argv)
    if args.configure_only:
        configure()
        print(BUILD_DIR)
        return 0
    found = build()
    for name, path in found.items():
        print(f"{name} {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
