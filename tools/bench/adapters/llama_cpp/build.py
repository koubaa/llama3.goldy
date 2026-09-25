"""Native Release GPU build of pinned llama.cpp. Does not edit refs/.

Windows: Ninja + CUDA inside a captured ``vcvars64.bat`` environment (VS 2022 or
newer, located with vswhere). The VS-bundled CMake/Ninja are preferred over
PATH because older CMake releases do not know newer MSVC toolsets.

macOS: Ninja + Metal (no CUDA).

Environment overrides:
  LLAMA_CPP_CUDA_ARCH   CMAKE_CUDA_ARCHITECTURES (default 89, RTX 40xx/Ada)
  LLAMA_CPP_VS_PATH     Visual Studio installation root (skips vswhere)
  LLAMA_CPP_CMAKE       cmake executable
  LLAMA_CPP_NINJA       ninja executable
"""

from __future__ import annotations

import argparse
import functools
import os
import pathlib
import shutil
import subprocess
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
    "llama-tokenize",
)
DEFAULT_CUDA_ARCH = "89"
# CUDA 13.1 host_config.h rejects _MSC_VER >= 1950 (VS 2026 ships 19.5x), and the
# VS 2026 STL static_asserts "STL1002: expected CUDA 13.2 or newer" under nvcc
# (cudafe++ 13.1 crashes with ACCESS_VIOLATION on that diagnostic).
WINDOWS_CUDA_FLAGS = ("-allow-unsupported-compiler", "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH")
VSWHERE = pathlib.Path(
    os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
VS_CMAKE_REL = pathlib.Path("Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe")
VS_NINJA_REL = pathlib.Path("Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe")


def exe_name(stem: str) -> str:
    return stem + (".exe" if os.name == "nt" else "")


def find_exe(stem: str) -> pathlib.Path:
    name = exe_name(stem)
    candidates = [
        BUILD_DIR / "bin" / name,
        BUILD_DIR / "bin" / "Release" / name,
        BUILD_DIR / "Release" / "bin" / name,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"{name} not found under {BUILD_DIR}")


def cuda_arch() -> str:
    return os.environ.get("LLAMA_CPP_CUDA_ARCH", DEFAULT_CUDA_ARCH)


@functools.lru_cache(maxsize=1)
def find_vs() -> pathlib.Path:
    override = os.environ.get("LLAMA_CPP_VS_PATH")
    if override:
        return pathlib.Path(override)
    if not VSWHERE.exists():
        raise SystemExit(f"vswhere not found at {VSWHERE}; set LLAMA_CPP_VS_PATH")
    proc = run(
        [
            str(VSWHERE),
            "-latest",
            "-products",
            "*",
            "-version",
            "[17.0,)",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ],
        capture=True,
    )
    path = proc.stdout.strip().splitlines()
    if not path:
        raise SystemExit("vswhere found no Visual Studio 17+ with the x64 C++ toolset")
    return pathlib.Path(path[0].strip())


@functools.lru_cache(maxsize=1)
def msvc_env() -> dict[str, str]:
    """Environment of a VS x64 developer prompt, keys upper-cased (Windows env is case-insensitive)."""
    vcvars = find_vs() / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    if not vcvars.exists():
        raise SystemExit(f"{vcvars} missing")
    # A string command is passed verbatim to CreateProcess; /s keeps cmd from mangling the quoted path.
    proc = subprocess.run(
        f'cmd /d /s /c ""{vcvars}" >nul 2>&1 && set"',
        capture_output=True,
        text=True,
        check=False,
    )
    env: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        key, sep, value = line.partition("=")
        if sep and key:
            env[key.upper()] = value
    if proc.returncode != 0 or "VCTOOLSINSTALLDIR" not in env:
        raise SystemExit(f"{vcvars} failed (exit {proc.returncode}): {proc.stderr.strip()}")
    return env


def build_env() -> dict[str, str] | None:
    if os.name != "nt":
        return None
    env = {k.upper(): v for k, v in os.environ.items()}
    env.update(msvc_env())
    return env


def _tool(env_var: str, vs_rel: pathlib.Path, name: str) -> str:
    override = os.environ.get(env_var)
    if override:
        return override
    if os.name == "nt":
        bundled = find_vs() / vs_rel
        if bundled.exists():
            return str(bundled)
    return shutil.which(name) or name


def cmake_exe() -> str:
    return _tool("LLAMA_CPP_CMAKE", VS_CMAKE_REL, "cmake")


def ninja_exe() -> str:
    return _tool("LLAMA_CPP_NINJA", VS_NINJA_REL, "ninja")


def cmake_configure_cmd() -> list[str]:
    cmd = [
        cmake_exe(),
        "-S",
        str(REFS_LLAMA_CPP),
        "-B",
        str(BUILD_DIR),
        "-G",
        "Ninja",
        f"-DCMAKE_MAKE_PROGRAM={ninja_exe()}",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLLAMA_BUILD_EXAMPLES=ON",
        "-DLLAMA_BUILD_TOOLS=ON",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_SERVER=OFF",
        "-DLLAMA_BUILD_APP=OFF",
        "-DLLAMA_OPENSSL=OFF",
    ]
    if sys.platform == "darwin":
        cmd += [
            "-DGGML_METAL=ON",
            "-DGGML_CUDA=OFF",
            "-DGGML_METAL_EMBED_LIBRARY=ON",
        ]
    else:
        cmd += [
            "-DGGML_CUDA=ON",
            f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch()}",
        ]
    if os.name == "nt":
        cmd += [
            "-DCMAKE_C_COMPILER=cl",
            "-DCMAKE_CXX_COMPILER=cl",
            f"-DCMAKE_CUDA_FLAGS={' '.join(WINDOWS_CUDA_FLAGS)}",
        ]
    return cmd


def build_cmd() -> list[str]:
    return [cmake_exe(), "--build", str(BUILD_DIR), "--config", "Release", "--target", *TARGETS]


def _run_logged(cmd: list[str]) -> None:
    """Stream tool output to stderr so callers can keep stdout for JSON Lines."""
    env = build_env()
    subprocess.run(cmd, env=env, check=True, stdout=sys.stderr, stderr=sys.stderr)


def configure() -> None:
    verify_ref_commit(REFS_LLAMA_CPP, common.LLAMA_CPP_COMMIT)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    _run_logged(cmake_configure_cmd())


def build() -> dict[str, pathlib.Path]:
    configure()
    _run_logged(build_cmd())
    return {stem: find_exe(stem) for stem in TARGETS}


def ensure_built() -> dict[str, pathlib.Path]:
    """Reuse existing binaries; configure/build only when one is missing."""
    try:
        return {stem: find_exe(stem) for stem in TARGETS}
    except FileNotFoundError:
        return build()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configure-only", action="store_true")
    p.add_argument("--print-cmd", action="store_true")
    args = p.parse_args(argv)
    if args.print_cmd:
        print(subprocess.list2cmdline(cmake_configure_cmd()))
        print(subprocess.list2cmdline(build_cmd()))
        return 0
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
