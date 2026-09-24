"""Copy pinned llama3.cuda, splice the JSON bench driver, build in WSL or natively.

Never edits refs/. The unpatched build compiles a byte-identical copy of llama3.cu.
Execution is chosen by KOBA_BENCH_LLAMA3_CUDA_EXECUTION (auto|wsl|native, default auto):
auto prefers WSL (the documented protocol) when the distro has nvcc + g++, otherwise a
native Windows nvcc + MSVC build that satisfies llama3.cu's POSIX includes via win_shim/.
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
from bench.adapters import util  # noqa: E402
from bench.adapters.util import (  # noqa: E402
    CACHE,
    REFS_LLAMA3_CUDA,
    SPLICE_MARKER,
    to_wsl_path,
    verify_ref_commit,
)

UNPATCHED_DIR = CACHE / "llama3.cuda-unpatched"
PATCHED_DIR = CACHE / "llama3.cuda-patched"
TAIL = HERE / "bench_tail.cu"
WIN_SHIM = HERE / "win_shim"
CCCL_COMPAT = WIN_SHIM / "cccl_compat.h"

EXECUTION_ENV = "KOBA_BENCH_LLAMA3_CUDA_EXECUTION"
ARCH_ENV = "KOBA_BENCH_CUDA_ARCH"
VSWHERE = pathlib.Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe")

# Upstream Makefile: nvcc -DUSE_CUBLAS=1 -g -o runcuda llama3.cu -lm -lcublas
UNPATCHED_OPT = ["-g"]
PATCHED_OPT = ["-O3", "-DNDEBUG"]


def cuda_arch() -> str:
    return os.environ.get(ARCH_ENV) or "native"


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
    out.write_text("".join(diff), encoding="utf-8", newline="\n")


def prepare(kind: str) -> pathlib.Path:
    verify_ref_commit(REFS_LLAMA3_CUDA, common.LLAMA3_CUDA_COMMIT)
    src = REFS_LLAMA3_CUDA / "llama3.cu"
    if kind == "unpatched":
        dest_dir = UNPATCHED_DIR
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest_dir / "llama3.cu")
        if (dest_dir / "llama3.cu").read_bytes() != src.read_bytes():
            raise SystemExit("unpatched copy is not byte-identical to refs/llama3.cuda/llama3.cu")
        return dest_dir
    dest_dir = PATCHED_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest_dir / "llama3.cu.orig")
    splice(src, dest_dir / "llama3.cu")
    write_audit_patch(dest_dir / "llama3.cu.orig", dest_dir / "llama3.cu", dest_dir / "llama3.cuda.bench.patch")
    shutil.copyfile(dest_dir / "llama3.cuda.bench.patch", HERE / "llama3.cuda.bench.patch")
    return dest_dir


# ---------------------------------------------------------------------------
# Native Windows toolchain (nvcc + MSVC via vcvars64)
# ---------------------------------------------------------------------------


def find_vcvars64() -> pathlib.Path | None:
    if os.name != "nt" or not VSWHERE.exists():
        return None
    proc = subprocess.run(
        [
            str(VSWHERE),
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ],
        capture_output=True,
        text=True,
    )
    for line in proc.stdout.splitlines():
        bat = pathlib.Path(line.strip()) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
        if bat.exists():
            return bat
    return None


@functools.lru_cache(maxsize=1)
def vcvars_env() -> dict[str, str]:
    bat = find_vcvars64()
    if bat is None:
        raise SystemExit("vcvars64.bat not found (vswhere + VC.Tools.x86.x64 required)")
    proc = subprocess.run(
        f'cmd /s /c ""{bat}" >nul && set"',
        capture_output=True,
        text=True,
        errors="replace",
    )
    env = dict(line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line and not line.startswith("="))
    if proc.returncode != 0 or not shutil.which("cl", path=env.get("Path") or env.get("PATH")):
        raise SystemExit(f"vcvars64.bat did not yield cl.exe (rc={proc.returncode})\n{proc.stderr}")
    return env


def native_available() -> bool:
    return os.name == "nt" and shutil.which("nvcc") is not None and find_vcvars64() is not None


def native_nvcc_cmd(kind: str, out_dir: pathlib.Path) -> list[str]:
    opt = UNPATCHED_OPT if kind == "unpatched" else PATCHED_OPT
    return [
        "nvcc",
        # CUDA 13.1 predates MSVC 14.5x (VS 2026); its STL also static_asserts CUDA >= 13.2
        # (and cudafe++ 13.1 crashes on that assert in C++20 mode instead of reporting it).
        "-allow-unsupported-compiler",
        "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH",
        # MSVC only accepts llama3.cu's `{.str = str}` designated initializer in C++20;
        # gcc takes it as an extension, so the Linux build needs no flag.
        "-std=c++20",
        *opt,
        "-DUSE_CUBLAS=1",
        f"-arch={cuda_arch()}",
        "-Xcompiler",
        "/Zc:preprocessor",
        "-D_CRT_SECURE_NO_WARNINGS",
        "-D_CRT_NONSTDC_NO_DEPRECATE",
        f"-I{WIN_SHIM}",
        "-include",
        str(CCCL_COMPAT),
        "-o",
        str(out_dir / "runcuda.exe"),
        "llama3.cu",
        str(out_dir / "posix_shim.obj"),
        "-lcublas",
    ]


def native_shim_cmd(out_dir: pathlib.Path) -> list[str]:
    return ["cl", "/nologo", "/O2", "/c", f"/Fo{out_dir / 'posix_shim.obj'}", str(WIN_SHIM / "posix_shim.c")]


def build_native(kind: str, dest_dir: pathlib.Path) -> pathlib.Path:
    env = vcvars_env()
    search = env.get("Path") or env.get("PATH")
    for cmd in (native_shim_cmd(dest_dir), native_nvcc_cmd(kind, dest_dir)):
        print("+ " + subprocess.list2cmdline(cmd), file=sys.stderr, flush=True)
        exe = shutil.which(cmd[0], path=search) or cmd[0]
        subprocess.run([exe, *cmd[1:]], cwd=dest_dir, env=env, check=True)
    return dest_dir / "runcuda.exe"


# ---------------------------------------------------------------------------
# WSL toolchain
# ---------------------------------------------------------------------------


def wsl_nvcc_script(kind: str, dest_dir: pathlib.Path) -> str:
    opt = " ".join(UNPATCHED_OPT if kind == "unpatched" else PATCHED_OPT)
    # -include only (never -I win_shim): the shim headers would shadow real unistd.h / sys/mman.h.
    return (
        f"set -euo pipefail; cd '{to_wsl_path(dest_dir)}' && "
        f"nvcc {opt} -DUSE_CUBLAS=1 -arch={cuda_arch()} -include '{to_wsl_path(CCCL_COMPAT)}' "
        f"-o runcuda llama3.cu -lm -lcublas"
    )


def build_wsl(kind: str, dest_dir: pathlib.Path) -> pathlib.Path:
    script = wsl_nvcc_script(kind, dest_dir)
    print(f"+ wsl -d {util.wsl_distro()}: {script}", file=sys.stderr, flush=True)
    util.wsl_bash(script)
    return dest_dir / "runcuda"


# ---------------------------------------------------------------------------


def detect_execution() -> str | None:
    """Return 'wsl' or 'native' for the build this machine supports, or None."""
    want = (os.environ.get(EXECUTION_ENV) or "auto").lower()
    if want not in ("auto", "wsl", "native"):
        raise SystemExit(f"{EXECUTION_ENV} must be auto|wsl|native, got {want!r}")
    if want in ("auto", "wsl") and util.wsl_has_cuda_toolchain():
        return "wsl"
    if want in ("auto", "native") and native_available():
        return "native"
    return None


@functools.lru_cache(maxsize=1)
def execution() -> str:
    got = detect_execution()
    if got is None:
        raise SystemExit(
            f"llama3.cuda: no toolchain (WSL distro {util.wsl_distro()!r} lacks nvcc/g++ "
            "and native nvcc + vcvars64 not found)"
        )
    return got


def build_info(kind: str = "patched") -> dict[str, object]:
    ex = execution()
    info: dict[str, object] = {
        "commit": common.LLAMA3_CUDA_COMMIT,
        "execution": ex,
        "opt": " ".join(UNPATCHED_OPT if kind == "unpatched" else PATCHED_OPT),
        "cublas": True,
        "arch": cuda_arch(),
    }
    if ex == "native":
        info["host_compiler"] = "msvc"
        info["nvcc"] = subprocess.list2cmdline(native_nvcc_cmd(kind, pathlib.Path(".")))
        info["posix_shim"] = "tools/bench/adapters/llama3_cuda/win_shim"
    else:
        info["wsl_distro"] = util.wsl_distro()
        info["nvcc"] = wsl_nvcc_script(kind, pathlib.Path("."))
    return info


def _build_key(kind: str, dest_dir: pathlib.Path) -> str:
    import hashlib

    h = hashlib.sha256()
    h.update(execution().encode())
    if execution() == "native":
        h.update(subprocess.list2cmdline(native_shim_cmd(dest_dir) + native_nvcc_cmd(kind, dest_dir)).encode())
        inputs = [dest_dir / "llama3.cu", *sorted(WIN_SHIM.rglob("*.[ch]"))]
    else:
        h.update(util.wsl_distro().encode())
        h.update(wsl_nvcc_script(kind, dest_dir).encode())
        inputs = [dest_dir / "llama3.cu", CCCL_COMPAT]
    for path in inputs:
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def build(kind: str, *, force: bool = False) -> pathlib.Path:
    dest_dir = prepare(kind)
    exe = dest_dir / ("runcuda.exe" if execution() == "native" else "runcuda")
    stamp = dest_dir / "build.stamp"
    key = _build_key(kind, dest_dir)
    if not force and exe.exists() and stamp.exists() and stamp.read_text().strip() == key:
        return exe
    stamp.unlink(missing_ok=True)
    exe = build_native(kind, dest_dir) if execution() == "native" else build_wsl(kind, dest_dir)
    stamp.write_text(key + "\n")
    return exe


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("kind", choices=("unpatched", "patched", "splice-only", "probe"))
    p.add_argument("--force", action="store_true", help="rebuild even if the build stamp matches")
    args = p.parse_args(argv)
    if args.kind == "probe":
        got = detect_execution()
        if got is None:
            print("llama3.cuda toolchain not found", file=sys.stderr)
            return 1
        print(got)
        return 0
    if args.kind == "splice-only":
        prepare("patched")
        print(PATCHED_DIR / "llama3.cu")
        return 0
    exe = build(args.kind, force=args.force)
    print(exe)
    return 0


if __name__ == "__main__":
    sys.exit(main())
