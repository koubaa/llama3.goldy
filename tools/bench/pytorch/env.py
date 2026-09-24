"""Locate the isolated CUDA-torch venv and build the environment bench.py runs in.

The global interpreter may carry a CPU-only torch; timed PyTorch runs use a venv at
tools/bench/.cache/torch-venv with a CUDA wheel (and triton-windows for --compile).
"""

from __future__ import annotations

import functools
import json
import os
import pathlib
import subprocess
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
BENCH = HERE.parent
CACHE = BENCH / ".cache"
VENV = CACHE / "torch-venv"
BENCH_PY = HERE / "bench.py"
ENV_OVERRIDE = "KOBA_BENCH_TORCH_PYTHON"
VCVARS_OVERRIDE = "KOBA_BENCH_VCVARS"
VCVARS_CANDIDATES = (
    r"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat",
    r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
    r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
)


def torch_python() -> pathlib.Path | None:
    """Interpreter for PyTorch benches: $KOBA_BENCH_TORCH_PYTHON, else the venv, else None."""
    override = os.environ.get(ENV_OVERRIDE)
    if override:
        path = pathlib.Path(override)
        return path if path.exists() else None
    for rel in (("Scripts", "python.exe"), ("bin", "python")):
        path = VENV.joinpath(*rel)
        if path.exists():
            return path
    return None


def vcvars_path() -> pathlib.Path | None:
    override = os.environ.get(VCVARS_OVERRIDE)
    candidates = (override,) if override else VCVARS_CANDIDATES
    for c in candidates:
        if c and pathlib.Path(c).exists():
            return pathlib.Path(c)
    return None


@functools.lru_cache(maxsize=1)
def _vcvars_env() -> dict[str, str] | None:
    bat = vcvars_path()
    if os.name != "nt" or bat is None:
        return None
    proc = subprocess.run(
        f'cmd /d /s /c ""{bat}" >nul 2>&1 && set"',
        capture_output=True,
        text=True,
        shell=False,
    )
    if proc.returncode != 0:
        return None
    env: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        key, sep, value = line.partition("=")
        if sep and key:
            env[key] = value
    return env


def bench_env(
    *, compile: bool = False, msvc: bool = False, base: dict[str, str] | None = None
) -> dict[str, str]:
    """Environment for launching bench.py.

    Inductor/Triton caches go to tools/bench/.cache (short, private paths). The CUDA
    inductor path (Python wrapper + Triton kernels, launcher built by triton-windows'
    bundled TinyCC) does not need MSVC; pass msvc=True to apply vcvars64 anyway (needed
    only for CPU inductor or cpp_wrapper, which the bench does not use).
    """
    env = dict(os.environ if base is None else base)
    env.setdefault("TORCHINDUCTOR_CACHE_DIR", str(CACHE / "torchinductor"))
    env.setdefault("TRITON_CACHE_DIR", str(CACHE / "triton"))
    env.setdefault("PYTHONIOENCODING", "utf-8")
    if compile and msvc and os.name == "nt" and not _has_cl(env):
        vc = _vcvars_env()
        if vc:
            merged = {k.upper(): k for k in env}
            for key, value in vc.items():
                env[merged.get(key.upper(), key)] = value
    return env


def _has_cl(env: dict[str, str]) -> bool:
    path = next((v for k, v in env.items() if k.upper() == "PATH"), "")
    return any((pathlib.Path(p) / "cl.exe").exists() for p in path.split(os.pathsep) if p)


def process_affinity_mask() -> str | None:
    """Hex CPU affinity mask of this process (Windows only, read-only), e.g. '0xfffffff'."""
    if os.name != "nt":
        return None
    import ctypes

    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    proc_mask = ctypes.c_size_t()
    sys_mask = ctypes.c_size_t()
    ok = k32.GetProcessAffinityMask(
        ctypes.c_void_p(k32.GetCurrentProcess()), ctypes.byref(proc_mask), ctypes.byref(sys_mask)
    )
    return hex(proc_mask.value) if ok else None


def bench_command(*args: str, compile: bool = False, python: pathlib.Path | None = None) -> list[str]:
    """argv for bench.py under the torch interpreter (falls back to the current one)."""
    import sys

    py = python or torch_python() or pathlib.Path(sys.executable)
    cmd = [str(py), str(BENCH_PY), *args]
    if compile and "--compile" not in args:
        cmd.append("--compile")
    return cmd


_PROBE = (
    "import json, torch\n"
    "info = {'torch': torch.__version__, 'cuda': torch.version.cuda,"
    " 'cuda_available': torch.cuda.is_available()}\n"
    "if info['cuda_available']:\n"
    "    info['device_name'] = torch.cuda.get_device_name(0)\n"
    "try:\n"
    "    import triton\n"
    "    info['triton'] = triton.__version__\n"
    "except Exception as exc:\n"
    "    info['triton'] = None\n"
    "    info['triton_error'] = f'{type(exc).__name__}: {exc}'\n"
    "print(json.dumps(info))\n"
)


@functools.lru_cache(maxsize=2)
def probe(python: str | None = None) -> dict[str, Any]:
    """Import torch in the bench interpreter; {'error': ...} if that fails."""
    py = python or (str(torch_python()) if torch_python() else None)
    if py is None:
        return {"error": f"no torch venv at {VENV} (set {ENV_OVERRIDE})"}
    proc = subprocess.run([py, "-c", _PROBE], capture_output=True, text=True, env=bench_env())
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["unknown error"]
        return {"error": f"torch import failed in {py}: {tail[0]}"}
    try:
        info = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"error": f"unparseable torch probe output from {py}"}
    info["python"] = py
    return info


def torch_ready(*, compile: bool = False) -> tuple[bool, str]:
    """(ready, reason) for run.py's engine_ready: needs a CUDA torch, plus Triton for compile."""
    info = probe()
    if "error" in info:
        return False, info["error"]
    if not info.get("cuda_available"):
        return False, f"torch {info['torch']} in {info['python']} has no CUDA"
    if compile and not info.get("triton"):
        return False, f"triton unavailable: {info.get('triton_error', 'not installed')}"
    return True, ""


if __name__ == "__main__":
    print(json.dumps({"python": str(torch_python()), **probe()}, indent=2))
