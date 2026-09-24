"""Convert a llama2.c checkpoint to F32 GGUF using pinned llama.cpp."""

from __future__ import annotations

import argparse
import pathlib
import struct
import subprocess
import sys
from typing import BinaryIO

HERE = pathlib.Path(__file__).resolve().parent
TOOLS = HERE.parents[1].parent
sys.path.insert(0, str(TOOLS))

from bench import common  # noqa: E402
from bench.adapters.llama_cpp import build as cpp_build  # noqa: E402

GGUF_MAGIC = b"GGUF"
GGML_TYPE_F32 = 0
# GGUF metadata value type -> struct format for fixed-size scalars.
_GGUF_SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
_GGUF_STRING = 8
_GGUF_ARRAY = 9


def gguf_path_for(checkpoint: pathlib.Path) -> pathlib.Path:
    return checkpoint.with_suffix(".f32.gguf")


def convert_cmd(converter: pathlib.Path, checkpoint: pathlib.Path, tokenizer: pathlib.Path, output: pathlib.Path) -> list[str]:
    return [
        str(converter),
        "--copy-vocab-from-model",
        str(tokenizer),
        "--llama2c-model",
        str(checkpoint),
        "--llama2c-output-model",
        str(output),
    ]


def _read(f: BinaryIO, fmt: str):
    size = struct.calcsize("<" + fmt)
    return struct.unpack("<" + fmt, f.read(size))[0]


def _read_str(f: BinaryIO) -> str:
    return f.read(_read(f, "Q")).decode("utf-8", errors="replace")


def _skip_value(f: BinaryIO, vtype: int) -> None:
    if vtype in _GGUF_SCALARS:
        f.seek(struct.calcsize("<" + _GGUF_SCALARS[vtype]), 1)
    elif vtype == _GGUF_STRING:
        f.seek(_read(f, "Q"), 1)
    elif vtype == _GGUF_ARRAY:
        item_type = _read(f, "I")
        count = _read(f, "Q")
        if item_type in _GGUF_SCALARS:
            f.seek(count * struct.calcsize("<" + _GGUF_SCALARS[item_type]), 1)
        else:
            for _ in range(count):
                _skip_value(f, item_type)
    else:
        raise ValueError(f"unknown GGUF value type {vtype}")


def gguf_tensor_types(path: pathlib.Path) -> dict[str, int]:
    """Tensor name -> ggml type id, read from the GGUF header only."""
    with path.open("rb") as f:
        if f.read(4) != GGUF_MAGIC:
            raise ValueError(f"{path} is not GGUF")
        _read(f, "I")  # version
        n_tensors = _read(f, "Q")
        n_kv = _read(f, "Q")
        for _ in range(n_kv):
            _read_str(f)
            _skip_value(f, _read(f, "I"))
        types: dict[str, int] = {}
        for _ in range(n_tensors):
            name = _read_str(f)
            n_dims = _read(f, "I")
            f.seek(8 * n_dims, 1)
            types[name] = _read(f, "I")
            _read(f, "Q")  # offset
        return types


def verify_f32(path: pathlib.Path) -> int:
    types = gguf_tensor_types(path)
    bad = {name: t for name, t in types.items() if t != GGML_TYPE_F32}
    if not types or bad:
        raise SystemExit(f"{path}: expected all-F32 tensors, non-F32: {bad or 'no tensors'}")
    return len(types)


def convert(
    checkpoint: pathlib.Path,
    tokenizer: pathlib.Path,
    output: pathlib.Path | None = None,
    *,
    force: bool = False,
) -> pathlib.Path:
    output = output or gguf_path_for(checkpoint)
    if output.exists() and not force:
        verify_f32(output)
        return output
    tools = cpp_build.ensure_built()
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = convert_cmd(tools["llama-convert-llama2c-to-ggml"], checkpoint, tokenizer, output)
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if proc.returncode != 0 or not output.exists():
        sys.stderr.write(proc.stdout[-4000:] + proc.stderr[-4000:])
        raise SystemExit(f"conversion failed (exit {proc.returncode}); {output} missing or incomplete")
    verify_f32(output)
    return output


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=str(common.MODELS / "stories15M.bin"))
    p.add_argument("--tokenizer", default=str(common.MODELS / "tokenizer.bin"))
    p.add_argument("--output", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--print-cmd", action="store_true")
    args = p.parse_args(argv)
    ckpt = pathlib.Path(args.checkpoint)
    tok = pathlib.Path(args.tokenizer)
    out = pathlib.Path(args.output) if args.output else gguf_path_for(ckpt)
    dummy_converter = pathlib.Path("llama-convert-llama2c-to-ggml")
    if args.print_cmd:
        print(" ".join(convert_cmd(dummy_converter, ckpt, tok, out)))
        return 0
    path = convert(ckpt, tok, out, force=args.force)
    print(f"{path} ({verify_f32(path)} tensors, all F32)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
