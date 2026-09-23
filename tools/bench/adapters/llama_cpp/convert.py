"""Convert a llama2.c checkpoint to F32 GGUF using pinned llama.cpp."""

from __future__ import annotations

import argparse
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
TOOLS = HERE.parents[1].parent
sys.path.insert(0, str(TOOLS))

from bench import common  # noqa: E402
from bench.adapters.llama_cpp import build as cpp_build  # noqa: E402


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


def convert(
    checkpoint: pathlib.Path,
    tokenizer: pathlib.Path,
    output: pathlib.Path | None = None,
    *,
    force: bool = False,
) -> pathlib.Path:
    output = output or gguf_path_for(checkpoint)
    if output.exists() and not force:
        return output
    tools = cpp_build.build()
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = convert_cmd(tools["llama-convert-llama2c-to-ggml"], checkpoint, tokenizer, output)
    from bench.adapters.util import run

    run(cmd)
    if not output.exists():
        raise SystemExit(f"conversion did not produce {output}")
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
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
