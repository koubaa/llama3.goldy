#!/usr/bin/env python3
"""Fetch the llama3.cuda TinyStories checkpoint and tokenizer (gitignored)."""

from __future__ import annotations

import hashlib
import pathlib
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

STORIES15M = (
    "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin",
    "cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a",
)
TOKENIZER = (
    "https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin",
    "50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361",
)


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: pathlib.Path, expected: str | None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and expected:
        got = sha256(dest)
        if got == expected:
            print(f"ok {dest.name} ({got})")
            return
        print(f"hash mismatch for {dest.name}: {got} != {expected}; re-downloading")
    print(f"downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    urllib.request.urlretrieve(url, tmp)
    if expected:
        got = sha256(tmp)
        if got != expected:
            tmp.unlink(missing_ok=True)
            raise SystemExit(f"hash mismatch after download: {got} != {expected}")
        print(f"sha256 {got}")
    else:
        print(f"sha256 {sha256(tmp)}")
    tmp.replace(dest)


def main() -> int:
    fetch(STORIES15M[0], MODELS / "stories15M.bin", STORIES15M[1])
    fetch(TOKENIZER[0], MODELS / "tokenizer.bin", TOKENIZER[1])
    tok = MODELS / "tokenizer.bin"
    if tok.stat().st_size < 100_000:
        raise SystemExit(f"tokenizer.bin looks too small: {tok.stat().st_size} bytes")
    print(f"assets in {MODELS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
