#!/usr/bin/env python3
"""Fetch and hash-check the shared TinyStories tokenizer and checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

# One tokenizer + one llama2.c blob per model, consumed by every engine.
ASSETS: dict[str, dict[str, object]] = {
    "tokenizer.bin": {
        "url": "https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin",
        "sha256": "50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361",
        "min_bytes": 100_000,
        "required": True,
    },
    "stories15M.bin": {
        "url": "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin",
        "sha256": "cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a",
        "bytes": 60_816_028,
        "config": {
            "dim": 288,
            "hidden_dim": 768,
            "n_layers": 6,
            "n_heads": 6,
            "n_kv_heads": 6,
            "vocab_size": 32000,
            "max_seq_len": 256,
        },
        "required": True,
    },
    "stories42M.bin": {
        "url": "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin",
        "sha256": "9f65a1000e17d0bc167dd6332e0ce5119a0222a3d920cead5bce413bfab2ee7b",
        "bytes": 167_020_572,
        "config": {
            "dim": 512,
            "hidden_dim": 1376,
            "n_layers": 8,
            "n_heads": 8,
            "n_kv_heads": 8,
            "vocab_size": 32000,
            "max_seq_len": 1024,
        },
        "required": False,
    },
    "stories110M.bin": {
        "url": "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin",
        "sha256": "515267168726a1ed1317a64a408492e6af3b67c1f71c5bd98c01d9d721803a24",
        "bytes": 438_381_596,
        "config": {
            "dim": 768,
            "hidden_dim": 2048,
            "n_layers": 12,
            "n_heads": 12,
            "n_kv_heads": 12,
            "vocab_size": 32000,
            "max_seq_len": 1024,
        },
        "required": False,
    },
}

TOKENIZER_SHA256 = str(ASSETS["tokenizer.bin"]["sha256"])
STORIES15M_SHA256 = str(ASSETS["stories15M.bin"]["sha256"])
STORIES42M_SHA256 = str(ASSETS["stories42M.bin"]["sha256"])
STORIES110M_SHA256 = str(ASSETS["stories110M.bin"]["sha256"])


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_file(name: str, path: pathlib.Path) -> None:
    spec = ASSETS[name]
    expected = str(spec["sha256"])
    got = sha256(path)
    if got != expected:
        raise SystemExit(f"hash mismatch for {name}: {got} != {expected}")
    size = path.stat().st_size
    if "bytes" in spec and size != spec["bytes"]:
        raise SystemExit(f"size mismatch for {name}: {size} != {spec['bytes']}")
    if "min_bytes" in spec and size < int(spec["min_bytes"]):
        raise SystemExit(f"{name} looks too small: {size} bytes")


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


def names_for(kind: str) -> list[str]:
    kind = kind.lower()
    aliases = {
        "tokenizer": ["tokenizer.bin"],
        "15m": ["stories15M.bin"],
        "42m": ["stories42M.bin"],
        "110m": ["stories110M.bin"],
        "all": list(ASSETS),
        "required": [n for n, s in ASSETS.items() if s.get("required")],
    }
    if kind not in aliases:
        raise SystemExit(f"unknown asset {kind!r}; choose from {sorted(aliases)}")
    return aliases[kind]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--only",
        default="all",
        help="comma-separated: tokenizer,15m,42m,110m,required,all (default all)",
    )
    p.add_argument(
        "--verify-only",
        action="store_true",
        help="hash-check existing files; do not download",
    )
    args = p.parse_args(argv)

    wanted: list[str] = []
    for part in args.only.split(","):
        for name in names_for(part.strip()):
            if name not in wanted:
                wanted.append(name)

    for name in wanted:
        spec = ASSETS[name]
        dest = MODELS / name
        if args.verify_only:
            if not dest.exists():
                raise SystemExit(f"missing {dest}")
            verify_file(name, dest)
            print(f"ok {name}")
            continue
        fetch(str(spec["url"]), dest, str(spec["sha256"]))
        verify_file(name, dest)
    print(f"assets in {MODELS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
