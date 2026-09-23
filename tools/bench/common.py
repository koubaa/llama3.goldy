"""Shared TinyStories benchmark protocol constants and helpers."""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import statistics
import time
from typing import Any, Iterable

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
CACHE = pathlib.Path(__file__).resolve().parent / ".cache"
REFS = ROOT.parent / "refs"

SCHEMA_VERSION = 1
DREAM_PROMPT = "I have a dream"
DREAM_STORY = (
    "I have a dream. He dreams of a big, beautiful garden full of flowers and trees. "
    "He dreams of playing with his friends and eating yummy snacks.\n"
    "One day, he was walking in the garden when he saw"
)
COMPAT_TOTAL_POSITIONS = 50
SCALING_CONTEXT_LENGTHS = (8, 32, 128, 224)
SCALING_DECODE_STEPS = 16
ROPE_THETA = 10000.0
RMS_EPS = 1e-5

ENGINES = (
    "goldy",
    "pytorch-eager",
    "pytorch-compile",
    "llama3.cuda",
    "llama.cpp",
    "llama.cpp-bench",
)

LLAMA3_CUDA_COMMIT = "424333d1651d2b0fc17d38e9f790e824947e284b"
LLAMA_CPP_COMMIT = "f072b103714dfa1eee531f80b24512faf38e3dd2"

CHECKPOINT_SHA256 = {
    "stories15M.bin": "cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a",
    "stories42M.bin": "9f65a1000e17d0bc167dd6332e0ce5119a0222a3d920cead5bce413bfab2ee7b",
    "stories110M.bin": "515267168726a1ed1317a64a408492e6af3b67c1f71c5bd98c01d9d721803a24",
    "tokenizer.bin": "50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361",
}


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def monotonic_s() -> float:
    return time.perf_counter()


def percentile(samples: list[float], p: float) -> float:
    if not samples:
        return 0.0
    if len(samples) == 1:
        return float(samples[0])
    ordered = sorted(samples)
    k = (len(ordered) - 1) * (p / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return float(ordered[lo])
    return float(ordered[lo] * (hi - k) + ordered[hi] * (k - lo))


def summarize(samples: Iterable[float]) -> dict[str, float]:
    values = [float(x) for x in samples]
    if not values:
        return {"median": 0.0, "p10": 0.0, "p90": 0.0, "n": 0.0}
    return {
        "median": float(statistics.median(values)),
        "p10": percentile(values, 10),
        "p90": percentile(values, 90),
        "n": float(len(values)),
    }


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
