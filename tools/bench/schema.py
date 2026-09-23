"""JSON Lines schema for one benchmark repetition."""

from __future__ import annotations

from typing import Any

from .common import ENGINES, SCHEMA_VERSION

REQUIRED_TOP = (
    "schema_version",
    "engine",
    "execution",
    "checkpoint",
    "workload",
    "precision",
    "tokens",
    "phases",
    "metrics",
    "engine_native_notes",
    "build",
    "replay_stats",
)

REQUIRED_CHECKPOINT = ("path", "sha256", "config")
REQUIRED_CONFIG = (
    "dim",
    "hidden_dim",
    "n_layers",
    "n_heads",
    "n_kv_heads",
    "vocab_size",
    "max_seq_len",
)
REQUIRED_WORKLOAD = (
    "tier",
    "prompt",
    "batch",
    "context_len",
    "total_positions",
    "decode_steps",
    "sampling",
)
REQUIRED_PRECISION = ("weights", "activations", "kv", "tf32")
REQUIRED_TOKENS = ("prompt", "generated", "text", "match_expected")
REQUIRED_PHASES = (
    "load_s",
    "warmup_s",
    "prompt_s",
    "ttft_s",
    "decode_step_s",
    "compat_elapsed_s",
)
REQUIRED_METRICS = ("prompt_tok_s", "decode_tok_s", "legacy_compat_tok_s")


class SchemaError(ValueError):
    pass


def _require(obj: dict[str, Any], keys: tuple[str, ...], where: str) -> None:
    missing = [k for k in keys if k not in obj]
    if missing:
        raise SchemaError(f"{where} missing {missing}")


def validate_result(obj: Any) -> None:
    if not isinstance(obj, dict):
        raise SchemaError("result must be an object")
    _require(obj, REQUIRED_TOP, "result")
    if obj["schema_version"] != SCHEMA_VERSION:
        raise SchemaError(f"schema_version {obj['schema_version']!r} != {SCHEMA_VERSION}")
    if obj["engine"] not in ENGINES:
        raise SchemaError(f"unknown engine {obj['engine']!r}")
    if obj["execution"] not in ("native", "wsl"):
        raise SchemaError(f"execution must be native|wsl, got {obj['execution']!r}")
    if obj["workload"]["tier"] not in ("compatibility", "scaling"):
        raise SchemaError(f"unknown tier {obj['workload']['tier']!r}")
    if obj["workload"]["sampling"] not in ("greedy", "engine-native"):
        raise SchemaError("sampling must be greedy or engine-native")
    if obj["precision"]["weights"] != "fp32" or obj["precision"]["activations"] != "fp32":
        raise SchemaError("primary contract is fp32 weights/activations")
    if not isinstance(obj["engine_native_notes"], list):
        raise SchemaError("engine_native_notes must be a list")
    _require(obj["checkpoint"], REQUIRED_CHECKPOINT, "checkpoint")
    _require(obj["checkpoint"]["config"], REQUIRED_CONFIG, "checkpoint.config")
    _require(obj["workload"], REQUIRED_WORKLOAD, "workload")
    _require(obj["precision"], REQUIRED_PRECISION, "precision")
    _require(obj["tokens"], REQUIRED_TOKENS, "tokens")
    _require(obj["phases"], REQUIRED_PHASES, "phases")
    _require(obj["metrics"], REQUIRED_METRICS, "metrics")
    if not isinstance(obj["phases"]["decode_step_s"], list):
        raise SchemaError("phases.decode_step_s must be a list")
    if not isinstance(obj["tokens"]["prompt"], list) or not isinstance(
        obj["tokens"]["generated"], list
    ):
        raise SchemaError("token id lists must be arrays")
    if obj["replay_stats"] is not None and not isinstance(obj["replay_stats"], dict):
        raise SchemaError("replay_stats must be null or an object")


def base_result(
    *,
    engine: str,
    execution: str,
    checkpoint: dict[str, Any],
    workload: dict[str, Any],
    precision: dict[str, Any] | None = None,
    tokens: dict[str, Any],
    phases: dict[str, Any],
    metrics: dict[str, Any],
    engine_native_notes: list[str] | None = None,
    build: dict[str, Any] | None = None,
    replay_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    obj = {
        "schema_version": SCHEMA_VERSION,
        "engine": engine,
        "execution": execution,
        "checkpoint": checkpoint,
        "workload": workload,
        "precision": precision
        or {
            "weights": "fp32",
            "activations": "fp32",
            "kv": "fp32",
            "tf32": False,
        },
        "tokens": tokens,
        "phases": phases,
        "metrics": metrics,
        "engine_native_notes": list(engine_native_notes or []),
        "build": build or {},
        "replay_stats": replay_stats,
    }
    validate_result(obj)
    return obj
