"""Summarize JSON Lines results into the checked-in baseline report."""

from __future__ import annotations

import json
import pathlib
from collections import defaultdict
from typing import Any, Iterable

from . import common


def _model_name(path: str) -> str:
    return pathlib.Path(path).name


def _group_key(row: dict[str, Any]) -> tuple:
    w = row["workload"]
    return (
        row["engine"],
        row.get("execution", ""),
        _model_name(row["checkpoint"]["path"]),
        w["tier"],
        int(w["context_len"]),
        w["sampling"],
    )


def _metric_samples(rows: list[dict[str, Any]], name: str) -> list[float]:
    return [float(r["metrics"][name]) for r in rows]


def _phase_samples(rows: list[dict[str, Any]], name: str) -> list[float]:
    return [float(r["phases"][name]) for r in rows]


def _step_medians(rows: list[dict[str, Any]]) -> list[float]:
    out = []
    for r in rows:
        steps = [float(x) for x in r["phases"]["decode_step_s"]]
        if steps:
            out.append(common.summarize(steps)["median"])
    return out


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    notes: list[str] = []
    for r in rows:
        for note in r.get("engine_native_notes") or []:
            if note not in notes:
                notes.append(note)
    replay = None
    for r in rows:
        if r.get("replay_stats"):
            replay = r["replay_stats"]
    matches = [bool(r["tokens"]["match_expected"]) for r in rows]
    return {
        "engine": first["engine"],
        "execution": first["execution"],
        "checkpoint": _model_name(first["checkpoint"]["path"]),
        "tier": first["workload"]["tier"],
        "context_len": first["workload"]["context_len"],
        "sampling": first["workload"]["sampling"],
        "n": len(rows),
        "match_expected": all(matches),
        "legacy_compat_tok_s": common.summarize(_metric_samples(rows, "legacy_compat_tok_s")),
        "decode_tok_s": common.summarize(_metric_samples(rows, "decode_tok_s")),
        "prompt_tok_s": common.summarize(_metric_samples(rows, "prompt_tok_s")),
        "ttft_s": common.summarize(_phase_samples(rows, "ttft_s")),
        "prompt_s": common.summarize(_phase_samples(rows, "prompt_s")),
        "decode_step_median_s": common.summarize(_step_medians(rows)),
        "load_s": common.summarize(_phase_samples(rows, "load_s")),
        "engine_native_notes": notes,
        "replay_stats": replay,
    }


def group_results(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[_group_key(row)].append(row)
    groups = [summarize_group(v) for v in buckets.values()]
    groups.sort(key=lambda g: (g["tier"], g["checkpoint"], g["context_len"], g["engine"]))
    return groups


def _fmt(summary: dict[str, float], digits: int = 1) -> str:
    if not summary or summary.get("n", 0) == 0:
        return "—"
    if summary["p10"] == 0 and summary["p90"] == 0:
        return "—"
    med = summary["median"]
    p10 = summary["p10"]
    p90 = summary["p90"]
    return f"{med:.{digits}f} ({p10:.{digits}f}–{p90:.{digits}f})"


def render_markdown(groups: list[dict[str, Any]], metadata: dict[str, Any], skipped: list[str]) -> str:
    lines = [
        "# Llama benchmark baseline",
        "",
        "Informational throughput. Protocol: [`BENCHMARKS.md`](../BENCHMARKS.md).",
        "",
        "## Machine",
        "",
        f"- captured: `{metadata.get('captured_at', '')}`",
        f"- host: `{metadata.get('hostname', '')}`",
        f"- os: `{metadata.get('os', '')}`",
        f"- gpu: `{metadata.get('gpu', '')}`",
        f"- rustc: `{metadata.get('rustc', '')}`",
        f"- torch: `{metadata.get('torch', '')}`",
        f"- llama3.cuda pin: `{metadata.get('llama3_cuda_commit', common.LLAMA3_CUDA_COMMIT)}`",
        f"- llama.cpp pin: `{metadata.get('llama_cpp_commit', common.LLAMA_CPP_COMMIT)}`",
        "",
    ]
    if skipped:
        lines.append("## Skipped engines")
        lines.append("")
        for item in skipped:
            lines.append(f"- {item}")
        lines.append("")

    compat = [g for g in groups if g["tier"] == "compatibility" and g["engine"] != "llama.cpp-bench"]
    lines.extend(
        [
            "## Compatibility (stories15M, \"I have a dream\", 50 positions)",
            "",
            "Headline metric is llama3.cuda-style `legacy_compat_tok_s = (pos - 1) / elapsed`.",
            "llama.cpp uses its engine-native tokenizer and is **not** rewritten to match; it has no legacy metric.",
            "",
            "| Engine | Exec | Match | Legacy tok/s | Decode tok/s | Prompt tok/s | TTFT s |",
            "|--------|------|-------|--------------|--------------|--------------|--------|",
        ]
    )
    for g in compat:
        lines.append(
            f"| {g['engine']} | {g['execution']} | {'yes' if g['match_expected'] else 'no'} | "
            f"{_fmt(g['legacy_compat_tok_s'])} | {_fmt(g['decode_tok_s'])} | "
            f"{_fmt(g['prompt_tok_s'])} | {_fmt(g['ttft_s'], 4)} |"
        )
    if not compat:
        lines.append("| *(no compatibility rows)* | | | | | | |")
    lines.append("")
    executions = sorted({f"{g['engine']}={g['execution']}" for g in groups})
    lines.append(f"Execution: {', '.join(executions)}.")
    lines.append("")

    bench = [g for g in groups if g["engine"] == "llama.cpp-bench"]
    if bench:
        lines.extend(
            [
                "## llama.cpp-bench (engine-native, random tokens)",
                "",
                "Do not mix into the compatibility headline.",
                "",
                "| Checkpoint | Context | Decode tok/s | Prompt tok/s |",
                "|------------|---------|--------------|--------------|",
            ]
        )
        for g in bench:
            lines.append(
                f"| {g['checkpoint']} | {g['context_len']} | {_fmt(g['decode_tok_s'])} | {_fmt(g['prompt_tok_s'])} |"
            )
        lines.append("")

    scaling = [g for g in groups if g["tier"] == "scaling" and g["engine"] != "llama.cpp-bench"]
    checkpoints = sorted({g["checkpoint"] for g in scaling})
    engines = []
    for g in scaling:
        if g["engine"] not in engines:
            engines.append(g["engine"])
    lines.append("## Scaling decode tok/s (median, p10–p90)")
    lines.append("")
    if not scaling:
        lines.append("No scaling rows.")
        lines.append("")
    for ckpt in checkpoints:
        subset = [g for g in scaling if g["checkpoint"] == ckpt]
        ctxs = sorted({g["context_len"] for g in subset})
        lines.append(f"### {ckpt}")
        lines.append("")
        header = "| Context | " + " | ".join(engines) + " |"
        sep = "|---------|" + "|".join(["--------"] * len(engines)) + "|"
        lines.append(header)
        lines.append(sep)
        for ctx in ctxs:
            cells = [str(ctx)]
            for engine in engines:
                hit = next((g for g in subset if g["engine"] == engine and g["context_len"] == ctx), None)
                cells.append(_fmt(hit["decode_tok_s"]) if hit else "—")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        lines.append("TTFT s / prompt tok/s / decode-step median s:")
        lines.append("")
        lines.append("| Context | Engine | TTFT s | Prompt tok/s | Step median s |")
        lines.append("|---------|--------|--------|--------------|---------------|")
        for g in subset:
            lines.append(
                f"| {g['context_len']} | {g['engine']} | {_fmt(g['ttft_s'], 4)} | "
                f"{_fmt(g['prompt_tok_s'])} | {_fmt(g['decode_step_median_s'], 5)} |"
            )
        lines.append("")

    goldy = [g for g in groups if g["engine"] == "goldy" and g.get("replay_stats")]
    if goldy:
        lines.append("## Goldy replay stats (last row per group)")
        lines.append("")
        lines.append("| Checkpoint | Tier | Context | records | topology | clean | resubmit_hits |")
        lines.append("|------------|------|---------|---------|----------|-------|---------------|")
        for g in goldy:
            rs = g["replay_stats"]
            lines.append(
                f"| {g['checkpoint']} | {g['tier']} | {g['context_len']} | "
                f"{rs.get('records', '')} | {rs.get('topology_records', '')} | "
                f"{rs.get('clean_submits', '')} | {rs.get('resubmit_hits', '')} |"
            )
        lines.append("")

    noted = []
    for g in groups:
        for note in g.get("engine_native_notes") or []:
            key = (g["engine"], note)
            if key not in noted:
                noted.append(key)
    if noted:
        lines.append("## Engine-native notes")
        lines.append("")
        for engine, note in noted:
            lines.append(f"- `{engine}`: {note}")
        lines.append("")
    return "\n".join(lines) + "\n"


def write_report(
    rows: list[dict[str, Any]],
    *,
    metadata: dict[str, Any],
    skipped: list[str],
    out_md: pathlib.Path,
    out_json: pathlib.Path,
) -> tuple[pathlib.Path, dict[str, Any]]:
    groups = group_results(rows)
    md = render_markdown(groups, metadata, skipped)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")
    headline = [
        {
            "engine": g["engine"],
            "execution": g["execution"],
            "legacy_compat_tok_s": g["legacy_compat_tok_s"],
            "match_expected": g["match_expected"],
        }
        for g in groups
        if g["tier"] == "compatibility" and g["engine"] != "llama.cpp-bench"
    ]
    summary = {
        "metadata": metadata,
        "skipped": skipped,
        "groups": groups,
        "headline": headline,
        "n_rows": len(rows),
    }
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return out_md, summary


def load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("{"):
            rows.append(json.loads(line))
    return rows


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("jsonl", nargs="+", type=pathlib.Path)
    p.add_argument("--out-dir", default=str(pathlib.Path(__file__).resolve().parents[2] / "benchmarks"))
    p.add_argument("--skip", action="append", default=[])
    p.add_argument("--metadata", type=pathlib.Path, default=None)
    args = p.parse_args(argv)
    rows: list[dict[str, Any]] = []
    for path in args.jsonl:
        rows.extend(load_jsonl(path))
    out_dir = pathlib.Path(args.out_dir)
    metadata = {}
    if args.metadata and args.metadata.exists():
        metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    write_report(
        rows,
        metadata=metadata,
        skipped=args.skip,
        out_md=out_dir / "baseline.md",
        out_json=out_dir / "baseline.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
