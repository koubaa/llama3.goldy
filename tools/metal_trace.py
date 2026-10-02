#!/usr/bin/env python3
"""Summarize a Metal System Trace (xctrace) of one LLM decode process.

Usage:
    metal_trace.py TRACE [--process NAME] [--dump SCHEMA]

Exports the GPU interval, encoder, and command-buffer tables with
`xcrun xctrace export`, then prints per-command-buffer GPU time, encoder counts,
and the gaps between consecutive command buffers (CPU encode / sync bubbles).
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import statistics
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET


def export_table(trace: pathlib.Path, schema: str, out_dir: pathlib.Path) -> pathlib.Path:
    out = out_dir / f"{schema}.xml"
    if not out.exists():
        xpath = f'/trace-toc/run[@number="1"]/data/table[@schema="{schema}"]'
        subprocess.run(
            ["xcrun", "xctrace", "export", "--input", str(trace), "--xpath", xpath, "--output", str(out)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    return out


def read_table(path: pathlib.Path) -> list[dict[str, str]]:
    """Rows as `{mnemonic: fmt}`; xctrace `ref` attributes resolve to earlier `id`s."""
    root = ET.parse(path).getroot()
    node = root.find(".//node")
    if node is None:
        return []
    cols = [c.findtext("mnemonic") for c in node.find("schema").findall("col")]
    ids: dict[str, ET.Element] = {}
    for el in root.iter():
        if "id" in el.attrib:
            ids[el.attrib["id"]] = el

    def value(el: ET.Element) -> tuple[str, str]:
        if "ref" in el.attrib:
            el = ids.get(el.attrib["ref"], el)
        return el.attrib.get("fmt", el.text or ""), (el.text or "")

    rows = []
    for row in node.findall("row"):
        rec: dict[str, str] = {}
        for col, el in zip(cols, list(row)):
            fmt, raw = value(el)
            rec[col] = fmt
            rec[col + "#raw"] = raw
        rows.append(rec)
    return rows


def num(rec: dict[str, str], key: str) -> int:
    try:
        return int(rec.get(key + "#raw") or 0)
    except ValueError:
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", type=pathlib.Path)
    ap.add_argument("--process", default=None, help="substring of the process name to keep")
    ap.add_argument("--dump", default=None, help="print the first rows of this schema and exit")
    ap.add_argument("--skip", type=int, default=5, help="leading command buffers to drop (load/prompt)")
    ap.add_argument("--timeline", type=int, default=0, help="print this many kept command buffers in order")
    args = ap.parse_args()

    out_dir = pathlib.Path(tempfile.gettempdir()) / "metal_trace" / args.trace.name
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dump:
        rows = read_table(export_table(args.trace, args.dump, out_dir))
        for r in rows[:40]:
            print({k: v for k, v in r.items() if not k.endswith("#raw")})
        return 0

    rows = read_table(export_table(args.trace, "metal-gpu-intervals", out_dir))
    if args.process:
        rows = [r for r in rows if args.process in r.get("process", "")]
    compute = [r for r in rows if r.get("channel-name") == "Compute"]
    by_cb: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    for r in compute:
        by_cb[r.get("cmdbuffer-id#raw", "")].append(r)

    cbs = []
    for cb, rs in by_cb.items():
        start = min(num(r, "start") for r in rs)
        end = max(num(r, "start") + num(r, "duration") for r in rs)
        encoders = {r.get("encoder-id#raw") for r in rs}
        busy = sum(num(r, "duration") for r in rs if r.get("event-depth") == "0")
        cbs.append((start, end, len(encoders), busy, rs[0].get("event-label", "")))
    cbs.sort()
    cbs = cbs[args.skip :]
    if not cbs:
        print("no compute command buffers")
        return 1

    prev_end = None
    for start, end, n_enc, busy, label in cbs[: args.timeline]:
        gap = (start - prev_end) / 1e3 if prev_end is not None else 0.0
        print(
            f"{start / 1e6:10.3f} ms  span {(end - start) / 1e3:8.1f} us  busy {busy / 1e3:8.1f} us"
            f"  gap {gap:7.1f} us  encoders {n_enc:3d}  {label[:50]}"
        )
        prev_end = end

    spans = [(e - s) / 1e3 for s, e, *_ in cbs]
    gaps = [(cbs[i + 1][0] - cbs[i][1]) / 1e3 for i in range(len(cbs) - 1)]
    encs = [n for _, _, n, _, _ in cbs]
    print(f"compute command buffers: {len(cbs)} (after skipping {args.skip})")
    print(f"GPU span per command buffer  us: median {statistics.median(spans):8.1f}  p10 {sorted(spans)[len(spans)//10]:8.1f}  p90 {sorted(spans)[len(spans)*9//10]:8.1f}")
    if gaps:
        print(f"idle gap between buffers     us: median {statistics.median(gaps):8.1f}  p10 {sorted(gaps)[len(gaps)//10]:8.1f}  p90 {sorted(gaps)[len(gaps)*9//10]:8.1f}")
    print(f"encoders per command buffer    : median {statistics.median(encs)}  max {max(encs)}")
    total = (cbs[-1][1] - cbs[0][0]) / 1e3
    print(f"wall across kept buffers     ms: {total/1e3:.2f}  GPU-busy fraction {sum(spans)/total:.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
