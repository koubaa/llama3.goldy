"""Condense a metal_api_trace log into one line per dispatch for the last N submits.

A submit ends at a `waitUntilCompleted.return` (llama.cpp) or, with --by-commit, at
every Nth group of commits; pass --last to choose how many trailing groups to print.

    python3 tools/metal_api_trace/summarize.py /tmp/trace.txt --last 1
"""

from __future__ import annotations

import argparse
import collections
import re


def parse(path):
    for line in open(path, encoding="utf-8", errors="replace"):
        parts = line.rstrip("\n").split(" ", 4)
        if len(parts) < 4:
            continue
        seq, thread, obj, call = parts[:4]
        yield int(seq), thread, obj, call, parts[4] if len(parts) > 4 else ""


def groups(rows, boundary):
    cur = []
    for r in rows:
        cur.append(r)
        if r[3] == boundary:
            yield cur
            cur = []
    if cur:
        yield cur


def condense(rows):
    pso = {}
    pending = collections.defaultdict(lambda: collections.Counter())
    label = {}
    barrier = collections.defaultdict(int)
    out = []
    counts = collections.Counter()
    for seq, thread, obj, call, args in rows:
        counts[call] += 1
        if call == "setComputePipelineState":
            pso[obj] = args
        elif call in ("setBuffer", "setBytes", "setThreadgroupMemoryLength", "setBufferOffset", "setBuffers"):
            pending[obj][call] += 1
        elif call == "pushDebugGroup":
            label[obj] = args
        elif call in ("memoryBarrierWithScope", "memoryBarrierWithResources"):
            barrier[obj] += 1
        elif call.startswith("dispatch"):
            p = pending.pop(obj, collections.Counter())
            binds = " ".join(f"{k.replace('setThreadgroupMemoryLength', 'tgmem')}x{v}" for k, v in sorted(p.items()))
            mark = "|B " if barrier.pop(obj, 0) else "   "
            name = label.pop(obj, "") or pso.get(obj, "?")
            out.append(f"{thread:>3} {obj:<6} {mark}{name[:58]:<58} {args:<26} {binds}")
        elif call in ("useResources", "useHeaps", "useHeap", "useResource"):
            out.append(f"{thread:>3} {obj:<6}    {call} {args}")
        else:
            out.append(f"{thread:>3} {obj:<6} -- {call} {args}")
    return out, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--boundary", default="waitUntilCompleted.return")
    ap.add_argument("--last", type=int, default=1)
    args = ap.parse_args()
    gs = list(groups(parse(args.trace), args.boundary))
    for g in gs[-args.last :]:
        lines, counts = condense(g)
        print("\n".join(lines))
        print("counts:", dict(sorted(counts.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
