#!/usr/bin/env python3
"""Per-step CPU/GPU timeline from a timestamped metal_api_trace log.

A step ends at each `waitUntilCompleted.return`. For every step the report gives
wall time, when the first command buffer was created, and per command buffer its
encode window, commit, and GPU execution window, all relative to the step start.

With `--split blit`, a step instead ends with a command buffer that encodes a blit
(the readback) and that the host waited for, and starts when the previous step's GPU
work ended. This works for hosts that do not wait with `waitUntilCompleted`;
`first_cb` then includes how long the host took to observe completion.

  python3 tools/metal_api_trace/timeline.py /tmp/trace.txt [--last 20] [--show 2] [--split blit]
"""

import argparse
import re
import statistics as st

LINE = re.compile(r"^(\d+) t(\d+) (\S+) (\S+) ?(.*?) @([\d.]+)$")
GPU = re.compile(r"start=([\d.]+) end=([\d.]+)")


def parse(path):
    for line in open(path):
        m = LINE.match(line.rstrip("\n"))
        if m:
            yield {"tid": int(m[2]), "obj": m[3], "call": m[4], "args": m[5], "t": float(m[6])}


def steps(rows):
    rows = list(rows)
    gpu = {}
    for r in rows:
        if r["call"] == "gpu":
            m = GPU.search(r["args"])
            gpu[r["obj"]] = (float(m[1]), float(m[2]))
    ends = [i for i, r in enumerate(rows) if r["call"] == "waitUntilCompleted.return"]
    for a, b in zip(ends, ends[1:]):
        t0 = rows[a]["t"]
        cbs = {}
        for r in rows[a + 1 : b + 1]:
            if r["call"].startswith("queue.commandBuffer"):
                cbs[r["obj"]] = {"created": r["t"] - t0}
            elif r["obj"] in cbs and r["call"] == "commit":
                cbs[r["obj"]]["commit"] = r["t"] - t0
            elif r["call"] == "endEncoding":
                pass
        for name, cb in cbs.items():
            if name in gpu:
                cb["gpu"] = (gpu[name][0] - t0, gpu[name][1] - t0)
        yield {"wall": rows[b]["t"] - t0, "wait": rows[b - 1]["t"] - t0, "cbs": cbs}


def blit_steps(rows):
    rows = list(rows)
    gpu = {}
    for r in rows:
        if r["call"] == "gpu":
            m = GPU.search(r["args"])
            gpu[r["obj"]] = (float(m[1]), float(m[2]))
    order, created, commit, ends = [], {}, {}, []
    for r in rows:
        if r["call"].startswith("queue.commandBuffer"):
            order.append(r["obj"])
            created[r["obj"]] = r["t"]
        elif r["call"] == "commit":
            commit[r["obj"]] = r["t"]
        elif r["call"] == "blitCommandEncoder" and r["obj"] in created:
            ends.append(order.index(r["obj"]))
    # A step ends only where the host waited: the next command buffer comes after it.
    ends = [
        i
        for i in sorted(set(ends))
        if i + 1 < len(order) and order[i] in gpu and created[order[i + 1]] > gpu[order[i]][1]
    ]
    for a, b in zip(ends, ends[1:]):
        if order[a] not in gpu or order[b] not in gpu:
            continue
        t0 = gpu[order[a]][1]
        cbs = {}
        for name in order[a + 1 : b + 1]:
            cb = {"created": created[name] - t0}
            if name in commit:
                cb["commit"] = commit[name] - t0
            if name in gpu:
                cb["gpu"] = (gpu[name][0] - t0, gpu[name][1] - t0)
            cbs[name] = cb
        wall = gpu[order[b]][1] - t0
        yield {"wall": wall, "wait": float("nan"), "cbs": cbs}


def summarize(step):
    cbs = [c for c in step["cbs"].values() if "gpu" in c]
    if not cbs:
        return None
    busy = sum(c["gpu"][1] - c["gpu"][0] for c in cbs)
    first_cb = min(c["created"] for c in step["cbs"].values())
    gpu_first = min(c["gpu"][0] for c in cbs)
    gpu_last = max(c["gpu"][1] for c in cbs)
    last_commit = max(c.get("commit", 0) for c in step["cbs"].values())
    return {
        "wall": step["wall"],
        "first_cb": first_cb,
        "last_commit": last_commit,
        "gpu_first_start": gpu_first,
        "gpu_busy": busy,
        "gpu_span": gpu_last - gpu_first,
        "gpu_end_to_return": step["wall"] - gpu_last,
        "cbs": len(step["cbs"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--last", type=int, default=20, help="summarize the last N steps")
    ap.add_argument("--show", type=int, default=1, help="print the last N steps in detail")
    ap.add_argument("--split", choices=["wait", "blit"], default="wait")
    a = ap.parse_args()
    all_steps = list((blit_steps if a.split == "blit" else steps)(parse(a.trace)))
    tail = all_steps[-a.last :]
    for s in all_steps[-a.show :]:
        print(f"step wall={s['wall']:.1f}us wait_called={s['wait']:.1f}us")
        for name, c in s["cbs"].items():
            g = c.get("gpu")
            gs = f"gpu {g[0]:8.1f} .. {g[1]:8.1f} ({g[1] - g[0]:6.1f})" if g else "gpu ?"
            print(f"  {name:>6} created {c['created']:8.1f} commit {c.get('commit', float('nan')):8.1f}  {gs}")
    rows = [r for r in map(summarize, tail) if r]
    print(f"\nmedian over {len(rows)} steps (us):")
    for k in rows[0]:
        print(f"  {k:>18} {st.median(r[k] for r in rows):9.1f}")


if __name__ == "__main__":
    main()
