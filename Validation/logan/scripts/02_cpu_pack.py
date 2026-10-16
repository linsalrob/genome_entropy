#!/usr/bin/env python3
"""Run many 02_cpu_stage.py accessions inside one allocation, packed by memory.

Setonix bills an allocation, not the work done in it: phaseb1's half-node
elements were billed 128 SU/h while running 12 single-threaded accessions,
~10.7 SU per process-hour. This launcher keeps as many accessions running as
the allocation can hold:

  * at most --max-procs at once (bounded by the cores);
  * the sum of their *estimated* peak memory stays under --mem-gb.

Peak memory is estimated from the accession's expected contig bp, from the
pilot (7 GB at 287 Mbp, ~1 GB floor): est = 1 + 2.3 GB per 100 Mbp, times a
1.25 safety factor. Largest accessions are started first so a big one is
never left to run alone at the end of the element (the 12 h timeouts of
phaseb1 were exactly that).

Each accession's output goes to logs/cpu/<acc>.log; failures are reported
and do not stop the others. Exit status is 1 if any accession failed.

  02_cpu_pack.py todo.txt --max-procs 40 --mem-gb 105 [--repro-every 100]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import logan_common as lc  # noqa: E402

HERE = Path(__file__).resolve().parent


def est_gb(acc: str) -> float:
    bp = lc.load_state(acc).get("info", {}).get("expected_contig_bp") or 1e8
    return 1.25 * (1.0 + 2.3 * bp / 1e8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("todo", help="lines of '<row> <accession>'")
    ap.add_argument("--max-procs", type=int, default=40)
    ap.add_argument("--mem-gb", type=float, default=105)
    ap.add_argument("--repro-every", type=int, default=100)
    args = ap.parse_args()

    items = [l.split() for l in open(args.todo) if l.strip()]
    items = [(int(r), a, est_gb(a)) for r, a in items]
    items.sort(key=lambda x: -x[2])          # largest first
    logdir = lc.root() / "logs" / "cpu"
    logdir.mkdir(parents=True, exist_ok=True)

    running = {}                              # Popen -> (acc, gb, t0)
    ok, failed, peak_procs, peak_gb = [], [], 0, 0.0
    while items or running:
        for p in [p for p in running if p.poll() is not None]:
            acc, gb, t0 = running.pop(p)
            (ok if p.returncode == 0 else failed).append(acc)
            print(f"{'ok' if p.returncode == 0 else 'FAILED'} {acc} {time.time() - t0:.0f}s "
                  f"(est {gb:.1f} GB)", flush=True)
        used = sum(v[1] for v in running.values())
        started = False
        for i, (row, acc, gb) in enumerate(items):
            if len(running) >= args.max_procs:
                break
            # Always allow one process, even if its estimate exceeds the budget.
            if running and used + gb > args.mem_gb:
                continue
            cmd = [sys.executable, str(HERE / "02_cpu_stage.py"), acc]
            if args.repro_every and row % args.repro_every == 0:
                cmd.append("--repro-check")
            fh = open(logdir / f"{acc}.log", "w")
            running[subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)] = (acc, gb, time.time())
            fh.close()
            used += gb
            items.pop(i)
            started = True
            break
        peak_procs = max(peak_procs, len(running))
        peak_gb = max(peak_gb, used)
        if not started:
            time.sleep(5)
    print(f"done: {len(ok)} ok, {len(failed)} failed; peak {peak_procs} processes, "
          f"{peak_gb:.0f} GB estimated", flush=True)
    for a in failed:
        print(f"FAILED {a}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
