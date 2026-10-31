#!/usr/bin/env python
"""Run the quick Phase 0 spikes (A, C, D, E-audit, F) and build the report.

Spike B (the blueprint pilot) is long-running and CPU-heavy, so it is run separately:
    python scripts/phase0/spike_b_pilot.py --iters 10000000
    python scripts/phase0/spike_b_pilot.py --iters 30000000
    python scripts/phase0/spike_b_extrapolate.py
Spike D needs the GPU venv; here it records PENDING if torch is missing from this one.
Spike F defaults to the MOCK backend; pass real backend arguments via --f-args "...".
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(script: str, *extra: str) -> int:
    cmd = [sys.executable, str(HERE / script), *extra]
    print(f"\n=== {' '.join(Path(c).name if c.endswith('.py') else c for c in cmd[1:])}")
    return subprocess.run(cmd, check=False).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--quick", action="store_true", help="short benchmark windows")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--skip", nargs="*", default=[], choices=["A", "C", "D", "E", "F"])
    ap.add_argument(
        "--f-args",
        default="--backend mock --label mock",
        help='arguments for spike F, e.g. "--backend direct --base-url http://127.0.0.1:8080/v1"',
    )
    args = ap.parse_args()
    quick = ["--quick"] if args.quick else []

    codes = {}
    if not args.report_only:
        if "A" not in args.skip:
            codes["A"] = run("spike_a_engine_bench.py", *quick)
        if "C" not in args.skip:
            codes["C"] = run("spike_c_equity_bench.py", *quick)
        if "D" not in args.skip:
            codes["D"] = run("spike_d_gpu_check.py", *quick)
        if "E" not in args.skip:
            codes["E"] = run("spike_e_license_audit.py")
        if "F" not in args.skip:
            codes["F"] = run("spike_f_local_decision.py", *shlex.split(args.f_args))
    run("report.py")
    print("\nexit codes:", codes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
