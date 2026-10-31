#!/usr/bin/env python
"""Spike B, step 2: project training memory and time from pilot runs (gate G0.2).

Reads results/phase0/spike_b_pilot_*.json (successful runs only), fits peak RSS as a power law
of iteration count (needs two different pilot sizes; with one it assumes linear growth, which is
conservative), anchors on the largest pilot, and projects 50M / 100M / 200M / 400M iterations against a memory budget.

Rule from the plan: keep projected peak memory under ~24 GiB on a 30.6 GiB machine (OS and
desktop need headroom). Time projections use the pilot's iterations/s and are optimistic: tables
grow and caches miss more as training proceeds, so a pessimistic factor is shown as well.
"""

from __future__ import annotations

import argparse
import json
import math

from common import RESULTS, gate, write_result

TARGETS = [50_000_000, 100_000_000, 200_000_000, 400_000_000]
DEFAULT_BUDGET_GIB = 24.0
PESSIMISTIC_EXPONENT_ADD = 0.15
PESSIMISTIC_TIME_FACTOR = 1.5


def load_pilots() -> list[dict]:
    pilots = []
    for path in sorted(RESULTS.glob("spike_b_pilot_*.json")):
        data = json.loads(path.read_text())
        if data.get("exit_code") == 0 and data.get("peak_rss_gib") and data.get("iters"):
            pilots.append(data)
    return pilots


def fit_power_law(points: list[tuple[int, float]]) -> tuple[float, float, str]:
    """Return (a, b, note) for rss = a * iters**b."""
    distinct = sorted(set(points))
    if len({p[0] for p in distinct}) < 2:
        iters, rss = distinct[-1]
        return rss / iters, 1.0, "single pilot size: assumed linear growth (conservative)"
    xs = [math.log(p[0]) for p in distinct]
    ys = [math.log(p[1]) for p in distinct]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sum(
        (x - mx) ** 2 for x in xs
    )
    b = min(max(b, 0.3), 1.2)
    a = math.exp(my - b * mx)
    return a, b, f"fitted exponent {b:.2f} from {n} pilot sizes"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--budget-gib", type=float, default=DEFAULT_BUDGET_GIB)
    args = ap.parse_args()

    pilots = load_pilots()
    if not pilots:
        detail = "no successful pilot results yet: run spike_b_pilot.py first"
        path = write_result("spike_b_extrapolation", {"gates": [gate("G0.2", None, detail)]})
        print(f"G0.2 PENDING: {detail}\nwrote {path}")
        return 0

    points = [(p["iters"], p["peak_rss_gib"]) for p in pilots]
    a, b, note = fit_power_law(points)
    biggest = max(pilots, key=lambda p: p["iters"])
    rate = biggest["iters_per_s"]

    rows = []
    anchor_iters, anchor_rss = biggest["iters"], biggest["peak_rss_gib"]
    for target in TARGETS:
        # Anchor on the largest observed pilot and scale by the fitted growth exponent.
        rss = anchor_rss * (target / anchor_iters) ** b
        rss_pess = anchor_rss * (target / anchor_iters) ** (b + PESSIMISTIC_EXPONENT_ADD)
        hours = target / rate / 3600 if rate else None
        rows.append(
            {
                "iters": target,
                "projected_peak_rss_gib": round(rss, 1),
                "pessimistic_peak_rss_gib": round(rss_pess, 1),
                "optimistic_hours": round(hours, 2) if hours else None,
                "pessimistic_hours": round(hours * PESSIMISTIC_TIME_FACTOR, 2) if hours else None,
                "fits_budget": rss <= args.budget_gib,
                "fits_budget_pessimistic": rss_pess <= args.budget_gib,
            }
        )
    recommended = None
    for row in rows:
        if row["iters"] <= 200_000_000 and row["fits_budget_pessimistic"]:
            recommended = row["iters"]
    fits_plan = any(r["iters"] >= 100_000_000 and r["fits_budget"] for r in rows)
    detail = (
        f"{note}. Budget {args.budget_gib} GiB. Recommended target: "
        + (
            f"{recommended // 1_000_000}M iterations (fits even pessimistically)"
            if recommended
            else "none of 50M/100M/200M fits pessimistically: shrink the abstraction"
        )
        + ". Check the pilot's eval output before trusting a blueprint."
    )
    print(f"{'iters':>6} {'RSS GiB':>8} {'pess GiB':>9} {'opt h':>7} {'pess h':>7}  fits")
    for r in rows:
        print(
            f"{r['iters'] // 1_000_000:>5}M {r['projected_peak_rss_gib']:>8} "
            f"{r['pessimistic_peak_rss_gib']:>9} {r['optimistic_hours']!s:>7} "
            f"{r['pessimistic_hours']!s:>7}  {r['fits_budget']}/{r['fits_budget_pessimistic']}"
        )
    path = write_result(
        "spike_b_extrapolation",
        {
            "fit": {"a": a, "b": b, "note": note},
            "pilots_used": [
                {"iters": p["iters"], "peak_rss_gib": p["peak_rss_gib"]} for p in pilots
            ],
            "budget_gib": args.budget_gib,
            "projections": rows,
            "recommended_iters": recommended,
            "gates": [gate("G0.2", fits_plan, detail)],
        },
    )
    print(f"G0.2 {'PASS' if fits_plan else 'FAIL'}: {detail}\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
