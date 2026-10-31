#!/usr/bin/env python
"""Aggregate all Phase 0 results into results/phase0/PHASE0_REPORT.md."""

from __future__ import annotations

import json

from common import RESULTS, now_iso

GATE_ORDER = ["G0.1", "G0.2", "G0.3", "G3.1-early", "C-sanity", "E-auto", "E-manual", "F"]
HINTS = {
    "G0.1": "Engine too slow: build the Rust fast path with PokerKit as test oracle (plan risk table).",
    "G0.2": "Blueprint too large for RAM: lower the iteration target or shrink the abstraction.",
    "G0.3": "GPU not usable: neural work moves to CPU or waits; the arena does not depend on it.",
    "G3.1-early": "Python evaluation too slow for level 4: plan the compiled equity core from Phase 3.",
    "C-sanity": "Evaluators disagree or equity looks wrong: investigate before trusting any benchmark.",
    "E-auto": "A dependency has a copyleft/unknown licence: review before any commercial release.",
    "F": "Decision model weak: try another model/backend or drop it from the arena.",
}


def status(passed) -> str:
    return {True: "PASS", False: "FAIL", None: "PENDING"}[passed]


def build() -> str:
    results = {}
    for path in sorted(RESULTS.glob("*.json")):
        data = json.loads(path.read_text())
        if "spike" in data:
            results[data["spike"]] = data
    gates: dict[str, tuple[str, dict]] = {}
    for name, data in results.items():
        for g in data.get("gates", []):
            key = g["id"]
            if (
                key == "F" and name != "spike_f_local_decision"
            ):  # keep real runs; labelled variants listed below
                key = f"F[{name.removeprefix('spike_f_local_decision_')}]"
            gates[key] = (name, g)

    system = next((d["system"] for d in results.values() if "system" in d), {})
    lines = ["# Phase 0 report", "", f"Generated {now_iso()}", ""]
    if system:
        lines += [
            f"Machine: {system.get('cpu_model', '?')} ({system.get('logical_cpus', '?')} logical CPUs, "
            f"{system.get('mem_total_gib', '?')} GiB RAM), Python {system.get('python', '?')}",
            "",
        ]
    lines += ["## Gates", "", "| Gate | Status | Detail |", "|---|---|---|"]
    ordered = [k for k in GATE_ORDER if k in gates] + sorted(
        k for k in gates if k not in GATE_ORDER
    )
    for key in ordered:
        spike, g = gates[key]
        mock = results[spike].get("is_mock")
        label = "MOCK (not a model result)" if mock else status(g["passed"])
        lines.append(f"| {key} | {label} | {g['detail'].replace('|', '/')} |")
    missing = [k for k in ("G0.1", "G0.2", "G0.3") if k not in gates]
    if missing:
        lines += ["", f"Not yet run: {', '.join(missing)}."]

    todo = [
        i
        for i in results.get("spike_e_license_audit", {}).get("manual_items", [])
        if i.get("status") == "TODO"
    ]
    if todo:
        lines += ["", "## Manual tasks still open (Spike E)", ""]
        lines += [f"- [ ] {i['task']}" for i in todo]
    failing = [
        k
        for k in ordered
        if gates[k][1]["passed"] is False and not results[gates[k][0]].get("is_mock")
    ]
    if failing:
        lines += ["", "## What the failures mean", ""]
        lines += [f"- **{k}**: {HINTS.get(k.split('[')[0], 'see detail')}" for k in failing]
    lines += ["", "## Raw results", ""] + [f"- `{n}.json`" for n in sorted(results)]
    return "\n".join(lines) + "\n"


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    text = build()
    out = RESULTS / "PHASE0_REPORT.md"
    out.write_text(text)
    print(text)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
