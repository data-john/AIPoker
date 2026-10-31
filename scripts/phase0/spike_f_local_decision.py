#!/usr/bin/env python
"""Spike F: can an open, local decision model make sensible preflop decisions? (no paid APIs)

Replaces the hosted Jev experiment. Every one of the 169 hand classes is posed in five unopened-pot
positions (845 spots) as a fold/raise decision to a local model, and the answers are compared with
the rough reference in aipoker.poker_ref (Chen ranking + typical six-max opening widths). This is a
plausibility check, not a strength measurement: the reference is NOT GTO.

Backends (all local; non-loopback hosts are refused unless --allow-remote):
  mock    deterministic fake policy, for dry runs and CI (results are labelled MOCK)
  direct  dependency-free: max_tokens=1 chat call to llama-server / Ollama / vLLM, label logprobs
  jevper  the Apache-2.0 `jevper` package; use --api systemone for kev, Ollaya or CLM servers

Examples:
  python scripts/phase0/spike_f_local_decision.py --backend mock
  python scripts/phase0/stub_llm_server.py --port 8099 &      # plumbing test, no model needed
  python scripts/phase0/spike_f_local_decision.py --backend direct --base-url http://127.0.0.1:8099/v1 --limit 100
  python scripts/phase0/spike_f_local_decision.py --backend direct --base-url http://127.0.0.1:8080/v1 --model qwen
  python scripts/phase0/spike_f_local_decision.py --backend jevper --api systemone \
      --base-url http://127.0.0.1:8009/v1 --model kev-latest
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

from common import REPO, gate, write_result

sys.path.insert(0, str(REPO / "src"))
from aipoker.agents.external.local_decision import (  # noqa: E402
    CachedBackend,
    DirectLogprobBackend,
    JevperBackend,
    MockBackend,
    PermutationAveraged,
    RemoteHostError,
)
from aipoker.agents.external.poker_prompts import preflop_unopened_spot  # noqa: E402
from aipoker.poker_ref import (  # noqa: E402
    POSITIONS,
    RFI_WIDTH_PCT,
    combos,
    hand_classes,
    percentile_table,
    reference_action,
    spearman,
)

# Proposed usefulness thresholds (informational, not a gate): tune once real numbers exist.
MIN_VALID_RATE = 0.99
MIN_SPEARMAN = 0.60
MIN_AGREEMENT = 0.75


def mock_policy(state: str, question) -> dict[str, float]:
    """Noisy version of the reference so the pipeline can be exercised without a model."""
    import hashlib
    import re

    from stub_llm_server import hand_class_from_prompt

    cls = hand_class_from_prompt(state)
    pos = re.search(r"Your position: (\w+)", state).group(1)
    noise = (int(hashlib.sha256(state.encode()).hexdigest()[:6], 16) / 0xFFFFFF - 0.5) * 6
    p = 1 / (1 + math.exp(-(RFI_WIDTH_PCT[pos] - percentile_table()[cls] + noise) / 5))
    return {"raise": p, "fold": 1 - p}


def build_backend(args):
    if args.backend == "mock":
        inner = MockBackend(mock_policy)
    elif args.backend == "direct":
        inner = DirectLogprobBackend(args.base_url, args.model, allow_remote=args.allow_remote)
    else:
        inner = JevperBackend(
            args.base_url,
            args.model,
            api=args.api,
            method=args.method,
            allow_remote=args.allow_remote,
        )
    backend = inner
    if args.backend != "mock":
        if args.permutations > 1:  # option order biases every model a little; average it out
            backend = PermutationAveraged(backend, args.permutations)
        if args.cache:
            backend = CachedBackend(backend, Path(args.cache))
    return backend


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--backend", choices=["mock", "direct", "jevper"], default="mock")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="local")
    ap.add_argument("--api", choices=["chat_completions", "systemone"], default="chat_completions")
    ap.add_argument("--method", default="logprobs", help="jevper method for chat_completions")
    ap.add_argument("--permutations", type=int, default=2, help="option-order rotations to average")
    ap.add_argument("--limit", type=int, help="only the first N spots (quick test)")
    ap.add_argument("--cache", default=str(REPO / "work" / "spike_f_cache.jsonl"))
    ap.add_argument("--label", help="name for the result file, e.g. kev4b or qwen-instruct")
    ap.add_argument("--allow-remote", action="store_true", help="allow a LAN (non-loopback) server")
    args = ap.parse_args()

    try:
        backend = build_backend(args)
    except (RemoteHostError, ImportError) as exc:
        print(f"error: {exc}")
        return 2

    spots = [(c, pos) for pos in POSITIONS for c in hand_classes()]
    if args.limit:  # stratified: every k-th spot, so all positions and hand strengths appear
        spots = spots[:: max(1, math.ceil(len(spots) / args.limit))]
    is_mock = args.backend == "mock"
    print(
        f"Spike F: backend={backend.name} spots={len(spots)}{' (MOCK: not a model result)' if is_mock else ''}"
    )

    rows, failures, latencies, first_error = [], 0, [], None
    for i, (cls, pos) in enumerate(spots, 1):
        state, question = preflop_unopened_spot(cls, pos)
        try:
            ans = backend.decide(state, question)
        except Exception as exc:  # noqa: BLE001 - count and continue; report the first error
            failures += 1
            first_error = first_error or f"{type(exc).__name__}: {exc}"
            if failures >= 5 and not rows:
                print("first 5 calls failed; aborting. Is the server running?\n  " + first_error)
                break
            continue
        if not ans.cached:
            latencies.append(ans.latency_ms)
        rows.append(
            {
                "cls": cls,
                "pos": pos,
                "p_raise": ans.probabilities["raise"],
                "choice": ans.choice,
                "ref": reference_action(cls, pos),
            }
        )
        if i % 100 == 0:
            print(f"  {i}/{len(spots)}")

    if not rows:
        detail = f"no successful decisions. {first_error}"
        path = write_result(
            "spike_f_local_decision",
            {"backend": backend.name, "is_mock": is_mock, "gates": [gate("F", False, detail)]},
        )
        print(f"F FAIL: {detail}\nwrote {path}")
        return 1

    n = len(rows)
    valid_rate = n / len(spots)
    agree = sum(r["choice"] == r["ref"] for r in rows) / n
    agree_w = sum(combos(r["cls"]) for r in rows if r["choice"] == r["ref"]) / sum(
        combos(r["cls"]) for r in rows
    )
    by_pos = {}
    for pos in POSITIONS:
        sub = [r for r in rows if r["pos"] == pos]
        if not sub:
            continue
        weight = sum(combos(r["cls"]) for r in sub)
        by_pos[pos] = {
            "mean_p_raise_combo_weighted": round(
                sum(r["p_raise"] * combos(r["cls"]) for r in sub) / weight, 3
            ),
            "target_width": RFI_WIDTH_PCT[pos] / 100,
            "agreement": round(sum(r["choice"] == r["ref"] for r in sub) / len(sub), 3),
            "n": len(sub),
        }
    # Rank correlation between P(raise) and hand strength, averaged over positions.
    rhos = []
    for pos in POSITIONS:
        sub = [r for r in rows if r["pos"] == pos]
        if len(sub) >= 20:
            rhos.append(
                spearman([r["p_raise"] for r in sub], [-percentile_table()[r["cls"]] for r in sub])
            )
    rho = statistics.mean(rhos) if rhos else float("nan")
    order = [
        by_pos[p]["mean_p_raise_combo_weighted"] for p in ("UTG", "HJ", "CO", "BTN") if p in by_pos
    ]
    monotone = all(a <= b + 0.02 for a, b in zip(order, order[1:], strict=False))
    lat = {"calls_timed": len(latencies)}
    if latencies:
        lat |= {
            "p50_ms": round(pct(latencies, 0.5), 1),
            "p95_ms": round(pct(latencies, 0.95), 1),
            "mean_ms": round(statistics.mean(latencies), 1),
        }

    useful = valid_rate >= MIN_VALID_RATE and rho >= MIN_SPEARMAN and agree >= MIN_AGREEMENT
    verdict = (
        "MOCK" if is_mock else ("promising" if useful else "weak: not worth an arena slot yet")
    )
    detail = (
        f"{backend.name}: valid {valid_rate:.1%}, agreement {agree:.1%} (combo-weighted "
        f"{agree_w:.1%}), Spearman {rho:.2f}, widens toward button: {monotone}, "
        f"p50 latency {lat.get('p50_ms', 'n/a')} ms. Verdict: {verdict}."
    )
    name = "spike_f_local_decision" + (f"_{args.label}" if args.label else "")
    path = write_result(
        name,
        {
            "backend": backend.name,
            "is_mock": is_mock,
            "spots": len(spots),
            "answered": n,
            "failures": failures,
            "first_error": first_error,
            "metrics": {
                "valid_rate": round(valid_rate, 4),
                "agreement": round(agree, 4),
                "agreement_combo_weighted": round(agree_w, 4),
                "spearman_mean": round(rho, 3),
                "position_widening": monotone,
                "by_position": by_pos,
                "latency": lat,
            },
            "thresholds": {
                "valid_rate": MIN_VALID_RATE,
                "spearman": MIN_SPEARMAN,
                "agreement": MIN_AGREEMENT,
            },
            "verdict": verdict,
            "reference_note": "Chen-formula ranking with typical RFI widths; not GTO",
            "gates": [gate("F", None if is_mock else useful, detail)],
        },
    )
    for pos, v in by_pos.items():
        print(
            f"  {pos:<4} P(raise)={v['mean_p_raise_combo_weighted']:.2f} target={v['target_width']:.2f} "
            f"agreement={v['agreement']:.2f}"
        )
    print(f"F {verdict.upper()}: {detail}\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
