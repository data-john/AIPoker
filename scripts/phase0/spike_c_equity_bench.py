#!/usr/bin/env python
"""Spike C: 7-card evaluator and Monte Carlo equity throughput (early read on gate G3.1).

Benchmarks every evaluator that imports (PokerKit's built-in, plus eval7, phevaluator and treys
if installed via `pip install -e '.[bench]'`), checks that they agree on who wins, and estimates
how long one ladder level-4 decision would take.

Planning assumption (documented, adjustable): one level-4 decision needs about 10,000 7-card
evaluations (reweighting 1,326 opponent combos on the board plus a Monte Carlo equity estimate
against up to five opponents). To stay under about 50 ms per decision, an evaluator needs
>= 200,000 evaluations/s; otherwise plan the compiled equity core (Rust + PyO3) from Phase 3.
"""

from __future__ import annotations

import argparse
import random
import time
from collections.abc import Callable

from common import gate, write_result
from pokerkit import Card, StandardHighHand

EVALS_PER_L4_DECISION = 10_000
DECISION_BUDGET_MS = 50.0

RANKS = "23456789TJQKA"
SUITS = "cdhs"
ALL_CARDS = [r + s for r in RANKS for s in SUITS]


def _pokerkit_native() -> tuple[Callable[[str], object], Callable[[list, list], object]]:
    cache = {c: next(iter(Card.parse(c))) for c in ALL_CARDS}
    return cache.__getitem__, lambda hole, board: StandardHighHand.from_game(hole, board)


def _eval7_native():
    import eval7

    cache = {c: eval7.Card(c) for c in ALL_CARDS}
    return cache.__getitem__, lambda hole, board: eval7.evaluate(hole + board)


def _phevaluator_native():
    from phevaluator import evaluate_cards

    return (lambda c: c), lambda hole, board: -evaluate_cards(*hole, *board)


def _treys_native():
    import treys

    evaluator = treys.Evaluator()
    cache = {c: treys.Card.new(c) for c in ALL_CARDS}
    return cache.__getitem__, lambda hole, board: -evaluator.evaluate(board, hole)


EVALUATORS = {
    "pokerkit": _pokerkit_native,
    "eval7": _eval7_native,
    "phevaluator": _phevaluator_native,
    "treys": _treys_native,
}


def load_evaluators() -> tuple[dict, dict]:
    ok, skipped = {}, {}
    for name, factory in EVALUATORS.items():
        try:
            ok[name] = factory()
        except Exception as exc:  # noqa: BLE001 - optional dependency may be missing or broken
            skipped[name] = f"{type(exc).__name__}: {exc}"
    return ok, skipped


def random_deals(n: int, seed: int) -> list[tuple[list[str], list[str]]]:
    rng = random.Random(seed)
    deals = []
    for _ in range(n):
        cards = rng.sample(ALL_CARDS, 7)
        deals.append((cards[:2], cards[2:]))
    return deals


def time_evals(native, score, deals, min_seconds: float) -> float:
    prepared = [([native(c) for c in h], [native(c) for c in b]) for h, b in deals]
    done = 0
    start = time.perf_counter()
    while True:
        for hole, board in prepared:
            score(hole, board)
        done += len(prepared)
        elapsed = time.perf_counter() - start
        if elapsed >= min_seconds:
            return done / elapsed


def agreement(evaluators: dict, n_pairs: int, seed: int) -> dict:
    """Compare winner decisions between evaluators on identical random showdowns."""
    rng = random.Random(seed)
    cases = []
    for _ in range(n_pairs):
        cards = rng.sample(ALL_CARDS, 9)
        cases.append((cards[:2], cards[2:4], cards[4:]))

    def outcomes(native, score) -> list[int]:
        out = []
        for a, b, board in cases:
            nb = [native(c) for c in board]
            sa = score([native(c) for c in a], nb)
            sb = score([native(c) for c in b], nb)
            out.append((sa > sb) - (sa < sb))
        return out

    results = {name: outcomes(*ev) for name, ev in evaluators.items()}
    reference_name = "pokerkit" if "pokerkit" in results else next(iter(results))
    ref = results[reference_name]
    return {
        "reference": reference_name,
        "pairs": n_pairs,
        "mismatches": {
            name: sum(1 for x, y in zip(ref, out, strict=True) if x != y)
            for name, out in results.items()
            if name != reference_name
        },
    }


def mc_equity(native, score, hero: list[str], samples: int, seed: int) -> tuple[float, float]:
    """Hero vs one random hand, full random 5-card board. Returns (equity, seconds)."""
    rng = random.Random(seed)
    remaining = [c for c in ALL_CARDS if c not in hero]
    hero_n = [native(c) for c in hero]
    wins = 0.0
    start = time.perf_counter()
    for _ in range(samples):
        draw = rng.sample(remaining, 7)
        opp = [native(c) for c in draw[:2]]
        board = [native(c) for c in draw[2:]]
        a, b = score(hero_n, board), score(opp, board)
        wins += 1.0 if a > b else 0.5 if a == b else 0.0
    return wins / samples, time.perf_counter() - start


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--deals", type=int, default=5000)
    parser.add_argument("--seconds", type=float, default=2.0, help="timing window per evaluator")
    parser.add_argument("--mc-samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.deals, args.seconds, args.mc_samples = 1000, 0.5, 500

    evaluators, skipped = load_evaluators()
    if not evaluators:
        print("no evaluator available")
        return 2
    deals = random_deals(args.deals, args.seed)

    print(f"Spike C: evaluators={list(evaluators)} skipped={list(skipped)}")
    rows = {}
    for name, (native, score) in evaluators.items():
        eps = time_evals(native, score, deals, args.seconds)
        equity, secs = mc_equity(native, score, ["Ac", "Ad"], args.mc_samples, args.seed)
        rows[name] = {
            "evals_per_s": round(eps),
            "est_l4_decision_ms": round(EVALS_PER_L4_DECISION / eps * 1000, 1),
            "mc_samples": args.mc_samples,
            "mc_seconds": round(secs, 3),
            "aa_equity_vs_random": round(equity, 3),
        }
        print(
            f"  {name:<12} {eps:>10,.0f} evals/s  L4 est {rows[name]['est_l4_decision_ms']:>8.1f} ms"
            f"  AA equity {equity:.3f}"
        )

    agree = agreement(evaluators, 1500 if not args.quick else 300, args.seed)
    best_name = max(rows, key=lambda n: rows[n]["evals_per_s"])
    best = rows[best_name]
    equity_ok = all(0.80 <= r["aa_equity_vs_random"] <= 0.90 for r in rows.values())
    agree_ok = all(m == 0 for m in agree["mismatches"].values())
    fast_enough = best["est_l4_decision_ms"] <= DECISION_BUDGET_MS
    detail = (
        f"best evaluator {best_name}: {best['evals_per_s']:,} evals/s, est L4 decision "
        f"{best['est_l4_decision_ms']} ms (budget {DECISION_BUDGET_MS} ms). "
        + (
            "Python-level evaluation is enough for now."
            if fast_enough
            else "Plan the compiled equity core (Rust + PyO3) from Phase 3."
        )
    )
    path = write_result(
        "spike_c_equity",
        {
            "assumption_evals_per_l4_decision": EVALS_PER_L4_DECISION,
            "evaluators": rows,
            "skipped": skipped,
            "agreement": agree,
            "sanity": {"aa_equity_in_range": equity_ok, "evaluators_agree": agree_ok},
            "best_evaluator": best_name,
            "gates": [
                gate("G3.1-early", fast_enough, detail),
                gate(
                    "C-sanity",
                    equity_ok and agree_ok,
                    f"AA equity in 0.80-0.90 for all: {equity_ok}; evaluators agree: {agree_ok}",
                ),
            ],
        },
    )
    print(f"G3.1-early {'PASS' if fast_enough else 'COMPILE'}: {detail}")
    print(f"agreement mismatches vs {agree['reference']}: {agree['mismatches']}\nwrote {path}")
    return 0 if equity_ok and agree_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
