#!/usr/bin/env python
"""Spike A: PokerKit engine throughput, correctness and determinism (gate G0.1).

Plays six-max NLHE hands (50/100 blinds, 10,000-chip stacks reset every hand: the planned arena
format) with a cheap random policy, from a seeded deck dealt card by card. It measures hands/s
per process and scaling across worker processes, checks chip conservation on every hand, and
checks that the same seed reproduces identical results.

G0.1 passes when the best measured throughput can play 100k hands in about 10 minutes
(>= 167 hands/s overall). Run on the real machine; sandbox numbers mean nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import multiprocessing as mp
import os
import random
import time
import warnings

from common import gate, write_result
from pokerkit import Automation, Deck, Mode, NoLimitTexasHoldem

PLAYERS = 6
BLINDS = (50, 100)
STACK = 10_000
GATE_HANDS_PER_S = 100_000 / 600  # 100k hands in 10 minutes

# Burning and dealing are manual so that every card comes from our own seeded deck.
AUTOMATIONS = (
    Automation.ANTE_POSTING,
    Automation.BET_COLLECTION,
    Automation.BLIND_OR_STRADDLE_POSTING,
    Automation.RUNOUT_COUNT_SELECTION,
    Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
    Automation.HAND_KILLING,
    Automation.CHIPS_PUSHING,
    Automation.CHIPS_PULLING,
)


def play_hand(rng: random.Random, stack: int = STACK) -> tuple[tuple[int, ...], int]:
    """Play one hand with a random policy. Returns (final stacks, number of player actions)."""
    state = NoLimitTexasHoldem.create_state(
        AUTOMATIONS, True, 0, BLINDS, BLINDS[1], stack, PLAYERS, mode=Mode.CASH_GAME
    )
    deck = list(Deck.STANDARD)
    rng.shuffle(deck)
    used = 0
    actions = 0
    while state.status:
        if state.can_deal_hole():
            state.deal_hole(deck[used : used + 1])
            used += 1
        elif state.can_burn_card():
            state.burn_card(deck[used : used + 1])
            used += 1
        elif state.can_deal_board():
            count = 3 if not state.board_cards else 1
            state.deal_board(deck[used : used + count])
            used += count
        elif state.actor_index is not None:
            seat = state.actor_index
            roll = rng.random()
            facing_bet = state.bets[seat] < max(state.bets)
            if facing_bet and roll < 0.15:
                state.fold()
            elif state.can_complete_bet_or_raise_to() and roll > 0.78:
                low = state.min_completion_betting_or_raising_to_amount
                high = state.max_completion_betting_or_raising_to_amount
                if rng.random() < 0.05:
                    amount = high
                else:
                    amount = min(high, max(low, int(low * (1 + 3 * rng.random()))))
                state.complete_bet_or_raise_to(amount)
            else:
                state.check_or_call()
            actions += 1
        else:
            raise RuntimeError(f"engine stalled: board={state.board_cards} stacks={state.stacks}")
    return tuple(state.stacks), actions


def _worker(args: tuple[int, int, float]) -> dict:
    worker_id, seed, seconds = args
    warnings.simplefilter("error")  # PokerKit warns on questionable actions: count them as faults
    hands = actions = violations = 0
    first_error = None
    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        rng = random.Random(seed + worker_id * 1_000_003 + hands + violations)
        try:
            stacks, n_actions = play_hand(rng)
            if sum(stacks) != PLAYERS * STACK:
                raise AssertionError(f"chips not conserved: {stacks}")
            hands += 1
            actions += n_actions
        except Exception as exc:  # noqa: BLE001 - a spike records faults rather than crashing
            violations += 1
            first_error = first_error or f"{type(exc).__name__}: {exc}"
    return {
        "hands": hands,
        "actions": actions,
        "violations": violations,
        "first_error": first_error,
        "seconds": time.perf_counter() - start,
    }


def run_level(workers: int, seconds: float, seed: int) -> dict:
    ctx = mp.get_context("spawn")
    t0 = time.perf_counter()
    with ctx.Pool(workers) as pool:
        parts = pool.map(_worker, [(i, seed, seconds) for i in range(workers)])
    wall = time.perf_counter() - t0
    hands = sum(p["hands"] for p in parts)
    return {
        "workers": workers,
        "hands": hands,
        "actions": sum(p["actions"] for p in parts),
        "violations": sum(p["violations"] for p in parts),
        "first_error": next((p["first_error"] for p in parts if p["first_error"]), None),
        "measured_seconds": round(max(p["seconds"] for p in parts), 3),
        "hands_per_s": round(hands / max(p["seconds"] for p in parts), 1),
        "wall_seconds_incl_startup": round(wall, 2),
    }


def reproducibility_check(n_hands: int = 200, seed: int = 12345) -> dict:
    def digest(s: int) -> str:
        rng = random.Random(s)
        h = hashlib.sha256()
        for _ in range(n_hands):
            stacks, _ = play_hand(rng)
            h.update(repr(stacks).encode())
        return h.hexdigest()

    same = digest(seed) == digest(seed)
    different = digest(seed) != digest(seed + 1)
    return {"same_seed_identical": same, "different_seed_differs": different, "hands": n_hands}


def choose_levels(cpus: int) -> list[int]:
    levels = {1, 2, 4, cpus // 2, max(1, cpus - 2), cpus}
    return sorted(level for level in levels if 1 <= level <= cpus)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seconds", type=float, default=8.0, help="duration of each level")
    parser.add_argument("--workers", type=int, nargs="*", help="worker counts to test")
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--quick", action="store_true", help="2 s per level, fewer levels")
    args = parser.parse_args()

    cpus = os.cpu_count() or 1
    levels = args.workers or ([1, cpus] if args.quick else choose_levels(cpus))
    seconds = 2.0 if args.quick else args.seconds
    levels = sorted(set(levels))

    print(f"Spike A: PokerKit throughput, levels={levels}, {seconds}s each")
    results = []
    for workers in levels:
        res = run_level(workers, seconds, args.seed)
        results.append(res)
        print(
            f"  workers={workers:>2}  {res['hands_per_s']:>8.1f} hands/s  "
            f"violations={res['violations']}"
        )
    base = next((r["hands_per_s"] for r in results if r["workers"] == 1), results[0]["hands_per_s"])
    for res in results:
        res["speedup_vs_1"] = round(res["hands_per_s"] / base, 2) if base else None
        res["efficiency"] = round(res["speedup_vs_1"] / res["workers"], 2) if base else None

    repro = reproducibility_check()
    best = max(results, key=lambda r: r["hands_per_s"])
    total_violations = sum(r["violations"] for r in results)
    ok = (
        best["hands_per_s"] >= GATE_HANDS_PER_S
        and total_violations == 0
        and repro["same_seed_identical"]
        and repro["different_seed_differs"]
    )
    detail = (
        f"best {best['hands_per_s']} hands/s at {best['workers']} workers "
        f"(need >= {GATE_HANDS_PER_S:.0f}); violations={total_violations}; "
        f"reproducible={repro['same_seed_identical']}"
    )
    path = write_result(
        "spike_a_engine",
        {
            "format": {"players": PLAYERS, "blinds": BLINDS, "stack": STACK, "reset_stacks": True},
            "levels": results,
            "reproducibility": repro,
            "best": best,
            "projected_minutes_100k_hands": round(100_000 / best["hands_per_s"] / 60, 1),
            "projected_minutes_1M_hands": round(1_000_000 / best["hands_per_s"] / 60, 1),
            "note": "random-policy hands; real agents add decision time on top",
            "gates": [gate("G0.1", ok, detail)],
        },
    )
    print(f"G0.1 {'PASS' if ok else 'FAIL'}: {detail}\nwrote {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
