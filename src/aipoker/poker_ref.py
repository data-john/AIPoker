"""Rough preflop reference used by the Phase 0 decision-model spike.

NOT a GTO solution. It ranks the 169 starting-hand classes with the Chen formula and opens the
top N% of combos by position, using commonly quoted six-max raise-first-in widths. It exists so a
candidate decision model can be sanity-checked (does it open wider on the button than UTG? does it
prefer AA to 72o?) before any real reference (NoRegrets blueprint, push/fold solver) is available.
"""

from __future__ import annotations

import math
from functools import lru_cache

RANKS = "AKQJT98765432"  # high to low
POSITIONS = ("UTG", "HJ", "CO", "BTN", "SB")
# Approximate raise-first-in widths, percent of the 1,326 starting combos.
RFI_WIDTH_PCT = {"UTG": 15.0, "HJ": 19.0, "CO": 27.0, "BTN": 43.0, "SB": 40.0}
PLAYERS_BEHIND = {"UTG": 5, "HJ": 4, "CO": 3, "BTN": 2, "SB": 1}

_CHEN_POINTS = {
    "A": 10.0,
    "K": 8.0,
    "Q": 7.0,
    "J": 6.0,
    "T": 5.0,
    "9": 4.5,
    "8": 4.0,
    "7": 3.5,
    "6": 3.0,
    "5": 2.5,
    "4": 2.0,
    "3": 1.5,
    "2": 1.0,
}


def hand_classes() -> list[str]:
    """All 169 classes: 'AA', 'AKs', 'AKo', ..."""
    out = []
    for i, hi in enumerate(RANKS):
        for j, lo in enumerate(RANKS):
            if i == j:
                out.append(hi + lo)
            elif i < j:
                out.append(hi + lo + "s")
            else:
                out.append(RANKS[j] + RANKS[i] + "o")
    return out


def combos(cls: str) -> int:
    if len(cls) == 2:
        return 6
    return 4 if cls.endswith("s") else 12


def chen_score(cls: str) -> float:
    hi, lo = cls[0], cls[1]
    points = _CHEN_POINTS[hi]
    if hi == lo:
        return max(points * 2, 5.0)
    if cls.endswith("s"):
        points += 2
    gap = RANKS.index(lo) - RANKS.index(hi) - 1
    points -= {0: 0, 1: 1, 2: 2, 3: 4}.get(gap, 5)
    if gap <= 1 and RANKS.index(hi) > RANKS.index("Q"):  # both cards below a queen
        points += 1
    return float(math.ceil(points))


@lru_cache(maxsize=1)
def strength_order() -> tuple[str, ...]:
    def key(cls: str):
        hi, lo = cls[0], cls[1]
        return (-chen_score(cls), RANKS.index(hi), RANKS.index(lo), 0 if cls.endswith("s") else 1)

    return tuple(sorted(hand_classes(), key=key))


@lru_cache(maxsize=1)
def percentile_table() -> dict[str, float]:
    """Cumulative combo percentile (0-100] of each class from strongest to weakest."""
    total = 1326
    seen = 0
    table = {}
    for cls in strength_order():
        seen += combos(cls)
        table[cls] = 100.0 * seen / total
    return table


def reference_action(cls: str, position: str) -> str:
    return "raise" if percentile_table()[cls] <= RFI_WIDTH_PCT[position] else "fold"


def representative_cards(cls: str) -> tuple[str, str]:
    """One concrete combo for a class, e.g. 'AKs' -> ('Ah', 'Kh')."""
    hi, lo = cls[0], cls[1]
    if hi == lo:
        return hi + "h", lo + "d"
    if cls.endswith("s"):
        return hi + "h", lo + "h"
    return hi + "h", lo + "d"


def spearman(xs: list[float], ys: list[float]) -> float:
    def ranks(vals: list[float]) -> list[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        out = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0
