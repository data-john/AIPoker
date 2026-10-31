"""Turn poker spots into (state text, DecisionQuestion) for a decision model."""

from __future__ import annotations

from aipoker.agents.external.local_decision import DecisionQuestion
from aipoker.poker_ref import PLAYERS_BEHIND, representative_cards

RAISE_TO_BB = 2.5


def preflop_unopened_spot(hand_class: str, position: str) -> tuple[str, DecisionQuestion]:
    c1, c2 = representative_cards(hand_class)
    behind = PLAYERS_BEHIND[position]
    state = "\n".join(
        [
            "Game: six-max No-Limit Texas Hold'em cash game, 100 big blinds effective, "
            "blinds 0.5/1, no ante.",
            "Street: preflop.",
            f"Your position: {position} ({behind} player{'s' if behind != 1 else ''} still to act "
            "after you, including the blinds).",
            f"Your hand: {c1} {c2}.",
            "Action so far: every player before you folded. The pot is unopened (1.5 big blinds).",
        ]
    )
    question = DecisionQuestion(
        instructions="You are a strong, disciplined six-max cash game player. "
        "What is the best action?",
        options={"fold": "Fold", "raise": f"Raise to {RAISE_TO_BB} big blinds"},
    )
    return state, question
