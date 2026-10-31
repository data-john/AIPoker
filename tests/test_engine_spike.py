import random

from spike_a_engine_bench import PLAYERS, STACK, play_hand, reproducibility_check


def test_hands_conserve_chips_and_terminate():
    rng = random.Random(99)
    for _ in range(60):
        stacks, actions = play_hand(rng)
        assert sum(stacks) == PLAYERS * STACK
        assert actions > 0


def test_same_seed_reproduces_and_different_seed_differs():
    res = reproducibility_check(n_hands=30, seed=5)
    assert res["same_seed_identical"] and res["different_seed_differs"]
