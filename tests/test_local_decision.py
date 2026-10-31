import json

import pytest
from stub_llm_server import start

from aipoker.agents.external.local_decision import (
    CachedBackend,
    DecisionQuestion,
    DirectLogprobBackend,
    LogprobsUnavailable,
    MockBackend,
    PermutationAveraged,
    RemoteHostError,
    assert_local_url,
    probs_from_top_logprobs,
)
from aipoker.agents.external.poker_prompts import preflop_unopened_spot


def test_remote_hosts_refused_by_default():
    assert_local_url("http://127.0.0.1:8080/v1")
    assert_local_url("http://localhost:11434/v1")
    for url in ("https://api.openai.com/v1", "https://api.typesafe.ai/v1", "http://example.com/v1"):
        with pytest.raises(RemoteHostError):
            assert_local_url(url)
    assert_local_url("http://192.168.1.20:8080/v1", allow_remote=True)
    with pytest.raises(RemoteHostError):
        DirectLogprobBackend(base_url="https://api.openai.com/v1")


def test_probs_from_top_logprobs_handles_spaces_case_and_missing_labels():
    top = [
        {"token": " A", "logprob": -0.1},
        {"token": "b", "logprob": -2.5},
        {"token": "zzz", "logprob": -9.0},
    ]
    p = probs_from_top_logprobs(top, ["A", "B", "C"])
    assert abs(sum(p.values()) - 1) < 1e-9
    assert p["A"] > p["B"] > p["C"] > 0
    with pytest.raises(LogprobsUnavailable):
        probs_from_top_logprobs([], ["A", "B"])


def test_direct_backend_against_stub_reads_reference_like_policy():
    server, counter = start()
    try:
        backend = DirectLogprobBackend(base_url=f"http://127.0.0.1:{server.server_port}/v1")
        state, q = preflop_unopened_spot("AA", "UTG")
        strong = backend.decide(state, q)
        state, q = preflop_unopened_spot("72o", "UTG")
        weak = backend.decide(state, q)
    finally:
        server.shutdown()
    assert strong.choice == "raise" and strong.probabilities["raise"] > 0.9
    assert weak.choice == "fold" and weak.probabilities["raise"] < 0.1
    assert counter["calls"] == 2


def test_permutation_averaging_cancels_position_bias():
    state, q = preflop_unopened_spot("T9s", "HJ")

    def p_fold(bias, permutations):
        server, _ = start(bias_a=bias)
        try:
            raw = DirectLogprobBackend(base_url=f"http://127.0.0.1:{server.server_port}/v1")
            backend = PermutationAveraged(raw, permutations) if permutations > 1 else raw
            return backend.decide(state, q).probabilities["fold"]
        finally:
            server.shutdown()

    unbiased = p_fold(0.0, 1)
    biased = p_fold(1.5, 1)  # label A is 'fold' in the default order, so fold is inflated
    averaged = p_fold(1.5, 2)
    assert biased > unbiased + 0.1
    assert abs(averaged - unbiased) < abs(biased - unbiased) / 2


def test_cache_makes_repeat_calls_free(tmp_path):
    calls = {"n": 0}

    def policy(state, q):
        calls["n"] += 1
        return {"fold": 1.0, "raise": 3.0}

    inner = MockBackend(policy)
    cached = CachedBackend(inner, tmp_path / "cache.jsonl")
    q = DecisionQuestion("pick", {"fold": "Fold", "raise": "Raise"})
    a1 = cached.decide("s", q)
    a2 = cached.decide("s", q)
    assert calls["n"] == 1 and a2.cached and a1.probabilities == a2.probabilities
    # a fresh object reading the same file also hits
    again = CachedBackend(MockBackend(policy), tmp_path / "cache.jsonl")
    assert again.decide("s", q).cached
    assert json.loads((tmp_path / "cache.jsonl").read_text().splitlines()[0])["choice"] == "raise"


def test_jevper_backend_against_stub():
    pytest.importorskip("jevper")
    pytest.importorskip("openai")
    from aipoker.agents.external.local_decision import JevperBackend

    server, _ = start()
    try:
        backend = JevperBackend(
            f"http://127.0.0.1:{server.server_port}/v1",
            "stub",
            api="chat_completions",
            method="logprobs",
        )
        state, q = preflop_unopened_spot("KK", "CO")
        ans = backend.decide(state, q)
    finally:
        server.shutdown()
    assert ans.choice == "raise" and ans.probabilities["raise"] > 0.8
    assert abs(sum(ans.probabilities.values()) - 1) < 1e-6
