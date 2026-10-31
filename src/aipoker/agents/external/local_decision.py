"""Local, open-source stand-in for a hosted 'Jev-style' decision model. No paid APIs.

A decision model takes a situation and a closed set of options and returns a probability for each
option. Here that is done with open-weight models served on this machine:

* DirectLogprobBackend: dependency-free. One chat call with max_tokens=1 against any local
  OpenAI-compatible server (llama.cpp's llama-server, Ollama, vLLM...). The options are labelled
  A, B, C...; the probabilities are the softmax over the label tokens' logprobs.
* JevperBackend: uses the Apache-2.0 `jevper` package, which implements the Jev interface over any
  OpenAI-compatible model and can also talk to the open decision servers kev, Ollaya and CLM
  (api='systemone'). Optional dependency; pinned in pyproject because it is young.
* MockBackend: deterministic fake for tests and dry runs.

Wrappers: PermutationAveraged (cancels option-order bias), CachedBackend (disk cache, makes runs
reproducible and cheap to repeat). Every network backend refuses non-loopback hosts unless
allow_remote=True, so nothing can silently go to a paid cloud endpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SYSTEM_PROMPT = (
    "You are a decision engine. Read the situation and choose the single best option. "
    "Reply with only the letter of that option."
)


class RemoteHostError(ValueError):
    """Raised when a backend is pointed at a non-loopback host without allow_remote=True."""


class LogprobsUnavailable(RuntimeError):
    """The server answered but returned no usable token log-probabilities."""


def assert_local_url(url: str, allow_remote: bool = False) -> str:
    host = urllib.parse.urlparse(url).hostname or ""
    if host in {"localhost", "::1"} or host.startswith("127."):
        return url
    if allow_remote:
        return url
    raise RemoteHostError(
        f"refusing non-local host {host!r}: this project uses local open models only. "
        "Pass allow_remote=True only for a server on your own network."
    )


@dataclass(frozen=True)
class DecisionQuestion:
    instructions: str
    options: dict[str, str]  # option key -> description, in presentation order

    def rotated(self, k: int) -> DecisionQuestion:
        items = list(self.options.items())
        k %= len(items)
        return DecisionQuestion(self.instructions, dict(items[k:] + items[:k]))


@dataclass
class DecisionAnswer:
    choice: str
    probabilities: dict[str, float]
    latency_ms: float
    n_calls: int = 1
    cached: bool = False

    @property
    def confidence(self) -> float:
        k = len(self.probabilities)
        return (max(self.probabilities.values()) - 1 / k) / (1 - 1 / k) if k > 1 else 1.0


class DecisionBackend(Protocol):
    name: str

    def decide(self, state: str, question: DecisionQuestion) -> DecisionAnswer: ...


def normalise(probs: dict[str, float]) -> dict[str, float]:
    total = sum(probs.values())
    if not total or not math.isfinite(total):
        raise ValueError(f"cannot normalise {probs}")
    return {k: v / total for k, v in probs.items()}


def build_user_prompt(state: str, question: DecisionQuestion) -> str:
    lines = [f"Situation:\n{state.strip()}", "", question.instructions.strip(), "", "Options:"]
    for label, (_, desc) in zip(LABELS, question.options.items(), strict=False):
        lines.append(f"{label}) {desc}")
    lines += ["", "Answer with the single letter of the best option."]
    return "\n".join(lines)


def probs_from_top_logprobs(top: list[dict], labels: list[str]) -> dict[str, float]:
    """Softmax over label tokens. Labels missing from the top-k get half the smallest seen mass."""
    if not top:
        raise LogprobsUnavailable("empty top_logprobs")
    mass = dict.fromkeys(labels, 0.0)
    for entry in top:
        tok = str(entry.get("token", "")).strip().upper()
        if tok in mass:
            mass[tok] += math.exp(float(entry["logprob"]))
    floor = math.exp(min(float(e["logprob"]) for e in top)) * 0.5
    return normalise({k: (v if v > 0 else floor) for k, v in mass.items()})


class MockBackend:
    """Deterministic fake. `policy(state, question) -> {option_key: weight}`; default uniform."""

    name = "mock"

    def __init__(self, policy: Callable[[str, DecisionQuestion], dict[str, float]] | None = None):
        self.policy = policy
        self.calls = 0

    def decide(self, state: str, question: DecisionQuestion) -> DecisionAnswer:
        self.calls += 1
        default = dict.fromkeys(question.options, 1.0)
        weights = self.policy(state, question) if self.policy else default
        probs = normalise({k: weights[k] for k in question.options})
        return DecisionAnswer(max(probs, key=probs.get), probs, 0.0)


class DirectLogprobBackend:
    """One max_tokens=1 chat call to a local OpenAI-compatible server; read label logprobs."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080/v1",
        model: str = "local",
        api_key: str = "local",
        timeout: float = 120.0,
        top_logprobs: int = 20,
        extra_body: dict | None = None,
        allow_remote: bool = False,
    ):
        self.base_url = assert_local_url(base_url, allow_remote).rstrip("/")
        self.model, self.api_key, self.timeout = model, api_key, timeout
        self.top_logprobs = top_logprobs
        # Qwen3-style templates think by default; a label must be the first token, so switch it off.
        self.extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        self.extra_body.update(extra_body or {})
        self.name = f"direct:{model}"

    def _post(self, body: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310 - loopback only
            return json.loads(resp.read())

    def decide(self, state: str, question: DecisionQuestion) -> DecisionAnswer:
        labels = list(LABELS[: len(question.options)])
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(state, question)},
            ],
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": self.top_logprobs,
            **self.extra_body,
        }
        t0 = time.perf_counter()
        data = self._post(body)
        latency = (time.perf_counter() - t0) * 1000
        try:
            top = data["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LogprobsUnavailable(f"no logprobs in response: {str(data)[:200]}") from exc
        by_label = probs_from_top_logprobs(top, labels)
        probs = {key: by_label[label] for label, key in zip(labels, question.options, strict=True)}
        return DecisionAnswer(max(probs, key=probs.get), probs, latency)


class JevperBackend:
    """Jev-interface client from the Apache-2.0 `jevper` package (optional dependency)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api: str = "chat_completions",
        method: str = "logprobs",
        extra_body: dict | None = None,
        allow_remote: bool = False,
    ):
        try:
            from jevper import Choice, SystemOneClient
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on the optional extra
            raise ImportError("install the extra: pip install -e '.[llm]'") from exc
        self._Choice = Choice
        self.method = method
        self.api = api
        self.base_url = assert_local_url(base_url, allow_remote)
        self.client = SystemOneClient(
            OpenAI(base_url=self.base_url, api_key="local"),
            model=model,
            api=api,
            extra_body=extra_body or {},
        )
        self.name = f"jevper[{api}]:{model}"

    def decide(self, state: str, question: DecisionQuestion) -> DecisionAnswer:
        kwargs = {} if self.api == "systemone" else {"method": self.method}
        t0 = time.perf_counter()
        response = self.client.system_one(
            state=state,
            questions={
                "decision": self._Choice(
                    instructions=question.instructions, criteria=dict(question.options)
                )
            },
            **kwargs,
        )
        latency = (time.perf_counter() - t0) * 1000
        answer = response.answers["decision"]
        probs = normalise({k: float(answer.probabilities.get(k, 0.0)) for k in question.options})
        return DecisionAnswer(
            max(probs, key=probs.get),
            probs,
            latency,
            n_calls=getattr(response.usage, "n_calls", 1) or 1,
        )


class PermutationAveraged:
    """Ask with several option orders and average, cancelling position bias toward 'A'."""

    def __init__(self, inner: DecisionBackend, permutations: int = 2):
        self.inner, self.permutations = inner, permutations
        self.name = f"{inner.name}+perm{permutations}"

    def decide(self, state: str, question: DecisionQuestion) -> DecisionAnswer:
        n = len(question.options)
        k = max(1, min(self.permutations, n))
        total = dict.fromkeys(question.options, 0.0)
        latency = 0.0
        calls = 0
        for shift in range(k):
            ans = self.inner.decide(state, question.rotated(shift))
            for key, p in ans.probabilities.items():
                total[key] += p / k
            latency += ans.latency_ms
            calls += ans.n_calls
        return DecisionAnswer(max(total, key=total.get), total, latency, calls)


class CachedBackend:
    """JSONL disk cache keyed by (backend, state, question): repeat runs are free and identical."""

    def __init__(self, inner: DecisionBackend, path: Path):
        self.inner, self.path = inner, Path(path)
        self.name = inner.name
        self.hits = self.misses = 0
        self._mem: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self._mem[rec["key"]] = rec

    @staticmethod
    def _key(name: str, state: str, q: DecisionQuestion) -> str:
        blob = json.dumps([name, state, q.instructions, list(q.options.items())])
        return hashlib.sha256(blob.encode()).hexdigest()

    def decide(self, state: str, question: DecisionQuestion) -> DecisionAnswer:
        key = self._key(self.inner.name, state, question)
        if key in self._mem:
            self.hits += 1
            rec = self._mem[key]
            return DecisionAnswer(rec["choice"], rec["probabilities"], 0.0, 0, cached=True)
        self.misses += 1
        ans = self.inner.decide(state, question)
        rec = {"key": key, "choice": ans.choice, "probabilities": ans.probabilities}
        self._mem[key] = rec
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        return ans
