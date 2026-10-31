#!/usr/bin/env python
"""A tiny fake OpenAI-compatible chat server for testing the decision-model plumbing offline.

It answers /v1/chat/completions with token logprobs shaped like llama.cpp / vLLM output. The
'model' is the rough preflop reference from aipoker.poker_ref plus an optional bias toward the
first option label (to exercise the permutation-averaging wrapper). It is NOT a language model:
use it only to check that spike F's parsing, caching and metrics work end to end.

    python scripts/phase0/stub_llm_server.py --port 8099 --bias-a 0.8
    python scripts/phase0/spike_f_local_decision.py --backend direct \
        --base-url http://127.0.0.1:8099/v1 --limit 40
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from aipoker.poker_ref import RFI_WIDTH_PCT, percentile_table  # noqa: E402


def hand_class_from_prompt(text: str) -> str | None:
    m = re.search(r"Your hand: (\w\w) (\w\w)\.", text)
    if not m:
        return None
    (r1, s1), (r2, s2) = m.group(1), m.group(2)
    if r1 == r2:
        return r1 + r2
    order = "AKQJT98765432"
    hi, lo = sorted((r1, r2), key=order.index)
    return hi + lo + ("s" if s1 == s2 else "o")


def stub_probabilities(text: str, bias_a: float) -> dict[str, float]:
    cls = hand_class_from_prompt(text)
    pos = re.search(r"Your position: (\w+)", text)
    # Accepts both our own prompt ("B) Raise to ...") and jevper's ("B: raise — Raise to ...").
    raise_label = re.search(r"^([A-Z])[:)] .*[Rr]aise", text, flags=re.M)
    labels = re.findall(r"^([A-Z])[:)] ", text, flags=re.M)
    if not (cls and pos and raise_label and labels):
        return {lab: 1 / max(1, len(labels)) for lab in labels}
    width = RFI_WIDTH_PCT[pos.group(1)]
    p_raise = 1 / (1 + math.exp(-(width - percentile_table()[cls]) / 5))
    logits = {lab: 0.0 for lab in labels}
    logits[raise_label.group(1)] = math.log(p_raise / (1 - p_raise + 1e-9) + 1e-9)
    logits["A"] = logits.get("A", 0.0) + bias_a
    z = sum(math.exp(v) for v in logits.values())
    return {k: math.exp(v) / z for k, v in logits.items()}


def make_handler(bias_a: float, counter: dict):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # silence
            pass

        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            counter["calls"] = counter.get("calls", 0) + 1
            user = "\n".join(str(m["content"]) for m in body["messages"] if m["role"] == "user")
            probs = stub_probabilities(user, bias_a)
            top = [
                {"token": k, "logprob": math.log(v), "bytes": [ord(k)]} for k, v in probs.items()
            ]
            top += [{"token": f"x{i}", "logprob": -25.0, "bytes": [120]} for i in range(18)]
            best = max(probs, key=probs.get)
            payload = {
                "id": "stub",
                "object": "chat.completion",
                "created": 0,
                "model": body.get("model"),
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": best},
                        "logprobs": {
                            "content": [
                                {
                                    "token": best,
                                    "logprob": math.log(probs[best]),
                                    "bytes": [ord(best)],
                                    "top_logprobs": top,
                                }
                            ]
                        },
                    }
                ],
                "usage": {"prompt_tokens": 50, "completion_tokens": 1, "total_tokens": 51},
            }
            raw = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    return Handler


def start(port: int = 0, bias_a: float = 0.0) -> tuple[ThreadingHTTPServer, dict]:
    counter: dict = {}
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(bias_a, counter))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, counter


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--bias-a", type=float, default=0.0)
    args = ap.parse_args()
    srv, _ = start(args.port, args.bias_a)
    print(
        f"stub LLM server on http://127.0.0.1:{srv.server_port}/v1 (bias_a={args.bias_a}); Ctrl-C to stop"
    )
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        srv.shutdown()
