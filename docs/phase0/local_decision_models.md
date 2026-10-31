# Open, local alternatives to the hosted Jev model (no paid APIs)

Jev is a hosted, paid service: you send a state and typed questions, it returns choices with
probabilities. Spike F replaces it with open models running on your own machine, behind one small
interface (`src/aipoker/agents/external/local_decision.py`).

None of these models is poker-trained. They are general decision/classification models or
language models. Spike F measures whether they make *plausible* preflop decisions; it does not
measure strength. A weak result is a perfectly good outcome: drop it and move on.

## Candidates (facts as described in jevper's own documentation; I did not run kev, Ollaya or CLM)

| Option | What it is | Fit for your machine | Licence |
|---|---|---|---|
| Plain instruct LLM via llama.cpp (`llama-server`) + `direct` backend | Any open-weight chat model. We read the next-token log-probabilities over option letters, so no sampling and no JSON parsing. | A 4-8B model at 4-bit fits 8 GB or 16 GB cards (check yours). Vulkan or HIP backends of llama.cpp; try Vulkan first, it does not need ROCm. | Choose weights under Apache-2.0 or MIT. Llama/Gemma-style custom licences need reading before any product use. |
| **kev** (jaredpalmer/kev) | Open decision models built on Qwen3.5, wire-compatible with the hosted System One API. 4B is the suggested starting size. | 4B is comfortable on a 16 GB card; 0.8B runs on CPU. | **Unverified: check before shipping** |
| **Ollaya** (ollaya-dev/ollaya, model `laya:en`) | A 421M-parameter ModernBERT model exported to ONNX (about 850 MB); one forward pass per decision, no text generation. | CPU is fine; fastest and smallest option. The installer is a `curl | sh` script: read it first. | **Unverified** |
| **CLM** (Contrastive-LM/CLM) | A frozen Qwen3-8B encoder plus small heads; deterministic. | Heavy (about 8-16 GB of model). | **Unverified** |
| **jevper** (PyPI, zhulinchng/jevper) | An independent implementation of the Jev interface on top of any OpenAI-compatible server; also speaks to kev, Ollaya and CLM with `--api systemone`. | Pure Python client. | Apache-2.0 (verified from the package metadata). Young (many releases in its first weeks, one maintainer): the version is pinned in `pyproject.toml`. |

## How to run each one against Spike F

Always start with the plumbing test, which needs no model:

```bash
python scripts/phase0/stub_llm_server.py --port 8099 &
python scripts/phase0/spike_f_local_decision.py --backend direct --base-url http://127.0.0.1:8099/v1 --model stub --label stub --limit 100
```

### A. llama.cpp + a plain instruct model (recommended first real test)

```bash
# build or install llama.cpp (Vulkan or HIP backend), download a GGUF of an Apache-2.0/MIT instruct model, then:
llama-server -m /path/to/model.gguf --host 127.0.0.1 --port 8080 -c 4096 -ngl 99 --jinja
python scripts/phase0/spike_f_local_decision.py --backend direct --base-url http://127.0.0.1:8080/v1 --model local --label llm-a
# same server through the Jev interface instead of the direct reader:
python scripts/phase0/spike_f_local_decision.py --backend jevper --base-url http://127.0.0.1:8080/v1 --model local --label llm-a-jevper
```

Qwen3-style "thinking" templates must have thinking disabled so the answer letter is the first
token; the direct backend sends `chat_template_kwargs: {enable_thinking: false}` by default.

### B. kev (dedicated decision model)

```bash
git clone https://github.com/jaredpalmer/kev.git && cd kev && pip install -e '.[serve]'   # `pip install kev` is a different package
python -m kev.serve --run jaredpalmer/kev-4b --port 8009
python scripts/phase0/spike_f_local_decision.py --backend jevper --api systemone --base-url http://127.0.0.1:8009/v1 --model kev-latest --label kev4b
```

### C. Ollaya

```bash
curl -fsSL https://ollaya.dev/install.sh -o /tmp/ollaya-install.sh   # read it before running it
sh /tmp/ollaya-install.sh && ollaya pull laya:en && ollaya serve      # listens on 127.0.0.1:11435
python scripts/phase0/spike_f_local_decision.py --backend jevper --api systemone --base-url http://127.0.0.1:11435/v1 --model laya:en --label ollaya
```

Check each project's README for current commands: these were taken from jevper's docs and may
have moved on. Servers on other machines on your LAN need `--allow-remote`; non-loopback hosts are
otherwise refused, so a typo cannot send a request to a paid cloud endpoint.

## Reading the result

Spike F reports valid-answer rate, agreement with a rough Chen-formula reference, Spearman rank
correlation with hand strength, whether opening range widens toward the button, and latency.
Proposed thresholds for "promising": valid >= 99%, Spearman >= 0.60, agreement >= 75%. These are
placeholders to tune, and the reference is **not** GTO. If a model is promising, Phase 5d tests it
on harder spots and tiny hand samples; if not, record the result and drop it.
