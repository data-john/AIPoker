# Phase 0 report

Generated 2026-10-30T22:35:29+00:00

Machine: AMD Ryzen 7 5700G with Radeon Graphics (16 logical CPUs, 30.65 GiB RAM), Python 3.11.4

## Gates

| Gate | Status | Detail |
|---|---|---|
| G0.1 | PASS | best 1253.9 hands/s at 14 workers (need >= 167); violations=0; reproducible=True |
| G0.3 | PENDING | torch not importable (No module named 'torch'). Create the GPU venv and install the ROCm build of PyTorch; see docs/phase0/README.md |
| G3.1-early | PASS | best evaluator eval7: 2,150,653 evals/s, est L4 decision 4.6 ms (budget 50.0 ms). Python-level evaluation is enough for now. |
| C-sanity | PASS | AA equity in 0.80-0.90 for all: True; evaluators agree: True |
| E-auto | PASS | no copyleft or unknown licences among installed dependencies |
| E-manual | PENDING | 6 manual licence tasks open (see report) |
| F[mock] | MOCK (not a model result) | mock: valid 100.0%, agreement 98.2% (combo-weighted 98.6%), Spearman 1.00, widens toward button: True, p50 latency 0.0 ms. Verdict: MOCK. |

Not yet run: G0.2.

## Manual tasks still open (Spike E)

- [ ] Request NoRegrets commercial terms in writing (docs/phase0/noregrets_licence_email.md)
- [ ] Confirm in writing that a blueprint you train from the MIT source may be shipped
- [ ] Record the licence of every decision-model weight you intend to ship (kev, Ollaya, CLM, any GGUF)
- [ ] kev (jaredpalmer/kev): licence not verified
- [ ] Ollaya (ollaya-dev/ollaya) and its laya models: licence not verified
- [ ] CLM (Contrastive-LM/CLM) and its heads: licence not verified

## Raw results

- `spike_a_engine.json`
- `spike_c_equity.json`
- `spike_d_gpu.json`
- `spike_e_license_audit.json`
- `spike_f_local_decision_mock.json`
