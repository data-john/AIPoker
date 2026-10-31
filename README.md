# AIPoker: Phase 0 (feasibility spikes)

Runnable implementation of Phase 0 from the implementation plan: replace assumptions with
measurements on your machine before committing to the architecture. Nothing here needs a paid API.

```bash
scripts/phase0/setup_toolchain.sh            # apt + rustup + .venv (asks before each step; --yes to skip prompts)
source .venv/bin/activate
python scripts/phase0/run_all.py             # spikes A, C, D (pending), E, F (mock) + report
python scripts/phase0/spike_b_pilot.py --iters 10000000   # long, CPU-heavy: see docs/phase0/README.md
cat results/phase0/PHASE0_REPORT.md
```

Full instructions, the gate definitions and the GPU venv steps are in
[docs/phase0/README.md](docs/phase0/README.md).

| Spike | Script | Gate |
|---|---|---|
| A engine throughput, correctness, determinism | `spike_a_engine_bench.py` | G0.1 |
| B NoRegrets blueprint pilot and memory projection | `spike_b_pilot.py`, `spike_b_extrapolate.py` | G0.2 |
| C evaluator and equity speed | `spike_c_equity_bench.py` | G3.1 early read |
| D GPU / ROCm check (run in the GPU venv) | `spike_d_gpu_check.py` | G0.3 |
| E licence audit and manual licensing tasks | `spike_e_license_audit.py`, `docs/phase0/noregrets_licence_email.md` | E |
| F open local decision model (replaces hosted Jev) | `spike_f_local_decision.py`, `docs/phase0/local_decision_models.md` | informational |
