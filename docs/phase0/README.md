# Phase 0 guide

## 1. What this delivers

Phase 0 of the plan is a set of measurements, each with a pass/fail gate, so that later phases
rest on facts about *your* machine (Ryzen 7 5700G, 30.6 GiB RAM, RX 9060 XT, Mint 22.3):

| Gate | Question | Pass condition |
|---|---|---|
| G0.1 | Is PokerKit fast enough for bulk simulation? | at least 167 hands/s overall (100k hands in 10 min), zero faults, chips conserved, same seed gives identical results |
| G0.2 | Does the NoRegrets blueprint fit in RAM? | projected peak memory for 100M iterations is at most 24 GiB |
| G0.3 | Does PyTorch use the RX 9060 XT? | GPU visible, ROCm build, faster than CPU on a small MLP loop |
| G3.1 (early) | Is Python-level hand evaluation fast enough for ladder level 4? | estimated level-4 decision at most 50 ms (about 200k evaluations/s) |
| E | Are licences acceptable? | no copyleft or unknown licence in dependencies; manual items closed |
| F | Does an open local decision model make plausible preflop decisions? | informational: valid >= 99%, Spearman >= 0.60, agreement >= 75% |

## 2. Run order

1. `scripts/phase0/setup_toolchain.sh` (reads, then asks before each step).
2. Start the blueprint pilots first because they take the longest (spike B, section 3).
3. While they run, with the machine otherwise as idle as you can make it: `python scripts/phase0/run_all.py`.
   Benchmarks compete with the pilot for CPU, so for clean numbers run A and C when no pilot is running.
4. Spike D in the GPU venv (section 4). Spike F with a real model (`local_decision_models.md`).
5. `python scripts/phase0/report.py` and read `results/phase0/PHASE0_REPORT.md`.

## 3. Spike B: blueprint pilot

```bash
python scripts/phase0/spike_b_pilot.py --iters 10000000      # clones, builds, trains, samples memory
python scripts/phase0/spike_b_pilot.py --iters 30000000
python scripts/phase0/spike_b_extrapolate.py                  # projects 50M / 100M / 200M / 400M
```

- It clones `github.com/conorarmstrong/noregrets` into `vendor/` and records the commit hash. Use
  `--ref <commit>` to pin one. `--run-tests` also runs `cargo test --release` (some tests are
  wall-clock budgeted: leave the machine idle).
- Before any long job it runs `<binary> <subcommand> --help` and checks that the flags used here
  exist, and stops if the README and the binary have drifted apart.
- `--threads` defaults to logical CPUs minus 2, which leaves room for the desktop.
- Memory is sampled from `/proc` every 5 s into `results/phase0/spike_b_pilot_<N>M_rss.csv`.
- Rule of thumb from the plan: keep projected peak memory at or under about 24 GiB. Projection
  anchors on the largest pilot and scales by the fitted growth exponent; a pessimistic variant adds
  0.15 to the exponent. Time projections are optimistic: tables grow and caches miss more as
  training proceeds. Only trust a blueprint after reading the pilot's `eval` output.
- This is CPU only. The GPU plays no part in blueprint training.

## 4. Spike D: the GPU venv

Use a separate venv: the default PyPI torch wheel on Linux is a CUDA build.

```bash
sudo usermod -aG render,video "$USER"      # then log out and back in
python3 -m venv .venv-gpu && source .venv-gpu/bin/activate
# Install ROCm and the ROCm build of PyTorch for your ROCm version by following AMD's Radeon
# documentation (AMD recommends the wheels on repo.radeon.com). Do not guess the version in a URL.
python scripts/phase0/spike_d_gpu_check.py
```

Things worth knowing:
- The RX 9060 XT (gfx1200) is on AMD's support list from ROCm 6.4.2 for Ubuntu 24.04, which Mint 22.3 is based on. Mint itself is not an AMD-listed distribution, so an installer that refuses it may need the Ubuntu 24.04 repository added by hand.
- You run a mainline 6.18 kernel. ROCm's DKMS driver may not build against it: prefer the in-kernel amdgpu driver with ROCm user-space only, and keep Mint's stock kernel selectable at boot.
- The Ryzen's integrated GPU may also be listed. `HIP_VISIBLE_DEVICES=0` assumes the RX 9060 XT is device 0: check the script's printout.
- 8 GB and 16 GB versions of the card exist; the script prints the VRAM it sees.

## 5. What was tested before you got this (and what was not)

Everything was developed and run in a throwaway Linux sandbox (1 CPU core, 3 GB RAM, no GPU, no Rust).

| Piece | Status |
|---|---|
| Spike A hand loop on PokerKit 0.7.7 | Run: no faults, chips conserved, deterministic. Multi-worker scaling **not** measurable there |
| Spike C with PokerKit, eval7, phevaluator, treys | Run: all four evaluators agree on winners; AA equity vs a random hand sane |
| Spike B scripts | Run only against a **fake** binary (harness, memory sampler, flag check, extrapolation). The real NoRegrets build and flags are **untested**: the flag check will tell you quickly |
| Spike D | Skip path run; main path run only against a fake torch. **Untested on real ROCm** |
| Spike E | Run (audit of installed packages). Manual items are yours |
| Spike F | Run against a fake local server and the MOCK backend, including through `jevper`. **No real model was run**; kev, Ollaya and CLM paths are untested |
| `setup_toolchain.sh` | Syntax-checked only. **Never run** |
| Unit tests | 14 pass (`pytest`), ruff clean |

Numbers from the sandbox are not in the shipped results folder, on purpose.

## 6. Findings that already change the plan

- **PokerKit's own hand evaluator is slow** (about 6k evaluations/s on the sandbox core, against roughly 2.3M for eval7 and about 425k for phevaluator). Use PokerKit for game rules and settlement, and a C-backed evaluator for agent equity. The ratio will hold on your machine even though absolute numbers will not.
- **PokerKit 0.7.7 exists**, so the pin in the earlier spec is valid.
- **Dealing from your own seeded deck needs three details**: burn cards and hole cards must be dealt manually (do not automate card burning), the run-out count selection must be automated, and fold only when facing a bet (PokerKit warns otherwise). These are in `spike_a_engine_bench.py` and become the seed of Phase 1's adapter.
- **jevper** (Apache-2.0) already implements the Jev interface over any local OpenAI-compatible server, and can speak to the open decision servers kev, Ollaya and CLM, so no custom Jev clone is needed.
