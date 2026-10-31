#!/usr/bin/env bash
# Phase 0 toolchain setup for Linux Mint 22.x / Ubuntu 24.04 (noble).
#
#   scripts/phase0/setup_toolchain.sh            # show the plan and ask before each step
#   scripts/phase0/setup_toolchain.sh --yes      # do everything without prompting
#   scripts/phase0/setup_toolchain.sh --tauri    # also install Tauri 2 system libraries (Phase 4)
#
# What it does: apt build tools, rustup (official installer), a project venv with the dev, bench
# and llm extras. What it deliberately does NOT do: install ROCm or PyTorch (the ROCm build must
# match your ROCm version: see docs/phase0/README.md section 4) or Node (needed in Phase 4).
set -euo pipefail

YES=0; TAURI=0
for arg in "$@"; do
  case "$arg" in
    --yes) YES=1 ;;
    --tauri) TAURI=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

confirm() {
  [ "$YES" -eq 1 ] && return 0
  read -r -p "$1 [y/N] " reply
  [[ "$reply" =~ ^[Yy]$ ]]
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "== System"
. /etc/os-release || true
echo "OS: ${PRETTY_NAME:-unknown}  (base: ${UBUNTU_CODENAME:-unknown})"
echo "CPU threads: $(nproc)   RAM: $(free -g | awk '/Mem:/ {print $2}') GiB"
if [ "${UBUNTU_CODENAME:-}" != "noble" ]; then
  echo "note: tested assumptions are Ubuntu 24.04 'noble' (Mint 22.x); other bases may need package-name changes"
fi

APT_PKGS=(build-essential curl wget file git pkg-config libssl-dev python3 python3-venv python3-dev)
if [ "$TAURI" -eq 1 ]; then
  APT_PKGS+=(libwebkit2gtk-4.1-dev libxdo-dev libayatana-appindicator3-dev librsvg2-dev)
fi

echo; echo "== 1/4 apt packages: ${APT_PKGS[*]}"
if confirm "Install with sudo apt?"; then
  sudo apt update
  sudo apt install -y "${APT_PKGS[@]}"
fi

echo; echo "== 2/4 Rust toolchain (needed for the NoRegrets pilot and, later, the compiled core)"
if command -v cargo >/dev/null 2>&1; then
  echo "cargo already installed: $(cargo --version)"
elif confirm "Install rustup from https://sh.rustup.rs (official installer)?"; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  echo "open a new shell or run: source \$HOME/.cargo/env"
fi

echo; echo "== 3/4 Python venv (.venv) with dev, bench and llm extras"
if confirm "Create .venv and pip install -e '.[dev,bench,llm]'?"; then
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install -e '.[dev,bench,llm]'
  python --version
fi

echo; echo "== 4/4 Checks"
if [ -d .venv ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pytest -q || echo "tests failed: fix before running the spikes"
fi
echo
echo "Next:"
echo "  source .venv/bin/activate"
echo "  python scripts/phase0/run_all.py --quick     # spikes A, C, D(pending), E, F(mock) + report"
echo "  python scripts/phase0/spike_b_pilot.py --iters 10000000   # long run: see docs/phase0/README.md"
