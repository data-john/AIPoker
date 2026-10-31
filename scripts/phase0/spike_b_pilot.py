#!/usr/bin/env python
"""Spike B: NoRegrets blueprint training pilot (feeds gate G0.2).

Clones (or reuses) the MIT-licensed NoRegrets repo, builds it, trains a small pilot blueprint
while sampling the process's resident memory, evaluates it briefly, and writes one JSON result.
Run it twice (for example 10M and 30M iterations), then run spike_b_extrapolate.py.

    python scripts/phase0/spike_b_pilot.py --iters 10000000
    python scripts/phase0/spike_b_pilot.py --iters 30000000
    python scripts/phase0/spike_b_extrapolate.py

CPU only. Needs git and a Rust toolchain (rustup) unless --binary points at a built executable.
Close heavy applications first: the numbers are for the whole machine.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from common import REPO, RESULTS, write_result

REPO_URL = "https://github.com/conorarmstrong/noregrets"
SUBCOMMAND_FLAGS = {
    "bucket-table": ["--out"],
    "train": ["--iters", "--bucket-table", "--threads", "--checkpoint", "--out"],
    "inspect": ["--blueprint"],
    "eval": ["--blueprint", "--hands", "--baseline", "--aivat"],
}


def run(cmd: list[str], cwd: Path | None = None, log: Path | None = None, dry: bool = False) -> int:
    print("$", " ".join(map(str, cmd)))
    if dry:
        return 0
    out = log.open("w") if log else None
    try:
        return subprocess.run(
            cmd, cwd=cwd, stdout=out or None, stderr=subprocess.STDOUT if out else None, check=False
        ).returncode
    finally:
        if out:
            out.close()


def capture(cmd: list[str], cwd: Path | None = None) -> str:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False).stdout.strip()


def find_binary(repo: Path) -> Path | None:
    release = repo / "target" / "release"
    preferred = release / "pluribus"
    if preferred.exists():
        return preferred
    if release.exists():
        for path in sorted(release.iterdir()):
            if path.is_file() and os.access(path, os.X_OK) and path.suffix == "":
                return path
    return None


def check_flags(binary: Path) -> list[str]:
    """Return flags the binary's --help does not mention (guards against README drift)."""
    missing = []
    for sub, flags in SUBCOMMAND_FLAGS.items():
        help_text = capture([str(binary), sub, "--help"])
        missing += [f"{sub} {flag}" for flag in flags if flag not in help_text]
    return missing


class RssSampler(threading.Thread):
    """Samples VmRSS / VmHWM of a PID from /proc every `interval` seconds."""

    def __init__(self, pid: int, interval: float):
        super().__init__(daemon=True)
        self.pid, self.interval = pid, interval
        self.samples: list[tuple[float, float]] = []
        self.peak_kib = 0
        self._stop_flag = threading.Event()
        self._t0 = time.perf_counter()

    def read(self) -> tuple[int, int] | None:
        try:
            rss = hwm = 0
            for line in Path(f"/proc/{self.pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1])
                elif line.startswith("VmHWM:"):
                    hwm = int(line.split()[1])
            return rss, hwm
        except (OSError, ValueError):
            return None

    def run(self) -> None:
        while not self._stop_flag.is_set():
            got = self.read()
            if got:
                rss, hwm = got
                self.peak_kib = max(self.peak_kib, hwm, rss)
                self.samples.append(
                    (round(time.perf_counter() - self._t0, 1), round(rss / 1024, 1))
                )
            self._stop_flag.wait(self.interval)

    def stop(self) -> None:
        self._stop_flag.set()
        self.join(timeout=5)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--iters", type=int, default=10_000_000)
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--hands", type=int, default=20_000, help="evaluation hands (0 = skip eval)")
    ap.add_argument("--repo-dir", type=Path, default=REPO / "vendor" / "noregrets")
    ap.add_argument("--ref", help="git commit/tag to pin (recorded either way)")
    ap.add_argument(
        "--binary", type=Path, help="use an already-built executable; skips clone/build"
    )
    ap.add_argument("--workdir", type=Path, default=REPO / "work" / "noregrets")
    ap.add_argument("--sample-secs", type=float, default=5.0)
    ap.add_argument("--run-tests", action="store_true", help="also run cargo test --release")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)
    commit = None
    binary = args.binary

    if binary is None:
        if not args.repo_dir.exists():
            if run(["git", "clone", REPO_URL, str(args.repo_dir)], dry=args.dry_run):
                return 2
        if args.ref:
            run(["git", "checkout", args.ref], cwd=args.repo_dir, dry=args.dry_run)
        commit = (
            capture(["git", "rev-parse", "HEAD"], cwd=args.repo_dir) if not args.dry_run else None
        )
        if not shutil.which("cargo"):
            print("cargo not found: run scripts/phase0/setup_toolchain.sh (installs rustup) first")
            return 2
        if run(
            ["cargo", "build", "--release"],
            cwd=args.repo_dir,
            log=args.workdir / "build.log",
            dry=args.dry_run,
        ):
            print(f"build failed, see {args.workdir / 'build.log'}")
            return 2
        if args.run_tests:
            run(
                ["cargo", "test", "--release"],
                cwd=args.repo_dir,
                log=args.workdir / "test.log",
                dry=args.dry_run,
            )
        binary = find_binary(args.repo_dir)
    if args.dry_run:
        print("dry run: stopping before training")
        return 0
    if binary is None or not Path(binary).exists():
        print("could not find the built binary under target/release")
        return 2

    missing = check_flags(binary)
    if missing:
        print("The binary's --help does not mention these flags (README drift?):")
        for item in missing:
            print("  -", item)
        print("Fix SUBCOMMAND_FLAGS / commands in this script before running a long job.")
        return 2

    buckets = args.workdir / "buckets.bin"
    bucket_secs = None
    if not buckets.exists():
        t0 = time.perf_counter()
        if run(
            [str(binary), "bucket-table", "--out", str(buckets)], log=args.workdir / "buckets.log"
        ):
            print("bucket-table failed")
            return 2
        bucket_secs = time.perf_counter() - t0

    tag = f"{args.iters // 1_000_000}M" if args.iters % 1_000_000 == 0 else str(args.iters)
    blueprint = args.workdir / f"pilot_{tag}.bin"
    ckpt = args.workdir / f"pilot_{tag}.ckpt"
    train_cmd = [
        str(binary),
        "train",
        "--iters",
        str(args.iters),
        "--bucket-table",
        str(buckets),
        "--threads",
        str(args.threads),
        "--checkpoint",
        str(ckpt),
        "--out",
        str(blueprint),
    ]
    print("$", " ".join(train_cmd))
    log_path = args.workdir / f"train_{tag}.log"
    t0 = time.perf_counter()
    with log_path.open("w") as log:
        proc = subprocess.Popen(train_cmd, stdout=log, stderr=subprocess.STDOUT)
        sampler = RssSampler(proc.pid, args.sample_secs)
        sampler.start()
        code = proc.wait()
        sampler.stop()
    wall = time.perf_counter() - t0

    RESULTS.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS / f"spike_b_pilot_{tag}_rss.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["elapsed_s", "rss_mib"])
        writer.writerows(sampler.samples)

    inspect_out = eval_out = None
    if code == 0 and blueprint.exists():
        inspect_out = capture([str(binary), "inspect", "--blueprint", str(blueprint)])[-2000:]
        if args.hands:
            eval_out = capture(
                [
                    str(binary),
                    "eval",
                    "--blueprint",
                    str(blueprint),
                    "--hands",
                    str(args.hands),
                    "--baseline",
                    "caller",
                    "--aivat",
                ]
            )[-2000:]

    result = {
        "iters": args.iters,
        "threads": args.threads,
        "exit_code": code,
        "train_wall_seconds": round(wall, 1),
        "iters_per_s": round(args.iters / wall, 1) if wall else None,
        "peak_rss_gib": round(sampler.peak_kib / 1024 / 1024, 3),
        "blueprint_bytes": blueprint.stat().st_size if blueprint.exists() else None,
        "bucket_table_seconds": round(bucket_secs, 1) if bucket_secs else None,
        "commit": commit,
        "binary": str(binary),
        "rss_timeline_csv": str(csv_path),
        "inspect_tail": inspect_out,
        "eval_tail": eval_out,
        "train_log": str(log_path),
    }
    path = write_result(f"spike_b_pilot_{tag}", result)
    print(f"exit={code} wall={wall:.0f}s peak RSS={result['peak_rss_gib']} GiB -> {path}")
    return code


if __name__ == "__main__":
    sys.exit(main())
