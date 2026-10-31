"""Shared helpers for the Phase 0 spikes: system info, result files, gate records."""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = Path(os.environ.get("AIPOKER_RESULTS", REPO / "results" / "phase0"))


def now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def system_info() -> dict:
    """Best-effort machine description (Linux /proc); never raises."""
    info: dict = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
    }
    try:
        model = None
        cores = set()
        cur_phys = None
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name") and model is None:
                model = line.split(":", 1)[1].strip()
            elif line.startswith("physical id"):
                cur_phys = line.split(":", 1)[1].strip()
            elif line.startswith("core id"):
                cores.add((cur_phys, line.split(":", 1)[1].strip()))
        if model:
            info["cpu_model"] = model
        if cores:
            info["physical_cores"] = len(cores)
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                info["mem_total_gib"] = round(int(line.split()[1]) / 1024 / 1024, 2)
                break
    except OSError:
        pass
    return info


def write_result(name: str, data: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {"spike": name, "timestamp": now_iso(), "system": system_info(), **data}
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    return path


def load_result(name: str) -> dict | None:
    path = RESULTS / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def gate(gate_id: str, passed: bool | None, detail: str) -> dict:
    """passed=None means 'could not be evaluated' (reported as PENDING)."""
    return {"id": gate_id, "passed": passed, "detail": detail}
