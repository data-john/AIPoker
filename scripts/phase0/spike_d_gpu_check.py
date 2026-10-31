#!/usr/bin/env python
"""Spike D: GPU / ROCm check for the RX 9060 XT (gate G0.3).

Run this inside the *GPU venv* (see docs/phase0/README.md), not the main project venv:
    python3 -m venv .venv-gpu && source .venv-gpu/bin/activate
    # install the ROCm build of PyTorch matching your ROCm version, then:
    python scripts/phase0/spike_d_gpu_check.py

It reports what the system exposes (/dev/kfd, render group, rocminfo), what PyTorch sees, and
times a small MLP training loop on CPU and (if present) GPU. G0.3 passes when a GPU is visible and
the loop runs faster on it. If torch is not installed the spike is recorded as skipped.
"""

from __future__ import annotations

import argparse
import grp
import os
import shutil
import subprocess
import time
from pathlib import Path

from common import gate, write_result


def system_view() -> dict:
    view: dict = {
        "dev_kfd_exists": Path("/dev/kfd").exists(),
        "render_nodes": sorted(p.name for p in Path("/dev/dri").glob("renderD*"))
        if Path("/dev/dri").exists()
        else [],
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
        "rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES"),
    }
    try:
        groups = {grp.getgrgid(g).gr_name for g in os.getgroups()}
        view["in_render_group"] = "render" in groups
        view["in_video_group"] = "video" in groups
    except KeyError:
        pass
    for tool in ("rocminfo", "rocm-smi"):
        view[f"{tool}_path"] = shutil.which(tool)
    if view["rocminfo_path"]:
        try:
            out = subprocess.run(
                ["rocminfo"], capture_output=True, text=True, timeout=30, check=False
            ).stdout
            view["rocm_gfx_agents"] = sorted({w for w in out.split() if w.startswith("gfx")})
        except (OSError, subprocess.TimeoutExpired):
            view["rocm_gfx_agents"] = "rocminfo failed"
    return view


def mlp_bench(torch, device: str, steps: int, batch: int, width: int) -> float:
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(width, width),
        torch.nn.ReLU(),
        torch.nn.Linear(width, width),
        torch.nn.ReLU(),
        torch.nn.Linear(width, 16),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(batch, width, device=device)
    y = torch.randn(batch, 16, device=device)
    loss_fn = torch.nn.MSELoss()

    def sync() -> None:
        if device != "cpu":
            torch.cuda.synchronize()

    for _ in range(10):  # warm-up (kernel compilation, allocator)
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()
    sync()
    start = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()
    sync()
    return time.perf_counter() - start


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.steps = 100

    data: dict = {"system_view": system_view()}
    try:
        import torch
    except ImportError as exc:
        detail = (
            f"torch not importable ({exc}). Create the GPU venv and install the ROCm build of "
            "PyTorch; see docs/phase0/README.md"
        )
        data.update(status="skipped", gates=[gate("G0.3", None, detail)])
        path = write_result("spike_d_gpu", data)
        print(f"G0.3 PENDING: {detail}\nwrote {path}")
        return 0

    info = {
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "cuda_build": torch.version.cuda,
        "gpu_available": bool(torch.cuda.is_available()),
        "devices": [],
    }
    if info["gpu_available"]:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["devices"].append(
                {"index": i, "name": props.name, "vram_gib": round(props.total_memory / 2**30, 2)}
            )
    data["torch"] = info
    print(f"torch {info['torch']} hip={info['hip']} gpu_available={info['gpu_available']}")
    for dev in info["devices"]:
        print(f"  device {dev['index']}: {dev['name']} ({dev['vram_gib']} GiB)")

    cpu_s = mlp_bench(torch, "cpu", args.steps, args.batch, args.width)
    data["cpu_seconds"] = round(cpu_s, 3)
    print(f"  CPU: {cpu_s:.2f}s for {args.steps} steps")

    passed: bool | None
    if info["gpu_available"]:
        gpu_s = mlp_bench(torch, "cuda", args.steps, args.batch, args.width)
        data["gpu_seconds"] = round(gpu_s, 3)
        data["speedup"] = round(cpu_s / gpu_s, 2)
        passed = gpu_s < cpu_s
        detail = (
            f"GPU {info['devices'][0]['name']}: {gpu_s:.2f}s vs CPU {cpu_s:.2f}s "
            f"(speedup {data['speedup']}x). Check device index 0 is the RX 9060 XT, not the iGPU."
        )
        if info["hip"] is None:
            detail += " WARNING: this torch is not a ROCm build (torch.version.hip is None)."
            passed = False
    else:
        passed = False
        detail = (
            "No GPU visible to PyTorch. Check: ROCm-build wheel installed, /dev/kfd present, user "
            "in render and video groups, HIP_VISIBLE_DEVICES, kernel driver (6.18 mainline: use "
            "the in-kernel amdgpu driver, not DKMS)."
        )
    data["gates"] = [gate("G0.3", passed, detail)]
    path = write_result("spike_d_gpu", data)
    print(f"G0.3 {'PASS' if passed else 'FAIL'}: {detail}\nwrote {path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
