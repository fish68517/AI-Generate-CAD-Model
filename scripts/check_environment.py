"""Collect a machine-readable local CADmium environment report."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PACKAGES = [
    "torch",
    "transformers",
    "datasets",
    "pyarrow",
    "peft",
    "accelerate",
    "bitsandbytes",
    "trimesh",
    "manifold3d",
    "mapbox-earcut",
    "shapely",
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "plotly",
]


def run_command(command: list[str]) -> dict:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            check=False,
        )
        return {
            "available": True,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": str(exc)}


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def torch_report() -> dict:
    try:
        import torch

        report = {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count(),
            "gpus": [],
        }
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                report["gpus"].append(
                    {
                        "index": index,
                        "name": props.name,
                        "total_memory_mb": round(props.total_memory / 1024**2),
                        "compute_capability": f"{props.major}.{props.minor}",
                    }
                )
            tensor = torch.randn((256, 256), device="cuda")
            report["cuda_matrix_test"] = float((tensor @ tensor).mean().cpu())
        return report
    except Exception as exc:  # environment diagnostics must always be saved
        return {"import_error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("reports/environment.json"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    drive = Path.cwd().anchor
    disk = shutil.disk_usage(drive)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cwd": str(Path.cwd()),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "system": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "drive": drive,
            "disk_free_gb": round(disk.free / 1024**3, 2),
        },
        "tools": {
            "git": run_command(["git", "--version"]),
            "nvidia_smi": run_command(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version,compute_cap",
                    "--format=csv,noheader",
                ]
            ),
            "conda": run_command(["conda", "--version"]),
        },
        "packages": package_versions(),
        "torch": torch_report(),
    }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n环境报告已保存：{args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

