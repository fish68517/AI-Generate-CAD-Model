from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from common import PROJECT_ROOT, write_json


def command_output(command: list[str]) -> dict:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return {
            "available": True,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "reports/environment.json"
    )
    args = parser.parse_args()

    packages = {
        name: package_version(name)
        for name in ("cadquery", "cadquery-ocp", "numpy", "matplotlib", "pillow", "pytest")
    }
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "system": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": __import__("os").cpu_count(),
            "disk_free_gib": round(shutil.disk_usage(PROJECT_ROOT).free / 1024**3, 3),
        },
        "commands": {
            "git": command_output(["git", "--version"]),
            "nvidia_smi": command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version,compute_cap",
                    "--format=csv,noheader",
                ]
            ),
            "docker": command_output(["docker", "--version"]),
        },
        "packages": packages,
        "local_scope_ready": bool(packages["cadquery"] and packages["numpy"]),
        "model_scope": "not_checked; official 13B model is intentionally outside local duties",
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["local_scope_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

