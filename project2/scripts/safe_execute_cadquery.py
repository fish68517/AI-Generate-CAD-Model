from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from code_safety import validate_code
from common import PROJECT_ROOT, project_path, sha256_file, write_json


RUNNER = r'''from pathlib import Path
import json
import cadquery as cq

source_path = Path(__SOURCE_PATH__)
output_dir = Path(__OUTPUT_DIR__)
scope = {"__name__": "__cadquery_generated__"}
code = source_path.read_text(encoding="utf-8")
exec(compile(code, str(source_path), "exec"), scope, scope)
if "solid" not in scope:
    raise RuntimeError("CadQuery code did not define final variable 'solid'")
solid = scope["solid"]
output_dir.mkdir(parents=True, exist_ok=True)
cq.exporters.export(solid, str(output_dir / "model.step"))
cq.exporters.export(solid, str(output_dir / "model.stl"))
shape = solid.val() if hasattr(solid, "val") else solid
metrics = {
    "volume": float(shape.Volume()),
    "surface_area": float(shape.Area()),
    "edge_count": len(shape.Edges()),
    "face_count": len(shape.Faces()),
    "solid_count": len(shape.Solids()),
}
(output_dir / "geometry.json").write_text(
    json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
)
'''


def build_runner(source: Path, output_dir: Path) -> str:
    return RUNNER.replace(
        "__SOURCE_PATH__", json.dumps(str(source.resolve()))
    ).replace("__OUTPUT_DIR__", json.dumps(str(output_dir.resolve())))


def execute(
    source: Path,
    output_dir: Path,
    timeout_seconds: int = 60,
    validate_only: bool = False,
) -> dict:
    source = source.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    code = source.read_text(encoding="utf-8")
    safety = validate_code(code)
    result = {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "output_dir": str(output_dir),
        "timeout_seconds": timeout_seconds,
        "safety": safety,
        "success": False,
        "status": "F9_security_reject" if not safety["accepted"] else "C3_validated",
    }
    if not safety["accepted"] or validate_only:
        if validate_only and safety["accepted"]:
            result.update(success=True, status="C3_validate_only")
        write_json(output_dir / "result.json", result)
        return result

    runner_path = output_dir / "_restricted_runner.py"
    runner_path.write_text(build_runner(source, output_dir), encoding="utf-8")
    env = {
        "PATH": os.environ.get("PATH", ""),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        "WINDIR": os.environ.get("WINDIR", ""),
        "USERPROFILE": os.environ.get("USERPROFILE", ""),
        "HOMEDRIVE": os.environ.get("HOMEDRIVE", ""),
        "HOMEPATH": os.environ.get("HOMEPATH", ""),
        "APPDATA": os.environ.get("APPDATA", ""),
        "LOCALAPPDATA": os.environ.get("LOCALAPPDATA", ""),
        "TEMP": str(output_dir),
        "TMP": str(output_dir),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONNOUSERSITE": "1",
    }
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [sys.executable, "-I", str(runner_path)],
            cwd=output_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        result.update(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            elapsed_seconds=round(time.perf_counter() - started, 4),
        )
        step_path = output_dir / "model.step"
        stl_path = output_dir / "model.stl"
        success = completed.returncode == 0 and step_path.exists() and stl_path.exists()
        result.update(
            success=success,
            status="C6_step_exported" if success else "F4_syntax_or_runtime_error",
            step=str(step_path) if step_path.exists() else None,
            stl=str(stl_path) if stl_path.exists() else None,
        )
        if step_path.exists():
            result["step_sha256"] = sha256_file(step_path)
        if stl_path.exists():
            result["stl_sha256"] = sha256_file(stl_path)
    except subprocess.TimeoutExpired as exc:
        result.update(
            status="F9_timeout",
            elapsed_seconds=round(time.perf_counter() - started, 4),
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
        )
    finally:
        runner_path.unlink(missing_ok=True)
    write_json(output_dir / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    source = project_path(args.input)
    output_dir = project_path(args.output_dir)
    result = execute(source, output_dir, args.timeout, args.validate_only)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
