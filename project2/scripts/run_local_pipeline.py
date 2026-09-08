from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from common import PROJECT_ROOT, project_path, read_jsonl, write_json, write_jsonl


SCRIPTS = PROJECT_ROOT / "scripts"


def run(script: str, *arguments: str, allow_failure: bool = False) -> subprocess.CompletedProcess:
    command = [sys.executable, str(SCRIPTS / script), *arguments]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, check=False)
    if completed.returncode != 0 and not allow_failure:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(command)}")
    return completed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    reports = PROJECT_ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    run("check_local_environment.py", "--output", "reports/environment.json")
    manifest = PROJECT_ROOT / "manifests/figure4_cases.jsonl"
    if not args.skip_download or not manifest.exists():
        run("prepare_official_subset.py")
    cases = read_jsonl(manifest)
    case_results = []
    for case in cases:
        case_id = case["case_id"]
        output_dir = f"reports/ground_truth/{case_id}"
        run(
            "safe_execute_cadquery.py",
            "--input",
            case["ground_truth_code"],
            "--output-dir",
            output_dir,
            "--timeout",
            str(args.timeout),
        )
        generated_step = f"{output_dir}/model.step"
        run(
            "compute_iou_best.py",
            "--source",
            case["ground_truth_step"],
            "--target",
            case["ground_truth_step"],
            "--output",
            f"{output_dir}/iou_self.json",
        )
        run(
            "compute_iou_best.py",
            "--source",
            generated_step,
            "--target",
            case["ground_truth_step"],
            "--output",
            f"{output_dir}/iou_code_vs_official.json",
        )
        run(
            "render_step.py",
            "--input",
            case["ground_truth_step"],
            "--output",
            case["ground_truth_preview"],
            "--title",
            f"{case_id} / {case['deepcad_id']}",
        )
        iou_self = json.loads(
            (PROJECT_ROOT / output_dir / "iou_self.json").read_text(encoding="utf-8")
        )["iou_best"]
        iou_code = json.loads(
            (PROJECT_ROOT / output_dir / "iou_code_vs_official.json").read_text(
                encoding="utf-8"
            )
        )["iou_best"]
        case_results.append(
            {
                "case_id": case_id,
                "cad_execution_success": True,
                "iou_self": iou_self,
                "iou_code_vs_official": iou_code,
                "preview": case["ground_truth_preview"],
            }
        )

    first = cases[0]
    oracle_code = "data/figure5/oracle.py"
    run(
        "create_fillet_oracle.py",
        "--input",
        first["ground_truth_code"],
        "--output",
        oracle_code,
        "--radius",
        "0.01",
    )
    run(
        "safe_execute_cadquery.py",
        "--input",
        oracle_code,
        "--output-dir",
        "reports/figure5/oracle",
        "--timeout",
        str(args.timeout),
    )
    run(
        "render_step.py",
        "--input",
        "reports/figure5/oracle/model.step",
        "--output",
        "reports/figure5/oracle/preview.png",
        "--title",
        "Local CadQuery fillet oracle",
        "--color",
        "#7b8794",
    )
    fillet_check = run(
        "verify_fillet.py",
        "--base",
        f"reports/ground_truth/{first['case_id']}/model.step",
        "--fillet",
        "reports/figure5/oracle/model.step",
        "--output",
        "reports/figure5/oracle/fillet_verification.json",
        allow_failure=True,
    )
    oracle_status = (
        "local_oracle_verified" if fillet_check.returncode == 0 else "local_oracle_needs_review"
    )
    write_jsonl(
        reports / "figure5_like/results.jsonl",
        [
            {
                "name": "Local CadQuery oracle",
                "status": oracle_status,
                "preview": "reports/figure5/oracle/preview.png",
            }
        ],
    )
    run("render_figure4_grid.py")
    run("render_figure5_grid.py")

    summary = {
        "scope": "local CPU/data/CadQuery/metric/visualization duties only",
        "official_model_loaded": False,
        "case_count": len(cases),
        "cad_execution_success": sum(row["cad_execution_success"] for row in case_results),
        "iou_self_min": min(row["iou_self"] for row in case_results),
        "iou_code_vs_official_min": min(row["iou_code_vs_official"] for row in case_results),
        "iou_validation_note": (
            "iou_self validates the metric implementation. code_vs_official is a released-asset/"
            "CadQuery-kernel consistency diagnostic and can be lower for symmetric inertia axes or "
            "version-dependent boolean geometry; it is not a model score."
        ),
        "fillet_oracle_status": oracle_status,
        "cases": case_results,
        "figure4": "reports/figure4_like/figure4_local_skeleton.png",
        "figure5": "reports/figure5_like/figure5_local_skeleton.png",
        "pending": [
            "Capture five real photos",
            "Run official 13B model on a cloud GPU",
            "Fill figure4_results_template.jsonl with real model outputs",
        ],
    }
    write_json(reports / "local_pipeline_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
