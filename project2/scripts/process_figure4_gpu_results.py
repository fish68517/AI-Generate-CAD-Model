from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import PROJECT_ROOT, project_path, read_jsonl, write_jsonl
from compute_iou_best import compute
from extract_cadquery_code import extract_code
from render_step import render
from safe_execute_cadquery import execute


def validate_row(row: dict, valid_cases: set[str]) -> None:
    required = {"case_id", "input_domain"}
    missing = required - set(row)
    if missing:
        raise ValueError(f"GPU result row missing fields: {sorted(missing)}")
    if row["case_id"] not in valid_cases:
        raise ValueError(f"Unknown case_id: {row['case_id']}")
    if row["input_domain"] not in {"real", "rendered"}:
        raise ValueError(f"Invalid input_domain: {row['input_domain']}")
    if not row.get("response") and not row.get("code"):
        raise ValueError("GPU result row requires response or code")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--case-manifest",
        type=Path,
        default=PROJECT_ROOT / "manifests/figure4_cases.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "manifests/figure4_results.jsonl",
    )
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()
    cases = read_jsonl(project_path(args.case_manifest))
    case_map = {row["case_id"]: row for row in cases}
    rows = read_jsonl(project_path(args.input))
    seen = set()
    results = []
    for row in rows:
        validate_row(row, set(case_map))
        key = (row["case_id"], row["input_domain"])
        if key in seen:
            raise ValueError(f"Duplicate GPU result: {key}")
        seen.add(key)
        case_id, domain = key
        output_dir = PROJECT_ROOT / f"reports/predictions/{case_id}/{domain}"
        output_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "case_id": case_id,
            "input_domain": domain,
            "status": "pending_processing",
            "model": row.get("model"),
            "revision": row.get("revision"),
            "generation": row.get("generation", {}),
            "prediction_preview": f"reports/predictions/{case_id}/{domain}/preview.png",
            "prediction_step": f"reports/predictions/{case_id}/{domain}/model.step",
            "iou_best": None,
        }
        try:
            code = row.get("code") or extract_code(row["response"])
            code_path = output_dir / "model_code.py"
            code_path.write_text(code.rstrip() + "\n", encoding="utf-8")
            execution = execute(code_path, output_dir, args.timeout)
            record["execution"] = execution
            if not execution["success"]:
                record["status"] = execution["status"]
            else:
                prediction_step = output_dir / "model.step"
                metric = compute(
                    prediction_step,
                    project_path(case_map[case_id]["ground_truth_step"]),
                )
                record["iou_best"] = metric["iou_best"]
                preview_path = output_dir / "preview.png"
                render(
                    prediction_step,
                    preview_path,
                    f"{case_id} {domain} prediction",
                    color="#7b8794",
                )
                record["status"] = "C8_figure_rendered"
        except Exception as exc:
            record["status"] = "processing_error"
            record["error"] = f"{type(exc).__name__}: {exc}"
        results.append(record)

    write_jsonl(project_path(args.output), results)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0 if all(row["status"] == "C8_figure_rendered" for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

