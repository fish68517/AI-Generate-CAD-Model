from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Callable

import requests

from common import (
    FILLET_PROMPT,
    OFFICIAL_COMMIT,
    OFFICIAL_PROMPT,
    OFFICIAL_RAW_ROOT,
    PROJECT_ROOT,
    relative_to_project,
    sha256_file,
    write_json,
    write_jsonl,
)


TEST_JSONL_PATH = "inference/cadquery_test_data_subset100.jsonl"


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    destination.write_bytes(response.content)


def code_features(row: dict) -> dict:
    code = row["ground_truth"]
    return {
        "extrude_count": code.count(".extrude("),
        "cut_count": code.count(".cut("),
        "fuse_count": code.count(".fuse(") + code.count(".union("),
        "circle_count": code.count(".circle("),
        "arc_count": code.count(".threePointArc("),
        "line_count": code.count(".lineTo("),
        "code_chars": len(code),
        "code_lines": len(code.splitlines()),
    }


def select_cases(rows: list[dict], count: int = 5) -> list[tuple[str, dict, dict]]:
    enriched = [(row, code_features(row)) for row in rows]
    enriched.sort(key=lambda item: (item[1]["code_chars"], item[0]["question_id"]))
    selected: list[tuple[str, dict, dict]] = []
    used: set[int] = set()

    selectors: list[tuple[str, Callable[[dict], bool], bool]] = [
        (
            "single_extrude_prismatic",
            lambda f: f["extrude_count"] == 1
            and f["cut_count"] == 0
            and f["circle_count"] == 0
            and f["arc_count"] == 0,
            False,
        ),
        (
            "single_extrude_curved",
            lambda f: f["extrude_count"] == 1
            and f["cut_count"] == 0
            and (f["circle_count"] > 0 or f["arc_count"] > 0),
            False,
        ),
        (
            "two_feature_boolean",
            lambda f: f["extrude_count"] == 2 and f["cut_count"] > 0,
            False,
        ),
        (
            "multi_extrude",
            lambda f: f["extrude_count"] >= 3,
            False,
        ),
        (
            "complex_long_code",
            lambda f: f["extrude_count"] >= 2,
            True,
        ),
    ]

    for label, predicate, reverse in selectors:
        candidates = [item for item in enriched if predicate(item[1])]
        if reverse:
            candidates = list(reversed(candidates))
        choice = next(
            (item for item in candidates if item[0]["question_id"] not in used), None
        )
        if choice is None:
            continue
        row, features = choice
        used.add(row["question_id"])
        selected.append((label, row, features))
        if len(selected) >= count:
            break

    if len(selected) < count:
        for row, features in enriched:
            if row["question_id"] in used:
                continue
            selected.append(("fallback", row, features))
            used.add(row["question_id"])
            if len(selected) >= count:
                break
    if len(selected) != count:
        raise RuntimeError(f"只能选择 {len(selected)} 条，期望 {count} 条")
    return selected


def read_source_rows(source_path: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in source_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    required = {"question_id", "image", "text", "ground_truth"}
    if len(rows) != 100:
        raise ValueError(f"官方固定子集应为100条，当前为 {len(rows)} 条")
    for row in rows:
        missing = required - set(row)
        if missing:
            raise ValueError(f"样本缺少字段：{sorted(missing)}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.count != 5:
        raise ValueError("图4式本地固定清单要求 count=5")

    source_dir = PROJECT_ROOT / "data/official_source"
    source_jsonl = source_dir / "cadquery_test_data_subset100.jsonl"
    source_url = f"{OFFICIAL_RAW_ROOT}/{TEST_JSONL_PATH}"
    if args.force or not source_jsonl.exists():
        download(source_url, source_jsonl)
    rows = read_source_rows(source_jsonl)
    selected = select_cases(rows, args.count)

    manifest_rows = []
    download_records = []
    for index, (selection_reason, row, features) in enumerate(selected, start=1):
        case_id = f"f4_{index:02d}"
        image_name = Path(row["image"]).name
        deepcad_id = re.sub(r"_0$", "", Path(image_name).stem)
        case_dir = PROJECT_ROOT / f"data/cases/{case_id}"
        image_path = case_dir / "rendered.png"
        step_path = case_dir / "ground_truth.step"
        code_path = case_dir / "ground_truth.py"
        image_url = f"{OFFICIAL_RAW_ROOT}/inference/test100_images/{image_name}"
        step_url = f"{OFFICIAL_RAW_ROOT}/inference/test100_gt_steps/{deepcad_id}.step"
        for url, destination in ((image_url, image_path), (step_url, step_path)):
            if args.force or not destination.exists():
                download(url, destination)
            download_records.append(
                {
                    "case_id": case_id,
                    "url": url,
                    "path": relative_to_project(destination),
                    "sha256": sha256_file(destination),
                    "bytes": destination.stat().st_size,
                }
            )
        code_path.parent.mkdir(parents=True, exist_ok=True)
        code_path.write_text(row["ground_truth"].rstrip() + "\n", encoding="utf-8")
        manifest_rows.append(
            {
                "case_id": case_id,
                "question_id": row["question_id"],
                "deepcad_id": deepcad_id,
                "selection_reason": selection_reason,
                "features": features,
                "prompt": row["text"],
                "rendered_image": relative_to_project(image_path),
                "real_photo": f"data/real_photos/{case_id}.png",
                "real_photo_status": "pending_user_capture",
                "ground_truth_code": relative_to_project(code_path),
                "ground_truth_step": relative_to_project(step_path),
                "ground_truth_preview": f"reports/ground_truth/{case_id}/preview.png",
                "source_commit": OFFICIAL_COMMIT,
                "source_image_sha256": sha256_file(image_path),
                "source_step_sha256": sha256_file(step_path),
            }
        )

    manifest_path = PROJECT_ROOT / "manifests/figure4_cases.jsonl"
    write_jsonl(manifest_path, manifest_rows)
    figure4_template = []
    for row in manifest_rows:
        for domain in ("real", "rendered"):
            figure4_template.append(
                {
                    "case_id": row["case_id"],
                    "input_domain": domain,
                    "status": "pending_gpu_inference",
                    "prediction_preview": f"reports/predictions/{row['case_id']}/{domain}/preview.png",
                    "prediction_step": f"reports/predictions/{row['case_id']}/{domain}/model.step",
                    "iou_best": None,
                }
            )
    write_jsonl(
        PROJECT_ROOT / "manifests/figure4_results_template.jsonl", figure4_template
    )

    first = manifest_rows[0]
    figure5 = {
        "case_id": "f5_01",
        "source_case_id": first["case_id"],
        "input_image": first["rendered_image"],
        "base_code": first["ground_truth_code"],
        "prompt_first_turn": OFFICIAL_PROMPT,
        "prompt_second_turn": FILLET_PROMPT,
        "fillet_radius": 0.01,
        "base_preview": first["ground_truth_preview"],
        "variants": [
            {
                "name": "CAD-Coder released 13B",
                "status": "pending_gpu_inference",
                "preview": "reports/figure5/cadcoder/preview.png",
            },
            {
                "name": "Qwen text code-edit baseline",
                "status": "pending_optional_experiment",
                "preview": "reports/figure5/qwen_text/preview.png",
            },
            {
                "name": "Local CadQuery oracle",
                "status": "pending_local_execution",
                "preview": "reports/figure5/oracle/preview.png",
            },
        ],
    }
    write_jsonl(PROJECT_ROOT / "manifests/figure5_cases.jsonl", [figure5])
    write_json(
        PROJECT_ROOT / "data/source_manifest.json",
        {
            "official_repository": "https://github.com/anniedoris/CAD-Coder",
            "official_commit": OFFICIAL_COMMIT,
            "source_test_jsonl": {
                "url": source_url,
                "path": relative_to_project(source_jsonl),
                "sha256": sha256_file(source_jsonl),
                "bytes": source_jsonl.stat().st_size,
            },
            "selected_count": len(manifest_rows),
            "downloads": download_records,
        },
    )
    print(
        json.dumps(
            {
                "source_rows": len(rows),
                "selected": [
                    {
                        "case_id": row["case_id"],
                        "deepcad_id": row["deepcad_id"],
                        "reason": row["selection_reason"],
                    }
                    for row in manifest_rows
                ],
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
