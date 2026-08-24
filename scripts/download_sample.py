"""Stream a balanced, deterministic sample from the public CADmium test set."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset


CATEGORIES = ("cut", "multi_part", "arc", "circle", "line_only")


def describe_json(value: dict) -> tuple[str, Counter, Counter]:
    parts = [part for part in value.get("parts", {}).values() if part]
    primitives: Counter = Counter()
    operations: Counter = Counter()
    for part in parts:
        operations[part.get("extrusion", {}).get("operation", "missing")] += 1
        for face in part.get("sketch", {}).values():
            for loop in face.values():
                for key in loop:
                    primitives[key.split("_", 1)[0]] += 1

    if operations["CutFeatureOperation"]:
        category = "cut"
    elif len(parts) > 1:
        category = "multi_part"
    elif primitives["arc"]:
        category = "arc"
    elif primitives["circle"]:
        category = "circle"
    else:
        category = "line_only"
    return category, primitives, operations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="chandar-lab/CADmium-ds")
    parser.add_argument("--split", default="test")
    parser.add_argument("--per-category", type=int, default=4)
    parser.add_argument("--scan-limit", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, default=Path("data/samples"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_dir = args.output_dir / "json"
    prompt_dir = args.output_dir / "prompts"
    json_dir.mkdir(exist_ok=True)
    prompt_dir.mkdir(exist_ok=True)

    selected: dict[str, list[dict]] = defaultdict(list)
    stream = load_dataset(args.dataset, split=args.split, streaming=True)
    scanned = 0
    for row in stream:
        scanned += 1
        try:
            parsed = json.loads(row["json_desc"])
            category, primitives, operations = describe_json(parsed)
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        if len(selected[category]) < args.per_category:
            item = {
                "uid": row["uid"],
                "category": category,
                "annotation": row["annotation"],
                "json_desc": row["json_desc"],
                "primitive_counts": dict(primitives),
                "operation_counts": dict(operations),
            }
            selected[category].append(item)
        if all(len(selected[name]) >= args.per_category for name in CATEGORIES):
            break
        if scanned >= args.scan_limit:
            break

    rows = [item for category in CATEGORIES for item in selected[category]]
    jsonl_path = args.output_dir / f"{args.split}_sample.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for item in rows:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            safe_uid = item["uid"].replace("/", "_")
            (json_dir / f"{safe_uid}.json").write_text(
                json.dumps(json.loads(item["json_desc"]), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            (prompt_dir / f"{safe_uid}.txt").write_text(item["annotation"], encoding="utf-8")

    digest = hashlib.sha256(jsonl_path.read_bytes()).hexdigest()
    manifest = {
        "dataset": args.dataset,
        "split": args.split,
        "streaming": True,
        "scan_limit": args.scan_limit,
        "scanned_rows": scanned,
        "selected_rows": len(rows),
        "category_counts": {name: len(selected[name]) for name in CATEGORIES},
        "sample_sha256": digest,
        "jsonl": str(jsonl_path.resolve()),
    }
    (args.output_dir / "sample_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if len(rows) != args.per_category * len(CATEGORIES):
        print("警告：在扫描上限内未找到足量的所有类别样本。")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

