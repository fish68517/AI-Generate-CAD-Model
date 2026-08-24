"""Generate repeatable example figures from existing local reconstruction output."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


CATEGORY_NAMES = {
    "cut": "Boolean cut",
    "multi_part": "Multiple bodies",
    "arc": "Contains arc",
    "circle": "Contains circle",
    "line_only": "Lines only",
}


def _load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _preview_path(reconstruction_dir: Path, uid: str) -> Path:
    return reconstruction_dir / uid.replace("/", "_") / "preview.png"


def _draw_image(axis, path: Path, title: str) -> None:
    with Image.open(path) as source:
        axis.imshow(source.convert("RGB"))
    axis.set_title(title, fontsize=9)
    axis.set_axis_off()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, default=Path("data/samples/test_sample.jsonl"))
    parser.add_argument("--reconstruction-dir", type=Path, default=Path("reports/reconstruction"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/examples"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(args.sample)
    available: list[tuple[dict, Path]] = []
    missing: list[str] = []
    for row in rows:
        preview = _preview_path(args.reconstruction_dir, row["uid"])
        if preview.exists():
            available.append((row, preview))
        else:
            missing.append(row["uid"])

    if not available:
        raise FileNotFoundError("没有找到可用于例图的 preview.png，请先运行 reconstruct_json.py")

    grouped: dict[str, list[tuple[dict, Path]]] = defaultdict(list)
    for item in available:
        grouped[item[0].get("category", "unknown")].append(item)

    categories = list(grouped)
    max_columns = max(len(items) for items in grouped.values())
    fig, axes = plt.subplots(
        len(categories), max_columns,
        figsize=(3.4 * max_columns, 3.0 * len(categories)),
        squeeze=False,
    )
    for row_index, category in enumerate(categories):
        for column in range(max_columns):
            axis = axes[row_index][column]
            if column < len(grouped[category]):
                record, preview = grouped[category][column]
                _draw_image(
                    axis,
                    preview,
                    f"{CATEGORY_NAMES.get(category, category)}\n{record['uid']}",
                )
            else:
                axis.set_axis_off()
    fig.suptitle("CADmium local CPU reconstruction gallery: 20 fixed ground-truth samples", fontsize=15)
    fig.tight_layout()
    gallery_path = args.output_dir / "ground_truth_gallery_20.png"
    fig.savefig(gallery_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    representatives = [items[0] for items in grouped.values()]
    fig, axes = plt.subplots(1, len(representatives), figsize=(3.5 * len(representatives), 3.5), squeeze=False)
    representative_rows = []
    for index, (record, preview) in enumerate(representatives):
        category = record.get("category", "unknown")
        _draw_image(
            axes[0][index],
            preview,
            f"{CATEGORY_NAMES.get(category, category)}\n{record['uid']}",
        )
        representative_rows.append(
            {
                "uid": record["uid"],
                "category": category,
                "preview": str(preview.resolve()),
                "primitive_counts": record.get("primitive_counts", {}),
                "operation_counts": record.get("operation_counts", {}),
            }
        )
    fig.suptitle("One representative per local validation category", fontsize=14)
    fig.tight_layout()
    representative_path = args.output_dir / "category_representatives_5.png"
    fig.savefig(representative_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "sample": str(args.sample.resolve()),
        "available_count": len(available),
        "missing_count": len(missing),
        "missing_uids": missing,
        "category_counts": {key: len(value) for key, value in grouped.items()},
        "gallery": str(gallery_path.resolve()),
        "representative_gallery": str(representative_path.resolve()),
        "representatives": representative_rows,
        "note": "这些例图展示真实 JSON 的 CPU 轻量重建覆盖面，不代表模型预测质量。",
    }
    (args.output_dir / "example_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
