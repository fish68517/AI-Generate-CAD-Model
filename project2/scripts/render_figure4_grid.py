from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from common import PROJECT_ROOT, project_path, read_jsonl


def draw_image_or_pending(axis, path: Path, title: str, pending: str) -> None:
    if path.exists():
        with Image.open(path) as source:
            axis.imshow(source.convert("RGB"))
    else:
        axis.set_facecolor("#f3f4f6")
        axis.text(
            0.5,
            0.5,
            pending,
            ha="center",
            va="center",
            wrap=True,
            fontsize=9,
            color="#6b7280",
            transform=axis.transAxes,
        )
    axis.set_title(title, fontsize=9)
    axis.set_axis_off()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=PROJECT_ROOT / "manifests/figure4_cases.jsonl"
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=PROJECT_ROOT / "manifests/figure4_results_template.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports/figure4_like/figure4_local_skeleton.png",
    )
    args = parser.parse_args()
    cases = read_jsonl(project_path(args.manifest))
    results = read_jsonl(project_path(args.results))
    result_map = {(row["case_id"], row["input_domain"]): row for row in results}
    fig, axes = plt.subplots(4, len(cases), figsize=(3.2 * len(cases), 11.2), squeeze=False)

    for column, case in enumerate(cases):
        case_id = case["case_id"]
        real_path = project_path(case["real_photo"])
        rendered_path = project_path(case["rendered_image"])
        gt_path = project_path(case["ground_truth_preview"])
        real_result = result_map.get((case_id, "real"), {})
        rendered_result = result_map.get((case_id, "rendered"), {})
        real_iou = real_result.get("iou_best")
        rendered_iou = rendered_result.get("iou_best")
        draw_image_or_pending(
            axes[0][column],
            real_path,
            f"{case_id} Real photo",
            f"PENDING\n{real_path.name}",
        )
        draw_image_or_pending(
            axes[1][column],
            project_path(real_result.get("prediction_preview", "__missing__")),
            f"Real prediction\nIOUbest={real_iou if real_iou is not None else 'pending'}",
            "PENDING GPU\nreal-photo inference",
        )
        draw_image_or_pending(
            axes[2][column],
            project_path(rendered_result.get("prediction_preview", "__missing__")),
            f"Rendered prediction\nIOUbest={rendered_iou if rendered_iou is not None else 'pending'}",
            "PENDING GPU\nrendered-image inference",
        )
        draw_image_or_pending(
            axes[3][column], gt_path, "Ground truth", "PENDING local GT render"
        )
        if column == 0:
            axes[0][column].set_ylabel("Real Image", fontsize=11)
            axes[1][column].set_ylabel("Real Prediction", fontsize=11)
            axes[2][column].set_ylabel("Rendered Prediction", fontsize=11)
            axes[3][column].set_ylabel("Ground Truth", fontsize=11)
    fig.suptitle(
        "CAD-Coder Figure 4 local skeleton - pending rows are not model results",
        fontsize=15,
    )
    fig.tight_layout()
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

