from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from common import PROJECT_ROOT, project_path, read_jsonl


def image_or_placeholder(axis, path: Path, text: str) -> None:
    if path.exists():
        with Image.open(path) as source:
            axis.imshow(source.convert("RGB"))
    else:
        axis.set_facecolor("#f3f4f6")
        axis.text(0.5, 0.5, text, ha="center", va="center", transform=axis.transAxes)
    axis.set_axis_off()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=PROJECT_ROOT / "manifests/figure5_cases.jsonl"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports/figure5_like/figure5_local_skeleton.png",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=PROJECT_ROOT / "reports/figure5_like/results.jsonl",
    )
    args = parser.parse_args()
    case = read_jsonl(project_path(args.manifest))[0]
    results_path = project_path(args.results)
    result_map = {}
    if results_path.exists():
        result_map = {row["name"]: row for row in read_jsonl(results_path)}
    fig = plt.figure(figsize=(13, 7.5))
    grid = fig.add_gridspec(2, 3, height_ratios=(1.0, 1.2))
    input_axis = fig.add_subplot(grid[0, 0])
    image_or_placeholder(input_axis, project_path(case["input_image"]), "Missing input")
    input_axis.set_title("Input rendered CAD image")
    prompt_axis = fig.add_subplot(grid[0, 1:])
    prompt_axis.set_facecolor("#e8f0fb")
    prompt_axis.text(
        0.03,
        0.66,
        "First turn:\n" + case["prompt_first_turn"],
        ha="left",
        va="center",
        wrap=True,
        fontsize=10,
        transform=prompt_axis.transAxes,
    )
    prompt_axis.text(
        0.03,
        0.27,
        "Second turn:\n" + case["prompt_second_turn"],
        ha="left",
        va="center",
        wrap=True,
        fontsize=10,
        transform=prompt_axis.transAxes,
    )
    prompt_axis.set_axis_off()
    for index, variant in enumerate(case["variants"]):
        resolved = {**variant, **result_map.get(variant["name"], {})}
        axis = fig.add_subplot(grid[1, index])
        image_or_placeholder(axis, project_path(resolved["preview"]), resolved["status"])
        axis.set_title(resolved["name"] + "\n" + resolved["status"], fontsize=10)
    fig.suptitle(
        "CAD-Coder Figure 5 local skeleton - oracle is not a model prediction",
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
