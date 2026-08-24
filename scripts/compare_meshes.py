"""Create reproducible lightweight ground-truth/prediction comparisons.

The primary Chamfer distance intentionally mirrors CADmium's public evaluator:
each point cloud is divided by its own global coordinate range, but is not
centered or rigidly aligned.  A second centered metric is reported only as a
diagnostic to separate translation error from the remaining shape/pose error.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy.spatial import cKDTree


def normalize_pc(points: np.ndarray) -> np.ndarray:
    """Match CADmium's public ``normalize_pc`` implementation exactly."""
    scale = float(np.max(points) - np.min(points))
    return points / scale if scale > 0 else points


def center_and_normalize_pc(points: np.ndarray) -> np.ndarray:
    """Remove only bounding-box translation, then apply CADmium-style scaling."""
    bounds_center = (points.min(axis=0) + points.max(axis=0)) / 2.0
    centered = points - bounds_center
    return normalize_pc(centered)


def chamfer_distance(first: np.ndarray, second: np.ndarray) -> float:
    first_to_second = cKDTree(second).query(first)[0]
    second_to_first = cKDTree(first).query(second)[0]
    return float(np.square(first_to_second).mean() + np.square(second_to_first).mean())


def draw_mesh(
    axis,
    mesh: trimesh.Trimesh,
    title: str,
    elev: float = 25,
    azim: float = 40,
) -> None:
    vertices, faces = mesh.vertices, mesh.faces
    axis.plot_trisurf(
        vertices[:, 0], vertices[:, 1], faces, vertices[:, 2],
        color="#60a5fa", edgecolor="#1e3a8a", linewidth=0.08, alpha=1.0,
    )
    center = mesh.bounding_box.centroid
    radius = float(max(mesh.extents.max() / 2, 1e-4))
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev, azim)
    axis.set_title(title)
    axis.set_axis_off()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--points", type=int, default=8192)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ground_truth = trimesh.load_mesh(args.ground_truth, force="mesh")
    prediction = trimesh.load_mesh(args.prediction, force="mesh")
    np.random.seed(42)
    gt_points_raw = ground_truth.sample(args.points)
    pred_points_raw = prediction.sample(args.points)
    gt_points = normalize_pc(gt_points_raw)
    pred_points = normalize_pc(pred_points_raw)
    cd = chamfer_distance(gt_points, pred_points) * 1000.0
    centered_cd = chamfer_distance(
        center_and_normalize_pc(gt_points_raw),
        center_and_normalize_pc(pred_points_raw),
    ) * 1000.0
    gt_volume = float(abs(ground_truth.volume))
    pred_volume = float(abs(prediction.volume))
    volume_error = abs(gt_volume - pred_volume)
    gt_scale = float(np.max(ground_truth.vertices) - np.min(ground_truth.vertices))
    centroid_distance = float(
        np.linalg.norm(ground_truth.bounding_box.centroid - prediction.bounding_box.centroid)
    )
    report = {
        "ground_truth": str(args.ground_truth.resolve()),
        "prediction": str(args.prediction.resolve()),
        "sample_points": args.points,
        "chamfer_distance_x1000": cd,
        "paper_style_chamfer_distance_x1000": cd,
        "centered_chamfer_distance_x1000": centered_cd,
        "metric_note": (
            "paper_style follows the public CADmium normalize_pc: divide by each point cloud's "
            "global coordinate range without centering or rigid alignment; centered removes only "
            "bounding-box translation and remains a local diagnostic, not a paper metric."
        ),
        "ground_truth_volume": gt_volume,
        "prediction_volume": pred_volume,
        "volume_absolute_error": volume_error,
        "volume_relative_error": volume_error / gt_volume if gt_volume > 0 else None,
        "ground_truth_extents": ground_truth.extents.tolist(),
        "prediction_extents": prediction.extents.tolist(),
        "extent_absolute_error": np.abs(ground_truth.extents - prediction.extents).tolist(),
        "centroid_distance": centroid_distance,
        "centroid_distance_normalized_by_gt_scale": centroid_distance / gt_scale if gt_scale > 0 else None,
        "ground_truth_euler": int(ground_truth.euler_number),
        "prediction_euler": int(prediction.euler_number),
        "both_watertight": bool(ground_truth.is_watertight and prediction.is_watertight),
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fig = plt.figure(figsize=(12, 5.5))
    draw_mesh(fig.add_subplot(121, projection="3d"), ground_truth, "Ground truth")
    draw_mesh(fig.add_subplot(122, projection="3d"), prediction, "CADmium-1.5B CPU prediction")
    fig.suptitle(f"CD x1000 = {cd:.4f} | both watertight = {report['both_watertight']}")
    fig.tight_layout()
    fig.savefig(args.output_dir / "comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    views = [(25, 40, "Isometric"), (0, 0, "Front"), (90, -90, "Top")]
    fig = plt.figure(figsize=(13, 8))
    for column, (elev, azim, view_name) in enumerate(views, start=1):
        draw_mesh(
            fig.add_subplot(2, 3, column, projection="3d"),
            ground_truth,
            f"Ground truth - {view_name}",
            elev,
            azim,
        )
        draw_mesh(
            fig.add_subplot(2, 3, column + 3, projection="3d"),
            prediction,
            f"Prediction - {view_name}",
            elev,
            azim,
        )
    fig.suptitle(
        f"Multi-view error analysis | paper-style CD x1000={cd:.4f} | "
        f"centered CD x1000={centered_cd:.4f}"
    )
    fig.tight_layout()
    fig.savefig(args.output_dir / "comparison_multiview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    labels = ["X extent", "Y extent", "Z extent", "Volume"]
    gt_values = [*ground_truth.extents.tolist(), gt_volume]
    pred_values = [*prediction.extents.tolist(), pred_volume]
    positions = np.arange(len(labels))
    width = 0.36
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    axis.bar(positions - width / 2, gt_values, width, label="Ground truth", color="#1d4ed8")
    axis.bar(positions + width / 2, pred_values, width, label="Prediction", color="#f97316")
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Value in reconstructed coordinate system")
    axis.set_title("Bounding extents and volume: geometry is closed but semantically different")
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "comparison_dimensions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
