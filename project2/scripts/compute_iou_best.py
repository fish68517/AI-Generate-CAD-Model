from __future__ import annotations

import argparse
import json
from pathlib import Path

import cadquery as cq
import numpy as np

from common import project_path, sha256_file, write_json


def align_and_iou(source: cq.Workplane, target: cq.Workplane) -> tuple[cq.Workplane | None, float]:
    """Reproduce the CAD-Coder paper's scale/translation/principal-axis IOU."""
    source_shape = source.val()
    target_shape = target.val()
    center_source = cq.Shape.centerOfMass(source_shape)
    center_target = cq.Shape.centerOfMass(target_shape)
    inertia_source = np.array(cq.Shape.matrixOfInertia(source_shape), dtype=float)
    inertia_target = np.array(cq.Shape.matrixOfInertia(target_shape), dtype=float)
    volume_source = float(cq.Shape.computeMass(source_shape))
    volume_target = float(cq.Shape.computeMass(target_shape))
    if volume_source <= 0 or volume_target <= 0:
        raise ValueError("IOU requires two positive-volume solids")

    eig_source, axes_source = np.linalg.eigh(inertia_source)
    eig_target, axes_target = np.linalg.eigh(inertia_target)
    scale_source = float(np.sqrt(np.abs(eig_source).sum() / volume_source))
    scale_target = float(np.sqrt(np.abs(eig_target).sum() / volume_target))
    normalized_source = source.translate(-center_source).val().scale(1.0 / scale_source)
    normalized_target = target.translate(-center_target).val().scale(1.0 / scale_target)

    rotations = np.zeros((4, 3, 3))
    rotations[0] = axes_target @ axes_source.T
    for index in range(3):
        alignment = 1 - 2 * np.array(
            [index > 0, (index + 1) % 2, index % 3 <= 1], dtype=float
        )
        rotations[index + 1] = axes_target @ (alignment[None, :] * axes_source).T

    best_iou = 0.0
    best_transform: np.ndarray | None = None
    for rotation in rotations:
        transform = np.zeros((4, 4))
        transform[:3, :3] = rotation
        transform[-1, -1] = 1.0
        aligned = normalized_source.transformGeometry(cq.Matrix(transform.tolist()))
        try:
            intersection = aligned.intersect(normalized_target)
            union = aligned.fuse(normalized_target)
            union_volume = float(union.Volume())
            score = float(intersection.Volume()) / union_volume if union_volume > 0 else 0.0
        except Exception:
            score = 0.0
        if score > best_iou:
            best_iou = score
            best_transform = transform

    if best_transform is None:
        return None, 0.0
    aligned = normalized_source.transformGeometry(cq.Matrix(best_transform.tolist()))
    aligned = aligned.scale(scale_target).translate(center_target)
    return cq.Workplane(aligned), best_iou


def compute(source_path: Path, target_path: Path) -> dict:
    source = cq.importers.importStep(str(source_path))
    target = cq.importers.importStep(str(target_path))
    _, score = align_and_iou(source, target)
    return {
        "source": str(source_path.resolve()),
        "target": str(target_path.resolve()),
        "source_sha256": sha256_file(source_path),
        "target_sha256": sha256_file(target_path),
        "iou_best": score,
        "metric_scope": "paper-style principal-axis best-aligned solid IOU",
        "success": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compute(project_path(args.source), project_path(args.target))
    write_json(project_path(args.output), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

