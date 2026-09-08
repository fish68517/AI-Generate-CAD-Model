from __future__ import annotations

import argparse
import json
from pathlib import Path

import cadquery as cq
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from common import project_path, write_json


def render(
    step_path: Path,
    output_path: Path,
    title: str,
    color: str = "#4f7f3f",
    elevation: float = 25.0,
    azimuth: float = -45.0,
    tolerance: float = 0.001,
) -> dict:
    workplane = cq.importers.importStep(str(step_path))
    shape = workplane.val()
    vertices, triangles = shape.tessellate(tolerance)
    points = np.array([[v.x, v.y, v.z] for v in vertices], dtype=float)
    faces = np.array(triangles, dtype=int)
    if points.size == 0 or faces.size == 0:
        raise ValueError(f"STEP tessellation is empty: {step_path}")
    polygons = points[faces]

    fig = plt.figure(figsize=(5.2, 4.5))
    axis = fig.add_subplot(111, projection="3d")
    collection = Poly3DCollection(
        polygons, facecolor=color, edgecolor="#1f2937", linewidth=0.08, alpha=1.0
    )
    axis.add_collection3d(collection)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) / 2.0
    radius = max(float((maximum - minimum).max()) / 2.0, 1e-6)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elevation, azimuth)
    axis.set_title(title)
    axis.set_axis_off()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "step": str(step_path.resolve()),
        "preview": str(output_path.resolve()),
        "vertices": int(len(points)),
        "triangles": int(len(faces)),
        "extents": (maximum - minimum).tolist(),
        "success": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="CAD solid")
    parser.add_argument("--color", default="#4f7f3f")
    args = parser.parse_args()
    output = project_path(args.output)
    report = render(project_path(args.input), output, args.title, args.color)
    write_json(output.with_suffix(".json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
