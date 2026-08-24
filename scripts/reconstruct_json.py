"""CPU-only CADmium minimal-JSON reconstruction using Shapely and Trimesh.

This is a lightweight local implementation for the subset used by CADmium:
line/arc/circle sketches, two-sided extrusion, and New/Join/Cut/Intersect booleans.
It intentionally does not replace the paper's PythonOCC evaluation pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import traceback
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import trimesh
from scipy.spatial.transform import Rotation
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union


def _arc_points(start: np.ndarray, middle: np.ndarray, end: np.ndarray, segments: int) -> np.ndarray:
    x1, y1 = start
    x2, y2 = middle
    x3, y3 = end
    determinant = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(determinant) < 1e-10:
        return np.linspace(start, end, max(2, segments // 2))
    ux = (
        (x1 * x1 + y1 * y1) * (y2 - y3)
        + (x2 * x2 + y2 * y2) * (y3 - y1)
        + (x3 * x3 + y3 * y3) * (y1 - y2)
    ) / determinant
    uy = (
        (x1 * x1 + y1 * y1) * (x3 - x2)
        + (x2 * x2 + y2 * y2) * (x1 - x3)
        + (x3 * x3 + y3 * y3) * (x2 - x1)
    ) / determinant
    center = np.array([ux, uy])
    radius = float(np.linalg.norm(start - center))
    angles = np.mod(np.arctan2(np.array([y1, y2, y3]) - uy, np.array([x1, x2, x3]) - ux), 2 * np.pi)
    start_angle, middle_angle, end_angle = angles
    ccw_total = (end_angle - start_angle) % (2 * np.pi)
    ccw_middle = (middle_angle - start_angle) % (2 * np.pi)
    if ccw_middle <= ccw_total + 1e-9:
        sweep = ccw_total
    else:
        sweep = -((start_angle - end_angle) % (2 * np.pi))
    count = max(4, int(math.ceil(abs(sweep) / (2 * np.pi) * segments)) + 1)
    theta = np.linspace(start_angle, start_angle + sweep, count)
    return center + radius * np.column_stack((np.cos(theta), np.sin(theta)))


def _curve_points(key: str, value: dict, arc_segments: int, circle_segments: int) -> np.ndarray:
    kind = key.split("_", 1)[0]
    if kind == "line":
        return np.asarray([value["Start Point"], value["End Point"]], dtype=float)
    if kind == "arc":
        return _arc_points(
            np.asarray(value["Start Point"], dtype=float),
            np.asarray(value["Mid Point"], dtype=float),
            np.asarray(value["End Point"], dtype=float),
            arc_segments,
        )
    if kind == "circle":
        center = np.asarray(value["Center"], dtype=float)
        radius = float(value["Radius"])
        theta = np.linspace(0, 2 * np.pi, circle_segments, endpoint=False)
        return center + radius * np.column_stack((np.cos(theta), np.sin(theta)))
    raise ValueError(f"不支持的曲线类型：{key}")


def _connect_curves(curves: list[np.ndarray], tolerance: float = 1e-5) -> np.ndarray:
    if len(curves) == 1:
        return curves[0]
    remaining = [curve.copy() for curve in curves]
    chain = remaining.pop(0)
    while remaining:
        endpoint = chain[-1]
        choices = []
        for index, curve in enumerate(remaining):
            choices.append((float(np.linalg.norm(endpoint - curve[0])), index, False))
            choices.append((float(np.linalg.norm(endpoint - curve[-1])), index, True))
        distance, index, reverse = min(choices, key=lambda item: item[0])
        curve = remaining.pop(index)
        if reverse:
            curve = curve[::-1]
        if distance <= tolerance:
            chain = np.vstack((chain, curve[1:]))
        else:
            chain = np.vstack((chain, curve))
    if np.linalg.norm(chain[0] - chain[-1]) <= tolerance:
        chain[-1] = chain[0]
    else:
        chain = np.vstack((chain, chain[0]))
    return chain


def _loop_ring(loop: dict, arc_segments: int, circle_segments: int) -> np.ndarray:
    curves = [
        _curve_points(key, value, arc_segments, circle_segments)
        for key, value in loop.items()
        if value is not None
    ]
    if not curves:
        raise ValueError("loop 中没有可用曲线")
    return _connect_curves(curves)


def _face_polygon(face: dict, arc_segments: int, circle_segments: int) -> Polygon | MultiPolygon:
    rings = [_loop_ring(loop, arc_segments, circle_segments) for loop in face.values() if loop]
    if not rings:
        raise ValueError("face 中没有有效 loop")
    ring_polygons = [Polygon(ring).buffer(0) for ring in rings]
    outer_index = max(range(len(ring_polygons)), key=lambda index: ring_polygons[index].area)
    outer = rings[outer_index]
    holes = [rings[index] for index in range(len(rings)) if index != outer_index]
    polygon = Polygon(outer, holes=holes).buffer(0)
    if polygon.is_empty or polygon.area <= 0:
        raise ValueError("草图多边形为空或面积为零")
    return polygon


def _extrude_geometry(geometry, height: float) -> trimesh.Trimesh:
    polygons: Iterable[Polygon]
    if isinstance(geometry, Polygon):
        polygons = [geometry]
    elif isinstance(geometry, MultiPolygon):
        polygons = list(geometry.geoms)
    else:
        raise ValueError(f"不支持的平面几何：{geometry.geom_type}")
    meshes = [trimesh.creation.extrude_polygon(polygon, height=height) for polygon in polygons]
    if not meshes:
        raise ValueError("没有生成挤出 mesh")
    return meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)


def _part_mesh(part: dict, arc_segments: int, circle_segments: int) -> tuple[trimesh.Trimesh, str]:
    sketch = part["sketch"]
    face_polygons = [
        _face_polygon(face, arc_segments, circle_segments) for face in sketch.values() if face
    ]
    plane_geometry = unary_union(face_polygons).buffer(0)
    extrusion = part["extrusion"]
    towards = float(extrusion.get("extrude_depth_towards_normal", 0.0))
    opposite = float(extrusion.get("extrude_depth_opposite_normal", 0.0))
    height = towards + opposite
    if height <= 0:
        raise ValueError(f"拉伸总深度必须大于 0，当前为 {height}")
    mesh = _extrude_geometry(plane_geometry, height)
    mesh.apply_translation([0.0, 0.0, -opposite])

    coordinate_system = part["coordinate_system"]
    angles = coordinate_system.get("Euler Angles", [0.0, 0.0, 0.0])
    origin = np.asarray(coordinate_system.get("Translation Vector", [0.0, 0.0, 0.0]), dtype=float)
    rotation = Rotation.from_euler("zyx", angles, degrees=True).as_matrix()
    transform = np.eye(4)
    # Match the official row-vector operation: local_vector @ rotation.
    transform[:3, :3] = rotation.T
    transform[:3, 3] = origin
    mesh.apply_transform(transform)
    mesh.remove_unreferenced_vertices()
    return mesh, extrusion.get("operation", "NewBodyFeatureOperation")


def reconstruct(value: dict, arc_segments: int = 32, circle_segments: int = 64) -> tuple[trimesh.Trimesh, list[str]]:
    current: trimesh.Trimesh | None = None
    warnings: list[str] = []
    for part_name, part in value.get("parts", {}).items():
        if part is None:
            continue
        part_mesh, operation = _part_mesh(part, arc_segments, circle_segments)
        try:
            if current is None:
                if operation in {"CutFeatureOperation", "IntersectFeatureOperation"}:
                    raise ValueError(f"首个实体不能执行 {operation}")
                current = part_mesh
            elif operation in {"NewBodyFeatureOperation", "JoinFeatureOperation"}:
                current = trimesh.boolean.union([current, part_mesh], engine="manifold")
            elif operation == "CutFeatureOperation":
                current = trimesh.boolean.difference([current, part_mesh], engine="manifold")
            elif operation == "IntersectFeatureOperation":
                current = trimesh.boolean.intersection([current, part_mesh], engine="manifold")
            else:
                raise ValueError(f"未知布尔操作：{operation}")
        except Exception as exc:
            raise RuntimeError(f"{part_name} 的 {operation} 失败：{type(exc).__name__}: {exc}") from exc
    if current is None or len(current.faces) == 0:
        raise ValueError("未生成任何有效实体")
    current.remove_unreferenced_vertices()
    return current, warnings


def _save_preview(mesh: trimesh.Trimesh, path: Path, title: str) -> None:
    vertices = mesh.vertices
    faces = mesh.faces
    fig = plt.figure(figsize=(7, 6))
    axis = fig.add_subplot(111, projection="3d")
    axis.plot_trisurf(
        vertices[:, 0], vertices[:, 1], faces, vertices[:, 2],
        color="#60a5fa", edgecolor="#1e3a8a", linewidth=0.08, alpha=0.95,
    )
    extents = np.maximum(mesh.extents, 1e-6)
    center = mesh.bounding_box.centroid
    radius = float(extents.max() / 2)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev=25, azim=40)
    axis.set_title(title)
    axis.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_html(mesh: trimesh.Trimesh, path: Path, title: str) -> None:
    v, f = mesh.vertices, mesh.faces
    figure = go.Figure(
        data=[
            go.Mesh3d(
                x=v[:, 0], y=v[:, 1], z=v[:, 2],
                i=f[:, 0], j=f[:, 1], k=f[:, 2],
                color="#60a5fa", opacity=1.0, flatshading=False,
            )
        ]
    )
    figure.update_layout(
        title=title,
        scene={"aspectmode": "data"},
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
    )
    figure.write_html(path, include_plotlyjs="cdn", full_html=True)


def reconstruct_record(
    uid: str,
    value: dict,
    output_root: Path,
    category: str = "unknown",
    arc_segments: int = 32,
    circle_segments: int = 64,
    export_html: bool = True,
) -> dict:
    safe_uid = uid.replace("/", "_")
    output_dir = output_root / safe_uid
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "input.json").write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    status = {"uid": uid, "category": category, "status": "S3", "success": False}
    try:
        mesh, warnings = reconstruct(value, arc_segments, circle_segments)
        stl_path = output_dir / "model.stl"
        mesh.export(stl_path)
        _save_preview(mesh, output_dir / "preview.png", uid)
        if export_html:
            _save_html(mesh, output_dir / "interactive.html", uid)
        status.update(
            {
                "status": "S5" if mesh.is_watertight else "S4",
                "success": True,
                "watertight": bool(mesh.is_watertight),
                "vertices": int(len(mesh.vertices)),
                "faces": int(len(mesh.faces)),
                "volume": float(mesh.volume),
                "surface_area": float(mesh.area),
                "euler_number": int(mesh.euler_number),
                "extents": [float(value) for value in mesh.extents],
                "warnings": warnings,
                "stl": str(stl_path.resolve()),
            }
        )
    except Exception as exc:
        status.update(
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=4),
            }
        )
    (output_dir / "status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    return status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/samples/test_sample.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/reconstruction"))
    parser.add_argument("--arc-segments", type=int, default=32)
    parser.add_argument("--circle-segments", type=int, default=64)
    parser.add_argument("--no-html", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records: list[tuple[str, str, dict]] = []
    if args.input.suffix.lower() == ".jsonl":
        for line in args.input.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            records.append((row["uid"], row.get("category", "unknown"), json.loads(row["json_desc"])))
    else:
        records.append((args.input.stem, "single", json.loads(args.input.read_text(encoding="utf-8"))))

    statuses = [
        reconstruct_record(
            uid, value, args.output_dir, category,
            args.arc_segments, args.circle_segments, not args.no_html,
        )
        for uid, category, value in records
    ]
    state_counts = Counter(item["status"] for item in statuses)
    category_success: dict[str, dict[str, int]] = {}
    for item in statuses:
        bucket = category_success.setdefault(item["category"], {"total": 0, "success": 0, "watertight": 0})
        bucket["total"] += 1
        bucket["success"] += int(item["success"])
        bucket["watertight"] += int(item.get("watertight", False))
    summary = {
        "input": str(args.input.resolve()),
        "total": len(statuses),
        "success": sum(item["success"] for item in statuses),
        "watertight": sum(item.get("watertight", False) for item in statuses),
        "state_counts": dict(state_counts),
        "category_results": category_success,
        "results": statuses,
        "note": "本结果来自 Shapely+Trimesh 本机轻量重建器，不等同于论文 PythonOCC 指标口径。",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    fig, axis = plt.subplots(figsize=(8, 4.5))
    names = list(category_success)
    totals = [category_success[name]["total"] for name in names]
    successes = [category_success[name]["success"] for name in names]
    axis.bar(names, totals, color="#dbeafe", label="total")
    axis.bar(names, successes, color="#2563eb", label="reconstructed")
    axis.set_ylim(0, max(totals + [1]) + 1)
    axis.set_title("CPU reconstruction by category")
    axis.tick_params(axis="x", rotation=25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "reconstruction_summary.png", dpi=180)
    plt.close(fig)

    print(json.dumps({key: summary[key] for key in ("total", "success", "watertight", "state_counts", "category_results")}, indent=2, ensure_ascii=False))
    return 0 if summary["success"] == summary["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

