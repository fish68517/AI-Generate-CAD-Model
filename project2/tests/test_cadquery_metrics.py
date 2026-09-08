from pathlib import Path

import cadquery as cq

from compute_iou_best import align_and_iou
from verify_fillet import verify


def test_iou_is_invariant_to_translation_scale_and_axis_rotation() -> None:
    original = cq.Workplane("XY").box(1.0, 2.0, 3.0)
    transformed = (
        cq.Workplane("XY")
        .box(2.0, 4.0, 6.0)
        .rotate((0, 0, 0), (0, 0, 1), 90)
        .translate((4.0, -3.0, 2.0))
    )
    _, score = align_and_iou(transformed, original)
    assert score > 0.999


def test_fillet_verification_detects_changed_curved_edges(tmp_path: Path) -> None:
    base = cq.Workplane("XY").box(1.0, 1.0, 0.4)
    filleted = base.edges().fillet(0.05)
    base_path = tmp_path / "base.step"
    fillet_path = tmp_path / "fillet.step"
    cq.exporters.export(base, str(base_path))
    cq.exporters.export(filleted, str(fillet_path))
    report = verify(base_path, fillet_path)
    assert report["geometry_changed"] is True
    assert report["new_curved_edge_evidence"] is True
    assert report["fillet_verified"] is True
