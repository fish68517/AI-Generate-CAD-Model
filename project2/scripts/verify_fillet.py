from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cadquery as cq

from common import project_path, write_json


def shape_summary(path: Path) -> dict:
    shape = cq.importers.importStep(str(path)).val()
    edge_types = Counter(edge.geomType() for edge in shape.Edges())
    return {
        "volume": float(shape.Volume()),
        "surface_area": float(shape.Area()),
        "edge_count": len(shape.Edges()),
        "face_count": len(shape.Faces()),
        "edge_types": dict(sorted(edge_types.items())),
        "curved_edges": sum(
            count for kind, count in edge_types.items() if kind.upper() != "LINE"
        ),
    }


def verify(base_path: Path, fillet_path: Path) -> dict:
    base = shape_summary(base_path)
    fillet = shape_summary(fillet_path)
    volume_delta = fillet["volume"] - base["volume"]
    surface_delta = fillet["surface_area"] - base["surface_area"]
    changed = abs(volume_delta) > 1e-10 or abs(surface_delta) > 1e-10
    curve_evidence = fillet["curved_edges"] > base["curved_edges"]
    return {
        "base_step": str(base_path.resolve()),
        "fillet_step": str(fillet_path.resolve()),
        "base": base,
        "fillet": fillet,
        "volume_delta": volume_delta,
        "surface_area_delta": surface_delta,
        "geometry_changed": changed,
        "new_curved_edge_evidence": curve_evidence,
        "fillet_verified": bool(changed and curve_evidence),
        "note": "Programmatic evidence must still be paired with preview inspection.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--fillet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify(project_path(args.base), project_path(args.fillet))
    write_json(project_path(args.output), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["fillet_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

