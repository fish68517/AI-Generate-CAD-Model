from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import PROJECT_ROOT, project_path, write_json


def create_oracle(source: Path, output: Path, radius: float) -> dict:
    code = source.read_text(encoding="utf-8").rstrip()
    addition = (
        "\n\n# Local oracle: verify the CadQuery fillet execution path.\n"
        f"fillet_radius = {radius!r}\n"
        "solid = solid.edges().fillet(fillet_radius)\n"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(code + addition, encoding="utf-8")
    report = {
        "source": str(source.resolve()),
        "output": str(output.resolve()),
        "fillet_radius": radius,
        "scope": "oracle executor validation; not a model prediction",
    }
    write_json(output.with_suffix(".json"), report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "data/figure5/oracle.py"
    )
    parser.add_argument("--radius", type=float, default=0.01)
    args = parser.parse_args()
    report = create_oracle(project_path(args.input), project_path(args.output), args.radius)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

