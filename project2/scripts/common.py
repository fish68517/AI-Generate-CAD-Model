from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_COMMIT = "7f40b4827b029731ef97313a116d62519e313960"
OFFICIAL_RAW_ROOT = (
    "https://raw.githubusercontent.com/anniedoris/CAD-Coder/"
    f"{OFFICIAL_COMMIT}"
)
OFFICIAL_PROMPT = (
    "Generate the CadQuery code needed to create the CAD for the provided image. "
    "Just the code, no other words."
)
FILLET_PROMPT = (
    "Add fillets to all the edges. Recall that you can do this using "
    "solid.edges().fillet(fillet_radius), where I would like fillet_radius "
    "to be 0.01."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def relative_to_project(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()

