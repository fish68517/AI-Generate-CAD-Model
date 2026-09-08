from __future__ import annotations

import argparse
import re
from pathlib import Path


FENCE_PATTERN = re.compile(
    r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL
)


def extract_code(text: str) -> str:
    matches = FENCE_PATTERN.findall(text)
    if matches:
        candidates = [candidate.strip() for candidate in matches if candidate.strip()]
        if not candidates:
            raise ValueError("模型回答中的代码围栏为空")
        code = max(candidates, key=len)
    else:
        code = text.strip()
    if not code:
        raise ValueError("模型回答为空")
    if "import cadquery" not in code and "from cadquery" not in code:
        raise ValueError("回答中没有 CadQuery import")
    return code.rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    code = extract_code(args.input.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(code, encoding="utf-8")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

