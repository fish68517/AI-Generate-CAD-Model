import pytest

from extract_cadquery_code import extract_code


def test_extracts_python_fence() -> None:
    response = "Before\n```python\nimport cadquery as cq\nsolid = cq.Workplane().box(1, 2, 3)\n```"
    code = extract_code(response)
    assert code.startswith("import cadquery as cq")
    assert "solid =" in code


def test_accepts_raw_code() -> None:
    code = extract_code("import cadquery as cq\nsolid = cq.Workplane().box(1, 2, 3)")
    assert code.endswith("\n")


def test_rejects_non_cadquery_text() -> None:
    with pytest.raises(ValueError, match="CadQuery"):
        extract_code("This is only an explanation")

