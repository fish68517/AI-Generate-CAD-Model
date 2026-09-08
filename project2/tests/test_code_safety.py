from code_safety import validate_code


def test_accepts_ground_truth_style_code() -> None:
    code = """import cadquery as cq
wp = cq.Workplane('XY')
solid = wp.box(1.0, 2.0, 3.0)
"""
    result = validate_code(code)
    assert result["accepted"] is True
    assert result["issues"] == []


def test_rejects_os_and_open() -> None:
    code = """import os
import cadquery as cq
open('bad.txt', 'w').write('x')
solid = cq.Workplane('XY').box(1.0, 1.0, 1.0)
"""
    result = validate_code(code)
    assert result["accepted"] is False
    codes = {issue["code"] for issue in result["issues"]}
    assert "unsafe_import" in codes
    assert "unsafe_call" in codes


def test_requires_solid_assignment() -> None:
    result = validate_code("import cadquery as cq\npart = cq.Workplane().box(1, 1, 1)\n")
    assert result["accepted"] is False
    assert any(issue["code"] == "missing_solid" for issue in result["issues"])

