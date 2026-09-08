from prepare_official_subset import code_features, select_cases
from process_figure4_gpu_results import validate_row


def row(question_id: int, code: str) -> dict:
    return {
        "question_id": question_id,
        "image": f"{question_id:08d}_0.png",
        "text": "prompt",
        "ground_truth": "import cadquery as cq\n" + code + "\nsolid=solid0\n",
    }


def test_features_and_selection_are_deterministic() -> None:
    rows = [
        row(1, "solid0=cq.Workplane().rect(1,1).extrude(1)"),
        row(2, "solid0=cq.Workplane().circle(1).extrude(1)"),
        row(
            3,
            "solid0=cq.Workplane().box(1,1,1)\nsolid1=cq.Workplane().circle(1).extrude(1)\nsolid0=solid0.cut(solid1)",
        ),
        row(
            4,
            "a=cq.Workplane().circle(1).extrude(1)\nb=cq.Workplane().circle(1).extrude(1)\nc=cq.Workplane().circle(1).extrude(1)",
        ),
        row(
            5,
            "a=cq.Workplane().circle(1).extrude(1)\n"
            "b=cq.Workplane().circle(1).extrude(1)\n"
            "solid0=a.cut(b)\n# make this candidate deliberately longer",
        ),
    ]
    first = select_cases(rows)
    second = select_cases(list(reversed(rows)))
    assert [item[1]["question_id"] for item in first] == [
        item[1]["question_id"] for item in second
    ]
    assert len({item[1]["question_id"] for item in first}) == 5
    assert code_features(rows[1])["circle_count"] == 1


def test_gpu_result_schema_validation() -> None:
    validate_row(
        {
            "case_id": "f4_01",
            "input_domain": "rendered",
            "response": "import cadquery as cq\nsolid = cq.Workplane().box(1,1,1)",
        },
        {"f4_01"},
    )
