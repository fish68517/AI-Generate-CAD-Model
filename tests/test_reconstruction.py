import unittest

import numpy as np

from scripts.compare_meshes import center_and_normalize_pc, chamfer_distance, normalize_pc
from scripts.infer_batch_gpu import build_summary, extract_json
from scripts.prepare_lora_subset import split_rows
from scripts.reconstruct_json import reconstruct


def rectangle_part(operation="NewBodyFeatureOperation"):
    return {
        "coordinate_system": {"Euler Angles": [0.0, 0.0, 0.0], "Translation Vector": [0.0, 0.0, 0.0]},
        "sketch": {
            "face_1": {
                "loop_1": {
                    "line_1": {"Start Point": [0.0, 0.0], "End Point": [1.0, 0.0]},
                    "line_2": {"Start Point": [1.0, 0.0], "End Point": [1.0, 0.5]},
                    "line_3": {"Start Point": [1.0, 0.5], "End Point": [0.0, 0.5]},
                    "line_4": {"Start Point": [0.0, 0.5], "End Point": [0.0, 0.0]},
                }
            }
        },
        "extrusion": {
            "extrude_depth_towards_normal": 0.2,
            "extrude_depth_opposite_normal": 0.0,
            "operation": operation,
            "sketch_scale": 1.0,
        },
    }


class ReconstructionTests(unittest.TestCase):
    def test_rectangle_volume(self):
        mesh, _ = reconstruct({"parts": {"part_1": rectangle_part()}})
        self.assertTrue(mesh.is_watertight)
        self.assertAlmostEqual(mesh.volume, 0.1, places=6)

    def test_circle_with_hole(self):
        part = rectangle_part()
        part["sketch"]["face_1"]["loop_2"] = {
            "circle_1": {"Center": [0.5, 0.25], "Radius": 0.1}
        }
        mesh, _ = reconstruct({"parts": {"part_1": part}})
        self.assertTrue(mesh.is_watertight)
        self.assertLess(mesh.volume, 0.1)
        self.assertEqual(mesh.euler_number, 0)


class LocalPreparationTests(unittest.TestCase):
    def test_official_normalization_and_centered_diagnostic(self):
        points = np.asarray([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        shifted = points + 10.0
        self.assertGreater(chamfer_distance(normalize_pc(points), normalize_pc(shifted)), 0.0)
        self.assertAlmostEqual(
            chamfer_distance(center_and_normalize_pc(points), center_and_normalize_pc(shifted)),
            0.0,
        )

    def test_batch_json_extraction_and_summary(self):
        self.assertEqual(extract_json('prefix {"parts": {}} suffix'), {"parts": {}})
        summary = build_summary(
            [
                {"generation_success": True, "json_parse_success": True, "generation_seconds": 1.0},
                {"generation_success": True, "json_parse_success": False, "generation_seconds": 3.0},
            ],
            total_input=2,
        )
        self.assertEqual(summary["attempted"], 2)
        self.assertEqual(summary["json_parse_rate"], 0.5)
        self.assertEqual(summary["mean_generation_seconds"], 2.0)

    def test_deterministic_balanced_subset(self):
        rows = [
            {"uid": f"{category}/{index}", "category": category}
            for index in range(4)
            for category in ("a", "b", "c", "d", "e")
        ]
        first = split_rows(rows, 10, 5, 5, seed=42)
        second = split_rows(rows, 10, 5, 5, seed=42)
        self.assertEqual(first, second)
        self.assertEqual({row["category"] for row in first["validation"]}, {"a", "b", "c", "d", "e"})


if __name__ == "__main__":
    unittest.main()
