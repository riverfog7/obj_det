from __future__ import annotations

import pathlib
import unittest


class BoundaryTest(unittest.TestCase):
    def test_model_layer_does_not_import_raw_sources(self):
        root = pathlib.Path("src/obj_det/models")
        forbidden = ("obj_det.datasets.sources", "BaseSourceDataset", "source_from_config")
        offenders = []
        for path in root.rglob("*.py"):
            text = path.read_text()
            for item in forbidden:
                if item in text:
                    offenders.append(f"{path}:{item}")
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
