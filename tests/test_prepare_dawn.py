from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.prepare_dawn import prepare_dawn


class PrepareDawnTest(unittest.TestCase):
    def test_creates_deterministic_disjoint_splits_and_groups_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images"
            label_dir = root / "labels"
            image_dir.mkdir()
            label_dir.mkdir()

            for index in range(20):
                (image_dir / f"image-{index}.jpg").write_bytes(
                    f"image-{index}".encode()
                )
                (label_dir / f"image-{index}.txt").write_text(
                    "3 0.5 0.5 0.2 0.2\n", encoding="utf-8"
                )

            (image_dir / "duplicate.jpg").write_bytes(b"image-0")
            (label_dir / "duplicate.txt").write_text(
                "3 0.5 0.5 0.2 0.2\n", encoding="utf-8"
            )

            first_counts = prepare_dawn(root)
            first_mapping = self._split_mapping(root)
            second_counts = prepare_dawn(root)
            second_mapping = self._split_mapping(root)

            self.assertEqual(first_counts, second_counts)
            self.assertEqual(first_mapping, second_mapping)
            self.assertEqual(sum(first_counts.values()), 21)
            self.assertTrue(all(first_counts[split] for split in first_counts))
            self.assertEqual(first_mapping["image-0.jpg"], first_mapping["duplicate.jpg"])

    @staticmethod
    def _split_mapping(root: Path) -> dict[str, str]:
        mapping = {}
        for split in ("train", "val", "test"):
            for image_path in (root / "splits" / split / "images").iterdir():
                if image_path.name in mapping:
                    raise AssertionError(f"Image appears in multiple splits: {image_path.name}")
                mapping[image_path.name] = split
        return mapping


if __name__ == "__main__":
    unittest.main()
