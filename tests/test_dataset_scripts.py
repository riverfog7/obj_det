from __future__ import annotations

import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class DatasetScriptsTest(unittest.TestCase):
    def test_staged_download_fails_without_archives_and_leaves_no_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_root = Path(tmp) / "source"
            env = {**os.environ, "SOURCE_DATASET_ROOT": str(source_root)}
            env.pop("BDD100K_IMAGES_ARCHIVE", None)
            env.pop("BDD100K_LABELS_ARCHIVE", None)

            result = subprocess.run(
                ["bash", "scripts/download.sh", "bdd100k"],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((source_root / "bdd100k").exists())
            self.assertIn("requires BDD100K_IMAGES_ARCHIVE", result.stderr)

    def test_nuscenes_download_requires_staged_archives(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_root = Path(tmp) / "source"
            env = {**os.environ, "SOURCE_DATASET_ROOT": str(source_root)}
            env.pop("NUSCENES_ARCHIVES", None)

            result = subprocess.run(
                ["bash", "scripts/download.sh", "nuscenes"],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((source_root / "nuscenes").exists())
            self.assertIn("requires NUSCENES_ARCHIVES", result.stderr)

    def test_download_preserves_existing_dataset_without_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_root = Path(tmp) / "source"
            dataset_root = source_root / "hazydet"
            dataset_root.mkdir(parents=True)
            marker = dataset_root / "keep.txt"
            marker.write_text("keep", encoding="utf-8")

            result = subprocess.run(
                ["bash", "scripts/download.sh", "hazydet"],
                cwd=REPO_ROOT,
                env={**os.environ, "SOURCE_DATASET_ROOT": str(source_root)},
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(marker.exists())
            self.assertIn("Refusing to replace", result.stderr)

    def test_convert_runs_only_selected_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_root = tmp_path / "configs"
            output_root = tmp_path / "outputs"
            bin_root = tmp_path / "bin"
            config_root.mkdir()
            bin_root.mkdir()
            (config_root / "alpha.yaml").write_text("key: alpha\n", encoding="utf-8")
            (config_root / "beta.yaml").write_text("key: beta\n", encoding="utf-8")

            log_path = tmp_path / "uv.log"
            fake_uv = bin_root / "uv"
            fake_uv.write_text(
                '#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$UV_LOG"\nmkdir -p "${!#}"\n',
                encoding="utf-8",
            )
            fake_uv.chmod(fake_uv.stat().st_mode | stat.S_IXUSR)
            env = {
                **os.environ,
                "PATH": f"{bin_root}:{os.environ['PATH']}",
                "UV_LOG": str(log_path),
                "DATASET_CONFIG_DIR": str(config_root),
                "DATASET_OUTPUT_ROOT": str(output_root),
            }

            subprocess.run(
                ["bash", "scripts/convert.sh", "alpha"],
                cwd=REPO_ROOT,
                env=env,
                check=True,
            )

            log = log_path.read_text(encoding="utf-8")
            self.assertIn(str(config_root / "alpha.yaml"), log)
            self.assertNotIn(str(config_root / "beta.yaml"), log)
            self.assertTrue((output_root / "alpha").is_dir())
            self.assertFalse((output_root / "beta").exists())

    def test_convert_force_replaces_only_selected_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_root = tmp_path / "configs"
            output_root = tmp_path / "outputs"
            bin_root = tmp_path / "bin"
            config_root.mkdir()
            bin_root.mkdir()
            (config_root / "alpha.yaml").write_text("key: alpha\n", encoding="utf-8")
            (config_root / "beta.yaml").write_text("key: beta\n", encoding="utf-8")
            (output_root / "alpha").mkdir(parents=True)
            (output_root / "beta").mkdir()
            alpha_marker = output_root / "alpha" / "old.txt"
            beta_marker = output_root / "beta" / "keep.txt"
            alpha_marker.write_text("old", encoding="utf-8")
            beta_marker.write_text("keep", encoding="utf-8")

            fake_uv = bin_root / "uv"
            fake_uv.write_text('#!/usr/bin/env bash\nmkdir -p "${!#}"\n', encoding="utf-8")
            fake_uv.chmod(fake_uv.stat().st_mode | stat.S_IXUSR)
            env = {
                **os.environ,
                "PATH": f"{bin_root}:{os.environ['PATH']}",
                "DATASET_CONFIG_DIR": str(config_root),
                "DATASET_OUTPUT_ROOT": str(output_root),
            }

            subprocess.run(
                ["bash", "scripts/convert.sh", "--force", "alpha"],
                cwd=REPO_ROOT,
                env=env,
                check=True,
            )

            self.assertFalse(alpha_marker.exists())
            self.assertTrue(beta_marker.exists())


if __name__ == "__main__":
    unittest.main()
