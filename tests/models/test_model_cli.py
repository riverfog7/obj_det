from __future__ import annotations

import unittest

from typer.testing import CliRunner

from obj_det.cli.main import app as root_app
from obj_det.models.cli import app as models_app


class ModelCliTest(unittest.TestCase):
    def test_root_cli_exposes_models_group(self):
        result = CliRunner().invoke(root_app, ["--help"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("models", result.output)

    def test_models_cli_exposes_core_commands(self):
        result = CliRunner().invoke(models_app, ["--help"])

        self.assertEqual(result.exit_code, 0)
        for command in ["train", "evaluate", "optimize", "final", "plan"]:
            self.assertIn(command, result.output)


if __name__ == "__main__":
    unittest.main()
