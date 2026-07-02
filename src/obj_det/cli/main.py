from __future__ import annotations

import typer

from obj_det.datasets.cli import app as datasets_app
from obj_det.models.cli import app as models_app


app = typer.Typer(no_args_is_help=True)
app.add_typer(datasets_app, name="datasets")
app.add_typer(models_app, name="models")


if __name__ == "__main__":
    app()
