from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from obj_det.datasets.converters import convert_config_to_dataset_dict


app = typer.Typer(no_args_is_help=True)


@app.command()
def convert(
    config: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
    out: Annotated[
        Path,
        typer.Argument(file_okay=False),
    ],
    split: Annotated[
        list[str] | None,
        typer.Option("--split", "-s", help="Split to convert. Repeat to convert multiple splits."),
    ] = None,
    hub_id: Annotated[
        str | None,
        typer.Option("--hub-id", help="Optional Hugging Face Hub dataset repo id."),
    ] = None,
    private: Annotated[
        bool,
        typer.Option("--private", help="Create/use a private Hub repo when pushing."),
    ] = False,
    token: Annotated[
        str | None,
        typer.Option("--token", envvar="HF_TOKEN", help="Hugging Face token for push."),
    ] = None,
    config_name: Annotated[
        str,
        typer.Option("--config-name", help="Hub dataset config name."),
    ] = "default",
    max_shard_size: Annotated[
        str | None,
        typer.Option("--max-shard-size", help="Max shard size passed to save/push."),
    ] = None,
) -> None:
    dataset = convert_config_to_dataset_dict(config, splits=split)

    dataset.save_to_disk(out, max_shard_size=max_shard_size)
    typer.echo(f"Saved dataset to {out}")

    if hub_id is not None:
        dataset.push_to_hub(
            hub_id,
            config_name=config_name,
            private=private,
            token=token,
            max_shard_size=max_shard_size,
            embed_external_files=True,
        )
        typer.echo(f"Pushed dataset to {hub_id}")
