from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator

from .base import ModelSchema


LoggingBackend = Literal["none", "local", "wandb"]
WandbMode = Literal["online", "offline", "disabled"]


class LocalLoggingConfig(ModelSchema):
    path: Path | None = None


class WandbLoggingConfig(ModelSchema):
    project: str = "obj-det"
    entity: str | None = None
    group: str | None = None
    name: str | None = None
    mode: WandbMode = "online"
    tags: list[str] = Field(default_factory=list)


class LoggingConfig(ModelSchema):
    backends: list[LoggingBackend] = Field(default_factory=lambda: ["none"])
    local: LocalLoggingConfig = Field(default_factory=LocalLoggingConfig)
    wandb: WandbLoggingConfig = Field(default_factory=WandbLoggingConfig)

    @field_validator("backends")
    @classmethod
    def validate_backends(cls, value: list[LoggingBackend]) -> list[LoggingBackend]:
        if not value:
            raise ValueError("logging backends cannot be empty")
        deduped = list(dict.fromkeys(value))
        if "none" in deduped and len(deduped) > 1:
            raise ValueError("logging backend 'none' cannot be combined with other backends")
        return deduped
