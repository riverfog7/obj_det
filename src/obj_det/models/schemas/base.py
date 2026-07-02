from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelSchema(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )
