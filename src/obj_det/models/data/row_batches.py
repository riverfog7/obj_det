from __future__ import annotations

from typing import Any, Iterator


def iter_hf_row_batches(ds, batch_size: int) -> Iterator[list[dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    for start in range(0, len(ds), batch_size):
        end = min(start + batch_size, len(ds))
        yield [ds[idx] for idx in range(start, end)]
