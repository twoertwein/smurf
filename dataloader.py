#!/usr/bin/env python3
from pathlib import Path

from python_tools import caching
from python_tools.ml.data_loader import DataLoader


def get_partitions(dimension: str, batch_size: int) -> dict[int, dict[str, DataLoader]]:
    """Return pickled dataloader to be portable."""
    assert batch_size == 256
    dataset, dimension = dimension.split("/", 1)
    return caching.read_pickle(Path(f"cache/{dataset}.pickle"))[dimension]
