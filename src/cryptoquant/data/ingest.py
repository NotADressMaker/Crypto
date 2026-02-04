from __future__ import annotations

from pathlib import Path
from typing import List
from cryptoquant.config import AppConfig
from cryptoquant.data.exchange_client import fetch_all
from cryptoquant.data.storage import save_snapshot


def ingest_all(config: AppConfig) -> List[Path]:
    snapshots = fetch_all(
        symbols=config.data.default_symbols,
        horizons=config.data.default_horizons,
        venues=config.data.default_venues,
        seed=config.seed,
    )
    paths = []
    for snapshot in snapshots:
        paths.append(save_snapshot(config, snapshot))
    return paths
