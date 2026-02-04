from __future__ import annotations

from pathlib import Path
import pandas as pd
from cryptoquant.config import AppConfig
from cryptoquant.data.exchange_client import MarketSnapshot
from cryptoquant.utils.hash import stable_hash


def snapshot_path(config: AppConfig, snapshot: MarketSnapshot) -> Path:
    key = stable_hash(f"{snapshot.symbol}-{snapshot.horizon}-{snapshot.venue}")
    return Path(config.data.data_path) / "snapshots" / f"{key}.csv"


def save_snapshot(config: AppConfig, snapshot: MarketSnapshot) -> Path:
    path = snapshot_path(config, snapshot)
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.data.to_csv(path, index=False)
    return path


def load_snapshot(config: AppConfig, snapshot: MarketSnapshot) -> pd.DataFrame:
    return pd.read_csv(snapshot_path(config, snapshot))
