from __future__ import annotations

from pathlib import Path
import pandas as pd
from cryptoquant.config import AppConfig
from cryptoquant.data.exchange_client import MarketSnapshot
from cryptoquant.utils.hash import stable_hash


def compute_features(snapshot: MarketSnapshot) -> pd.DataFrame:
    df = snapshot.data.copy()
    df["return"] = df["close"].pct_change().fillna(0)
    df["range"] = (df["high"] - df["low"]) / df["close"].replace(0, 1)
    df["volume_z"] = (df["volume"] - df["volume"].mean()) / df["volume"].std(ddof=0)
    df["ma_5"] = df["close"].rolling(5).mean().fillna(method="bfill")
    return df


def feature_path(config: AppConfig, snapshot: MarketSnapshot) -> Path:
    key = stable_hash(f"{snapshot.symbol}-{snapshot.horizon}-{snapshot.venue}")
    return Path(config.features.feature_store_path) / f"{key}.csv"


def save_features(config: AppConfig, snapshot: MarketSnapshot, features: pd.DataFrame) -> Path:
    path = feature_path(config, snapshot)
    path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(path, index=False)
    return path


def load_features(config: AppConfig, snapshot: MarketSnapshot) -> pd.DataFrame:
    return pd.read_csv(feature_path(config, snapshot))
