from __future__ import annotations

from pathlib import Path
import logging
import pandas as pd
from cryptoquant.config import AppConfig
from cryptoquant.data.exchange_client import fetch_all
from cryptoquant.data.storage import load_snapshot
from cryptoquant.features.feature_store import compute_features, save_features
from cryptoquant.datasets.labels import build_labels

logger = logging.getLogger(__name__)


def build_dataset(config: AppConfig) -> Path:
    snapshots = fetch_all(
        symbols=config.data.default_symbols,
        horizons=config.data.default_horizons,
        venues=config.data.default_venues,
        seed=config.seed,
    )
    dataset_frames = []
    for snapshot in snapshots:
        raw = load_snapshot(config, snapshot)
        snapshot = snapshot.__class__(
            symbol=snapshot.symbol,
            horizon=snapshot.horizon,
            venue=snapshot.venue,
            data=raw,
        )
        features = compute_features(snapshot)
        save_features(config, snapshot, features)
        labels = build_labels(features, config.datasets.label_horizon)
        features["label"] = labels
        features["symbol"] = snapshot.symbol
        features["horizon"] = snapshot.horizon
        features["venue"] = snapshot.venue
        dataset_frames.append(features)
        logger.info("Built features for %s %s %s", snapshot.symbol, snapshot.horizon, snapshot.venue)

    dataset = pd.concat(dataset_frames, ignore_index=True)
    path = Path(config.datasets.dataset_path) / "dataset.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(path, index=False)
    logger.info("Saved dataset to %s", path)
    return path
