from __future__ import annotations

from pathlib import Path
import pandas as pd
from cryptoquant.config import AppConfig
from cryptoquant.datasets.builder import build_dataset
from cryptoquant.models.baseline_gbdt import load as load_baseline
from cryptoquant.models.baseline_gbdt import predict as predict_baseline
from cryptoquant.models.baseline_gbdt import save as save_baseline
from cryptoquant.models.baseline_gbdt import train as train_baseline

def model_path(config: AppConfig) -> Path:
    return Path(config.models.model_path) / f"baseline_{config.models.baseline_model}.pkl"


def _apply_addons(config: AppConfig, data: pd.DataFrame, probs: pd.Series) -> pd.Series:
    if not config.addon.enabled:
        return probs
    adjusted = probs.copy()
    if config.addon.quantum_predictor.enabled and "return" in data.columns:
        quantum_signal = data["return"].fillna(0).clip(-0.05, 0.05)
        adjusted = adjusted * (1 + config.addon.quantum_predictor.amplitude * quantum_signal)
    if config.addon.nash_analyzer.enabled and "range" in data.columns:
        imbalance = data["range"].fillna(0)
        adjusted = adjusted + 0.05 * (imbalance - imbalance.mean())
    return adjusted.clip(0, 1)


def train_router(config: AppConfig) -> Path:
    dataset_path = build_dataset(config)
    data = pd.read_csv(dataset_path)
    path = model_path(config)
    model = train_baseline(data, seed=config.seed)
    save_baseline(model, path)
    return path


def predict_with_router(config: AppConfig, data: pd.DataFrame) -> pd.Series:
    path = model_path(config)
    model = load_baseline(path)
    probs = predict_baseline(model, data)
    return _apply_addons(config, data, probs)
