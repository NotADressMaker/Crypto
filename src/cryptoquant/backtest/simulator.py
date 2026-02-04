from __future__ import annotations

import logging
import pandas as pd
from cryptoquant.backtest.metrics import compute_metrics
from cryptoquant.config import AppConfig
from cryptoquant.datasets.builder import build_dataset
from cryptoquant.models.router import predict_with_router
from cryptoquant.risk.gates import apply_risk_gates

logger = logging.getLogger(__name__)


def run_backtest(config: AppConfig) -> dict:
    dataset_path = build_dataset(config)
    data = pd.read_csv(dataset_path)
    probs = predict_with_router(config, data)
    signals = (probs > 0.5).astype(int)

    capped_signals = apply_risk_gates(config, signals)
    returns = data["return"]
    equity = (1 + returns * capped_signals).cumprod() * config.backtest.initial_capital

    metrics = compute_metrics(equity)
    logger.info("Backtest metrics: %s", metrics)
    return metrics
