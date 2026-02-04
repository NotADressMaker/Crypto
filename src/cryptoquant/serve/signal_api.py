from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from cryptoquant.config import AppConfig
from cryptoquant.datasets.builder import build_dataset
from cryptoquant.models.router import predict_with_router, train_router, model_path
from cryptoquant.risk.gates import apply_risk_gates


def _latest_features(config: AppConfig) -> pd.DataFrame:
    dataset_path = build_dataset(config)
    data = pd.read_csv(dataset_path)
    return data.tail(1)


def _ensure_model(config: AppConfig) -> None:
    path = model_path(config)
    if not Path(path).exists():
        train_router(config)


def create_app(config: AppConfig) -> FastAPI:
    app = FastAPI(title="CryptoQuant Signal API")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/signal")
    def signal() -> dict[str, Any]:
        _ensure_model(config)
        features = _latest_features(config)
        probs = predict_with_router(config, features)
        signal_value = (probs > 0.5).astype(int)
        gated_signal = apply_risk_gates(config, signal_value)

        action = "HOLD"
        if config.risk.execution_on and int(gated_signal.iloc[0]) == 1:
            action = "BUY"

        return {
            "symbol": features["symbol"].iloc[0],
            "horizon": features["horizon"].iloc[0],
            "venue": features["venue"].iloc[0],
            "probability": float(probs.iloc[0]),
            "action": action,
            "execution_on": config.risk.execution_on,
        }

    return app


def create_dashboard_app(config: AppConfig) -> FastAPI:
    app = FastAPI(title="Prop Trader Dashboard")

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        addon_status = "ENABLED" if config.addon.enabled else "DISABLED"
        quantum_status = "ENABLED" if config.addon.quantum_predictor.enabled else "DISABLED"
        nash_status = "ENABLED" if config.addon.nash_analyzer.enabled else "DISABLED"
        html = f"""
        <html>
          <head><title>Prop Trader Dashboard</title></head>
          <body>
            <h1>Prop Trader Dashboard</h1>
            <p>Addon Status: {addon_status}</p>
            <h2>Core Status</h2>
            <ul>
              <li>Symbols: {', '.join(config.data.default_symbols)}</li>
              <li>Horizons: {', '.join(config.data.default_horizons)}</li>
              <li>Venues: {', '.join(config.data.default_venues)}</li>
            </ul>
            <h2>Addon Modules</h2>
            <ul>
              <li>Quantum-Inspired Predictor: {quantum_status}</li>
              <li>Nash Equilibrium Analyzer: {nash_status}</li>
            </ul>
            <p>This dashboard uses placeholder data for visualization.</p>
          </body>
        </html>
        """
        return html

    return app
