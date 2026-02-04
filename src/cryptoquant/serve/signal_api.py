from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from cryptoquant.config import AppConfig
from cryptoquant.datasets.builder import build_dataset
from cryptoquant.market.liquidity import forecast_slippage
from cryptoquant.models.router import predict_with_router, train_router, model_path
from cryptoquant.risk.gates import apply_risk_gates


def _latest_features(
    config: AppConfig,
    symbol: Optional[str] = None,
    horizon: Optional[str] = None,
    venue: Optional[str] = None,
) -> pd.DataFrame:
    dataset_path = build_dataset(config)
    data = pd.read_csv(dataset_path)
    if symbol:
        data = data[data["symbol"] == symbol]
    if horizon:
        data = data[data["horizon"] == horizon]
    if venue:
        data = data[data["venue"] == venue]
    if data.empty:
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

    @app.get("/slippage")
    def slippage(
        size: float,
        symbol: Optional[str] = None,
        horizon: Optional[str] = None,
        venue: Optional[str] = None,
    ) -> dict[str, Any]:
        features = _latest_features(config, symbol=symbol, horizon=horizon, venue=venue)
        resolved_symbol = features["symbol"].iloc[0]
        resolved_horizon = features["horizon"].iloc[0]
        resolved_venue = features["venue"].iloc[0]
        forecast = forecast_slippage(
            config,
            features,
            size=size,
            venue=resolved_venue,
            horizon=resolved_horizon,
        )
        return {
            "symbol": resolved_symbol,
            "horizon": resolved_horizon,
            "venue": resolved_venue,
            "forecast": forecast,
        }

    return app


def create_dashboard_app(config: AppConfig) -> FastAPI:
    app = FastAPI(title="Prop Trader Dashboard")

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        addon_status = "ENABLED" if config.addon.enabled else "DISABLED"
        quantum_status = "ENABLED" if config.addon.quantum_predictor.enabled else "DISABLED"
        nash_status = "ENABLED" if config.addon.nash_analyzer.enabled else "DISABLED"
        features = _latest_features(config)
        slippage_preview = forecast_slippage(
            config,
            features,
            size=config.liquidity.size_reference,
            venue=features["venue"].iloc[0],
            horizon=features["horizon"].iloc[0],
        )
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
            <h2>Liquidity + Slippage Forecaster</h2>
            <p>
              If you trade size {slippage_preview["inputs"]["size"]:,.0f} now,
              expected slippage is {slippage_preview["expected_slippage_bps"]} bps with
              fill probability {slippage_preview["fill_probability"] * 100:.1f}%.
            </p>
            <p>
              Try the API: <code>/slippage?size=25000&amp;symbol=BTCUSDT&amp;horizon=5m&amp;venue=binance</code>
            </p>
            <p>This dashboard uses placeholder data for visualization.</p>
          </body>
        </html>
        """
        return html

    return app
