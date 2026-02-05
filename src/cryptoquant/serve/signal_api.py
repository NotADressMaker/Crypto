from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from threading import Lock
from typing import Any, Optional
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from cryptoquant.config import AppConfig
from cryptoquant.addons.nash_equilibrium import ne_enabled, run_ne_analysis
from cryptoquant.datasets.builder import build_dataset
from cryptoquant.market.liquidity import forecast_slippage
from cryptoquant.models.router import predict_with_router, train_router, model_path
from cryptoquant.risk.gates import apply_risk_gates

MAX_TRADE_SIZE = 1_000_000_000.0


@dataclass
class _DatasetCache:
    path: Optional[Path] = None
    mtime: float = 0.0
    data: Optional[pd.DataFrame] = None
    lock: Lock = field(default_factory=Lock)

    def load(self, config: AppConfig) -> pd.DataFrame:
        dataset_path = Path(build_dataset(config))
        with self.lock:
            mtime = dataset_path.stat().st_mtime
            if self.path == dataset_path and self.data is not None and self.mtime == mtime:
                return self.data
            data = pd.read_csv(dataset_path)
            self.path = dataset_path
            self.mtime = mtime
            self.data = data
            return data


def _latest_features(
    config: AppConfig,
    cache: _DatasetCache,
    symbol: Optional[str] = None,
    horizon: Optional[str] = None,
    venue: Optional[str] = None,
) -> pd.DataFrame:
    data = cache.load(config)
    if symbol:
        data = data[data["symbol"] == symbol]
    if horizon:
        data = data[data["horizon"] == horizon]
    if venue:
        data = data[data["venue"] == venue]
    if data.empty:
        data = cache.load(config)
    result = data.tail(1)
    if result.empty:
        raise HTTPException(status_code=404, detail="No feature data available")
    return result


def _ensure_model(config: AppConfig) -> None:
    path = model_path(config)
    if not Path(path).exists():
        train_router(config)


def create_app(config: AppConfig) -> FastAPI:
    app = FastAPI(title="CryptoQuant Signal API")
    cache = _DatasetCache()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/signal")
    def signal(
        symbol: Optional[str] = None,
        horizon: Optional[str] = None,
        venue: Optional[str] = None,
    ) -> dict[str, Any]:
        _ensure_model(config)
        features = _latest_features(config, cache, symbol=symbol, horizon=horizon, venue=venue)
        probs = predict_with_router(config, features)
        signal_value = (probs > 0.5).astype(int)
        gated_signal = apply_risk_gates(config, signal_value)

        action = "HOLD"
        if config.risk.execution_on and int(gated_signal.iloc[0]) == 1:
            action = "BUY"

        return {
            "symbol": str(features["symbol"].iloc[0]),
            "horizon": str(features["horizon"].iloc[0]),
            "venue": str(features["venue"].iloc[0]),
            "probability": float(probs.iloc[0]),
            "action": action,
            "execution_on": config.risk.execution_on,
        }

    @app.get("/slippage")
    def slippage(
        size: float = Query(..., gt=0, le=MAX_TRADE_SIZE),
        symbol: Optional[str] = None,
        horizon: Optional[str] = None,
        venue: Optional[str] = None,
    ) -> dict[str, Any]:
        features = _latest_features(config, cache, symbol=symbol, horizon=horizon, venue=venue)
        resolved_symbol = str(features["symbol"].iloc[0])
        resolved_horizon = str(features["horizon"].iloc[0])
        resolved_venue = str(features["venue"].iloc[0])
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
    cache = _DatasetCache()

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        addon_status = "ENABLED" if config.addon.enabled else "DISABLED"
        quantum_status = "ENABLED" if config.addon.quantum_predictor.enabled else "DISABLED"
        nash_status = "ENABLED" if config.addon.ne.enabled else "DISABLED"
        features = _latest_features(config, cache)
        slippage_preview = forecast_slippage(
            config,
            features,
            size=config.liquidity.size_reference,
            venue=str(features["venue"].iloc[0]),
            horizon=str(features["horizon"].iloc[0]),
        )
        ne_panel = ""
        if ne_enabled(config):
            artifacts = run_ne_analysis(config, Path(build_dataset(config)))
            with open(artifacts["summary"], "r", encoding="utf-8") as handle:
                summary_payload = json.load(handle)
            summary = summary_payload["summary"]
            deviation_cost = summary_payload["deviation_cost"]
            asset_scores = pd.read_csv(artifacts["assets"]).sort_values("ads", ascending=False).head(10)
            regimes = pd.read_csv(artifacts["regimes"])
            latest_regime = regimes.tail(1).to_dict(orient="records")[0] if not regimes.empty else {}
            ads_rows = "\n".join(
                f"<li>{row['asset']}: {row['ads']:.1f}</li>" for _, row in asset_scores.iterrows()
            )
            ne_panel = f"""
            <h2>Nash Equilibrium Lens</h2>
            <ul>
              <li>FDI: {summary['fdi']:.3f}</li>
              <li>Centrality: {summary['centrality']:.3f}</li>
              <li>Influence: {summary['influence']:.3f}</li>
              <li>Benchmark share: {summary['benchmark_share']:.3f}</li>
              <li>Regime: {latest_regime.get('regime', 'N/A')} ({latest_regime.get('confidence', 0.0):.2f})</li>
            </ul>
            <h3>Top BTC-Dependent Alts (ADS)</h3>
            <ol>
              {ads_rows}
            </ol>
            <h3>Deviation Cost</h3>
            <ul>
              <li>All-alt drawdown: {deviation_cost['all_alt_drawdown']:.4f}</li>
              <li>Mixed drawdown: {deviation_cost['mixed_drawdown']:.4f}</li>
              <li>Deviation cost: {deviation_cost['deviation_cost']:.4f}</li>
            </ul>
            <p>Detailed artifacts written to {config.addon.ne.output_path}.</p>
            """
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
            {ne_panel}
            <p>This dashboard uses placeholder data for visualization.</p>
          </body>
        </html>
        """
        return html

    return app
