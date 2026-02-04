from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
from cryptoquant.backtest.simulator import run_backtest
from cryptoquant.config import AppConfig
from cryptoquant.data.ingest import ingest_all
from cryptoquant.datasets.builder import build_dataset
from cryptoquant.market.agent_simulation import run_agent_based_simulation
from cryptoquant.market.microstructure import simulate_order_book, summarize_order_book
from cryptoquant.market.liquidity import forecast_slippage
from cryptoquant.models.router import predict_with_router, train_router
from cryptoquant.monitoring.snapshot import write_snapshot
from cryptoquant.utils.time import utc_now


PERSONAS = {
    "retail": "Reproducible research loop with guardrails before live execution.",
    "prop": "Deterministic scaffold with optional dashboarding and risk gating.",
    "protocol": "Microstructure, liquidity modeling, and agent-based simulations.",
    "exchange": "Reference pipeline for signals, slippage estimates, and monitoring outputs.",
}


def _guardrails(config: AppConfig) -> Dict[str, Any]:
    notes: List[str] = []
    if config.risk.execution_on:
        notes.append("Execution is enabled; retail/protocol workflows expect execution_off.")
    return {
        "execution_on": config.risk.execution_on,
        "max_position": config.risk.max_position,
        "max_drawdown": config.risk.max_drawdown,
        "notes": notes,
    }


def _load_latest_features(config: AppConfig) -> pd.DataFrame:
    dataset_path = build_dataset(config)
    data = pd.read_csv(dataset_path)
    return data.tail(1)


def run_persona(config: AppConfig, persona: str) -> Dict[str, Any]:
    if persona not in PERSONAS:
        raise ValueError(f"Unknown persona: {persona}")
    if persona == "retail":
        return _run_retail(config)
    if persona == "prop":
        return _run_prop(config)
    if persona == "protocol":
        return _run_protocol(config)
    if persona == "exchange":
        return _run_exchange(config)
    raise ValueError(f"Unhandled persona: {persona}")


def _base_report(config: AppConfig, persona: str) -> Dict[str, Any]:
    return {
        "persona": persona,
        "description": PERSONAS[persona],
        "timestamp": utc_now().isoformat(),
        "seed": config.seed,
        "guardrails": _guardrails(config),
    }


def _run_retail(config: AppConfig) -> Dict[str, Any]:
    ingest_paths = ingest_all(config)
    dataset_path = build_dataset(config)
    model_path = train_router(config)
    metrics = run_backtest(config)
    report = _base_report(config, "retail")
    report.update(
        {
            "ingest_paths": [str(path) for path in ingest_paths],
            "dataset_path": str(dataset_path),
            "model_path": str(model_path),
            "backtest_metrics": metrics,
            "next_step": "Review metrics and keep execution_off before live experiments.",
        }
    )
    return report


def _run_prop(config: AppConfig) -> Dict[str, Any]:
    dataset_path = build_dataset(config)
    model_path = train_router(config)
    metrics = run_backtest(config)
    report = _base_report(config, "prop")
    report.update(
        {
            "dataset_path": str(dataset_path),
            "model_path": str(model_path),
            "backtest_metrics": metrics,
            "dashboard_ready": True,
            "addon_status": {
                "addon_enabled": config.addon.enabled,
                "quantum_predictor": config.addon.quantum_predictor.enabled,
                "nash_analyzer": config.addon.nash_analyzer.enabled,
            },
            "next_step": "Launch dashboard or wire risk gates into execution service.",
        }
    )
    return report


def _run_protocol(config: AppConfig) -> Dict[str, Any]:
    features = _load_latest_features(config)
    order_book = simulate_order_book(seed=config.seed, mid_price=float(features["close"].iloc[0]))
    micro_summary = summarize_order_book(order_book, mid_price=float(features["close"].iloc[0]))
    slippage = forecast_slippage(
        config,
        features,
        size=config.liquidity.size_reference,
        venue=features["venue"].iloc[0],
        horizon=features["horizon"].iloc[0],
    )
    agent_result = run_agent_based_simulation(seed=config.seed, start_price=float(features["close"].iloc[0]))
    report = _base_report(config, "protocol")
    report.update(
        {
            "microstructure": {
                "order_book_levels": order_book.head(6).to_dict(orient="records"),
                "summary": micro_summary.__dict__,
            },
            "liquidity_model": slippage,
            "agent_simulation": {
                "price_start": agent_result.price_series[0],
                "price_end": agent_result.price_series[-1],
                "trade_count": agent_result.trade_count,
                "notional_volume": agent_result.notional_volume,
            },
            "next_step": "Extend simulations with custom agent behaviors before deployment.",
        }
    )
    return report


def _run_exchange(config: AppConfig) -> Dict[str, Any]:
    dataset_path = build_dataset(config)
    model_path = train_router(config)
    data = pd.read_csv(dataset_path)
    latest = data.tail(1)
    probs = predict_with_router(config, latest)
    slippage = forecast_slippage(
        config,
        latest,
        size=config.liquidity.size_reference,
        venue=latest["venue"].iloc[0],
        horizon=latest["horizon"].iloc[0],
    )
    payload = {
        "symbol": latest["symbol"].iloc[0],
        "horizon": latest["horizon"].iloc[0],
        "venue": latest["venue"].iloc[0],
        "probability": float(probs.iloc[0]),
        "slippage": slippage,
    }
    snapshot_path = write_snapshot(config, payload=payload, name="exchange-snapshot")
    report = _base_report(config, "exchange")
    report.update(
        {
            "dataset_path": str(dataset_path),
            "model_path": str(model_path),
            "signal_payload": payload,
            "monitoring_snapshot": str(snapshot_path),
            "next_step": "Stream payload to monitoring or risk gateway dashboards.",
        }
    )
    return report


def write_persona_report(config: AppConfig, persona: str, report: Dict[str, Any]) -> Path:
    report_path = Path("reports") / f"{persona}-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    return report_path
