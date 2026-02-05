from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
from cryptoquant.addons.nash_equilibrium.metrics import (
    align_returns,
    compute_ads,
    compute_deviation_cost,
    compute_fdi,
)
from cryptoquant.addons.nash_equilibrium.regimes import compute_regime_series
from cryptoquant.addons.nash_equilibrium.report import build_report
from cryptoquant.config import AppConfig


def ne_enabled(config: AppConfig) -> bool:
    return bool(config.addon.enabled and config.addon.ne.enabled)


def _resolve_symbols(config: AppConfig, data: pd.DataFrame) -> Tuple[str, list[str]]:
    btc_symbol = config.addon.ne.btc_symbol
    alt_universe = list(config.addon.ne.alt_universe)
    if not alt_universe:
        alt_universe = [sym for sym in data["symbol"].unique() if sym != btc_symbol]
    return btc_symbol, alt_universe


def _volume_share(data: pd.DataFrame, symbols: list[str], horizon: str, venue: str) -> pd.Series:
    subset = data[data["symbol"].isin(symbols)]
    subset = subset[(subset["horizon"] == horizon) & (subset["venue"] == venue)]
    return subset.groupby("symbol")["volume"].mean()


def run_ne_analysis(config: AppConfig, dataset_path: Path) -> Dict[str, Path]:
    data = pd.read_csv(dataset_path)
    btc_symbol, alt_symbols = _resolve_symbols(config, data)
    symbols = [btc_symbol] + alt_symbols
    horizon = config.addon.ne.horizon
    venue = config.addon.ne.venue
    returns = align_returns(
        data,
        symbols=symbols,
        horizon=horizon,
        venue=venue,
        window=config.addon.ne.window,
    )
    volumes = _volume_share(data, symbols, horizon, venue)
    fdi_components = compute_fdi(
        returns,
        btc_symbol=btc_symbol,
        alt_symbols=alt_symbols,
        lead_lag=config.addon.ne.lead_lag,
        corr_threshold=config.addon.ne.corr_threshold,
        weights=config.addon.ne.fdi_weights,
        volumes=volumes,
    )
    ads = compute_ads(
        returns,
        btc_symbol=btc_symbol,
        alt_symbols=alt_symbols,
        shock_sigma=config.addon.ne.shock_sigma,
        tail_sigma=config.addon.ne.tail_sigma,
        reversal_window=config.addon.ne.reversal_window,
        weights=config.addon.ne.ads_weights,
    )
    regimes = compute_regime_series(
        returns,
        btc_symbol=btc_symbol,
        alt_symbols=alt_symbols,
        window=config.addon.ne.regime_window,
        lead_lag=config.addon.ne.lead_lag,
        corr_threshold=config.addon.ne.corr_threshold,
        fdi_weights=config.addon.ne.fdi_weights,
        thresholds=config.addon.ne.regime_thresholds,
    )
    deviation_cost = compute_deviation_cost(
        returns,
        btc_symbol=btc_symbol,
        alt_symbols=alt_symbols,
        shock_sigma=config.addon.ne.shock_sigma,
        shock_window=config.addon.ne.shock_window,
        btc_mix_weight=config.addon.ne.btc_mix_weight,
    )
    summary = {
        "fdi": fdi_components.fdi,
        "centrality": fdi_components.centrality,
        "influence": fdi_components.influence,
        "benchmark_share": fdi_components.benchmark_share,
        "volume_share": fdi_components.volume_share,
    }
    report = build_report(summary, ads, regimes, deviation_cost)

    output_dir = Path(config.addon.ne.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "ne_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "summary": summary,
                "deviation_cost": deviation_cost,
                "config": {
                    "btc_symbol": btc_symbol,
                    "alt_universe": alt_symbols,
                    "horizon": horizon,
                    "venue": venue,
                },
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    assets_path = output_dir / "ne_asset_scores.csv"
    ads.to_csv(assets_path, index=False)
    regimes_path = output_dir / "ne_regimes.csv"
    regimes.to_csv(regimes_path, index=False)
    report_path = output_dir / "ne_report.md"
    report_path.write_text(report, encoding="utf-8")

    return {
        "summary": summary_path,
        "assets": assets_path,
        "regimes": regimes_path,
        "report": report_path,
    }
