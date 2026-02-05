from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd
from cryptoquant.addons.nash_equilibrium.metrics import compute_fdi


@dataclass(frozen=True)
class RegimeResult:
    label: str
    confidence: float
    fdi: float
    centrality: float
    influence: float
    benchmark_share: float
    alt_dispersion_ratio: float


def _alt_dispersion_ratio(returns: pd.DataFrame, alt_symbols: List[str], btc_series: pd.Series) -> float:
    alts = [symbol for symbol in alt_symbols if symbol in returns.columns]
    if not alts:
        return 0.0
    dispersion = float(returns[alts].std(axis=1, ddof=0).mean())
    btc_vol = float(btc_series.std(ddof=0))
    return dispersion / btc_vol if btc_vol > 0 else 0.0


def classify_regime(
    fdi: float,
    centrality: float,
    influence: float,
    alt_dispersion_ratio: float,
    thresholds: Dict[str, float],
) -> RegimeResult:
    fdi_high = thresholds.get("fdi_high", 0.6)
    influence_high = thresholds.get("influence_high", 0.5)
    centrality_low = thresholds.get("centrality_low", 0.4)
    alt_dispersion_high = thresholds.get("alt_dispersion_high", 0.6)

    if fdi >= fdi_high and influence >= influence_high:
        confidence = min(1.0, fdi / fdi_high, influence / influence_high)
        label = "BTC_LED"
    elif centrality <= centrality_low and alt_dispersion_ratio >= alt_dispersion_high:
        centrality_score = (centrality_low - centrality) / centrality_low if centrality_low > 0 else 0.0
        confidence = min(1.0, alt_dispersion_ratio / alt_dispersion_high, centrality_score)
        label = "ALT_LED"
    else:
        confidence = 0.5
        label = "MIXED"

    return RegimeResult(
        label=label,
        confidence=confidence,
        fdi=fdi,
        centrality=centrality,
        influence=influence,
        benchmark_share=0.0,
        alt_dispersion_ratio=alt_dispersion_ratio,
    )


def compute_regime_series(
    returns: pd.DataFrame,
    btc_symbol: str,
    alt_symbols: List[str],
    window: int,
    lead_lag: int,
    corr_threshold: float,
    fdi_weights: Dict[str, float],
    thresholds: Dict[str, float],
) -> pd.DataFrame:
    records = []
    if window <= 1:
        window = max(2, int(window))

    for end in range(window, len(returns) + 1):
        window_returns = returns.iloc[end - window : end]
        fdi_components = compute_fdi(
            window_returns,
            btc_symbol=btc_symbol,
            alt_symbols=alt_symbols,
            lead_lag=lead_lag,
            corr_threshold=corr_threshold,
            weights=fdi_weights,
        )
        btc_series = window_returns[btc_symbol]
        alt_dispersion_ratio = _alt_dispersion_ratio(window_returns, alt_symbols, btc_series)
        regime = classify_regime(
            fdi=fdi_components.fdi,
            centrality=fdi_components.centrality,
            influence=fdi_components.influence,
            alt_dispersion_ratio=alt_dispersion_ratio,
            thresholds=thresholds,
        )
        records.append(
            {
                "index": end - 1,
                "fdi": fdi_components.fdi,
                "centrality": fdi_components.centrality,
                "influence": fdi_components.influence,
                "benchmark_share": fdi_components.benchmark_share,
                "alt_dispersion_ratio": alt_dispersion_ratio,
                "regime": regime.label,
                "confidence": regime.confidence,
            }
        )

    return pd.DataFrame(records)
