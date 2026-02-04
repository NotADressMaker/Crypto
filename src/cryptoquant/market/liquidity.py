from __future__ import annotations

import math
from typing import Any, Dict

import pandas as pd

from cryptoquant.config import AppConfig


def _coerce_positive(value: float, minimum: float = 1e-6) -> float:
    return max(value, minimum)


def forecast_slippage(
    config: AppConfig,
    features: pd.DataFrame,
    size: float,
    venue: str,
    horizon: str,
) -> Dict[str, Any]:
    row = features.iloc[0]
    volume = _coerce_positive(float(row.get("volume", 0.0)), 1.0)
    price_range = _coerce_positive(float(row.get("range", 0.0)))

    volume_scale = math.log1p(volume)
    volatility_bps = min(price_range * 10_000, 250.0)

    base_spread = config.liquidity.base_spread_bps * (1 + volatility_bps / 200.0)
    venue_multiplier = config.liquidity.venue_liquidity.get(venue, 1.0)
    horizon_multiplier = config.liquidity.horizon_multipliers.get(horizon, 1.0)

    size_ratio = _coerce_positive(size, 1.0) / _coerce_positive(config.liquidity.size_reference, 1.0)
    impact = config.liquidity.max_impact_bps * math.sqrt(size_ratio) / volume_scale

    expected_slippage_bps = (base_spread + impact) * venue_multiplier * horizon_multiplier
    liquidity_score = (volume_scale / (1 + volatility_bps / 50.0)) * venue_multiplier * horizon_multiplier
    fill_probability = max(0.05, min(0.99, liquidity_score / (1 + size_ratio)))

    return {
        "expected_slippage_bps": round(expected_slippage_bps, 3),
        "fill_probability": round(fill_probability, 4),
        "liquidity_score": round(liquidity_score, 3),
        "inputs": {
            "size": size,
            "venue": venue,
            "horizon": horizon,
        },
    }
