from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List
import numpy as np
import pandas as pd
from cryptoquant.utils.hash import stable_hash


@dataclass(frozen=True)
class MarketSnapshot:
    symbol: str
    horizon: str
    venue: str
    data: pd.DataFrame


def _seed_for(symbol: str, horizon: str, venue: str, seed: int) -> int:
    digest = stable_hash(f"{seed}-{symbol}-{horizon}-{venue}")
    return int(digest[:8], 16)


def _horizon_to_freq(horizon: str) -> str:
    mapping = {"5m": "5min", "1h": "1h", "1d": "1d"}
    return mapping.get(horizon, "1min")


def _generate_synthetic_candles(
    symbol: str,
    horizon: str,
    venue: str,
    seed: int,
    periods: int = 320,
) -> pd.DataFrame:
    rng = np.random.default_rng(_seed_for(symbol, horizon, venue, seed))
    freq = _horizon_to_freq(horizon)
    timestamps = pd.date_range(end=pd.Timestamp.utcnow(), periods=periods, freq=freq)
    base_price = 10000 + (int(stable_hash(symbol)[:6], 16) % 5000)
    returns = rng.normal(loc=0, scale=0.0015, size=periods)
    close = base_price * np.exp(np.cumsum(returns))
    open_price = np.concatenate([[close[0]], close[:-1]])
    spread = rng.uniform(0.0005, 0.0025, size=periods)
    high = close * (1 + spread)
    low = close * (1 - spread)
    volume = rng.lognormal(mean=8.0, sigma=0.35, size=periods)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "symbol": symbol,
            "horizon": horizon,
            "venue": venue,
        }
    )


def fetch_all(
    symbols: Iterable[str],
    horizons: Iterable[str],
    venues: Iterable[str],
    seed: int,
) -> List[MarketSnapshot]:
    snapshots = []
    for symbol in symbols:
        for horizon in horizons:
            for venue in venues:
                data = _generate_synthetic_candles(symbol, horizon, venue, seed)
                snapshots.append(MarketSnapshot(symbol=symbol, horizon=horizon, venue=venue, data=data))
    return snapshots
