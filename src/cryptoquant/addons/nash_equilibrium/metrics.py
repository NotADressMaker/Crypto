from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FDIComponents:
    centrality: float
    influence: float
    benchmark_share: float
    fdi: float
    volume_share: float


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(value, 0.0) for value in weights.values())
    if total <= 0:
        raise ValueError("Weights must sum to a positive value")
    return {key: max(value, 0.0) / total for key, value in weights.items()}


def align_returns(
    data: pd.DataFrame,
    symbols: Iterable[str],
    horizon: str,
    venue: str,
    window: int,
) -> pd.DataFrame:
    subset = data[data["symbol"].isin(symbols)]
    subset = subset[(subset["horizon"] == horizon) & (subset["venue"] == venue)]
    if subset.empty:
        raise ValueError("No data available for requested symbols/horizon/venue")
    subset = subset.sort_values(["symbol", "timestamp"])
    subset["seq"] = subset.groupby(["symbol", "horizon", "venue"]).cumcount()
    pivot = subset.pivot_table(index="seq", columns="symbol", values="return")
    pivot = pivot.dropna(how="any")
    if pivot.empty:
        raise ValueError("Aligned returns are empty after pivot")
    if window > 0:
        pivot = pivot.tail(window)
    return pivot


def compute_fdi(
    returns: pd.DataFrame,
    btc_symbol: str,
    alt_symbols: List[str],
    lead_lag: int,
    corr_threshold: float,
    weights: Dict[str, float],
    volumes: pd.Series | None = None,
) -> FDIComponents:
    if btc_symbol not in returns.columns:
        raise ValueError("BTC symbol not found in returns")
    alts = [symbol for symbol in alt_symbols if symbol in returns.columns]
    if not alts:
        raise ValueError("No alt symbols available for FDI computation")

    corr = returns.corr()
    btc_corr = corr.loc[btc_symbol, alts].abs()
    centrality = float(btc_corr.mean())

    btc_series = returns[btc_symbol]
    influence_scores = []
    for alt in alts:
        alt_series = returns[alt]
        max_lead_corr = 0.0
        for lag in range(1, lead_lag + 1):
            lead_corr = btc_series.shift(lag).corr(alt_series)
            if pd.isna(lead_corr):
                continue
            max_lead_corr = max(max_lead_corr, abs(float(lead_corr)))
        influence_scores.append(max_lead_corr)
    influence = float(np.mean(influence_scores)) if influence_scores else 0.0

    benchmark_share = float((btc_corr >= corr_threshold).mean())

    normalized = _normalize_weights(weights)
    fdi = (
        normalized.get("centrality", 0.0) * centrality
        + normalized.get("influence", 0.0) * influence
        + normalized.get("benchmark", 0.0) * benchmark_share
    )
    volume_share = 0.0
    if volumes is not None and not volumes.empty:
        volume_share = float(volumes.get(btc_symbol, 0.0) / volumes.sum())
    return FDIComponents(
        centrality=centrality,
        influence=influence,
        benchmark_share=benchmark_share,
        fdi=fdi,
        volume_share=volume_share,
    )


def compute_ads(
    returns: pd.DataFrame,
    btc_symbol: str,
    alt_symbols: List[str],
    shock_sigma: float,
    tail_sigma: float,
    reversal_window: int,
    weights: Dict[str, float],
) -> pd.DataFrame:
    if btc_symbol not in returns.columns:
        raise ValueError("BTC symbol not found in returns")
    normalized = _normalize_weights(weights)
    btc = returns[btc_symbol]
    btc_var = float(btc.var(ddof=0))
    btc_sigma = float(btc.std(ddof=0))
    btc_shock = btc < (-shock_sigma * btc_sigma)
    results = []
    for alt in alt_symbols:
        if alt not in returns.columns:
            continue
        alt_series = returns[alt]
        alt_sigma = float(alt_series.std(ddof=0))
        beta = float(np.cov(alt_series, btc, ddof=0)[0, 1] / btc_var) if btc_var > 0 else 0.0
        beta_score = min(1.0, abs(beta) / 2.0)
        if btc_shock.any() and alt_sigma > 0:
            tail_dep = float((alt_series[btc_shock] < (-tail_sigma * alt_sigma)).mean())
        else:
            tail_dep = 0.0
        reversal_penalty = 0.0
        if reversal_window > 0 and btc_shock.any() and alt_sigma > 0:
            penalties = []
            shock_indices = np.where(btc_shock.values)[0]
            for idx in shock_indices:
                window_slice = alt_series.iloc[idx + 1 : idx + 1 + reversal_window]
                if window_slice.empty:
                    continue
                window_mean = float(window_slice.mean())
                penalties.append(min(1.0, max(0.0, -window_mean / alt_sigma)))
            if penalties:
                reversal_penalty = float(np.mean(penalties))

        score = (
            normalized.get("beta", 0.0) * beta_score
            + normalized.get("tail", 0.0) * tail_dep
            + normalized.get("reversal", 0.0) * reversal_penalty
        )
        results.append(
            {
                "asset": alt,
                "beta": beta,
                "beta_score": beta_score,
                "tail_dependence": tail_dep,
                "reversal_penalty": reversal_penalty,
                "ads": float(score * 100.0),
            }
        )
    return pd.DataFrame(results)


def compute_deviation_cost(
    returns: pd.DataFrame,
    btc_symbol: str,
    alt_symbols: List[str],
    shock_sigma: float,
    shock_window: int,
    btc_mix_weight: float,
) -> Dict[str, float]:
    if btc_symbol not in returns.columns:
        raise ValueError("BTC symbol not found in returns")
    alts = [symbol for symbol in alt_symbols if symbol in returns.columns]
    if not alts:
        return {"all_alt_drawdown": 0.0, "mixed_drawdown": 0.0, "deviation_cost": 0.0}

    btc = returns[btc_symbol]
    btc_sigma = float(btc.std(ddof=0))
    shock_mask = btc < (-shock_sigma * btc_sigma)
    if not shock_mask.any():
        return {"all_alt_drawdown": 0.0, "mixed_drawdown": 0.0, "deviation_cost": 0.0}

    alt_portfolio = returns[alts].mean(axis=1)
    mixed_portfolio = btc_mix_weight * btc + (1.0 - btc_mix_weight) * alt_portfolio

    def _avg_drawdown(portfolio: pd.Series) -> float:
        drawdowns = []
        shock_indices = np.where(shock_mask.values)[0]
        for idx in shock_indices:
            window_slice = portfolio.iloc[idx : idx + shock_window]
            if window_slice.empty:
                continue
            cumulative = (1 + window_slice).cumprod()
            drawdown = float(cumulative.min() - 1.0)
            drawdowns.append(drawdown)
        return float(np.mean(drawdowns)) if drawdowns else 0.0

    all_alt_dd = _avg_drawdown(alt_portfolio)
    mixed_dd = _avg_drawdown(mixed_portfolio)
    deviation_cost = all_alt_dd - mixed_dd
    return {
        "all_alt_drawdown": all_alt_dd,
        "mixed_drawdown": mixed_dd,
        "deviation_cost": deviation_cost,
    }
