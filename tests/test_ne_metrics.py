import numpy as np
import pandas as pd
from cryptoquant.addons.nash_equilibrium.metrics import compute_ads, compute_fdi
from cryptoquant.addons.nash_equilibrium.regimes import classify_regime


def _synthetic_returns() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    btc = rng.normal(0, 0.01, 200)
    alt_high = btc * 1.2 + rng.normal(0, 0.002, 200)
    alt_low = rng.normal(0, 0.01, 200)
    return pd.DataFrame(
        {
            "BTCUSDT": btc,
            "ALT1USDT": alt_high,
            "ALT2USDT": alt_low,
        }
    )


def test_fdi_and_ads_ranges() -> None:
    returns = _synthetic_returns()
    fdi = compute_fdi(
        returns,
        btc_symbol="BTCUSDT",
        alt_symbols=["ALT1USDT", "ALT2USDT"],
        lead_lag=2,
        corr_threshold=0.3,
        weights={"centrality": 0.4, "influence": 0.4, "benchmark": 0.2},
    )
    assert 0.0 <= fdi.fdi <= 1.0
    assert 0.0 <= fdi.centrality <= 1.0
    assert 0.0 <= fdi.influence <= 1.0
    assert 0.0 <= fdi.benchmark_share <= 1.0

    ads = compute_ads(
        returns,
        btc_symbol="BTCUSDT",
        alt_symbols=["ALT1USDT", "ALT2USDT"],
        shock_sigma=2.0,
        tail_sigma=2.0,
        reversal_window=2,
        weights={"beta": 0.5, "tail": 0.3, "reversal": 0.2},
    )
    assert ads["ads"].between(0, 100).all()
    assert ads.sort_values("ads", ascending=False)["asset"].iloc[0] == "ALT1USDT"


def test_regime_label_valid() -> None:
    regime = classify_regime(
        fdi=0.7,
        centrality=0.7,
        influence=0.7,
        alt_dispersion_ratio=0.3,
        thresholds={
            "fdi_high": 0.6,
            "influence_high": 0.5,
            "centrality_low": 0.4,
            "alt_dispersion_high": 0.6,
        },
    )
    assert regime.label in {"BTC_LED", "ALT_LED", "MIXED"}
