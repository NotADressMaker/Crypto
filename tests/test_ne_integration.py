from pathlib import Path
import pandas as pd
from cryptoquant.backtest.simulator import run_backtest
from cryptoquant.config import (
    AddonConfig,
    AppConfig,
    BacktestConfig,
    DataConfig,
    DatasetConfig,
    FeatureConfig,
    LiquidityConfig,
    ModelConfig,
    MonitoringConfig,
    NashAnalyzerConfig,
    QuantumPredictorConfig,
    RiskConfig,
    ServeConfig,
)
from cryptoquant.data.ingest import ingest_all
from cryptoquant.models.router import train_router


def _ne_config(tmp_path: Path, enabled: bool) -> NashAnalyzerConfig:
    return NashAnalyzerConfig(
        enabled=enabled,
        horizon="5m",
        venue="binance",
        btc_symbol="BTCUSDT",
        alt_universe=["ETHUSDT"],
        window=120,
        regime_window=60,
        lead_lag=2,
        corr_threshold=0.3,
        shock_sigma=2.0,
        tail_sigma=2.0,
        reversal_window=2,
        fdi_weights={"centrality": 0.4, "influence": 0.4, "benchmark": 0.2},
        ads_weights={"beta": 0.5, "tail": 0.3, "reversal": 0.2},
        regime_thresholds={
            "fdi_high": 0.6,
            "influence_high": 0.5,
            "centrality_low": 0.4,
            "alt_dispersion_high": 0.6,
        },
        btc_mix_weight=0.5,
        shock_window=4,
        output_path=str(tmp_path / "outputs" / "ne"),
    )


def _make_config(tmp_path: Path, addon_enabled: bool, ne_enabled: bool) -> AppConfig:
    return AppConfig(
        repo_name="test",
        system_goal="test",
        base_model="UNKNOWN",
        seed=42,
        data=DataConfig(
            data_sources=["EXCHANGE_APIS"],
            default_symbols=["BTCUSDT", "ETHUSDT"],
            default_horizons=["5m"],
            default_venues=["binance"],
            data_path=str(tmp_path / "data"),
        ),
        features=FeatureConfig(feature_store_path=str(tmp_path / "features")),
        datasets=DatasetConfig(dataset_path=str(tmp_path / "datasets"), label_horizon=1),
        models=ModelConfig(
            model_path=str(tmp_path / "models"),
            baseline_model="gbdt",
            calibration="none",
        ),
        backtest=BacktestConfig(initial_capital=1000, fee_bps=2, slippage_bps=1),
        risk=RiskConfig(max_position=1.0, max_drawdown=0.2, execution_on=False),
        serve=ServeConfig(host="0.0.0.0", port=8000),
        addon=AddonConfig(
            enabled=addon_enabled,
            quantum_predictor=QuantumPredictorConfig(enabled=False, amplitude=0.5),
            ne=_ne_config(tmp_path, enabled=ne_enabled),
        ),
        liquidity=LiquidityConfig(
            base_spread_bps=1.5,
            max_impact_bps=25,
            size_reference=100000,
            venue_liquidity={"binance": 1.0},
            horizon_multipliers={"5m": 1.05},
        ),
        monitoring=MonitoringConfig(
            log_path=str(tmp_path / "logs"),
            logging_config="configs/logging.yaml",
        ),
    )


def test_ne_addon_artifacts(tmp_path: Path) -> None:
    disabled_config = _make_config(tmp_path / "disabled", addon_enabled=False, ne_enabled=False)
    ingest_all(disabled_config)
    train_router(disabled_config)
    disabled_metrics = run_backtest(disabled_config)

    ne_output = Path(disabled_config.addon.ne.output_path)
    assert not ne_output.exists()

    enabled_config = _make_config(tmp_path / "enabled", addon_enabled=True, ne_enabled=True)
    ingest_all(enabled_config)
    train_router(enabled_config)
    enabled_metrics = run_backtest(enabled_config)

    pd.testing.assert_series_equal(
        pd.Series(disabled_metrics).sort_index(),
        pd.Series(enabled_metrics).sort_index(),
    )

    output_dir = Path(enabled_config.addon.ne.output_path)
    assert (output_dir / "ne_summary.json").exists()
    assert (output_dir / "ne_asset_scores.csv").exists()
    assert (output_dir / "ne_regimes.csv").exists()
    assert (output_dir / "ne_report.md").exists()
