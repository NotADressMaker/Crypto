import pandas as pd
from cryptoquant.risk.gates import apply_risk_gates
from cryptoquant.config import AppConfig, RiskConfig, DataConfig, FeatureConfig, DatasetConfig, ModelConfig, BacktestConfig, ServeConfig, AddonConfig, QuantumPredictorConfig, NashAnalyzerConfig, MonitoringConfig, LiquidityConfig


def _ne_config(output_path: str) -> NashAnalyzerConfig:
    return NashAnalyzerConfig(
        enabled=False,
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
        output_path=output_path,
    )


def make_config(execution_on: bool) -> AppConfig:
    return AppConfig(
        repo_name="test",
        system_goal="test",
        base_model="UNKNOWN",
        seed=42,
        data=DataConfig(
            data_sources=["EXCHANGE_APIS"],
            default_symbols=["BTCUSDT"],
            default_horizons=["5m"],
            default_venues=["binance"],
            data_path="./data",
        ),
        features=FeatureConfig(feature_store_path="./data/features"),
        datasets=DatasetConfig(dataset_path="./data/datasets", label_horizon=1),
        models=ModelConfig(model_path="./models", baseline_model="gbdt", calibration="none"),
        backtest=BacktestConfig(initial_capital=1000, fee_bps=2, slippage_bps=1),
        risk=RiskConfig(max_position=1.0, max_drawdown=0.2, execution_on=execution_on),
        serve=ServeConfig(host="0.0.0.0", port=8000),
        addon=AddonConfig(
            enabled=False,
            quantum_predictor=QuantumPredictorConfig(enabled=True, amplitude=0.5),
            ne=_ne_config(output_path="./outputs/ne"),
        ),
        liquidity=LiquidityConfig(
            base_spread_bps=1.5,
            max_impact_bps=25,
            size_reference=100000,
            venue_liquidity={"binance": 1.0},
            horizon_multipliers={"5m": 1.05},
        ),
        monitoring=MonitoringConfig(log_path="./logs", logging_config="configs/logging.yaml"),
    )


def test_risk_gate_blocks_when_execution_off() -> None:
    config = make_config(execution_on=False)
    signal = pd.Series([1, 1, 0])
    gated = apply_risk_gates(config, signal)
    assert gated.sum() == 0


def test_risk_gate_allows_when_execution_on() -> None:
    config = make_config(execution_on=True)
    signal = pd.Series([1, 1, 0])
    gated = apply_risk_gates(config, signal)
    assert gated.sum() == 2
