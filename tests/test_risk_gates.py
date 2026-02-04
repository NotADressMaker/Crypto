import pandas as pd
from cryptoquant.risk.gates import apply_risk_gates
from cryptoquant.config import AppConfig, RiskConfig, DataConfig, FeatureConfig, DatasetConfig, ModelConfig, BacktestConfig, ServeConfig, AddonConfig, QuantumPredictorConfig, NashAnalyzerConfig, MonitoringConfig


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
            nash_analyzer=NashAnalyzerConfig(enabled=True, players=2),
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
