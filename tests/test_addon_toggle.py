import pandas as pd
from cryptoquant.config import AppConfig, RiskConfig, DataConfig, FeatureConfig, DatasetConfig, ModelConfig, BacktestConfig, ServeConfig, AddonConfig, QuantumPredictorConfig, NashAnalyzerConfig, MonitoringConfig
from cryptoquant.models.baseline_gbdt import train, save
from cryptoquant.models.router import predict_with_router


def make_base_config(model_path: str, addon_enabled: bool) -> AppConfig:
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
        models=ModelConfig(model_path=model_path, baseline_model="gbdt", calibration="none"),
        backtest=BacktestConfig(initial_capital=1000, fee_bps=2, slippage_bps=1),
        risk=RiskConfig(max_position=1.0, max_drawdown=0.2, execution_on=True),
        serve=ServeConfig(host="0.0.0.0", port=8000),
        addon=AddonConfig(
            enabled=addon_enabled,
            quantum_predictor=QuantumPredictorConfig(enabled=True, amplitude=0.5),
            nash_analyzer=NashAnalyzerConfig(enabled=True, players=2),
        ),
        monitoring=MonitoringConfig(log_path="./logs", logging_config="configs/logging.yaml"),
    )


def test_addon_toggle_changes_probabilities(tmp_path) -> None:
    data = pd.DataFrame(
        {
            "return": [0.01, -0.02, 0.03, 0.0],
            "range": [0.02, 0.01, 0.03, 0.02],
            "volume_z": [0.1, -0.1, 0.2, 0.0],
            "ma_5": [100, 101, 102, 103],
            "label": [1, 0, 1, 0],
        }
    )
    model = train(data)
    path = tmp_path / "baseline_gbdt.pkl"
    save(model, path)

    base_config = make_base_config(str(tmp_path), addon_enabled=False)
    addon_config = make_base_config(str(tmp_path), addon_enabled=True)

    base_probs = predict_with_router(base_config, data)
    addon_probs = predict_with_router(addon_config, data)

    assert not base_probs.equals(addon_probs)
