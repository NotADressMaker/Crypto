"""Tests for edge cases, error handling, and input validation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cryptoquant.backtest.metrics import compute_metrics
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
    load_config,
)
from cryptoquant.data.exchange_client import MarketSnapshot
from cryptoquant.datasets.labels import build_labels
from cryptoquant.features.feature_store import compute_features
from cryptoquant.market.liquidity import forecast_slippage
from cryptoquant.models.baseline_gbdt import load as load_model
from cryptoquant.models.baseline_gbdt import predict, save, train
from cryptoquant.personas import PERSONAS, run_persona
from cryptoquant.risk.gates import apply_risk_gates


def _make_config(tmp_path: Path, execution_on: bool = True) -> AppConfig:
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
        risk=RiskConfig(max_position=1.0, max_drawdown=0.2, execution_on=execution_on),
        serve=ServeConfig(host="0.0.0.0", port=8000),
        addon=AddonConfig(
            enabled=False,
            quantum_predictor=QuantumPredictorConfig(enabled=False, amplitude=0.05),
            nash_analyzer=NashAnalyzerConfig(enabled=False, players=2),
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


# --- compute_metrics edge cases ---


class TestComputeMetrics:
    def test_empty_equity_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            compute_metrics(pd.Series([], dtype=float))

    def test_zero_initial_equity_raises(self) -> None:
        with pytest.raises(ValueError, match="zero"):
            compute_metrics(pd.Series([0.0, 100.0, 200.0]))

    def test_constant_equity(self) -> None:
        metrics = compute_metrics(pd.Series([100.0, 100.0, 100.0]))
        assert metrics["total_return"] == 0.0
        assert metrics["max_drawdown"] == 0.0
        assert metrics["volatility"] == 0.0

    def test_increasing_equity(self) -> None:
        metrics = compute_metrics(pd.Series([100.0, 110.0, 121.0]))
        assert metrics["total_return"] == pytest.approx(0.21, abs=1e-6)
        assert metrics["max_drawdown"] == 0.0

    def test_drawdown_computed(self) -> None:
        metrics = compute_metrics(pd.Series([100.0, 80.0, 90.0]))
        assert metrics["max_drawdown"] < 0.0


# --- build_labels ---


class TestBuildLabels:
    def test_basic_labeling(self) -> None:
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0]})
        labels = build_labels(df, horizon=1)
        assert len(labels) == 5
        assert labels.iloc[-1] == 0  # last row should be filled with 0

    def test_horizon_larger_than_data(self) -> None:
        df = pd.DataFrame({"close": [1.0, 2.0]})
        labels = build_labels(df, horizon=10)
        assert (labels == 0).all()

    def test_flat_prices(self) -> None:
        df = pd.DataFrame({"close": [100.0] * 5})
        labels = build_labels(df, horizon=1)
        assert (labels == 0).all()


# --- compute_features ---


class TestComputeFeatures:
    def test_features_contain_expected_columns(self) -> None:
        data = pd.DataFrame({
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [1000.0, 1100.0],
        })
        snapshot = MarketSnapshot(symbol="TEST", horizon="5m", venue="test", data=data)
        features = compute_features(snapshot)
        for col in ["return", "range", "volume_z", "ma_5"]:
            assert col in features.columns

    def test_constant_volume_no_division_error(self) -> None:
        data = pd.DataFrame({
            "open": [100.0] * 5,
            "high": [101.0] * 5,
            "low": [99.0] * 5,
            "close": [100.0] * 5,
            "volume": [500.0] * 5,
        })
        snapshot = MarketSnapshot(symbol="TEST", horizon="5m", venue="test", data=data)
        features = compute_features(snapshot)
        assert not features["volume_z"].isna().any()
        assert np.isfinite(features["volume_z"]).all()

    def test_zero_close_no_division_error(self) -> None:
        data = pd.DataFrame({
            "open": [0.0, 0.0],
            "high": [1.0, 1.0],
            "low": [0.0, 0.0],
            "close": [0.0, 0.0],
            "volume": [100.0, 100.0],
        })
        snapshot = MarketSnapshot(symbol="TEST", horizon="5m", venue="test", data=data)
        features = compute_features(snapshot)
        assert np.isfinite(features["range"]).all()


# --- forecast_slippage ---


class TestForecastSlippage:
    def test_empty_features_raises(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        empty = pd.DataFrame(columns=["volume", "range"])
        with pytest.raises(ValueError, match="empty"):
            forecast_slippage(config, empty, size=1000, venue="binance", horizon="5m")

    def test_basic_forecast(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        features = pd.DataFrame({"volume": [1000.0], "range": [0.01]})
        result = forecast_slippage(config, features, size=1000, venue="binance", horizon="5m")
        assert "expected_slippage_bps" in result
        assert "fill_probability" in result
        assert result["fill_probability"] >= 0.05
        assert result["fill_probability"] <= 0.99

    def test_unknown_venue_defaults(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        features = pd.DataFrame({"volume": [1000.0], "range": [0.01]})
        result = forecast_slippage(config, features, size=1000, venue="unknown_venue", horizon="5m")
        assert result["expected_slippage_bps"] > 0


# --- train/predict ---


class TestModelTrainPredict:
    def test_train_with_empty_features_raises(self) -> None:
        data = pd.DataFrame({"label": [1, 0]})
        with pytest.raises(ValueError, match="no numeric features"):
            train(data)

    def test_predict_probabilities_in_range(self) -> None:
        data = pd.DataFrame({
            "f1": np.random.randn(50),
            "f2": np.random.randn(50),
            "label": np.random.randint(0, 2, 50),
        })
        model = train(data)
        probs = predict(model, data)
        assert probs.between(0, 1).all()

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        data = pd.DataFrame({
            "f1": np.random.randn(20),
            "label": np.random.randint(0, 2, 20),
        })
        model = train(data)
        path = tmp_path / "model.pkl"
        save(model, path)
        loaded = load_model(path)
        original_probs = predict(model, data)
        loaded_probs = predict(loaded, data)
        pd.testing.assert_series_equal(original_probs, loaded_probs)

    def test_load_invalid_object_raises(self, tmp_path: Path) -> None:
        import pickle

        path = tmp_path / "bad_model.pkl"
        with open(path, "wb") as f:
            pickle.dump({"not": "a model"}, f)
        with pytest.raises(TypeError, match="GradientBoostingClassifier"):
            load_model(path)


# --- risk gates ---


class TestRiskGates:
    def test_empty_signal(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path, execution_on=True)
        result = apply_risk_gates(config, pd.Series([], dtype=int))
        assert len(result) == 0

    def test_capping(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path, execution_on=True)
        signal = pd.Series([0, 1, 5, 10])
        result = apply_risk_gates(config, signal)
        assert result.max() <= config.risk.max_position


# --- config loading ---


class TestConfigLoading:
    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")

    def test_invalid_yaml_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("not_a_mapping: true\n")
        with pytest.raises((ValueError, KeyError)):
            load_config(str(bad))

    def test_valid_config_loads(self) -> None:
        config = load_config("configs/default.yaml")
        assert config.seed > 0
        assert len(config.data.default_symbols) > 0


# --- personas ---


class TestPersonas:
    def test_invalid_persona_raises(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        with pytest.raises(ValueError, match="Valid options"):
            run_persona(config, "nonexistent")

    def test_all_personas_registered(self) -> None:
        expected = {"retail", "prop", "protocol", "exchange"}
        assert set(PERSONAS.keys()) == expected
