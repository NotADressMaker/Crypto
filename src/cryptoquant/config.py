from __future__ import annotations

from dataclasses import dataclass
from typing import List
import yaml


@dataclass(frozen=True)
class DataConfig:
    data_sources: List[str]
    default_symbols: List[str]
    default_horizons: List[str]
    default_venues: List[str]
    data_path: str


@dataclass(frozen=True)
class FeatureConfig:
    feature_store_path: str


@dataclass(frozen=True)
class DatasetConfig:
    dataset_path: str
    label_horizon: int


@dataclass(frozen=True)
class ModelConfig:
    model_path: str
    baseline_model: str
    calibration: str


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float
    fee_bps: float
    slippage_bps: float


@dataclass(frozen=True)
class RiskConfig:
    max_position: float
    max_drawdown: float
    execution_on: bool


@dataclass(frozen=True)
class ServeConfig:
    host: str
    port: int


@dataclass(frozen=True)
class QuantumPredictorConfig:
    enabled: bool
    amplitude: float


@dataclass(frozen=True)
class NashAnalyzerConfig:
    enabled: bool
    players: int


@dataclass(frozen=True)
class AddonConfig:
    enabled: bool
    quantum_predictor: QuantumPredictorConfig
    nash_analyzer: NashAnalyzerConfig


@dataclass(frozen=True)
class MonitoringConfig:
    log_path: str
    logging_config: str


@dataclass(frozen=True)
class AppConfig:
    repo_name: str
    system_goal: str
    base_model: str
    seed: int
    data: DataConfig
    features: FeatureConfig
    datasets: DatasetConfig
    models: ModelConfig
    backtest: BacktestConfig
    risk: RiskConfig
    serve: ServeConfig
    addon: AddonConfig
    monitoring: MonitoringConfig


def load_config(path: str) -> AppConfig:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    return AppConfig(
        repo_name=raw["repo_name"],
        system_goal=raw["system_goal"],
        base_model=raw["base_model"],
        seed=int(raw["seed"]),
        data=DataConfig(**raw["data"]),
        features=FeatureConfig(**raw["features"]),
        datasets=DatasetConfig(**raw["datasets"]),
        models=ModelConfig(**raw["models"]),
        backtest=BacktestConfig(**raw["backtest"]),
        risk=RiskConfig(**raw["risk"]),
        serve=ServeConfig(**raw["serve"]),
        addon=AddonConfig(
            enabled=raw["addon"]["enabled"],
            quantum_predictor=QuantumPredictorConfig(**raw["addon"]["quantum_predictor"]),
            nash_analyzer=NashAnalyzerConfig(**raw["addon"]["nash_analyzer"]),
        ),
        monitoring=MonitoringConfig(**raw["monitoring"]),
    )
