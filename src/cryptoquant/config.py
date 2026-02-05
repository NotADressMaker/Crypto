from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
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
class LiquidityConfig:
    base_spread_bps: float
    max_impact_bps: float
    size_reference: float
    venue_liquidity: Dict[str, float]
    horizon_multipliers: Dict[str, float]


@dataclass(frozen=True)
class QuantumPredictorConfig:
    enabled: bool
    amplitude: float


@dataclass(frozen=True)
class NashAnalyzerConfig:
    enabled: bool
    horizon: str
    venue: str
    btc_symbol: str
    alt_universe: List[str]
    window: int
    regime_window: int
    lead_lag: int
    corr_threshold: float
    shock_sigma: float
    tail_sigma: float
    reversal_window: int
    fdi_weights: Dict[str, float]
    ads_weights: Dict[str, float]
    regime_thresholds: Dict[str, float]
    btc_mix_weight: float
    shock_window: int
    output_path: str


@dataclass(frozen=True)
class AddonConfig:
    enabled: bool
    quantum_predictor: QuantumPredictorConfig
    ne: NashAnalyzerConfig


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
    liquidity: LiquidityConfig
    addon: AddonConfig
    monitoring: MonitoringConfig


_REQUIRED_TOP_KEYS = [
    "repo_name", "system_goal", "base_model", "seed",
    "data", "features", "datasets", "models", "backtest",
    "risk", "serve", "liquidity", "addon", "monitoring",
]


def load_config(path: str) -> AppConfig:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping, got {type(raw).__name__}")
    missing = [k for k in _REQUIRED_TOP_KEYS if k not in raw]
    if missing:
        raise ValueError(f"Config file {path} missing required keys: {', '.join(missing)}")

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
        liquidity=LiquidityConfig(**raw["liquidity"]),
        addon=AddonConfig(
            enabled=raw["addon"]["enabled"],
            quantum_predictor=QuantumPredictorConfig(**raw["addon"]["quantum_predictor"]),
            ne=NashAnalyzerConfig(**raw["addon"]["ne"]),
        ),
        monitoring=MonitoringConfig(**raw["monitoring"]),
    )
