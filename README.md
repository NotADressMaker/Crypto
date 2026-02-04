# crypto-quant-system

A minimal, deterministic crypto quant system scaffold for prop trading research. It includes ingestion, feature store, dataset builder, training, backtesting, signal serving, risk gates, optional execution gating, and a monitoring-friendly layout. Addon modules provide a prop trader dashboard, a quantum-inspired predictor, and a Nash equilibrium analyzer that are fully optional and leave core behavior unchanged when disabled.

## Who this is for
- **Retail traders** who want a reproducible research loop with clear guardrails before experimenting with live execution.
- **Prop traders** who need a deterministic scaffold with optional dashboarding and risk gating.
- **Protocol teams** exploring market microstructure, liquidity modeling, and agent-based simulations without shipping production capital flows.
- **Crypto exchanges** that want a reference pipeline for signals, slippage estimates, and monitoring-friendly outputs.

## Goals
- Deterministic defaults (`seed=42`).
- No claims of profitability or completed training.
- Addon-off mode keeps core behavior unchanged.
- No orders unless `EXECUTION_ON=true`.

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Ingest synthetic data
python scripts/run_ingest.py --config configs/default.yaml

# Build dataset
python scripts/build_dataset.py --config configs/default.yaml

# Train baseline model
python scripts/train.py --config configs/default.yaml

# Backtest
python scripts/backtest.py --config configs/default.yaml

# Serve signals
python scripts/serve_signal.py --config configs/default.yaml

# Run dashboard (addon)
python scripts/run_dashboard.py --config configs/default.yaml
```

## Configuration
- Primary config: `configs/default.yaml`
- Logging: `configs/logging.yaml`
- Environment variables: see `.env.example`

## Addons
Enable addons by setting `addon.enabled: true` in `configs/default.yaml`. When disabled, the core pipeline uses the baseline model only and skips addon logic.

## Liquidity + Slippage Forecaster
The signal API now exposes a lightweight liquidity forecaster. Use `/slippage?size=25000&symbol=BTCUSDT&horizon=5m&venue=binance` to estimate expected slippage and fill probability for a trade size at the latest snapshot.

## Notes
- This repo is a scaffold; data ingestion uses deterministic synthetic data unless real credentials are provided.
- Secrets and API keys must be supplied via environment variables.
- The system does not place orders unless `EXECUTION_ON=true`.
