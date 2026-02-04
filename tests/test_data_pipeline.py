from cryptoquant.config import load_config
from cryptoquant.data.exchange_client import fetch_all
from cryptoquant.data.ingest import ingest_all
from cryptoquant.data.storage import load_snapshot


def test_ingest_generates_snapshots() -> None:
    config = load_config("configs/default.yaml")
    paths = ingest_all(config)
    assert paths
    snapshots = fetch_all(
        symbols=config.data.default_symbols,
        horizons=config.data.default_horizons,
        venues=config.data.default_venues,
        seed=config.seed,
    )
    sample = load_snapshot(config, snapshots[0])
    assert {"close", "high", "low", "volume"}.issubset(sample.columns)
