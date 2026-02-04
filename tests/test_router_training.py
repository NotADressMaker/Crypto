import pandas as pd
from cryptoquant.config import load_config
from cryptoquant.datasets.builder import build_dataset
from cryptoquant.models.router import predict_with_router, train_router


def test_router_training_and_prediction() -> None:
    config = load_config("configs/default.yaml")
    train_router(config)
    dataset_path = build_dataset(config)
    data = pd.read_csv(dataset_path).head(5)
    probs = predict_with_router(config, data)
    assert probs.between(0, 1).all()
