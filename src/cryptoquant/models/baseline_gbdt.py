from __future__ import annotations

from pathlib import Path
import pickle
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier


def _split_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    features = data.drop(columns=["label"], errors="ignore")
    features = features.select_dtypes(include=["number"])
    labels = data["label"].astype(int) if "label" in data.columns else pd.Series([0] * len(data))
    return features, labels


def train(data: pd.DataFrame, seed: int = 42) -> GradientBoostingClassifier:
    features, labels = _split_features(data)
    model = GradientBoostingClassifier(random_state=seed)
    model.fit(features, labels)
    return model


def predict(model: GradientBoostingClassifier, data: pd.DataFrame) -> pd.Series:
    features = data.drop(columns=["label"], errors="ignore")
    features = features.select_dtypes(include=["number"])
    probs = model.predict_proba(features)[:, 1]
    return pd.Series(probs, index=data.index)


def save(model: GradientBoostingClassifier, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(model, handle)


def load(path: Path) -> GradientBoostingClassifier:
    with open(path, "rb") as handle:
        return pickle.load(handle)
