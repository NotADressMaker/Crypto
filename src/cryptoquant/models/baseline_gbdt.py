from __future__ import annotations

import logging
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

logger = logging.getLogger(__name__)


def _split_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    features = data.drop(columns=["label"], errors="ignore")
    features = features.select_dtypes(include=["number"])
    labels = data["label"].astype(int) if "label" in data.columns else pd.Series([0] * len(data))
    return features, labels


def train(data: pd.DataFrame, seed: int = 42) -> GradientBoostingClassifier:
    features, labels = _split_features(data)
    if features.empty:
        raise ValueError("Cannot train model: no numeric features found")
    unique_labels = labels.nunique()
    if unique_labels < 2:
        logger.warning("Training data contains only %d class(es); model may not generalize", unique_labels)
    model = GradientBoostingClassifier(random_state=seed)
    model.fit(features, labels)
    return model


def predict(model: GradientBoostingClassifier, data: pd.DataFrame) -> pd.Series:
    features = data.drop(columns=["label"], errors="ignore")
    features = features.select_dtypes(include=["number"])
    proba = model.predict_proba(features)
    if proba.shape[1] < 2:
        return pd.Series(np.zeros(len(data)), index=data.index)
    return pd.Series(proba[:, 1], index=data.index)


def save(model: GradientBoostingClassifier, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(model, handle)


def load(path: Path) -> GradientBoostingClassifier:
    with open(path, "rb") as handle:
        obj = pickle.load(handle)
    if not isinstance(obj, GradientBoostingClassifier):
        raise TypeError(
            f"Expected GradientBoostingClassifier, got {type(obj).__name__}"
        )
    return obj
