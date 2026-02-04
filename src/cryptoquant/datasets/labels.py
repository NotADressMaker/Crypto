from __future__ import annotations

import pandas as pd


def build_labels(df: pd.DataFrame, horizon: int) -> pd.Series:
    future = df["close"].shift(-horizon)
    label = (future > df["close"]).astype(int)
    return label.fillna(0)
