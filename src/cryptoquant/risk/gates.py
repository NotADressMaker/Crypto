from __future__ import annotations

import pandas as pd
from cryptoquant.config import AppConfig


def apply_risk_gates(config: AppConfig, signal: pd.Series) -> pd.Series:
    capped = signal.clip(upper=int(config.risk.max_position))
    if not config.risk.execution_on:
        return capped * 0
    return capped
