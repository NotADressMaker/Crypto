from __future__ import annotations

import pandas as pd


def compute_metrics(equity_curve: pd.Series) -> dict:
    returns = equity_curve.pct_change().fillna(0)
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    max_drawdown = ((equity_curve / equity_curve.cummax()) - 1).min()
    return {
        "total_return": float(total_return),
        "max_drawdown": float(max_drawdown),
        "volatility": float(returns.std(ddof=0)),
    }
