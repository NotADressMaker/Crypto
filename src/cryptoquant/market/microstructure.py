from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OrderBookSummary:
    mid_price: float
    spread_bps: float
    bid_depth: float
    ask_depth: float


def simulate_order_book(
    seed: int,
    mid_price: float = 30000.0,
    levels: int = 8,
    tick_size: float = 0.5,
    base_size: float = 1.2,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for level in range(1, levels + 1):
        bid_price = mid_price - level * tick_size
        ask_price = mid_price + level * tick_size
        bid_size = base_size * (1 + rng.random())
        ask_size = base_size * (1 + rng.random())
        rows.append({"side": "bid", "price": bid_price, "size": bid_size, "level": level})
        rows.append({"side": "ask", "price": ask_price, "size": ask_size, "level": level})
    return pd.DataFrame(rows)


def summarize_order_book(order_book: pd.DataFrame, mid_price: float) -> OrderBookSummary:
    bids = order_book[order_book["side"] == "bid"]
    asks = order_book[order_book["side"] == "ask"]
    best_bid = bids["price"].max()
    best_ask = asks["price"].min()
    spread_bps = ((best_ask - best_bid) / mid_price) * 10000
    bid_depth = bids["size"].sum()
    ask_depth = asks["size"].sum()
    return OrderBookSummary(
        mid_price=mid_price,
        spread_bps=float(round(spread_bps, 3)),
        bid_depth=float(round(bid_depth, 3)),
        ask_depth=float(round(ask_depth, 3)),
    )
