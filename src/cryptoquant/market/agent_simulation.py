from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class AgentSimulationResult:
    price_series: list[float]
    inventories: dict[str, float]
    trade_count: int
    notional_volume: float


def run_agent_based_simulation(
    seed: int,
    steps: int = 120,
    num_agents: int = 6,
    start_price: float = 30000.0,
) -> AgentSimulationResult:
    rng = np.random.default_rng(seed)
    inventories = {f"agent_{idx}": 0.0 for idx in range(num_agents)}
    price = start_price
    price_series = []
    trade_count = 0
    notional_volume = 0.0

    for _ in range(steps):
        order_flow = rng.normal(loc=0.0, scale=0.8, size=num_agents)
        imbalance = float(order_flow.sum())
        price *= 1 + (imbalance * 0.00015)
        for idx, qty in enumerate(order_flow):
            inventories[f"agent_{idx}"] += float(qty)
            if qty != 0:
                trade_count += 1
                notional_volume += abs(float(qty)) * price
        price_series.append(float(price))

    return AgentSimulationResult(
        price_series=price_series,
        inventories=inventories,
        trade_count=trade_count,
        notional_volume=float(round(notional_volume, 2)),
    )
