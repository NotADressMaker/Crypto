from __future__ import annotations

from typing import Dict
import pandas as pd


def build_report(
    summary: Dict[str, float | str],
    asset_scores: pd.DataFrame,
    regimes: pd.DataFrame,
    deviation_cost: Dict[str, float],
) -> str:
    latest_regime = regimes.tail(1).to_dict(orient="records")[0] if not regimes.empty else {}
    top_ads = asset_scores.sort_values("ads", ascending=False).head(10)
    lines = [
        "# Nash Equilibrium Lens Report",
        "",
        "## Summary",
        f"- BTC Focal Dominance Index (FDI): {summary['fdi']:.3f}",
        f"- Centrality: {summary['centrality']:.3f}",
        f"- Influence: {summary['influence']:.3f}",
        f"- Benchmark share: {summary['benchmark_share']:.3f}",
        f"- BTC volume share: {summary['volume_share']:.3f}",
        "",
        "## Regime (Latest Window)",
        f"- Regime: {latest_regime.get('regime', 'N/A')}",
        f"- Confidence: {latest_regime.get('confidence', 0.0):.2f}",
        f"- Alt dispersion ratio: {latest_regime.get('alt_dispersion_ratio', 0.0):.3f}",
        "",
        "## Altcoin Dependence (Top 10 ADS)",
    ]
    for _, row in top_ads.iterrows():
        lines.append(f"- {row['asset']}: ADS {row['ads']:.1f} (beta {row['beta']:.2f})")

    lines.extend(
        [
            "",
            "## Deviation Cost (Shock Windows)",
            f"- All-alt drawdown: {deviation_cost['all_alt_drawdown']:.4f}",
            f"- Mixed drawdown: {deviation_cost['mixed_drawdown']:.4f}",
            f"- Deviation cost (all-alt minus mixed): {deviation_cost['deviation_cost']:.4f}",
            "",
            "_Diagnostics only. Not financial advice._",
        ]
    )
    return "\n".join(lines)
