# Nash Equilibrium in Cryptocurrency Markets: Bitcoin as a Strategic Focal Point

This module maps the paper’s conceptual ideas into deterministic, measurable signals that can plug into the CryptoQuant scaffold. The implementation is diagnostic-only and does **not** claim predictive power.

## Conceptual Mapping (Paper → Metrics)

### BTC as focal point (“dominant strategy anchor”)
**Proxy:** BTC Focal Dominance Index (FDI).

FDI combines three bounded components (0..1):
1. **Centrality** – BTC’s average absolute correlation to alt returns (graph centrality proxy).
2. **Influence** – lead–lag cross-correlation where BTC moves precede alt moves.
3. **Benchmark share** – fraction of alts whose correlation with BTC exceeds a threshold.

These capture the paper’s idea that BTC functions as a coordination anchor in the market.

### Coordination / benchmarking
**Proxy:** lead–lag influence and benchmark share from the FDI.

When BTC shocks precede alt moves and a large share of alts track BTC, the system flags stronger coordination.

### Costly deviation
**Proxy:** deviation-cost diagnostics.

We compare drawdowns during BTC shock windows between:
- An **all-alt** equal-weight portfolio.
- A **BTC-mixed** portfolio (configurable BTC weight).

The difference in drawdown is a cost-of-deviation proxy: larger deltas indicate higher penalty for deviating from BTC exposure during shocks.

## Metric Definitions (Math-lite, reproducible)

Let \( r_{btc,t} \) be BTC returns and \( r_{i,t} \) be returns for alt \( i \).

### Focal Dominance Index (FDI)
- **Centrality:** \( C = \frac{1}{N}\sum_i |\text{corr}(r_{btc}, r_i)| \)
- **Influence:** \( I = \frac{1}{N}\sum_i \max_{\ell\in[1,L]} |\text{corr}(r_{btc,t-\ell}, r_{i,t})| \)
- **Benchmark share:** \( B = \frac{1}{N}\sum_i \mathbb{1}[|\text{corr}(r_{btc}, r_i)| \ge \tau] \)

Weighted sum (weights sum to 1):

```
FDI = w_c * C + w_i * I + w_b * B
```

### Altcoin Dependence Score (ADS)
Per asset \( i \):
- **Beta component:** \( \beta_i = \frac{\text{cov}(r_i, r_{btc})}{\text{var}(r_{btc})} \)
- **Tail dependence:** co-crash frequency when \( r_{btc} < -k \sigma_{btc} \) and \( r_i < -k \sigma_i \)
- **Reversal penalty:** average underperformance of \( r_i \) after BTC shock windows

ADS is a weighted sum (0..100):

```
ADS_i = 100 * (w_beta * beta_score + w_tail * tail_dependence + w_rev * reversal_penalty)
```

### Regime classifier
Rules (no ML):
- **BTC-led:** FDI high and influence high
- **ALT-led:** BTC centrality low and alt dispersion high
- **MIXED:** otherwise

### Deviation cost proxy
Compute average drawdown during BTC shock windows for:
- all-alt portfolio
- BTC-mixed portfolio

```
DeviationCost = DD_all_alt - DD_mixed
```

## Limitations
- This is a conceptual-to-empirical mapping, not a direct proof of Nash equilibrium.
- Correlation- and lead–lag-based proxies are sensitive to windowing choices.
- Outputs are diagnostics only and should not be interpreted as investment advice.
