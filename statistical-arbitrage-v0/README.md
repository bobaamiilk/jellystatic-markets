# statarb — Statistical Arbitrage Engine (APAC focus)

A from-first-principles statistical-arbitrage system: screen a universe for
cointegration, estimate a time-varying hedge ratio, build volatility-adjusted
mean-reversion signals, and backtest with the discipline that stops a stat-arb
researcher from fooling themselves. Validated end-to-end against synthetic data
with **known cointegration ground truth**.

## What it does

```
universe ─▶ correlation clustering ─▶ within-cluster Engle–Granger tests
         ─▶ Benjamini–Hochberg FDR ─▶ Kalman hedge ratio ─▶ vol-adjusted z-score
         ─▶ walk-forward backtest (t+1 execution, costs) ─▶ break monitoring
```

## Method

**Cointegration (Engle–Granger).** Two I(1) prices are cointegrated if a linear
combination is I(0). Step 1: OLS hedge ratio `y = α + βx + e` (β is
*super-consistent* under cointegration — converges at rate `T`, not `√T`).
Step 2: ADF-test the residual with cointegration-appropriate critical values.
Half-life of mean reversion from an Ornstein–Uhlenbeck fit,
`half-life = −ln2 / λ`, is tracked as a live health metric.

**Cointegrated ≠ tradeable.** The A–H share premium is the canonical APAC trap:
statistically cointegrated (same firm, dual-listed) yet untradeable because
capital controls block the cross-boundary trade. `is_tradeable()` encodes that
filter alongside half-life bounds.

**Multi-asset screening.** `n` assets ⇒ `n(n−1)/2` pairs; at `n=200` ≈ 20,000
tests, ≈ 1,000 false "discoveries" at 5%. Defended by **economic clustering**
(`d_ij = √(2(1−ρ_ij))` on returns) so only within-cluster pairs are tested, plus
**Benjamini–Hochberg** FDR control. Johansen trace test handles >2-asset systems.

**Signals.** Time-varying hedge ratio via a **Kalman filter** (β as a latent
random walk) or rolling OLS. Spread → z-score; a **vol-adjusted** z-score (EWMA
dispersion) fixes the hidden "z=2 is a fixed probability" assumption under fat
tails. Entry `|z|>2`, exit `|z|<0.5`, hard stop `|z|>4`, max-holding cap, and a
post-stop cooldown so a structurally-broken spread isn't immediately re-entered.

**Backtesting.** Walk-forward: fit on in-sample, **freeze**, trade out-of-sample,
roll. **Signal at t, execution at t+1** (`PnL_t = position_{t−1}·Δspread_t`).
Explicit cost model (commission + ½-spread + impact). In-sample Sharpe is
reported only as a sanity check; the **OOS** Sharpe / max-drawdown / Calmar is
the verdict. A unit test asserts no look-ahead (appending future data cannot
change past signals).

**Failure analysis.** Rolling-ADF spread health + CUSUM structural-break
detection. Because detection is inherently lagged, risk is bounded by
pre-committed hard stops rather than relying on prediction.

## Layout

```
src/statarb/
  cointegration/  engle_granger.py   screening.py     (EG, ADF, half-life, Johansen, FDR)
  signals/        kalman.py          zscore.py        (time-varying β, signals, break health)
  backtest/       walkforward.py                      (t+1 walk-forward, costs, metrics)
  utils/          synthetic_data.py                   (known-truth pairs & universe)
  examples/       run_pipeline.py                     (full demo + plot)
tests/            test_statarb.py                     (17 tests)
```

## Run

```bash
pip install -e .
pytest -q                                   # 17 passing
python -m statarb.examples.run_pipeline     # screen + backtest + reports/statarb_backtest.png
```

## Validation highlights
- Detects known cointegrated pairs; rejects independent random walks.
- Recovers the known hedge ratio (±0.15) and half-life (right order of magnitude).
- Screen recovers 18/18 true within-sector pairs with 0 false discoveries; FDR
  monotone in level.
- Kalman tracks a β drifting 1.0→2.0; rolling OLS is causal.
- Costs strictly reduce returns; OOS execution proven look-ahead-free.
- Example run: in-sample Sharpe ≈ 1.9 collapses to OOS Sharpe ≈ 1.0 — the honest
  degradation every real backtest shows.

*Synthetic data only; no investment advice. Built as a learning/portfolio project.*
