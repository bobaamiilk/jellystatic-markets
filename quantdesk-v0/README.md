# quantdesk

Integration layer that wires three independently-validated APAC quant packages
into a single regime-scaled statistical-arbitrage pipeline.

```
prices
  → cointegration screen   (statarb: cluster + FDR)
  → tradeability filter    (statarb: half-life, capital controls)
  → walk-forward backtest  (statarb: t+1 execution, realistic costs)
  → vol forecast           (volforecast: GARCH on equal-weight market return)
  → regime scalar          (volforecast: vol-percentile de-grossing)
  → book_return[t] = scalar[t-1] × mean(pair_returns[t])
  → HTML report + Parquet  (quantdesk.report)
```

The three upstream packages are treated as **frozen libraries** — quantdesk
calls their public APIs, never duplicates their logic.

---

## Package map

| Package | Role in quantdesk |
|---|---|
| `statarb` | `screen_universe` (FDR screen), `is_tradeable`, `backtest_pair` (walk-forward OOS) |
| `volforecast` | `fit_garch` (market vol), `vol_percentile_regime` (exposure scalar), `detect_structural_break_cusum` |
| `volsurface` | Available for vega/skew context; not yet wired into the default pipeline |

---

## Quick start

```bash
# Install all four packages (editable)
cd jellystatic-markets/statistical-arbitrage-v0  && pip install -e .
cd ../volatility-forecast-v0                     && pip install -e .
cd ../options-pricing-v0                         && pip install -e .
cd ../quantdesk-v0                               && pip install -e ".[dev]"

# Run on synthetic data — no CSVs needed, completes in < 10s
python -m quantdesk run --synthetic

# Run on real data
python -m quantdesk run --config config.yaml --data-dir /path/to/csvs/
```

Outputs land in `out/`:
- `out/report.html` — self-contained HTML (base64 figures, no external assets)
- `out/results.parquet` — signal/return/regime timeseries for reproducibility

### Reference numbers (deterministic `--synthetic` run)

The synthetic run prints, reproducibly (byte-identical Parquet across runs):
**16 of 18** planted pairs traded (screen recovers 17/18 with 0 false
discoveries; the tradeability filter drops one more), **OOS Sharpe +2.51
regime-scaled vs +2.78 unscaled** (the overlay did not help on this universe —
the report says so), max drawdown −0.3%, Calmar 13.4, 20% of days de-grossed.
The printout is the source of truth: if code or seeds change these numbers,
update this section to match.

---

## The pipeline in detail

### 1. Data layer (`quantdesk.data`)

**`PriceData`** loads per-ticker CSVs (one file per ticker, date-indexed) into a
single aligned DataFrame.  Forward-fills gaps and flags them in `.is_filled`.
The `as_of(date)` method is the look-ahead guard: every downstream caller
receives only data available up to that date.

**`make_synthetic_prices`** delegates to `statarb.utils.make_universe` —
a clustered universe where within-cluster pairs share a common stochastic trend
(cointegrated with known ground-truth hedge ratios).

**`events.py`** is a stub YAML loader for APAC macro events (BOJ/RBA/Fed/China
prints). Events are attached to report output for context; they are never used
in signal generation.

### 2. Configuration (`quantdesk.config`)

`QuantDeskConfig` is a pydantic model that holds every pipeline parameter:
universe/tickers, walk-forward windows, `SignalConfig`, `CostModel`,
`VolTargetConfig`, and regime de-grossing knobs.  Load from YAML:

```python
cfg = QuantDeskConfig.from_yaml("config.yaml")
```

Validated on load — `train_size > 0`, `0 < regime_stress_pct < 1`, etc.

### 3. Engine (`quantdesk.engine.portfolio`)

`run_regime_scaled_statarb(prices, cfg)` runs the full pipeline and returns a
`PortfolioResult`.

**Step 1 — Cointegration screen.**
`statarb.screen_universe(log_prices, n_clusters, fdr)` runs hierarchical
correlation clustering, tests within-cluster pairs with Engle-Granger, and
applies Benjamini-Hochberg FDR control to cap false discoveries.

**Step 2 — Tradeability filter + walk-forward backtest.**
For each FDR-surviving pair, `statarb.is_tradeable` applies the half-life
bounds (1–60 days) and capital-control flags. Surviving pairs get a
`statarb.backtest_pair` walk-forward OOS backtest: Kalman hedge ratio fit and
z-score statistics computed on the train window, FROZEN, then traded on the
test window at t+1 execution with explicit per-leg transaction costs.

**Step 3 — Market vol forecast.**
`volforecast.fit_garch` is called on the equal-weight daily return of the
universe. The GARCH conditional variance is converted to a daily sigma path
aligned to the prices index.  (Falls back to EWMA if the GARCH optimiser
diverges on degenerate data.)

**Step 4 — Regime scalar.**
`volforecast.vol_percentile_regime(full_sigma, lookback, stress_pct, min_scalar)`
classifies each day as calm/stressed by its trailing percentile and returns
an `exposure_scalar ∈ [min_scalar, 1]`.

**Step 5 — De-grossed book return.**

```
lagged_scalar[t] = scalar[t-1]      ← the no-look-ahead invariant
book_return[t]   = lagged_scalar[t] × mean(pair_returns[t])
```

The one-step lag means the de-grossing decision at time `t` was made from
information available at `t-1`.  A unit test asserts this algebraic identity
directly (`test_regime_scalar_is_lagged_by_one_period`).

**Step 6 — CUSUM break detection.**
`volforecast.detect_structural_break_cusum` is run on each pair's live spread
(OOS portion only).  Break dates are flagged in the result and shown in the
report — they inform position review but do not override the hard z-score stop
that `statarb.generate_signals` already applies.

### 4. Report (`quantdesk.report`)

`build_report(result, cfg, output_path)` writes a single self-contained HTML:
- OOS equity curve with regime-stressed periods shaded in red
- Regime scalar on a twin-axis with 21-day rolling portfolio vol
- Per-pair OOS Sharpe table (sorted best → worst)
- Headline metrics grid labelled **OUT-OF-SAMPLE ONLY**
- An honest regime-overlay verdict: if de-grossing did not improve Sharpe,
  the report says so and suggests recalibrating `stress_pct` / `min_scalar`

`to_parquet(result, output_path)` dumps the signal/return/regime timeseries
so results are reproducible and shareable without re-running the engine.

---

## Quality gates

| Gate | Test | What it proves |
|---|---|---|
| Look-ahead | `test_regime_scalar_is_lagged_by_one_period` | `book_return[t] == unscaled[t] × scalar[t-1]` for all t — contemporaneous scalar would be a look-ahead violation |
| Regime-link | `test_regime_degrosses_during_crisis_window` | In a synthetic crisis (3× calm vol) the mean exposure scalar is strictly lower than in the calm window |
| Reproducibility | `test_reproducibility` | Two runs on identical inputs produce byte-identical Parquet output |
| Upstream unchanged | `pytest statistical-arbitrage-v0 volatility-forecast-v0 options-pricing-v0` | 64 upstream tests pass untouched |

Run all 97 tests:

```bash
# From any directory
pytest jellystatic-markets/quantdesk-v0/tests/            # 33 quantdesk tests
pytest jellystatic-markets/statistical-arbitrage-v0/tests/ # 17
pytest jellystatic-markets/volatility-forecast-v0/tests/   # 17
pytest jellystatic-markets/options-pricing-v0/tests/       # 30
```

---

## Design decisions

**Why log prices for cointegration?**
Engle-Granger is run on `log(prices)` so the hedge ratio has the
interpretation of an elasticity (% return of Y vs % return of X). The OLS
beta is super-consistent under cointegration on non-stationary regressors.

**Why GARCH on the equal-weight return?**
The regime signal needs a market-level vol estimate, not a per-pair estimate.
The equal-weight portfolio return is a simple, interpretable proxy for the
APAC market factor that drives correlated drawdowns across pairs.

**Why a one-step lag on the regime scalar?**
The same invariant `statarb.backtest_pair` enforces for entry signals: the
decision made at time `t` uses only information available at `t`. Applying
the contemporaneous scalar would be a regime-timing bet, not de-grossing.

**Why not adjust the headline Sharpe to look good?**
The regime overlay verdict is computed from the actual OOS Sharpe of both the
scaled and unscaled book. On synthetic data with mild vol clustering the
de-grossing may not help — and the report says so. The brief is explicit:
*do not tune until it looks good*.
