# volforecast — Volatility Forecasting Engine (APAC focus)

A from-first-principles volatility-forecasting system: GARCH-family models fit by
**hand-rolled maximum likelihood** (no black-box library), the EWMA/rolling
benchmarks they must beat, *robust* forecast evaluation, volatility targeting,
and a regime signal that governs position sizing. Validated against simulated
returns with **known latent volatility**.

## What it does

```
returns ─▶ GARCH(1,1)/EGARCH MLE ─▶ walk-forward OOS vs EWMA/rolling
        ─▶ QLIKE + Diebold–Mariano ─▶ vol targeting ─▶ regime exposure signal
```

## Method

**Volatility is latent.** Returns are ~serially uncorrelated but *squared*
returns cluster. GARCH(1,1):
`σ²_t = ω + α·ε²_{t−1} + β·σ²_{t−1}`, with `α` = news reaction, `β` =
persistence, `α+β<1` for stationarity, long-run variance `σ̄² = ω/(1−α−β)`.
Estimated by MLE with **variance targeting** (ω pinned to the sample variance,
removing a near-flat direction). **EGARCH** models `ln σ²` and adds a leverage
term `γ`; `γ<0` means negative shocks raise vol more — the same asymmetry that
produces equity skew.

**Benchmarks & the key identity.** Rolling variance suffers the **ghosting**
artifact (a shock holds the estimate up for exactly `window` days, then drops off
a cliff). **EWMA = GARCH(1,1) with ω=0, α=1−λ, β=λ**, i.e. `α+β=1`: unit
persistence ⇒ **no mean reversion** ⇒ a *flat* multi-step forecast. A fine level
estimator, a poor forecaster at horizon. Both facts are asserted in tests.

**Robust evaluation (Patton).** The target is latent; we score against the noisy
proxy `r²`. Only some losses are *robust* (noisy proxy doesn't change the ranking):
- `MSE = mean((σ̂² − r²)²)` — robust, but dominated by a few extreme days.
- `QLIKE = mean(r²/σ̂² + ln σ̂²)` — robust **and asymmetric**, penalising
  *under*-prediction more, matching the cost of under-sizing risk into a crisis.
`MAE-on-vol` is included as a deliberately **non-robust** counter-example.
**Diebold–Mariano** (with a Newey–West **HAC** variance) tests significance —
including the honest "no significant difference, don't pay for GARCH" verdict.

**Vol targeting.** `w_t = σ_target / σ̂_t` holds risk roughly constant; realised
vol `= σ_target·(σ_t/σ̂_t)`, so forecast error *is* tracking error. Limitation:
it de-levers *after* vol rises (protects against persistence, not the onset jump)
and sizes up most in calm — so **leverage cap, vol floor, and band rebalancing**
are built in.

**Regime.** Switch (recurring, crisis has a correlation signature) vs break
(permanent). MS-GARCH is fragile, so a **robust vol-percentile / realized-
correlation proxy** is preferred — *size to the regime, don't time the turn*. The
exposure scalar it outputs is the cross-system link that de-grosses the stat-arb
book.

## Layout

```
src/volforecast/
  models/      garch.py        benchmarks.py     (GARCH/EGARCH MLE; rolling/EWMA + identity)
  evaluation/  losses.py       walkforward.py    (MSE/QLIKE/MAE, DM+HAC, OOS harness)
  trading/     sizing.py                         (vol targeting, cap/floor/band)
  regime/      regime.py                         (vol-percentile regime, CUSUM break)
  utils/       synthetic_data.py                 (known-truth GARCH/EGARCH/regime sims)
  examples/    run_pipeline.py                   (full demo + plot)
tests/         test_volforecast.py               (17 tests)
```

## Run

```bash
pip install -e .
pytest -q                                       # 17 passing
python -m volforecast.examples.run_pipeline      # full demo + reports/volforecast_pipeline.png
```

## Validation highlights
- GARCH MLE recovers persistence to ±0.03 (0.981 vs true 0.98) and α to ±0.04.
- Stationarity (`α+β<1`) and `ω>0` enforced; multi-step forecast mean-reverts to `σ̄²`.
- EGARCH recovers a negative leverage `γ`.
- EWMA proven equal to constrained GARCH and flat in forecast; ghosting reproduced.
- QLIKE/MSE prefer the true-variance forecast; QLIKE penalises under-prediction more.
- DM returns "neither" for identical models; example run honestly reports GARCH ≈ EWMA at 1-step.
- Vol targeting lands realised vol on target and respects the leverage cap.

*Synthetic data only; no investment advice. Built as a learning/portfolio project.*
