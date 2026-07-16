# APAC Quant Systems

This started because I was tired of being able to *say* the words (cointegration, vol surface, GARCH) without really being able to build any of it. So I sat down and built the three systems a derivatives desk actually runs on, from the maths up, focused on Asia-Pacific markets (Nikkei, Hang Seng, KOSPI 200, ASX 200) since that's where I am.

The one rule I gave myself: no calling a library to do the hard part. If the project is GARCH, the likelihood is coded in this repo. If it's a vol surface, so is the calibration. Otherwise I'd just be importing someone else's understanding and pretending it was mine.

Worth saying plainly, since this repo is about not fooling yourself: I built these systems with an AI assistant writing much of the code alongside me — my work is the system design, the choice of methods, and the validation (deciding what to test, and checking every result against ground truth I generated). The "no library for the hard part" rule still holds — the GARCH likelihood, SVI calibration, IV solver and Kalman filter are all implemented here, not imported — but "I typed every line" would be the wrong claim, so I'm not making it.

| # | System | What it does | Tests |
|---|--------|--------------|:-----:|
| 1 | [Options pricing → IV surface](./options-pricing-v0) | Raw option chain in, clean arbitrage-free vol surface out | 30 ✅ |
| 2 | [Statistical arbitrage](./statistical-arbitrage-v0) | Finds mean-reverting pairs, hedges them, backtests them honestly | 17 ✅ |
| 3 | [Volatility forecasting](./volatility-forecast-v0) | GARCH/EGARCH forecasts that turn into position sizes | 17 ✅ |
| 4 | [Integration layer (quantdesk)](./quantdesk-v0) | Wires 2 + 3 into one runner: the vol regime scalar de-grosses the stat-arb book | 33 ✅ |

97 tests passing across the four packages. All plain Python: NumPy, SciPy, pandas, statsmodels. The GARCH MLE, the SVI fit, the implied-vol solver and the Kalman filter are all implemented in this repo, not imported.

**Worth saying plainly:** everything here runs on synthetic data I generated with known parameters, and the tests pass when the code recovers those parameters. That tells you the implementations are *correct*. It does not tell you any of this makes money, which needs real data, and that's the next job. Nothing here is financial advice.

---

## Why three, and why they're in one repo

I didn't want three random scripts in a folder. These are genuinely the three problems one desk has to solve at once, and the interesting part is where they touch.

```text
                 +--------------------------+
   option chain →| 1 · Pricing → IV surface |→ vega / skew / pricing context
                 +--------------------------+
                 +--------------------------+
   price series →| 3 · Volatility forecast  |→ regime "size down" signal ──┐
                 +--------------------------+                              │
                 +--------------------------+      cut risk going INTO     │
   universe     →| 2 · Statistical arbitrage|◀───── the stress, not after ─┘
                 +--------------------------+
```

The bit I actually care about: the vol forecaster (3) spits out a regime scalar, and that scalar is what shrinks the stat-arb book (2) when things get scary, so you're cutting risk on the way *into* a crisis instead of reading about it afterwards. That hand-off is real code now — `quantdesk-v0` runs the whole chain end-to-end (see system 4 below). The surface (1) is the pricing and vega backdrop. And honestly the thing that ties them together more than any arrow is the habits: test out-of-sample, never peek at the future, and report the ugly results instead of burying them.

---

## The three systems — and the layer that wires them together

### 1 · Options pricing → implied vol surface · [`options-pricing-v0/`](./options-pricing-v0)

Chain goes in, a clean arbitrage-free volatility surface comes out. It's all in Black-76 / forward terms, because APAC index options are quoted off futures, not spot, a detail that trips people up and that I wanted to get right from the start.

Inside it: closed-form Black-Scholes/Black-76 with the Greeks, Monte Carlo with antithetic and control variates, and a binomial tree for the American-style HKEX single names (Tencent, HSBC and friends), where you can exercise early around dividends and closed-form pricing just can't help you.

The implied-vol solver was the part that taught me the most. The obvious choice is Newton, and Newton falls apart in the deep wings (think KOSPI weeklies) where vega goes to nearly zero and the steps explode. So it's Brent, with an arbitrage check up front so it never even tries to invert a price that has no solution.

The surface is SVI, fit with the butterfly no-arbitrage condition `b(1+|ρ|) ≤ 4` baked in as a hard constraint, because a least-squares fit will happily give you a great-looking curve that secretly contains an arbitrage if you let it.

Things I made sure of: put-call parity to 1e-10, the solver round-trips a known vol to ~1e-6, the fitted surface is arbitrage-free, and a small numerical trick (parity-aware Richardson extrapolation) makes the tree 10×+ more accurate.

### 2 · Statistical arbitrage · [`statistical-arbitrage-v0/`](./statistical-arbitrage-v0)

Find pairs that move together, trade the gap when it stretches, and (the hard part) backtest it without quietly lying to yourself.

It uses Engle-Granger for cointegration, with the half-life of mean reversion tracked as a sort of health bar for each pair. One thing I deliberately built in is the reminder that cointegrated isn't the same as tradeable: the A-share/H-share premium is textbook-cointegrated and completely untradeable because capital controls won't let you put the trade on. Easy to forget when you're staring at a p-value.

Scaling from one pair to a whole universe is where it gets dangerous. At 200 names you're running ~20,000 tests, and a chunk of them will look "significant" by pure luck. So there's correlation clustering to only test pairs that make economic sense, plus Benjamini-Hochberg to control the false-discovery rate.

The hedge ratio drifts over time (a Kalman filter handles that), the signal is a vol-adjusted z-score, and the backtest is strict about it: decide today, trade tomorrow, pay real costs, and there's a test that literally fails if future data ever leaks into a past decision.

On this package's demo universe (`python -m statarb.examples.run_pipeline`) the screen pulls out all 18 planted pairs with zero false positives; on the quantdesk integration universe it recovers 17 of 18 — still zero false discoveries — and 16 survive the tradeability filter. Two different synthetic universes, so the numbers differ; each README quotes the printout of its own run. And the demo shows a Sharpe of ~1.9 in-sample dropping to ~1.0 out-of-sample, which isn't a bug. That gap is what an honest backtest looks like, and pretending it isn't there is how people fool themselves.

### 3 · Volatility forecasting · [`volatility-forecast-v0/`](./volatility-forecast-v0)

Forecast how volatile things are about to be, then turn that into how big a position to hold.

The models are GARCH(1,1) and EGARCH, both fit by hand-rolled maximum likelihood. EGARCH was worth the extra effort because its leverage term captures the asymmetry (bad news moves vol more than good news), which is the same thing that gives you equity skew back in system 1.

I benchmark against rolling and EWMA vol, partly because there's a genuinely nice fact hiding there: EWMA is just GARCH with the parameters pinned (ω=0, α+β=1), which is exactly why it never mean-reverts. Scoring is QLIKE (which stays trustworthy even though you can never actually observe true volatility) plus a Diebold-Mariano test for whether the difference between two models is real or just noise.

And here's the part I'm weirdly proud of: when I ran it, GARCH did *not* significantly beat EWMA at a one-step horizon (Diebold-Mariano p = 0.51), and the code says so out loud instead of massaging the numbers until GARCH wins. It recovers the true persistence to within ±0.03 (0.981 vs 0.98), and the vol-targeting layer that sits on top lands realised volatility right on its target, with a leverage cap and vol floor so it can't do anything stupid.

### 4 · The integration layer · [`quantdesk-v0/`](./quantdesk-v0)

This is the single runner the other three were pointing at: prices in, a regime-scaled stat-arb book out, with a self-contained HTML report and a Parquet dump at the end. It treats the three packages as frozen libraries — it calls their public APIs and duplicates none of their logic — and the regime scalar is applied with a one-step lag, enforced by a unit test, so the de-grossing decision at time *t* only ever uses information from *t−1*.

The headline from the deterministic synthetic run: 16 of 18 planted pairs traded, and an out-of-sample Sharpe of **+2.51 with the regime overlay versus +2.78 without it**. Read that again: on this universe the de-grossing *cost* Sharpe, and the report prints that verdict instead of hiding it. The rule for every number in this repo is the same: the printout of a fresh clean-clone run is the truth, and the documentation follows it — never the other way around.

---

## What's actually in here

```text
jellystatic-markets/
├── options-pricing-v0/          # system 1: volsurface  (30 tests)
├── statistical-arbitrage-v0/    # system 2: statarb     (17 tests)
├── volatility-forecast-v0/      # system 3: volforecast (17 tests)
├── quantdesk-v0/                # system 4: integration (33 tests)
└── README.md                    # (plus a few demo plots at the root)
```

Each folder is its own installable Python package with its own README and its own tests. Systems 1–3 each have a `run_pipeline.py` you can just run to watch the whole thing work and spit out plots; system 4 runs as `python -m quantdesk`.

## Running it

```bash
# any single system, from inside its folder
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
python -m volsurface.examples.run_pipeline   # or statarb / volforecast

# the integrated pipeline (needs all four packages installed)
pip install -e ./statistical-arbitrage-v0 -e ./volatility-forecast-v0 \
            -e ./options-pricing-v0 -e "./quantdesk-v0[dev]"
(cd quantdesk-v0 && python -m quantdesk run --synthetic)   # < 10 s, writes out/report.html
```

---

## How I went about it

A few principles I stuck to, because they're the difference between understanding something and being able to recite it.

Derive it before coding it, with the maths in the comments rather than abstracted away in a dependency. Test against ground truth: I know the right answer because I generated the data, so the tests check the code actually finds it. Never look ahead, which means causal stats, walk-forward evaluation, and tests that break if I cheat. And treat the bad results as results, because a pair that loses money and gets correctly diagnosed as a regime break is more useful than three suspiciously clean Sharpe ratios.

## What it doesn't do yet (the honest list)

It's all synthetic so far, so real data adapters are the obvious next step. The SVI fit is done one maturity at a time, so calendar arbitrage gets checked but not prevented (the proper fix is SSVI). The single runner exists now (`quantdesk-v0`), but it only wires in systems 2 and 3 — the options surface is available for vega/skew context and doesn't drive positions yet. On the synthetic universe the regime overlay didn't improve Sharpe, so `stress_pct` / `min_scalar` need recalibrating rather than celebrating. And there's a whole shopping list of models I'd like to add: local vol, Heston, Longstaff-Schwartz for American options.

---

*A personal project to understand quant trading end to end. Not advice, and nothing here trades real money.*
