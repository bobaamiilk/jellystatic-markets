"""volsurface: an institutional-style options pricing -> IV surface engine.

Project 1 of a three-system APAC derivatives toolkit. Modules:
  pricing  -- Black-Scholes / Black-76 closed form, Monte Carlo, binomial trees
  iv       -- robust implied-vol inversion (Brent/Newton hybrid + arb pre-check)
  data     -- option-chain cleaning and no-arbitrage filtering
  surface  -- SVI calibration and arbitrage-checked vol surface construction
"""
__version__ = "0.1.0"
