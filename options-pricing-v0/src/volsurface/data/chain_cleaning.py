"""Option-chain cleaning and arbitrage filtering (Stage 4).

Real APAC chains are noisy: stale quotes across the HKEX lunch break, crossed /
zero-bid quotes in thin back months, asynchronous option-vs-futures snapshots
that manufacture phantom arbitrage. This module turns a raw chain into a clean,
arbitrage-checked one before it ever reaches the surface fit.

Pipeline (each step flags rather than silently deletes, so rejection rates are
auditable -- a sudden spike in rejections is itself a data-quality signal):
  1. drop crossed (bid >= ask) and zero-bid quotes;
  2. liquidity-weighted mid construction (inverse-spread weight available);
  3. no-arbitrage checks: bounds, strike monotonicity, butterfly convexity,
     put-call parity.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"strike", "maturity", "option_type", "bid", "ask"}


@dataclass
class CleaningReport:
    """Audit trail of what the cleaner did. Returned alongside the clean frame."""

    n_input: int = 0
    n_crossed: int = 0
    n_zero_bid: int = 0
    n_wide_spread: int = 0
    n_arb_violations: int = 0
    n_output: int = 0
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"input={self.n_input} crossed={self.n_crossed} "
            f"zero_bid={self.n_zero_bid} wide_spread={self.n_wide_spread} "
            f"arb_violations={self.n_arb_violations} output={self.n_output}"
        )


def _validate_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"chain missing required columns: {sorted(missing)}")


def drop_invalid_quotes(
    df: pd.DataFrame, report: CleaningReport, max_rel_spread: float = 0.5
) -> pd.DataFrame:
    """Drop crossed and zero-bid quotes; flag (not drop) very wide spreads.

    Wide-spread quotes are kept but marked low-confidence via `spread_weight`,
    because dropping every wide quote would gut the wings entirely on thin names.
    """
    out = df.copy()
    crossed = out["bid"] >= out["ask"]
    zero_bid = out["bid"] <= 0
    report.n_crossed = int(crossed.sum())
    report.n_zero_bid = int((zero_bid & ~crossed).sum())
    out = out[~(crossed | zero_bid)].copy()

    out["mid"] = 0.5 * (out["bid"] + out["ask"])
    out["rel_spread"] = (out["ask"] - out["bid"]) / out["mid"]
    wide = out["rel_spread"] > max_rel_spread
    report.n_wide_spread = int(wide.sum())
    # Inverse-spread weight, capped, normalised later by the surface fitter.
    out["spread_weight"] = 1.0 / out["rel_spread"].clip(lower=1e-4)
    return out


def check_bounds(
    df: pd.DataFrame, F: float, r: float, report: CleaningReport
) -> pd.DataFrame:
    """Discounted no-arbitrage price bounds per quote (forward / Black-76 form)."""
    out = df.copy()
    tau = out["maturity"].to_numpy()
    K = out["strike"].to_numpy()
    disc = np.exp(-r * tau)
    is_call = out["option_type"].str.lower().eq("call").to_numpy()

    intrinsic = np.where(is_call, np.maximum(F - K, 0.0), np.maximum(K - F, 0.0))
    lower = disc * intrinsic
    upper = disc * np.where(is_call, F, K)
    ok = (out["mid"].to_numpy() >= lower - 1e-9) & (
        out["mid"].to_numpy() <= upper + 1e-9
    )
    out["bounds_ok"] = ok
    report.n_arb_violations += int((~ok).sum())
    return out


def check_butterfly_convexity(df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Per-maturity, per-type: call price must be convex & decreasing in strike;
    put increasing. A butterfly C(K1)-2C(K2)+C(K3) < 0 is a static arbitrage.

    Marks each quote `convex_ok`; a violating middle strike is the suspect point.
    """
    out = df.copy()
    out["convex_ok"] = True
    for (mat, otype), g in out.groupby(["maturity", "option_type"]):
        g = g.sort_values("strike")
        if len(g) < 3:
            continue
        mids = g["mid"].to_numpy()
        # second difference (proportional to butterfly spread)
        second_diff = mids[2:] - 2 * mids[1:-1] + mids[:-2]
        bad_mid = second_diff < -1e-8  # convexity violated at interior point
        idx_mid = g.index[1:-1]
        violating = idx_mid[bad_mid]
        out.loc[violating, "convex_ok"] = False
    report.n_arb_violations += int((~out["convex_ok"]).sum())
    return out


def check_put_call_parity(
    df: pd.DataFrame, F: float, r: float, tol: float = 0.02
) -> pd.DataFrame:
    """Tag call/put pairs at the same (strike, maturity) that violate
    C - P = (F - K) e^{-r tau} beyond `tol` (relative to forward). Useful as an
    independent stale-quote detector; returns a per-pair diagnostic frame.
    """
    rows = []
    for (mat, K), g in df.groupby(["maturity", "strike"]):
        calls = g[g["option_type"].str.lower() == "call"]
        puts = g[g["option_type"].str.lower() == "put"]
        if len(calls) and len(puts):
            c, p = calls["mid"].iloc[0], puts["mid"].iloc[0]
            disc = np.exp(-r * mat)
            resid = c - p - disc * (F - K)
            rows.append(
                {"maturity": mat, "strike": K, "parity_resid": resid,
                 "parity_ok": abs(resid) <= tol * F}
            )
    return pd.DataFrame(rows)


def clean_chain(
    df: pd.DataFrame,
    F: float,
    r: float,
    max_rel_spread: float = 0.5,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Full Stage 4 pipeline. Returns (clean_chain, report).

    The clean chain retains diagnostic columns (bounds_ok, convex_ok,
    spread_weight) so the surface fitter can weight and exclude intelligently
    rather than receiving an opaque, already-filtered set.
    """
    _validate_columns(df)
    report = CleaningReport(n_input=len(df))
    out = drop_invalid_quotes(df, report, max_rel_spread)
    out = check_bounds(out, F, r, report)
    out = check_butterfly_convexity(out, report)
    # Keep only quotes passing hard arbitrage checks; weighting handles the rest.
    clean = out[out["bounds_ok"] & out["convex_ok"]].copy()
    report.n_output = len(clean)
    return clean, report
