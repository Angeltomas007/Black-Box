"""Primary alpha signals.

Both signals are deliberately simple, statistically-motivated models --
not curve-fit indicator soup. Each emits a "primary side" in
{-1, 0, +1}; the ML layer in ``model.py`` then meta-labels these calls
rather than replacing them, per Narang's warning that a pure black-box
ML signal with no economic rationale is fragile and hard to risk-manage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from blackbox.alpha.features import rolling_zscore


@dataclass(frozen=True)
class SignalResult:
    side: pd.Series  # -1 (short), 0 (flat), +1 (long)
    strength: pd.Series  # unsigned confidence proxy in [0, inf)


class MeanReversionSignal:
    """Bollinger/z-score mean reversion, gated by an Augmented
    Dickey-Fuller stationarity test so the strategy only trades names
    that are statistically mean-reverting over the lookback window
    (Chan, "Quantitative Trading", ch.2 on cointegration/stationarity
    screening before deploying a mean-reversion book)."""

    def __init__(self, lookback: int = 60, entry_z: float = 2.0, exit_z: float = 0.5, adf_pvalue: float = 0.05):
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.adf_pvalue = adf_pvalue

    def is_stationary(self, close: pd.Series) -> bool:
        from statsmodels.tsa.stattools import adfuller

        window = close.tail(self.lookback).dropna()
        if len(window) < max(20, self.lookback // 2):
            return False
        stat, pvalue, *_ = adfuller(window, autolag="AIC")
        return pvalue < self.adf_pvalue

    def generate(self, df: pd.DataFrame) -> SignalResult:
        close = df["close"]
        z = rolling_zscore(close, self.lookback)

        stationary = close.rolling(self.lookback).apply(
            lambda w: float(self.is_stationary(pd.Series(w))), raw=False
        ).fillna(0).astype(bool)

        side = pd.Series(0, index=close.index)
        side[(z < -self.entry_z) & stationary] = 1
        side[(z > self.entry_z) & stationary] = -1
        side[z.abs() < self.exit_z] = 0
        side = side.replace(0, np.nan).ffill().fillna(0)
        side[~stationary] = 0

        return SignalResult(side=side, strength=z.abs())


class MomentumSignal:
    """Dual moving-average trend filter with a volatility-normalized
    strength measure. Momentum and mean-reversion are complementary
    regimes (Narang ch.4): running both and letting the risk layer
    allocate between them by realized Sharpe is more robust than
    picking one regime permanently.
    """

    def __init__(self, fast: int = 20, slow: int = 100):
        self.fast = fast
        self.slow = slow

    def generate(self, df: pd.DataFrame) -> SignalResult:
        close = df["close"]
        fast_ma = close.rolling(self.fast).mean()
        slow_ma = close.rolling(self.slow).mean()
        spread = (fast_ma - slow_ma) / slow_ma

        side = pd.Series(0, index=close.index)
        side[spread > 0] = 1
        side[spread < 0] = -1

        return SignalResult(side=side, strength=spread.abs())
