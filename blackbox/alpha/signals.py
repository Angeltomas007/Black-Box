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

from blackbox.alpha.candles import detect_patterns
from blackbox.alpha.features import bollinger_bands, higher_timeframe_trend, rolling_zscore, rsi


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


class PriceActionConfluenceSignal:
    """Multi-factor confluence: a higher-timeframe trend filter, an
    RSI/Bollinger extremity, and a confirming candlestick pattern
    (Nison-style price action) must all agree before a side is emitted.

    Every input is computed from the same OHLCV series -- the HTF
    trend is a real resample of ``df``, shifted to only reference
    closed higher-timeframe bars (see
    ``features.higher_timeframe_trend``) -- so there is no possibility
    of the HTF filter being an artifact of an independently-simulated
    series, and no lookahead into an unclosed bar.

    ``strength`` is a continuous confluence score in [0, 1] built from
    how far price sits into oversold/overbought territory, not a fixed
    constant: three trades that all technically qualify are not
    equally convincing, and the risk layer's confidence-scaled sizing
    needs a real gradient to act on.
    """

    def __init__(
        self,
        htf_rule: str = "15min",
        rsi_window: int = 14,
        bb_window: int = 20,
        bb_std: float = 2.0,
        rsi_oversold: float = 35.0,
        rsi_overbought: float = 65.0,
    ):
        self.htf_rule = htf_rule
        self.rsi_window = rsi_window
        self.bb_window = bb_window
        self.bb_std = bb_std
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def generate(self, df: pd.DataFrame) -> SignalResult:
        close = df["close"]
        trend = higher_timeframe_trend(df, self.htf_rule)
        rsi_values = rsi(close, self.rsi_window)
        upper, _, lower = bollinger_bands(close, self.bb_window, self.bb_std)
        patterns = detect_patterns(df)

        rsi_oversold_cond = rsi_values < self.rsi_oversold
        touch_lower = close <= lower
        rsi_overbought_cond = rsi_values > self.rsi_overbought
        touch_upper = close >= upper

        bullish_pattern = patterns["is_bullish_engulfing"] | patterns["is_doji"]
        bearish_pattern = patterns["is_bearish_engulfing"] | patterns["is_doji"]

        long_cond = (trend > 0) & (rsi_oversold_cond | touch_lower) & bullish_pattern
        short_cond = (trend < 0) & (rsi_overbought_cond | touch_upper) & bearish_pattern

        side = pd.Series(0, index=close.index)
        side[long_cond.fillna(False)] = 1
        side[short_cond.fillna(False)] = -1

        rsi_extremity_long = ((self.rsi_oversold - rsi_values) / self.rsi_oversold).clip(lower=0)
        rsi_extremity_short = (
            (rsi_values - self.rsi_overbought) / (100 - self.rsi_overbought)
        ).clip(lower=0)
        bb_depth_long = ((lower - close) / lower.replace(0, np.nan)).clip(lower=0)
        bb_depth_short = ((close - upper) / upper.replace(0, np.nan)).clip(lower=0)

        strength = pd.Series(0.0, index=close.index)
        base_confidence = 0.5  # all three gating conditions already agreed
        strength[long_cond.fillna(False)] = (
            base_confidence
            + 0.25 * rsi_extremity_long[long_cond.fillna(False)].fillna(0)
            + 0.25 * bb_depth_long[long_cond.fillna(False)].fillna(0)
        ).clip(upper=1.0)
        strength[short_cond.fillna(False)] = (
            base_confidence
            + 0.25 * rsi_extremity_short[short_cond.fillna(False)].fillna(0)
            + 0.25 * bb_depth_short[short_cond.fillna(False)].fillna(0)
        ).clip(upper=1.0)

        return SignalResult(side=side, strength=strength)
