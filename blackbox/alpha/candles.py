"""Candlestick pattern recognition (price action layer, Nison-style).

Kept separate from ``signals.py`` because pattern detection is a pure
function of OHLC geometry -- it has no notion of lookback windows,
thresholds tuned on history, or statistical significance, so it
shouldn't share a class hierarchy with the statistically-gated signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_patterns(df: pd.DataFrame, doji_body_ratio: float = 0.1) -> pd.DataFrame:
    """Return a copy of ``df`` with boolean pattern columns appended:
    ``is_doji``, ``is_bullish_engulfing``, ``is_bearish_engulfing``.

    Each row only uses that row's and the prior row's OHLC -- no
    forward-looking data, so these columns are safe to use as of the
    bar's close.
    """
    out = df.copy()
    body = out["close"] - out["open"]
    candle_range = (out["high"] - out["low"]).replace(0, np.nan)

    out["is_doji"] = (body.abs() <= doji_body_ratio * candle_range).fillna(False)

    prev_body = body.shift(1)
    prev_open = out["open"].shift(1)
    prev_close = out["close"].shift(1)

    out["is_bullish_engulfing"] = (
        (body > 0)
        & (prev_body < 0)
        & (out["close"] >= prev_open)
        & (out["open"] <= prev_close)
    ).fillna(False)

    out["is_bearish_engulfing"] = (
        (body < 0)
        & (prev_body > 0)
        & (out["close"] <= prev_open)
        & (out["open"] >= prev_close)
    ).fillna(False)

    return out
