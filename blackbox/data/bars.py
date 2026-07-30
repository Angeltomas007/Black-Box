"""Information-driven bars (Lopez de Prado, AFML ch.2).

Time bars (1 bar per minute/day) oversample low-activity periods and
undersample high-activity ones, which breaks the i.i.d.-ish assumptions
most statistical tests rely on. Dollar bars sample every time a fixed
notional value has traded, giving a series with more stable statistical
properties -- useful when the alpha layer feeds a model that assumes
roughly stationary, normally-behaved increments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def to_dollar_bars(ticks: pd.DataFrame, dollar_threshold: float) -> pd.DataFrame:
    """Aggregate a tick/trade stream into dollar bars.

    ``ticks`` must have a DatetimeIndex and columns ['price', 'size'].
    """
    dollar_volume = (ticks["price"] * ticks["size"]).cumsum()
    bar_id = (dollar_volume // dollar_threshold).astype(int)

    grouped = ticks.groupby(bar_id)
    bars = pd.DataFrame(
        {
            "open": grouped["price"].first(),
            "high": grouped["price"].max(),
            "low": grouped["price"].min(),
            "close": grouped["price"].last(),
            "volume": grouped["size"].sum(),
        }
    )
    bars.index = grouped.apply(lambda g: g.index[-1])
    return bars


def get_daily_vol(close: pd.Series, span: int = 100) -> pd.Series:
    """Exponentially-weighted daily volatility of returns, used both as
    a triple-barrier width and as a position-sizing input (AFML ch.3)."""
    prior_idx = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    prior_idx = prior_idx[prior_idx > 0]
    prior_dates = pd.Series(
        close.index[prior_idx - 1], index=close.index[close.shape[0] - prior_idx.shape[0]:]
    )
    returns = close.loc[prior_dates.index] / close.loc[prior_dates.array].values - 1
    return returns.ewm(span=span).std()


def fractional_diff(series: pd.Series, d: float, threshold: float = 1e-4) -> pd.Series:
    """Fixed-window fractional differentiation (AFML ch.5).

    Standard integer differencing (returns, diff) achieves stationarity
    but destroys long-memory information the model could otherwise
    exploit. Fractional differencing finds the minimum ``d`` that makes
    the series stationary while preserving as much memory as possible.
    """
    weights = [1.0]
    k = 1
    while True:
        w_k = -weights[-1] / k * (d - k + 1)
        if abs(w_k) < threshold:
            break
        weights.append(w_k)
        k += 1
    weights = np.array(weights[::-1])
    width = len(weights)

    out = pd.Series(index=series.index, dtype=float)
    values = series.values
    for i in range(width - 1, len(values)):
        window = values[i - width + 1: i + 1]
        if np.isnan(window).any():
            continue
        out.iloc[i] = np.dot(weights, window)
    return out
