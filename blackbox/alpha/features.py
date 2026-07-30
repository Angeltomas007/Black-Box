"""Feature engineering shared by the mean-reversion and momentum
signals, and by the meta-labeling model. Every feature here is computed
strictly from information available at time t (no centered rolling
windows, no future close in the denominator) to avoid the lookahead
bias De Prado warns is the single most common cause of backtest
overfitting (AFML ch.4).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from blackbox.data.bars import fractional_diff


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std.replace(0, np.nan)


def average_true_range(df: pd.DataFrame, window: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(span=window, min_periods=window).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def realized_volatility(close: pd.Series, window: int = 20, annualize: bool = True) -> pd.Series:
    returns = np.log(close).diff()
    vol = returns.rolling(window).std()
    return vol * np.sqrt(252) if annualize else vol


def bollinger_bands(
    close: pd.Series, window: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    middle = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


def higher_timeframe_trend(df: pd.DataFrame, rule: str, sma_window: int = 20) -> pd.Series:
    """Resample ``df`` to a higher timeframe (e.g. "15min") and compute
    a trend filter (close vs SMA) on *fully closed* HTF bars only, then
    project it back onto ``df``'s index.

    Two lookahead traps this avoids, both present in the naive
    "request higher timeframe close" pattern:
      1. Resampling from the *same* underlying series (not an
         independently-fetched/simulated series) so the HTF bars are
         actually derived from the LTF data being traded.
      2. Shifting the resampled trend by one HTF bar before
         reindexing, so a bar's trend value only reflects HTF bars that
         had already closed by that timestamp -- the equivalent of
         Pine's ``lookahead=barmerge.lookahead_off``.
    """
    htf = df.resample(rule).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    htf_sma = htf["close"].rolling(sma_window).mean()
    htf_trend = np.sign(htf["close"] - htf_sma).shift(1)
    return htf_trend.reindex(df.index, method="ffill")


def build_feature_matrix(df: pd.DataFrame, lookback: int = 60, frac_diff_d: float = 0.4) -> pd.DataFrame:
    """Assemble the full feature set used to train the meta-labeling
    classifier. ``df`` must contain OHLCV columns indexed by timestamp."""
    close = df["close"]
    feats = pd.DataFrame(index=df.index)
    feats["zscore"] = rolling_zscore(close, lookback)
    feats["rsi"] = rsi(close)
    feats["atr_pct"] = average_true_range(df) / close
    feats["realized_vol"] = realized_volatility(close)
    feats["frac_diff"] = fractional_diff(np.log(close), d=frac_diff_d)
    feats["momentum_20"] = close.pct_change(20)
    feats["momentum_60"] = close.pct_change(60)
    feats["volume_zscore"] = rolling_zscore(df["volume"], lookback)
    return feats
