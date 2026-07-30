"""Shared synthetic-data fixtures.

None of these hit a network provider -- the point of unit tests is to
be fast and deterministic. Real-data validation belongs in a separate
backtest run against a live provider (see README), not in the test
suite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _ohlc_from_close(close: np.ndarray, index: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "open": close + rng.normal(0, 0.05, len(close)),
            "high": close + np.abs(rng.normal(0, 0.2, len(close))),
            "low": close - np.abs(rng.normal(0, 0.2, len(close))),
            "close": close,
            "volume": rng.integers(1000, 10_000, len(close)),
        },
        index=index,
    )
    # guarantee OHLC integrity regardless of the independent noise above
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)
    return df


@pytest.fixture
def random_walk_df() -> pd.DataFrame:
    n = 400
    idx = pd.date_range("2023-01-01", periods=n, freq="1D", tz="UTC")
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return _ohlc_from_close(close, idx, rng)


@pytest.fixture
def mean_reverting_df() -> pd.DataFrame:
    """Ornstein-Uhlenbeck process: genuinely mean-reverting, so the
    ADF-gated mean-reversion signal and the confluence signal both have
    something real to fire on."""
    n = 800
    idx = pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(3)
    theta, mu, sigma = 0.15, 100.0, 0.5
    x = np.full(n, mu)
    for i in range(1, n):
        x[i] = x[i - 1] + theta * (mu - x[i - 1]) + sigma * rng.normal()
    return _ohlc_from_close(x, idx, rng)


@pytest.fixture
def trending_df() -> pd.DataFrame:
    n = 300
    idx = pd.date_range("2023-01-01", periods=n, freq="1D", tz="UTC")
    rng = np.random.default_rng(5)
    close = 100 + np.arange(n) * 0.3 + rng.normal(0, 0.5, n)
    return _ohlc_from_close(close, idx, rng)
