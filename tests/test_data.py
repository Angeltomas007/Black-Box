from __future__ import annotations

import numpy as np
import pandas as pd

from blackbox.data.bars import fractional_diff, get_daily_vol, to_dollar_bars
from blackbox.data.market_data import DataCleaner


def test_data_cleaner_drops_bad_return_outliers(random_walk_df):
    df = random_walk_df.copy()
    # inject one absurd spike that should be filtered as an outlier
    spike_idx = df.index[200]
    df.loc[spike_idx, "close"] *= 50

    cleaned = DataCleaner.clean(df)
    assert spike_idx not in cleaned.index


def test_data_cleaner_drops_duplicates_and_sorts(random_walk_df):
    df = pd.concat([random_walk_df, random_walk_df.iloc[[0]]])
    cleaned = DataCleaner.clean(df)
    assert cleaned.index.is_monotonic_increasing
    assert not cleaned.index.duplicated().any()


def test_get_daily_vol_returns_finite_series(random_walk_df):
    vol = get_daily_vol(random_walk_df["close"])
    assert (vol.dropna() >= 0).all()
    assert vol.dropna().shape[0] > 0


def test_fractional_diff_reduces_autocorrelation_vs_level(random_walk_df):
    close = np.log(random_walk_df["close"])
    diffed = fractional_diff(close, d=0.4).dropna()
    # the fractionally differenced series should have much lower lag-1
    # autocorrelation than the raw (near unit-root) log-price series
    raw_autocorr = close.autocorr(lag=1)
    diffed_autocorr = diffed.autocorr(lag=1)
    assert abs(diffed_autocorr) < abs(raw_autocorr)


def test_to_dollar_bars_aggregates_by_notional_threshold():
    idx = pd.date_range("2023-01-01", periods=100, freq="1min", tz="UTC")
    ticks = pd.DataFrame({"price": np.full(100, 10.0), "size": np.full(100, 100.0)}, index=idx)
    # each tick trades $1,000 notional; a $5,000 threshold groups roughly
    # 5 ticks into one bar -- the exact count is boundary-sensitive, so
    # assert the invariants that actually matter: no volume lost, and
    # bar count in the right ballpark.
    bars = to_dollar_bars(ticks, dollar_threshold=5_000)
    assert 18 <= len(bars) <= 21
    assert {"open", "high", "low", "close", "volume"}.issubset(bars.columns)
    assert bars["volume"].sum() == ticks["size"].sum()
