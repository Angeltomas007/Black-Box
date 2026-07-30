from __future__ import annotations

import numpy as np
import pandas as pd

from blackbox.alpha.features import (
    average_true_range,
    bollinger_bands,
    build_feature_matrix,
    higher_timeframe_trend,
    rsi,
)


def test_rsi_bounded_between_0_and_100(random_walk_df):
    values = rsi(random_walk_df["close"]).dropna()
    assert (values >= 0).all() and (values <= 100).all()


def test_rsi_handles_flat_series_without_exploding():
    flat = pd.Series(100.0, index=pd.date_range("2023-01-01", periods=50, freq="1D"))
    values = rsi(flat)
    # no gain and no loss -> undefined, but must not raise or produce inf
    assert not np.isinf(values.dropna()).any()


def test_atr_is_non_negative(random_walk_df):
    atr = average_true_range(random_walk_df).dropna()
    assert (atr >= 0).all()


def test_bollinger_bands_ordering(random_walk_df):
    upper, middle, lower = bollinger_bands(random_walk_df["close"])
    valid = upper.notna() & middle.notna() & lower.notna()
    assert (upper[valid] >= middle[valid]).all()
    assert (middle[valid] >= lower[valid]).all()


def test_higher_timeframe_trend_only_uses_closed_bars(mean_reverting_df):
    """The whole point of shifting the resampled trend by one HTF bar is
    that a value at time t must not depend on any HTF bar still open at
    t. Concretely: truncating the input at some bar T must not change
    the trend value already computed for any bar strictly before the
    HTF bar containing T (no lookahead into the future)."""
    df = mean_reverting_df
    full_trend = higher_timeframe_trend(df, rule="1D")

    cutoff = df.index[len(df) // 2]
    truncated_trend = higher_timeframe_trend(df.loc[:cutoff], rule="1D")

    # any bar at least one full HTF period before the cutoff must be
    # identical whether or not future data beyond the cutoff exists
    safe_before = cutoff - pd.Timedelta(days=2)
    common_idx = full_trend.index[full_trend.index <= safe_before]
    pd.testing.assert_series_equal(
        full_trend.loc[common_idx], truncated_trend.loc[common_idx], check_names=False
    )


def test_build_feature_matrix_has_expected_columns(random_walk_df):
    feats = build_feature_matrix(random_walk_df, lookback=60)
    expected = {
        "zscore", "rsi", "atr_pct", "realized_vol", "frac_diff",
        "momentum_20", "momentum_60", "volume_zscore",
    }
    assert expected.issubset(feats.columns)
    assert len(feats.dropna()) > 0
