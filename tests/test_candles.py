from __future__ import annotations

import pandas as pd

from blackbox.alpha.candles import detect_patterns


def test_bullish_engulfing_detected():
    idx = pd.date_range("2023-01-01", periods=2, freq="1D", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [10.0, 9.5],
            "high": [10.2, 11.2],
            "low": [9.4, 9.4],
            "close": [9.6, 11.0],
            "volume": [1000, 1000],
        },
        index=idx,
    )
    result = detect_patterns(df)
    assert bool(result["is_bullish_engulfing"].iloc[-1]) is True
    assert bool(result["is_bearish_engulfing"].iloc[-1]) is False


def test_bearish_engulfing_detected():
    idx = pd.date_range("2023-01-01", periods=2, freq="1D", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [9.6, 11.0],
            "high": [10.2, 11.2],
            "low": [9.4, 8.8],
            "close": [10.0, 9.0],
            "volume": [1000, 1000],
        },
        index=idx,
    )
    result = detect_patterns(df)
    assert bool(result["is_bearish_engulfing"].iloc[-1]) is True
    assert bool(result["is_bullish_engulfing"].iloc[-1]) is False


def test_doji_detected_on_near_zero_body():
    idx = pd.date_range("2023-01-01", periods=1, freq="1D", tz="UTC")
    df = pd.DataFrame(
        {"open": [10.0], "high": [10.5], "low": [9.5], "close": [10.02]}, index=idx
    )
    df["volume"] = 1000
    result = detect_patterns(df)
    assert bool(result["is_doji"].iloc[-1]) is True


def test_detect_patterns_does_not_mutate_input(random_walk_df):
    original = random_walk_df.copy()
    detect_patterns(random_walk_df)
    pd.testing.assert_frame_equal(random_walk_df, original)
