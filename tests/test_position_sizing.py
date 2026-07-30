from __future__ import annotations

import pytest

from blackbox.risk.position_sizing import (
    confidence_scaled_size,
    kelly_fraction,
    volatility_target_size,
)


def test_kelly_fraction_zero_edge_bets_nothing():
    assert kelly_fraction(win_prob=0.5, win_loss_ratio=1.0) == 0.0


def test_kelly_fraction_positive_edge_bets_positive_fraction():
    f = kelly_fraction(win_prob=0.6, win_loss_ratio=1.0)
    assert f > 0.0


def test_kelly_fraction_respects_cap():
    f = kelly_fraction(win_prob=0.99, win_loss_ratio=5.0, cap=0.25)
    assert f == pytest.approx(0.25)


def test_kelly_fraction_never_negative_on_bad_edge():
    f = kelly_fraction(win_prob=0.2, win_loss_ratio=1.0)
    assert f == 0.0


def test_volatility_target_size_scales_inversely_with_asset_vol():
    small_vol_size = volatility_target_size(capital=1_000_000, target_annual_vol=0.10, asset_annual_vol=0.10, price=100)
    large_vol_size = volatility_target_size(capital=1_000_000, target_annual_vol=0.10, asset_annual_vol=0.40, price=100)
    assert small_vol_size > large_vol_size


def test_volatility_target_size_zero_on_zero_vol():
    assert volatility_target_size(capital=1_000_000, target_annual_vol=0.10, asset_annual_vol=0.0, price=100) == 0.0


def test_confidence_scaled_size_respects_max_position_pct():
    size = confidence_scaled_size(
        capital=1_000_000,
        price=100.0,
        predicted_proba=0.99,
        win_loss_ratio=10.0,
        kelly_cap=1.0,
        max_position_pct=0.05,
    )
    max_shares = 1_000_000 * 0.05 / 100.0
    assert size <= max_shares + 1e-9
