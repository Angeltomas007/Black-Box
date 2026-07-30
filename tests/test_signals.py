from __future__ import annotations

from blackbox.alpha.signals import MeanReversionSignal, MomentumSignal, PriceActionConfluenceSignal


def test_mean_reversion_fires_on_genuinely_mean_reverting_series(mean_reverting_df):
    signal = MeanReversionSignal(lookback=60, entry_z=1.5)
    result = signal.generate(mean_reverting_df)
    assert (result.side != 0).any()
    assert set(result.side.unique()).issubset({-1, 0, 1})


def test_mean_reversion_stays_flat_on_pure_trend(trending_df):
    """A one-directional trend should rarely satisfy the ADF
    stationarity gate, so the strategy should mostly stay flat rather
    than repeatedly fading a trend."""
    signal = MeanReversionSignal(lookback=60, entry_z=1.5)
    result = signal.generate(trending_df)
    active_fraction = (result.side != 0).mean()
    assert active_fraction < 0.5


def test_momentum_side_matches_trend_direction(trending_df):
    signal = MomentumSignal(fast=10, slow=50)
    result = signal.generate(trending_df)
    tail = result.side.dropna().iloc[-20:]
    assert (tail == 1).mean() > 0.8  # uptrend -> mostly long


def test_momentum_strength_is_non_negative(trending_df):
    result = MomentumSignal().generate(trending_df)
    assert (result.strength.dropna() >= 0).all()


def test_price_action_confluence_strength_is_continuous_not_constant(mean_reverting_df):
    signal = PriceActionConfluenceSignal(htf_rule="1D")
    result = signal.generate(mean_reverting_df)
    active = result.strength[result.side != 0]
    if len(active) > 1:
        # a real confluence score should vary with how extreme the
        # triggering conditions were, not collapse to one constant
        assert active.nunique() > 1
    assert ((result.strength >= 0) & (result.strength <= 1)).all()


def test_price_action_confluence_side_values_valid(mean_reverting_df):
    result = PriceActionConfluenceSignal(htf_rule="1D").generate(mean_reverting_df)
    assert set(result.side.unique()).issubset({-1, 0, 1})
