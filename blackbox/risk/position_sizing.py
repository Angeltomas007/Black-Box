"""Position sizing: volatility targeting and fractional Kelly.

Narang (ch.5) is explicit that sizing, not signal generation, is where
most of a systematic desk's edge-to-PnL conversion actually happens.
Two complementary sizers are provided: a volatility-target sizer for
baseline exposure, and a Kelly-based sizer that scales with the
meta-model's predicted confidence.
"""

from __future__ import annotations


def volatility_target_size(
    capital: float, target_annual_vol: float, asset_annual_vol: float, price: float
) -> float:
    """Number of shares/contracts such that this position's standalone
    volatility contribution matches ``target_annual_vol`` of capital."""
    if asset_annual_vol <= 0 or price <= 0:
        return 0.0
    dollar_size = capital * target_annual_vol / asset_annual_vol
    return dollar_size / price


def kelly_fraction(win_prob: float, win_loss_ratio: float, cap: float = 0.5) -> float:
    """Fractional Kelly bet size as a fraction of capital.

    f* = p - (1 - p) / b, where b is the win/loss payoff ratio.
    Full Kelly is provably growth-optimal but assumes known, stationary
    edge -- never true in markets -- so we cap at half-Kelly (or
    tighter) by default, standard practice per Chan ch.4.
    """
    if win_loss_ratio <= 0:
        return 0.0
    p = min(max(win_prob, 0.0), 1.0)
    f = p - (1 - p) / win_loss_ratio
    return max(0.0, min(f, cap))


def confidence_scaled_size(
    capital: float,
    price: float,
    predicted_proba: float,
    win_loss_ratio: float,
    kelly_cap: float,
    max_position_pct: float,
) -> float:
    """Combine meta-label confidence with a Kelly fraction, then clip to
    the hard per-position cap enforced regardless of model confidence."""
    fraction = kelly_fraction(predicted_proba, win_loss_ratio, cap=kelly_cap)
    fraction = min(fraction, max_position_pct)
    dollar_size = capital * fraction
    return dollar_size / price if price > 0 else 0.0
