from __future__ import annotations

import pytest

from blackbox.risk.risk_manager import RiskLimits, RiskManager


def test_register_fill_sets_symmetric_atr_stop_and_take_profit():
    rm = RiskManager(limits=RiskLimits(atr_stop_multiple=2.0, atr_tp_multiple=4.0), starting_equity=100_000.0)
    rm.register_fill("AAA", side=1, quantity=10, price=100.0, atr=1.0)
    pos = rm.positions["AAA"]
    assert pos.stop_price == pytest.approx(98.0)
    assert pos.take_profit_price == pytest.approx(104.0)


def test_stop_and_take_profit_trigger_correctly_for_short():
    rm = RiskManager(limits=RiskLimits(atr_stop_multiple=2.0, atr_tp_multiple=4.0), starting_equity=100_000.0)
    rm.register_fill("BBB", side=-1, quantity=5, price=50.0, atr=0.5)

    assert rm.check_stop_triggered("BBB", 51.5) is True
    assert rm.check_stop_triggered("BBB", 50.5) is False
    assert rm.check_take_profit_triggered("BBB", 47.5) is True
    assert rm.check_take_profit_triggered("BBB", 49.0) is False


def test_total_drawdown_triggers_kill_switch():
    rm = RiskManager(
        limits=RiskLimits(max_total_drawdown=0.10, max_daily_drawdown=0.99),
        starting_equity=100_000.0,
    )
    rm.update_equity(95_000.0)
    assert rm.kill_switch_active is False
    rm.update_equity(89_000.0)
    assert rm.kill_switch_active is True
    assert rm.can_trade() is False


def test_daily_drawdown_triggers_kill_switch_independent_of_peak():
    rm = RiskManager(limits=RiskLimits(max_daily_drawdown=0.02, max_total_drawdown=0.99), starting_equity=100_000.0)
    rm.start_new_trading_day(100_000.0)
    rm.update_equity(97_500.0)
    assert rm.kill_switch_active is True


def test_clip_order_size_enforces_max_position_pct():
    rm = RiskManager(limits=RiskLimits(max_position_pct=0.10, max_gross_leverage=10.0), starting_equity=100_000.0)
    clipped = rm.clip_order_size("AAA", proposed_quantity=1000, price=100.0, equity=100_000.0)
    # 10% of 100k = 10k notional -> 100 shares at $100
    assert clipped == pytest.approx(100.0)


def test_clip_order_size_enforces_gross_leverage_across_positions():
    rm = RiskManager(limits=RiskLimits(max_position_pct=1.0, max_gross_leverage=1.0), starting_equity=100_000.0)
    rm.register_fill("AAA", side=1, quantity=900, price=100.0, atr=1.0)  # $90k already deployed
    clipped = rm.clip_order_size("BBB", proposed_quantity=500, price=100.0, equity=100_000.0)
    # gross leverage cap of 1x equity leaves only $10k of room
    assert clipped == pytest.approx(100.0)


def test_clip_order_size_preserves_sign():
    rm = RiskManager(limits=RiskLimits(max_position_pct=0.10), starting_equity=100_000.0)
    clipped = rm.clip_order_size("AAA", proposed_quantity=-1000, price=100.0, equity=100_000.0)
    assert clipped < 0
