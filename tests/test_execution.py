from __future__ import annotations

import pytest

from blackbox.execution.broker import OrderSide, PaperBroker
from blackbox.execution.order_manager import OrderManager


def test_paper_broker_applies_slippage_in_the_adverse_direction():
    broker = PaperBroker(starting_cash=100_000.0, commission_bps=0.0, slippage_bps=100.0)  # 1%
    broker.mark_price("AAA", 100.0)
    buy = broker.submit_order("AAA", OrderSide.BUY, 10)
    assert buy.filled_price > 100.0  # buys fill worse (higher) than mid

    broker.mark_price("AAA", 100.0)
    sell = broker.submit_order("AAA", OrderSide.SELL, 10)
    assert sell.filled_price < 100.0  # sells fill worse (lower) than mid


def test_paper_broker_commission_reduces_cash():
    broker = PaperBroker(starting_cash=100_000.0, commission_bps=100.0, slippage_bps=0.0)  # 1%
    broker.mark_price("AAA", 100.0)
    broker.submit_order("AAA", OrderSide.BUY, 10)
    # 10 * 100 = 1000 notional + 1% commission = 1010 total cash outflow
    assert broker.cash == pytest.approx(100_000.0 - 1010.0)


def test_paper_broker_liquidate_all_flattens_positions():
    broker = PaperBroker(starting_cash=100_000.0)
    broker.mark_price("AAA", 50.0)
    broker.submit_order("AAA", OrderSide.BUY, 20)
    assert broker.get_positions()["AAA"] == pytest.approx(20.0)

    broker.liquidate_all()
    assert broker.get_positions().get("AAA", 0.0) == pytest.approx(0.0)


def test_order_manager_rebalance_nets_a_single_order():
    broker = PaperBroker(starting_cash=100_000.0)
    broker.mark_price("AAA", 100.0)
    manager = OrderManager(broker)

    result = manager.rebalance_to_target("AAA", current_quantity=0.0, target_quantity=50.0)
    assert result is not None
    assert result.side == OrderSide.BUY
    assert result.quantity == pytest.approx(50.0)


def test_order_manager_rebalance_noop_when_already_at_target():
    broker = PaperBroker(starting_cash=100_000.0)
    manager = OrderManager(broker)
    result = manager.rebalance_to_target("AAA", current_quantity=50.0, target_quantity=50.0)
    assert result is None


def test_order_manager_emergency_flatten_cancels_and_liquidates():
    broker = PaperBroker(starting_cash=100_000.0)
    broker.mark_price("AAA", 100.0)
    broker.submit_order("AAA", OrderSide.BUY, 10)
    manager = OrderManager(broker)

    manager.emergency_flatten()
    assert broker.get_positions().get("AAA", 0.0) == pytest.approx(0.0)
