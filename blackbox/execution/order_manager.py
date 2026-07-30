"""Translates a risk-approved target position into concrete broker
orders, and is the single choke point through which every order must
pass -- including the emergency liquidation path.
"""

from __future__ import annotations

import logging

from blackbox.execution.broker import BrokerAPI, OrderResult, OrderSide

logger = logging.getLogger(__name__)


class OrderManager:
    def __init__(self, broker: BrokerAPI):
        self.broker = broker

    def rebalance_to_target(
        self, symbol: str, current_quantity: float, target_quantity: float
    ) -> OrderResult | None:
        """Send the single order needed to move from the current
        position to the target position (netting, not stacking)."""
        delta = target_quantity - current_quantity
        if abs(delta) < 1e-9:
            return None
        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        return self.broker.submit_order(symbol, side, abs(delta))

    def emergency_flatten(self) -> None:
        logger.critical("Emergency flatten: cancelling orders and liquidating all positions")
        self.broker.cancel_all_orders()
        self.broker.liquidate_all()
