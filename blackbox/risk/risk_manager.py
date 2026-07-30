"""Risk management: the layer that sits between "the model wants this
trade" and "an order is sent" (Narang ch.5). Nothing reaches the
execution module without passing through here, including the
kill-switch check on every single loop iteration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    max_position_pct: float = 0.20
    max_gross_leverage: float = 2.0
    atr_stop_multiple: float = 2.5
    atr_tp_multiple: float = 5.0
    max_daily_drawdown: float = 0.03
    max_total_drawdown: float = 0.15
    risk_per_trade: float = 0.0025


@dataclass
class PositionState:
    symbol: str
    side: int  # -1, 0, +1
    quantity: float
    entry_price: float
    stop_price: float
    take_profit_price: float | None = None


class RiskManager:
    """Tracks equity, enforces drawdown-based kill switches, computes
    ATR-based dynamic stops, and vetoes/clips any proposed order that
    would breach a hard risk limit."""

    def __init__(self, limits: RiskLimits, starting_equity: float):
        self.limits = limits
        self.starting_equity = starting_equity
        self.peak_equity = starting_equity
        self.day_start_equity = starting_equity
        self.equity_curve: list[float] = [starting_equity]
        self.positions: dict[str, PositionState] = {}
        self._kill_switch = False

    # -- Kill switch & drawdown -------------------------------------------------

    @property
    def kill_switch_active(self) -> bool:
        return self._kill_switch

    def trigger_kill_switch(self, reason: str) -> None:
        logger.critical("KILL SWITCH TRIGGERED: %s", reason)
        self._kill_switch = True

    def reset_kill_switch(self) -> None:
        self._kill_switch = False

    def update_equity(self, equity: float) -> None:
        self.equity_curve.append(equity)
        self.peak_equity = max(self.peak_equity, equity)

        total_dd = 1 - equity / self.peak_equity
        daily_dd = 1 - equity / self.day_start_equity if self.day_start_equity > 0 else 0.0

        if total_dd >= self.limits.max_total_drawdown:
            self.trigger_kill_switch(
                f"Total drawdown {total_dd:.2%} >= limit {self.limits.max_total_drawdown:.2%}"
            )
        elif daily_dd >= self.limits.max_daily_drawdown:
            self.trigger_kill_switch(
                f"Daily drawdown {daily_dd:.2%} >= limit {self.limits.max_daily_drawdown:.2%}"
            )

    def start_new_trading_day(self, equity: float) -> None:
        self.day_start_equity = equity

    # -- Dynamic stop loss -------------------------------------------------

    def compute_stop_price(self, entry_price: float, atr: float, side: int) -> float:
        """ATR-based dynamic stop: wider stops in higher-volatility
        regimes rather than a fixed percentage (Chan ch.3)."""
        offset = self.limits.atr_stop_multiple * atr
        return entry_price - offset if side > 0 else entry_price + offset

    def check_stop_triggered(self, symbol: str, current_price: float) -> bool:
        pos = self.positions.get(symbol)
        if pos is None or pos.side == 0:
            return False
        if pos.side > 0:
            return current_price <= pos.stop_price
        return current_price >= pos.stop_price

    def compute_take_profit_price(self, entry_price: float, atr: float, side: int) -> float:
        """ATR-based take-profit, symmetric to the stop: the reward
        leg of the risk/reward ratio, sized off the same volatility
        estimate rather than a fixed pip/percent target."""
        offset = self.limits.atr_tp_multiple * atr
        return entry_price + offset if side > 0 else entry_price - offset

    def check_take_profit_triggered(self, symbol: str, current_price: float) -> bool:
        pos = self.positions.get(symbol)
        if pos is None or pos.side == 0 or pos.take_profit_price is None:
            return False
        if pos.side > 0:
            return current_price >= pos.take_profit_price
        return current_price <= pos.take_profit_price

    # -- Order-level risk checks -------------------------------------------------

    def clip_order_size(
        self, symbol: str, proposed_quantity: float, price: float, equity: float
    ) -> float:
        """Enforce the hard per-position and gross-leverage caps
        regardless of what the sizing model recommends."""
        max_position_value = equity * self.limits.max_position_pct
        max_quantity = max_position_value / price if price > 0 else 0.0

        gross_exposure = sum(
            abs(p.quantity) * price for p in self.positions.values()
        ) + abs(proposed_quantity) * price
        if gross_exposure > equity * self.limits.max_gross_leverage:
            allowed_extra = max(
                0.0, equity * self.limits.max_gross_leverage - gross_exposure + abs(proposed_quantity) * price
            )
            max_quantity = min(max_quantity, allowed_extra / price if price > 0 else 0.0)

        clipped = min(abs(proposed_quantity), max_quantity)
        return clipped if proposed_quantity >= 0 else -clipped

    def register_fill(
        self, symbol: str, side: int, quantity: float, price: float, atr: float
    ) -> None:
        stop = self.compute_stop_price(price, atr, side)
        take_profit = self.compute_take_profit_price(price, atr, side)
        self.positions[symbol] = PositionState(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=price,
            stop_price=stop,
            take_profit_price=take_profit,
        )

    def close_position(self, symbol: str) -> None:
        self.positions.pop(symbol, None)

    def can_trade(self) -> bool:
        return not self._kill_switch
