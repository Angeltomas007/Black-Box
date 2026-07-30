"""Broker adapters. The engine only ever talks to the ``BrokerAPI``
interface, so swapping Alpaca for Interactive Brokers, or dropping in
the paper broker for dry runs, never touches alpha/risk code.
"""

from __future__ import annotations

import abc
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class OrderResult:
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    status: str
    filled_price: float | None = None


@dataclass
class Account:
    equity: float
    cash: float
    buying_power: float


class BrokerAPI(abc.ABC):
    @abc.abstractmethod
    def submit_order(self, symbol: str, side: OrderSide, quantity: float) -> OrderResult:
        ...

    @abc.abstractmethod
    def get_positions(self) -> dict[str, float]:
        ...

    @abc.abstractmethod
    def get_account(self) -> Account:
        ...

    @abc.abstractmethod
    def cancel_all_orders(self) -> None:
        ...

    @abc.abstractmethod
    def liquidate_all(self) -> None:
        """Flatten every open position immediately -- the action
        actually taken when the risk layer trips the kill switch."""


class PaperBroker(BrokerAPI):
    """In-memory simulated broker for research and dry-run deployments.
    Applies a simple slippage + commission model so PnL is not
    unrealistically clean (Chan ch.5's warning against backtests that
    assume frictionless fills)."""

    def __init__(self, starting_cash: float, commission_bps: float = 0.5, slippage_bps: float = 1.0):
        self.cash = starting_cash
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        self._positions: dict[str, float] = {}
        self._last_prices: dict[str, float] = {}

    def mark_price(self, symbol: str, price: float) -> None:
        self._last_prices[symbol] = price

    def submit_order(self, symbol: str, side: OrderSide, quantity: float) -> OrderResult:
        price = self._last_prices.get(symbol)
        if price is None:
            raise ValueError(f"No mark price for {symbol}; call mark_price() first")

        slip = price * self.slippage_bps / 10_000
        fill_price = price + slip if side == OrderSide.BUY else price - slip
        notional = fill_price * quantity
        commission = notional * self.commission_bps / 10_000

        signed_qty = quantity if side == OrderSide.BUY else -quantity
        self._positions[symbol] = self._positions.get(symbol, 0.0) + signed_qty
        self.cash -= signed_qty * fill_price + commission

        logger.info(
            "PAPER FILL %s %s %.4f @ %.4f (commission %.4f)",
            side.value, symbol, quantity, fill_price, commission,
        )
        return OrderResult(
            order_id=str(uuid.uuid4()),
            symbol=symbol,
            side=side,
            quantity=quantity,
            status="filled",
            filled_price=fill_price,
        )

    def get_positions(self) -> dict[str, float]:
        return dict(self._positions)

    def get_account(self) -> Account:
        positions_value = sum(
            qty * self._last_prices.get(sym, 0.0) for sym, qty in self._positions.items()
        )
        equity = self.cash + positions_value
        return Account(equity=equity, cash=self.cash, buying_power=self.cash)

    def cancel_all_orders(self) -> None:
        return  # paper fills are immediate; nothing resting to cancel

    def liquidate_all(self) -> None:
        for symbol, qty in list(self._positions.items()):
            if qty == 0:
                continue
            side = OrderSide.SELL if qty > 0 else OrderSide.BUY
            self.submit_order(symbol, side, abs(qty))


class AlpacaBroker(BrokerAPI):
    """Live/paper execution via Alpaca's trading API."""

    def __init__(self, api_key: str, api_secret: str, base_url: str):
        from alpaca.trading.client import TradingClient

        self._client = TradingClient(
            api_key, api_secret, paper="paper-api" in base_url
        )

    def submit_order(self, symbol: str, side: OrderSide, quantity: float) -> OrderResult:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide as AlpacaSide, TimeInForce

        request = MarketOrderRequest(
            symbol=symbol,
            qty=quantity,
            side=AlpacaSide.BUY if side == OrderSide.BUY else AlpacaSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        order = self._client.submit_order(request)
        return OrderResult(
            order_id=str(order.id),
            symbol=symbol,
            side=side,
            quantity=quantity,
            status=order.status.value,
            filled_price=float(order.filled_avg_price) if order.filled_avg_price else None,
        )

    def get_positions(self) -> dict[str, float]:
        return {p.symbol: float(p.qty) for p in self._client.get_all_positions()}

    def get_account(self) -> Account:
        acct = self._client.get_account()
        return Account(
            equity=float(acct.equity), cash=float(acct.cash), buying_power=float(acct.buying_power)
        )

    def cancel_all_orders(self) -> None:
        self._client.cancel_orders()

    def liquidate_all(self) -> None:
        self._client.close_all_positions(cancel_orders=True)
