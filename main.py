"""Entry point.

    python main.py backtest --symbol SPY --start 2018-01-01
    python main.py live
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from blackbox.config import CONFIG
from blackbox.data.market_data import AlpacaDataProvider, YFinanceProvider
from blackbox.execution.broker import AlpacaBroker, PaperBroker
from blackbox.data.storage import RedisStore

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
)
logger = logging.getLogger("blackbox.main")


def build_data_provider():
    if CONFIG.data.provider == "alpaca":
        return AlpacaDataProvider(
            CONFIG.execution.alpaca_key_id, CONFIG.execution.alpaca_secret_key
        )
    return YFinanceProvider()


def build_broker():
    if CONFIG.execution.broker == "alpaca":
        return AlpacaBroker(
            CONFIG.execution.alpaca_key_id,
            CONFIG.execution.alpaca_secret_key,
            CONFIG.execution.alpaca_base_url,
        )
    broker = PaperBroker(
        starting_cash=CONFIG.risk.capital,
        commission_bps=CONFIG.execution.commission_bps,
        slippage_bps=CONFIG.execution.slippage_bps,
    )
    return broker


def cmd_backtest(args: argparse.Namespace) -> None:
    from blackbox.backtest.backtester import run_backtest

    provider = build_data_provider()
    df = provider.get_historical_bars(args.symbol, start=args.start, bar_size="1D")
    result = run_backtest(
        df,
        starting_capital=CONFIG.risk.capital,
        commission_bps=CONFIG.execution.commission_bps,
        slippage_bps=CONFIG.execution.slippage_bps,
        lookback=CONFIG.alpha.lookback,
    )
    logger.info(
        "Backtest %s | Sharpe=%.2f Sortino=%.2f MaxDD=%.2f%% CAGR=%.2f%% Turnover=%.3f",
        args.symbol, result.sharpe, result.sortino, result.max_drawdown * 100,
        result.cagr * 100, result.turnover,
    )


def cmd_live(args: argparse.Namespace) -> None:
    from blackbox.core.engine import TradingEngine

    provider = build_data_provider()
    broker = build_broker()
    try:
        redis_store = RedisStore(CONFIG.data.redis_url)
    except Exception:
        logger.warning("Redis unavailable; running without external kill-switch control plane")
        redis_store = None

    if isinstance(broker, PaperBroker):
        for symbol in CONFIG.universe:
            history = provider.get_historical_bars(symbol, start="30d ago", bar_size="1D")
            broker.mark_price(symbol, float(history["close"].iloc[-1]))

    engine = TradingEngine(CONFIG, provider, broker, redis_store)

    for symbol in CONFIG.universe:
        try:
            history = provider.get_historical_bars(symbol, start="500d ago", bar_size="1D")
            engine.train_meta_model(symbol, history)
        except Exception:
            logger.exception("Failed to train meta-model for %s", symbol)

    asyncio.run(engine.run_forever())


def main() -> None:
    parser = argparse.ArgumentParser(description="Black-Box systematic trading engine")
    sub = parser.add_subparsers(dest="command", required=True)

    backtest_parser = sub.add_parser("backtest", help="Run a vectorized single-asset backtest")
    backtest_parser.add_argument("--symbol", required=True)
    backtest_parser.add_argument("--start", default="2018-01-01")
    backtest_parser.set_defaults(func=cmd_backtest)

    live_parser = sub.add_parser("live", help="Run the live/paper trading engine loop")
    live_parser.set_defaults(func=cmd_live)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
