"""Standalone demonstration of the price-action confluence signal,
end to end: real market data -> confluence signal -> ATR-based
stop/take-profit -> Discord dispatch.

This replaces an earlier template that generated synthetic OHLCV with
``np.random`` and, when no signal fired, silently dispatched a
fabricated "demo" trade indistinguishable from a real alert. Both are
operational hazards outside a research notebook: don't dress up random
noise as a market feed, and never let a placeholder trade reach the
same channel a real signal would use without being unmistakably
labeled as such.

Usage:
    python examples/demo_confluence_alert.py --symbol SPY
    python examples/demo_confluence_alert.py --symbol SPY --show-format
"""

from __future__ import annotations

import argparse
import logging
import sys

sys.path.insert(0, ".")

from blackbox.alpha.features import average_true_range
from blackbox.alpha.signals import PriceActionConfluenceSignal
from blackbox.data.market_data import YFinanceProvider
from blackbox.notifications.discord_bot import DiscordNotifier
from blackbox.risk.risk_manager import RiskLimits, RiskManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("demo_confluence_alert")


def build_alert_message(symbol: str, side: int, entry: float, stop: float, take_profit: float, confidence: float) -> str:
    direction = "LONG" if side > 0 else "SHORT"
    risk = abs(entry - stop)
    reward = abs(take_profit - entry)
    rr = reward / risk if risk > 0 else 0.0
    return (
        f":large_blue_circle: **SIGNAL DE CONFLUENCE PRICE ACTION**\n"
        f"----------------------------------------\n"
        f"- **Type** : `{direction}`\n"
        f"- **Actif** : `{symbol}`\n"
        f"- **Entrée** : `{entry:.4f}`\n"
        f"- **Stop-loss (ATR)** : `{stop:.4f}`\n"
        f"- **Take-profit (ATR)** : `{take_profit:.4f}`\n"
        f"- **Risque/Rendement** : `1:{rr:.2f}`\n"
        f"- **Score de confluence** : `{confidence * 100:.1f}%`\n"
        f"----------------------------------------"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Price-action confluence signal demo")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--bar-size", default="1H", choices=["1min", "1H", "1D"])
    parser.add_argument("--start", default="60d ago")
    parser.add_argument("--webhook-url", default="", help="Discord webhook; omitted = log only")
    parser.add_argument(
        "--show-format",
        action="store_true",
        help="If no live signal fires, print (never dispatch) an example of the alert "
        "format using the last bar's data, clearly marked as a simulation.",
    )
    args = parser.parse_args()

    provider = YFinanceProvider()
    df = provider.get_historical_bars(args.symbol, start=args.start, bar_size=args.bar_size)

    htf_rule = {"1min": "15min", "1H": "1D", "1D": "1W"}[args.bar_size]
    signal = PriceActionConfluenceSignal(htf_rule=htf_rule)
    result = signal.generate(df)

    side = int(result.side.iloc[-1])
    confidence = float(result.strength.iloc[-1])
    price = float(df["close"].iloc[-1])
    atr = float(average_true_range(df).iloc[-1])

    notifier = DiscordNotifier(args.webhook_url, enabled=bool(args.webhook_url))
    risk_manager = RiskManager(limits=RiskLimits(), starting_equity=1_000_000.0)

    if side == 0:
        logger.info("Aucune confluence validée pour %s à cette heure.", args.symbol)
        if args.show_format:
            stop = risk_manager.compute_stop_price(price, atr, side=1)
            take_profit = risk_manager.compute_take_profit_price(price, atr, side=1)
            message = build_alert_message(args.symbol, 1, price, stop, take_profit, confidence=0.5)
            print("\n:test_tube: **[SIMULATION - AUCUN SIGNAL RÉEL, EXEMPLE DE FORMAT UNIQUEMENT]**")
            print(message)
        return

    stop = risk_manager.compute_stop_price(price, atr, side)
    take_profit = risk_manager.compute_take_profit_price(price, atr, side)
    message = build_alert_message(args.symbol, side, price, stop, take_profit, confidence)
    logger.info("\n%s", message)
    notifier.send_alert(message)


if __name__ == "__main__":
    main()
