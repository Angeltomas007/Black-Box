"""The main decision loop: data -> alpha -> risk -> execution, wired
together exactly as laid out in the architecture (Narang ch.2's
four-module decomposition). This is intentionally the only file that
imports from every other module -- everything else stays decoupled
behind the abstractions defined there.
"""

from __future__ import annotations

import asyncio
import logging

import pandas as pd

from blackbox.alpha.features import average_true_range, build_feature_matrix, realized_volatility
from blackbox.alpha.labeling import apply_triple_barrier, meta_label
from blackbox.alpha.model import MetaLabelingModel
from blackbox.alpha.signals import MeanReversionSignal, MomentumSignal
from blackbox.config import EngineConfig
from blackbox.data.bars import get_daily_vol
from blackbox.data.market_data import MarketDataProvider
from blackbox.data.storage import RedisStore
from blackbox.execution.broker import BrokerAPI, PaperBroker
from blackbox.execution.order_manager import OrderManager
from blackbox.notifications.discord_bot import DiscordNotifier
from blackbox.risk.position_sizing import confidence_scaled_size
from blackbox.risk.risk_manager import RiskLimits, RiskManager

logger = logging.getLogger(__name__)


class TradingEngine:
    def __init__(
        self,
        config: EngineConfig,
        data_provider: MarketDataProvider,
        broker: BrokerAPI,
        redis_store: RedisStore | None = None,
    ):
        self.config = config
        self.data_provider = data_provider
        self.broker = broker
        self.order_manager = OrderManager(broker)
        self.redis = redis_store
        self.notifier = DiscordNotifier(
            config.notifications.discord_webhook_url, config.notifications.enabled
        )

        self.mean_reversion = MeanReversionSignal(
            lookback=config.alpha.lookback,
            entry_z=config.alpha.zscore_entry,
            exit_z=config.alpha.zscore_exit,
        )
        self.momentum = MomentumSignal()
        self.meta_models: dict[str, MetaLabelingModel] = {}

        account = broker.get_account()
        self.risk_manager = RiskManager(
            limits=RiskLimits(
                max_position_pct=config.risk.max_position_pct,
                max_gross_leverage=config.risk.max_gross_leverage,
                atr_stop_multiple=config.risk.atr_stop_multiple,
                max_daily_drawdown=config.risk.max_daily_drawdown,
                max_total_drawdown=config.risk.max_total_drawdown,
                risk_per_trade=config.risk.risk_per_trade,
            ),
            starting_equity=account.equity,
        )

    # -- Model training --------------------------------------------------

    def train_meta_model(self, symbol: str, history: pd.DataFrame) -> None:
        """Fit the secondary (meta-labeling) model for one symbol using
        purged cross-validation, per AFML ch.7. Call this offline /
        periodically, not on every loop iteration."""
        primary = self.mean_reversion.generate(history).side
        primary = primary[primary != 0]
        if len(primary) < 100:
            logger.warning("Not enough primary signals to train meta-model for %s", symbol)
            return

        daily_vol = get_daily_vol(history["close"])
        barriers = apply_triple_barrier(history["close"], primary.index, daily_vol)
        labels = meta_label(primary, barriers)

        features = build_feature_matrix(history, lookback=self.config.alpha.lookback)
        aligned = features.reindex(labels.index).dropna()
        labels = labels.reindex(aligned.index)

        if len(aligned) < 50:
            logger.warning("Not enough aligned samples to train meta-model for %s", symbol)
            return

        model = MetaLabelingModel()
        label_end_times = barriers.reindex(aligned.index)["t1"]
        result = model.fit_with_purged_cv(aligned, labels, label_end_times)
        logger.info("Meta-model for %s: OOS accuracy=%.3f", symbol, result.oos_accuracy)
        self.meta_models[symbol] = model

    # -- Single decision cycle --------------------------------------------------

    def process_symbol(self, symbol: str, history: pd.DataFrame) -> None:
        if not self.risk_manager.can_trade():
            logger.info("Kill switch active; skipping %s", symbol)
            return

        mr_signal = self.mean_reversion.generate(history)
        mom_signal = self.momentum.generate(history)

        last_mr_side = int(mr_signal.side.iloc[-1])
        last_mom_side = int(mom_signal.side.iloc[-1])
        primary_side = last_mr_side if last_mr_side != 0 else last_mom_side

        price = float(history["close"].iloc[-1])
        atr = float(average_true_range(history).iloc[-1])
        annual_vol = float(realized_volatility(history["close"]).iloc[-1])

        if isinstance(self.broker, PaperBroker):
            self.broker.mark_price(symbol, price)
        account = self.broker.get_account()

        confidence = 0.5
        model = self.meta_models.get(symbol)
        if model is not None and primary_side != 0:
            features = build_feature_matrix(history, lookback=self.config.alpha.lookback)
            latest_features = features.iloc[[-1]].dropna(axis=1)
            try:
                confidence = float(model.predict_confidence(latest_features).iloc[-1])
            except Exception:
                logger.exception("Meta-model inference failed for %s; using neutral confidence", symbol)

        target_quantity = 0.0
        if primary_side != 0 and confidence >= self.config.alpha.meta_label_min_proba:
            raw_quantity = confidence_scaled_size(
                capital=account.equity,
                price=price,
                predicted_proba=confidence,
                win_loss_ratio=1.0,
                kelly_cap=self.config.risk.kelly_cap,
                max_position_pct=self.config.risk.max_position_pct,
            )
            target_quantity = primary_side * raw_quantity
            target_quantity = self.risk_manager.clip_order_size(
                symbol, target_quantity, price, account.equity
            )

        if self.risk_manager.check_stop_triggered(symbol, price):
            logger.info("Stop-loss triggered for %s at %.4f", symbol, price)
            target_quantity = 0.0

        current_positions = self.broker.get_positions()
        current_quantity = current_positions.get(symbol, 0.0)

        result = self.order_manager.rebalance_to_target(symbol, current_quantity, target_quantity)
        if result is not None:
            self.notifier.send_trade(symbol, result.side.value, result.quantity, result.filled_price or price)
            if target_quantity == 0:
                self.risk_manager.close_position(symbol)
            else:
                self.risk_manager.register_fill(
                    symbol, 1 if target_quantity > 0 else -1, abs(target_quantity), price, atr
                )

        self.risk_manager.update_equity(self.broker.get_account().equity)
        if self.risk_manager.kill_switch_active:
            self.notifier.send_kill_switch("Drawdown limit breached")
            self.order_manager.emergency_flatten()
            if self.redis:
                self.redis.set_kill_switch(True)

    # -- Main loop --------------------------------------------------

    async def run_forever(self) -> None:
        while True:
            if self.redis and self.redis.is_kill_switch_active():
                self.risk_manager.trigger_kill_switch("External kill switch (Discord)")

            for symbol in self.config.universe:
                try:
                    history = self.data_provider.get_historical_bars(
                        symbol, start="500d ago", bar_size=self.config.data.bar_size
                    )
                    self.process_symbol(symbol, history)
                except Exception:
                    logger.exception("Error processing %s", symbol)

            account = self.broker.get_account()
            positions = self.broker.get_positions()
            if self.redis:
                self.redis.set_equity(account.equity)
                self.redis.set_positions(positions)
            self.notifier.send_pnl_update(account.equity, account.equity - self.risk_manager.starting_equity, positions)

            await asyncio.sleep(self.config.poll_interval_seconds)
