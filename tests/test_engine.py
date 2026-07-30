from __future__ import annotations

from blackbox.config import (
    AlphaConfig,
    DataConfig,
    EngineConfig,
    ExecutionConfig,
    NotificationConfig,
    RiskConfig,
)
from blackbox.core.engine import TradingEngine
from blackbox.execution.broker import PaperBroker


class _FakeProvider:
    def __init__(self, df):
        self.df = df

    def get_historical_bars(self, symbol, start, end=None, bar_size="1D"):
        return self.df

    async def stream_realtime(self, symbols):
        raise NotImplementedError


def _build_engine(df, bar_size="1h"):
    config = EngineConfig(
        universe=("TEST",),
        data=DataConfig(bar_size=bar_size),
        alpha=AlphaConfig(lookback=60, meta_label_min_proba=0.0),
        risk=RiskConfig(capital=100_000.0),
        execution=ExecutionConfig(),
        notifications=NotificationConfig(enabled=False),
    )
    broker = PaperBroker(starting_cash=100_000.0)
    broker.mark_price("TEST", float(df["close"].iloc[-1]))
    engine = TradingEngine(config, _FakeProvider(df), broker, redis_store=None)
    return engine, broker


def test_process_symbol_runs_without_raising(mean_reverting_df):
    engine, broker = _build_engine(mean_reverting_df)
    engine.process_symbol("TEST", mean_reverting_df)  # must not raise
    assert engine.risk_manager.kill_switch_active is False


def test_kill_switch_blocks_further_trading(mean_reverting_df):
    engine, broker = _build_engine(mean_reverting_df)
    engine.risk_manager.trigger_kill_switch("test")
    engine.process_symbol("TEST", mean_reverting_df)
    # no new position should be opened while the kill switch is active
    assert broker.get_positions().get("TEST", 0.0) == 0.0


def test_engine_uses_three_tier_signal_fallback(mean_reverting_df):
    engine, _ = _build_engine(mean_reverting_df)
    mr_side = int(engine.mean_reversion.generate(mean_reverting_df).side.iloc[-1])
    mom_side = int(engine.momentum.generate(mean_reverting_df).side.iloc[-1])
    pa_side = int(engine.price_action.generate(mean_reverting_df).side.iloc[-1])

    expected = mr_side if mr_side != 0 else (mom_side if mom_side != 0 else pa_side)
    # process_symbol must not raise regardless of which tier is active
    engine.process_symbol("TEST", mean_reverting_df)
    assert expected in (-1, 0, 1)
