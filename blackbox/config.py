"""Centralized configuration, loaded from environment variables.

Keeping every tunable in one immutable dataclass makes the engine's
behavior reproducible and auditable -- a hard requirement for anything
that touches live capital (cf. Chan, "Quantitative Trading", ch. on
infrastructure).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class DataConfig:
    provider: str = os.environ.get("DATA_PROVIDER", "yfinance")  # yfinance | alpaca
    bar_size: str = os.environ.get("BAR_SIZE", "1D")
    postgres_dsn: str = os.environ.get(
        "POSTGRES_DSN", "postgresql://blackbox:blackbox@localhost:5432/blackbox"
    )
    redis_url: str = os.environ.get("REDIS_URL", "redis://localhost:6379/0")


@dataclass(frozen=True)
class AlphaConfig:
    lookback: int = _int("ALPHA_LOOKBACK", 60)
    zscore_entry: float = _float("ZSCORE_ENTRY", 2.0)
    zscore_exit: float = _float("ZSCORE_EXIT", 0.5)
    frac_diff_d: float = _float("FRAC_DIFF_D", 0.4)
    meta_label_min_proba: float = _float("META_LABEL_MIN_PROBA", 0.55)


@dataclass(frozen=True)
class RiskConfig:
    capital: float = _float("CAPITAL", 1_000_000.0)
    risk_per_trade: float = _float("RISK_PER_TRADE", 0.0025)  # 25 bps of equity
    target_annual_vol: float = _float("TARGET_ANNUAL_VOL", 0.10)
    max_position_pct: float = _float("MAX_POSITION_PCT", 0.20)
    max_gross_leverage: float = _float("MAX_GROSS_LEVERAGE", 2.0)
    atr_stop_multiple: float = _float("ATR_STOP_MULTIPLE", 2.5)
    max_daily_drawdown: float = _float("MAX_DAILY_DRAWDOWN", 0.03)
    max_total_drawdown: float = _float("MAX_TOTAL_DRAWDOWN", 0.15)
    kelly_cap: float = _float("KELLY_CAP", 0.5)  # half-Kelly


@dataclass(frozen=True)
class ExecutionConfig:
    broker: str = os.environ.get("BROKER", "paper")  # paper | alpaca | ibkr
    alpaca_key_id: str = os.environ.get("ALPACA_KEY_ID", "")
    alpaca_secret_key: str = os.environ.get("ALPACA_SECRET_KEY", "")
    alpaca_base_url: str = os.environ.get(
        "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
    )
    commission_bps: float = _float("COMMISSION_BPS", 0.5)
    slippage_bps: float = _float("SLIPPAGE_BPS", 1.0)


@dataclass(frozen=True)
class NotificationConfig:
    discord_webhook_url: str = os.environ.get("DISCORD_WEBHOOK_URL", "")
    discord_bot_token: str = os.environ.get("DISCORD_BOT_TOKEN", "")
    discord_alert_channel_id: str = os.environ.get("DISCORD_ALERT_CHANNEL_ID", "")
    enabled: bool = _bool("NOTIFICATIONS_ENABLED", True)


@dataclass(frozen=True)
class EngineConfig:
    universe: tuple[str, ...] = tuple(
        s.strip() for s in os.environ.get("UNIVERSE", "SPY,QQQ,IWM").split(",") if s.strip()
    )
    poll_interval_seconds: int = _int("POLL_INTERVAL_SECONDS", 60)
    data: DataConfig = field(default_factory=DataConfig)
    alpha: AlphaConfig = field(default_factory=AlphaConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)


CONFIG = EngineConfig()
