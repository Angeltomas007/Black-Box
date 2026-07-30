"""Market data ingestion: a broker-agnostic interface for historical and
real-time bars, plus the cleaning routines that make raw ticks fit to
feed a statistical model (Narang ch.3: garbage in, garbage out).
"""

from __future__ import annotations

import abc
import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

_RELATIVE_START_RE = re.compile(r"^\s*(\d+)\s*d\s*ago\s*$", re.IGNORECASE)


def resolve_start_date(start: str) -> str:
    """Convert a "<N>d ago" convenience string into an actual
    "YYYY-MM-DD" date. yfinance's start= parameter only accepts real
    dates (or datetime objects) -- passing it the literal string
    "10d ago" doesn't raise, it just silently matches no rows, which
    surfaces downstream as a confusing "No data returned" error. Any
    string that isn't of this relative form is passed through
    unchanged (e.g. an already-real "2018-01-01")."""
    match = _RELATIVE_START_RE.match(start)
    if not match:
        return start
    days = int(match.group(1))
    return (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")


@dataclass(frozen=True)
class Bar:
    symbol: str
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float


class MarketDataProvider(abc.ABC):
    """Abstract source of OHLCV data. Swap implementations without
    touching the alpha/risk/execution layers -- the whole point of an
    adapter boundary around a broker's idiosyncratic API."""

    @abc.abstractmethod
    def get_historical_bars(
        self, symbol: str, start: str, end: str | None = None, bar_size: str = "1D"
    ) -> pd.DataFrame:
        """Return a cleaned OHLCV DataFrame indexed by UTC timestamp."""

    @abc.abstractmethod
    async def stream_realtime(self, symbols: list[str]) -> AsyncIterator[Bar]:
        """Yield bars as they close, for live/paper trading."""


class YFinanceProvider(MarketDataProvider):
    """Free, delayed data source used for research/backtesting when no
    brokerage credentials are configured. Never use for live execution."""

    def get_historical_bars(
        self, symbol: str, start: str, end: str | None = None, bar_size: str = "1D"
    ) -> pd.DataFrame:
        import yfinance as yf

        interval_map = {
            "1D": "1d", "1H": "1h", "1min": "1m",
            "5min": "5m", "15min": "15m", "30min": "30m",
        }
        if bar_size not in interval_map:
            raise ValueError(f"Unsupported bar_size {bar_size!r}; expected one of {sorted(interval_map)}")
        interval = interval_map[bar_size]
        resolved_start = resolve_start_date(start)
        raw = yf.download(
            symbol, start=resolved_start, end=end, interval=interval, progress=False, auto_adjust=True
        )
        if raw.empty:
            raise ValueError(f"No data returned for {symbol} between {resolved_start} and {end}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns=str.lower)[OHLCV_COLUMNS]
        raw.index = pd.to_datetime(raw.index, utc=True)
        return DataCleaner.clean(raw)

    async def stream_realtime(self, symbols: list[str]) -> AsyncIterator[Bar]:
        # yfinance has no real streaming endpoint; poll the last daily bar
        # as a placeholder so the engine's control flow is exercised end
        # to end in a paper/demo setting.
        while True:
            for symbol in symbols:
                df = self.get_historical_bars(symbol, start="30d ago", bar_size="1D")
                last = df.iloc[-1]
                yield Bar(
                    symbol=symbol,
                    timestamp=df.index[-1],
                    open=last.open,
                    high=last.high,
                    low=last.low,
                    close=last.close,
                    volume=last.volume,
                )
            await asyncio.sleep(60)


class AlpacaDataProvider(MarketDataProvider):
    """Production data source via Alpaca's market data API."""

    def __init__(self, api_key: str, api_secret: str) -> None:
        from alpaca.data.historical import StockHistoricalDataClient

        self._client = StockHistoricalDataClient(api_key, api_secret)

    def get_historical_bars(
        self, symbol: str, start: str, end: str | None = None, bar_size: str = "1D"
    ) -> pd.DataFrame:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        timeframe_map = {
            "1D": TimeFrame.Day, "1H": TimeFrame.Hour, "1min": TimeFrame.Minute,
            "5min": TimeFrame(5, TimeFrameUnit.Minute), "15min": TimeFrame(15, TimeFrameUnit.Minute),
            "30min": TimeFrame(30, TimeFrameUnit.Minute),
        }
        if bar_size not in timeframe_map:
            raise ValueError(f"Unsupported bar_size {bar_size!r}; expected one of {sorted(timeframe_map)}")
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe_map[bar_size],
            start=resolve_start_date(start),
            end=end,
        )
        bars = self._client.get_stock_bars(req).df
        bars = bars.reset_index(level=0, drop=True) if "symbol" in bars.index.names else bars
        bars = bars.rename(columns=str.lower)[OHLCV_COLUMNS]
        return DataCleaner.clean(bars)

    async def stream_realtime(self, symbols: list[str]) -> AsyncIterator[Bar]:
        from alpaca.data.live import StockDataStream

        queue: asyncio.Queue[Bar] = asyncio.Queue()
        stream = StockDataStream(self._client._api_key, self._client._secret_key)

        async def _handler(bar) -> None:
            await queue.put(
                Bar(
                    symbol=bar.symbol,
                    timestamp=pd.Timestamp(bar.timestamp, tz="UTC"),
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume,
                )
            )

        stream.subscribe_bars(_handler, *symbols)
        run_task = asyncio.create_task(stream._run_forever())
        try:
            while True:
                yield await queue.get()
        finally:
            run_task.cancel()


class DataCleaner:
    """Cleaning pipeline applied to every raw OHLCV frame before it
    reaches feature engineering: dedupe, gap handling, and outlier
    winsorization on returns (bad ticks distort volatility estimates
    far more than they distort the mean -- Lopez de Prado, AFML ch.2)."""

    @staticmethod
    def clean(df: pd.DataFrame, max_return_zscore: float = 8.0) -> pd.DataFrame:
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df = df.dropna(subset=["close"])
        df = df[(df[OHLCV_COLUMNS] > 0).all(axis=1)]

        returns = np.log(df["close"]).diff()
        # The baseline std must exclude the return under test -- a
        # rolling window that includes its own outlier inflates its own
        # std and can mask exactly the single-bar spike it's meant to
        # catch (a 50x price spike can score under an 8-sigma threshold
        # once it's allowed to dominate its own 20-bar window).
        rolling_std = returns.rolling(20, min_periods=5).std().shift(1)
        z = (returns / rolling_std.replace(0, np.nan)).abs()
        df = df[(z < max_return_zscore) | z.isna()]

        df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].ffill()
        df["volume"] = df["volume"].fillna(0)
        return df
