"""Persistence layer: PostgreSQL for durable historical OHLCV storage,
Redis for low-latency real-time state (last prices, positions, the
kill-switch flag) and pub/sub between the Discord bot and the engine.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

BARS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS bars (
    symbol      TEXT NOT NULL,
    ts          TIMESTAMPTZ NOT NULL,
    open        DOUBLE PRECISION NOT NULL,
    high        DOUBLE PRECISION NOT NULL,
    low         DOUBLE PRECISION NOT NULL,
    close       DOUBLE PRECISION NOT NULL,
    volume      DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (symbol, ts)
);
"""


class PostgresStore:
    """Thin wrapper around psycopg2 for bar persistence. Kept dependency
    lazy (imported in __init__) so the rest of the codebase can be
    imported/tested without a live Postgres instance."""

    def __init__(self, dsn: str) -> None:
        import psycopg2

        self._conn = psycopg2.connect(dsn)
        self._conn.autocommit = True
        with self._conn.cursor() as cur:
            cur.execute(BARS_TABLE_DDL)

    def upsert_bars(self, symbol: str, df: pd.DataFrame) -> None:
        from psycopg2.extras import execute_values

        rows = [
            (symbol, ts.to_pydatetime(), r.open, r.high, r.low, r.close, r.volume)
            for ts, r in df.iterrows()
        ]
        with self._conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO bars (symbol, ts, open, high, low, close, volume)
                VALUES %s
                ON CONFLICT (symbol, ts) DO UPDATE SET
                    open = EXCLUDED.open, high = EXCLUDED.high,
                    low = EXCLUDED.low, close = EXCLUDED.close,
                    volume = EXCLUDED.volume
                """,
                rows,
            )

    def load_bars(self, symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
        query = "SELECT ts, open, high, low, close, volume FROM bars WHERE symbol = %s AND ts >= %s"
        params: list[Any] = [symbol, start]
        if end:
            query += " AND ts <= %s"
            params.append(end)
        query += " ORDER BY ts"
        df = pd.read_sql(query, self._conn, params=params, index_col="ts", parse_dates=["ts"])
        return df

    def close(self) -> None:
        self._conn.close()


class RedisStore:
    """Real-time key/value + pub/sub state. This is the shared control
    plane between the Discord command bot and the trading engine: the
    bot writes ``kill_switch=1`` here, the engine polls it every loop."""

    KILL_SWITCH_KEY = "blackbox:kill_switch"
    EQUITY_KEY = "blackbox:equity"
    POSITIONS_KEY = "blackbox:positions"
    COMMAND_CHANNEL = "blackbox:commands"

    def __init__(self, url: str) -> None:
        import redis

        self._client = redis.Redis.from_url(url, decode_responses=True)

    def set_kill_switch(self, active: bool) -> None:
        self._client.set(self.KILL_SWITCH_KEY, "1" if active else "0")

    def is_kill_switch_active(self) -> bool:
        return self._client.get(self.KILL_SWITCH_KEY) == "1"

    def set_equity(self, equity: float) -> None:
        self._client.set(self.EQUITY_KEY, str(equity))

    def get_equity(self) -> float | None:
        val = self._client.get(self.EQUITY_KEY)
        return float(val) if val is not None else None

    def set_positions(self, positions: dict[str, float]) -> None:
        self._client.set(self.POSITIONS_KEY, json.dumps(positions))

    def get_positions(self) -> dict[str, float]:
        raw = self._client.get(self.POSITIONS_KEY)
        return json.loads(raw) if raw else {}

    def publish_command(self, command: str) -> None:
        self._client.publish(self.COMMAND_CHANNEL, command)

    def subscribe_commands(self):
        pubsub = self._client.pubsub()
        pubsub.subscribe(self.COMMAND_CHANNEL)
        return pubsub
