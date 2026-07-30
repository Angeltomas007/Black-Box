"""Engine-local state, mirrored into Redis so the Discord bot and any
monitoring dashboard can read it without a direct line into the engine
process."""

from __future__ import annotations

from dataclasses import dataclass, field

from blackbox.data.storage import RedisStore


@dataclass
class EngineState:
    equity: float
    positions: dict[str, float] = field(default_factory=dict)
    daily_start_equity: float = 0.0

    def __post_init__(self) -> None:
        if self.daily_start_equity == 0.0:
            self.daily_start_equity = self.equity

    @property
    def daily_pnl(self) -> float:
        return self.equity - self.daily_start_equity

    def sync_to_redis(self, redis_store: RedisStore) -> None:
        redis_store.set_equity(self.equity)
        redis_store.set_positions(self.positions)
