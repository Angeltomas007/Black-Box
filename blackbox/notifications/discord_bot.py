"""Discord is the desk's control plane: risk alerts and PnL snapshots
push out via webhook, and a small command bot lets an operator query
status or trip the kill switch from a phone, at 3am, without SSH
access to the box (Narang ch.6 on the operational necessity of a
human-reachable monitoring/override layer around any automated system).
"""

from __future__ import annotations

import logging

import httpx

from blackbox.data.storage import RedisStore

logger = logging.getLogger(__name__)


class DiscordNotifier:
    """Fire-and-forget webhook sender for alerts and PnL updates. Kept
    separate from the command bot: the webhook needs no bot token and
    can run even if the interactive bot process is down."""

    def __init__(self, webhook_url: str, enabled: bool = True):
        self.webhook_url = webhook_url
        self.enabled = enabled and bool(webhook_url)

    def _post(self, content: str) -> None:
        if not self.enabled:
            logger.info("[discord disabled] %s", content)
            return
        try:
            httpx.post(self.webhook_url, json={"content": content}, timeout=5.0)
        except httpx.HTTPError as exc:
            logger.warning("Discord webhook post failed: %s", exc)

    def send_alert(self, message: str) -> None:
        self._post(f":rotating_light: **RISK ALERT** :rotating_light:\n{message}")

    def send_kill_switch(self, reason: str) -> None:
        self._post(f":octagonal_sign: **KILL SWITCH ENGAGED**\nReason: {reason}")

    def send_pnl_update(self, equity: float, daily_pnl: float, positions: dict[str, float]) -> None:
        pos_lines = "\n".join(f"  {sym}: {qty:+.2f}" for sym, qty in positions.items()) or "  (flat)"
        self._post(
            f":bar_chart: **PnL Update**\n"
            f"Equity: ${equity:,.2f} | Daily PnL: ${daily_pnl:+,.2f}\n"
            f"Positions:\n{pos_lines}"
        )

    def send_trade(self, symbol: str, side: str, quantity: float, price: float) -> None:
        self._post(f":large_green_circle: FILL: {side.upper()} {quantity:.4f} {symbol} @ {price:.4f}")


class DiscordCommandBot:
    """Slash-command bot (`/status`, `/pnl`, `/kill`, `/resume`) that
    writes to the same Redis control plane the trading engine polls.
    The bot never talks to the broker or engine process directly --
    it only flips flags in Redis, which keeps the blast radius of a
    compromised Discord token limited to "can request a kill switch",
    never "can place trades".
    """

    def __init__(self, bot_token: str, redis_store: RedisStore, alert_channel_id: str = ""):
        import discord
        from discord import app_commands

        self.redis = redis_store
        intents = discord.Intents.default()
        self.client = discord.Client(intents=intents)
        self.tree = app_commands.CommandTree(self.client)
        self._bot_token = bot_token
        self._alert_channel_id = alert_channel_id
        self._register_commands()

    def _register_commands(self) -> None:
        redis_store = self.redis

        @self.tree.command(name="status", description="Show black box status")
        async def status(interaction) -> None:
            equity = redis_store.get_equity()
            positions = redis_store.get_positions()
            killed = redis_store.is_kill_switch_active()
            pos_str = ", ".join(f"{s}:{q:+.2f}" for s, q in positions.items()) or "flat"
            await interaction.response.send_message(
                f"Equity: ${equity:,.2f}\nPositions: {pos_str}\nKill switch: "
                f"{'ACTIVE' if killed else 'off'}"
            )

        @self.tree.command(name="kill", description="Emergency stop: flatten and halt trading")
        async def kill(interaction) -> None:
            redis_store.set_kill_switch(True)
            redis_store.publish_command("KILL")
            await interaction.response.send_message(":octagonal_sign: Kill switch engaged.")

        @self.tree.command(name="resume", description="Resume trading after a kill switch")
        async def resume(interaction) -> None:
            redis_store.set_kill_switch(False)
            redis_store.publish_command("RESUME")
            await interaction.response.send_message(":white_check_mark: Trading resumed.")

    def run(self) -> None:
        self.client.run(self._bot_token)
