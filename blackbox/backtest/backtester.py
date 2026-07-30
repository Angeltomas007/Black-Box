"""Walk-forward, cost-aware backtester.

Three rules from Chan ("Quantitative Trading") this engine follows
strictly:
  1. Never fit a model on data it will later be tested against
     (walk-forward, expanding window, refit periodically).
  2. Every fill pays realistic commission + slippage; a strategy that
     only works with zero transaction costs is not a strategy.
  3. Report the full risk profile (Sharpe, Sortino, max drawdown,
     turnover) -- a headline return number alone is close to
     meaningless for judging a systematic strategy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from blackbox.alpha.signals import MeanReversionSignal, MomentumSignal


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    returns: pd.Series
    sharpe: float
    sortino: float
    max_drawdown: float
    cagr: float
    turnover: float


def _performance_stats(equity_curve: pd.Series, periods_per_year: int = 252) -> dict[str, float]:
    returns = equity_curve.pct_change().dropna()
    ann_return = returns.mean() * periods_per_year
    ann_vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(periods_per_year) if len(downside) else np.nan
    sortino = ann_return / downside_vol if downside_vol and downside_vol > 0 else 0.0

    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    max_dd = drawdown.min()

    n_years = len(equity_curve) / periods_per_year
    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / n_years) - 1 if n_years > 0 else 0.0

    return {
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(max_dd),
        "cagr": float(cagr),
    }


def run_backtest(
    df: pd.DataFrame,
    starting_capital: float = 1_000_000.0,
    commission_bps: float = 0.5,
    slippage_bps: float = 1.0,
    lookback: int = 60,
) -> BacktestResult:
    """Single-asset vectorized backtest of the combined mean-reversion /
    momentum signal, charging costs on every position change. This is a
    research tool for signal validation, not a substitute for the
    event-driven simulation an execution-quality review would need.
    """
    mr = MeanReversionSignal(lookback=lookback).generate(df).side
    mom = MomentumSignal().generate(df).side
    combined_side = mr.where(mr != 0, mom)

    positions = combined_side.shift(1).fillna(0)  # trade on next bar's open, not same-bar close
    returns = df["close"].pct_change().fillna(0)
    gross_pnl = positions * returns

    turnover = positions.diff().abs().fillna(0)
    cost = turnover * (commission_bps + slippage_bps) / 10_000
    net_pnl = gross_pnl - cost

    equity_curve = starting_capital * (1 + net_pnl).cumprod()
    stats = _performance_stats(equity_curve)

    return BacktestResult(
        equity_curve=equity_curve,
        returns=net_pnl,
        sharpe=stats["sharpe"],
        sortino=stats["sortino"],
        max_drawdown=stats["max_drawdown"],
        cagr=stats["cagr"],
        turnover=float(turnover.mean()),
    )


def walk_forward_windows(
    n_obs: int, train_size: int, test_size: int
) -> list[tuple[slice, slice]]:
    """Expanding-window walk-forward split: train on [0, i), test on
    [i, i+test_size). Never look backward across a refit boundary in a
    way that would let the test window influence the training window.
    """
    windows = []
    start = train_size
    while start + test_size <= n_obs:
        train_slice = slice(0, start)
        test_slice = slice(start, start + test_size)
        windows.append((train_slice, test_slice))
        start += test_size
    return windows
