"""Triple-barrier labeling and meta-labeling (Lopez de Prado, AFML ch.3).

Fixed-horizon labeling (label = sign of return over the next N bars)
ignores path: a trade that hits a stop-loss intraday but closes flat by
the horizon gets mislabeled as a scratch. The triple barrier method
labels each observation by whichever of three barriers is touched
first: an upper barrier (profit-take), a lower barrier (stop-loss), or
a vertical barrier (max holding period) -- which is what actually
happens to a position under real risk management.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def apply_triple_barrier(
    close: pd.Series,
    events: pd.DatetimeIndex,
    daily_vol: pd.Series,
    pt_sl: tuple[float, float] = (1.0, 1.0),
    max_holding_days: int = 5,
) -> pd.DataFrame:
    """For each event timestamp, find the first barrier touched.

    Returns a DataFrame indexed by event time with columns:
    ``t1`` (barrier touch time), ``ret`` (realized return), ``label``
    (+1 / -1 / 0).
    """
    out = pd.DataFrame(index=events, columns=["t1", "ret", "label"])
    vertical_barriers = close.index.searchsorted(events + pd.Timedelta(days=max_holding_days))
    vertical_barriers = np.minimum(vertical_barriers, len(close.index) - 1)

    for i, t0 in enumerate(events):
        vol = daily_vol.get(t0, np.nan)
        if pd.isna(vol) or vol <= 0:
            continue
        t1 = close.index[vertical_barriers[i]]
        path = close.loc[t0:t1]
        if len(path) < 2:
            continue

        path_returns = path / close.loc[t0] - 1
        upper, lower = pt_sl[0] * vol, -pt_sl[1] * vol

        touches = path_returns[(path_returns > upper) | (path_returns < lower)]
        touch_time = touches.index[0] if len(touches) else t1
        realized_ret = path_returns.loc[touch_time]

        out.loc[t0, "t1"] = touch_time
        out.loc[t0, "ret"] = realized_ret
        out.loc[t0, "label"] = np.sign(realized_ret) if touch_time != t1 else np.sign(realized_ret)

    return out.dropna()


def meta_label(primary_side: pd.Series, triple_barrier: pd.DataFrame) -> pd.Series:
    """Meta-labeling (AFML ch.3): given a primary model's directional
    call (``primary_side`` in {-1, +1}), label 1 if the trade would have
    been profitable and 0 otherwise. A secondary ML model is then
    trained to predict this binary outcome, which lets the risk layer
    size (or skip) trades by predicted confidence instead of taking
    every primary signal at face value.
    """
    aligned = triple_barrier.reindex(primary_side.index).dropna()
    realized_sign = np.sign(aligned["ret"].astype(float))
    side = primary_side.reindex(aligned.index)
    return (realized_sign == side).astype(int)
