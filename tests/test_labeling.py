from __future__ import annotations

import numpy as np
import pandas as pd

from blackbox.alpha.labeling import apply_triple_barrier, meta_label
from blackbox.data.bars import get_daily_vol


def test_triple_barrier_hits_upper_barrier_on_sustained_rally():
    idx = pd.date_range("2023-01-01", periods=30, freq="1D", tz="UTC")
    # flat for the vol estimation window, then a sharp sustained rally
    close = pd.Series([100.0] * 20 + [100 + i * 3 for i in range(1, 11)], index=idx)

    daily_vol = get_daily_vol(close).fillna(0.01)
    daily_vol[daily_vol == 0] = 0.01
    event = idx[[20]]

    result = apply_triple_barrier(close, event, daily_vol, pt_sl=(1.0, 1.0), max_holding_days=5)
    assert len(result) == 1
    assert result["label"].iloc[0] > 0


def test_triple_barrier_hits_lower_barrier_on_sustained_selloff():
    idx = pd.date_range("2023-01-01", periods=30, freq="1D", tz="UTC")
    close = pd.Series([100.0] * 20 + [100 - i * 3 for i in range(1, 11)], index=idx)

    daily_vol = get_daily_vol(close).fillna(0.01)
    daily_vol[daily_vol == 0] = 0.01
    event = idx[[20]]

    result = apply_triple_barrier(close, event, daily_vol, pt_sl=(1.0, 1.0), max_holding_days=5)
    assert len(result) == 1
    assert result["label"].iloc[0] < 0


def test_meta_label_agrees_when_primary_side_matches_realized_sign():
    idx = pd.date_range("2023-01-01", periods=3, freq="1D", tz="UTC")
    primary_side = pd.Series([1, -1, 1], index=idx)
    barriers = pd.DataFrame(
        {"t1": idx, "ret": [0.02, 0.02, -0.01]}, index=idx
    )
    labels = meta_label(primary_side, barriers)
    # side=+1 with ret=+0.02 -> correct call (1); side=-1 with ret=+0.02 -> wrong call (0);
    # side=+1 with ret=-0.01 -> wrong call (0)
    assert labels.tolist() == [1, 0, 0]
