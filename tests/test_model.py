from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from blackbox.alpha.model import MetaLabelingModel, PurgedKFold


def _overlapping_label_end_times(n: int, overlap_bars: int = 5) -> pd.Series:
    """Each observation's label spans ``overlap_bars`` bars into the
    future -- exactly the kind of overlap that would leak into a naive
    k-fold split."""
    idx = pd.date_range("2023-01-01", periods=n, freq="1D", tz="UTC")
    t1 = idx[np.minimum(np.arange(n) + overlap_bars, n - 1)]
    return pd.Series(t1, index=idx)


def test_purged_kfold_removes_overlapping_training_observations():
    n = 200
    label_end_times = _overlapping_label_end_times(n, overlap_bars=10)
    X = pd.DataFrame({"f": np.arange(n)}, index=label_end_times.index)

    cv = PurgedKFold(n_splits=5, label_end_times=label_end_times, embargo_pct=0.01)

    for train_idx, test_idx in cv.split(X):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        test_start_time = label_end_times.index[test_idx[0]]
        test_end_time = label_end_times.iloc[test_idx[-1]]

        train_label_starts = label_end_times.index[train_idx]
        train_label_ends = label_end_times.iloc[train_idx]

        # no training observation's label window may overlap the test
        # fold's [start, end] span
        overlaps = (train_label_starts <= test_end_time) & (train_label_ends >= test_start_time)
        assert not overlaps.any()


def test_purged_kfold_embargo_excludes_bars_immediately_after_test_fold():
    n = 100
    label_end_times = _overlapping_label_end_times(n, overlap_bars=1)
    X = pd.DataFrame({"f": np.arange(n)}, index=label_end_times.index)

    cv = PurgedKFold(n_splits=4, label_end_times=label_end_times, embargo_pct=0.05)
    embargo = int(n * 0.05)

    for train_idx, test_idx in cv.split(X):
        test_end = test_idx[-1]
        embargo_range = set(range(test_end + 1, min(test_end + 1 + embargo, n)))
        assert not (embargo_range & set(train_idx))


def test_meta_labeling_model_fit_predict_shapes():
    n = 300
    rng = np.random.default_rng(0)
    idx = pd.date_range("2023-01-01", periods=n, freq="1D", tz="UTC")
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)}, index=idx)
    y = pd.Series(rng.integers(0, 2, n), index=idx)
    label_end_times = pd.Series(idx, index=idx)

    model = MetaLabelingModel(n_estimators=20, min_samples_leaf=5)
    result = model.fit_with_purged_cv(X, y, label_end_times, n_splits=3)

    assert 0.0 <= result.oos_accuracy <= 1.0 or np.isnan(result.oos_accuracy)
    proba = model.predict_confidence(X.tail(5))
    assert len(proba) == 5
    assert ((proba >= 0) & (proba <= 1)).all()
