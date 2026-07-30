"""Meta-labeling ML layer with purged, embargoed cross-validation
(Lopez de Prado, AFML ch.3 and ch.7).

Financial labels overlap in time (a triple-barrier label spans multiple
bars), so a standard k-fold split leaks information: the training set
can contain observations whose label window overlaps the test set's,
inflating validation performance. PurgedKFold removes ("purges")
training samples whose evaluation window overlaps the test fold, and
adds an embargo period after each test fold to further block leakage
from serial correlation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import BaseCrossValidator


class PurgedKFold(BaseCrossValidator):
    """K-fold CV where training observations overlapping a test
    sample's label window are purged, plus an embargo after each test
    fold (AFML ch.7)."""

    def __init__(self, n_splits: int, label_end_times: pd.Series, embargo_pct: float = 0.01):
        self.n_splits = n_splits
        self.label_end_times = label_end_times  # t1 per observation, indexed like X
        self.embargo_pct = embargo_pct

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None):
        indices = np.arange(len(X))
        embargo = int(len(X) * self.embargo_pct)
        fold_bounds = np.array_split(indices, self.n_splits)

        for test_idx in fold_bounds:
            test_start, test_end = test_idx[0], test_idx[-1]
            t0 = self.label_end_times.index[test_start]
            t1 = self.label_end_times.iloc[test_end]

            train_mask = np.ones(len(X), dtype=bool)
            train_mask[test_start: test_end + 1] = False

            label_ends = self.label_end_times.array
            label_starts = self.label_end_times.index.array
            overlaps = (label_starts <= t1) & (label_ends >= t0)
            train_mask &= ~overlaps

            embargo_end = min(test_end + 1 + embargo, len(X))
            train_mask[test_end + 1: embargo_end] = False

            yield indices[train_mask], test_idx


@dataclass
class MetaLabelModelResult:
    model: BaseEstimator
    oos_accuracy: float
    feature_importances: pd.Series


class MetaLabelingModel:
    """Secondary classifier that predicts whether the primary signal's
    call will be profitable. Its predicted probability becomes the bet
    sizing input in the risk layer (probability -> Kelly fraction),
    exactly the "primary model for direction, secondary model for bet
    size" architecture AFML advocates."""

    def __init__(self, n_estimators: int = 200, max_depth: int = 4, min_samples_leaf: int = 50):
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )

    def fit_with_purged_cv(
        self, X: pd.DataFrame, y: pd.Series, label_end_times: pd.Series, n_splits: int = 5
    ) -> MetaLabelModelResult:
        cv = PurgedKFold(n_splits=n_splits, label_end_times=label_end_times)
        accuracies = []
        for train_idx, test_idx in cv.split(X):
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            self.clf.fit(X.iloc[train_idx], y.iloc[train_idx])
            accuracies.append(self.clf.score(X.iloc[test_idx], y.iloc[test_idx]))

        # Final fit on the full history for live inference.
        self.clf.fit(X, y)
        importances = pd.Series(self.clf.feature_importances_, index=X.columns).sort_values(
            ascending=False
        )
        return MetaLabelModelResult(
            model=self.clf,
            oos_accuracy=float(np.mean(accuracies)) if accuracies else float("nan"),
            feature_importances=importances,
        )

    def predict_confidence(self, X: pd.DataFrame) -> pd.Series:
        """P(trade is profitable | features), used to size the bet."""
        proba = self.clf.predict_proba(X)
        classes = list(self.clf.classes_)
        positive_idx = classes.index(1) if 1 in classes else -1
        return pd.Series(proba[:, positive_idx], index=X.index)
