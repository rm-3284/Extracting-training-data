"""Random Forest + scaling membership classifier (paper Sec. 2, Membership Inference Classifier)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    _SKLEARN = True
except ImportError:
    _SKLEARN = False


@dataclass
class MemTraceClassifier:
    """Fitted pipeline: StandardScaler + RandomForestClassifier."""

    pipeline: Any
    feature_names: Optional[list] = None

    def predict_proba_member(self, X: np.ndarray) -> np.ndarray:
        """Return P(member | x) as the positive class probability (column index 1)."""
        proba = self.pipeline.predict_proba(X)
        return proba[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict(X)


def train_memtrace_classifier(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 200,
    max_depth: int = 8,
    min_samples_leaf: int = 4,
    feature_names: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Train memTrace-style RF with z-score standardization (paper: StandardScaler on train only).

    The paper uses 5-fold CV + RandomizedSearchCV; this helper uses a single stratified split
    and fixed hyperparameters for a simple reproducible baseline. Adjust ``n_estimators`` /
    ``max_depth`` to explore the paper's search ranges (100–400 trees, depth 3–10).
    """
    if not _SKLEARN:
        raise ImportError("train_memtrace_classifier requires scikit-learn (pip install scikit-learn)")

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    auc = float(roc_auc_score(y_te, proba))

    return {
        "classifier": MemTraceClassifier(pipeline=pipe, feature_names=feature_names),
        "auc_holdout": auc,
        "X_test": X_te,
        "y_test": y_te,
        "y_test_proba": proba,
    }
