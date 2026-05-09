"""Shared helpers for train/val/test splits and AUC scoring (used by run_pipeline and hp_transfer)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def auc_direction(y: np.ndarray, s: np.ndarray) -> tuple[float, np.ndarray]:
    """Return max(AUC(s), AUC(-s)) and the score vector that achieved it."""
    s = np.asarray(s, dtype=np.float64)
    a0 = float(roc_auc_score(y, s))
    a1 = float(roc_auc_score(y, -s))
    if a1 > a0:
        return a1, -s
    return a0, s


def orient_scores_full(
    y: np.ndarray, mask: np.ndarray, scores_full: np.ndarray
) -> np.ndarray:
    """Flip sign on all samples if -score has higher AUC than score on mask (higher = member)."""
    s = np.asarray(scores_full, dtype=np.float64)
    ym, sm = y[mask], s[mask]
    if ym.size == 0 or len(np.unique(ym)) < 2:
        return s
    try:
        a0 = float(roc_auc_score(ym, sm))
        a1 = float(roc_auc_score(ym, -sm))
        return -s if a1 > a0 else s
    except ValueError:
        return s


def precision_at_k(
    scores: np.ndarray,
    labels: np.ndarray,
    k: int,
    *,
    lower_better: bool,
) -> float:
    """
    Fraction of positives among the top-k samples when sorting by ``scores``.
    ``labels``: binary (1 = member / positive class for the metric).
    ``lower_better``: if True, sort ascending (smallest scores first).
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if k <= 0 or k > len(scores):
        return float("nan")
    if lower_better:
        top_k_idx = np.argsort(scores)[:k]
    else:
        top_k_idx = np.argsort(scores)[::-1][:k]
    return float(labels[top_k_idx].mean())


def split_masks(
    y: np.ndarray,
    test_fraction: float,
    val_fraction: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(y)
    idx = np.arange(n)
    idx_train, idx_test = train_test_split(
        idx,
        test_size=test_fraction,
        stratify=y,
        random_state=random_state,
    )
    rel_val = val_fraction / (1.0 - test_fraction)
    idx_train, idx_val = train_test_split(
        idx_train,
        test_size=rel_val,
        stratify=y[idx_train],
        random_state=random_state,
    )
    tr = np.zeros(n, dtype=bool)
    va = np.zeros(n, dtype=bool)
    te = np.zeros(n, dtype=bool)
    tr[idx_train] = True
    va[idx_val] = True
    te[idx_test] = True
    return tr, va, te


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
