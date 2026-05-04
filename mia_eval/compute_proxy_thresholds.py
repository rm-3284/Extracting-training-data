#!/usr/bin/env python3
"""
Build a small JSON file of **fixed thresholds** for ``mia_gt_primary`` scalars, from a scored
proxy JSONL with integer ``label`` (0/1).

For **infilling** and **wbc**, picks a threshold that **maximizes balanced accuracy** (same as
maximizing the average of class-specific recall) on the calibration sample, over candidate
cutpoints at midpoints between sorted unique scores. Rules: WBC / memtrace → predict member (1) if
**score > t**; infilling → member if **score < t** (lower infilling = more member-like).

For **memtrace_p_member**, use ``--memtrace-p-fixed`` (default **0.5**) — no data fit (and use
``--skip-memtrace`` on ``score_proxy_jsonl`` to avoid memTrace compute when building the calib file).

Pass the result to ``run_carlini_table2 --proxy-thresholds-json ...``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        from sklearn.metrics import balanced_accuracy_score

        return float(balanced_accuracy_score(y_true, y_pred))
    except Exception:
        yt = y_true.astype(np.int64)
        yp = y_pred.astype(np.int64)
        n0 = int((yt == 0).sum())
        n1 = int((yt == 1).sum())
        if n0 == 0 or n1 == 0:
            return 0.0
        tn = int(((yt == 0) & (yp == 0)).sum())
        tp = int(((yt == 1) & (yp == 1)).sum())
        return 0.5 * (tn / n0 + tp / n1)


def _best_threshold_balanced_accuracy(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    higher_is_member: bool,
) -> Optional[Tuple[float, float]]:
    """
    Return (threshold, balanced_accuracy_on_calibration) maximizing balanced accuracy.
    Candidates: midpoints between consecutive sorted unique scores (standard 1D ROC-style sweep).
    """
    y = labels.astype(np.int64)
    if len(np.unique(y)) < 2:
        return None
    s = np.asarray(scores, dtype=np.float64)
    uniq = np.unique(s)
    if len(uniq) < 2:
        return None
    candidates = [float((uniq[i] + uniq[i + 1]) / 2.0) for i in range(len(uniq) - 1)]

    best_t: Optional[float] = None
    best_ba = -1.0
    for t in candidates:
        if higher_is_member:
            pred = (s > t).astype(np.int64)
        else:
            pred = (s < t).astype(np.int64)
        if int(pred.sum()) in (0, len(pred)):
            continue
        ba = _balanced_accuracy(y, pred)
        if ba > best_ba + 1e-15:
            best_ba = ba
            best_t = t
        elif abs(ba - best_ba) <= 1e-15 and best_t is not None:
            if t < best_t:
                best_t = t
    if best_t is None:
        return None
    return best_t, float(best_ba)


def _midpoint_class_medians(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    higher_is_member: bool,
) -> Optional[Tuple[float, float]]:
    """Fallback when ROC-style sweep has no valid split (e.g. near-constant scores)."""
    y = labels.astype(np.int64)
    if len(np.unique(y)) < 2:
        return None
    s = np.asarray(scores, dtype=np.float64)
    m0 = float(np.median(s[y == 0]))
    m1 = float(np.median(s[y == 1]))
    t = (m0 + m1) / 2.0
    if higher_is_member:
        pred = (s > t).astype(np.int64)
    else:
        pred = (s < t).astype(np.int64)
    if int(pred.sum()) in (0, len(pred)):
        t = m1 if higher_is_member else m0
        if higher_is_member:
            pred = (s > t).astype(np.int64)
        else:
            pred = (s < t).astype(np.int64)
    if int(pred.sum()) in (0, len(pred)):
        smin, smax = float(np.min(s)), float(np.max(s))
        if smin == smax:
            return None
        t = smax if higher_is_member else smin
        if higher_is_member:
            pred = (s > t).astype(np.int64)
            if int(pred.sum()) in (0, len(pred)):
                t = smin - 1e-9
                pred = (s > t).astype(np.int64)
        else:
            pred = (s < t).astype(np.int64)
            if int(pred.sum()) in (0, len(pred)):
                t = smax + 1e-9
                pred = (s < t).astype(np.int64)
    if int(pred.sum()) in (0, len(pred)):
        return None
    ba = _balanced_accuracy(y, pred)
    return float(t), float(ba)


def _majority_class_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    higher_is_member: bool,
) -> Tuple[float, float]:
    """Last resort when scores are constant: pick t so the score rule matches majority vote; BA matches that vote."""
    y = labels.astype(np.int64)
    s = np.asarray(scores, dtype=np.float64)
    maj = 0 if int((y == 0).sum()) >= int((y == 1).sum()) else 1
    pred = np.full_like(y, maj)
    ba = _balanced_accuracy(y, pred)
    smin, smax = float(np.min(s)), float(np.max(s))
    eps = 1e-9 * max(1.0, abs(smax))
    if higher_is_member:
        t = (smin - eps) if maj == 1 else (smax + eps)
    else:
        t = (smax + eps) if maj == 1 else (smin - eps)
    return t, float(ba)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Write proxy MIA thresholds JSON from a scored JSONL with label 0/1."
    )
    ap.add_argument("--input", type=str, required=True, help="JSONL with mia_gt_primary + label.")
    ap.add_argument("--output", type=str, required=True, help="Output JSON path.")
    ap.add_argument(
        "--memtrace-p-fixed",
        type=float,
        default=0.5,
        help="Fixed memtrace_p_member threshold (higher_is_member=True). Skip data-driven memtrace.",
    )
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.is_file():
        raise SystemExit(f"Not found: {inp}")

    rows = _load_rows(inp)
    inf, wbc, lab = [], [], []
    for r in rows:
        gp = r.get("mia_gt_primary")
        if not isinstance(gp, dict):
            continue
        if not all(k in gp for k in ("infilling", "wbc")):
            continue
        try:
            y = int(r.get("label", -1))
        except (TypeError, ValueError):
            continue
        if y not in (0, 1):
            continue
        inf.append(float(gp["infilling"]))
        wbc.append(float(gp["wbc"]))
        lab.append(y)

    if len(lab) < 8:
        raise SystemExit(f"Need at least 8 rows with label 0/1 and mia_gt_primary; got {len(lab)}.")

    labels = np.asarray(lab, dtype=np.int64)
    out_thr: Dict[str, Any] = {}

    for name, arr, hi_mem in (
        ("wbc", np.asarray(wbc, dtype=np.float64), True),
        ("infilling", np.asarray(inf, dtype=np.float64), False),
    ):
        got = _best_threshold_balanced_accuracy(arr, labels, higher_is_member=hi_mem)
        method = "max_balanced_accuracy"
        if got is None:
            got = _midpoint_class_medians(arr, labels, higher_is_member=hi_mem)
            method = "midpoint_class_medians_fallback"
        if got is None and len(np.unique(arr)) == 1 and len(np.unique(labels)) == 2:
            t, ba = _majority_class_threshold(arr, labels, higher_is_member=hi_mem)
            got = (t, ba)
            method = "constant_scores_majority_baseline"
        if got is None:
            nu = int(len(np.unique(arr)))
            n0, n1 = int((labels == 0).sum()), int((labels == 1).sum())
            print(
                f"Warning: no threshold for {name} (unique_scores={nu}, n_label0={n0}, n_label1={n1}).",
                file=sys.stderr,
            )
            continue
        t, ba = got
        out_thr[name] = {
            "threshold": float(t),
            "higher_is_member": hi_mem,
            "method": method,
            "balanced_accuracy_on_calibration": float(ba),
        }

    if len(out_thr) < 2:
        n0, n1 = int((labels == 0).sum()), int((labels == 1).sum())
        raise SystemExit(
            "Could not derive infilling and wbc thresholds after max-BA and class-median fallback. "
            f"Check proxy labels (n0={n0}, n1={n1}) and that infilling/wbc vary in the merged JSONL."
        )

    mt_fix = float(args.memtrace_p_fixed)
    out_thr["memtrace_p_member"] = {
        "threshold": mt_fix,
        "higher_is_member": True,
        "method": "fixed_user",
    }

    payload: Dict[str, Any] = {
        "schema": "mia_eval.proxy_thresholds.v1",
        "description": (
            "infilling/wbc: threshold maximizing balanced accuracy on proxy stream labels; "
            f"memtrace_p_member: fixed {mt_fix} (higher_is_member). "
            "Eval: wbc/memtrace → y=1 if score > t; infilling → y=1 if score < t."
        ),
        "source_jsonl": str(inp.resolve()),
        "n_rows_used": len(lab),
        "thresholds": out_thr,
    }

    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps({"wrote": str(outp.resolve()), **{k: payload[k] for k in ("n_rows_used", "thresholds")}}, indent=2))


if __name__ == "__main__":
    main()
