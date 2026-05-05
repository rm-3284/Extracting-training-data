"""Aggregate WBC diagnostics (exact zeros vs ``n < 2`` / short-NLL path)."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def wbc_quantification_summary(
    wbc: np.ndarray,
    labels: Optional[np.ndarray],
    short: Optional[np.ndarray],
) -> Dict[str, Any]:
    """
    Parameters
    ----------
    wbc
        WBC scores (float).
    labels
        Optional binary proxy labels in ``{0, 1}`` (same length as ``wbc``).
    short
        Optional boolean mask: true when ``min(len(target_nll), len(ref_nll)) < 2``
        (score forced to ``0.0`` in ``scoring_wbc.score_texts``). If omitted, only
        exact-zero counts are reported.
    """
    n = int(wbc.size)
    zero = wbc == 0.0
    n_zero = int(zero.sum())
    out: Dict[str, Any] = {
        "n_rows": n,
        "n_wbc_exactly_zero": n_zero,
        "frac_wbc_exactly_zero": float(n_zero / n) if n else 0.0,
    }
    if short is not None and short.size == n:
        n_short = int(short.sum())
        n_short_zero = int((short & zero).sum())
        n_not_short_zero = int((~short & zero).sum())
        out.update(
            {
                "wbc_short_available": True,
                "n_wbc_short_nll": n_short,
                "frac_wbc_short_nll": float(n_short / n) if n else 0.0,
                "n_wbc_short_and_exactly_zero": n_short_zero,
                "n_wbc_exactly_zero_not_short": n_not_short_zero,
                "frac_wbc_exactly_zero_not_short": float(n_not_short_zero / n) if n else 0.0,
            }
        )
    else:
        out["wbc_short_available"] = False

    if labels is not None and labels.size == n:
        by_l: Dict[str, Any] = {}
        for lb in (0, 1):
            m = labels == lb
            cn = int(m.sum())
            if not cn:
                continue
            block: Dict[str, Any] = {
                "n": cn,
                "n_wbc_exactly_zero": int((zero & m).sum()),
                "frac_wbc_exactly_zero": float((zero & m).sum() / cn),
            }
            if short is not None and short.size == n:
                sm = short & m
                block["n_wbc_short_nll"] = int(sm.sum())
                block["frac_wbc_short_nll"] = float(sm.sum() / cn)
            by_l[str(lb)] = block
        if by_l:
            out["by_label"] = by_l
    return out
