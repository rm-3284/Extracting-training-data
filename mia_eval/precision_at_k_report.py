#!/usr/bin/env python3
"""
Precision@k tables for **each ranking method** (Carlini-style scores and MIA scalars).

Two modes:

1. **GPU-backed (optional):** reads ``carlini_scores.jsonl`` from ``run_carlini_table2 --scores-jsonl``
   for Carlini-ranked P@k plus MIA-ranked P@k.

2. **MIA-only, no GPU:** pass ``--mia-only`` and ``--samples-jsonl`` (e.g. ``samples_mia_gt.jsonl``).
   Uses ``mia_gt_primary`` scalars (and optional ``select_alignment_mc``).

   To **also** compute Carlini-ranked P@k without re-running the LM, pass ``--carlini-scores-jsonl``
   pointing at the file from ``run_carlini_table2 --scores-jsonl`` (often named
   ``carlini_table2_scores.jsonl``). Top-k rows for Perplexity / Small / zlib / … are taken from
   those saved per-row scores (same indices as ``i`` in the file).

When using scores-jsonl, integer ``label`` (0/1) can come from the scores file or ``samples-jsonl``.

Reports:

- **carlini_ranking_***: sort by each Carlini-family score, take top-k, measure fraction with
  ``y`` = true member label and/or proxy pseudo-label from ``--proxy-thresholds-json``.
- **mia_ranking_***: sort by each MIA scalar (infilling, wbc, memtrace; optional
  ``select_alignment_mc`` from samples), same P@k definitions.

Example::

  # Per model (after running Carlini scoring with --scores-jsonl):
  python -m mia_eval.precision_at_k_report \\
    --scores-jsonl mia_eval_outputs/qwen25_7b_base/carlini_scores.jsonl \\
    --samples-jsonl mia_eval_outputs/qwen25_7b_base/samples_mia_gt.jsonl \\
    --proxy-thresholds-json data/proxy_scored/qwen25_7b_base/proxy_thresholds_inf_wbc_mtfix.json \\
    --output mia_eval_outputs/qwen25_7b_base/precision_at_k_report.json

  # All models — Carlini + MIA (needs carlini_scores.jsonl per model):
  python -m mia_eval.precision_at_k_report \\
    --scan-carlini-glob 'mia_eval_outputs/*/carlini_scores.jsonl' \\
    --samples-glob 'mia_eval_outputs/{model}/samples_mia_gt.jsonl' \\
    --proxy-threshold-template 'data/proxy_scored/{model}/proxy_thresholds_inf_wbc_mtfix.json' \\
    --output mia_eval_outputs/precision_at_k_all_models.json

  # All Qwen models — MIA P@k + Carlini P@k using saved scores (no GPU):
  python -m mia_eval.precision_at_k_report \\
    --scan-samples-glob 'mia_eval_outputs/*/samples_mia_gt.jsonl' \\
    --carlini-scores-template 'mia_eval_outputs/{model}/carlini_table2_scores.jsonl' \\
    --proxy-threshold-template 'data/proxy_scored/{model}/proxy_thresholds_inf_wbc_mtfix.json' \\
    --output mia_eval_outputs/precision_at_k_hybrid_all_models.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mia_eval.evaluation_common import jsonable, precision_at_k

# Keys in --scores-jsonl from run_carlini_table2 (see that module).
CARLINI_METRICS: List[Tuple[str, str, bool]] = [
    ("Perplexity", "perplexity_target", True),
    ("Small", "small_log_ratio", True),
    ("zlib", "zlib_log_ratio", True),
    ("Lowercase", "lowercase_log_ratio", True),
    ("Window", "perplexity_window_min", True),
]

# MIA scalars: json field suffix after mia_gt_primary_ in scores jsonl, then orientation.
MIA_METRICS: List[Tuple[str, str, bool]] = [
    ("infilling", "mia_gt_primary_infilling", True),
    ("wbc", "mia_gt_primary_wbc", False),
    ("memtrace_p_member", "mia_gt_primary_memtrace_p_member", False),
]

SELECT_KEY_OUTER = "mia_gt_primary_select_alignment_mc"


def _load_proxy_thresholds_json(path: Path) -> Dict[str, Tuple[float, bool]]:
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("schema") != "mia_eval.proxy_thresholds.v1":
        raise ValueError(f"Expected mia_eval.proxy_thresholds.v1 in {path}")
    raw = doc.get("thresholds") or {}
    out: Dict[str, Tuple[float, bool]] = {}
    for key in ("memtrace_p_member", "wbc", "infilling"):
        b = raw.get(key)
        if not isinstance(b, dict):
            raise ValueError(f"Missing thresholds[{key!r}] in {path}")
        out[key] = (float(b["threshold"]), bool(b["higher_is_member"]))
    return out


def _proxy_column_labels(
    mia: Dict[str, np.ndarray],
    thr: Dict[str, Tuple[float, bool]],
) -> Dict[str, np.ndarray]:
    """Binary pseudo-labels (1 = member-like) per proxy channel."""
    out: Dict[str, np.ndarray] = {}
    for key, (t, hi) in thr.items():
        s = mia[key]
        if hi:
            out[key] = (s > t).astype(np.float64)
        else:
            out[key] = (s < t).astype(np.float64)
    return out


def _block_pk(
    rank_scores: np.ndarray,
    y: np.ndarray,
    ks: List[int],
    lower_better: bool,
) -> Dict[str, float]:
    row: Dict[str, float] = {}
    for k in ks:
        if k > len(rank_scores):
            row[f"@{k}"] = float("nan")
        else:
            row[f"@{k}"] = precision_at_k(rank_scores, y, k, lower_better=lower_better)
    return row


def _load_scores_and_optional_samples(
    scores_path: Path,
    samples_path: Optional[Path],
) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    with open(scores_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    n = len(rows)
    if n == 0:
        raise SystemExit(f"No rows in {scores_path}")

    def col_float(key: str) -> np.ndarray:
        a = np.full(n, np.nan, dtype=np.float64)
        for i, r in enumerate(rows):
            v = r.get(key)
            if v is None:
                continue
            try:
                a[i] = float(v)
            except (TypeError, ValueError):
                pass
        return a

    bundle: Dict[str, np.ndarray] = {}
    for _, field, _ in CARLINI_METRICS:
        bundle[field] = col_float(field)
    for name, field, _ in MIA_METRICS:
        bundle[field] = col_float(field)

    y_scores_file: Optional[np.ndarray] = None
    lab = col_float("label")
    if np.all(np.isfinite(lab)) and np.all((lab == 0) | (lab == 1)):
        y_scores_file = lab

    y_final: Optional[np.ndarray] = None
    if samples_path is not None and samples_path.is_file():
        syr: List[Optional[int]] = []
        sel: List[float] = []
        with open(samples_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if "label" in r and r["label"] is not None:
                    try:
                        syr.append(int(r["label"]))
                    except (TypeError, ValueError):
                        syr.append(None)
                else:
                    syr.append(None)
                gp = r.get("mia_gt_primary")
                if isinstance(gp, dict) and "select_alignment_mc" in gp:
                    try:
                        sel.append(float(gp["select_alignment_mc"]))
                    except (TypeError, ValueError):
                        sel.append(float("nan"))
                else:
                    sel.append(float("nan"))
        if len(syr) != n:
            raise SystemExit(f"Line count mismatch: {scores_path} ({n}) vs {samples_path} ({len(syr)})")
        if all(v is not None for v in syr):
            y_final = np.array([int(v) for v in syr], dtype=np.float64)
        if len(sel) == n and not all(np.isnan(sel)):
            bundle[SELECT_KEY_OUTER] = np.asarray(sel, dtype=np.float64)

    if y_final is None and y_scores_file is not None:
        y_final = y_scores_file

    return bundle, y_final, rows


def _load_samples_mia_only(
    samples_path: Path,
) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray]]:
    """
    Load only MIA scalars from ``samples_mia_gt``-style JSONL (no Carlini / no model forward).
    """
    rows: List[Dict[str, Any]] = []
    with open(samples_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    n = len(rows)
    if n == 0:
        raise SystemExit(f"No rows in {samples_path}")

    inf: List[float] = []
    wbc: List[float] = []
    mt: List[float] = []
    sel: List[float] = []
    lab_or_sentinel: List[int] = []
    for r in rows:
        gp = r.get("mia_gt_primary")
        if not isinstance(gp, dict) or not all(
            k in gp for k in ("infilling", "wbc", "memtrace_p_member")
        ):
            raise SystemExit(
                f"Each row needs mia_gt_primary with infilling, wbc, memtrace_p_member: {samples_path}"
            )
        try:
            inf.append(float(gp["infilling"]))
            wbc.append(float(gp["wbc"]))
            mt.append(float(gp["memtrace_p_member"]))
        except (TypeError, ValueError) as e:
            raise SystemExit(f"Non-numeric mia_gt_primary in {samples_path}: {e}") from e
        v = gp.get("select_alignment_mc")
        if v is not None:
            try:
                sel.append(float(v))
            except (TypeError, ValueError):
                sel.append(float("nan"))
        else:
            sel.append(float("nan"))
        if "label" in r and r["label"] is not None:
            try:
                lab_or_sentinel.append(int(r["label"]))
            except (TypeError, ValueError):
                lab_or_sentinel.append(-1)
        else:
            lab_or_sentinel.append(-1)

    bundle: Dict[str, np.ndarray] = {
        "mia_gt_primary_infilling": np.asarray(inf, dtype=np.float64),
        "mia_gt_primary_wbc": np.asarray(wbc, dtype=np.float64),
        "mia_gt_primary_memtrace_p_member": np.asarray(mt, dtype=np.float64),
    }
    if not all(np.isnan(sel)):
        bundle[SELECT_KEY_OUTER] = np.asarray(sel, dtype=np.float64)

    y_final: Optional[np.ndarray] = None
    if all(x in (0, 1) for x in lab_or_sentinel):
        y_final = np.asarray(lab_or_sentinel, dtype=np.float64)

    return bundle, y_final


def _load_carlini_columns_from_jsonl(path: Path) -> Tuple[Dict[str, np.ndarray], int]:
    """
    Load Carlini per-row scores as written by ``run_carlini_table2 --scores-jsonl``.
    If each object has ``i``, values are placed at that index (supports sparse / reorder).
    Otherwise rows are assumed order 0..n-1.
    """
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"No rows in {path}")
    has_i = [("i" in r and isinstance(r.get("i"), int)) for r in rows]
    if any(has_i) and not all(has_i):
        raise SystemExit(
            f"{path}: either every line must have integer 'i' or none; mixed handling is not supported."
        )
    use_i = bool(rows and has_i[0])
    if use_i:
        n = max(int(r["i"]) for r in rows) + 1
    else:
        n = len(rows)
    fields = [f for _, f, _ in CARLINI_METRICS]
    out: Dict[str, np.ndarray] = {f: np.full(n, np.nan, dtype=np.float64) for f in fields}
    for j, r in enumerate(rows):
        idx = int(r["i"]) if use_i else j
        if idx < 0 or idx >= n:
            raise SystemExit(f"{path}: line {j}: bad index i={idx} (n={n})")
        for field in fields:
            v = r.get(field)
            if v is None:
                continue
            try:
                out[field][idx] = float(v)
            except (TypeError, ValueError):
                pass
    return out, n


def _merge_carlini_scores_into_bundle(
    bundle: Dict[str, np.ndarray],
    carlini_path: Path,
) -> None:
    """In-place: add Carlini columns; require same n as existing bundle rows."""
    carlini_cols, n_c = _load_carlini_columns_from_jsonl(carlini_path)
    n_b = len(next(iter(bundle.values())))
    if n_c != n_b:
        raise SystemExit(
            f"Row count mismatch: MIA bundle has n={n_b} but {carlini_path} has n={n_c}. "
            "Use the same eval run / same line order."
        )
    for k, arr in carlini_cols.items():
        bundle[k] = arr


def build_report_for_arrays(
    bundle: Dict[str, np.ndarray],
    y_true: Optional[np.ndarray],
    proxy_thr: Optional[Dict[str, Tuple[float, bool]]],
    ks: List[int],
    *,
    include_carlini: bool = True,
) -> Dict[str, Any]:
    n = len(next(iter(bundle.values())))
    rep: Dict[str, Any] = {
        "n_samples": n,
        "k": ks,
        "has_true_labels": y_true is not None
        and len(y_true) == n
        and len(np.unique(y_true[~np.isnan(y_true)])) >= 2,
        "has_proxy_thresholds": proxy_thr is not None,
        "mia_only": not include_carlini,
    }

    y_proxy_by_metric: Optional[Dict[str, np.ndarray]] = None
    if proxy_thr is not None:
        mia = {
            "infilling": bundle["mia_gt_primary_infilling"],
            "wbc": bundle["mia_gt_primary_wbc"],
            "memtrace_p_member": bundle["mia_gt_primary_memtrace_p_member"],
        }
        y_proxy_by_metric = _proxy_column_labels(mia, proxy_thr)

    yt = y_true if rep["has_true_labels"] else None

    # --- Carlini score ranks (skip in --mia-only mode) ---
    carlini_true: Dict[str, Dict[str, float]] = {}
    carlini_proxy: Dict[str, Dict[str, Dict[str, float]]] = {}
    if include_carlini:
        for disp, field, lb in CARLINI_METRICS:
            if field not in bundle:
                continue
            s = bundle[field]
            if np.all(np.isnan(s)):
                continue
            if yt is not None:
                carlini_true[disp] = _block_pk(s, yt, ks, lb)
            if y_proxy_by_metric is not None:
                carlini_proxy[disp] = {}
                for pkname, ypv in y_proxy_by_metric.items():
                    carlini_proxy[disp][f"{pkname}_proxy"] = _block_pk(s, ypv, ks, lb)

    rep["carlini_ranking_true_labels"] = carlini_true or None
    rep["carlini_ranking_proxy_labels"] = carlini_proxy or None

    # --- MIA score ranks ---
    mia_true: Dict[str, Dict[str, float]] = {}
    mia_proxy: Dict[str, Dict[str, float]] = {}
    for disp, field, lb in MIA_METRICS:
        s = bundle[field]
        if np.all(np.isnan(s)):
            continue
        if yt is not None:
            mia_true[disp] = _block_pk(s, yt, ks, lb)
        if y_proxy_by_metric is not None:
            mia_proxy[disp] = {}
            for pkname, ypv in y_proxy_by_metric.items():
                mia_proxy[disp][f"{pkname}_proxy"] = _block_pk(s, ypv, ks, lb)

    sel_key = SELECT_KEY_OUTER
    if sel_key in bundle and not np.all(np.isnan(bundle[sel_key])):
        s = bundle[sel_key]
        higher_member = True
        if yt is not None:
            mia_true["select_alignment_mc"] = _block_pk(s, yt, ks, not higher_member)
        if y_proxy_by_metric is not None:
            mia_proxy["select_alignment_mc"] = {}
            for pkname, ypv in y_proxy_by_metric.items():
                mia_proxy["select_alignment_mc"][f"{pkname}_proxy"] = _block_pk(
                    s, ypv, ks, not higher_member
                )

    rep["mia_ranking_true_labels"] = mia_true or None
    rep["mia_ranking_proxy_labels"] = mia_proxy or None

    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description="Precision@k report from carlini_scores.jsonl.")
    ap.add_argument(
        "--mia-only",
        action="store_true",
        help="Use samples JSONL for MIA (required). No unified --scores-jsonl. GPU not required.",
    )
    ap.add_argument(
        "--carlini-scores-jsonl",
        type=str,
        default="",
        help=(
            "With --mia-only: path to Carlini per-row scores (e.g. carlini_table2_scores.jsonl). "
            "Top-k for Carlini metrics uses these values (field ``i`` = row index). "
            "MIA columns still come from --samples-jsonl."
        ),
    )
    ap.add_argument("--scores-jsonl", type=str, default="", help="From run_carlini_table2 --scores-jsonl.")
    ap.add_argument(
        "--samples-jsonl",
        type=str,
        default="",
        help="samples_mia_gt.jsonl (required for --mia-only; optional otherwise for labels / select).",
    )
    ap.add_argument(
        "--proxy-thresholds-json",
        type=str,
        default="",
        help="mia_eval.proxy_thresholds.v1 from compute_proxy_thresholds.",
    )
    ap.add_argument(
        "--precision-k",
        type=str,
        default="10,50,100",
        help="Comma-separated k (clamped to n samples).",
    )
    ap.add_argument("--output", type=str, default="", help="Write JSON summary.")
    ap.add_argument(
        "--scan-carlini-glob",
        type=str,
        default="",
        help="If set, run one report per matching carlini_scores.jsonl and aggregate.",
    )
    ap.add_argument(
        "--samples-glob",
        type=str,
        default="",
        help="Template with {model} for samples path when using --scan-carlini-glob.",
    )
    ap.add_argument(
        "--proxy-threshold-template",
        type=str,
        default="",
        help="Template with {model} for thresholds path when using --scan-carlini-glob.",
    )
    ap.add_argument(
        "--scan-samples-glob",
        type=str,
        default="",
        help="Glob of samples_mia_gt.jsonl paths; one report per file.",
    )
    ap.add_argument(
        "--carlini-scores-template",
        type=str,
        default="",
        help=(
            "With --scan-samples-glob: path template with {model} e.g. "
            "mia_eval_outputs/{model}/carlini_table2_scores.jsonl"
        ),
    )
    args = ap.parse_args()

    ks_all = [int(x.strip()) for x in args.precision_k.split(",") if x.strip().isdigit()]
    if not ks_all:
        raise SystemExit("No valid k in --precision-k")

    if args.scan_samples_glob:
        paths = sorted(glob.glob(args.scan_samples_glob))
        if not paths:
            raise SystemExit(f"No files matched: {args.scan_samples_glob}")
        combined: Dict[str, Any] = {"k": ks_all, "models": {}}
        for sp in paths:
            samples_path = Path(sp)
            model_key = samples_path.parent.name
            proxy_path: Optional[Path] = None
            if str(args.proxy_threshold_template).strip():
                pp = Path(str(args.proxy_threshold_template).format(model=model_key))
                proxy_path = pp if pp.is_file() else None

            carlini_path: Optional[Path] = None
            if str(args.carlini_scores_template).strip():
                cp = Path(str(args.carlini_scores_template).format(model=model_key))
                carlini_path = cp if cp.is_file() else None

            bundle, y_true = _load_samples_mia_only(samples_path)
            include_carlini = False
            if carlini_path is not None:
                _merge_carlini_scores_into_bundle(bundle, carlini_path)
                include_carlini = True
            proxy_thr = _load_proxy_thresholds_json(proxy_path) if proxy_path else None
            n = len(next(iter(bundle.values())))
            ks = [k for k in ks_all if k <= n]
            sub = build_report_for_arrays(
                bundle, y_true, proxy_thr, ks, include_carlini=include_carlini
            )
            sub["samples_jsonl"] = str(samples_path.resolve())
            sub["carlini_scores_jsonl"] = str(carlini_path.resolve()) if carlini_path else None
            sub["proxy_thresholds_json"] = str(proxy_path.resolve()) if proxy_path else None
            sub["mia_only_samples"] = True
            combined["models"][model_key] = sub

        out = Path(args.output) if args.output else Path("mia_eval_outputs/precision_at_k_mia_only_all_models.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(jsonable(combined), f, indent=2)
        msg = "MIA + Carlini (from saved scores)" if str(args.carlini_scores_template).strip() else "MIA-only"
        print(f"Wrote {out} ({len(combined['models'])} models, {msg})", file=sys.stderr)
        return

    if args.scan_carlini_glob:
        paths = sorted(glob.glob(args.scan_carlini_glob))
        if not paths:
            raise SystemExit(f"No files matched: {args.scan_carlini_glob}")
        combined: Dict[str, Any] = {"k": ks_all, "models": {}}
        for scores_p in paths:
            scores_path = Path(scores_p)
            model_key = scores_path.parent.name
            samples_path: Optional[Path] = None
            if str(args.samples_glob).strip():
                sp = Path(str(args.samples_glob).format(model=model_key))
                samples_path = sp if sp.is_file() else None
            proxy_path: Optional[Path] = None
            if str(args.proxy_threshold_template).strip():
                pp = Path(str(args.proxy_threshold_template).format(model=model_key))
                proxy_path = pp if pp.is_file() else None

            bundle, y_true, _ = _load_scores_and_optional_samples(scores_path, samples_path)
            proxy_thr = _load_proxy_thresholds_json(proxy_path) if proxy_path else None
            n = len(next(iter(bundle.values())))
            ks = [k for k in ks_all if k <= n]
            sub = build_report_for_arrays(bundle, y_true, proxy_thr, ks, include_carlini=True)
            sub["scores_jsonl"] = str(scores_path.resolve())
            sub["samples_jsonl"] = str(samples_path.resolve()) if samples_path else None
            sub["proxy_thresholds_json"] = str(proxy_path.resolve()) if proxy_path else None
            combined["models"][model_key] = sub

        out = Path(args.output) if args.output else Path("mia_eval_outputs/precision_at_k_all_models.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(jsonable(combined), f, indent=2)
        print(f"Wrote {out} ({len(combined['models'])} models)", file=sys.stderr)
        return

    if args.mia_only:
        if not str(args.samples_jsonl).strip():
            raise SystemExit("--mia-only requires --samples-jsonl.")
        if str(args.scores_jsonl).strip():
            raise SystemExit("With --mia-only use --carlini-scores-jsonl for Carlini columns, not --scores-jsonl.")
        samples_path = Path(args.samples_jsonl)
        proxy_path = Path(args.proxy_thresholds_json) if str(args.proxy_thresholds_json).strip() else None
        bundle, y_true = _load_samples_mia_only(samples_path)
        include_carlini = False
        carlini_path: Optional[Path] = None
        if str(args.carlini_scores_jsonl).strip():
            carlini_path = Path(args.carlini_scores_jsonl)
            if not carlini_path.is_file():
                raise SystemExit(f"Not found: {carlini_path}")
            _merge_carlini_scores_into_bundle(bundle, carlini_path)
            include_carlini = True
        proxy_thr = _load_proxy_thresholds_json(proxy_path) if proxy_path and proxy_path.is_file() else None
        n = len(next(iter(bundle.values())))
        ks = [k for k in ks_all if k <= n]
        rep = build_report_for_arrays(
            bundle, y_true, proxy_thr, ks, include_carlini=include_carlini
        )
        rep["samples_jsonl"] = str(samples_path.resolve())
        rep["carlini_scores_jsonl"] = str(carlini_path.resolve()) if carlini_path else None
        rep["proxy_thresholds_json"] = str(proxy_path.resolve()) if proxy_path and proxy_path.is_file() else None
        out = Path(args.output) if args.output else samples_path.parent / (
            "precision_at_k_mia_carlini.json" if include_carlini else "precision_at_k_mia_only.json"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(jsonable(rep), f, indent=2)
        tag = "MIA + Carlini (scores file)" if include_carlini else "MIA-only"
        print(f"Wrote {out} ({tag})", file=sys.stderr)
        return

    if not str(args.scores_jsonl).strip():
        raise SystemExit("Provide --scores-jsonl, --scan-carlini-glob, --scan-samples-glob, or --mia-only.")

    scores_path = Path(args.scores_jsonl)
    samples_path = Path(args.samples_jsonl) if str(args.samples_jsonl).strip() else None
    proxy_path = Path(args.proxy_thresholds_json) if str(args.proxy_thresholds_json).strip() else None

    bundle, y_true, _ = _load_scores_and_optional_samples(scores_path, samples_path)
    proxy_thr = _load_proxy_thresholds_json(proxy_path) if proxy_path and proxy_path.is_file() else None
    n = len(next(iter(bundle.values())))
    ks = [k for k in ks_all if k <= n]

    rep = build_report_for_arrays(bundle, y_true, proxy_thr, ks, include_carlini=True)
    rep["scores_jsonl"] = str(scores_path.resolve())
    rep["samples_jsonl"] = str(samples_path.resolve()) if samples_path else None
    rep["proxy_thresholds_json"] = str(proxy_path.resolve()) if proxy_path and proxy_path.is_file() else None

    out = Path(args.output) if args.output else scores_path.parent / "precision_at_k_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(jsonable(rep), f, indent=2)
    print(f"Wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
