"""
Carlini-style generation + **reference MIA scores** (no shingle ground truth).

Each sample is annotated with two triples of method-native scores:
  - ``mia_gt_primary``: infilling + WBC at **open-model transferred** hyperparameters,
    memTrace ``p_member`` from a **pre-trained** ``*_memtrace_rf.joblib`` (Qwen proxy RF).
  - ``mia_gt_sensitivity``: same memTrace RF; infilling + WBC at an alternate HP combo.

Downstream ``mia_evaluate`` reports Spearman correlation matrices and score stability
between the two HP combos (not AUC vs training-data shingles).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from mia_eval.model_utils import load_causal_lm, pick_device, torch_dtype_from_str
from mia_eval.scoring_infilling import score_texts as infilling_scores
from mia_eval.scoring_memtrace import extract_memtrace_features_with_model
from mia_eval.scoring_wbc import score_texts as wbc_scores


def resolve_memtrace_rf_joblib(cfg: Dict[str, Any], model_key: str) -> Path:
    block = (cfg.get("mia_gt_pipeline") or {}) or {}
    raw = block.get("memtrace_rf_joblib")
    if raw:
        return Path(str(raw).format(active_model=model_key))
    base = Path(block.get("memtrace_rf_dir", "data/memtrace_rfs_qwen"))
    return base / f"{model_key}_memtrace_rf.joblib"


def _memtrace_p_batch(X: np.ndarray, joblib_path: Path) -> np.ndarray:
    import joblib

    obj: Any = joblib.load(joblib_path)
    # NumPy<2 does not support np.asarray(..., copy=...); use np.array for compatibility.
    Xs = np.array(X, dtype=np.float64, copy=True).reshape(X.shape[0], -1)
    np.nan_to_num(Xs, copy=False, nan=0.0, posinf=1e10, neginf=-1e10)
    if hasattr(obj, "predict_proba"):
        return np.asarray(obj.predict_proba(Xs)[:, 1], dtype=np.float64)
    if isinstance(obj, dict) and "scaler" in obj and "rf" in obj:
        X2 = obj["scaler"].transform(Xs)
        return np.asarray(obj["rf"].predict_proba(X2)[:, 1], dtype=np.float64)
    raise ValueError(
        f"Unsupported memtrace artifact {joblib_path}: expected Pipeline or dict scaler+rf."
    )


def _wbc_params_merge(cfg: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    m_wbc = (cfg.get("methods") or {}).get("wbc") or {}
    out = {k: m_wbc[k] for k in ("ensemble_variants", "use_ensemble", "ensemble_aggregate") if k in m_wbc}
    out.update(params)
    if out.get("window_sizes") is None:
        out.pop("window_sizes", None)
    return out


def load_samples_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def annotate_samples_mia_gt(
    cfg: Dict[str, Any],
    bundle: Dict[str, Any],
    model_key: str,
    samples_path: Path,
    out_path: Path,
) -> int:
    """
    Read ``samples.jsonl``, write ``samples_mia_gt.jsonl`` with ``mia_gt_primary`` /
    ``mia_gt_sensitivity`` triples.
    """
    block = (cfg.get("mia_gt_pipeline") or {}) or {}
    rows = load_samples_jsonl(samples_path)
    texts = [str(r.get("text", "")) for r in rows]
    if not texts:
        raise RuntimeError(f"No rows in {samples_path}")

    rf_path = resolve_memtrace_rf_joblib(cfg, model_key)
    if not rf_path.is_file():
        raise FileNotFoundError(
            f"memTrace RF not found: {rf_path}. Set mia_gt_pipeline.memtrace_rf_joblib or memtrace_rf_dir."
        )

    exp = cfg.get("experiment") or {}
    device = pick_device(exp.get("device"))
    dtype = torch_dtype_from_str(bundle.get("torch_dtype"))
    target = bundle["target_model"]
    ref_name = bundle["reference_model"]
    tok_id = bundle.get("tokenizer") or target
    max_len = int(exp.get("max_length_tokens", 512))

    inf_p = dict(block.get("open_model_infilling") or {"m": 1, "k": 0.2})
    inf_s = dict(block.get("open_model_infilling_sensitivity") or {"m": 5, "k": 0.2})
    wbc_p = dict(block.get("open_model_wbc") or {"min_window": 2, "max_window": 40, "num_windows": 12})
    wbc_s = dict(
        block.get("open_model_wbc_sensitivity") or {"min_window": 2, "max_window": 56, "num_windows": 10}
    )

    mt_len = int(block.get("memtrace_max_length") or (cfg.get("methods") or {}).get("memtrace", {}).get("max_length") or max_len)

    print(f"[mia_annotate] Loading target {target} (eager attn)…", file=sys.stderr)
    model_t, tok = load_causal_lm(target, tok_id, device, dtype, attn_implementation="eager")

    s_inf_p = np.asarray(infilling_scores(model_t, tok, texts, inf_p, max_length=max_len), dtype=np.float64)
    s_inf_s = np.asarray(infilling_scores(model_t, tok, texts, inf_s, max_length=max_len), dtype=np.float64)

    print(f"[mia_annotate] Loading reference {ref_name}…", file=sys.stderr)
    model_r, _ = load_causal_lm(ref_name, tok_id, device, dtype)
    s_wbc_p = np.asarray(
        wbc_scores(model_t, model_r, tok, texts, device, _wbc_params_merge(cfg, wbc_p), max_length=max_len),
        dtype=np.float64,
    )
    s_wbc_s = np.asarray(
        wbc_scores(model_t, model_r, tok, texts, device, _wbc_params_merge(cfg, wbc_s), max_length=max_len),
        dtype=np.float64,
    )

    print("[mia_annotate] memTrace features + RF…", file=sys.stderr)
    X_mt = extract_memtrace_features_with_model(
        model_t, tok, texts, mt_len, device, show_progress=True
    )
    s_mt = _memtrace_p_batch(X_mt, rf_path)

    del model_t, model_r
    if device.type == "cuda":
        import torch

        torch.cuda.empty_cache()

    s_sel: np.ndarray | None = None
    sel_block = block.get("select") or {}
    if sel_block.get("enabled"):
        from mia_eval.scoring_morris2025_select import compute_select_last_layer_scores

        base_id = str(sel_block.get("base_model") or "").strip()
        if not base_id:
            raise ValueError(
                "mia_gt_pipeline.select.enabled requires mia_gt_pipeline.select.base_model "
                "(HF model id for θ_0, same family as target θ_f)."
            )
        sel_max_len = int(sel_block.get("max_length", max_len))
        s_sel = compute_select_last_layer_scores(
            cfg,
            bundle,
            texts,
            base_model_id=base_id,
            tokenizer_id=tok_id,
            max_length=min(max_len, sel_max_len),
            n_monte_carlo=int(sel_block.get("n_monte_carlo", 4096)),
            random_state=int(sel_block.get("random_state", 42)),
            batch_size=int(sel_block.get("batch_size", 1)),
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            primary = {
                "infilling": float(s_inf_p[i]),
                "wbc": float(s_wbc_p[i]),
                "memtrace_p_member": float(s_mt[i]),
            }
            sensitivity = {
                "infilling": float(s_inf_s[i]),
                "wbc": float(s_wbc_s[i]),
                "memtrace_p_member": float(s_mt[i]),
            }
            if s_sel is not None:
                v = float(s_sel[i])
                primary["select_alignment_mc"] = v
                sensitivity["select_alignment_mc"] = v
            doc = {
                "text": r.get("text", ""),
                "source": r.get("source", ""),
                "mia_gt_primary": primary,
                "mia_gt_sensitivity": sensitivity,
            }
            for k in ("strategy", "top_k", "top_p"):
                if k in r:
                    doc[k] = r[k]
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"[mia_annotate] Wrote {out_path} ({len(rows)} rows)", file=sys.stderr)
    return len(rows)


def _spearman_matrix(a: np.ndarray, names: List[str]) -> Tuple[List[List[float]], List[str]]:
    """``a`` shape (num_samples, num_metrics); Spearman between columns."""
    n = a.shape[1]
    mat = [[1.0] * n for _ in range(n)]
    try:
        from scipy.stats import spearmanr as _spearmanr
    except Exception:  # pragma: no cover
        _spearmanr = None
    for i in range(n):
        for j in range(i + 1, n):
            xi, xj = a[:, i], a[:, j]
            if np.std(xi) < 1e-12 or np.std(xj) < 1e-12:
                rho = float("nan")
            elif _spearmanr is not None:
                rho, _ = _spearmanr(xi, xj)
                rho = float(rho)
            else:
                ri = np.argsort(np.argsort(xi)).astype(np.float64)
                rj = np.argsort(np.argsort(xj)).astype(np.float64)
                rho = float(np.corrcoef(ri, rj)[0, 1])
            mat[i][j] = rho
            mat[j][i] = rho
    return mat, names


def evaluate_mia_gt_jsonl(annotated_path: Path) -> Dict[str, Any]:
    rows = load_samples_jsonl(annotated_path)
    p_inf, p_wbc, p_mt, p_sel = [], [], [], []
    s_inf, s_wbc, s_mt, s_sel = [], [], [], []
    has_sel = bool(
        rows
        and all("select_alignment_mc" in (r.get("mia_gt_primary") or {}) for r in rows)
    )
    for r in rows:
        gp = r.get("mia_gt_primary") or {}
        gs = r.get("mia_gt_sensitivity") or {}
        p_inf.append(float(gp.get("infilling", 0.0)))
        p_wbc.append(float(gp.get("wbc", 0.0)))
        p_mt.append(float(gp.get("memtrace_p_member", 0.0)))
        s_inf.append(float(gs.get("infilling", 0.0)))
        s_wbc.append(float(gs.get("wbc", 0.0)))
        s_mt.append(float(gs.get("memtrace_p_member", 0.0)))
        if has_sel:
            p_sel.append(float(gp.get("select_alignment_mc", 0.0)))
            s_sel.append(float(gs.get("select_alignment_mc", 0.0)))

    names_p = ["infilling", "wbc", "memtrace_p_member"]
    cols_p = [p_inf, p_wbc, p_mt]
    cols_s = [s_inf, s_wbc, s_mt]
    if has_sel:
        names_p.append("select_alignment_mc")
        cols_p.append(p_sel)
        cols_s.append(s_sel)

    A_p = np.column_stack(cols_p)
    A_s = np.column_stack(cols_s)
    mat_p, names = _spearman_matrix(A_p, names_p)
    mat_s, _ = _spearman_matrix(A_s, names_p)

    d_inf = np.mean(np.abs(A_p[:, 0] - A_s[:, 0]))
    d_wbc = np.mean(np.abs(A_p[:, 1] - A_s[:, 1]))
    d_mt = np.mean(np.abs(A_p[:, 2] - A_s[:, 2]))

    out: Dict[str, Any] = {
        "n_samples": len(rows),
        "spearman_primary": {"rows": names, "matrix": mat_p},
        "spearman_sensitivity": {"rows": names, "matrix": mat_s},
        "mean_abs_delta_primary_vs_sensitivity": {
            "infilling": float(d_inf),
            "wbc": float(d_wbc),
            "memtrace_p_member": float(d_mt),
        },
        "note": (
            "Spearman among infilling, WBC, memTrace p_member"
            + (", and select_alignment_mc" if has_sel else "")
            + " at fixed HP; not supervised AUC vs training-data shingles."
        ),
    }

    # Optional: rank agreement with memTrace (median split) — label as exploratory only.
    if len(p_mt) >= 4 and len(np.unique(p_mt)) > 1:
        med = float(np.median(p_mt))
        y = (np.asarray(p_mt) > med).astype(np.int64)
        if y.sum() not in (0, len(y)):
            from sklearn.metrics import roc_auc_score

            try:
                auc_i = float(roc_auc_score(y, -np.asarray(p_inf)))
                auc_w = float(roc_auc_score(y, np.asarray(p_wbc)))
                out["exploratory_auc_vs_memtrace_median_split"] = {
                    "infilling_negated_for_higher_with_high_memtrace": auc_i,
                    "wbc": auc_w,
                    "note": "y=1 if memtrace_p > median; infilling negated so higher may align with y under infilling convention.",
                }
            except ValueError:
                pass

    return out
