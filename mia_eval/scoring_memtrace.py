"""memTrace features + RF; train on labeled subset, score all texts."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from memtrace.features import extract_memtrace_features_from_tensors

from .model_utils import load_causal_lm, pick_device, resolve_lm_head, torch_dtype_from_str


@torch.inference_mode()
def _features_one(
    model,
    tokenizer,
    lm_head,
    text: str,
    device: torch.device,
    max_length: int,
) -> np.ndarray:
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(
        **enc,
        output_attentions=True,
        output_hidden_states=True,
        return_dict=True,
    )
    hs = out.hidden_states
    att = out.attentions
    if hs is None:
        raise RuntimeError("Model returned no hidden_states")
    vec, _ = extract_memtrace_features_from_tensors(
        hs,
        att,
        lm_head,
        attention_mask=enc.get("attention_mask"),
        device=device,
    )
    return vec


def compute_feature_matrix(
    cfg: Dict[str, Any],
    model_bundle: Dict[str, Any],
    texts: List[str],
    max_length: int,
) -> np.ndarray:
    exp = cfg.get("experiment") or {}
    device = pick_device(exp.get("device"))
    dtype = torch_dtype_from_str(model_bundle.get("torch_dtype"))
    target = model_bundle["target_model"]
    tok_id = model_bundle.get("tokenizer") or target
    # SDPA / flash attention ignores output_attentions; eager returns real weights for memTrace.
    model, tokenizer = load_causal_lm(
        target, tok_id, device, dtype, attn_implementation="eager"
    )
    lm_head = resolve_lm_head(model)
    rows: List[np.ndarray] = []
    for t in tqdm(texts, desc="memTrace features"):
        rows.append(_features_one(model, tokenizer, lm_head, t, device, max_length))
    return np.stack(rows, axis=0)


def _sanitize_feature_matrix(X: np.ndarray) -> np.ndarray:
    """Finite values only; StandardScaler rejects inf/nan."""
    # np.asarray(..., copy=) is NumPy 2+ only; np.array(..., copy=True) works on 1.x.
    out = np.array(X, dtype=np.float64, copy=True)
    np.nan_to_num(out, copy=False, nan=0.0, posinf=1e10, neginf=-1e10)
    return out


def fit_rf_on_splits(
    X: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    params: Dict[str, Any],
    random_state: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    X = _sanitize_feature_matrix(X)
    X_tr, y_tr = X[train_mask], labels[train_mask]
    X_va, y_va = X[val_mask], labels[val_mask]
    X_te, y_te = X[test_mask], labels[test_mask]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    rf = RandomForestClassifier(
        n_estimators=int(params.get("n_estimators", 200)),
        max_depth=int(params.get("max_depth", 8)),
        min_samples_leaf=int(params.get("min_samples_leaf", 4)),
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_tr_s, y_tr)
    try:
        val_auc = float(roc_auc_score(y_va, rf.predict_proba(X_va_s)[:, 1]))
    except ValueError:
        val_auc = 0.5
    try:
        test_auc = float(roc_auc_score(y_te, rf.predict_proba(X_te_s)[:, 1]))
    except ValueError:
        test_auc = 0.5
    X_all_s = scaler.transform(X)
    proba_all = rf.predict_proba(X_all_s)[:, 1]
    return proba_all, {"val_auc": val_auc, "test_auc": test_auc}
