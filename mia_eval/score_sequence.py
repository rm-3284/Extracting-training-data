#!/usr/bin/env python3
"""
Score one or more text sequences with infilling, WBC, and memTrace-style features.

This does **not** build shingle ground truth. Outputs are **method-native scores**:
  - infilling: lower is more ``member-like'' under the infilling construction (same as pipeline).
  - wbc: WBC aggregate score (loss-based).
  - memtrace: optional ``p_member`` if you pass ``--memtrace-rf`` (sklearn Pipeline or dict with scaler+rf);
    always includes ``feature_l2`` / ``feature_dim`` as weak summaries of the feature vector.

Example:
  python -m mia_eval.score_sequence \\
    --config mia_eval/config/defaults.yaml \\
    --experiment mia_eval/config/qwen2p5.yaml \\
    --preset qwen25_7b_base \\
    --text "The quick brown fox."

  python -m mia_eval.score_sequence ... --text-file candidates.txt --jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mia_eval.config_loader import active_model_bundle, load_merged_config, apply_dot_overrides
from mia_eval.model_utils import load_causal_lm, pick_device, torch_dtype_from_str
from mia_eval.scoring_infilling import score_texts as infilling_scores
from mia_eval.scoring_memtrace import extract_memtrace_features_with_model
from mia_eval.scoring_wbc import score_texts as wbc_scores


def _sanitize_row(x: np.ndarray) -> np.ndarray:
    out = np.array(x, dtype=np.float64, copy=True).reshape(1, -1)
    np.nan_to_num(out, copy=False, nan=0.0, posinf=1e10, neginf=-1e10)
    return out


def _memtrace_p_from_artifact(vec: np.ndarray, path: Path) -> float:
    import joblib

    obj: Any = joblib.load(path)
    X = _sanitize_row(vec)
    if hasattr(obj, "predict_proba"):
        return float(obj.predict_proba(X)[0, 1])
    if isinstance(obj, dict) and "scaler" in obj and "rf" in obj:
        Xs = obj["scaler"].transform(X)
        return float(obj["rf"].predict_proba(Xs)[0, 1])
    raise ValueError(
        f"Unsupported memtrace artifact {path}: expected sklearn Pipeline / "
        "Classifier with predict_proba, or dict with keys 'scaler', 'rf'."
    )


def _score_one(
    texts: List[str],
    cfg: Dict[str, Any],
    bundle: Dict[str, Any],
    *,
    memtrace_rf: Optional[Path],
    max_len_inf_wbc: int,
    mt_max_len: int,
) -> Dict[str, Any]:
    exp = cfg.get("experiment") or {}
    device = pick_device(exp.get("device"))
    dtype = torch_dtype_from_str(bundle.get("torch_dtype"))
    target = bundle["target_model"]
    ref_name = bundle["reference_model"]
    tok_id = bundle.get("tokenizer") or target

    sq = cfg.get("score_sequence") or {}
    sq_inf = sq.get("infilling") or {}
    p_inf = {"m": int(sq_inf.get("m", 5)), "k": float(sq_inf.get("k", 0.2))}
    sq_wbc = sq.get("wbc") or {}
    p_wbc = {
        "min_window": int(sq_wbc.get("min_window", 2)),
        "max_window": int(sq_wbc.get("max_window", 40)),
        "num_windows": int(sq_wbc.get("num_windows", 8)),
    }
    mw = (cfg.get("methods") or {}).get("wbc") or {}
    for k in ("ensemble_variants", "use_ensemble", "ensemble_aggregate"):
        if k in mw:
            p_wbc[k] = mw[k]
    if p_wbc.get("window_sizes") is None:
        p_wbc.pop("window_sizes", None)

    print("Loading target model (eager attention for memTrace)...", file=sys.stderr)
    model_t, tok = load_causal_lm(
        target, tok_id, device, dtype, attn_implementation="eager"
    )

    s_inf = infilling_scores(model_t, tok, texts, p_inf, max_length=max_len_inf_wbc)

    print("Loading reference model for WBC...", file=sys.stderr)
    model_r, _ = load_causal_lm(ref_name, tok_id, device, dtype)
    s_wbc = wbc_scores(
        model_t, model_r, tok, texts, device, p_wbc, max_length=max_len_inf_wbc
    )

    print("memTrace features...", file=sys.stderr)
    X_mt = extract_memtrace_features_with_model(
        model_t, tok, texts, mt_max_len, device, show_progress=len(texts) > 1
    )

    del model_r
    if device.type == "cuda":
        import torch

        torch.cuda.empty_cache()

    out_rows: List[Dict[str, Any]] = []
    for i, t in enumerate(texts):
        vec = X_mt[i]
        mem: Dict[str, Any] = {
            "feature_dim": int(vec.shape[0]),
            "feature_l2": float(np.linalg.norm(vec)),
            "feature_mean_abs": float(np.mean(np.abs(vec))),
        }
        if memtrace_rf is not None:
            mem["p_member"] = _memtrace_p_from_artifact(vec, memtrace_rf)
        out_rows.append(
            {
                "text_preview": t[:200] + ("…" if len(t) > 200 else ""),
                "infilling_score": float(s_inf[i]),
                "wbc_score": float(s_wbc[i]),
                "memtrace": mem,
            }
        )

    del model_t
    if device.type == "cuda":
        import torch

        torch.cuda.empty_cache()

    return {
        "target_model": target,
        "reference_model": ref_name,
        "tokenizer": tok_id,
        "params": {"infilling": p_inf, "wbc": p_wbc, "memtrace_max_length": mt_max_len},
        "max_length_infilling_wbc": max_len_inf_wbc,
        "notes": {
            "infilling": (
                "Raw bottom-k infilling aggregate (same construction as mia_eval). "
                "Across candidates, lower often correlates with stronger membership signal; "
                "run_pipeline orients on val so that higher AUC — compare only within a fixed setup."
            ),
            "wbc": "Raw WBC score from wbc_attack (compare within fixed HPs / model pair).",
            "memtrace": (
                "Without --memtrace-rf, only feature norms are returned. "
                "p_member requires a scaler+RF trained on your proxy labels (see docs/qwen_memtrace_proxy.md)."
            ),
        },
        "rows": out_rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Score sequence(s) with infilling, WBC, memTrace.")
    ap.add_argument("--config", type=str, default=str(ROOT / "mia_eval/config/defaults.yaml"))
    ap.add_argument("--experiment", type=str, default="", help="YAML merged over config (e.g. qwen2p5.yaml).")
    ap.add_argument("--set", action="append", default=[], help="Overrides, e.g. active_model=qwen25_7b_math")
    ap.add_argument(
        "--preset",
        type=str,
        default="",
        help="Shorthand: set active_model to this key (must exist under models:).",
    )
    ap.add_argument("--text", type=str, default="", help="Single input string.")
    ap.add_argument("--text-file", type=str, default="", help="UTF-8 file, one text per line.")
    ap.add_argument(
        "--memtrace-rf",
        type=str,
        default="",
        help="Optional joblib: sklearn Pipeline, or dict {{scaler, rf}} trained on memTrace features.",
    )
    ap.add_argument(
        "--jsonl",
        action="store_true",
        help="With --text-file, print one JSON object per line (scores only that row).",
    )
    args = ap.parse_args()

    exp_path = Path(args.experiment) if args.experiment else None
    cfg = load_merged_config(Path(args.config), exp_path)
    cfg = apply_dot_overrides(cfg, args.set)
    if args.preset:
        cfg["active_model"] = args.preset

    bundle = active_model_bundle(cfg)
    texts: List[str] = []
    if args.text:
        texts.append(args.text)
    if args.text_file:
        raw = Path(args.text_file).read_text(encoding="utf-8")
        for line in raw.splitlines():
            line = line.strip()
            if line:
                texts.append(line)
    if not texts:
        ap.error("Provide --text and/or --text-file with non-empty lines.")

    exp = cfg.get("experiment") or {}
    max_len = int(exp.get("max_length_tokens", 512))
    sq = cfg.get("score_sequence") or {}
    mt_max_len = int(sq.get("memtrace_max_length", max_len))

    rf_path = Path(args.memtrace_rf) if args.memtrace_rf else None
    if rf_path is not None and not rf_path.is_file():
        ap.error(f"--memtrace-rf not found: {rf_path}")

    payload = _score_one(
        texts,
        cfg,
        bundle,
        memtrace_rf=rf_path,
        max_len_inf_wbc=max_len,
        mt_max_len=mt_max_len,
    )
    payload["active_model"] = cfg.get("active_model")

    if args.jsonl and args.text_file:
        for row in payload["rows"]:
            print(json.dumps(row, ensure_ascii=False))
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
