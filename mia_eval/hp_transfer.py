#!/usr/bin/env python3
"""
Evaluate hyperparameter transfer: apply best_params from a source model's results.json
to the target model's labeled data (same train/val/test split as run_pipeline).

Requires target artifacts from a full run:
  python -m mia_eval.run_pipeline --config ... --set active_model=<target> --steps all

Then (example):
  python -m mia_eval.hp_transfer \\
    --config mia_eval/config/defaults.yaml \\
    --source-model gpt_neo_2p7 \\
    --target-model pythia_2p8

Writes: mia_eval_outputs/<target_model>/hp_transfer_from_<source_model>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mia_eval.config_loader import active_model_bundle, load_merged_config, apply_dot_overrides
from mia_eval.evaluation_common import (
    auc_direction,
    jsonable,
    orient_scores_full,
    split_masks,
)
from mia_eval.labeling import load_labeled
from mia_eval.model_utils import load_causal_lm, pick_device, torch_dtype_from_str
from mia_eval.scoring_infilling import score_texts as infilling_scores
from mia_eval.scoring_memtrace import compute_feature_matrix, fit_rf_on_splits
from mia_eval.scoring_wbc import score_texts as wbc_scores


def _load_results(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _eval_infilling(
    y: np.ndarray,
    va_m: np.ndarray,
    te_m: np.ndarray,
    texts: list[str],
    params: dict,
    *,
    model_t,
    tok,
    max_len: int,
) -> dict:
    scores = np.array(
        infilling_scores(model_t, tok, texts, params, max_length=max_len),
        dtype=np.float64,
    )
    auc_va, _ = auc_direction(y[va_m], scores[va_m])
    auc_te, _ = auc_direction(y[te_m], scores[te_m])
    oriented = orient_scores_full(y, va_m, scores)
    return {
        "params": jsonable(params),
        "val_auc": auc_va,
        "test_auc": float(auc_te),
        "score_orientation": "higher_is_member_on_val_split",
        "_scores_oriented": oriented,
        "_scores_raw": scores,
    }


def _eval_wbc(
    y: np.ndarray,
    va_m: np.ndarray,
    te_m: np.ndarray,
    texts: list[str],
    params: dict,
    *,
    model_t,
    model_r,
    tok,
    device,
    max_len: int,
) -> dict:
    p2 = dict(params)
    if p2.get("window_sizes") is None:
        p2.pop("window_sizes", None)
    scores = np.array(
        wbc_scores(model_t, model_r, tok, texts, device, p2, max_length=max_len),
        dtype=np.float64,
    )
    auc_va, _ = auc_direction(y[va_m], scores[va_m])
    auc_te, _ = auc_direction(y[te_m], scores[te_m])
    oriented = orient_scores_full(y, va_m, scores)
    return {
        "params": jsonable(p2),
        "val_auc": auc_va,
        "test_auc": float(auc_te),
        "score_orientation": "higher_is_member_on_val_split",
        "_scores_oriented": oriented,
        "_scores_raw": scores,
    }


def _eval_memtrace(
    y: np.ndarray,
    tr_m: np.ndarray,
    va_m: np.ndarray,
    te_m: np.ndarray,
    texts: list[str],
    params: dict,
    *,
    cfg: dict,
    bundle: dict,
    mt_max_len: int,
    random_state: int,
) -> dict:
    rf_keys = {"n_estimators", "max_depth", "min_samples_leaf"}
    rf_params = {k: params[k] for k in rf_keys if k in params}
    X_mt = compute_feature_matrix(cfg, bundle, texts, mt_max_len)
    proba_all, meta = fit_rf_on_splits(
        X_mt, y, tr_m, va_m, te_m, rf_params, random_state
    )
    return {
        "params": jsonable(rf_params),
        "val_auc": meta["val_auc"],
        "test_auc": float(meta["test_auc"]),
        "max_length": mt_max_len,
        "score_orientation": "memtrace_p_member_is_RF_P(class=1)",
        "_proba": np.asarray(proba_all, dtype=np.float64),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Hyperparameter transfer evaluation.")
    ap.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "mia_eval/config/defaults.yaml"),
    )
    ap.add_argument("--experiment", type=str, default="", help="Optional YAML merged over defaults.")
    ap.add_argument("--set", action="append", default=[], help="Config overrides, e.g. active_model=pythia_2p8")
    ap.add_argument(
        "--source-model",
        type=str,
        required=True,
        help="Preset whose results.json supplies best_params (e.g. gpt_neo_2p7).",
    )
    ap.add_argument(
        "--target-model",
        type=str,
        required=True,
        help="Preset to evaluate (must have samples_labeled.jsonl for this preset).",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Root for mia_eval_outputs (default: from config experiment.output_dir).",
    )
    ap.add_argument(
        "--source-results",
        type=str,
        default="",
        help="Override path to source results.json (default: <output-dir>/<source-model>/results.json).",
    )
    ap.add_argument(
        "--target-labeled",
        type=str,
        default="",
        help="Override path to target samples_labeled.jsonl.",
    )
    args = ap.parse_args()

    defaults_path = Path(args.config)
    exp_path = Path(args.experiment) if args.experiment else None
    cfg = load_merged_config(defaults_path, exp_path)
    cfg = apply_dot_overrides(cfg, args.set)

    out_root = Path(
        args.output_dir or cfg.get("experiment", {}).get("output_dir", "mia_eval_outputs")
    )
    src_results_path = (
        Path(args.source_results)
        if args.source_results
        else out_root / args.source_model / "results.json"
    )
    if not src_results_path.is_file():
        raise FileNotFoundError(
            f"Source results not found: {src_results_path}\n"
            "Run the pipeline for the source model first, or pass --source-results."
        )

    cfg = {**cfg, "active_model": args.target_model}
    bundle = active_model_bundle(cfg)
    model_key = args.target_model
    run_dir = out_root / model_key
    labeled_path = (
        Path(args.target_labeled)
        if args.target_labeled
        else run_dir / "samples_labeled.jsonl"
    )
    if not labeled_path.is_file():
        raise FileNotFoundError(
            f"Target labeled data not found: {labeled_path}\n"
            "Run: python -m mia_eval.run_pipeline --config ... "
            f"--set active_model={args.target_model} --steps all"
        )

    source_payload = _load_results(src_results_path)
    methods_src = source_payload.get("methods") or {}

    texts, labels, sources = load_labeled(labeled_path)
    y = np.array(labels, dtype=np.int64)
    if y.sum() == 0 or y.sum() == len(y):
        raise RuntimeError("Labels are all one class; cannot compute AUC.")

    ev = cfg.get("evaluation") or {}
    test_f = float(ev.get("test_fraction", 0.2))
    val_f = float(ev.get("val_fraction", 0.2))
    rs = int(ev.get("random_state", 42))
    tr_m, va_m, te_m = split_masks(y, test_f, val_f, rs)

    exp = cfg.get("experiment") or {}
    device = pick_device(exp.get("device"))
    dtype = torch_dtype_from_str(bundle.get("torch_dtype"))
    max_len = int(exp.get("max_length_tokens", 512))

    report: dict = {
        "source_model": args.source_model,
        "target_model": args.target_model,
        "source_results_path": str(src_results_path),
        "target_labeled_path": str(labeled_path),
        "n_samples": len(y),
        "split": {
            "test_fraction": test_f,
            "val_fraction": val_f,
            "random_state": rs,
        },
        "transferred": {},
    }

    target = bundle["target_model"]
    tok_id = bundle.get("tokenizer") or target
    ref_name = bundle["reference_model"]

    # Infilling + WBC share target model load
    if "infilling" in methods_src or "wbc" in methods_src:
        print("Loading target / reference models for infilling & WBC...")
        model_t, tok = load_causal_lm(target, tok_id, device, dtype)
        model_r, _ = load_causal_lm(ref_name, tok_id, device, dtype)

        if "infilling" in methods_src:
            p_inf = methods_src["infilling"].get("best_params") or {}
            if not p_inf:
                report["transferred"]["infilling"] = {"error": "no best_params in source"}
            else:
                r = _eval_infilling(
                    y, va_m, te_m, texts, p_inf,
                    model_t=model_t, tok=tok, max_len=max_len,
                )
                report["transferred"]["infilling"] = {
                    k: v for k, v in r.items() if not k.startswith("_")
                }

        if "wbc" in methods_src:
            p_wbc = methods_src["wbc"].get("best_params") or {}
            if not p_wbc:
                report["transferred"]["wbc"] = {"error": "no best_params in source"}
            else:
                r = _eval_wbc(
                    y, va_m, te_m, texts, p_wbc,
                    model_t=model_t, model_r=model_r, tok=tok,
                    device=device, max_len=max_len,
                )
                report["transferred"]["wbc"] = {
                    k: v for k, v in r.items() if not k.startswith("_")
                }

        del model_t, model_r
        if device.type == "cuda":
            import torch
            torch.cuda.empty_cache()
    else:
        print("Skipping infilling/WBC: not present in source results.")

    if "memtrace" in methods_src:
        p_mt = methods_src["memtrace"].get("best_params") or {}
        m_mt = (cfg.get("methods") or {}).get("memtrace") or {}
        mt_max_len = int(m_mt.get("max_length", int(methods_src["memtrace"].get("max_length", 256))))
        if not p_mt:
            report["transferred"]["memtrace"] = {"error": "no best_params in source"}
        else:
            print("Computing memTrace features on target model (slow)...")
            r = _eval_memtrace(
                y, tr_m, va_m, te_m, texts, p_mt,
                cfg=cfg, bundle=bundle, mt_max_len=mt_max_len, random_state=rs,
            )
            report["transferred"]["memtrace"] = {
                k: v for k, v in r.items() if not k.startswith("_")
            }

    # Optional oracle comparison: target's own results.json from a full grid search
    oracle_path = out_root / args.target_model / "results.json"
    if oracle_path.is_file():
        oracle = _load_results(oracle_path)
        report["oracle_on_target"] = {
            m: {
                "best_params": (oracle.get("methods") or {}).get(m, {}).get("best_params"),
                "val_auc": (oracle.get("methods") or {}).get(m, {}).get("val_auc"),
                "test_auc": (oracle.get("methods") or {}).get(m, {}).get("test_auc"),
            }
            for m in ("infilling", "wbc", "memtrace")
            if (oracle.get("methods") or {}).get(m)
        }
        report["oracle_results_path"] = str(oracle_path)

        report["comparison_test_auc"] = {}
        for m in ("infilling", "wbc", "memtrace"):
            tr = report["transferred"].get(m, {})
            orc = (oracle.get("methods") or {}).get(m, {})
            if tr.get("error") or "test_auc" not in tr:
                continue
            ot = orc.get("test_auc")
            tt = tr.get("test_auc")
            if ot is None or tt is None:
                continue
            report["comparison_test_auc"][m] = {
                "transferred_from_source": float(tt),
                "oracle_on_target": float(ot),
                "oracle_minus_transferred": float(ot) - float(tt),
            }

    out_path = run_dir / f"hp_transfer_from_{args.source_model}.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(jsonable(report), f, indent=2)
    print(json.dumps(jsonable(report), indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
