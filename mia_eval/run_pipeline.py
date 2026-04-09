#!/usr/bin/env python3
"""
End-to-end MIA evaluation vs training-data-derived labels.

Steps: build_index | generate | label | evaluate

Run from repository root:
  python -m mia_eval.run_pipeline --config mia_eval/config/defaults.yaml --steps all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Repository root (parent of mia_eval/)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from mia_eval.config_loader import active_model_bundle, iter_grid, load_merged_config, apply_dot_overrides
from mia_eval.generation import generate_diverse_samples
from mia_eval.ground_truth import TrainingShingleIndex, build_index_from_hf
from mia_eval.labeling import label_jsonl, load_labeled
from mia_eval.model_utils import load_causal_lm, pick_device, torch_dtype_from_str
from mia_eval.scoring_infilling import score_texts as infilling_scores
from mia_eval.scoring_memtrace import compute_feature_matrix, fit_rf_on_splits
from mia_eval.scoring_wbc import score_texts as wbc_scores


def _auc_direction(y: np.ndarray, s: np.ndarray) -> tuple[float, np.ndarray]:
    """Return max(AUC(s), AUC(-s)) and the score vector to use."""
    s = np.asarray(s, dtype=np.float64)
    a0 = float(roc_auc_score(y, s))
    a1 = float(roc_auc_score(y, -s))
    if a1 > a0:
        return a1, -s
    return a0, s


def _split_masks(
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "mia_eval/config/defaults.yaml"),
    )
    ap.add_argument(
        "--experiment",
        type=str,
        default="",
        help="Optional second YAML merged over defaults.",
    )
    ap.add_argument(
        "--steps",
        type=str,
        default="all",
        help="Comma list: build_index,generate,label,evaluate,all",
    )
    ap.add_argument("--set", action="append", default=[], help="Override e.g. active_model=pythia_2p8")
    args = ap.parse_args()

    defaults_path = Path(args.config)
    exp_path = Path(args.experiment) if args.experiment else None
    cfg = load_merged_config(defaults_path, exp_path)
    cfg = apply_dot_overrides(cfg, args.set)

    steps = {x.strip() for x in args.steps.lower().split(",")}
    if "all" in steps:
        steps = {"build_index", "generate", "label", "evaluate"}

    out_root = Path(cfg.get("experiment", {}).get("output_dir", "mia_eval_outputs"))
    model_key = cfg.get("active_model", "gpt_neo_2p7")
    run_dir = out_root / model_key
    run_dir.mkdir(parents=True, exist_ok=True)

    bundle = active_model_bundle(cfg)
    gt_cfg = bundle.get("ground_truth") or {}

    index_path = run_dir / "training_shingle_index.json"
    samples_path = run_dir / "samples.jsonl"
    labeled_path = run_dir / "samples_labeled.jsonl"
    results_path = run_dir / "results.json"

    if "build_index" in steps:
        print("Building training shingle index...")
        idx = build_index_from_hf(
            gt_cfg["dataset_name"],
            gt_cfg.get("dataset_config"),
            gt_cfg.get("text_field", "text"),
            int(gt_cfg.get("max_documents", 2000)),
            int(gt_cfg.get("shingle_chars", 200)),
            int(gt_cfg.get("max_shingles", 2_000_000)),
        )
        idx.save(index_path)
        print(f"Saved index ({len(idx)} shingles) -> {index_path}")

    if "generate" in steps:
        print("Generating samples...")
        generate_diverse_samples(cfg, bundle, samples_path)
        print(f"Wrote {samples_path}")

    if "label" in steps:
        print("Labeling samples...")
        idx = TrainingShingleIndex.load(index_path)
        label_jsonl(
            samples_path,
            idx,
            int(gt_cfg.get("min_match_chars", 150)),
            labeled_path,
        )
        print(f"Wrote {labeled_path}")

    if "evaluate" in steps:
        texts, labels, sources = load_labeled(labeled_path)
        y = np.array(labels, dtype=np.int64)
        if y.sum() == 0 or y.sum() == len(y):
            raise RuntimeError(
                "Labels are all one class; adjust generation, index size, or min_match_chars."
            )

        ev = cfg.get("evaluation") or {}
        test_f = float(ev.get("test_fraction", 0.2))
        val_f = float(ev.get("val_fraction", 0.15))
        rs = int(ev.get("random_state", 42))
        tr_m, va_m, te_m = _split_masks(y, test_f, val_f, rs)

        exp = cfg.get("experiment") or {}
        device = pick_device(exp.get("device"))
        dtype = torch_dtype_from_str(bundle.get("torch_dtype"))
        max_len = int(exp.get("max_length_tokens", 512))

        results: dict = {"model": model_key, "n_samples": len(y), "methods": {}}

        # ----- Infilling -----
        m_inf = (cfg.get("methods") or {}).get("infilling") or {}
        grid_inf = (m_inf.get("hyperparameter_search") or {}).get("grid") or {"m": [5], "k": [0.1]}
        print("Loading target model for infilling / WBC...")
        target = bundle["target_model"]
        tok_id = bundle.get("tokenizer") or target
        model_t, tok = load_causal_lm(target, tok_id, device, dtype)

        best_inf = None
        for params in iter_grid(grid_inf):
            scores = np.array(
                infilling_scores(model_t, tok, texts, params, max_length=max_len),
                dtype=np.float64,
            )
            auc_va, _ = _auc_direction(y[va_m], scores[va_m])
            if best_inf is None or auc_va > best_inf[0]:
                auc_te, _ = _auc_direction(y[te_m], scores[te_m])
                best_inf = (auc_va, params, float(auc_te))

        assert best_inf is not None
        results["methods"]["infilling"] = {
            "best_params": best_inf[1],
            "val_auc": best_inf[0],
            "test_auc": best_inf[2],
        }

        # ----- WBC -----
        m_wbc = (cfg.get("methods") or {}).get("wbc") or {}
        grid_wbc = (m_wbc.get("hyperparameter_search") or {}).get("grid") or {
            "min_window": [2],
            "max_window": [40],
            "num_windows": [10],
        }
        # filter out null window_sizes in grid expansion
        ref_name = bundle["reference_model"]
        model_r, _ = load_causal_lm(ref_name, tok_id, device, dtype)

        best_wbc = None
        for params in iter_grid(grid_wbc):
            p2 = dict(params)
            if p2.get("window_sizes") is None:
                p2.pop("window_sizes", None)
            scores = np.array(
                wbc_scores(model_t, model_r, tok, texts, device, p2, max_length=max_len),
                dtype=np.float64,
            )
            auc_va, _ = _auc_direction(y[va_m], scores[va_m])
            if best_wbc is None or auc_va > best_wbc[0]:
                auc_te, _ = _auc_direction(y[te_m], scores[te_m])
                best_wbc = (auc_va, params, float(auc_te))

        assert best_wbc is not None
        results["methods"]["wbc"] = {
            "best_params": best_wbc[1],
            "val_auc": best_wbc[0],
            "test_auc": best_wbc[2],
        }

        del model_t, model_r
        if device.type == "cuda":
            import torch

            torch.cuda.empty_cache()

        # ----- memTrace -----
        m_mt = (cfg.get("methods") or {}).get("memtrace") or {}
        grid_mt = (m_mt.get("hyperparameter_search") or {}).get("grid") or {
            "n_estimators": [200],
            "max_depth": [8],
            "min_samples_leaf": [4],
        }
        mt_max_len = int(m_mt.get("max_length", 512))
        print("Computing memTrace features (slow, large memory)...")
        X_mt = compute_feature_matrix(cfg, bundle, texts, mt_max_len)
        best_mt = None
        for params in iter_grid(grid_mt):
            params = {**params, "max_length": mt_max_len}
            _, meta = fit_rf_on_splits(X_mt, y, tr_m, va_m, te_m, params, rs)
            auc_va = meta["val_auc"]
            if best_mt is None or auc_va > best_mt[0]:
                best_mt = (auc_va, params, meta["test_auc"])

        assert best_mt is not None
        results["methods"]["memtrace"] = {
            "best_params": {k: v for k, v in best_mt[1].items() if k != "max_length"},
            "val_auc": best_mt[0],
            "test_auc": best_mt[2],
            "max_length": mt_max_len,
        }

        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
