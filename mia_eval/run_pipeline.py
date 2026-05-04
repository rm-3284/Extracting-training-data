#!/usr/bin/env python3
"""
End-to-end MIA evaluation vs training-data-derived labels.

Steps: build_index | generate | label | evaluate

**MIA reference scores (no shingle labels):** ``generate``, ``mia_annotate``,
``mia_evaluate`` — or ``all_mia_gt`` for all three. See ``mia_gt_pipeline.py``
and ``mia_gt_pipeline`` in experiment YAML.

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

from mia_eval.config_loader import active_model_bundle, iter_grid, load_merged_config, apply_dot_overrides
from mia_eval.evaluation_common import (
    auc_direction as _auc_direction,
    jsonable as _jsonable,
    orient_scores_full as _orient_scores_full,
    split_masks as _split_masks,
)
from mia_eval.generation import generate_diverse_samples
from mia_eval.ground_truth import TrainingShingleIndex, build_index_from_hf
from mia_eval.labeling import label_jsonl, load_labeled
from mia_eval.mia_gt_pipeline import (
    annotate_samples_mia_gt,
    evaluate_mia_gt_jsonl,
    resolve_memtrace_rf_joblib,
)
from mia_eval.model_utils import load_causal_lm, pick_device, torch_dtype_from_str
from mia_eval.scoring_infilling import score_texts as infilling_scores
from mia_eval.scoring_memtrace import compute_feature_matrix, fit_rf_on_splits
from mia_eval.scoring_wbc import score_texts as wbc_scores


def _save_evaluation_artifacts(
    run_dir: Path,
    *,
    y: np.ndarray,
    sources: list,
    tr_m: np.ndarray,
    va_m: np.ndarray,
    te_m: np.ndarray,
    infilling: np.ndarray,
    wbc: np.ndarray,
    memtrace_p: np.ndarray,
    infilling_raw: np.ndarray | None = None,
    wbc_raw: np.ndarray | None = None,
) -> None:
    splits = {
        "train_indices": np.nonzero(tr_m)[0].astype(int).tolist(),
        "val_indices": np.nonzero(va_m)[0].astype(int).tolist(),
        "test_indices": np.nonzero(te_m)[0].astype(int).tolist(),
        "n_samples": int(len(y)),
    }
    with open(run_dir / "evaluation_splits.json", "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    out_scores = run_dir / "scores_per_sample.jsonl"
    with open(out_scores, "w", encoding="utf-8") as f:
        for i in range(len(y)):
            if tr_m[i]:
                sp = "train"
            elif va_m[i]:
                sp = "val"
            else:
                sp = "test"
            row = {
                "i": i,
                "label": int(y[i]),
                "source": sources[i],
                "split": sp,
                "infilling_score": float(infilling[i]),
                "wbc_score": float(wbc[i]),
                "memtrace_p_member": float(memtrace_p[i]),
            }
            if infilling_raw is not None:
                row["infilling_score_raw"] = float(infilling_raw[i])
            if wbc_raw is not None:
                row["wbc_score_raw"] = float(wbc_raw[i])
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {run_dir / 'evaluation_splits.json'}")
    print(f"Wrote {out_scores}")


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
        help=(
            "Comma list: build_index,generate,label,evaluate,all | "
            "generate,mia_annotate,mia_evaluate | all_mia_gt"
        ),
    )
    ap.add_argument("--set", action="append", default=[], help="Override e.g. active_model=pythia_2p8")
    args = ap.parse_args()

    defaults_path = Path(args.config)
    exp_path = Path(args.experiment) if args.experiment else None
    cfg = load_merged_config(defaults_path, exp_path)
    cfg = apply_dot_overrides(cfg, args.set)

    steps = {x.strip() for x in args.steps.lower().split(",")}
    if "all_mia_gt" in steps:
        steps.discard("all_mia_gt")
        steps |= {"generate", "mia_annotate", "mia_evaluate"}
    if "all" in steps:
        steps = {"build_index", "generate", "label", "evaluate"}

    out_root = Path(cfg.get("experiment", {}).get("output_dir", "mia_eval_outputs"))
    model_key = cfg.get("active_model", "gpt_neo_2p7")
    run_dir = out_root / model_key
    run_dir.mkdir(parents=True, exist_ok=True)

    bundle = active_model_bundle(cfg)
    gt_cfg = bundle.get("ground_truth") or {}

    if ("mia_annotate" in steps or "mia_evaluate" in steps) and not gt_cfg:
        gcfg = cfg.setdefault("generation", {})
        n_ex = int(gcfg.get("add_training_excerpts_members", 0))
        if n_ex > 0:
            print(
                "mia_gt pipeline: disabling add_training_excerpts_members (no ground_truth on this preset).",
                file=sys.stderr,
            )
            gcfg["add_training_excerpts_members"] = 0

    index_path = run_dir / "training_shingle_index.json"
    samples_path = run_dir / "samples.jsonl"
    labeled_path = run_dir / "samples_labeled.jsonl"
    results_path = run_dir / "results.json"
    mia_gt_path = run_dir / "samples_mia_gt.jsonl"
    results_mia_gt_path = run_dir / "results_mia_gt.json"

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

    if "mia_annotate" in steps:
        print("Annotating samples with MIA reference score triples (no shingle labels)...")
        n_done = annotate_samples_mia_gt(cfg, bundle, model_key, samples_path, mia_gt_path)
        print(f"MIA-GT annotated n={n_done} -> {mia_gt_path}")

    if "mia_evaluate" in steps:
        if not mia_gt_path.is_file():
            raise FileNotFoundError(f"Missing {mia_gt_path}; run mia_annotate after generate.")
        print("Evaluating cross-method agreement (Spearman, HP sensitivity)...")
        mia_res = evaluate_mia_gt_jsonl(mia_gt_path)
        mia_res["model"] = model_key
        mia_res["artifacts"] = {"samples_mia_gt": str(mia_gt_path.name)}
        block = (cfg.get("mia_gt_pipeline") or {}) or {}
        mia_res["hyperparams_used"] = {
            "open_model_infilling_primary": block.get("open_model_infilling"),
            "open_model_infilling_sensitivity": block.get("open_model_infilling_sensitivity"),
            "open_model_wbc_primary": block.get("open_model_wbc"),
            "open_model_wbc_sensitivity": block.get("open_model_wbc_sensitivity"),
            "memtrace_rf_joblib": str(resolve_memtrace_rf_joblib(cfg, model_key)),
            "memtrace_max_length": block.get("memtrace_max_length"),
            "morris2025_select": block.get("select"),
        }
        with open(results_mia_gt_path, "w", encoding="utf-8") as f:
            json.dump(_jsonable(mia_res), f, indent=2)
        print(json.dumps(_jsonable(mia_res), indent=2))
        print(f"Wrote {results_mia_gt_path}")

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
        val_f = float(ev.get("val_fraction", 0.2))
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
                best_inf = (auc_va, dict(params), float(auc_te), scores.copy())

        assert best_inf is not None
        s_inf_oriented = _orient_scores_full(y, va_m, best_inf[3])
        results["methods"]["infilling"] = {
            "best_params": _jsonable(best_inf[1]),
            "val_auc": best_inf[0],
            "test_auc": best_inf[2],
            "score_orientation": "higher_is_member_on_val_split",
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
            p2 = {
                k: m_wbc[k]
                for k in ("ensemble_variants", "use_ensemble", "ensemble_aggregate")
                if k in m_wbc
            }
            p2.update(params)
            if p2.get("window_sizes") is None:
                p2.pop("window_sizes", None)
            scores = np.array(
                wbc_scores(model_t, model_r, tok, texts, device, p2, max_length=max_len),
                dtype=np.float64,
            )
            auc_va, _ = _auc_direction(y[va_m], scores[va_m])
            if best_wbc is None or auc_va > best_wbc[0]:
                auc_te, _ = _auc_direction(y[te_m], scores[te_m])
                best_wbc = (auc_va, dict(p2), float(auc_te), scores.copy())

        assert best_wbc is not None
        s_wbc_oriented = _orient_scores_full(y, va_m, best_wbc[3])
        results["methods"]["wbc"] = {
            "best_params": _jsonable(best_wbc[1]),
            "val_auc": best_wbc[0],
            "test_auc": best_wbc[2],
            "score_orientation": "higher_is_member_on_val_split",
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
        best_mt_proba: np.ndarray | None = None
        for params in iter_grid(grid_mt):
            params = {**params, "max_length": mt_max_len}
            proba_all, meta = fit_rf_on_splits(X_mt, y, tr_m, va_m, te_m, params, rs)
            auc_va = meta["val_auc"]
            if best_mt is None or auc_va > best_mt[0]:
                best_mt = (auc_va, dict(params), meta["test_auc"])
                best_mt_proba = np.asarray(proba_all, dtype=np.float64).copy()

        assert best_mt is not None and best_mt_proba is not None
        bp = {k: v for k, v in best_mt[1].items() if k != "max_length"}
        results["methods"]["memtrace"] = {
            "best_params": _jsonable(bp),
            "val_auc": best_mt[0],
            "test_auc": best_mt[2],
            "max_length": mt_max_len,
            "score_orientation": "memtrace_p_member_is_RF_P(class=1)",
        }

        results["saved_artifacts"] = {
            "evaluation_splits": "evaluation_splits.json",
            "scores_per_sample": "scores_per_sample.jsonl",
        }

        _save_evaluation_artifacts(
            run_dir,
            y=y,
            sources=sources,
            tr_m=tr_m,
            va_m=va_m,
            te_m=te_m,
            infilling=s_inf_oriented,
            wbc=s_wbc_oriented,
            memtrace_p=best_mt_proba,
            infilling_raw=best_inf[3],
            wbc_raw=best_wbc[3],
        )

        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(_jsonable(results), f, indent=2)
        print(json.dumps(_jsonable(results), indent=2))


if __name__ == "__main__":
    main()
