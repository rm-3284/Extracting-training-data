#!/usr/bin/env python3
"""
Add integer ``label`` ∈ {0, 1} to Carlini ``samples.jsonl`` rows.

**Default:** stream the **public training corpora** listed for each ``run_key`` in
``mia_eval/config/carlini_training_corpora.yaml`` (Hugging Face Hub — downloaded /
streamed over the internet), build a hashed character-shingle index from those
documents, then set ``label=1`` iff a long substring of the sample matches that
index (see ``labeling.label_jsonl`` and ``ground_truth.iter_training_documents``).

This approximates “text overlap with the **released** training mixture” for each
open model. It is **not** a proof that weights memorized a string; coverage is
limited by ``max_documents`` / ``max_shingles`` and Shingle parameters.

**Fallback:** ``--legacy-generic-corpora`` uses coarse project→dataset heuristics
(C4 / Pile / …) instead of the curated YAML.

Examples::

  python -m mia_eval.label_carlini_extract_samples \\
    --batch-carlini-root mia_eval_outputs/carlini_extract \\
    --save-index

  python -m mia_eval.label_carlini_extract_samples \\
    --run-dir mia_eval_outputs/carlini_extract/olmo2_7b_base \\
    --training-corpora-yaml mia_eval/config/carlini_training_corpora.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mia_eval.ground_truth import (
    TrainingShingleIndex,
    build_index_from_hf,
    build_index_from_training_sources,
)
from mia_eval.labeling import label_jsonl

DEFAULT_CORPORA_YAML = ROOT / "mia_eval" / "config" / "carlini_training_corpora.yaml"


def _parse_only(s: str | None) -> Optional[Set[str]]:
    if not s or not str(s).strip():
        return None
    return {x.strip() for x in s.split(",") if x.strip()}


# Legacy coarse map (only with ``--legacy-generic-corpora``).
_LEGACY_PROJECT_CORPUS: Dict[str, Dict[str, Any]] = {
    "Pythia": {
        "dataset_name": "EleutherAI/the_pile_deduplicated",
        "dataset_config": None,
        "text_field": "text",
    },
    "RedPajama": {
        "dataset_name": "togethercomputer/RedPajama-Data-1T",
        "dataset_config": "default",
        "text_field": "text",
    },
    "StarCoder2": {
        "dataset_name": "allenai/c4",
        "dataset_config": "en",
        "text_field": "text",
    },
    "OLMo2": {"dataset_name": "allenai/c4", "dataset_config": "en", "text_field": "text"},
    "OLMo1": {"dataset_name": "allenai/c4", "dataset_config": "en", "text_field": "text"},
    "LLM360": {"dataset_name": "allenai/c4", "dataset_config": "en", "text_field": "text"},
    "DCLM": {"dataset_name": "allenai/c4", "dataset_config": "en", "text_field": "text"},
}

_DEFAULT_GT_NUM = {
    "max_documents": 2000,
    "shingle_chars": 200,
    "max_shingles": 2000000,
    "min_match_chars": 150,
}


def _load_run_meta(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "run_meta.json"
    if not p.is_file():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_corpora_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_yaml_training_spec(
    run_dir: Path, corpora_path: Path
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Optional[str]]:
    meta = _load_run_meta(run_dir)
    run_key = str(meta.get("run_key") or run_dir.name)
    doc = _load_corpora_yaml(corpora_path)
    defaults = dict(doc.get("defaults") or {})
    runs = doc.get("run_training_corpora") or {}
    if run_key not in runs:
        raise KeyError(
            f"No entry run_training_corpora[{run_key!r}] in {corpora_path}. "
            f"Try --legacy-generic-corpora or add a block to the YAML. "
            f"Known keys: {sorted(runs.keys())}"
        )
    block = runs[run_key]
    sources = list(block.get("sources") or [])
    if not sources:
        raise ValueError(f"{run_key}: sources list is empty in {corpora_path}")
    note = block.get("note")
    return sources, defaults, str(note) if note else None


def _resolve_legacy_gt_kwargs(
    run_dir: Path,
    args: argparse.Namespace,
) -> Tuple[str, Optional[str], str, int, int, int, int]:
    meta = _load_run_meta(run_dir)
    project = str(meta.get("project") or "")
    base = dict(_DEFAULT_GT_NUM)
    corp = dict(_LEGACY_PROJECT_CORPUS.get(project, _LEGACY_PROJECT_CORPUS["OLMo2"]))
    base.update(corp)

    dataset_name = args.dataset_name or base["dataset_name"]
    dataset_config = args.dataset_config if args.dataset_config is not None else base.get("dataset_config")
    text_field = args.text_field or base["text_field"]
    max_documents = args.max_documents if args.max_documents is not None else int(base["max_documents"])
    shingle_chars = args.shingle_chars if args.shingle_chars is not None else int(base["shingle_chars"])
    max_shingles = args.max_shingles if args.max_shingles is not None else int(base["max_shingles"])
    min_match_chars = args.min_match_chars if args.min_match_chars is not None else int(base["min_match_chars"])

    if dataset_config == "":
        dataset_config = None

    return (
        dataset_name,
        dataset_config,
        text_field,
        max_documents,
        shingle_chars,
        max_shingles,
        min_match_chars,
    )


def _write_constant_labels(samples_path: Path, out_path: Path, value: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(samples_path, "r", encoding="utf-8") as fin, open(
        out_path, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            r["label"] = int(value)
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")


def _discover_carlini_dirs(carlini_root: Path) -> List[Path]:
    out: List[Path] = []
    for p in sorted(carlini_root.iterdir()):
        if p.is_dir() and (p / "samples.jsonl").is_file():
            out.append(p)
    return out


def _label_one_run(
    run_dir: Path,
    args: argparse.Namespace,
) -> None:
    samples_path = run_dir / "samples.jsonl"
    out_path = run_dir / (args.output_name or "samples_labeled.jsonl")
    index_path = run_dir / "training_shingle_index.json"
    manifest_path = run_dir / "training_corpus_manifest.json"

    if args.skip_existing and out_path.is_file():
        print(f"[skip] {run_dir.name}: exists {out_path.name}", file=sys.stderr)
        return

    if args.constant_label is not None:
        print(
            f"[label] {run_dir.name}: constant label={int(args.constant_label)} → {out_path}",
            file=sys.stderr,
        )
        _write_constant_labels(samples_path, out_path, int(args.constant_label))
        return

    if args.reuse_index and index_path.is_file():
        print(f"[index] {run_dir.name}: load {index_path.name}", file=sys.stderr)
        idx = TrainingShingleIndex.load(index_path)
        mmc_final = args.min_match_chars
        if mmc_final is None:
            if args.legacy_generic_corpora:
                _, _, _, _, _, _, mmc = _resolve_legacy_gt_kwargs(run_dir, args)
                mmc_final = mmc
            else:
                doc = _load_corpora_yaml(Path(args.training_corpora_yaml))
                mmc_final = int((doc.get("defaults") or {}).get("min_match_chars", 150))
    else:
        if args.legacy_generic_corpora:
            ds_name, ds_cfg, text_field, max_doc, sc, ms, mmc = _resolve_legacy_gt_kwargs(
                run_dir, args
            )
            print(
                f"[index] {run_dir.name}: LEGACY single-stream {ds_name!r} config={ds_cfg!r}",
                file=sys.stderr,
            )
            idx = build_index_from_hf(
                ds_name, ds_cfg, text_field, max_doc, sc, ms
            )
            mmc_final = args.min_match_chars if args.min_match_chars is not None else mmc
        else:
            corp_path = Path(args.training_corpora_yaml)
            sources, corp_defaults, note = _resolve_yaml_training_spec(run_dir, corp_path)
            sc = (
                args.shingle_chars
                if args.shingle_chars is not None
                else int(corp_defaults.get("shingle_chars", 200))
            )
            ms = (
                args.max_shingles
                if args.max_shingles is not None
                else int(corp_defaults.get("max_shingles", 2_000_000))
            )
            mmc_def = int(corp_defaults.get("min_match_chars", 150))
            mmc_final = (
                args.min_match_chars
                if args.min_match_chars is not None
                else mmc_def
            )
            if note:
                print(f"[note] {run_dir.name}: {note}", file=sys.stderr)
            print(
                f"[index] {run_dir.name}: {len(sources)} streaming source(s) from {corp_path.name}",
                file=sys.stderr,
            )
            idx = build_index_from_training_sources(sources, sc, ms)
            manifest = {
                "training_corpora_yaml": str(corp_path.resolve()),
                "run_key": _load_run_meta(run_dir).get("run_key", run_dir.name),
                "sources": sources,
                "shingle_chars": sc,
                "max_shingles_cap": ms,
                "min_match_chars": mmc_final,
            }
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
            print(f"[manifest] wrote {manifest_path}", file=sys.stderr)

        if args.save_index:
            idx.save(index_path)
            print(f"[index] wrote {index_path} ({len(idx)} shingles)", file=sys.stderr)

    print(
        f"[label] {run_dir.name}: min_match_chars={mmc_final} → {out_path}",
        file=sys.stderr,
    )
    label_jsonl(samples_path, idx, int(mmc_final), out_path)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Add label 0/1 via overlap with streamed open training corpora (HF)."
    )
    ap.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Single samples.jsonl (parent dir must match a Carlini run folder with run_meta).",
    )
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Carlini run directory with samples.jsonl + run_meta.json.",
    )
    ap.add_argument(
        "--batch-carlini-root",
        type=Path,
        default=None,
        help="Process each */samples.jsonl under this directory.",
    )
    ap.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated run_key names when batching.",
    )
    ap.add_argument(
        "--output-name",
        type=str,
        default="samples_labeled.jsonl",
        help="Output JSONL filename inside each run directory.",
    )
    ap.add_argument(
        "--training-corpora-yaml",
        type=str,
        default=str(DEFAULT_CORPORA_YAML),
        help="Per-run training corpus definitions (default: carlini_training_corpora.yaml).",
    )
    ap.add_argument(
        "--legacy-generic-corpora",
        action="store_true",
        help="Ignore YAML; use legacy single generic dataset per project (C4/Pile/…).",
    )
    ap.add_argument(
        "--constant-label",
        type=int,
        choices=(0, 1),
        default=None,
        help="Set every row to this label (skip indexing).",
    )
    ap.add_argument(
        "--dataset-name",
        type=str,
        default="",
        help="(Legacy mode only) Override HF dataset id.",
    )
    ap.add_argument(
        "--dataset-config",
        type=str,
        default=None,
        help='(Legacy mode) HF config; use "" for null.',
    )
    ap.add_argument(
        "--text-field",
        type=str,
        default="",
        help="(Legacy mode) Text column name.",
    )
    ap.add_argument("--max-documents", type=int, default=None)
    ap.add_argument("--shingle-chars", type=int, default=None)
    ap.add_argument("--max-shingles", type=int, default=None)
    ap.add_argument("--min-match-chars", type=int, default=None)
    ap.add_argument(
        "--save-index",
        action="store_true",
        help="Save training_shingle_index.json after building.",
    )
    ap.add_argument(
        "--reuse-index",
        action="store_true",
        help="Load existing training_shingle_index.json instead of streaming corpora.",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip run if output JSONL already exists.",
    )
    args = ap.parse_args()

    if args.dataset_config == "":
        args.dataset_config = None

    if not args.legacy_generic_corpora and not Path(args.training_corpora_yaml).is_file():
        raise SystemExit(
            f"Missing --training-corpora-yaml file: {args.training_corpora_yaml}\n"
            "Use --legacy-generic-corpora for old heuristics, or restore the YAML."
        )

    only = _parse_only(args.only)
    runs: List[Path] = []

    if args.batch_carlini_root is not None:
        runs = _discover_carlini_dirs(args.batch_carlini_root)
        if only is not None:
            runs = [p for p in runs if p.name in only]
    elif args.run_dir is not None:
        runs = [args.run_dir]
    elif args.input is not None:
        runs = [args.input.parent]
        if args.input.name != "samples.jsonl":
            raise SystemExit("--input must be named samples.jsonl")
    else:
        raise SystemExit("Provide --batch-carlini-root, --run-dir, or --input")

    if not runs:
        msg = "No runs to process."
        if only:
            msg += " (--only left nothing matching.)"
        raise SystemExit(msg)

    failures: List[str] = []
    for run_dir in runs:
        try:
            _label_one_run(run_dir.resolve(), args)
        except Exception as e:
            failures.append(f"{run_dir.name}: {e}")
            print(f"[fail] {run_dir.name}: {e}", file=sys.stderr)

    if failures:
        raise SystemExit("Failures:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
