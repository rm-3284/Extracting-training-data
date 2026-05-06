#!/usr/bin/env python3
"""
Summarize Carlini extraction outputs before labeling/evaluation.

Reads run folders under ``mia_eval_outputs/carlini_extract`` and computes:
- row counts and source breakdown,
- duplicate rates (exact text),
- text length stats,
- optional pairwise base vs post deltas if pairs are provided.

Useful when you only have ``run_meta.json`` + ``samples.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _map_remote_to_local(p: Path, repo_root: Path) -> Path:
    if p.exists():
        return p
    marker = '/mia_eval_outputs/carlini_extract/'
    s = str(p)
    i = s.find(marker)
    if i >= 0:
        rel = s[i + 1 :]
        cand = repo_root / rel
        if cand.exists():
            return cand
    # fallback: use filename-based suffix under local output dir
    try:
        parts = p.parts
        j = parts.index('carlini_extract')
        tail = Path(*parts[j + 1 :])
        cand = repo_root / 'mia_eval_outputs' / 'carlini_extract' / tail
        if cand.exists():
            return cand
    except ValueError:
        pass
    return p


def _safe_quantiles(vals: List[int]) -> Dict[str, float]:
    if not vals:
        return {'p50': 0.0, 'p90': 0.0, 'p99': 0.0}
    s = sorted(vals)

    def q(p: float) -> float:
        idx = int(round((len(s) - 1) * p))
        return float(s[idx])

    return {'p50': q(0.50), 'p90': q(0.90), 'p99': q(0.99)}


def summarize_run(run_dir: Path) -> Dict[str, Any]:
    meta_path = run_dir / 'run_meta.json'
    samples_path = run_dir / 'samples.jsonl'
    out: Dict[str, Any] = {
        'run_key': run_dir.name,
        'run_dir': str(run_dir),
        'has_meta': meta_path.exists(),
        'has_samples': samples_path.exists(),
    }

    if meta_path.exists():
        meta = _read_json(meta_path)
        out.update(
            {
                'project': meta.get('project'),
                'hf_model_id': meta.get('hf_model_id'),
                'variant': meta.get('variant'),
                'post_training': meta.get('post_training'),
                'generation': meta.get('generation'),
            }
        )

    if not samples_path.exists():
        out['status'] = 'missing_samples'
        return out

    rows = list(_iter_jsonl(samples_path))
    if not rows:
        out['status'] = 'empty_samples'
        out['n_rows'] = 0
        return out

    sources = [str(r.get('source', '')) for r in rows]
    texts = [str(r.get('text', '')) for r in rows]
    lengths_chars = [len(t) for t in texts]
    lengths_words = [len(t.split()) for t in texts]

    source_counts = Counter(sources)
    text_counts = Counter(texts)
    n = len(texts)
    n_unique = len(text_counts)
    n_dupe_rows = n - n_unique

    out['status'] = 'ok'
    out['n_rows'] = n
    out['source_counts'] = dict(sorted(source_counts.items(), key=lambda kv: kv[0]))
    out['duplicate_rows'] = n_dupe_rows
    out['duplicate_rate'] = float(n_dupe_rows / n)
    out['unique_text_ratio'] = float(n_unique / n)
    out['length_chars'] = {
        'mean': float(statistics.fmean(lengths_chars)),
        'min': int(min(lengths_chars)),
        'max': int(max(lengths_chars)),
        **_safe_quantiles(lengths_chars),
    }
    out['length_words'] = {
        'mean': float(statistics.fmean(lengths_words)),
        'min': int(min(lengths_words)),
        'max': int(max(lengths_words)),
        **_safe_quantiles(lengths_words),
    }
    return out


def _fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return '-'
    return f'{100.0 * x:.2f}%'


def _pair_delta(summary: Dict[str, Any], base_key: str, post_key: str) -> Dict[str, Any]:
    b = summary.get(base_key, {})
    p = summary.get(post_key, {})
    out = {'base': base_key, 'post': post_key}
    if b.get('status') != 'ok' or p.get('status') != 'ok':
        out['status'] = 'missing'
        return out
    out['status'] = 'ok'
    out['n_rows_base'] = b['n_rows']
    out['n_rows_post'] = p['n_rows']
    out['duplicate_rate_base'] = b['duplicate_rate']
    out['duplicate_rate_post'] = p['duplicate_rate']
    out['duplicate_rate_delta_post_minus_base'] = p['duplicate_rate'] - b['duplicate_rate']
    out['unique_text_ratio_base'] = b['unique_text_ratio']
    out['unique_text_ratio_post'] = p['unique_text_ratio']
    out['unique_text_ratio_delta_post_minus_base'] = p['unique_text_ratio'] - b['unique_text_ratio']
    return out


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description='Summarize carlini_extract outputs')
    p.add_argument(
        '--manifest',
        type=Path,
        default=root / 'mia_eval_outputs' / 'carlini_extract' / 'manifest.json',
        help='Path to manifest.json',
    )
    p.add_argument(
        '--output-json',
        type=Path,
        default=root / 'mia_eval_outputs' / 'carlini_extract' / 'summary_local.json',
        help='Where to write machine-readable summary',
    )
    p.add_argument(
        '--output-md',
        type=Path,
        default=root / 'mia_eval_outputs' / 'carlini_extract' / 'summary_local.md',
        help='Where to write markdown table summary',
    )
    p.add_argument(
        '--pairs',
        type=str,
        default='',
        help='Optional comma-separated base:post run_key pairs',
    )
    p.add_argument(
        '--run-root',
        type=Path,
        default=root / 'mia_eval_outputs' / 'carlini_extract',
        help='Directory containing per-run folders (used as fallback if manifest is missing/incomplete).',
    )
    p.add_argument(
        '--append-utc-to-output',
        action='store_true',
        help='Append UTC timestamp to output filenames to avoid overwriting prior summaries.',
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    runs: List[Dict[str, Any]] = []
    if args.manifest.exists():
        manifest = _read_json(args.manifest)
        runs = list(manifest.get('runs') or [])

    run_dirs: List[Path] = []
    for r in runs:
        p = r.get('samples_jsonl')
        if p:
            local_samples = _map_remote_to_local(Path(str(p)), repo_root)
            run_dirs.append(local_samples.parent)
        else:
            rk = r.get('run_key')
            if rk:
                run_dirs.append(args.run_root / rk)

    # Fallback / supplement: scan run_root for directories with run_meta.json or samples.jsonl.
    if args.run_root.exists():
        for d in sorted(args.run_root.iterdir()):
            if not d.is_dir():
                continue
            if d.name.startswith('.'):
                continue
            if (d / 'run_meta.json').exists() or (d / 'samples.jsonl').exists():
                run_dirs.append(d)

    # Deduplicate while preserving order.
    seen = set()
    uniq_dirs: List[Path] = []
    for d in run_dirs:
        if str(d) in seen:
            continue
        seen.add(str(d))
        uniq_dirs.append(d)

    summaries = [summarize_run(d) for d in uniq_dirs]
    by_key = {s['run_key']: s for s in summaries}

    pair_specs: List[Dict[str, Any]] = []
    if args.pairs.strip():
        for token in args.pairs.split(','):
            token = token.strip()
            if not token or ':' not in token:
                continue
            base, post = token.split(':', 1)
            pair_specs.append(_pair_delta(by_key, base.strip(), post.strip()))

    payload = {
        'manifest': str(args.manifest),
        'n_runs_manifest': len(runs),
        'n_runs_summarized': len(summaries),
        'runs': summaries,
        'pairs': pair_specs,
    }
    output_json = args.output_json
    output_md = args.output_md
    if args.append_utc_to_output:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        output_json = output_json.with_name(f'{output_json.stem}_{stamp}{output_json.suffix}')
        output_md = output_md.with_name(f'{output_md.stem}_{stamp}{output_md.suffix}')

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    lines: List[str] = []
    lines.append('# Carlini Extract Summary')
    lines.append('')
    lines.append('| run_key | status | variant | rows | duplicate_rate | sources | model |')
    lines.append('|---|---|---|---:|---:|---|---|')
    for s in summaries:
        src = s.get('source_counts', {})
        src_txt = ', '.join(f'{k}:{v}' for k, v in sorted(src.items())) if src else '-'
        lines.append(
            '| {rk} | {st} | {va} | {n} | {dr} | {src} | {mid} |'.format(
                rk=s.get('run_key', '-'),
                st=s.get('status', '-'),
                va=str(s.get('variant', '-')),
                n=s.get('n_rows', 0),
                dr=_fmt_pct(s.get('duplicate_rate')),
                src=src_txt,
                mid=str(s.get('hf_model_id', '-')),
            )
        )

    if pair_specs:
        lines.append('')
        lines.append('## Base vs Post Deltas')
        lines.append('')
        lines.append('| base | post | status | Δ duplicate_rate (post-base) | Δ unique_ratio (post-base) |')
        lines.append('|---|---|---|---:|---:|')
        for p in pair_specs:
            lines.append(
                '| {b} | {p} | {st} | {dd} | {du} |'.format(
                    b=p.get('base', '-'),
                    p=p.get('post', '-'),
                    st=p.get('status', '-'),
                    dd=_fmt_pct(p.get('duplicate_rate_delta_post_minus_base')),
                    du=_fmt_pct(p.get('unique_text_ratio_delta_post_minus_base')),
                )
            )


    with open(output_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    print(f'Wrote {output_json}')
    print(f'Wrote {output_md}')


if __name__ == '__main__':
    main()
