"""Carlini-style generation: top-k, temperature decay (paper §5.1.1), optional nucleus."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import torch
from tqdm import tqdm

from .ground_truth import stream_texts
from .model_utils import load_causal_lm, pick_device, torch_dtype_from_str

try:
    from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
except ImportError:
    from transformers import LogitsProcessor, LogitsProcessorList


_DECAY_MAP = {
    1: 10.0,
    2: 9.53,
    3: 9.06,
    4: 8.59,
    5: 8.12,
    6: 7.65,
    7: 7.18,
    8: 6.71,
    9: 6.24,
    10: 5.77,
    11: 5.30,
    12: 4.83,
    13: 4.36,
    14: 3.89,
    15: 3.42,
    16: 2.95,
    17: 2.49,
    18: 2.01,
    19: 1.54,
    20: 1.0,
}


class DecayingTemperatureLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.Tensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cur_len = int(input_ids.shape[-1])
        t = float(_DECAY_MAP.get(cur_len, 1.0))
        return scores / max(t, 1e-6)


def _generate_batch(
    model,
    tokenizer,
    device: torch.device,
    batch_size: int,
    seq_len: int,
    *,
    top_k: int,
    top_p: float,
    do_sample: bool,
    logits_processors: LogitsProcessorList | None,
    temperature: float = 1.0,
) -> List[str]:
    prompts = [tokenizer.eos_token or tokenizer.pad_token or ""]
    # Repeat single prompt type like the extraction repo (EOS-only prompt).
    prompts = prompts * batch_size
    enc = tokenizer(prompts, return_tensors="pt", padding=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    gen_kwargs: Dict[str, Any] = dict(
        max_length=seq_len,
        do_sample=do_sample,
        top_k=top_k if top_k > 0 else 50,
        top_p=top_p,
        temperature=temperature,
        pad_token_id=tokenizer.pad_token_id,
    )
    if logits_processors is not None:
        gen_kwargs["logits_processor"] = logits_processors
        gen_kwargs["renormalize_logits"] = True
    with torch.inference_mode():
        out = model.generate(**enc, **gen_kwargs)
    return tokenizer.batch_decode(out, skip_special_tokens=True)


def _run_strategy(
    model,
    tokenizer,
    device: torch.device,
    n: int,
    bs: int,
    seq_len: int,
    *,
    source: str,
    top_k: int,
    top_p: float,
    logits_processors: LogitsProcessorList | None,
    temperature: float = 1.0,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    need = n
    while need > 0:
        b = min(bs, need)
        texts = _generate_batch(
            model,
            tokenizer,
            device,
            b,
            seq_len,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            logits_processors=logits_processors,
            temperature=temperature,
        )
        for t in texts:
            rows.append({"text": t, "source": source})
        need = n - len(rows)
    return rows[:n]


def generate_diverse_samples(
    cfg: Dict[str, Any],
    model_bundle: Dict[str, Any],
    out_path: Path,
) -> Path:
    gcfg = cfg.get("generation") or {}
    exp = cfg.get("experiment") or {}
    device = pick_device(exp.get("device"))
    dtype = torch_dtype_from_str(model_bundle.get("torch_dtype"))

    target = model_bundle["target_model"]
    tokenizer_id = model_bundle.get("tokenizer") or target
    model, tokenizer = load_causal_lm(target, tokenizer_id, device, dtype)

    seq_len = int(gcfg.get("seq_len", 256))
    n_per = int(gcfg.get("num_samples_per_strategy", 32))
    bs = int(gcfg.get("batch_size", 4))
    top_k = int(gcfg.get("top_k", 40))
    top_p = float(gcfg.get("top_p", 1.0))

    records: List[Dict[str, Any]] = []

    records.extend(
        _run_strategy(
            model,
            tokenizer,
            device,
            n_per,
            bs,
            seq_len,
            source="top_k",
            top_k=top_k,
            top_p=top_p,
            logits_processors=None,
        )
    )

    td = gcfg.get("temperature_decay") or {}
    if td.get("enabled", True):
        lp = LogitsProcessorList([DecayingTemperatureLogitsProcessor()])
        records.extend(
            _run_strategy(
                model,
                tokenizer,
                device,
                n_per,
                bs,
                seq_len,
                source="temperature_decay",
                top_k=top_k,
                top_p=top_p,
                logits_processors=lp,
            )
        )

    nuc = gcfg.get("nucleus") or {}
    if nuc.get("enabled"):
        n_nuc = int(nuc.get("num_samples", n_per))
        records.extend(
            _run_strategy(
                model,
                tokenizer,
                device,
                n_nuc,
                bs,
                seq_len,
                source="nucleus",
                top_k=0,
                top_p=float(nuc.get("top_p", 0.95)),
                logits_processors=None,
                temperature=float(nuc.get("temperature", 1.0)),
            )
        )

    gt = model_bundle.get("ground_truth") or {}
    n_train_ex = int(gcfg.get("add_training_excerpts_members", 0))
    if n_train_ex > 0:
        ds_name = gt.get("dataset_name")
        ds_conf = gt.get("dataset_config")
        text_field = gt.get("text_field", "text")
        for i, doc in enumerate(
            stream_texts(ds_name, ds_conf, text_field, max_documents=n_train_ex * 3)
        ):
            if len(doc) < 400:
                continue
            chunk = doc[100:360]
            records.append({"text": chunk, "source": "training_excerpt"})
            if len([r for r in records if r["source"] == "training_excerpt"]) >= n_train_ex:
                break

    n_wiki = int(gcfg.get("add_wikipedia_nonmembers", 0))
    if n_wiki > 0:
        wname = gcfg.get("wikipedia_dataset", "wikipedia")
        wconf = gcfg.get("wikipedia_config", "20220301.en")
        wsplit = gcfg.get("wikipedia_split", "train")
        try:
            from datasets import load_dataset

            wds = load_dataset(wname, wconf, split=wsplit, streaming=True, trust_remote_code=True)
        except Exception:
            wds = load_dataset(wname, wconf, split=wsplit, streaming=True)
        got = 0
        for row in wds:
            t = row.get("text", "")
            if len(t) < 200:
                continue
            records.append({"text": t[:800], "source": "wikipedia_ood"})
            got += 1
            if got >= n_wiki:
                break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out_path
