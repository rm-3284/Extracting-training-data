"""Carlini-style generation: top-k, temperature decay (paper §5.1.1), optional nucleus,
and optional ``memorization_detection`` decoders (baseline / risk-aware / WBC)."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None  # type: ignore[misc, assignment]

import torch

from .carlini_sample_sources import MEMORIZATION_DETECTION_SOURCES
from .ground_truth import stream_texts
from .model_utils import load_causal_lm, pick_device, torch_dtype_from_str


def _carlini_sampling_enabled(gcfg: Dict[str, Any]) -> bool:
    """When false, skip top_k / internet_prefix / temperature_decay / nucleus (memorization_detection only)."""
    cs = gcfg.get("carlini_sampling") or {}
    if not isinstance(cs, dict):
        return True
    return bool(cs.get("enabled", True))


def _carlini_log(msg: str, *, verbose: bool = False) -> None:
    """Progress / debug lines (always flush for Slurm). Use CARLINI_VERBOSE=1 for extra detail."""
    if verbose and os.environ.get("CARLINI_VERBOSE", "").strip().lower() not in (
        "1",
        "true",
        "yes",
    ):
        return
    print(msg, flush=True)


def _carlini_pbar(total: int, desc: str):
    if tqdm is None or os.environ.get("CARLINI_NO_TQDM", "").strip() in ("1", "true", "yes"):
        return None
    file = sys.stderr if getattr(sys.stderr, "isatty", lambda: False)() else sys.stdout
    return tqdm(total=total, desc=desc, unit="sample", leave=True, file=file)


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


def _internet_applies_to(ipc: Dict[str, Any], strategy: str) -> bool:
    """
    Carlini et al. (§5.1.2): after a web prefix, they continue with **top-n sampling**
    (§4.1), not decaying-temperature sampling. Default ``apply_to`` matches that
    (internet only paired with ``top_k``). Set ``apply_to: [top_k, temperature_decay]``
    to also run internet + temperature decay (not in the paper's main three-way split).
    """
    raw = ipc.get("apply_to")
    if raw is None:
        return strategy == "top_k"
    if isinstance(raw, str):
        seq = [raw]
    else:
        seq = list(raw)
    return strategy in seq


# When ``allenai/c4`` streaming fails (cluster network, Hub gzip resolution), try these.
_DEFAULT_INTERNET_FALLBACKS: Tuple[Tuple[str, Optional[str], str], ...] = (
    ("wikipedia", "20220301.en", "text"),
)


def _internet_text_stream(ipc: Dict[str, Any]) -> Iterator[str]:
    """Yield long text lines from a streaming HF dataset (web-like / C4-style).

    Tries the configured dataset first, then optional YAML ``fallback_datasets``, then
    built-in fallbacks. Streaming ``allenai/c4`` often raises ``FileNotFoundError`` on
    compute nodes; Wikipedia English is a reliable substitute for long-document prefixes.
    """
    from datasets import load_dataset

    split = str(ipc.get("split", "train"))
    min_chars = int(ipc.get("min_doc_chars", 80))

    primary_name = str(ipc.get("dataset_name", "allenai/c4"))
    primary_conf = ipc.get("dataset_config")
    if primary_conf is None or primary_conf == "null":
        primary_conf = None
    primary_field = str(ipc.get("text_field", "text"))

    attempts: List[Tuple[str, Optional[str], str]] = [
        (primary_name, primary_conf, primary_field),
    ]
    raw_fb = ipc.get("fallback_datasets")
    if isinstance(raw_fb, list):
        for item in raw_fb:
            if not isinstance(item, dict):
                continue
            dn = item.get("dataset_name")
            if not dn:
                continue
            dc = item.get("dataset_config")
            if dc is None or dc == "null":
                dc = None
            tf = str(item.get("text_field", "text"))
            attempts.append((str(dn), dc, tf))
    for trip in _DEFAULT_INTERNET_FALLBACKS:
        attempts.append(trip)

    # De-dupe by (hub id, config) keeping first text_field
    seen: set = set()
    uniq: List[Tuple[str, Optional[str], str]] = []
    for name, conf, field in attempts:
        key = (name, conf)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((name, conf, field))

    last_err: Optional[BaseException] = None
    for name, conf, field in uniq:
        try:
            kwargs = dict(split=split, streaming=True)
            try:
                ds = load_dataset(name, conf, trust_remote_code=True, **kwargs)
            except Exception:
                ds = load_dataset(name, conf, **kwargs)
            for row in ds:
                t = row.get(field, "")
                if isinstance(t, str) and len(t) >= min_chars:
                    yield t
            return
        except Exception as e:
            last_err = e
            continue

    hint = (
        "internet_prefix: could not open any streaming dataset (network / Hub cache?). "
        "Set HF_HOME, ensure compute nodes reach huggingface.co, or add "
        "``internet_prefix.fallback_datasets`` in YAML. "
        f"Last error: {last_err!r}"
    )
    raise RuntimeError(hint) from last_err


def _next_prefix_id_batch(
    tokenizer,
    text_iter: Iterator[str],
    prefix_tokens: int,
    batch_size: int,
    *,
    max_attempts_factor: int = 400,
) -> Optional[torch.LongTensor]:
    """
    Build ``batch_size`` prefix sequences, each exactly ``prefix_tokens`` tokens
    (Carlini Common Crawl-style fixed-length prompts for batched generate).
    """
    rows: List[torch.Tensor] = []
    attempts = 0
    max_attempts = max(batch_size * max_attempts_factor, 1000)
    while len(rows) < batch_size and attempts < max_attempts:
        attempts += 1
        try:
            text = next(text_iter)
        except StopIteration:
            return None
        enc = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=prefix_tokens,
            return_tensors="pt",
        )
        ids = enc["input_ids"][0]
        if int(ids.shape[0]) < prefix_tokens:
            continue
        rows.append(ids[:prefix_tokens].to(torch.long))
    if len(rows) < batch_size:
        return None
    return torch.stack(rows, dim=0)


def _generate_batch_from_input_ids(
    model,
    tokenizer,
    device: torch.device,
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    seq_len: int,
    *,
    top_k: int,
    top_p: float,
    do_sample: bool,
    logits_processors: LogitsProcessorList | None,
    temperature: float = 1.0,
) -> List[str]:
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    gen_kwargs: Dict[str, Any] = dict(
        max_length=seq_len,
        do_sample=do_sample,
        top_k=top_k if top_k > 0 else 50,
        top_p=top_p,
        temperature=temperature,
        pad_token_id=tokenizer.pad_token_id,
        attention_mask=attention_mask,
    )
    if logits_processors is not None:
        gen_kwargs["logits_processor"] = logits_processors
        gen_kwargs["renormalize_logits"] = True
    with torch.inference_mode():
        out = model.generate(input_ids=input_ids, **gen_kwargs)
    return tokenizer.batch_decode(out, skip_special_tokens=True)


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
    input_ids = enc["input_ids"].to(device)
    attn = enc.get("attention_mask")
    if attn is None:
        attn = torch.ones_like(input_ids, device=device)
    else:
        attn = attn.to(device)
    return _generate_batch_from_input_ids(
        model,
        tokenizer,
        device,
        input_ids,
        attn,
        seq_len,
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample,
        logits_processors=logits_processors,
        temperature=temperature,
    )


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
    pbar = _carlini_pbar(n, desc=f"generate:{source}")
    try:
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
            if pbar is not None:
                pbar.update(len(texts))
    finally:
        if pbar is not None:
            pbar.close()
    return rows[:n]


def _run_strategy_internet(
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
    temperature: float,
    text_iter: Iterator[str],
    prefix_tokens: int,
    prefix_chars_max: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    need = n
    pbar = _carlini_pbar(n, desc=f"generate:{source}")
    try:
        while need > 0:
            b = min(bs, need)
            batch_ids = _next_prefix_id_batch(tokenizer, text_iter, prefix_tokens, b)
            if batch_ids is None:
                break
            attn = torch.ones(batch_ids.shape, dtype=torch.long)
            texts = _generate_batch_from_input_ids(
                model,
                tokenizer,
                device,
                batch_ids,
                attn,
                seq_len,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                logits_processors=logits_processors,
                temperature=temperature,
            )
            for i, t in enumerate(texts):
                pfx = tokenizer.decode(batch_ids[i], skip_special_tokens=True)
                if prefix_chars_max > 0:
                    pfx = pfx[:prefix_chars_max]
                rows.append(
                    {
                        "text": t,
                        "source": source,
                        "prompt_prefix": pfx,
                        "prefix_tokens": int(prefix_tokens),
                    }
                )
            need = n - len(rows)
            if pbar is not None:
                pbar.update(len(texts))
    finally:
        if pbar is not None:
            pbar.close()
    if len(rows) < n:
        raise RuntimeError(
            f"internet_prefix: only collected {len(rows)}/{n} samples "
            "(dataset exhausted or too many short docs)"
        )
    return rows[:n]


def _import_memorization_detection():
    """Load ``memorization_detection/memorization_detection.py`` without package name clash."""
    root = Path(__file__).resolve().parents[1]
    path = root / "memorization_detection" / "memorization_detection.py"
    if not path.is_file():
        raise FileNotFoundError(
            f"memorization_detection enabled but file missing: {path} "
            "(clone must include memorization_detection/)."
        )
    spec = importlib.util.spec_from_file_location("mia_eval_memorization_detection_impl", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _eos_like_prompt_and_max_new(tokenizer, seq_len: int) -> Tuple[str, int]:
    """Match Carlini empty-prompt style: EOS (or pad) only, then fill up to ``seq_len`` tokens total."""
    p = tokenizer.eos_token or tokenizer.pad_token or ""
    enc = tokenizer(p, return_tensors="pt", add_special_tokens=False)
    plen = int(enc["input_ids"].shape[1])
    # ``generate_*`` APIs extend by max_new_tokens; total length ≈ plen + max_new.
    return p, max(8, int(seq_len) - plen)


def _append_memorization_detection_records(
    records: List[Dict[str, Any]],
    *,
    md_cfg: Dict[str, Any],
    model: Any,
    tokenizer: Any,
    device: torch.device,
    dtype: Optional[torch.dtype],
    model_bundle: Dict[str, Any],
    seq_len: int,
) -> None:
    """Append samples from ``memorization_detection`` decoders (baseline / risk-aware / WBC)."""
    strategies = list(md_cfg.get("strategies") or ())
    if not strategies:
        strategies = [
            "memorization_baseline",
            "memorization_risk_fast",
            "memorization_risk_slow",
        ]
    unknown = [s for s in strategies if s not in MEMORIZATION_DETECTION_SOURCES]
    if unknown:
        raise ValueError(
            f"memorization_detection.strategies unknown keys {unknown!r}; "
            f"allowed: {list(MEMORIZATION_DETECTION_SOURCES)}"
        )

    n_per = int(md_cfg.get("num_samples_per_strategy", 0))
    if n_per <= 0:
        raise ValueError("memorization_detection.num_samples_per_strategy must be positive when enabled")

    mod = _import_memorization_detection()
    prompt, max_new = _eos_like_prompt_and_max_new(tokenizer, seq_len)
    decode_top_k = int(md_cfg.get("decode_top_k", 20))
    temperature = float(md_cfg.get("temperature", 1.0))
    gate_gamma = float(md_cfg.get("gate_gamma", 5.0))
    risk_every = int(md_cfg.get("risk_every", 1))
    lam_fast = float(md_cfg.get("lambda_penalty_fast", 0.3))
    lam_slow = float(md_cfg.get("lambda_penalty_slow", 0.5))
    wbc_lambda = float(md_cfg.get("wbc_lambda", 0.5))
    wbc_infilling_lambda = float(md_cfg.get("wbc_infilling_lambda", 0.3))
    wbc_gate_gamma = md_cfg.get("wbc_gate_gamma")
    wbc_gate_every = int(md_cfg.get("wbc_gate_every", 4))

    ref_model = None
    ref_tokenizer = None
    if "memorization_wbc" in strategies:
        ref_id = md_cfg.get("reference_model") or model_bundle.get("reference_model")
        if not ref_id:
            _carlini_log(
                "[generate] memorization_detection: memorization_wbc listed but no "
                "``reference_model`` in YAML (set generation.memorization_detection.reference_model "
                "or model_bundle.reference_model); skipping memorization_wbc.",
                verbose=False,
            )
            strategies = [s for s in strategies if s != "memorization_wbc"]
        else:
            ref_tok = md_cfg.get("reference_tokenizer") or ref_id
            _carlini_log(
                f"[generate] memorization_detection: loading reference {ref_id!r} for WBC …",
                verbose=False,
            )
            ref_model, ref_tokenizer = load_causal_lm(str(ref_id), str(ref_tok), device, dtype)

    try:
        for strat in strategies:
            desc = f"generate:{strat}"
            pbar = _carlini_pbar(n_per, desc=desc)
            try:
                for _ in range(n_per):
                    if strat == "memorization_baseline":
                        text = mod.generate_baseline(
                            prompt,
                            model,
                            tokenizer,
                            max_new_tokens=max_new,
                            temperature=temperature,
                        )
                    elif strat == "memorization_risk_fast":
                        text = mod.generate_risk_aware(
                            prompt,
                            model,
                            tokenizer,
                            max_new_tokens=max_new,
                            top_k=decode_top_k,
                            lambda_penalty=lam_fast,
                            temperature=temperature,
                            mode="fast",
                            gate_gamma=gate_gamma,
                            risk_every=risk_every,
                            reference_model=None,
                            reference_tokenizer=None,
                        )
                    elif strat == "memorization_risk_slow":
                        text = mod.generate_risk_aware(
                            prompt,
                            model,
                            tokenizer,
                            max_new_tokens=max_new,
                            top_k=decode_top_k,
                            lambda_penalty=lam_slow,
                            temperature=temperature,
                            mode="slow",
                            gate_gamma=gate_gamma,
                            risk_every=risk_every,
                            reference_model=None,
                            reference_tokenizer=None,
                        )
                    elif strat == "memorization_wbc":
                        assert ref_model is not None
                        text = mod.generate_risk_aware(
                            prompt,
                            model,
                            tokenizer,
                            max_new_tokens=max_new,
                            top_k=decode_top_k,
                            temperature=temperature,
                            mode="wbc",
                            gate_gamma=gate_gamma,
                            risk_every=risk_every,
                            reference_model=ref_model,
                            reference_tokenizer=ref_tokenizer,
                            wbc_lambda=wbc_lambda,
                            wbc_infilling_lambda=wbc_infilling_lambda,
                            wbc_gate_gamma=wbc_gate_gamma,
                            wbc_gate_every=wbc_gate_every,
                        )
                    else:
                        raise RuntimeError(f"unhandled strategy {strat!r}")
                    records.append({"text": text, "source": strat})
                    if pbar is not None:
                        pbar.update(1)
            finally:
                if pbar is not None:
                    pbar.close()
            _carlini_log(
                f"[generate] {strat} done: {len([r for r in records if r.get('source')==strat])} rows",
                verbose=False,
            )
    finally:
        if ref_model is not None:
            del ref_model
        if ref_tokenizer is not None:
            del ref_tokenizer
        if device.type == "cuda":
            torch.cuda.empty_cache()


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
    _carlini_log(
        f"[generate] run start model={target!r} tokenizer={tokenizer_id!r} "
        f"device={device} dtype={dtype} out={out_path}",
        verbose=False,
    )
    if "olmo" in target.lower():
        _carlini_log(
            "[generate] OLMo path: remote-code + compat shims in load_causal_lm; "
            "set CARLINI_VERBOSE=1 for tokenizer/model step logs.",
            verbose=False,
        )

    _carlini_log("[generate] loading tokenizer + model weights ...", verbose=False)
    model, tokenizer = load_causal_lm(target, tokenizer_id, device, dtype)
    try:
        param_dev = next(model.parameters()).device
    except StopIteration:
        param_dev = device
    _carlini_log(
        f"[generate] model ready: {type(model).__name__} params_device={param_dev}",
        verbose=False,
    )

    seq_len = int(gcfg.get("seq_len", 256))
    n_per = int(gcfg.get("num_samples_per_strategy", 32))
    bs = int(gcfg.get("batch_size", 4))
    top_k = int(gcfg.get("top_k", 40))
    top_p = float(gcfg.get("top_p", 1.0))

    records: List[Dict[str, Any]] = []

    carlini_on = _carlini_sampling_enabled(gcfg)
    ipc = gcfg.get("internet_prefix") or {}
    internet_on = bool(ipc.get("enabled", False))
    text_iter: Optional[Iterator[str]] = None
    prefix_tokens = int(ipc.get("prefix_tokens", 10))
    prefix_chars_max = int(ipc.get("prefix_chars_max", 512))
    n_per_inet = n_per
    if ipc.get("num_samples_per_strategy") is not None:
        n_per_inet = int(ipc["num_samples_per_strategy"])
    if carlini_on:
        if internet_on:
            _carlini_log("[generate] internet_prefix: opening text stream ...", verbose=False)
            text_iter = _internet_text_stream(ipc)
            skip0 = int(ipc.get("skip_initial_documents", 0))
            for _ in range(max(skip0, 0)):
                try:
                    next(text_iter)
                except StopIteration:
                    raise RuntimeError("internet_prefix: skip_initial_documents exceeds stream") from None

        _carlini_log(
            f"[generate] strategy top_k: target {n_per} samples (batch_size={bs}, seq_len={seq_len})",
            verbose=False,
        )
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
        _carlini_log(f"[generate] top_k done: {len(records)} rows so far", verbose=False)

    if carlini_on and internet_on and text_iter is not None and _internet_applies_to(ipc, "top_k"):
        _carlini_log(
            f"[generate] strategy top_k_internet: target {n_per_inet} samples "
            f"(prefix_tokens={prefix_tokens})",
            verbose=False,
        )
        records.extend(
            _run_strategy_internet(
                model,
                tokenizer,
                device,
                n_per_inet,
                bs,
                seq_len,
                source="top_k_internet",
                top_k=top_k,
                top_p=top_p,
                logits_processors=None,
                temperature=1.0,
                text_iter=text_iter,
                prefix_tokens=prefix_tokens,
                prefix_chars_max=prefix_chars_max,
            )
        )
        _carlini_log(f"[generate] top_k_internet done: {len(records)} rows so far", verbose=False)

    td = gcfg.get("temperature_decay") or {}
    if carlini_on and td.get("enabled", True):
        _carlini_log(
            f"[generate] strategy temperature_decay: target {n_per} samples",
            verbose=False,
        )
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
        _carlini_log(f"[generate] temperature_decay done: {len(records)} rows so far", verbose=False)

        if internet_on and text_iter is not None and _internet_applies_to(
            ipc, "temperature_decay"
        ):
            _carlini_log(
                f"[generate] strategy temperature_decay_internet: target {n_per_inet} samples",
                verbose=False,
            )
            records.extend(
                _run_strategy_internet(
                    model,
                    tokenizer,
                    device,
                    n_per_inet,
                    bs,
                    seq_len,
                    source="temperature_decay_internet",
                    top_k=top_k,
                    top_p=top_p,
                    logits_processors=lp,
                    temperature=1.0,
                    text_iter=text_iter,
                    prefix_tokens=prefix_tokens,
                    prefix_chars_max=prefix_chars_max,
                )
            )
            _carlini_log(
                f"[generate] temperature_decay_internet done: {len(records)} rows so far",
                verbose=False,
            )

    nuc = gcfg.get("nucleus") or {}
    if carlini_on and nuc.get("enabled"):
        n_nuc = int(nuc.get("num_samples", n_per))
        _carlini_log(f"[generate] strategy nucleus: target {n_nuc} samples", verbose=False)
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
        _carlini_log(f"[generate] nucleus done: {len(records)} rows so far", verbose=False)

        if internet_on and text_iter is not None and _internet_applies_to(ipc, "nucleus"):
            n_nuc_i = int(nuc.get("num_samples_internet", n_nuc))
            _carlini_log(
                f"[generate] strategy nucleus_internet: target {n_nuc_i} samples",
                verbose=False,
            )
            records.extend(
                _run_strategy_internet(
                    model,
                    tokenizer,
                    device,
                    n_nuc_i,
                    bs,
                    seq_len,
                    source="nucleus_internet",
                    top_k=0,
                    top_p=float(nuc.get("top_p", 0.95)),
                    logits_processors=None,
                    temperature=float(nuc.get("temperature", 1.0)),
                    text_iter=text_iter,
                    prefix_tokens=prefix_tokens,
                    prefix_chars_max=prefix_chars_max,
                )
            )
            _carlini_log(
                f"[generate] nucleus_internet done: {len(records)} rows so far",
                verbose=False,
            )

    if not carlini_on:
        _carlini_log(
            "[generate] carlini_sampling disabled: skipped top_k / internet_prefix / "
            "temperature_decay / nucleus",
            verbose=False,
        )

    md = gcfg.get("memorization_detection") or {}
    if isinstance(md, dict) and md.get("enabled"):
        md_eff = dict(md)
        if not md_eff.get("num_samples_per_strategy"):
            md_eff["num_samples_per_strategy"] = n_per
        _carlini_log(
            "[generate] memorization_detection: generating extra samples (see YAML strategies) …",
            verbose=False,
        )
        _append_memorization_detection_records(
            records,
            md_cfg=md_eff,
            model=model,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
            model_bundle=model_bundle,
            seq_len=seq_len,
        )
        _carlini_log(
            f"[generate] memorization_detection done: {len(records)} rows total",
            verbose=False,
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
    _carlini_log(
        f"[generate] wrote {len(records)} lines → {out_path}",
        verbose=False,
    )
    return out_path
