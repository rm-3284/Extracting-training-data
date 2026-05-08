"""Load causal LMs / tokenizers for evaluation."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.generation.utils import GenerationMixin

from mia_eval.openlm_hf_loader import ensure_openlm_hf_registered, is_openlm_load_error


def pick_device(cfg_device: Optional[str]) -> torch.device:
    if cfg_device and cfg_device != "auto":
        return torch.device(cfg_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def torch_dtype_from_str(s: Optional[str]) -> Optional[torch.dtype]:
    if not s:
        return None
    m = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return m.get(str(s).lower())


def _from_pretrained_causal_lm(model_name: str, kwargs: Dict[str, Any]) -> Any:
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except ValueError as e:
        # Some Dolly mirrors omit ``model_type`` in config.json, but weights are GPT-NeoX-compatible.
        # Fallback to a known GPT-NeoX config so model class resolution succeeds.
        msg = str(e).lower()
        if "unrecognized model" in msg and "model_type" in msg and "dolly" in model_name.lower():
            cfg_kwargs: Dict[str, Any] = {}
            if "token" in kwargs:
                cfg_kwargs["token"] = kwargs["token"]
            cfg = AutoConfig.from_pretrained("EleutherAI/pythia-12b", trust_remote_code=True, **cfg_kwargs)
            kw3 = dict(kwargs)
            kw3["config"] = cfg
            return AutoModelForCausalLM.from_pretrained(model_name, **kw3)
        raise
    except TypeError:
        kw2 = dict(kwargs)
        kw2.pop("attn_implementation", None)
        return AutoModelForCausalLM.from_pretrained(model_name, **kw2)


def _ensure_tied_weights_attr_for_compat(model_name: str) -> None:
    """Backfill `all_tied_weights_keys` for older model classes before load.

    Newer Transformers treats this as a mapping and calls ``.keys()``; some remote
    checkpoints expose a list (or nothing), which raises AttributeError.
    """
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}
    if "olmo" in model_name.lower():
        try:
            import hf_olmo  # noqa: F401
            from hf_olmo import OLMoForCausalLM

            if not hasattr(OLMoForCausalLM, "all_tied_weights_keys"):
                OLMoForCausalLM.all_tied_weights_keys = {}
            elif isinstance(OLMoForCausalLM.all_tied_weights_keys, list):
                OLMoForCausalLM.all_tied_weights_keys = {}
            # Newer Transformers may call tie_weights(missing_keys=...).
            # Older hf_olmo implementations expose tie_weights(self) only.
            try:
                import inspect

                sig = inspect.signature(OLMoForCausalLM.tie_weights)
                if "missing_keys" not in sig.parameters:
                    _orig_tie_weights = OLMoForCausalLM.tie_weights

                    def _tie_weights_compat(self, *args, **kwargs):
                        return _orig_tie_weights(self)

                    OLMoForCausalLM.tie_weights = _tie_weights_compat
            except Exception:
                pass
        except Exception:
            # If hf_olmo is unavailable or API differs, keep generic base-class shim.
            pass


def _normalize_all_tied_weights_keys(model: Any) -> None:
    """Ensure instance and class use a dict (Transformers >= ~4.48 expects mapping semantics)."""
    for target in (model, model.__class__):
        if not hasattr(target, "all_tied_weights_keys"):
            setattr(target, "all_tied_weights_keys", {})
        else:
            v = getattr(target, "all_tied_weights_keys")
            if isinstance(v, list):
                setattr(target, "all_tied_weights_keys", {})


def _ensure_dynamic_cache_flag(model: Any) -> None:
    """Remote-code models may omit flags newer ``generate`` reads."""
    cls = model.__class__
    default = getattr(PreTrainedModel, "_supports_default_dynamic_cache", True)
    # Always refresh OpenLM: some stacks report the flag via MRO but instance access still fails.
    if cls.__name__ == "OpenLMForCausalLM" or not hasattr(model, "_supports_default_dynamic_cache"):
        setattr(cls, "_supports_default_dynamic_cache", default)


def _ensure_generation_methods(model: Any) -> None:
    """Backfill missing GenerationMixin methods for older remote-code classes."""
    cls = model.__class__
    for name, attr in GenerationMixin.__dict__.items():
        if name.startswith("__"):
            continue
        if not callable(attr):
            continue
        if not hasattr(cls, name):
            # Assign unbound function to class; Python binds it on instance access.
            setattr(cls, name, attr)


def _ensure_tie_weights_signature_compat(model: Any) -> None:
    """Allow older remote-code tie_weights(self) under newer Transformers calls."""
    import inspect

    cls = model.__class__
    tie = getattr(cls, "tie_weights", None)
    if tie is None:
        return
    try:
        sig = inspect.signature(tie)
    except (TypeError, ValueError):
        return
    if "missing_keys" in sig.parameters:
        return
    _orig_tie = tie

    def _tie_weights_compat(self, *args, **kwargs):
        return _orig_tie(self)

    cls.tie_weights = _tie_weights_compat


def load_causal_lm(
    model_name: str,
    tokenizer_name: Optional[str],
    device: torch.device,
    torch_dtype: Optional[torch.dtype] = None,
    *,
    attn_implementation: Optional[str] = None,
) -> Tuple[Any, Any]:
    tok_name = tokenizer_name or model_name
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    tok_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if hf_token:
        tok_kwargs["token"] = hf_token

    def _is_tokenizer_backend_fail(exc: BaseException) -> bool:
        msg = str(exc).lower()
        return any(
            needle in msg
            for needle in (
                "backend tokenizer",
                "instantiate the backend",
                "couldn't instantiate",
                "could not instantiate",
                "sentencepiece",
                "tiktoken",
                "convert a slow tokenizer",
            )
        )

    def _build_tokenizer() -> Any:
        # GPT-NeoX–style checkpoints on HF often lack fast-tokenizer assets; default fast load then
        # tries to convert the slow tokenizer and requires sentencepiece/tiktoken even when slow
        # would work. Dolly mirrors are a common case.
        kw = dict(tok_kwargs)
        if "dolly" in tok_name.lower():
            kw["use_fast"] = False
        try:
            tok = AutoTokenizer.from_pretrained(tok_name, **kw)
        except Exception as e:
            if kw.get("use_fast") is not False and _is_tokenizer_backend_fail(e):
                tok = AutoTokenizer.from_pretrained(
                    tok_name, use_fast=False, **tok_kwargs
                )
            else:
                raise
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok

    tokenizer = _build_tokenizer()

    kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if hf_token:
        kwargs["token"] = hf_token
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    # Ensure OpenLM classes are registered and patched before from_pretrained for DCLM runs.
    if any(k in model_name.lower() for k in ("dclm", "openlm")):
        ensure_openlm_hf_registered()

    _ensure_tied_weights_attr_for_compat(model_name)

    try:
        model = _from_pretrained_causal_lm(model_name, kwargs)
    except (ValueError, OSError, TypeError) as e:
        if is_openlm_load_error(e):
            ensure_openlm_hf_registered()
            tokenizer = _build_tokenizer()
            model = _from_pretrained_causal_lm(model_name, kwargs)
        else:
            raise
    _normalize_all_tied_weights_keys(model)
    _ensure_dynamic_cache_flag(model)
    _ensure_tie_weights_signature_compat(model)
    _ensure_generation_methods(model)
    model.to(device)
    model.eval()
    if hasattr(model.config, "pad_token_id") and model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def resolve_lm_head(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "lm_head") and model.lm_head is not None:
        return model.lm_head
    if callable(getattr(model, "get_output_embeddings", None)):
        head = model.get_output_embeddings()
        if head is not None:
            return head
    raise ValueError(
        "Could not resolve LM head (lm_head or get_output_embeddings)"
    )
