"""Load causal LMs / tokenizers for evaluation."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def load_causal_lm(
    model_name: str,
    tokenizer_name: Optional[str],
    device: torch.device,
    torch_dtype: Optional[torch.dtype] = None,
    *,
    attn_implementation: Optional[str] = None,
) -> Tuple[Any, Any]:
    tok_name = tokenizer_name or model_name
    tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except TypeError:
        kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    # Compatibility shim for some remote-code models (e.g., older OLMo classes)
    # that do not define this attribute expected by newer generation helpers.
    if not hasattr(model, "all_tied_weights_keys"):
        model.all_tied_weights_keys = []
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
