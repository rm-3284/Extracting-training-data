"""Run Hugging Face causal LMs and extract memTrace feature vectors."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .features import extract_memtrace_features_from_tensors


def _resolve_lm_head(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "lm_head") and model.lm_head is not None:
        return model.lm_head
    out = getattr(model, "get_output_embeddings", None)
    if callable(out):
        head = model.get_output_embeddings()
        if head is not None:
            return head
    raise ValueError(
        "Could not resolve LM head; model must expose lm_head or get_output_embeddings()"
    )


class MemTraceHuggingFaceExtractor:
    """
    Forward-pass wrapper: collects hidden states and attentions, then memTrace features.

    Matches the paper's setting (white-box): full access to internal representations.
    """

    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        torch_dtype: torch.dtype | None = None,
        max_length: int = 512,
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        kwargs = {}
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.model.to(self.device)
        self.model.eval()
        self.lm_head = _resolve_lm_head(self.model)
        cfg_max = getattr(self.model.config, "max_position_embeddings", None)
        self.max_length = min(max_length, cfg_max) if cfg_max else max_length

    @torch.inference_mode()
    def features_for_text(self, text: str) -> Tuple[np.ndarray, List[str]]:
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model(
            **enc,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )
        hs = out.hidden_states
        att = out.attentions
        if hs is None:
            raise RuntimeError("Model did not return hidden_states")
        vec, names = extract_memtrace_features_from_tensors(
            hs,
            att,
            self.lm_head,
            attention_mask=enc.get("attention_mask"),
            device=self.device,
        )
        return vec, names

    @torch.inference_mode()
    def features_for_texts(self, texts: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Stack feature vectors for multiple strings (variable length → pad to batch of 1 each)."""
        rows = []
        names: List[str] | None = None
        for t in texts:
            v, n = self.features_for_text(t)
            if names is None:
                names = n
            elif names != n:
                raise ValueError(
                    "Feature name mismatch across texts; use the same model and truncation."
                )
            rows.append(v)
        return np.stack(rows, axis=0), names if names is not None else []
