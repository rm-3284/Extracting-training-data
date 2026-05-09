"""Load causal LMs / tokenizers for evaluation."""

from __future__ import annotations

import inspect
import os

from typing import Any, Dict, List, Optional, Set, Tuple

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


def _dtype_kwargs_for_from_pretrained(torch_dtype: torch.dtype) -> Dict[str, Any]:
    """Prefer ``dtype`` (Transformers 5+) over deprecated ``torch_dtype``.

    TF 5 ``from_pretrained`` often hides ``dtype`` behind ``**kwargs``, so signature
    inspection alone misses it and would keep emitting deprecation warnings.
    """
    try:
        import transformers

        ver = transformers.__version__.split("+")[0].split("-")[0]
        major_s = "".join(c for c in ver.split(".")[0] if c.isdigit())
        major = int(major_s or "0")
        if major >= 5:
            return {"dtype": torch_dtype}
    except Exception:
        pass
    try:
        sig = inspect.signature(AutoModelForCausalLM.from_pretrained)
        if "dtype" in sig.parameters:
            return {"dtype": torch_dtype}
    except (TypeError, ValueError):
        pass
    return {"torch_dtype": torch_dtype}


def _ensure_transformers_head_pruning_compat() -> None:
    """Backfill attention-head pruning helpers removed in Transformers 5.

    Hub checkpoints such as ``LLM360/Crystal`` ship ``modeling_*.py`` that still import
    ``find_pruneable_heads_and_indices`` / ``prune_conv1d_layer`` from
    ``transformers.pytorch_utils`` (removed in `PR #41417`_). Inference does not use these;
    they only need to exist for module import.

    .. _PR #41417: https://github.com/huggingface/transformers/pull/41417
    """
    try:
        import transformers.pytorch_utils as pu
    except ImportError:
        return
    if hasattr(pu, "find_pruneable_heads_and_indices") and hasattr(pu, "prune_conv1d_layer"):
        return

    Conv1D = getattr(pu, "Conv1D", None)
    if Conv1D is None:

        class Conv1D(torch.nn.Module):
            """GPT-style linear-as-conv (matches pre-v5 ``pytorch_utils.Conv1D``)."""

            def __init__(self, nf: int, nx: int):
                super().__init__()
                self.nf = nf
                self.nx = nx
                self.weight = torch.nn.Parameter(torch.empty(nx, nf))
                self.bias = torch.nn.Parameter(torch.zeros(nf))
                torch.nn.init.normal_(self.weight, std=0.02)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                size_out = x.size()[:-1] + (self.nf,)
                x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
                return x.view(size_out)

        pu.Conv1D = Conv1D
    else:
        Conv1D = pu.Conv1D

    def prune_conv1d_layer(layer: Any, index: torch.LongTensor, dim: int = 1) -> Any:
        index = index.to(layer.weight.device)
        W = layer.weight.index_select(dim, index).clone().detach()
        if dim == 0:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
        new_size = list(layer.weight.size())
        new_size[dim] = len(index)
        new_layer = Conv1D(new_size[1], new_size[0]).to(layer.weight.device)
        new_layer.weight.requires_grad = False
        new_layer.weight.copy_(W.contiguous())
        new_layer.weight.requires_grad = True
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
        return new_layer

    def find_pruneable_heads_and_indices(
        heads: List[int],
        n_heads: int,
        head_size: int,
        already_pruned_heads: Set[int],
    ) -> Tuple[Set[int], torch.LongTensor]:
        mask = torch.ones(n_heads, head_size)
        heads_set = set(heads) - already_pruned_heads
        for head in heads_set:
            head_adj = head - sum(1 if h < head else 0 for h in already_pruned_heads)
            mask[head_adj] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask), device=mask.device)[mask].long()
        return heads_set, index

    if not hasattr(pu, "prune_conv1d_layer"):
        pu.prune_conv1d_layer = prune_conv1d_layer
    if not hasattr(pu, "find_pruneable_heads_and_indices"):
        pu.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices


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
            # hf_olmo internals expect list/tuple KV cache per layer, not ``DynamicCache``.
            @classmethod
            def _olmo_supports_default_dynamic_cache(_cls) -> bool:
                return False

            OLMoForCausalLM._supports_default_dynamic_cache = _olmo_supports_default_dynamic_cache
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
    """Ensure ``_supports_default_dynamic_cache`` is GenerationMixin's classmethod.

    Transformers 5+ calls ``self._supports_default_dynamic_cache()`` inside ``generate``.
    A plain ``bool`` (from an older mistaken patch) raises ``TypeError: 'bool' object is not callable``.
    """
    cls = model.__class__
    cm = GenerationMixin.__dict__.get("_supports_default_dynamic_cache")
    if not isinstance(cm, classmethod):
        return
    cur = getattr(cls, "_supports_default_dynamic_cache", None)
    if cur is None or isinstance(cur, bool):
        setattr(cls, "_supports_default_dynamic_cache", cm)


def _ensure_generation_methods(model: Any) -> None:
    """Backfill missing GenerationMixin methods for older remote-code classes."""
    cls = model.__class__
    for name, attr in GenerationMixin.__dict__.items():
        if name.startswith("__"):
            continue
        if not callable(attr):
            continue
        # Some remote checkpoints expose names like ``generate`` as non-callable flags.
        # Only skip when an existing *callable* implementation is present.
        existing = getattr(cls, name, None)
        if callable(existing):
            continue
        setattr(cls, name, attr)


def _ensure_config_generation_attrs(model: Any) -> None:
    """Backfill config fields Transformers ``generate`` / cache code expects."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        return
    if not hasattr(cfg, "use_cache"):
        try:
            cfg.use_cache = True
        except (AttributeError, TypeError):
            pass
    # OpenLM/DCLM configs use open_lm names (``n_layers``, ``dim``) on ``OpenLMConfig``.
    mt = getattr(cfg, "model_type", None)
    if mt == "openlm" or type(cfg).__name__ == "OpenLMConfig":
        try:
            if not hasattr(cfg, "num_hidden_layers"):
                nl = getattr(cfg, "n_layers", None)
                if nl is not None:
                    cfg.num_hidden_layers = int(nl)
            if not hasattr(cfg, "hidden_size"):
                d = getattr(cfg, "dim", None)
                if d is not None:
                    cfg.hidden_size = int(d)
            if not hasattr(cfg, "num_attention_heads"):
                nh = getattr(cfg, "n_heads", None)
                if nh is not None:
                    cfg.num_attention_heads = int(nh)
        except (AttributeError, TypeError, ValueError):
            pass


def _ensure_olmo_generate_signature_compat(model: Any) -> None:
    """hf_olmo + newer ``generate``: forward must *name* args ``_validate_model_kwargs`` allows (e.g. ``attention_mask``)."""
    cls = model.__class__
    if cls.__name__ != "OLMoForCausalLM":
        return
    if getattr(cls, "_mia_eval_olmo_generate_sig_patched", False):
        return
    import inspect

    def _call_filtered(fn: Any, self: Any, /, *args: Any, **kwargs: Any) -> Any:
        sig = inspect.signature(fn)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return fn(self, *args, **kwargs)
        allowed = {n for n in sig.parameters if n != "self"}
        return fn(self, *args, **{k: v for k, v in kwargs.items() if k in allowed})

    _orig_forward = cls.forward

    def _forward(
        self,
        *args: Any,
        attention_mask: Any = None,
        cache_position: Any = None,
        position_ids: Any = None,
        past_key_values: Any = None,
        **kwargs: Any,
    ) -> Any:
        merged = dict(kwargs)
        for name, val in (
            ("attention_mask", attention_mask),
            ("cache_position", cache_position),
            ("position_ids", position_ids),
            ("past_key_values", past_key_values),
        ):
            if val is not None:
                merged[name] = val
        return _call_filtered(_orig_forward, self, *args, **merged)

    cls.forward = _forward

    _orig_prep = getattr(cls, "prepare_inputs_for_generation", None)
    if _orig_prep is not None:
        try:
            prep_names = set(inspect.signature(_orig_prep).parameters)
        except (TypeError, ValueError):
            prep_names = set()
        if (
            "attention_mask" not in prep_names
            and "kwargs" not in prep_names
            and "model_kwargs" not in prep_names
        ):

            def _prep(
                self,
                *args: Any,
                attention_mask: Any = None,
                cache_position: Any = None,
                position_ids: Any = None,
                **kwargs: Any,
            ) -> Any:
                merged = dict(kwargs)
                for name, val in (
                    ("attention_mask", attention_mask),
                    ("cache_position", cache_position),
                    ("position_ids", position_ids),
                ):
                    if val is not None:
                        merged[name] = val
                return _call_filtered(_orig_prep, self, *args, **merged)

            cls.prepare_inputs_for_generation = _prep

    cls._mia_eval_olmo_generate_sig_patched = True


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
    tok_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        # Avoid Transformers warning for BPE (e.g. OLMo fast tokenizer).
        "clean_up_tokenization_spaces": False,
    }
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

    _olmo_dbg = "olmo" in model_name.lower()
    if _olmo_dbg:
        print(
            f"[load_causal_lm] OLMo: tokenizer ok ({tok_name}); loading weights from {model_name!r} …",
            flush=True,
        )

    kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if hf_token:
        kwargs["token"] = hf_token
    if torch_dtype is not None:
        kwargs.update(_dtype_kwargs_for_from_pretrained(torch_dtype))
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    # Ensure OpenLM classes are registered and patched before from_pretrained for DCLM runs.
    if any(k in model_name.lower() for k in ("dclm", "openlm")):
        ensure_openlm_hf_registered()

    _ensure_tied_weights_attr_for_compat(model_name)
    _ensure_transformers_head_pruning_compat()

    if _olmo_dbg:
        print(
            "[load_causal_lm] OLMo: from_pretrained starting (remote code; first forward compat after load) …",
            flush=True,
        )

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
    _ensure_olmo_generate_signature_compat(model)
    _ensure_config_generation_attrs(model)
    model.to(device)
    model.eval()
    if hasattr(model.config, "pad_token_id") and model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if _olmo_dbg:
        try:
            pdev = next(model.parameters()).device
        except StopIteration:
            pdev = device
        print(
            f"[load_causal_lm] OLMo: ready {type(model).__name__} on {pdev}; "
            "compat hooks applied (KV cache / generate / forward).",
            flush=True,
        )

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
