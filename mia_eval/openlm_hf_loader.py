"""
Register Hugging Face ``openlm`` config/model/tokenizer classes from the ``open_lm`` package.

DataComp / DCLM checkpoints (``model_type: openlm``) ship integration in ``mlfoundations/open_lm``:
importing ``open_lm.hf`` runs ``AutoConfig`` / ``AutoModelForCausalLM`` / ``AutoTokenizer`` registration.

Install (inference / HF registration only; avoids ``mosaicml`` / ``ray`` and broken PyYAML 5.x sdists)::

    pip install -U --prefer-binary -c mia_eval/constraints_openlm_inference.txt 'PyYAML>=6.0.1'
    pip install -U --prefer-binary -c mia_eval/constraints_openlm_inference.txt \\
        torch transformers accelerate sentencepiece huggingface_hub tiktoken safetensors numpy
    pip install -U --prefer-binary -c mia_eval/constraints_openlm_inference.txt --no-deps \\
        'git+https://github.com/mlfoundations/open_lm.git'
    pip install -U --prefer-binary -c mia_eval/constraints_openlm_inference.txt xformers
"""

from __future__ import annotations

import sys
import types


def _ensure_openlm_generation_flags() -> None:
    """OpenLM HF class predates newer ``generate`` cache flags; set explicitly on the class."""
    try:
        from transformers import PreTrainedModel
        from open_lm.hf.modeling_openlm import OpenLMForCausalLM
    except Exception:
        return
    default = getattr(PreTrainedModel, "_supports_default_dynamic_cache", True)
    OpenLMForCausalLM._supports_default_dynamic_cache = default


def _ensure_openlm_tie_weights_compat() -> None:
    """Patch OpenLM tie_weights signature for newer Transformers kwargs."""
    try:
        import inspect
        from open_lm.hf.modeling_openlm import OpenLMForCausalLM
    except Exception:
        return
    try:
        sig = inspect.signature(OpenLMForCausalLM.tie_weights)
    except (TypeError, ValueError):
        return
    if "missing_keys" in sig.parameters:
        return
    _orig = OpenLMForCausalLM.tie_weights

    def _tie_weights_compat(self, *args, **kwargs):
        return _orig(self)

    OpenLMForCausalLM.tie_weights = _tie_weights_compat


def _install_xformers_ops_fallback() -> None:
    """Provide a minimal ``xformers.ops`` shim when binary extensions are broken."""
    import torch
    import torch.nn.functional as F
    from torch import nn

    class _LowerTriangularMask:
        pass

    class _SwiGLU(nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, bias: bool = True):
            super().__init__()
            self.w12 = nn.Linear(in_dim, 2 * hidden_dim, bias=bias)
            self.w3 = nn.Linear(hidden_dim, out_dim, bias=bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            gate, val = self.w12(x).chunk(2, dim=-1)
            return self.w3(F.silu(gate) * val)

    def _memory_efficient_attention(
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_bias=None,
    ) -> torch.Tensor:
        q = queries.transpose(1, 2)
        k = keys.transpose(1, 2)
        v = values.transpose(1, 2)
        is_causal = isinstance(attn_bias, _LowerTriangularMask)
        mask = None if is_causal else attn_bias
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=is_causal)
        return out.transpose(1, 2).contiguous()

    xformers_mod = types.ModuleType("xformers")
    ops_mod = types.ModuleType("xformers.ops")
    ops_mod.LowerTriangularMask = _LowerTriangularMask
    ops_mod.SwiGLU = _SwiGLU
    ops_mod.memory_efficient_attention = _memory_efficient_attention
    xformers_mod.ops = ops_mod
    sys.modules["xformers"] = xformers_mod
    sys.modules["xformers.ops"] = ops_mod


def ensure_openlm_hf_registered() -> None:
    """Import ``open_lm.hf`` so OpenLM classes are registered with Transformers."""
    try:
        try:
            import xformers.ops  # noqa: F401
        except Exception:
            _install_xformers_ops_fallback()
        import open_lm.hf  # noqa: F401
        _ensure_openlm_generation_flags()
        _ensure_openlm_tie_weights_compat()
    except ImportError as e:
        raise ImportError(
            "OpenLM (DCLM) checkpoints need the ``open_lm`` package so "
            "`AutoModelForCausalLM` can resolve `model_type=openlm`. "
            "Use ``pip install --no-deps git+...open_lm...`` plus torch/transformers/xformers "
            "(see module docstring / ``jobscript/run_carlini_open_extract_array.slurm``). "
            "Original import error:\n"
            f"  {e}"
        ) from e


def is_openlm_load_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "openlm" in msg or (
        "does not recognize this architecture" in msg and "model type" in msg
    )
