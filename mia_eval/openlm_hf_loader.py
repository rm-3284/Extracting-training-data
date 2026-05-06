"""
Register Hugging Face ``openlm`` config/model/tokenizer classes from the ``open_lm`` package.

DataComp / DCLM checkpoints (``model_type: openlm``) ship integration in ``mlfoundations/open_lm``:
importing ``open_lm.hf`` runs ``AutoConfig`` / ``AutoModelForCausalLM`` / ``AutoTokenizer`` registration.

Install::

    pip install \"git+https://github.com/mlfoundations/open_lm.git\"
"""

from __future__ import annotations


def ensure_openlm_hf_registered() -> None:
    """Import ``open_lm.hf`` so OpenLM classes are registered with Transformers."""
    try:
        import open_lm.hf  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "OpenLM (DCLM) checkpoints need the ``open_lm`` package so "
            "`AutoModelForCausalLM` can resolve `model_type=openlm`. Install:\n"
            '  pip install "git+https://github.com/mlfoundations/open_lm.git"\n'
            "Then retry. Original import error:\n"
            f"  {e}"
        ) from e


def is_openlm_load_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "openlm" in msg or (
        "does not recognize this architecture" in msg and "model type" in msg
    )
