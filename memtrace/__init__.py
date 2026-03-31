"""memTrace: membership inference from transformer hidden states and attention (Neural Breadcrumbs)."""

from .classifier import MemTraceClassifier, train_memtrace_classifier
from .features import extract_memtrace_features_from_tensors

# Hugging Face helper is optional (requires ``transformers``); use:
# ``from memtrace.huggingface import MemTraceHuggingFaceExtractor``

__all__ = [
    "extract_memtrace_features_from_tensors",
    "MemTraceClassifier",
    "train_memtrace_classifier",
]
