"""Window-Based Comparison (WBC) membership inference attack."""

from .core import (
    WBCConfig,
    evaluate_wbc_scores,
    geometric_window_sizes,
    wbc_score_from_losses,
)

__all__ = [
    "WBCConfig",
    "geometric_window_sizes",
    "wbc_score_from_losses",
    "evaluate_wbc_scores",
]
