"""Core implementation of the WBC attack from arXiv:2601.02751."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class WBCConfig:
    """Configuration for the WBC attack."""

    window_sizes: Sequence[int] | None = None
    min_window: int = 2
    max_window: int = 40
    num_windows: int = 10


def geometric_window_sizes(
    min_window: int = 2,
    max_window: int = 40,
    num_windows: int = 10,
) -> List[int]:
    """
    Generate geometrically-spaced window sizes.

    Matches the paper's default progression:
    w_k = round(2 * 20^((k-1)/9)), k in {1..10}
    -> [2, 3, 4, 6, 9, 13, 18, 25, 32, 40]
    """
    if min_window < 1:
        raise ValueError("min_window must be >= 1")
    if max_window < min_window:
        raise ValueError("max_window must be >= min_window")
    if num_windows < 1:
        raise ValueError("num_windows must be >= 1")

    if num_windows == 1:
        return [int(min_window)]

    # The paper reports this exact default schedule in the appendix.
    if min_window == 2 and max_window == 40 and num_windows == 10:
        return [2, 3, 4, 6, 9, 13, 18, 25, 32, 40]

    ratio = max_window / float(min_window)
    windows: List[int] = []
    for k in range(num_windows):
        exponent = k / float(num_windows - 1)
        w = int(round(min_window * (ratio**exponent)))
        windows.append(w)

    deduped = sorted(set(windows))
    return deduped


def _window_vote_fraction(
    target_losses: Sequence[float],
    reference_losses: Sequence[float],
    window_size: int,
) -> float:
    """
    T_sign(w) from Algorithm 1.

    A window votes "member" when sumR > sumT for that window.
    """
    n = len(target_losses)
    if window_size > n:
        raise ValueError(
            f"window_size={window_size} exceeds sequence length n={n}"
        )

    # Algorithm 1 style running sums over sliding windows.
    sum_t = sum(target_losses[:window_size])
    sum_r = sum(reference_losses[:window_size])
    votes = 1 if sum_r > sum_t else 0

    total_windows = n - window_size + 1
    for i in range(1, total_windows):
        sum_t = sum_t - target_losses[i - 1] + target_losses[i + window_size - 1]
        sum_r = sum_r - reference_losses[i - 1] + reference_losses[i + window_size - 1]
        if sum_r > sum_t:
            votes += 1

    return votes / float(total_windows)


def wbc_score_from_losses(
    target_losses: Sequence[float],
    reference_losses: Sequence[float],
    config: WBCConfig | None = None,
) -> float:
    """
    Compute S_WBC for a single sample from per-token losses.

    Inputs:
      - target_losses: losses from target model M_T
      - reference_losses: losses from reference model M_R
    Returns:
      - final membership score in [0, 1] (higher => more likely member)
    """
    cfg = config or WBCConfig()

    t = [float(x) for x in target_losses]
    r = [float(x) for x in reference_losses]

    if len(t) != len(r):
        raise ValueError("target_losses and reference_losses must have equal length")
    if len(t) < 1:
        raise ValueError("loss arrays must be non-empty")

    window_sizes = (
        list(cfg.window_sizes)
        if cfg.window_sizes is not None
        else geometric_window_sizes(
            min_window=cfg.min_window,
            max_window=cfg.max_window,
            num_windows=cfg.num_windows,
        )
    )

    valid_window_sizes = [w for w in window_sizes if 1 <= w <= len(t)]
    if not valid_window_sizes:
        raise ValueError(
            "No valid window sizes for this sequence length. "
            f"Sequence length={len(t)}, candidate windows={window_sizes}"
        )

    score_sum = 0.0
    for w in valid_window_sizes:
        score_sum += _window_vote_fraction(t, r, w)
    return score_sum / float(len(valid_window_sizes))


def _rankdata_average(values: Sequence[float]) -> List[float]:
    """Average-rank ties; used for AUC without sklearn."""
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    sorted_vals = [v for _, v in indexed]
    ranks = [0.0] * len(values)

    i = 0
    n = len(values)
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0  # 1-indexed ranks
        for idx, _ in indexed[i:j]:
            ranks[idx] = avg_rank
        i = j
    return ranks


def auc_from_scores(labels: Sequence[int], scores: Sequence[float]) -> float:
    """
    Compute ROC-AUC from binary labels and continuous scores.

    labels: 1 = member, 0 = non-member
    """
    y = [int(v) for v in labels]
    s = [float(v) for v in scores]
    if len(y) != len(s):
        raise ValueError("labels and scores length mismatch")

    n_pos = sum(1 for v in y if v == 1)
    n_neg = sum(1 for v in y if v == 0)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC requires both positive and negative examples")

    ranks = _rankdata_average(s)
    sum_ranks_pos = sum(rank for rank, label in zip(ranks, y) if label == 1)
    auc = (sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return auc


def evaluate_wbc_scores(
    target_loss_sequences: Iterable[Sequence[float]],
    reference_loss_sequences: Iterable[Sequence[float]],
    labels: Sequence[int],
    config: WBCConfig | None = None,
) -> dict:
    """Compute per-sample WBC scores and aggregate AUC."""
    target_list = list(target_loss_sequences)
    reference_list = list(reference_loss_sequences)
    if len(target_list) != len(reference_list):
        raise ValueError("target/reference sequence count mismatch")
    if len(target_list) != len(labels):
        raise ValueError("labels length mismatch")

    scores = [
        wbc_score_from_losses(t, r, config=config)
        for t, r in zip(target_list, reference_list)
    ]
    return {
        "scores": scores,
        "auc": auc_from_scores(labels, scores),
    }
