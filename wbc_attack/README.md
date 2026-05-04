# WBC Attack Implementation

This folder contains an implementation of the **Window-Based Comparison (WBC)** membership inference attack from:

- [arXiv:2601.02751](https://arxiv.org/abs/2601.02751)

## Intuition

WBC compares two causal language models on the **same** text: a **target** model \(M_T\) (e.g. a 7B checkpoint you suspect has memorized training data) and a **reference** model \(M_R\) (e.g. a smaller or less-trained model from the same family).

For each token, both models assign a **negative log-likelihood (NLL)** for the true next token given the prefix. On **training-like** text, the target often assigns **higher** probability (lower NLL) than the reference across **contiguous spans**, because the target has specialized to those patterns. WBC turns that idea into a **scalar score in \([0,1]\)** by sliding many window sizes along the token sequence and **voting** on whether the reference’s cumulative loss in each window exceeds the target’s.

## How it works (step by step)

### 1. Per-token losses (inputs to this package)

For a fixed text, tokenize once. For each valid next-token step you need two aligned sequences of length \(n\):

- **`target_losses[t]`** ≈ \(-\log p_{M_T}(x_{t+1} \mid x_{\le t})\)
- **`reference_losses[t]`** ≈ \(-\log p_{M_R}(x_{t+1} \mid x_{\le t})\)

In this repo, `mia_eval/losses.py` (`per_token_nll`) computes these from a single forward pass per model. **WBC itself does not run the models**; it consumes these two arrays (see `mia_eval/scoring_wbc.py` for the full path from text → score).

### 2. Sliding window vote (one window size `w`)

Consider all contiguous windows of **`w`** consecutive tokens. There are \(n - w + 1\) such windows (in the implementation’s indexing, `n` is the length of the loss arrays).

For each window, sum the target losses inside the window and the reference losses inside the window:

- \(S_T = \sum \text{target\_losses in window}\)
- \(S_R = \sum \text{reference\_losses in window}\)

**Vote = 1** (“member-like”) if **`S_R > S_T`**: the reference assigns **higher total NLL** (worse fit) to the same substring than the target. Equivalently, the target is **more confident** on that span than the reference.

**Vote = 0** otherwise.

The **window score** \(T_{\text{sign}}(w)\) is the **fraction** of windows that voted 1, in \([0,1]\).

This uses **running sums** so each window is updated in \(O(1)\) after the first, keeping the overall pass \(O(n)\) per window size.

### 3. Geometric ensemble over window sizes

Instead of a single `w`, WBC uses a **set of window sizes** (default a geometric schedule between `min_window` and `max_window` with `num_windows` values; the paper’s default `[2, 3, 4, 6, 9, 13, 18, 25, 32, 40]` is hard-coded as a special case when those three parameters match).

For each valid window size \(w\) (sizes larger than `n` are skipped), compute \(T_{\text{sign}}(w)\). The **final WBC score** is the **unweighted average** of these fractions across all valid sizes:

\[
S_{\text{WBC}} = \frac{1}{|\mathcal{W}|} \sum_{w \in \mathcal{W}} T_{\text{sign}}(w)
\]

So **higher** \(S_{\text{WBC}}\) means: across scales, **more** windows showed the reference struggling more than the target—**more member-like** under the paper’s construction.

### 4. Hyperparameters (`WBCConfig`)

| Field | Role |
|--------|------|
| `min_window`, `max_window`, `num_windows` | Define the geometric list of `w` (unless `window_sizes` is set explicitly). |
| `window_sizes` | Optional explicit list; if set, overrides the geometric generator. |

Optional ensemble fields used by `mia_eval` (`ensemble_variants`, `use_ensemble`, …) are merged in `mia_eval/scoring_wbc.py` when present in YAML.

## What is implemented

- Algorithm 1 (WBC attack) using per-token negative log-likelihood losses
- Sign-based sliding-window voting:
  - A window votes member if `sum(loss_ref_window) > sum(loss_target_window)`
- Geometric window ensemble with default schedule matching the paper:
  - `{2, 3, 4, 6, 9, 13, 18, 25, 32, 40}`
- Optional simple AUC evaluation helper for a set of samples

## Files

- `core.py`: main attack implementation
- `cli.py`: command-line utility for computing WBC from JSON loss arrays
- `__init__.py`: package exports

## Usage (Python)

```python
from wbc_attack import wbc_score_from_losses, WBCConfig

target_losses = [1.2, 0.9, 1.5, 0.7, 0.8, 1.1]
reference_losses = [1.4, 1.1, 1.7, 1.0, 1.2, 1.3]

score = wbc_score_from_losses(target_losses, reference_losses)
print(score)  # higher -> more likely member
```

## Usage (CLI)

Prepare two JSON files, each containing a list of per-token losses:

- `target.json`
- `reference.json`

Then run:

```bash
python -m wbc_attack.cli \
  --target-losses target.json \
  --reference-losses reference.json
```

Optional custom windows:

```bash
python -m wbc_attack.cli \
  --target-losses target.json \
  --reference-losses reference.json \
  --window-sizes 2 3 4 6 9 13 18 25 32 40
```

## Notes

- This implementation expects precomputed token-level losses.
- To exactly reproduce paper metrics, use the same target/reference models, datasets, and evaluation protocol as the paper.
