# WBC Attack Implementation

This folder contains an implementation of the **Window-Based Comparison (WBC)** membership inference attack from:

- [arXiv:2601.02751](https://arxiv.org/abs/2601.02751)

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
