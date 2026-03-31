# memTrace (Neural Breadcrumbs)

Implementation of **memTrace**, the membership inference framework from:

- Makhija et al., *Neural Breadcrumbs: Membership Inference Attacks on LLMs Through Hidden State and Attention Pattern Analysis* (EACL 2026). [ACL Anthology PDF](https://aclanthology.org/2026.eacl-long.262.pdf)

Official release: [amazonscience/NeuralBreadcrumbs-MIA-EACL-2026](https://github.com/amazonscience/NeuralBreadcrumbs-MIA-EACL-2026).

## What this code does

1. **Feature extraction** (paper §2): From one forward pass through a causal LM, builds a fixed-length vector from:
   - **Prediction confidence & entropy**: LM head applied to each layer’s hidden state; per-position entropy, top-1 probability, top-1 minus top-2 gap, aggregates, and confidence stability.
   - **Attention**: Head-mean attention matrix; row entropy, concentration, adaptive sparsity, self- and previous-token bias, mean attention distance, per-head entropy/focus.
   - **Layer transitions**: Euclidean “surprise” and cosine stability between consecutive layer representations; position of min/max surprise.
   - **Activation patterns**: Sparsity, peak magnitude, softmax entropy over hidden dimensions, utilization, positive/negative mass.
   - **Context evolution**: Norm of change in cumulative mean hidden state as the prefix grows.
   - **Token-position**: First–last cosine similarity per layer; local std of final-layer confidence near first/middle/last tokens.

2. **Classifier** (paper §2): Optional `StandardScaler` + `RandomForestClassifier` training helper (simplified single holdout split; the paper uses 5-fold CV + `RandomizedSearchCV`).

Feature dimensionality is **architecture-dependent** (the paper reports ~2,085 features for Pythia-1B with 16 layers).

## Dependencies

- `torch`, `numpy`
- `transformers` (for `MemTraceHuggingFaceExtractor`)
- `scikit-learn` (for `train_memtrace_classifier`)

## Usage

### Hugging Face causal LM

```python
from memtrace.huggingface import MemTraceHuggingFaceExtractor

ext = MemTraceHuggingFaceExtractor("gpt2", max_length=128)
vec, names = ext.features_for_text("Example text for auditing.")
```

### Tensors only (custom models)

```python
from memtrace import extract_memtrace_features_from_tensors

# hidden_states: tuple from output_hidden_states=True (batch size 1)
# attentions: tuple from output_attentions=True, or None
# lm_head: nn.Module mapping (..., d) -> (..., vocab)
vec, names = extract_memtrace_features_from_tensors(
    hidden_states, attentions, lm_head, attention_mask
)
```

### Train RF on labeled features

```python
import numpy as np
from memtrace import train_memtrace_classifier

# X: (n_samples, n_features), y: 0/1 non-member/member
out = train_memtrace_classifier(X, y)
print("Holdout AUC:", out["auc_holdout"])
clf = out["classifier"]
proba = clf.predict_proba_member(X_new)
```

## Limitations

- **White-box**: Requires hidden states and (for full features) attention weights.
- This repository implements the **methodology from the paper**; exact numeric match to their benchmark tables would require matching models, tokenization, masks, and their full CV / search protocol.
- Some models omit attention in certain configs; pass `attentions=None` only if you intentionally skip attention features (vector length and semantics change).
