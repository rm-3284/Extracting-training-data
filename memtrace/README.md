# memTrace (Neural Breadcrumbs)

Implementation of **memTrace**, the membership inference framework from:

- Makhija et al., *Neural Breadcrumbs: Membership Inference Attacks on LLMs Through Hidden State and Attention Pattern Analysis* (EACL 2026). [ACL Anthology PDF](https://aclanthology.org/2026.eacl-long.262.pdf)

Official release: [amazonscience/NeuralBreadcrumbs-MIA-EACL-2026](https://github.com/amazonscience/NeuralBreadcrumbs-MIA-EACL-2026).

## Intuition

Standard **loss-only** MIAs (perplexity, zlib ratios, etc.) only see the model’s **scalar** prediction at each step. **memTrace** treats membership as a **white-box** problem: a single forward pass through the target LM exposes **internal activations**—hidden states and attention maps—that may carry statistical traces of whether the model “behaves like” it saw a string during training.

The method **vectorizes** many hand-designed statistics over layers, heads, and positions into one **fixed-length feature vector** per input. A **supervised classifier** (in the paper and often in this repo: `StandardScaler` + `RandomForest`) maps that vector to a membership probability or score. The features are meant to capture **confidence shape**, **attention structure**, **layer-to-layer drift**, **activation sparsity**, and **how representations evolve** along the prefix—signals that are hard to recover from logits alone.

## How it works (step by step)

### 1. Forward pass (white-box requirement)

Run the **target** causal LM with:

- `output_hidden_states=True` — tuple of hidden activations per layer (including embedding layer depending on implementation).
- `output_attentions=True` — per-layer, per-head attention weights for the prefilled sequence.

**Important:** many fast attention kernels **do not materialize** full attention weights. For faithful memTrace features, use an attention implementation that returns real weights (e.g. **eager** attention in Hugging Face), as done in `mia_eval` when scoring.

### 2. Feature extraction (paper §2; `memtrace/features.py`)

From the hidden-state tuple, attention tuple, optional `attention_mask`, and the **LM head** (to probe vocabulary-level behavior at intermediate layers), the code builds a **single feature vector**. Broad families include:

1. **Prediction confidence and entropy (via LM head on layer hidden states)**  
   Apply the shared output head to chosen layer representations; per-position statistics include entropy, top-1 probability, top1−top2 margin, aggregates (mean / std / min / max), and **stability** of confidence along the sequence.

2. **Attention structure**  
   Pool head outputs to a mean attention matrix; row entropies, concentration, sparsity, self- vs previous-token mass, **mean attention distance**, per-head entropy and focus.

3. **Layer transitions**  
   Euclidean “surprise” and cosine similarity between consecutive layer vectors at aligned positions; positions of min/max surprise.

4. **Activation geometry**  
   Sparsity, peak magnitude, softmax entropy over hidden dimensions, utilization, positive vs negative mass in hidden states.

5. **Context evolution**  
   Norm of change in a **cumulative mean** hidden representation as the prefix grows (how fast the model’s internal summary drifts).

6. **Token-position structure**  
   First–last cosine similarity per layer; local variation of final-layer confidence near start / middle / end of the sequence.

The exact list and ordering of scalar features is **architecture-dependent** (depth, width, head count). The vector length must match what the downstream classifier was trained on.

### 3. Classifier (paper §2; optional in this package)

Given a labeled dataset of `(feature_vector, member/non-member)`:

- The reference pipeline uses **standardization** + **Random Forest** (with hyperparameter search in the full paper: 5-fold CV + `RandomizedSearchCV`).
- In **`mia_eval`**, the same RF family is used in `run_pipeline`’s `evaluate` step (fit on train/val splits), and **proxy-trained** RFs for Qwen are produced by `mia_eval/train_memtrace_rf.py` then loaded at scoring time (`--memtrace-rf`).

Output is typically **`P(member | features)`** from the forest’s `predict_proba`.

### 4. Limitations

- **White-box** only: needs hidden states and (for full feature set) attentions.
- **Transfer**: a forest trained on one model’s feature dimension **does not** apply to another architecture without retraining.
- Numeric parity with the EACL benchmark tables requires matching models, tokenization, masks, and their full CV / search protocol.

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
