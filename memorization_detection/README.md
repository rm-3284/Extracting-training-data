## Prefix-Aware MI-Guided Decoding

### Overview

This module explores an inference-time intervention for reducing extractive or memorization-prone continuations in language models.

Rather than modifying model weights or requiring access to the original training corpus, we use a **membership-inference-derived risk signal** during decoding. The core idea is to detect when the current prefix appears unusually training-like, then selectively downweight suspicious next-token continuations.

This work is intended as an extension of the broader project on membership inference in LLMs.

---

### Main Idea

Most membership inference methods are used only for **post hoc detection**: given a completed sequence, estimate whether it is likely to have appeared in training.

Here, we repurpose that signal for **online control**.

At decoding step \(t\), given prefix \(x_{<t}\):

1. compute a prefix-level membership/risk score using our MI method,
2. convert that score into a bounded risk gate,
3. use that gate to penalize suspicious candidate next tokens,
4. decode from the modified distribution.

This yields a decoding procedure that is:

- **prefix-conditioned**
- **model-agnostic**
- **training-data-free at inference**
- **non-invasive** (no finetuning, no weight editing)

---

### Motivation

Existing inference-time mitigation methods each have limitations:

- **n-gram / bloom-filter approaches** require access to training data and cannot catch near-matches,
- **activation steering** requires careful layer selection and hidden-state manipulation,
- **mechanistic localization** is informative but not directly deployable as a lightweight wrapper.

Our approach instead uses the model’s own behavior, through a membership-inference signal, to estimate when generation is entering a high-risk region.

---

### Method

Let the base language model define

\[
p_\theta(v \mid x_{<t}) = \mathrm{softmax}(z_t)_v
\]

where \(z_{t,v}\) is the logit for candidate token \(v\).

We compute a prefix-level membership score

\[
s_{\mathrm{MI}}(x_{<t})
\]

using our infilling-based membership inference method (or a surrogate trained to approximate it).

We convert this to a bounded risk gate:

\[
r_t = \sigma\!\big(\alpha(s_{\mathrm{MI}}(x_{<t}) - \tau)\big)
\]

where:
- \(\sigma\) is the sigmoid function,
- \(\tau\) is a threshold,
- \(\alpha\) controls how sharply risk activates.

We then define a token-level suspiciousness term \(g_t(v)\), and modify the distribution as:

\[
\tilde z_{t,v} = z_{t,v} - \lambda r_t g_t(v)
\]

\[
\tilde p(v \mid x_{<t}) = \mathrm{softmax}(\tilde z_t)_v
\]

Equivalently,

\[
\tilde p(v \mid x_{<t})
\propto
p_\theta(v \mid x_{<t})
\exp(-\lambda r_t g_t(v))
\]

where \(\lambda\) controls intervention strength.

---

### Interpreting the Components

- \(s_{\mathrm{MI}}(x_{<t})\): how membership-like or extraction-prone the current prefix appears
- \(r_t\): a bounded risk gate derived from the MI score
- \(g_t(v)\): how suspicious candidate token \(v\) is under the current prefix
- \(\lambda\): how strongly risky continuations are penalized

When \(r_t\) is low, decoding behaves almost like the original model.

When \(r_t\) is high, suspicious continuations are downweighted more aggressively.

---

### Using the Infilling Score

Our current MI signal is based on an **infilling score**, which measures how sensitive a sequence is to local perturbations. Intuitively, memorization-prone sequences may exhibit brittle, unusually peaked continuation behavior: replacing a token can cause a sharp drop in local continuation consistency.

In the current prototype, the infilling score is computed offline on text examples. This score can be used in two ways:

1. **Directly**, as a prefix-level risk signal during decoding, or
2. **Indirectly**, by training a small surrogate model to predict the infilling score from the LM hidden state for faster online use.

The second option is likely more practical for real-time decoding.

---

### Implementation Plan

#### Phase 1: Data and scoring
- Load MIMIR member/nonmember examples
- Split each text into prefix/suffix
- Compute suffix NLL and/or infilling-based MI scores
- Verify that member-like continuations are easier or more brittle than nonmember ones

#### Phase 2: Prefix risk modeling
- Compute a local prefix-level MI score
- Optionally train a lightweight surrogate risk model on LM hidden states

#### Phase 3: Decoding intervention
- At each generation step:
  - score the current prefix,
  - compute the risk gate \(r_t\),
  - reweight candidate next-token logits,
  - sample from the modified distribution

#### Phase 4: Evaluation
Compare:
- standard decoding
- global temperature / penalty baselines
- MI-guided prefix-aware decoding

---

### Evaluation Goals

We want to test whether MI-guided decoding can:

- reduce exact or near-exact continuation of member-like sequences,
- selectively activate on risky prefixes rather than everywhere,
- preserve utility better than blunt global penalties.

Potential metrics:
- suffix negative log-likelihood,
- exact continuation length,
- overlap with target suffix,
- change in generation quality / fluency.

---

### Claims and Scope

This module does **not** claim to perfectly detect memorization or fully solve extraction.

Instead, it tests the following hypothesis:

> Membership-inference signals can be repurposed as prefix-level extraction-risk proxies, enabling lightweight and selective decoding-time mitigation.

---

### Current Status

- [x] Load MIMIR data
- [x] Compute member vs nonmember suffix NLL on examples
- [x] Inspect infilling-based MI implementation
- [ ] Define prefix-local MI risk score
- [ ] Implement logit reweighting
- [ ] Compare against baseline decoding

---

### Practical Notes

This approach differs from prior work in three ways:

1. it does **not** require access to the training corpus at inference time,
2. it does **not** modify weights or hidden states directly,
3. it uses **membership inference as a control signal**, rather than only as an offline audit tool.