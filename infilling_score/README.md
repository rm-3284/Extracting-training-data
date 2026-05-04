# Infilling Scores

This implementation of **infilling-based membership inference** follows the construction in the OpenReview paper linked below. The same scoring logic is also used from `mia_eval/scoring_infilling.py` in the evaluation pipeline.

- Paper / write-up: [OpenReview `9QPH1YQCMn`](https://openreview.net/forum?id=9QPH1YQCMn)

## Intuition

For a causal language model, each token was chosen at generation time from a highly peaked distribution when the model **memorized** or strongly prefers a continuation; for unrelated text, the realized token is often more “ordinary” relative to the full vocabulary distribution at that position.

The **infilling score** asks: if we treat the observed text as fixed and measure how **forced** each token is—especially under a **counterfactual** where one position is replaced by the model’s greedy choice—do we see the kind of tight coupling between positions that is typical of memorized or training-like snippets?

Lower aggregate scores (after the construction below) tend to indicate **more member-like** behavior under the original infilling MIA convention; downstream tools (e.g. `mia_eval`) may **negate or flip** scores for evaluation so that “higher = member” on a validation split—always check the `score_orientation` field in your run config.

## How it works (step by step)

Settings: integer **`m`** (lookahead length) and fraction **`k`** (bottom-`k` mass). Defaults in many configs are on the order of `m ∈ {1,…,7}` and `k ∈ {0.05,…,0.2}`.

### 1. Encode the text

The input string is tokenized to `input_ids` of length `seq_len`. One forward pass on the **real** sequence yields `log_probs_real`: for each position `t`, the row `log_probs_real[t, :]` is `log softmax` over the vocabulary for the distribution **predicting the token at position `t+1`**.

### 2. Per-token infilling statistic (positions `i = 1 … seq_len-1`)

For each index `i` (focusing on how the model scores the **actual** token `x_i` given prefix `x_{<i}`):

1. **Z-score the realized token under the real prefix**  
   At position `i-1`, look at the full vocabulary distribution. Let `μ` and `σ` be the mean and standard deviation of `log p` under that distribution (weighted by probabilities). Let `x_i` be the true token. A standardized score compares `log p(x_i | x_{<i})` to that bulk.

2. **Subtract the greedy alternative**  
   Let `x*_i = argmax_v p(v | x_{<i})`. The construction subtracts a similar standardized term for `x*_i` so that the score emphasizes deviation **relative to the mode**, not only high absolute probability.

3. **Counterfactual sequence**  
   Build `input_ids*` identical to `input_ids` except position `i` is set to `x*_i`. Run a **second** forward pass to get `log_probs_star` for this counterfactual sequence.

4. **Lookahead alignment (`m`)**  
   For each `j` in `{i+1, …, min(i+m, seq_len-1)}`, add a standardized contribution from the **real** sequence at `j`, and subtract a standardized contribution from the **counterfactual** sequence at `j`.  
   So `m` controls how many **downstream** positions are compared when measuring how much changing position `i` to the greedy token perturbs the model’s story of the rest of the string.

The result is one scalar **`infilling_score_token`** per position `i`.

### 3. Sequence-level score

Collect all position scores into a vector of length `seq_len - 1`. Let **`k_count = max(1, floor((seq_len-1) * k))`**. Take the **`k_count` smallest** token scores (the **bottom-`k`** fraction of positions—the ones that look least “forced” or most anomalous under the construction). The **mean** of those bottom values is returned as the **infilling score** for the whole text.

So **`k`** makes the score **robust** to a few positions by focusing on the worst (most member-suspicious) tail; **`m`** couples each position to a short **local window** of counterfactual consistency.

### 4. Complexity

Each token index can trigger an extra forward pass for the counterfactual sequence, so cost scales roughly with **sequence length × (work per position)**. GPU runtime can be substantial for long texts or large models (see Notes below).

## Files

- `infilling_score.py`: main attack implementation (standalone script style).
- `test.py`: dumps the attack results into a json file.
- `infilling_score_gpt_neo.json`: the results of running the attack on GPT-Neo.

## Relation to `mia_eval`

`mia_eval/scoring_infilling.py` implements the same mathematical construction but is wired for batch evaluation and shared tokenizer/device handling with WBC and memTrace.

## Notes

- The results obtained from running `infilling_score.py` were similar to those of the original paper. However, it took substantially longer to run than expected at about 40 minutes on an Nvidia RTX 3070.
