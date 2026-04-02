# Infilling Scores

This implementation of infilling scoring for membership inference attacks was from: https://openreview.net/forum?id=9QPH1YQCMn

## Files

- `infilling_score.py`: main attack implementation. Uses top-k for sampling
- `test.py`: dumps the attack results into a json file
- `infilling_score_gpt_neo.json`: the results of running the attack on GPT-Neo

## Notes

- The results obtained from running `infilling_score.py` was similar to those of the original paper. However, it took substaintially longer to run than expected at about 40 minutes on a Nvidia RTX 3070.