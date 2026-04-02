import json

# with open("data/wikipedia_(en)_ngram_13_0.8.jsonl", "r") as f:
#     first_line = f.readline()
#     second_line = f.readline()
#     print(repr(first_line))
#     print(repr(second_line))


results = {
    "model": "EleutherAI/gpt-neo-1.3B",
    "dataset": "MIMIR wikipedia_(en)",
    "n_samples": 50,
    "infilling_score_m1_auc": 0.5500,
    "infilling_score_m5_auc": 0.5652,
}

with open("infilling_score_gpt_neo.json", "w") as f:
    json.dump(results, f, indent=2)