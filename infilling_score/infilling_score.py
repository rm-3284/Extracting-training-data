import json
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return model, tokenizer

def get_log_probs(model, input_ids):
    with torch.no_grad():
        outputs = model(input_ids.unsqueeze(0))
    logits = outputs.logits[0]
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs

def get_mean_std(log_probs_at_position):
    probs = torch.exp(log_probs_at_position)
    
    mean = (probs * log_probs_at_position).sum()
    variance = (probs * (log_probs_at_position - mean) ** 2).sum()
    std = torch.sqrt(variance + 1e-8)  
    
    return mean, std

def infilling_score_token(log_probs_real, model, input_ids, i, m):
    seq_len = input_ids.shape[0]

    log_p_xi = log_probs_real[i - 1, input_ids[i]]
    mean_i, std_i = get_mean_std(log_probs_real[i - 1])

    r = (log_p_xi - mean_i) / std_i
    x_star_i = torch.argmax(log_probs_real[i - 1]).item()

    log_p_xstar = log_probs_real[i - 1, x_star_i]
    r = r - (log_p_xstar - mean_i) / std_i

    input_ids_star = input_ids.clone()
    input_ids_star[i] = x_star_i
    log_probs_star = get_log_probs(model, input_ids_star)

    for j in range(i + 1, min(i + m + 1, seq_len)):
        log_p_xj_real = log_probs_real[j - 1, input_ids[j]]
        mean_j, std_j = get_mean_std(log_probs_real[j - 1])
        r += (log_p_xj_real - mean_j) / std_j

        log_p_xj_star = log_probs_star[j - 1, input_ids[j]]
        mean_j_star, std_j_star = get_mean_std(log_probs_star[j - 1])
        r -= (log_p_xj_star - mean_j_star) / std_j_star

    return r.item()

def infilling_score(model, tokenizer, text, m=5, k=0.1):
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(text, return_tensors="pt").squeeze(0).to(device)
    seq_len = input_ids.shape[0]

    log_probs_real = get_log_probs(model, input_ids)

    token_scores = []
    for i in range(1, seq_len):
        score = infilling_score_token(log_probs_real, model, input_ids, i, m)
        token_scores.append(score)

    token_scores = torch.tensor(token_scores)
    k_count = max(1, int(len(token_scores) * k))
    bottom_k_scores = torch.topk(token_scores, k_count, largest=False).values

    return bottom_k_scores.mean().item()

#----------------------------------------------------------------

def evaluate_on_mimir(model, tokenizer, n_samples=50, m=1, k=0.1):
    def load_jsonl(path, n):
        texts = []
        with open(path, "r") as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                texts.append(json.loads(line))
        return texts
    
    members = load_jsonl("data/wikipedia_(en)_train_ngram_13_0.8.jsonl", n_samples)
    nonmembers = load_jsonl("data/wikipedia_(en)_test_ngram_13_0.8.jsonl", n_samples)
    
    scores = []
    labels = []
    
    for i, (member, nonmember) in enumerate(zip(members, nonmembers)):
        member_score = infilling_score(model, tokenizer, member, m=m, k=k)
        scores.append(member_score)
        labels.append(1)
        
        nonmember_score = infilling_score(model, tokenizer, nonmember, m=m, k=k)
        scores.append(nonmember_score)
        labels.append(0)
        
        if i % 5 == 0:
            print(f"Processed {i}/{n_samples} pairs, member: {member_score:.4f}, nonmember: {nonmember_score:.4f}")
    
    auc = roc_auc_score(labels, scores)
    print(f"AUC-ROC on MIMIR wikipedia_(en): {auc:.4f}")
    return auc

def evaluate_on_wikimia(model, tokenizer, length=64, n_samples=50, m=5, k=0.1):
    dataset = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{length}")
    scores = []
    labels = []
    
    for i, example in enumerate(dataset):
        if i >= n_samples:
            break
        
        score = infilling_score(model, tokenizer, example["input"], m=m, k=k)
        scores.append(score)
        labels.append(example["label"])
        
        if i % 5 == 0:
            print(f"Processed {i}/{n_samples} samples...")
    
    auc = roc_auc_score(labels, scores)
    print(f"AUC-ROC on WikiMIA length {length}: {auc:.4f}")
    return auc

#----------------------------------------------------------------

if __name__ == "__main__":
    model, tokenizer = load_model("EleutherAI/gpt-neo-1.3B")
    print(f"Model device: {next(model.parameters()).device}")
    
    print("\nm=1:")
    evaluate_on_mimir(model, tokenizer, n_samples=50, m=1, k=0.1)
    
    print("\nm=5:")
    evaluate_on_mimir(model, tokenizer, n_samples=50, m=5, k=0.1)
    # evaluate_on_wikimia(model, tokenizer, length=32, n_samples=20, m=1)