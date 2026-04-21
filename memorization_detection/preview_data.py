import os
from huggingface_hub import login
from datasets import load_dataset

HF_TOKEN = os.environ["HF_TOKEN"]

login(token=HF_TOKEN)

config = "arxiv"   # change this if you want a different domain

ds = load_dataset(
    "iamgroot42/mimir",
    config,
    token=HF_TOKEN,
    trust_remote_code=True,
)

first_split = list(ds.keys())[0]

sample = ds["ngram_7_0.2"][0]

member_text = sample["member"]
nonmember_text = sample["nonmember"]

def split_text(text, prefix_len=200):
    return text[:prefix_len], text[prefix_len:prefix_len+200]

prefix_m, suffix_m = split_text(member_text)
prefix_nm, suffix_nm = split_text(nonmember_text)

print("MEMBER PREFIX:\n", prefix_m[:200])
print("\nMEMBER SUFFIX:\n", suffix_m[:200])

print("\n---\n")

print("NONMEMBER PREFIX:\n", prefix_nm[:200])
print("\nNONMEMBER SUFFIX:\n", suffix_nm[:200])