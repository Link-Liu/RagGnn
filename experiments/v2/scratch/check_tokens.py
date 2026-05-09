import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from transformers import AutoTokenizer

# 尝试本地路径
paths = [
    r"c:\LCY\checkpoints\modelscope\LLM-Research\Meta-Llama-3___1-8B-Instruct",
    "LLM-Research/Meta-Llama-3.1-8B-Instruct",
]
t = None
for p in paths:
    try:
        t = AutoTokenizer.from_pretrained(p, trust_remote_code=True)
        print(f"Loaded from: {p}")
        break
    except:
        continue

if t is None:
    print("Failed to load tokenizer")
    exit(1)

# 检查 label tokens
for label in ["Enzyme", "Non-enzyme", "enzyme", "non-enzyme", "0", "1", "Yes", "No"]:
    ids = t.encode(label, add_special_tokens=False)
    tokens = [t.decode([x]) for x in ids]
    print(f"  '{label}' -> IDs: {ids} -> tokens: {tokens}")

# 关键检查
e_ids = t.encode("Enzyme", add_special_tokens=False)
n_ids = t.encode("Non-enzyme", add_special_tokens=False)
print(f"\nEnzyme first token ID: {e_ids[0]}")
print(f"Non-enzyme first token ID: {n_ids[0]}")
print(f"Same first token? {e_ids[0] == n_ids[0]}")
