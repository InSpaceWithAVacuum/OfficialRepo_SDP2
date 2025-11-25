# scripts/calibrate_temperature.py
import json, numpy as np, torch, torch.nn.functional as F
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

MODEL_DIR = "models/hf_best"
DATA_PARQUET = "data/edge_text.parquet"
LABELS_JSON  = "data/labels.json"
OUT_JSON     = "models/calibration.json"

print("[*] Loading data/model…")
labels = json.load(open(LABELS_JSON))["labels"]
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()

df = pd.read_parquet(DATA_PARQUET)
# small held-out calibration set (e.g., 2000 samples or all if fewer)
cal = df.sample(min(2000, len(df)), random_state=17).reset_index(drop=True)

# tokenize
X = tok(cal["text"].tolist(), max_length=128, padding=True, truncation=True, return_tensors="pt")
y = torch.tensor(cal["label_id"].tolist(), dtype=torch.long)

with torch.no_grad():
    logits = model(input_ids=X["input_ids"], attention_mask=X["attention_mask"]).logits  # (N, C)

# Learn temperature T by minimizing NLL
T = torch.tensor(1.0, requires_grad=True)
opt = torch.optim.LBFGS([T], lr=0.1, max_iter=50, line_search_fn="strong_wolfe")

def nll_loss():
    opt.zero_grad()
    scaled = logits / T.clamp_min(0.5)  # keep T sane
    loss = F.cross_entropy(scaled, y)
    loss.backward()
    return loss

print("[*] Optimizing temperature…")
opt.step(nll_loss)
T_star = float(T.detach().clamp(0.5, 5.0))

print(f"[*] Best temperature T = {T_star:.4f}")
json.dump({"temperature": T_star}, open(OUT_JSON, "w"))
print(f"[*] Saved {OUT_JSON}")
