import numpy as np, json, matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd, torch

labels = json.load(open("data/labels.json"))["labels"]
tok = AutoTokenizer.from_pretrained("models/hf_best")
model = AutoModelForSequenceClassification.from_pretrained("models/hf_best").eval()

df = pd.read_parquet("data/edge_text.parquet").sample(2000, random_state=0)
X = tok(df["text"].tolist(), max_length=128, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    logits = model(**{k: v for k, v in X.items() if k != "token_type_ids"}).logits
pred = np.argmax(logits.numpy(), 1)
y = df["label_id"].to_numpy()

# Multiclass confusion matrix
cm = confusion_matrix(y, pred, labels=list(range(len(labels))))
fig, ax = plt.subplots(figsize=(12, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
plt.title("Multiclass Confusion Matrix", fontsize=14, pad=20)
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("True", fontsize=12)
plt.tight_layout(pad=3.0)
plt.savefig("models/cm_multiclass.png", dpi=200)
plt.close()

# Binary collapse: normal vs attack
normal = labels.index("Normal") if "Normal" in labels else 0
yb = (y == normal).astype(int)
pb = (pred == normal).astype(int)
cmb = confusion_matrix(yb, pb, labels=[0, 1])

fig, ax = plt.subplots(figsize=(6, 5))
dispb = ConfusionMatrixDisplay(confusion_matrix=cmb, display_labels=["Attack", "Normal"])
dispb.plot(ax=ax, cmap="Greens", colorbar=False)
plt.title("Binary Confusion Matrix", fontsize=12, pad=15)
plt.xlabel("Predicted", fontsize=11)
plt.ylabel("True", fontsize=11)
plt.tight_layout(pad=2.0)
plt.savefig("models/cm_binary.png", dpi=200)
plt.close()
