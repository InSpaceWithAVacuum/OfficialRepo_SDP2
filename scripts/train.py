# scripts/train.py
import os, json, hashlib, warnings, re
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict
from transformers import (
    DataCollatorWithPadding,
    AutoTokenizer,
    ModernBertConfig,
    ModernBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)

# ---- Paths ----
DATA_PARQUET = "data/edge_text.parquet"
LABELS_JSON  = "data/labels.json"
OUT_DIR_CKPT = "models/hf_ckpt"
OUT_DIR_BEST = "models/hf_best"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.makedirs("models", exist_ok=True)
torch.set_float32_matmul_precision("high")
set_seed(42)

# ---- Load ----
print("[*] Loading data…")
df = pd.read_parquet(DATA_PARQUET)
meta = json.load(open(LABELS_JSON, "r"))
labels = meta["labels"]
id2label = {i: l for i, l in enumerate(labels)}
label2id = {l: i for i, l in enumerate(labels)}

assert "text" in df.columns, "edge_text.parquet must contain 'text'."
label_col = "label_id" if "label_id" in df.columns else ("labels" if "labels" in df.columns else None)
assert label_col is not None, "edge_text.parquet must have 'label_id' or 'labels'."
if label_col == "labels":
    if not np.issubdtype(df["labels"].dtype, np.integer):
        df["labels"] = df["labels"].map(label2id).astype(int)
else:
    df.rename(columns={"label_id": "labels"}, inplace=True)

# ---- De-dup & sanity ----
print("[*] Deduplicating by exact text…")
n0 = len(df)
df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
print(f"    Removed {n0 - len(df)} duplicates. Now {len(df)} rows.")

print("[*] Scanning for label tokens inside text (sanity)…")
bad_labels = [l for l in labels if df["text"].str.contains(fr"\b{re.escape(l)}\b", case=False, regex=True).any()]
if bad_labels:
    print(f"    WARNING: Found label tokens in text (possible leakage): {bad_labels}")

# ---- Split ----
print("[*] Stratified train/test split…")
train_idx, test_idx = train_test_split(
    df.index, test_size=0.20, random_state=42, stratify=df["labels"]
)
df_train, df_test = df.loc[train_idx].reset_index(drop=True), df.loc[test_idx].reset_index(drop=True)
print(f"    Train: {len(df_train)} | Test: {len(df_test)}")

def md5_series(s: pd.Series) -> set:
    return set(s.astype(str).map(lambda x: hashlib.md5(x.encode()).hexdigest()))
overlap = md5_series(df_train["text"]).intersection(md5_series(df_test["text"]))
print(f"[*] Train/Test text overlap (should be 0): {len(overlap)}")
assert len(overlap) == 0, "Train/test text overlap detected — recheck preprocessing."

# ---- HF Datasets ----
print("[*] Building HuggingFace datasets…")
ds = DatasetDict({
    "train": Dataset.from_pandas(df_train),
    "test":  Dataset.from_pandas(df_test),
})

tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
def tokfn(batch):
    return tok(batch["text"], max_length=512, truncation=True)

ds = ds.map(tokfn, batched=True)
if "labels" not in ds["train"].column_names:
    raise ValueError("Label column must be 'labels' after preprocessing.")
keep = {"input_ids", "attention_mask", "labels"}
ds = ds.remove_columns([c for c in ds["train"].column_names if c not in keep])
ds.set_format(type="torch")
data_collator = DataCollatorWithPadding(tokenizer=tok, pad_to_multiple_of=8)

# ---- Class weights (inverse-freq^0.5) with Normal boost ----
cnt = Counter(df_train["labels"].tolist())
w = {i: 1.0 / (cnt[i] ** 0.5) for i in range(len(labels))}

if "Normal" in labels:
    normal_idx = labels.index("Normal")
    w[normal_idx] *= 1.1

weights_vec = torch.tensor([w[i] for i in range(len(labels))], dtype=torch.float)
print("[*] Class weights (after Normal boost):", {labels[i]: round(weights_vec[i].item(), 4) for i in range(len(labels))})


# ---- Model ----
print("[*] Loading model…")
cfg = ModernBertConfig.from_pretrained(
    "answerdotai/ModernBERT-base",
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
)

cfg.reference_compile = False  # avoid Triton/compile path
model = ModernBertForSequenceClassification.from_pretrained(
    "answerdotai/ModernBERT-base",
    config=cfg,
)
model.set_attn_implementation("sdpa")

# ---- Custom Trainer with weighted loss ----
class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = F.cross_entropy(logits, labels.long(), weight=self.class_weights.to(logits.device))
        return (loss, outputs) if return_outputs else loss

# ---- Training args (longer, lower LR, warmup) ----
print("[*] Setting up training…")
args = TrainingArguments(
    output_dir=OUT_DIR_CKPT,
    num_train_epochs=3,
    learning_rate=3e-5,
    warmup_ratio=0.06,
    per_device_train_batch_size=2,     # fill 12GB; drop to 20/16 if OOM
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=2,
    bf16=torch.cuda.is_bf16_supported(),# faster on Ada; else falls back
    fp16=False,                         # prefer bf16
    tf32=True,                          # speed up matmul
    optim="adamw_torch_fused",          # faster optimizer
    dataloader_num_workers=os.cpu_count()//2 or 4,
    dataloader_pin_memory=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",
    greater_is_better=True,
    logging_steps=200,
    report_to="none",
    save_total_limit=3,
)

def compute_metrics(eval_pred):
    logits, y_true = eval_pred
    y_pred = np.argmax(logits, axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }

trainer = WeightedTrainer(
    class_weights=weights_vec,
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    tokenizer=tok,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ---- Train ----
print("[*] Training…")
trainer.train()

# ---- Evaluate ----
print("[*] Evaluating…")
pred = trainer.predict(ds["test"])
y_true = pred.label_ids
y_pred = np.argmax(pred.predictions, axis=1)

print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=labels, digits=4))
print("=== Confusion Matrix (rows=true, cols=pred) ===")
print(confusion_matrix(y_true, y_pred))

# ---- Save best ----
print("[*] Saving best model and tokenizer…")
os.makedirs(OUT_DIR_BEST, exist_ok=True)
trainer.model.save_pretrained(OUT_DIR_BEST)
tok.save_pretrained(OUT_DIR_BEST)
print(f"Saved to {OUT_DIR_BEST}")