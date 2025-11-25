# scripts/export_onnx.py
import torch, onnx, onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "models/hf_best"
ONNX_OUT  = "models/model.onnx"

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
m   = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()

# dummy batch of size 1, seq len 128
dummy_ids  = torch.randint(0, tok.vocab_size, (1,128), dtype=torch.long)
dummy_mask = torch.ones(1,128, dtype=torch.long)

torch.onnx.export(
    m,
    (dummy_ids, dummy_mask),
    ONNX_OUT,
    input_names=["input_ids","attention_mask"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch"}, "attention_mask": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=18,                 # ← export directly to 18
    do_constant_folding=True
)

onnx.checker.check_model(ONNX_OUT)

# quick sanity inference
sess = ort.InferenceSession(ONNX_OUT, providers=["CUDAExecutionProvider","CPUExecutionProvider"])
sample = tok("At t, src 1.1.1.1 -> dst 2.2.2.2, flags S, ack 0, size 0, dport 80.", 
             max_length=128, padding="max_length", truncation=True, return_tensors="np")
outs = sess.run(None, {
    "input_ids": sample["input_ids"].astype("int64"),
    "attention_mask": sample["attention_mask"].astype("int64"),
})[0]
print("ONNX logits shape:", outs.shape)
print("OK ✓")
