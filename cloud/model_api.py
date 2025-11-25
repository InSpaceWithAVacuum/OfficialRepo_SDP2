import onnxruntime as ort
import numpy as np
import json, sys, os
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer

# --- 1. Load models and files on startup ---
MODEL_PATH = "models/model.onnx"
TOKENIZER_PATH = "models/hf_best" # The tokenizer is still loaded from the hf_best folder
LABELS_PATH = "data/labels.json"

try:
    # Load the ONNX inference session and use CUDA
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(MODEL_PATH, providers=providers)
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    
    # Load the labels
    labels = json.load(open(LABELS_PATH))["labels"]
    
    
except FileNotFoundError as e:
    print(f"FATAL ERROR: Could not find required file. {e}")
    print("Please make sure 'models/model.onnx', 'models/hf_best', and 'data/labels.json' exist.")
    sys.exit(1)
except Exception as e:
    print(f"FATAL ERROR loading model or tokenizer: {e}")
    sys.exit(1)

print(f"Successfully loaded ONNX model from {MODEL_PATH}")
print(f"Successfully loaded Tokenizer from {TOKENIZER_PATH}")
print(f"Successfully loaded {len(labels)} labels from {LABELS_PATH}")

CAL_PATH = "models/calibration.json"
TEMP = 1.0
if os.path.exists(CAL_PATH):
    try:
        TEMP = float(json.load(open(CAL_PATH))["temperature"])
    except Exception:
        TEMP = 1.0

app = FastAPI(title="LLM-IDS Model (ONNX)")

class Item(BaseModel):
    text: str

def softmax(x):
    """Compute softmax values for a set of scores x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# --- 2. Define the classification endpoint ---
@app.post("/classify")
def classify(it: Item):
    """
    Classifies the input text using the ONNX model.
    """
    # 1. Tokenize the input text
    inputs = tokenizer(
        it.text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="np" # Return NumPy arrays for ONNX
    )
    
    # 2. Prepare inputs for ONNX session
    # We need to make sure the data types are correct (int64)
    onnx_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64)
    }

    # 3. Run inference
    # The output is a list, we take the first element (logits)
    logits = session.run(None, onnx_inputs)[0][0] # Get the first (and only) logit array

    logits = logits / max(TEMP, 1e-6)
    # 4) Post-process
    probabilities = softmax(logits)

    # (c) Choose final label
    prediction_index = int(np.argmax(probabilities))
    prediction_label = labels[prediction_index]
    confidence = float(probabilities[prediction_index])
    return {"label": prediction_label, "confidence": confidence}
