import pandas as pd
import requests
import random
import json
import sys
import re

#  1. Simulation Configuration 
TOTAL_ALERTS = 100
NORMAL_TRAFFIC_RATIO = 0.8 # 80% of alerts will be "Normal"

#  2. Load helper files 
try:
    labels_list = json.load(open("data/labels.json"))["labels"]
    NORMAL_LABEL = "Normal"
    NORMAL_ID = labels_list.index(NORMAL_LABEL)
except (FileNotFoundError, ValueError) as e:
    print(f"ERROR: Could not load 'data/labels.json' or find 'Normal' label. {e}")
    print("       Did you run 'python scripts/prep.py'?")
    sys.exit(1)
    
try:
    DF_full = pd.read_parquet("data/edge_text.parquet")
except FileNotFoundError:
    print("ERROR: data/edge_text.parquet not found.")
    print("       Did you run 'python scripts/prep.py'?")
    sys.exit(1)

#  3. Create the Test Sample Mix 
num_normal = int(TOTAL_ALERTS * NORMAL_TRAFFIC_RATIO)
num_attack = TOTAL_ALERTS - num_normal

df_normal = DF_full[DF_full['label_id'] == NORMAL_ID]
df_attack = DF_full[DF_full['label_id'] != NORMAL_ID]

normal_samples = df_normal.sample(num_normal, replace=True)
attack_samples = df_attack.sample(num_attack, replace=True)

DF_test = pd.concat([normal_samples, attack_samples]).sample(frac=1).reset_index(drop=True)
print(f"--- Starting simulation of {TOTAL_ALERTS} alerts ({num_normal} Normal, {num_attack} Attack) ---")

#  4. Initialize analysis counters (using your report's definitions from 7.3) 
# Note: These definitions are swapped from the standard.
tp = 0  # True Positive:  Report defines as 'Normal' traffic correctly identified as 'Normal'
fn = 0  # False Negative: Report defines as 'Normal' traffic misclassified as 'Attack'
tn = 0  # True Negative:  Report defines as 'Attack' correctly identified as 'Attack'
fp = 0  # False Positive: Report defines as 'Attack' misclassified as 'Normal'

#  5. Run the replay loop 
for i, (_, r) in enumerate(DF_test.iterrows()):
    text = r["text"]
    true_label = labels_list[r["label_id"]]

    print(f"\n--- [Alert {i+1}/{TOTAL_ALERTS}] (True Label: {true_label}) ---")

    # 1) Classify
    try:
        c = requests.post("http://127.0.0.1:8000/classify", json={"text": text}).json()
        predicted_label = c.get('label', 'ERROR')
    except Exception as e:
        print(f"  [ERROR] Model API (port 8000) failed: {e}")
        continue
        
    # 2) Decide
    try:
        d = requests.post("http://127.0.0.1:8001/decide", json=c).json()
    except Exception as e:
        print(f"  [ERROR] Policy API (port 8001) failed: {e}")
        continue

    # 3. Corrected src/port parsing
    src = "Unknown"; port = 0
    try:
        src_match = re.search(r'src\s+([0-9a-zA-Z\.\-:]+)\s+(?:->|→)\s+dst', text)
        if src_match: src = src_match.group(1)
        port_match = re.search(r'dport\s+(\d+)', text)   # accepts "443" or "443."
        if port_match: port = int(port_match.group(1))
    except Exception:
        pass

    # 4) Deploy
    try:
        p = {
            "action": d.get("action", "LOG_ONLY"),
            "src": src,
            "port": port,
            "source_label": d.get("source_label", "Unknown")
        }
        o = requests.post("http://127.0.0.1:8002/deploy", json=p).json()
    except Exception as e:
        print(f"  [ERROR] Orchestrator API (port 8002) failed: {e}")
        continue

    # 5) Log for analysis
    print(f"  [RESULT] True: '{true_label}', Predicted: '{predicted_label}'")
    
    # Apply logic based on report's definitions
    is_true_normal = (true_label == NORMAL_LABEL)
    is_predicted_normal = (predicted_label == NORMAL_LABEL)

    if is_true_normal:
        if is_predicted_normal:
            tp += 1 # Normal as Normal
        else:
            fn += 1 # Normal as Attack
    else: # Is a true attack
        if is_predicted_normal:
            fp += 1 # Attack as Normal
        else:
            tn += 1 # Attack as Attack

# --- 6. Print Final Analysis ---
print("\n--- Simulation Complete ---")
print("--- Performance Analysis (based on report 7.3 definitions) ---")
print(f"\nTotal Alerts Processed: {TOTAL_ALERTS}")

print("\n--- Raw Counts ---")
print(f"True Positives  (TP) [Normal -> Normal]: {tp}")
print(f"False Negatives (FN) [Normal -> Attack]: {fn}")
print(f"True Negatives  (TN) [Attack -> Attack]: {tn}")
print(f"False Positives (FP) [Attack -> Normal]: {fp}")

#  Calculate Metrics (using report's formulas) 
try:
    # [cite_start]Accuracy (TP+TN)/(TP + TN+FP+FN) [cite: 768]
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    print(f"\nOverall Accuracy: {accuracy * 100:.2f}%")
except ZeroDivisionError:
    print("\nOverall Accuracy: N/A (no data)")

try:
    # [cite_start]False Positive Rate (FPR) = FP / (FP + TN) [cite: 1051]
    fpr = fp / (fp + tn)
    print(f"False Positive Rate (Attacks misclassified as Normal): {fpr * 100:.2f}%")
except ZeroDivisionError:
    print("False Positive Rate: N/A (no attacks tested)")
    
try:
    # [cite_start]False Negative Rate (FNR) = FN / (FN + TP) [cite: 1055]
    fnr = fn / (fn + tp)
    print(f"False Negative Rate (Normal misclassified as Attack): {fnr * 100:.2f}%")
except ZeroDivisionError:
    print("False Negative Rate: N/A (no normal traffic tested)")

print("\n----------------------- End of Report -----------------------")