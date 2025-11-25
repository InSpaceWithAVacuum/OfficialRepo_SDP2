# scripts/prep.py
import os, json
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- Configuration ---
SRC = "data/ML-EdgeIIoT-dataset.csv"
os.makedirs("data", exist_ok=True)
WINDOW_SIZE = 10  # Number of log lines to combine into one sample
STEP_SIZE = 5     # How many lines to slide the window forward. Overlapping windows (step < size) are good.

# Columns used to build the single-line text samples
KEEP = [
    "frame.time", "ip.src_host", "ip.dst_host",
    "tcp.flags", "tcp.ack", "tcp.len", "tcp.dstport",
    "Attack_type"
]

# --- Initial Loading and Text Formatting ---
print("[*] Loading CSV…")
df = pd.read_csv(SRC, usecols=KEEP, low_memory=False)

df.fillna({"ip.src_host": "Unknown", "ip.dst_host": "Unknown", "tcp.flags": "N/A"}, inplace=True)
for c in ["tcp.ack", "tcp.len", "tcp.dstport"]:
    df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(0).astype(str)

# Ensure data is sorted by time to make windows chronological
df = df.sort_values("frame.time", ascending=True).reset_index(drop=True)

def row_to_text(r):
    return (f"At {r['frame.time']}, src {r['ip.src_host']} -> dst {r['ip.dst_host']}, "
            f"flags {r['tcp.flags']}, ack {r['tcp.ack']}, size {r['tcp.len']}, dport {r['tcp.dstport']}.")

df["text"] = df.apply(row_to_text, axis=1)

labels = sorted(df["Attack_type"].unique().tolist())
# --- THIS IS THE CORRECTED LINE ---
label2id = {l: i for i, l in enumerate(labels)}
df["label_id"] = df["Attack_type"].map(label2id)

# --- Aggregated Window Creation Logic ---
print(f"[*] Creating aggregated window samples (size={WINDOW_SIZE}, step={STEP_SIZE})…")

def create_aggregated_windows(group):
    """
    Slides a window over a group of rows (from a single source IP),
    concatenates their text, and determines the window's single label.
    """
    new_rows = []
    # Ensure group is sorted if not already
    group = group.sort_values("frame.time").reset_index(drop=True)
    
    for i in range(0, len(group) - WINDOW_SIZE + 1, STEP_SIZE):
        window = group.iloc[i:i + WINDOW_SIZE]
        
        # 1. Concatenate Text: Join all text fields in the window
        combined_text = "\n".join(window["text"].tolist())
        
        # 2. Determine the Label: Prioritize attacks
        attack_types = window["Attack_type"].value_counts()
        
        # If the only type is "Normal", label is "Normal"
        if len(attack_types) == 1 and attack_types.index[0] == "Normal":
            final_label = "Normal"
        else:
            # Otherwise, remove "Normal" and find the most frequent attack
            if "Normal" in attack_types.index:
                attack_types = attack_types.drop("Normal")
            final_label = attack_types.index[0]

        new_rows.append({
            "text": combined_text,
            "label_id": label2id[final_label]
        })
        
    return pd.DataFrame(new_rows)

# Group by source IP and apply the windowing function
tqdm.pandas(desc="Building Aggregated Windows")
grouped = df.groupby("ip.src_host")
df_agg = grouped.progress_apply(create_aggregated_windows).reset_index(drop=True)

# --- Save the final aggregated dataset ---
df_agg.to_parquet("data/edge_text.parquet")
json.dump({"labels": labels}, open("data/labels.json", "w"))

print(f"\n[*] Saved data/edge_text.parquet with {len(df_agg)} aggregated rows.")
print("[*] Saved data/labels.json")