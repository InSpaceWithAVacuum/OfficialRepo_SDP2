import argparse, sys, json, re, random, math, time
import pandas as pd
import requests

# --- Configuration ---
MODEL_URL_SINGLE = "http://127.0.0.1:8000/classify"
MODEL_URL_BATCH  = "http://127.0.0.1:8000/classify_batch"
POLICY_URL       = "http://127.0.0.1:8001/decide"
ORCH_URL         = "http://127.0.0.1:8002/deploy"
LOG_FILE         = "demo_logs.jsonl" 

def parse_args():
    ap = argparse.ArgumentParser(description="Batch+Burst replay to stress the IDS loop")
    ap.add_argument("--total", type=int, default=300, help="Total alerts to send")
    ap.add_argument("--normal-ratio", type=float, default=0.8, help="Fraction that are Normal")
    ap.add_argument("--sources", type=int, default=20, help="Distinct attacker/benign source IPs")
    ap.add_argument("--burst", type=int, default=3, help="Burst factor per source (>=1)")
    ap.add_argument("--sleep", type=float, default=0.1, help="Sleep seconds between batches")
    return ap.parse_args()

def load_data():
    try:
        labels_list = json.load(open("data/labels.json"))["labels"]
        normal_label = "Normal"
        normal_id = labels_list.index(normal_label)
    except Exception as e:
        print(f"[FATAL] labels.json/Normal missing: {e}"); sys.exit(1)
    try:
        df = pd.read_parquet("data/edge_text.parquet")
        df["label_id"] = df["label_id"].astype(int)
    except Exception as e:
        print(f"[FATAL] edge_text.parquet missing: {e}"); sys.exit(1)
    return df, labels_list, normal_label, normal_id

def choose_sources(n):
    pool = []
    for _ in range(n):
        if random.random() < 0.6:
            pool.append(f"192.168.{random.randint(0,254)}.{random.randint(1,254)}")
        else:
            pool.append(f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}")
    return pool

def inject_dport(text: str, dport: int) -> str:
    t = re.sub(r'dport\s+(\d+)', f"dport {dport}", text)
    return t

def classify_batch(texts):
    try:
        r = requests.post(MODEL_URL_BATCH, json={"texts": texts}, timeout=10)
        if r.ok:
            return r.json()["results"]
    except Exception:
        pass
    out = []
    for tx in texts:
        try:
            c = requests.post(MODEL_URL_SINGLE, json={"text": tx}, timeout=5).json()
        except Exception as e:
            c = {"label": "ERROR", "confidence": 0.0}
        out.append(c)
    return out

def main():
    args = parse_args()
    df, labels_list, NORMAL, NORMAL_ID = load_data()

    # --- CHANGE: REMOVED THE FILE WIPE LOGIC SO DATA ACCUMULATES ---
    # (The dashboard reset button handles the wiping now)

    total = args.total
    num_normal = int(total * args.normal_ratio)
    num_attack = total - num_normal

    df_n = df[df["label_id"] == NORMAL_ID]
    df_a = df[df["label_id"] != NORMAL_ID]

    if len(df_n)==0 or len(df_a)==0:
        print("[FATAL] Need both Normal and Attack rows in dataset"); sys.exit(1)

    S = choose_sources(args.sources)
    ports_common = [22, 53, 80, 123, 443, 8080, 8443, 3389]

    # Build mixed test set
    normal_samples = df_n.sample(num_normal, replace=True).reset_index(drop=True)
    attack_samples = df_a.sample(num_attack, replace=True).reset_index(drop=True)
    test = pd.concat([normal_samples, attack_samples]).sample(frac=1, random_state=7).reset_index(drop=True)

    print(f"--- Starting batch replay: total={total} ---")
    
    tp=fn=tn=fp=0
    batch_size = max(8, min(64, args.sources * args.burst))
    i = 0
    while i < len(test):
        end = min(len(test), i+batch_size)
        chunk = test.iloc[i:end]
        chosen_srcs = [random.choice(S) for _ in range(len(chunk))]
        
        if args.burst > 1:
            for b in range(0, len(chunk), args.burst):
                src_b = random.choice(S)
                for j in range(b, min(b+args.burst, len(chunk))):
                    chosen_srcs[j] = src_b
                    
        chosen_ports = [random.choice(ports_common) for _ in range(len(chunk))]
        texts = [inject_dport(t, p) for t,p in zip(chunk["text"], chosen_ports)]
        
        t0 = time.time()
        results = classify_batch(texts)
        current_latency = (time.time() - t0) / len(chunk)

        for k, (idx, row) in enumerate(chunk.iterrows()):
            true_lab = labels_list[row["label_id"]]
            c = results[k]
            pred_lab = c.get("label", "ERROR")
            conf = float(c.get("confidence", 0.0))

            try:
                d = requests.post(POLICY_URL, json={"label": pred_lab, "confidence": conf}, timeout=5).json()
            except:
                d = {"action":"LOG_ONLY"}

            action = d.get("action", "LOG_ONLY")
            plan = {"action": action, "src": chosen_srcs[k], "port": chosen_ports[k], "source_label": pred_lab}
            try:
                requests.post(ORCH_URL, json=plan, timeout=1) 
            except: 
                pass

            is_true_attack = (true_lab != NORMAL)
            is_pred_attack = (pred_lab != NORMAL)
            
            if is_true_attack:
                if is_pred_attack: tp += 1
                else: fn += 1
            else:
                if is_pred_attack: fp += 1
                else: tn += 1

            log_entry = {
                "timestamp": time.time(),
                "time_str": time.strftime("%H:%M:%S"),
                "src_ip": chosen_srcs[k],
                "dst_port": chosen_ports[k],
                "label": pred_lab,
                "confidence": conf,
                "action": action,
                "latency_ms": current_latency * 1000,
                "is_attack": is_pred_attack,
                "is_correct": (is_true_attack == is_pred_attack)
            }
            # Append mode "a" is default for open, but explicit here
            with open(LOG_FILE, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        if args.sleep > 0: time.sleep(args.sleep)
        i = end

    print(f"\nFinished. Accuracy: {(tp+tn)/(tp+tn+fp+fn):.2%}")

if __name__ == "__main__":
    main()