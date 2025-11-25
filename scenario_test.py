import requests
import json

URL = "http://127.0.0.1:8000/classify"

print("--- 🧪 SCENARIO TEST: Context Awareness ---")

# 1. Define a "low-and-slow" Brute Force log
# A single log looks like a normal user login failure (or attempt).
single_log = "At 10:00:01, src 192.168.1.105 -> dst 10.0.0.5, flags S, ack 0, size 0, dport 22."

# 2. Define the "Aggregated Window"
# Repeating it simulates a burst of attempts in a short time.
# In your training, you likely taught the model that repetitions = Brute Force / DDoS.
window_log = "\n".join([single_log] * 10)

def test_payload(name, text):
    print(f"\n[?] Testing: {name}")
    try:
        res = requests.post(URL, json={"text": text}).json()
        print(f"    -> Label: {res['label']}")
        print(f"    -> Conf : {res['confidence']:.4f}")
        return res['label']
    except Exception as e:
        print(f"    Error: {e}")
        return "ERROR"

# Run Tests
l1 = test_payload("Single Packet (No Context)", single_log)
l2 = test_payload("Aggregated Window (10 Packets)", window_log)

print("\n" + "="*40)
if l1 == "Normal" and l2 != "Normal":
    print("✅ SUCCESS: The model utilized context to detect the attack!")
elif l1 == l2:
    print("⚠️  NOTE: Both classified as " + l1 + ". (Model might be too sensitive or insensitive).")
else:
    print("✅ Detection changed based on context.")