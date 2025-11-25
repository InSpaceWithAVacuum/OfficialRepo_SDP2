from fastapi import FastAPI
from pydantic import BaseModel

# This policy maps all 14 attack types from the report
# to a specific, logical "Action Recommendation".
policy = {
    # Benign Traffic
    "Normal": "LOG_ONLY",

    # DOS/DDoS Category
    "DDoS_TCP": "DEPLOY:BLOCK_IP", "DDoS_UDP": "DEPLOY:BLOCK_IP",
    "DDoS_ICMP": "DEPLOY:BLOCK_IP", "DDoS_HTTP": "DEPLOY:RATE_LIMIT_IP",

    # Information Gathering Category
    "Port_Scanning": "DEPLOY:BLOCK_IP", "Vulnerability_scanner": "DEPLOY:BLOCK_IP",
    "Fingerprinting": "DEPLOY:BLOCK_IP",

    # Injection Attack Category
    "SQL_injection": "DEPLOY:WAF_RULE_SQLI", "XSS": "DEPLOY:WAF_RULE_XSS",
    "Uploading": "DEPLOY:WAF_RULE_UPLOAD",

    # MITM / Malware Category (High Severity)
    "MITM": "DEPLOY:QUARANTINE_IP", "Ransomware": "DEPLOY:QUARANTINE_IP",
    "Backdoor": "DEPLOY:QUARANTINE_IP", "Password": "DEPLOY:BLOCK_IP",
}

# --- FIXED: Per-label thresholds to reduce false alarms ---
# Tune these based on validation set performance.
# Higher threshold = more evidence needed to trigger an action.
THRESHOLDS = {
    # Default for any unlisted attack type
    "DEFAULT": 0.95,

    # Labels prone to false positives need higher confidence
    "Port_Scanning": 0.98,
    "Fingerprinting": 0.97,
    "DDoS_HTTP": 0.96,

    # Critical labels can have a slightly lower threshold to ensure detection
    "Ransomware": 0.92,
    "MITM": 0.92,
    "SQL_injection": 0.94,
}

class Verdict(BaseModel):
    label: str
    confidence: float

app = FastAPI(title="Policy Engine")

@app.post("/decide")
def decide(v: Verdict):
    # Rule 1: Normal traffic is always logged and never triggers action.
    if v.label == "Normal":
        return {"action": "LOG_ONLY", "severity": "LOW", "source_label": v.label}

    # Rule 2: Check if the confidence for the predicted attack label meets its specific threshold.
    trigger_threshold = THRESHOLDS.get(v.label, THRESHOLDS["DEFAULT"])
    if v.confidence < trigger_threshold:
        # If confidence is too low, downgrade to LOG_ONLY to avoid false alarms.
        return {"action": "LOG_ONLY", "severity": "LOW", "source_label": v.label}

    # Rule 3: If confidence is high enough, determine action and severity.
    action = policy.get(v.label, "LOG_ONLY")
    severity = "HIGH" if v.confidence < 0.97 else "CRITICAL"

    return {"action": action, "severity": severity, "source_label": v.label}