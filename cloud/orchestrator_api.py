from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os

# Get the Edge Agent URL from an environment variable, default to localhost
EDGE_AGENT_URL = os.getenv("EDGE_URL", "http://127.0.0.1:7000")

class Plan(BaseModel):
    action: str
    src: str | None = None
    port: int | None = None
    source_label: str | None = None

app = FastAPI(title="Orchestrator")

@app.post("/deploy")
def deploy(p: Plan):
    """
    Receives a deployment plan from the Policy Engine
    and translates it into a specific rule for the Edge Agent.
    """
    rule = "NOOP" # Default rule, overwritten if a valid action is found
    ttl = os.getenv("RULE_TTL_MIN", "30")

    # --- FIXED: Logic re-ordered to build the rule *before* sending it ---

    # 1. Map the abstract policy action to a concrete rule string
    if p.action == "DEPLOY:BLOCK_IP":
        if p.src and p.src != "Unknown":
            rule = f"BLOCK_IP SRC {p.src}"
    elif p.action == "DEPLOY:RATE_LIMIT_IP":
        if p.src and p.src != "Unknown":
            rule = f"RATE_LIMIT_IP SRC {p.src} 100/min"
    elif p.action == "DEPLOY:QUARANTINE_IP":
        if p.src and p.src != "Unknown":
            rule = f"QUARANTINE_IP SRC {p.src}"
    elif p.action == "DEPLOY:WAF_RULE_SQLI":
        if p.port:
            rule = f"ADD_WAF_RULE type=SQLI target_port={p.port}"
    elif p.action == "DEPLOY:WAF_RULE_XSS":
        if p.port:
            rule = f"ADD_WAF_RULE type=XSS target_port={p.port}"
    elif p.action == "DEPLOY:WAF_RULE_UPLOAD":
        if p.port:
            rule = f"ADD_WAF_RULE type=FILE_UPLOAD target_port={p.port}"
    # Kept for compatibility
    elif p.action == "DEPLOY:ACL_BLOCK_SRC" and p.src:
        rule = f"BLOCK SRC {p.src}"
    elif p.action == "DEPLOY:BLOCK_PORT" and p.port:
        rule = f"BLOCK DSTPORT {p.port}"

    # 2. If no valid rule was created, skip sending.
    if rule == "NOOP":
        return {"sent_rule": "NOOP", "edge_status": {"skipped": "no valid rule for action or context"}}

    # 3. Send the final, concrete rule to the Edge Agent
    try:
        r = requests.post(
            f"{EDGE_AGENT_URL}/apply_rule",
            json={"rule": rule, "ttl_min": ttl},
            timeout=5
        )
        r.raise_for_status()
        edge_status = r.json()
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Could not contact Edge Agent at {EDGE_AGENT_URL}. {e}")
        edge_status = {"error": str(e)}

    return {"sent_rule": rule, "edge_status": edge_status}