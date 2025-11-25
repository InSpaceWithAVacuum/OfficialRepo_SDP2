from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime, timedelta

RULES = Path("edge/rules/active.rules"); RULES.parent.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="Edge Agent")
class Req(BaseModel):
    rule: str
    ttl_min: int | str = 30

@app.post("/apply_rule")
def apply_rule(req: Req):
    exp = datetime.utcnow() + timedelta(minutes=int(req.ttl_min))
    line = f"{req.rule}  # expires={exp.isoformat()}Z\n"
    RULES.write_text((RULES.read_text() if RULES.exists() else "") + line)
    return {"applied": req.rule, "expires_utc": exp.isoformat()+"Z", "file": str(RULES)}
