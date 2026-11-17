# FitMentor AI server (Python)
# --------------------------------------------------------------
# Kaam: website se sawaal lo -> AWS Bedrock (Claude) ko bhejo -> jawab wapas do.
# AWS key sirf Render ke "Environment Variables" me rehti hai. Is file me KABHI mat likhna.
# --------------------------------------------------------------

import os
import re
import datetime
import requests
from flask import Flask, request, jsonify

# ---------- 1. Settings (Render ke Environment Variables se aate hain) ----------
REGION = os.environ.get("AWS_REGION", "ap-south-1").strip()

# key ko saaf karo (agar galti se "export ..." ya quotes ke saath paste ho gayi ho)
TOKEN = os.environ.get("AWS_BEARER_TOKEN_BEDROCK", "")
TOKEN = re.sub(r"^\s*export\s+AWS_BEARER_TOKEN_BEDROCK\s*=\s*", "", TOKEN, flags=re.I)
TOKEN = TOKEN.strip().strip("\"'").strip()

# ek visitor din me kitni baar AI use kar sakta hai (bill control)
DAILY_LIMIT = int(os.environ.get("DAILY_LIMIT_PER_USER", "60"))

# sirf in websites se request allowed hai
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get("ALLOWED_ORIGINS", "https://fitnessdiet.onrender.com").split(",")
    if o.strip()
]

# ---------- 2. Kaunse Claude models try karne hain (pehla na chale to agla) ----------
def unique(items):
    out = []
    for m in items:
        if m and m not in out:
            out.append(m)
    return out

FAST_MODELS = unique([
    os.environ.get("BEDROCK_MODEL_ID"),
    "global.anthropic.claude-haiku-5-5",
    "global.anthropic.claude-haiku-4-5-20251001-v1:0",
    "in.anthropic.claude-haiku-4-5-20251001-v1:0",
    "global.anthropic.claude-sonnet-4-6",
])
SMART_MODELS = unique([
    os.environ.get("BEDROCK_SMART_MODEL_ID"),
    "global.anthropic.claude-sonnet-5-5",
    "global.anthropic.claude-sonnet-4-6",
] + FAST_MODELS)

MODELS = {"quick": FAST_MODELS, "default": FAST_MODELS, "complex": SMART_MODELS}
last_working = {}  # tier -> model jo pichhli baar chala

# ye error aaye to matlab "ye model yahan nahi chalta", agla try karo
MODEL_UNAVAILABLE = re.compile(
    r"end of (its )?life|legacy|retired|deprecated|not supported|invalid model|"
    r"model identifier|isn't supported|doesn't support|does not support|"
    r"access to the model|don't have access|on-demand throughput|inference profile",
    re.I,
)

SYSTEM_PROMPT = (
    "You are FitMentor, a friendly Indian diet and gym mentor. Reply in simple English. "
    "You are not a doctor: never diagnose, never prescribe medicine, and suggest seeing "
    "a doctor for medical problems. Never encourage extreme dieting. "
    "When asked for JSON, reply with JSON only."
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB (photo ke liye)


# ---------- 3. CORS: browser ko batana ki hamari website allowed hai ----------
@app.after_request
def add_cors(resp):
    origin = request.headers.get("Origin")
    if origin and (origin in ALLOWED_ORIGINS or "*" in ALLOWED_ORIGINS):
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Vary"] = "Origin"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


# ---------- 4. Daily limit (har visitor ka counter, roz reset) ----------
usage = {}
usage_day = datetime.date.today()

def over_limit(ip):
    global usage_day
    today = datetime.date.today()
    if today != usage_day:
        usage.clear()
        usage_day = today
    usage[ip] = usage.get(ip, 0) + 1
    return usage[ip] > DAILY_LIMIT


# ---------- 5. Simple pages ----------
@app.get("/")
def home():
    return "FitMentor AI server is running."

@app.get("/health")
def health():
    return jsonify(ok=True, ai=bool(TOKEN), region=REGION, images=True)


# ---------- 6. Main AI route ----------
def to_content(c):
    """Text ko Bedrock ke format [{'text': ...}] me badlo."""
    if isinstance(c, str):
        return [{"text": c}]
    if isinstance(c, list):
        out = []
        for p in c:
            if isinstance(p, str):
                out.append({"text": p})
            elif isinstance(p, dict) and p.get("text"):
                out.append({"text": str(p["text"])})
        return out
    return [{"text": str(c or "")}]


@app.route("/api/ai", methods=["POST", "OPTIONS"])
def ai():
    if request.method == "OPTIONS":
        return "", 204
    if not TOKEN:
        return jsonify(error="AI is not set up on the server yet."), 503

    ip = (request.headers.get("X-Forwarded-For") or request.remote_addr or "").split(",")[0].strip()
    if over_limit(ip):
        return jsonify(error="Daily AI limit reached. Please try again tomorrow."), 429

    body = request.get_json(silent=True) or {}
    tier = body.get("tier") if body.get("tier") in MODELS else "default"

    # chat history ya ek prompt
    if isinstance(body.get("messages"), list) and body["messages"]:
        messages = [
            {"role": "assistant" if m.get("role") == "assistant" else "user",
             "content": to_content(m.get("content"))}
            for m in body["messages"][-20:]
        ]
    else:
        messages = [{"role": "user", "content": to_content(body.get("prompt", ""))}]
    while messages and messages[0]["role"] != "user":
        messages.pop(0)
    if not messages:
        return jsonify(error="Empty request."), 400

    # photo (base64) ko aakhri message me jodo
    for im in (body.get("images") or [])[:3]:
        fmt = str(im.get("media", "image/jpeg")).split("/")[-1].replace("jpg", "jpeg")
        if fmt in ("jpeg", "png", "webp", "gif") and im.get("data"):
            messages[-1]["content"].insert(0, {"image": {"format": fmt, "source": {"bytes": im["data"]}}})

    default_max = 4000 if tier == "complex" else 1500
    max_tokens = min(int(body.get("maxTokens") or default_max), 4000)

    models = MODELS[tier]
    start = models.index(last_working[tier]) if last_working.get(tier) in models else 0
    last_error = None
    for model in models[start:]:
        try:
            text = call_bedrock(model, messages, max_tokens)
            last_working[tier] = model
            return jsonify(text=text, model=model)
        except ModelUnavailable as e:
            print(f"model {model} unavailable: {e}")
            last_error = e
            continue
        except Exception as e:
            last_error = e
            break
    print("AI error:", last_error)
    return jsonify(error="AI is busy right now. Please try again."), 502


class ModelUnavailable(Exception):
    pass


def call_bedrock(model, messages, max_tokens):
    """AWS Bedrock Converse API ko call karo aur jawab ka text lo."""
    url = f"https://bedrock-runtime.{REGION}.amazonaws.com/model/{requests.utils.quote(model, safe='')}/converse"
    r = requests.post(
        url,
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
        json={
            "system": [{"text": SYSTEM_PROMPT}],
            "messages": messages,
            "inferenceConfig": {"maxTokens": max_tokens, "temperature": 0.5},
        },
        timeout=90,
    )
    try:
        data = r.json()
    except ValueError:
        data = {}
    if not r.ok:
        msg = str(data.get("message") or f"HTTP {r.status_code}")
        if r.status_code in (400, 403, 404) and MODEL_UNAVAILABLE.search(msg):
            raise ModelUnavailable(msg)
        raise RuntimeError(msg)
    parts = (((data.get("output") or {}).get("message") or {}).get("content")) or []
    text = "".join(p.get("text", "") for p in parts).strip()
    if not text:
        raise RuntimeError("No text returned")
    return text


# local test ke liye: python app.py   (Render par gunicorn chalata hai)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
