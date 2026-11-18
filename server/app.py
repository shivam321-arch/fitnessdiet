# FitMentor AI server (Python)
# --------------------------------------------------------------
# Kaam: website se sawaal lo -> AWS Bedrock (Claude) ko bhejo -> jawab wapas do.
# AWS key sirf Render ke "Environment Variables" me rehti hai. Is file me KABHI mat likhna.
# --------------------------------------------------------------

import os
import re
import time
import datetime
import secrets
import hmac
import hashlib
import requests
from flask import Flask, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

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

# login settings (Render Environment Variables)
MONGODB_URI = os.environ.get("MONGODB_URI", "").strip()
# password alag variable me bhi de sakte ho (link me <db_password> rehne do)
_MPW = os.environ.get("MONGODB_PASSWORD", "").strip()
if _MPW and "<db_password>" in MONGODB_URI:
    from urllib.parse import quote_plus
    MONGODB_URI = MONGODB_URI.replace("<db_password>", quote_plus(_MPW))
SECRET_KEY = os.environ.get("SECRET_KEY", "").strip()          # lamba random text, sirf Render me
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
TOKEN_DAYS = 60

# OTP email (Brevo). Render free par Gmail SMTP band hai, isliye Brevo ki HTTP API use karte hain.
BREVO_API_KEY = os.environ.get("BREVO_API_KEY", "").strip()
MAIL_FROM = os.environ.get("MAIL_FROM", "").strip()            # Brevo me verify kiya hua sender email
EMAIL_OTP = bool(BREVO_API_KEY and MAIL_FROM)

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
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
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
    return jsonify(ok=True, ai=bool(TOKEN), region=REGION, images=True, login=bool(MONGODB_URI and SECRET_KEY))


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


# =====================================================================
# 7. LOGIN + CLOUD SAVE
#    - email/password (password hash hi save hota hai, asli password kabhi nahi)
#    - Google login (Google ka ID token verify karte hain)
#    - har user ka app data (profile, food log, weight...) MongoDB me
# =====================================================================
_users = None
def users_col():
    """MongoDB collection 'users' (pehli baar connect karta hai)."""
    global _users
    if _users is not None:
        return _users
    if not MONGODB_URI:
        return None
    try:
        from pymongo import MongoClient
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=8000)
        col = client["fitmentor"]["users"]
        col.create_index("email", unique=True, sparse=True)
        col.create_index("google_sub", unique=True, sparse=True)
        _users = col
    except Exception as e:
        print("MongoDB connect error:", e)
        return None
    return _users

def signer():
    return URLSafeTimedSerializer(SECRET_KEY, salt="fitmentor-login")

def make_token(user_id):
    return signer().dumps(str(user_id))

def current_user():
    """Authorization: Bearer <token> se user nikaalo."""
    col = users_col()
    if col is None or not SECRET_KEY:
        return None
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    try:
        uid = signer().loads(auth[7:], max_age=TOKEN_DAYS * 86400)
    except (BadSignature, SignatureExpired):
        return None
    from bson import ObjectId
    try:
        return col.find_one({"_id": ObjectId(uid)})
    except Exception:
        return None

def public_user(u):
    return {"email": u.get("email"), "name": u.get("name", ""), "google": bool(u.get("google_sub")),
            "data": u.get("data") or {}, "updated": u.get("updated", 0)}

# login attempts ki limit (password guess karne se rokne ke liye)
_attempts = {}
def too_many(key, limit=10, window=900):
    now = time.time()
    arr = [t for t in _attempts.get(key, []) if now - t < window]
    arr.append(now)
    _attempts[key] = arr
    return len(arr) > limit

def need_db():
    if users_col() is None or not SECRET_KEY:
        return jsonify(error="Login is not set up on the server yet."), 503
    return None

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

@app.get("/api/config")
def config():
    return jsonify(login=users_col() is not None and bool(SECRET_KEY), google_client_id=GOOGLE_CLIENT_ID,
                   email_otp=EMAIL_OTP)


# ---------- OTP (email par 6 number ka code) ----------
_otps = None
def otps_col():
    global _otps
    if _otps is None and users_col() is not None:
        _otps = users_col().database["otps"]
        _otps.create_index("exp", expireAfterSeconds=0)   # purane code MongoDB khud mita deta hai
    return _otps

def otp_hash(email, purpose, code):
    return hmac.new(SECRET_KEY.encode(), f"{purpose}:{email}:{code}".encode(), hashlib.sha256).hexdigest()

def send_mail(to, subject, html, text):
    r = requests.post("https://api.brevo.com/v3/smtp/email", timeout=15,
                      headers={"api-key": BREVO_API_KEY, "accept": "application/json"},
                      json={"sender": {"name": "FitMentor", "email": MAIL_FROM}, "to": [{"email": to}],
                            "subject": subject, "htmlContent": html, "textContent": text})
    if not r.ok:
        print("Brevo error:", r.status_code, r.text[:300], flush=True)
        raise RuntimeError("mail failed")

def check_otp(email, purpose, code):
    """Code sahi hai to True (aur code use ho gaya, dobara nahi chalega)."""
    col = otps_col()
    doc = col.find_one({"email": email, "purpose": purpose})
    if not doc or doc.get("tries", 0) >= 5 or doc["exp"] < datetime.datetime.utcnow():
        return False
    if not hmac.compare_digest(doc["hash"], otp_hash(email, purpose, str(code).strip())):
        col.update_one({"_id": doc["_id"]}, {"$inc": {"tries": 1}})
        return False
    col.delete_one({"_id": doc["_id"]})
    return True

@app.route("/api/auth/otp", methods=["POST", "OPTIONS"])
def send_otp():
    if request.method == "OPTIONS":
        return "", 204
    bad = need_db()
    if bad: return bad
    if not EMAIL_OTP:
        return jsonify(error="Email codes are not set up yet."), 503
    b = request.get_json(silent=True) or {}
    email = str(b.get("email", "")).strip().lower()
    purpose = "reset" if b.get("purpose") == "reset" else "signup"
    if not EMAIL_RE.match(email):
        return jsonify(error="Please enter a valid email."), 400
    ip = (request.headers.get("X-Forwarded-For") or request.remote_addr or "").split(",")[0].strip()
    if too_many("otpip:" + ip, 15, 3600) or too_many("otpmail:" + email, 5, 3600):
        return jsonify(error="Too many codes requested. Try again after some time."), 429
    u = users_col().find_one({"email": email})
    if purpose == "signup" and u and (u.get("pw") or u.get("google_sub")):
        return jsonify(error="An account with this email already exists. Please sign in."), 409
    if purpose == "reset" and not u:
        return jsonify(ok=True)   # kisi ko pata na chale ki email registered hai ya nahi
    old = otps_col().find_one({"email": email, "purpose": purpose})
    if old and time.time() - old.get("sent", 0) < 45:
        return jsonify(error="Please wait a few seconds before asking for a new code."), 429
    code = f"{secrets.randbelow(1000000):06d}"
    otps_col().update_one({"email": email, "purpose": purpose}, {"$set": {
        "hash": otp_hash(email, purpose, code), "tries": 0, "sent": time.time(),
        "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=10)}}, upsert=True)
    what = "create your FitMentor account" if purpose == "signup" else "reset your FitMentor password"
    try:
        send_mail(email, f"{code} is your FitMentor code",
                  f"<div style='font-family:Arial,sans-serif;font-size:15px'><p>Use this code to {what}:</p>"
                  f"<p style='font-size:30px;font-weight:bold;letter-spacing:6px'>{code}</p>"
                  f"<p>It works for 10 minutes. If you did not ask for it, you can ignore this email.</p></div>",
                  f"Your FitMentor code is {code}. Use it to {what}. It works for 10 minutes.")
    except Exception:
        return jsonify(error="Could not send the email. Please try again."), 502
    return jsonify(ok=True)

@app.route("/api/auth/reset", methods=["POST", "OPTIONS"])
def reset_pw():
    if request.method == "OPTIONS":
        return "", 204
    bad = need_db()
    if bad: return bad
    b = request.get_json(silent=True) or {}
    email = str(b.get("email", "")).strip().lower()
    pw = str(b.get("password", ""))
    if len(pw) < 8:
        return jsonify(error="Password must be at least 8 characters."), 400
    if not EMAIL_OTP or not check_otp(email, "reset", b.get("otp", "")):
        return jsonify(error="Code is wrong or expired. Ask for a new code."), 400
    col = users_col()
    u = col.find_one({"email": email})
    if not u:
        return jsonify(error="No account with this email."), 404
    col.update_one({"_id": u["_id"]}, {"$set": {"pw": generate_password_hash(pw), "verified": True}})
    return jsonify(token=make_token(u["_id"]), user=public_user(u))

@app.route("/api/auth/register", methods=["POST", "OPTIONS"])
def register():
    if request.method == "OPTIONS":
        return "", 204
    bad = need_db()
    if bad: return bad
    b = request.get_json(silent=True) or {}
    email = str(b.get("email", "")).strip().lower()
    pw = str(b.get("password", ""))
    name = str(b.get("name", "")).strip()[:60]
    if not EMAIL_RE.match(email):
        return jsonify(error="Please enter a valid email."), 400
    if len(pw) < 8:
        return jsonify(error="Password must be at least 8 characters."), 400
    ip = (request.headers.get("X-Forwarded-For") or request.remote_addr or "").split(",")[0].strip()
    if too_many("reg:" + ip, 8, 3600):
        return jsonify(error="Too many sign-ups from here. Try again later."), 429
    col = users_col()
    if col.find_one({"email": email}):
        return jsonify(error="An account with this email already exists. Please sign in."), 409
    if EMAIL_OTP and not check_otp(email, "signup", b.get("otp", "")):
        return jsonify(error="Code is wrong or expired. Ask for a new code."), 400
    doc = {"email": email, "name": name, "pw": generate_password_hash(pw), "created": int(time.time()),
           "verified": EMAIL_OTP, "data": {}, "updated": 0}
    uid = col.insert_one(doc).inserted_id
    return jsonify(token=make_token(uid), user=public_user(doc))

@app.route("/api/auth/login", methods=["POST", "OPTIONS"])
def login():
    if request.method == "OPTIONS":
        return "", 204
    bad = need_db()
    if bad: return bad
    b = request.get_json(silent=True) or {}
    email = str(b.get("email", "")).strip().lower()
    pw = str(b.get("password", ""))
    if too_many("login:" + email, 10, 900):
        return jsonify(error="Too many attempts. Wait 15 minutes and try again."), 429
    u = users_col().find_one({"email": email})
    if not u or not u.get("pw") or not check_password_hash(u["pw"], pw):
        return jsonify(error="Email or password is wrong."), 401
    return jsonify(token=make_token(u["_id"]), user=public_user(u))

@app.route("/api/auth/google", methods=["POST", "OPTIONS"])
def google_login():
    if request.method == "OPTIONS":
        return "", 204
    bad = need_db()
    if bad: return bad
    if not GOOGLE_CLIENT_ID:
        return jsonify(error="Google login is not set up yet."), 503
    cred = str((request.get_json(silent=True) or {}).get("credential", ""))
    try:
        r = requests.get("https://oauth2.googleapis.com/tokeninfo", params={"id_token": cred}, timeout=10)
        info = r.json() if r.ok else {}
    except Exception:
        info = {}
    if info.get("aud") != GOOGLE_CLIENT_ID or info.get("iss") not in ("accounts.google.com", "https://accounts.google.com") or str(info.get("email_verified")) != "true":
        return jsonify(error="Google sign-in failed. Please try again."), 401
    col = users_col()
    sub, email = info["sub"], info.get("email", "").lower()
    u = col.find_one({"google_sub": sub}) or (col.find_one({"email": email}) if email else None)
    if u:
        if not u.get("google_sub"):
            upd = {"$set": {"google_sub": sub, "verified": True}}
            # bina verify wala password kisi aur ne bhi bana ho sakta hai -> hata do (Google ne email confirm kiya hai)
            if u.get("pw") and not u.get("verified"):
                upd["$unset"] = {"pw": ""}
            col.update_one({"_id": u["_id"]}, upd)
    else:
        u = {"email": email, "name": info.get("given_name") or info.get("name", ""), "google_sub": sub, "verified": True, "created": int(time.time()), "data": {}, "updated": 0}
        u["_id"] = col.insert_one(dict(u)).inserted_id
    return jsonify(token=make_token(u["_id"]), user=public_user(u))

@app.route("/api/me", methods=["GET", "DELETE", "OPTIONS"])
def me():
    if request.method == "OPTIONS":
        return "", 204
    u = current_user()
    if not u:
        return jsonify(error="Please sign in again."), 401
    if request.method == "DELETE":
        users_col().delete_one({"_id": u["_id"]})
        return jsonify(ok=True)
    return jsonify(user=public_user(u))

@app.route("/api/data", methods=["PUT", "OPTIONS"])
def save_data():
    if request.method == "OPTIONS":
        return "", 204
    u = current_user()
    if not u:
        return jsonify(error="Please sign in again."), 401
    b = request.get_json(silent=True) or {}
    data = b.get("data")
    if not isinstance(data, dict):
        return jsonify(error="Bad data."), 400
    if len(request.data or b"") > 1_500_000:
        return jsonify(error="Data too large."), 413
    now = int(time.time() * 1000)
    users_col().update_one({"_id": u["_id"]}, {"$set": {"data": data, "updated": now}})
    return jsonify(ok=True, updated=now)


# local test ke liye: python app.py   (Render par gunicorn chalata hai)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
