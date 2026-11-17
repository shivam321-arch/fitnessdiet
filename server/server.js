// FitMentor AI helper — forwards requests from the FitMentor website to Claude on AWS Bedrock.
// The AWS key is read ONLY from environment variables (set in Render). Never put it in this file.
const express = require("express");

const PORT = process.env.PORT || 10000;
const REGION = (process.env.AWS_REGION || "ap-south-1").trim();
const TOKEN = String(process.env.AWS_BEARER_TOKEN_BEDROCK || "")
  .replace(/^\s*export\s+AWS_BEARER_TOKEN_BEDROCK\s*=\s*/i, "")
  .replace(/^["']|["']$/g, "")
  .trim();
const DAILY_LIMIT = Number(process.env.DAILY_LIMIT_PER_USER) || 60; // requests per visitor per day
const ALLOWED = (process.env.ALLOWED_ORIGINS || "https://fitnessdiet.onrender.com")
  .split(",").map((s) => s.trim()).filter(Boolean);

const FAST = [
  process.env.BEDROCK_MODEL_ID,
  "global.anthropic.claude-haiku-5-5",
  "global.anthropic.claude-haiku-4-5-20251001-v1:0",
  "in.anthropic.claude-haiku-4-5-20251001-v1:0",
  "global.anthropic.claude-sonnet-4-6",
];
const SMART = [
  process.env.BEDROCK_SMART_MODEL_ID,
  "global.anthropic.claude-sonnet-5-5",
  "global.anthropic.claude-sonnet-4-6",
  ...FAST,
];
const uniq = (a) => a.filter((m, i) => m && a.indexOf(m) === i);
const CANDIDATES = { quick: uniq(FAST), default: uniq(FAST), complex: uniq(SMART) };
const working = {}; // tier -> model that worked last
const MODEL_UNAVAILABLE = /end of (its )?life|legacy|retired|deprecated|not supported|invalid model|model identifier|isn't supported|is not supported|doesn't support|does not support|access to the model|don't have access|on-demand throughput|inference profile/i;

const SYSTEM = "You are FitMentor, a friendly Indian diet and gym mentor. Reply in simple English. " +
  "You are not a doctor: never diagnose, never prescribe medicine, and suggest seeing a doctor for medical problems. " +
  "Never encourage extreme dieting. When asked for JSON, reply with JSON only.";

const app = express();
app.use(express.json({ limit: "8mb" }));

app.use((req, res, next) => {
  const o = req.headers.origin;
  if (o && (ALLOWED.includes(o) || ALLOWED.includes("*"))) {
    res.setHeader("Access-Control-Allow-Origin", o);
    res.setHeader("Vary", "Origin");
    res.setHeader("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  }
  if (req.method === "OPTIONS") return res.sendStatus(204);
  next();
});

// simple per-visitor daily limit so nobody can run up the AWS bill
const usage = new Map();
let usageDay = new Date().toISOString().slice(0, 10);
function overLimit(ip) {
  const d = new Date().toISOString().slice(0, 10);
  if (d !== usageDay) { usage.clear(); usageDay = d; }
  const n = (usage.get(ip) || 0) + 1;
  usage.set(ip, n);
  return n > DAILY_LIMIT;
}

app.get("/", (req, res) => res.send("FitMentor AI server is running."));
app.get("/health", (req, res) => res.json({ ok: true, ai: !!TOKEN, region: REGION, images: true }));

function toContent(c) {
  if (typeof c === "string") return [{ text: c }];
  if (Array.isArray(c)) return c.map((p) => (typeof p === "string" ? { text: p } : p.text ? { text: String(p.text) } : null)).filter(Boolean);
  return [{ text: String(c || "") }];
}

app.post("/api/ai", async (req, res) => {
  if (!TOKEN) return res.status(503).json({ error: "AI is not set up on the server yet." });
  const ip = String(req.headers["x-forwarded-for"] || req.ip || "").split(",")[0].trim();
  if (overLimit(ip)) return res.status(429).json({ error: "Daily AI limit reached. Please try again tomorrow." });

  const b = req.body || {};
  const tier = ["quick", "default", "complex"].includes(b.tier) ? b.tier : "default";
  let messages = Array.isArray(b.messages) && b.messages.length
    ? b.messages.slice(-20).map((m) => ({ role: m.role === "assistant" ? "assistant" : "user", content: toContent(m.content) }))
    : [{ role: "user", content: toContent(b.prompt || "") }];
  while (messages.length && messages[0].role !== "user") messages.shift();
  if (!messages.length) return res.status(400).json({ error: "Empty request." });

  // images: [{ media: "image/jpeg", data: "<base64>" }] -> attach to the last user message
  const imgs = Array.isArray(b.images) ? b.images.slice(0, 3) : [];
  const last = messages[messages.length - 1];
  for (const im of imgs) {
    const fmt = String(im.media || "image/jpeg").split("/")[1].replace("jpg", "jpeg");
    if (!["jpeg", "png", "webp", "gif"].includes(fmt) || !im.data) continue;
    last.content.unshift({ image: { format: fmt, source: { bytes: im.data } } });
  }

  const maxTokens = Math.min(Number(b.maxTokens) || (tier === "complex" ? 4000 : 1500), 4000);
  const list = CANDIDATES[tier];
  const start = Math.max(0, list.indexOf(working[tier]));
  let lastErr;
  for (const model of list.slice(start)) {
    try {
      const text = await callBedrock(model, messages, maxTokens);
      working[tier] = model;
      return res.json({ text, model });
    } catch (e) {
      lastErr = e;
      if (!e.modelUnavailable) break;
      console.warn(`model ${model} unavailable: ${e.message}`);
    }
  }
  console.error("AI error:", lastErr && lastErr.message);
  res.status(502).json({ error: "AI is busy right now. Please try again." });
});

async function callBedrock(model, messages, maxTokens) {
  const url = `https://bedrock-runtime.${REGION}.amazonaws.com/model/${encodeURIComponent(model)}/converse`;
  const r = await fetch(url, {
    method: "POST",
    headers: { Authorization: `Bearer ${TOKEN}`, "Content-Type": "application/json" },
    body: JSON.stringify({ system: [{ text: SYSTEM }], messages, inferenceConfig: { maxTokens, temperature: 0.5 } }),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    const err = new Error(data.message || `HTTP ${r.status}`);
    err.modelUnavailable = [400, 403, 404].includes(r.status) && MODEL_UNAVAILABLE.test(String(data.message || ""));
    throw err;
  }
  const parts = (data.output && data.output.message && data.output.message.content) || [];
  const text = parts.map((p) => p.text || "").join("").trim();
  if (!text) throw new Error("No text returned");
  return text;
}

app.listen(PORT, () => console.log(`FitMentor AI server on ${PORT}, region ${REGION}, key ${TOKEN ? "set" : "MISSING"}`));
