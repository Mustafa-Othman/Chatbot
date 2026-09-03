"""Flask API for the RAG e-commerce customer-support chatbot.

Endpoints
---------
GET  /health      -> which modules are loaded / missing
POST /chat        -> {"message": "..."} -> full pipeline response
GET  /            -> tiny HTML demo page

Run (from the project root):
    python deploy/app.py            # default http://127.0.0.1:8000
Set GROQ_API_KEY first to enable generated (LLM) answers.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask import Flask, jsonify, request
from flask_cors import CORS

from support_bot import config as C
from support_bot.pipeline import get_bot

app = Flask(__name__)
CORS(app)

_INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Support Chatbot</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:760px;margin:2rem auto;padding:0 1rem}
 textarea{width:100%;height:80px;font-size:1rem}
 .row{display:flex;gap:8px;margin-top:8px}
 button{padding:8px 16px}
 .meta{color:#555;font-size:.85rem;white-space:pre-wrap}
 .reply{background:#f2f6fc;border:1px solid #cfdbe8;border-radius:8px;padding:12px;white-space:pre-wrap;margin-top:8px}
</style></head><body>
<h2>🛒 Customer Support Chatbot</h2>
<p>Language → sentiment → intent → RAG. LLM answers if <code>GROQ_API_KEY</code> is set.</p>
<textarea id="msg" placeholder="e.g. where is my order?"></textarea>
<div class="row"><button onclick="ask()">Send</button></div>
<div class="meta" id="meta"></div>
<div class="reply" id="reply"></div>
<script>
async function ask(){
  const r = await fetch('/chat',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({message:document.getElementById('msg').value})});
  const j = await r.json();
  document.getElementById('reply').innerText = j.reply || j.error;
  const m = [j.language&&'lang='+j.language, j.tone&&'tone='+j.tone,
             j.route&&'route='+j.route, j.policy&&'policy='+j.policy,
             j.escalate&&'⚠ escalated', j.used_llm&&'llm'].filter(Boolean).join(' | ');
  document.getElementById('meta').innerText = m;
}
</script></body></html>"""


def _missing_modules() -> list[str]:
    status = get_bot().status()
    missing = [k for k, ok in status.items() if k in ("language", "sentiment", "intent", "kb") and not ok]
    return missing


@app.get("/")
def index():
    return _INDEX_HTML


@app.get("/health")
def health():
    bot = get_bot()
    status = bot.status()
    ok = all(status.get(k) for k in ("language", "sentiment", "intent", "kb"))
    return jsonify({
        "status": "ok" if ok else "incomplete",
        "modules": status,
        "kb_chunks": _kb_count(),
        "hint": ("run the four notebooks 01..04 first" if not ok else "ready"),
    })


def _kb_count():
    try:
        return int((C.KB_DIR / "count.txt").read_text())
    except Exception:
        return 0


@app.post("/chat")
def chat():
    payload = request.get_json(force=True, silent=True) or {}
    message = (payload.get("message") or "").strip()
    if not message:
        return jsonify({"error": "empty message", "reply": "Please write a message."}), 400

    missing = _missing_modules()
    if missing:
        return jsonify({
            "error": f"missing trained modules: {missing}",
            "reply": ("I'm not ready yet — please run the notebooks 01..04 first "
                      "(see README) to train the modules."),
            "modules_missing": missing,
        }), 503

    try:
        out = get_bot().answer(message)
        return jsonify(out)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"{type(e).__name__}: {e}",
                        "reply": "Something went wrong on my side."}), 500


if __name__ == "__main__":
    print("Support chatbot API  ->  http://127.0.0.1:8000")
    print("Module status:", get_bot().status())
    app.run(host="127.0.0.1", port=8000, debug=False)
