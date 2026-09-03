"""Quick smoke test of the deployed /chat endpoint.

Usage:
    python deploy/test_api.py                # assumes server on 127.0.0.1:8000
Optionally:  python deploy/test_api.py 12345   (custom port)
"""
from __future__ import annotations

import json
import sys
import urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8000"
BASE = f"http://127.0.0.1:{PORT}"

MESSAGES = [
    "hi there!",
    "Where is my order? It should have arrived yesterday.",
    "I want to cancel my order, the button is broken and I'm angry!",
    "Can I get a refund for a faulty laptop?",
    "Quiero saber el estado de mi pedido",          # Spanish
]


def post(msg: str):
    req = urllib.request.Request(
        BASE + "/chat",
        data=json.dumps({"message": msg}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


if __name__ == "__main__":
    with urllib.request.urlopen(BASE + "/health", timeout=30) as r:
        print("HEALTH:", json.dumps(json.loads(r.read().decode()), indent=2))

    for m in MESSAGES:
        print("\n" + "=" * 80)
        print("USER:", m)
        try:
            out = post(m)
            if "error" in out and out.get("error") and not out.get("reply"):
                print("ERROR:", out["error"])
                continue
            print(f"lang={out.get('language')} tone={out.get('tone')} "
                  f"route={out.get('route')} policy={out.get('policy')} "
                  f"escalate={out.get('escalate')} llm={out.get('used_llm')}")
            print("BOT:", out.get("reply", "")[:400])
        except Exception as e:  # noqa: BLE001
            print("REQUEST FAILED:", e)
