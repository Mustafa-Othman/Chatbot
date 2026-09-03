"""Intent routing: keyword rules for small talk + the trained route
classifier + model loading helpers used by notebooks and the app.
"""
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

from .config import INTENT_DIR, INTENT_MAP, INTENT_PIPELINE, ROUTES, ROUTE_POLICY
from .textproc import clean

# ---------------------------------------------------------------------------
# Small-talk (greeting / goodbye / gratitude) rules.
# The Bitext corpus has no greeting rows, so this is a small rule set that
# runs BEFORE the classifier. Everything not matched here goes to the model.
# ---------------------------------------------------------------------------
_SMALLTALK_RE = re.compile(
    r"\b(hi|hello|hey|good\s*(morning|afternoon|evening)|howdy|greetings"
    r"|goodbye|bye\b|good\s*night|see\s*you|thanks|thank\s*you|thx|ty"
    r"|cheers|appreciate|grateful|have a good one)\b",
    re.IGNORECASE,
)

# Strong "I want a person" signals. These are decisive for routing: when a
# customer explicitly asks for a human/manager we escalate even if the rest of
# the message is about, say, a double charge.
_ESCALATION_PATTERNS = [
    re.compile(r"\btalk\s+(me\s+)?to\s+(a\s+|an\s+|the\s+)?"
               r"(human|person|manager|agent|representative|operator)\b", re.I),
    re.compile(r"\bspeak\s+(me\s+)?to\s+(a\s+|an\s+|the\s+)?"
               r"(human|person|manager|agent|representative|operator)\b", re.I),
    re.compile(r"\b(want|need|get|speak|see)\s+(to\s+)?(a\s+|an\s+|the\s+)?"
               r"(human|manager|representative|operator)\b", re.I),
    re.compile(r"\bhuman\s+agent\b", re.I),
    re.compile(r"\b(connect|transfer)\s+me\b", re.I),
    re.compile(r"\b(manager|supervisor|real\s+person)\b", re.I),
]


def is_small_talk(text: str) -> bool:
    """True if the message is essentially a greeting/thanks and has no
    substantive request keyword."""
    s = clean(text)
    return bool(_SMALLTALK_RE.search(s)) and not any(
        k in s for k in ("order", "refund", "invoice", "deliver", "account",
                         "password", "payment", "cancel", "shipping", "address",
                         "complaint", "return", "track")
    )


def is_escalation_request(text: str) -> bool:
    """True if the customer explicitly asks to reach a person / manager."""
    s = clean(text)
    return any(p.search(s) for p in _ESCALATION_PATTERNS)


# ---------------------------------------------------------------------------
# Loaders / prediction
# ---------------------------------------------------------------------------
def load_intent_model(intent_dir: Path | None = None):
    intent_dir = intent_dir or INTENT_DIR
    with open(intent_dir / INTENT_PIPELINE.name, "rb") as fh:
        pipe = pickle.load(fh)
    route_map = json.loads((intent_dir / INTENT_MAP.name).read_text(encoding="utf-8"))
    return pipe, route_map


def predict_route(text: str, pipe=None, route_map=None, intent_dir: Path | None = None):
    """Route a message.

    Returns (route, confidence). Confidence is 1.0 for rule-based small talk;
    otherwise the classifier's softmax probability of its argmax route.
    """
    s = clean(text)
    if is_small_talk(s):
        return "small_talk", 1.0
    if is_escalation_request(s):
        # decisive human-request signal -> escalation route (rule-based)
        return "contact_support", 0.99
    if pipe is None:
        pipe, route_map = load_intent_model(intent_dir)
    proba = pipe.predict_proba([s])[0]
    k = int(proba.argmax())
    route = pipe.classes_[k]
    conf = float(proba[k])
    return route, conf
