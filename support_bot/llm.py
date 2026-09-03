"""LLM generation wrapper (Groq) with a graceful retrieval-only fallback.

Design note: the LLM is only one *optional* stage. If GROQ_API_KEY is not
set (or the request fails) the pipeline still returns the best retrieved
support response, plus an honest note -- so the demo never breaks, and the
moment a key is added the answers become fully generated.
"""
from __future__ import annotations

import os

from .config import (GROQ_API_KEY, GROQ_FALLBACK_MODELS, GROQ_MAX_TOKENS,
                     GROQ_MODEL)

try:  # groq is an optional dependency for the notebooks/demo
    import groq
except Exception:  # pragma: no cover
    groq = None


class GroqClient:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or GROQ_API_KEY
        self.model = model or GROQ_MODEL
        self._client = None
        if self.api_key and groq is not None:
            self._client = groq.Groq(api_key=self.api_key)

    @property
    def available(self) -> bool:
        return self._client is not None

    def generate(self, system: str, user: str,
                 temperature: float = 0.3,
                 max_tokens: int | None = None) -> dict:
        """Call the model, trying fallback model ids if the primary id is
        unknown on this Groq account. Returns {"ok": bool, "text": ...,
        "error": ...}."""
        if self._client is None:
            return {"ok": False, "text": "",
                    "error": "GROQ_API_KEY not set"}
        if groq is None:
            return {"ok": False, "text": "", "error": "groq package missing"}
        max_tokens = max_tokens or GROQ_MAX_TOKENS
        candidates = [self.model] + [m for m in GROQ_FALLBACK_MODELS if m != self.model]
        last_err = ""
        for m in candidates:
            try:
                resp = self._client.chat.completions.create(
                    model=m,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return {"ok": True, "text": resp.choices[0].message.content.strip(),
                        "model": m, "error": ""}
            except Exception as e:  # noqa: BLE001
                last_err = f"{type(e).__name__}: {e}"
                # 404 / model-not-found -> try next id; auth/rate errors -> stop
                if "model" not in last_err.lower() or "not found" not in last_err.lower():
                    break
        return {"ok": False, "text": "", "error": last_err}


def build_generation_prompt(user_message: str, retrieved: list[dict],
                            detected_language: str,
                            detected_tone: str,
                            route: str,
                            escalate: bool = False) -> tuple[str, str]:
    """Build (system, user) prompt from the template in the task brief.

    retrieved: list of {"instruction", "response", "intent", "category", "score"}
    """
    system = (
        "You are a helpful, professional customer support assistant for an "
        "online retailer. Answer the customer's question using ONLY the "
        "information in the retrieved support responses below. "
    )
    if detected_tone == "negative":
        system += ("The customer sounds frustrated; acknowledge that "
                   "sincerely before answering. ")
    if escalate:
        system += ("The customer asked to speak to a person or filed a "
                   "complaint: be extra apologetic and clearly offer to "
                   "escalate to a human agent.")
    system += (
        "If the retrieved context does not cover the question, say so "
        "honestly and offer to escalate to a human agent rather than guessing. "
        "Do not invent policies, fees, links or phone numbers that are not in "
        "the context. "
        f"If the customer's language is '{detected_language}' (not English), "
        "you may translate your answer into that language, but keep the "
        "factual content identical to the English context."
    )

    parts = ["Context (retrieved past support responses):"]
    for i, c in enumerate(retrieved, 1):
        parts.append(f"[{i}] (intent={c.get('intent')}, category={c.get('category')})\n"
                     f"Q: {c.get('instruction', '')}\n"
                     f"A: {c.get('response', '')}")
    parts.append(f"Customer question: \"{user_message}\"")
    user = "\n\n".join(parts)
    return system, user
