"""End-to-end chat pipeline.

Order of processing (mirrors the task brief):

    message
      -> language detection   (which language / script)
      -> sentiment (emotion)  (frustrated / neutral / satisfied)
      -> intent routing       (small talk | 7 trained routes)
      -> response policy:
            * small talk        -> canned reply, NO RAG
            * complaint / human -> apologetic + escalate, NO auto-answer
            * out-of-scope      -> honest "can't help" + escalation offer
            * everything else   -> RAG (retrieve KB + grounded LLM answer)

All model loading is lazy + cached so the Flask app starts fast even when
the user has only trained some modules.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

import numpy as np

from . import config as C
from .config import LANG_PIPELINE, ROUTE_POLICY
from .intent import predict_route
from .llm import GroqClient, build_generation_prompt
from .sentiment import load_sentiment, predict_emotion, tone_of_emotion
from .textproc import clean

# minimum cosine similarity to consider a retrieved chunk "grounding"
KB_SIM_MIN = 0.45

_PLACEHOLDER_RE = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}")


def deplaceholder(text: str) -> str:
    """Turn Bitext's  {{Customer Support Phone Number}}  style slots into
    readable, still-honest text for retrieval-only fallback answers."""
    return _PLACEHOLDER_RE.sub(lambda m: m.group(1).strip().lower().replace("_", " "), text)


# Small set of canned small-talk replies (English + a few languages).
_SMALLTALK = {
    "en": ["Hi! I'm the customer support assistant. How can I help you today?",
           "You're welcome! Is there anything else I can do for you?",
           "Goodbye! Thanks for contacting us. Have a great day!"],
    "es": ["¡Hola! Soy el asistente de atención al cliente. ¿En qué puedo ayudarte hoy?"],
    "fr": ["Bonjour ! Je suis l'assistant du service client. Comment puis-je vous aider ?"],
    "de": ["Hallo! Ich bin der Kundenservice-Assistent. Wie kann ich Ihnen heute helfen?"],
    "pt": ["Olá! Sou o assistente de atendimento ao cliente. Como posso ajudar hoje?"],
    "it": ["Ciao! Sono l'assistente del servizio clienti. Come posso aiutarti oggi?"],
    "nl": ["Hallo! Ik ben de klantenservice-assistent. Hoe kan ik u vandaag helpen?"],
    "zh": ["你好！我是客户支持助理，今天有什么可以帮您的吗？"],
    "ar": ["مرحباً! أنا مساعد دعم العملاء. كيف يمكنني مساعدتك اليوم؟"],
    "hi": ["नमस्ते! मैं ग्राहक सहायता सहायक हूँ। आज मैं आपकी कैसे मदद कर सकता हूँ?"],
    "ja": ["こんにちは！カスタマーサポートです。本日はどのようなご用件でしょうか？"],
}


class SupportBot:
    def __init__(self):
        self._lang_pipe = None
        self._sent_model = None
        self._sent_meta = None
        self._intent_pipe = None
        self._route_map = None
        self._kb_index = None
        self._kb_meta = None
        self._embedder = None
        self._llm = None

    # ------------------------------------------------------------------ loaders
    def status(self) -> dict:
        return {
            "language": LANG_PIPELINE.exists(),
            "sentiment": (C.SENT_DIR / "model.pt").exists(),
            "intent": (C.INTENT_DIR / "intent_pipeline.pkl").exists(),
            "kb": C.KB_INDEX.exists(),
            "llm_available": bool(os.environ.get("GROQ_API_KEY", "")),
        }

    def load_language(self):
        import pickle
        if self._lang_pipe is None:
            with open(LANG_PIPELINE, "rb") as fh:
                self._lang_pipe = pickle.load(fh)
        return self._lang_pipe

    def load_sentiment(self):
        if self._sent_model is None:
            self._sent_model, self._sent_meta = load_sentiment()
        return self._sent_model, self._sent_meta

    def load_intent(self):
        if self._intent_pipe is None:
            from .intent import load_intent_model
            self._intent_pipe, self._route_map = load_intent_model()
        return self._intent_pipe, self._route_map

    def load_kb(self):
        if self._kb_index is None:
            from .retriever import load_index
            self._kb_index, self._kb_meta = load_index()
        return self._kb_index, self._kb_meta

    def embedder(self):
        if self._embedder is None:
            from .retriever import _get_embedder
            self._embedder = _get_embedder()
        return self._embedder

    def llm(self) -> GroqClient:
        if self._llm is None:
            self._llm = GroqClient()
        return self._llm

    def _detect_language(self, text: str) -> dict:
        pipe = self.load_language()
        s = clean(text)
        proba = pipe.predict_proba([s])[0]
        k = int(proba.argmax())
        iso = str(pipe.classes_[k])
        return {"language": iso, "lang_conf": float(proba[k]),
                "lang_name": C.LANG_NAMES.get(iso, iso)}

    def _detect_sentiment(self, text: str) -> dict:
        model, meta = self.load_sentiment()
        emo, conf = predict_emotion(model, meta, [clean(text)])
        return {"emotion": emo[0], "emotion_conf": conf[0],
                "tone": tone_of_emotion(emo[0], meta["tone_map"])}

    def _retrieve(self, text: str, top_k: int = 3) -> list[dict]:
        from .retriever import search
        index, metas = self.load_kb()
        qvec = self.embedder().encode([clean(text)], normalize_embeddings=True,
                                      convert_to_numpy=True)
        hits = search(index, metas, qvec, top_k=top_k)
        return [{"score": sc, **m} for sc, m in hits]

    # ---------------------------------------------------------------- analysis
    def analyze(self, message: str) -> dict:
        """Stage 1-3: language, sentiment, route. Pure analysis, no generation."""
        out = {"message": message}
        out.update(self._detect_language(message))
        out.update(self._detect_sentiment(message))
        route, conf = predict_route(message, self._intent_pipe, self._route_map)
        out["route"] = route
        out["route_conf"] = conf
        out["policy"] = ROUTE_POLICY.get(route, "rag")
        return out

    # ---------------------------------------------------------------- answers
    def answer(self, message: str, top_k: int = 3, use_llm: bool = True) -> dict:
        t0 = time.time()
        message = clean(message)
        if not message:
            return {"reply": "Please write a message so I can help you.",
                    "error": "empty_message", "latency": 0.0}

        analysis = self.analyze(message)
        route, policy = analysis["route"], analysis["policy"]
        negative = analysis["tone"] == "negative"
        result = {"reply": "", "route": route, "policy": policy,
                  "escalate": False, "priority": negative,
                  "sources": [], "used_llm": False,
                  "note": "", "latency": 0.0, **analysis}

        # ---- small talk: no RAG ----------------------------------------
        if route == "small_talk":
            result["reply"] = _SMALLTALK.get(
                analysis["language"], _SMALLTALK["en"])[0]
            result["note"] = "handled without retrieval (small talk)"
            result["latency"] = round(time.time() - t0, 3)
            return result

        # ---- complaint / ask-for-human: apologise + escalate ------------
        if policy == "escalate":
            apology = ("I'm very sorry for the frustration this has caused. "
                       if analysis["tone"] == "negative" else "")
            if route == "contact_support":
                reply = apology + ("Of course — I've flagged your request and a "
                                   "human agent will join you shortly. While you "
                                   "wait, is there anything about your order I "
                                   "can look up for you?")
            else:
                reply = apology + ("I've escalated your case to our support team "
                                   "for priority handling — a human agent will "
                                   "respond to you as soon as possible.")
            result["reply"] = reply
            result["escalate"] = True
            result["priority"] = True
            result["note"] = "priority path (complaint / human request): no auto-generated policy answer"
            result["latency"] = round(time.time() - t0, 3)
            return result

        # ---- RAG path ---------------------------------------------------
        try:
            hits = self._retrieve(message, top_k=top_k)
        except FileNotFoundError:
            result["reply"] = ("I can't answer that yet: the knowledge base "
                               "index has not been built. Please run notebook "
                               "04 (RAG) first.")
            result["note"] = "kb index missing"
            result["latency"] = round(time.time() - t0, 3)
            return result

        result["sources"] = hits
        if not hits or hits[0]["score"] < KB_SIM_MIN:
            empathy = ("I'm sorry this is frustrating. " if negative else "")
            result["reply"] = (empathy + "I couldn't find an answer for that "
                               "in our knowledge base. Would you like me to "
                               "connect you with a human agent?")
            result["note"] = "no confident retrieval -> honest out-of-scope reply"
            result["latency"] = round(time.time() - t0, 3)
            return result

        # grounded answer -------------------------------------------------
        system, user_prompt = build_generation_prompt(
            message, hits[:top_k], analysis["language"],
            analysis["tone"], route, escalate=False)

        if use_llm and self.llm().available:
            gen = self.llm().generate(system, user_prompt)
            if gen["ok"]:
                reply = gen["text"]
                if analysis["tone"] == "negative":
                    reply = reply if reply.lower().startswith(("i'm sorry", "i am sorry", "sorry", "i understand")) \
                        else ("I'm sorry to hear that. " + reply)
                result["reply"] = reply
                result["used_llm"] = True
                result["note"] = f"grounded answer via {gen['model']}"
                result["latency"] = round(time.time() - t0, 3)
                return result

        # retrieval-only fallback (no key / LLM error) -------------------
        best = hits[0]
        fallback = deplaceholder(best["response"])
        if analysis["tone"] == "negative":
            fallback = ("I'm sorry this is frustrating — let me help. " + fallback)
        result["reply"] = fallback
        result["note"] = "retrieval-only reply (no LLM available)"
        result["latency"] = round(time.time() - t0, 3)
        return result


_bot: SupportBot | None = None


def get_bot() -> SupportBot:
    global _bot
    if _bot is None:
        _bot = SupportBot()
    return _bot
