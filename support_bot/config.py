"""Shared configuration & label maps for the customer-support chatbot.

Single source of truth so that the *training* notebooks and the *serving*
app can never drift apart on artifact paths, dataset ids, label maps or
model ids.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
# support_bot/config.py  ->  parents[0]=<root>/support_bot  parents[1]=<root>
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
KB_DIR = ROOT / "kb_index"

SEED = 42

# ---------------------------------------------------------------------------
# Dataset identifiers (HuggingFace) & local caches
# ---------------------------------------------------------------------------
DATASET_LANG = "papluca/language-identification"     # ~90k, 20 languages
DATASET_EMOTION = "dair-ai/emotion"                  # ~20k, 6 emotions
BICTEXT_CSV = DATA_DIR / "bitext_raw.csv"            # cached copy of the Bitext KB
BICTEXT_HF = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"

# ---------------------------------------------------------------------------
# Artifact paths (what the notebooks produce, what the app consumes)
# ---------------------------------------------------------------------------
LANG_DIR = MODELS_DIR / "language"
LANG_PIPELINE = LANG_DIR / "language_pipeline.pkl"

SENT_DIR = MODELS_DIR / "sentiment"
SENT_MODEL = SENT_DIR / "model.pt"        # full torch model (pickle)
SENT_META = SENT_DIR / "meta.json"        # vocab, labels, tone map, config

INTENT_DIR = MODELS_DIR / "intent"
INTENT_PIPELINE = INTENT_DIR / "intent_pipeline.pkl"
INTENT_MAP = INTENT_DIR / "route_map.json"

KB_INDEX = KB_DIR / "faiss.index"
KB_META = KB_DIR / "meta.pkl"
KB_N = KB_DIR / "count.txt"

EMBEDDER_ID = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Language classifier labels (ISO 639-1) + friendly names
# ---------------------------------------------------------------------------
LANG_NAMES = {
    "ar": "Arabic", "bg": "Bulgarian", "de": "German", "el": "Greek",
    "en": "English", "es": "Spanish", "fr": "French", "hi": "Hindi",
    "it": "Italian", "ja": "Japanese", "nl": "Dutch", "pl": "Polish",
    "pt": "Portuguese", "ru": "Russian", "sw": "Swahili", "th": "Thai",
    "tr": "Turkish", "ur": "Urdu", "vi": "Vietnamese", "zh": "Chinese",
}

# ---------------------------------------------------------------------------
# Emotion dataset label ids -> names  (dair-ai/emotion ClassLabel order)
# ---------------------------------------------------------------------------
EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]

# Collapse the 6 fine emotions onto the 3 tone buckets used for routing.
# negative   = frustration / urgency (sadness, anger, fear)
# neutral    = surprise is genuinely ambiguous -> treated as neutral in support
# positive   = satisfaction (joy, love)
EMOTION_TO_TONE = {
    "sadness": "negative",
    "anger": "negative",
    "fear": "negative",
    "joy": "positive",
    "love": "positive",
    "surprise": "neutral",
}
TONES = ["negative", "neutral", "positive"]

# ---------------------------------------------------------------------------
# Bitext fine-grained intent (27) -> routing route.
#
# The task brief suggests condensing the 27 dataset intents into a small
# routing set. Two of the brief's suggested routes (greeting/goodbye and
# out_of_scope) have NO samples in Bitext, so they are *not* trainable
# classes here:
#   * small talk is caught by lightweight keyword rules before the model,
#   * out-of-scope is modelled as low classifier confidence / no retrieval.
# Instead the data's shipping & human-contact intents get dedicated routes.
# ---------------------------------------------------------------------------
FINE_TO_ROUTE = {
    # order_status  -- "where is my parcel / when will it arrive / options"
    "track_order": "order_status",
    "delivery_options": "order_status",
    "delivery_period": "order_status",
    # order_management -- actions on an order
    "cancel_order": "order_management",
    "change_order": "order_management",
    "place_order": "order_management",
    # billing_and_refunds -- money, invoices, payments, fees
    "check_invoice": "billing_and_refunds",
    "get_invoice": "billing_and_refunds",
    "get_refund": "billing_and_refunds",
    "track_refund": "billing_and_refunds",
    "check_refund_policy": "billing_and_refunds",
    "check_cancellation_fee": "billing_and_refunds",
    "check_payment_methods": "billing_and_refunds",
    "payment_issue": "billing_and_refunds",
    # account_management -- profiles & preferences
    "create_account": "account_management",
    "edit_account": "account_management",
    "delete_account": "account_management",
    "switch_account": "account_management",
    "recover_password": "account_management",
    "registration_problems": "account_management",
    "newsletter_subscription": "account_management",
    # shipping_address
    "set_up_shipping_address": "shipping_address",
    "change_shipping_address": "shipping_address",
    # complaint -- priority / escalation regardless of RAG
    "complaint": "complaint",
    "review": "complaint",
    # contact_support -- request to talk to a person
    "contact_customer_service": "contact_support",
    "contact_human_agent": "contact_support",
}

# Routes that are trained as classes (all 27 intents map onto these).
ROUTES = [
    "order_status",
    "order_management",
    "billing_and_refunds",
    "account_management",
    "shipping_address",
    "complaint",
    "contact_support",
]

# Routing policy applied by the pipeline for each route.
ROUTE_POLICY = {
    "small_talk": "no_rag",             # greeting/goodbye/gratitude
    "order_status": "rag",
    "order_management": "rag",
    "billing_and_refunds": "rag",
    "account_management": "rag",
    "shipping_address": "rag",
    "complaint": "escalate",            # priority -> human-style handling
    "contact_support": "escalate",      # user wants a person
    "out_of_scope": "no_rag",
}

# ---------------------------------------------------------------------------
# Groq (LLM) settings -- read from the environment; no key hard-coded.
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "gpt-oss-120b")  # or gpt-oss-20b
GROQ_FALLBACK_MODELS = ["gpt-oss-120b", "gpt-oss-20b"]
GROQ_MAX_TOKENS = int(os.environ.get("GROQ_MAX_TOKENS", "500"))
