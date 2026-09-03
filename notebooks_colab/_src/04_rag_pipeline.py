# %% [markdown]
# # 04 — Q&A RAG pipeline  (RAG e-commerce support chatbot) — **standalone / Colab**
#
# Stage 4 — Retrieval-Augmented Generation, wired together with stages 1–3.
#
# ```text
#  customer message
#     │  language detection (01) ── 20 languages / scripts
#     │  sentiment (02)          ── emotion -> negative/neutral/positive tone
#     │  intent routing (03)     ── small talk | 7 routes
#     ▼
#  response policy
#     ├─ small talk             -> canned reply            (no RAG)
#     ├─ complaint/human-request-> apologetic + escalate   (no auto-answer)
#     ├─ no confident retrieval -> honest "can't help"     (no guessing)
#     └─ everything else        -> RETRIEVE KB + GROUNDED LLM answer
#                                    │  embed question  (sentence-transformers)
#                                    │  top-k FAISS hits (MiniLM cosine)
#                                    ▼
#                            Groq gpt-oss  (only if GROQ_API_KEY set)
# ```
#
# **Components (as the brief asks):**
# * **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (384-d, local).
# * **Vector store:** local **FAISS `IndexFlatIP`** over L2-normalised vectors
#   (= cosine similarity) — no external account; cloud Qdrant is a drop-in swap.
# * **Knowledge base:** the Bitext `instruction → response` pairs: embed the
#   *question* for retrieval, inject the paired *response* as grounding.
# * **LLM:** **Groq** `gpt-oss-120b` — called only when `GROQ_API_KEY` is set;
#   otherwise the pipeline returns the best retrieved support response.
#
# This notebook is **fully standalone** except for one thing: it **loads the
# models that notebooks 01–03 saved** into the shared artifact folder
# (`/content/chatbot_artifacts`). So run **01 → 02 → 03 → 04** in one Colab
# session, keeping the runtime alive. If a model is missing you will get a clear
# message naming the notebook to run first.

# %%
# ===== Colab setup (no-op when run locally) ==============================
import os
import subprocess
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:                       # install whatever Colab is missing
    missing = []
    for _mod in ("datasets", "sentence_transformers", "groq"):
        try:
            __import__(_mod)
        except ImportError:
            missing.append(_mod.replace("_", "-"))
    try:
        import faiss  # noqa: F401
    except ImportError:
        missing.append("faiss-cpu")
    for _mod in ("sklearn", "pandas", "numpy"):
        try:
            __import__(_mod)
        except ImportError:
            missing.append("scikit-learn" if _mod == "sklearn" else _mod)
    if missing:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + missing,
                       check=False)
    print("Colab: installed missing packages:", missing or "none")
    # optional: read the Groq key from a Colab secret named GROQ_API_KEY
    try:
        from google.colab import userdata
        os.environ.setdefault("GROQ_API_KEY", userdata.get("GROQ_API_KEY"))
    except Exception:
        pass

default_dir = "/content/chatbot_artifacts" if IN_COLAB else "chatbot_artifacts"
ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", default_dir))
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
print("artifact dir:", ARTIFACT_DIR)

# ---- RUN SETTINGS -------------------------------------------------------
SMOKE = False                     # True -> tiny KB + fast smoke run
if os.environ.get("PROJECT_SMOKE") == "1":
    SMOKE = True

SEED = 42
REBUILD = True                    # False -> reuse an existing FAISS index
TOP_K = 3
print("SMOKE mode :", SMOKE, "| REBUILD =", REBUILD)

# %% [markdown]
# ## Part A — Build the retrieval store
#
# ### A.1 Load the knowledge base
# Each KB unit is a real `(customer question, agent response)` pair from the
# Bitext corpus (HuggingFace). We index the *question* text; the *response* is
# the ground truth we ground answers on.

# %%
import re
import time
import pickle
import json

import numpy as np
import pandas as pd

BICTEXT_HF = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
EMBEDDER_ID = "sentence-transformers/all-MiniLM-L6-v2"

KB_INDEX_PATH = ARTIFACT_DIR / "faiss.index"
KB_META_PATH = ARTIFACT_DIR / "kb_meta.pkl"

# %%
from datasets import load_dataset

if REBUILD or not (KB_INDEX_PATH.exists() and KB_META_PATH.exists()):
    df = load_dataset(BICTEXT_HF, split="train").to_pandas()
    df = df.dropna(subset=["instruction", "response", "intent"])
    print("KB rows:", len(df))

    if SMOKE:
        df = df.sample(n=2000, random_state=SEED).reset_index(drop=True)
        print("SMOKE subset:", len(df), "rows")

    metas = [{"instruction": r.instruction, "response": r.response,
              "intent": r.intent,
              "category": getattr(r, "category", "")}
             for r in df.itertuples()]
    texts = [m["instruction"] for m in metas]
    print("example chunk:\n - Q:", texts[0][:90], "\n - intent:", metas[0]["intent"])
else:
    print("Reusing existing index (REBUILD = False).")

# %% [markdown]
# ### A.2 Embed the instructions
# `all-MiniLM-L6-v2` maps a sentence to a **384-d** vector, L2-normalised so that
# a flat inner-product search equals cosine similarity. The model downloads once.

# %%
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer(EMBEDDER_ID)

if REBUILD or not (KB_INDEX_PATH.exists() and KB_META_PATH.exists()):
    t0 = time.time()
    vectors = embedder.encode(list(texts), batch_size=64, show_progress_bar=True,
                              normalize_embeddings=True, convert_to_numpy=True)
    vectors = np.asarray(vectors, dtype=np.float32)
    print(f"embedded {vectors.shape[0]} chunks -> shape {vectors.shape} "
          f"({time.time()-t0:.1f}s)")

# %% [markdown]
# ### A.3 Build the FAISS index & save
# `IndexFlatIP` is an exact (brute-force) cosine index — perfect for ~27k chunks.

# %%
import faiss

if REBUILD or not (KB_INDEX_PATH.exists() and KB_META_PATH.exists()):
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    print("index size:", index.ntotal, "| dimension:", dim)

    faiss.write_index(index, str(KB_INDEX_PATH))
    with open(KB_META_PATH, "wb") as fh:
        pickle.dump(metas, fh)
    print("saved:", KB_INDEX_PATH)
    print("saved:", KB_META_PATH)
else:
    index = faiss.read_index(str(KB_INDEX_PATH))
    with open(KB_META_PATH, "rb") as fh:
        metas = pickle.load(fh)
    print("loaded existing index, ntotal:", index.ntotal)

# %% [markdown]
# ### A.4 Sanity: do neighbours share a topic?
# For random KB rows we search for the row itself, skip the trivial self-match,
# and check whether the next hit shares the **category / intent**. High numbers
# mean retrieval is semantically meaningful.

# %%
rng = np.random.default_rng(SEED)
n_q = min(100 if SMOKE else 300, len(metas))
qidx = rng.choice(len(metas), size=n_q, replace=False)
same_cat = same_intent = 0
qvecs = embedder.encode([metas[int(i)]["instruction"] for i in qidx],
                        normalize_embeddings=True, convert_to_numpy=True)
qvecs = np.ascontiguousarray(qvecs, dtype=np.float32)
for n, i in enumerate(qidx):
    scores, ids = index.search(qvecs[n:n + 1], 5)
    for j in ids[0]:
        if j != i:
            other = metas[int(j)]
            same_cat += int(other["category"] == metas[int(i)]["category"])
            same_intent += int(other["intent"] == metas[int(i)]["intent"])
            break
print(f"checked {n_q} queries")
print(f"neighbour shares CATEGORY : {same_cat / n_q:.3f}")
print(f"neighbour shares INTENT   : {same_intent / n_q:.3f}")

# %% [markdown]
# ### A.5 Retrieval demo

# %%
def retrieve(query, top_k=TOP_K):
    q = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    scores, ids = index.search(np.ascontiguousarray(q, dtype=np.float32), top_k)
    return [(float(scores[0][k]), metas[int(ids[0][k])]) for k in range(top_k)]


for q in ["I need to cancel my order, how do I do that?",
          "when will my package arrive?",
          "i was charged twice, give me my money back"]:
    print("\nQ:", q)
    for sc, m in retrieve(q):
        print(f"   {sc:6.3f}  [{m['intent']:22s} {m['category']:14s}] {m['instruction'][:70]}")

# %% [markdown]
# ## Part B — Grounded generation with Groq
#
# The system prompt is strict: answer **only** from the retrieved responses,
# acknowledge frustration if the tone is negative, and honestly offer a human
# agent if the context doesn't cover the question. Groq is contacted only when
# `GROQ_API_KEY` is present (set it as a Colab **secret** named `GROQ_API_KEY`,
# or paste your key into the setup cell). Without a key the demo still works —
# it returns the best retrieved support response instead.

# %%
try:  # groq is an optional dependency
    import groq
except Exception:  # pragma: no cover
    groq = None

GROQ_MODEL = os.environ.get("GROQ_MODEL", "gpt-oss-120b")
GROQ_FALLBACK_MODELS = ["gpt-oss-120b", "gpt-oss-20b"]
GROQ_MAX_TOKENS = int(os.environ.get("GROQ_MAX_TOKENS", "500"))


class GroqClient:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.model = model or GROQ_MODEL
        self._client = None
        if self.api_key and groq is not None:
            self._client = groq.Groq(api_key=self.api_key)

    @property
    def available(self):
        return self._client is not None

    def generate(self, system: str, user: str,
                 temperature: float = 0.3, max_tokens: int | None = None) -> dict:
        if self._client is None:
            return {"ok": False, "text": "", "error": "GROQ_API_KEY not set"}
        max_tokens = max_tokens or GROQ_MAX_TOKENS
        candidates = [self.model] + [m for m in GROQ_FALLBACK_MODELS if m != self.model]
        last_err = ""
        for m in candidates:
            try:
                resp = self._client.chat.completions.create(
                    model=m,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    temperature=temperature, max_tokens=max_tokens)
                return {"ok": True, "text": resp.choices[0].message.content.strip(),
                        "model": m, "error": ""}
            except Exception as e:  # noqa: BLE001
                last_err = f"{type(e).__name__}: {e}"
                if "model" not in last_err.lower() or "not found" not in last_err.lower():
                    break
        return {"ok": False, "text": "", "error": last_err}


def build_generation_prompt(user_message, retrieved, detected_language,
                            detected_tone, route, escalate=False):
    system = ("You are a helpful, professional customer support assistant for an "
              "online retailer. Answer the customer's question using ONLY the "
              "information in the retrieved support responses below. ")
    if detected_tone == "negative":
        system += ("The customer sounds frustrated; acknowledge that "
                   "sincerely before answering. ")
    if escalate:
        system += ("The customer asked to speak to a person or filed a complaint: "
                   "be extra apologetic and clearly offer to escalate to a human agent.")
    system += ("If the retrieved context does not cover the question, say so "
               "honestly and offer to escalate to a human agent rather than guessing. "
               "Do not invent policies, fees, links or phone numbers that are not in "
               "the context. "
               f"If the customer's language is '{detected_language}' (not English), "
               "you may translate your answer into that language, but keep the "
               "factual content identical to the English context.")
    parts = ["Context (retrieved past support responses):"]
    for i, c in enumerate(retrieved, 1):
        parts.append(f"[{i}] (intent={c.get('intent')}, category={c.get('category')})\n"
                     f"Q: {c.get('instruction', '')}\nA: {c.get('response', '')}")
    parts.append(f"Customer question: \"{user_message}\"")
    return system, "\n\n".join(parts)


client = GroqClient()
print("Groq API key present :", client.available)
print("primary model        :", client.model)
if not client.available:
    print("\n[info] No GROQ_API_KEY -> generation cell shows the built prompt only.")
    print("       Retrieval-only answers (best KB response) still work everywhere.\n")


def generate_answer(query, top_k=TOP_K):
    hits = retrieve(query, top_k=top_k)
    system, user = build_generation_prompt(
        query, [m for _, m in hits], "english", "neutral", "rag")
    if client.available:
        return client.generate(system, user), hits
    return None, hits


for q in ["How do I return a faulty item and get my money back?",
          "What are the shipping options for France?",
          "Can you sing me a song about penguins?"]:
    res, hits = generate_answer(q)
    print("\n" + "=" * 80)
    print("Q:", q)
    print("retrieved intents:", [h[1]["intent"] for h in hits],
          "| top score:", round(hits[0][0], 3))
    if res and res["ok"]:
        print("A:", res["text"])
    else:
        print("(no LLM call — key not set)")

# %% [markdown]
# ## Part C — Full four-stage chat pipeline
#
# This loads the stage-1/2/3 models that notebooks **01–03** saved into the
# shared artifact folder. Run them first — if anything is missing, the message
# below tells you exactly which notebook to run (in the **same** Colab runtime,
# because the artifact folder lives on the runtime disk).

# %%
import torch
import torch.nn as nn

# --- shared pre-processing (identical to notebooks 01-03) --------------------
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_WS_RE = re.compile(r"\s+")


def clean(text: str) -> str:
    if text is None:
        return ""
    s = str(text).lower()
    s = _URL_RE.sub(" ", s)
    s = _EMAIL_RE.sub(" ", s)
    s = s.replace("\r", " ").replace("\n", " ")
    s = _WS_RE.sub(" ", s)
    return s.strip()


# --- sentiment model architecture (identical to notebook 02) ----------------
class EmotionBiLSTM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128,
                 hidden_dim: int = 128, num_layers: int = 2,
                 num_classes: int = 6, dropout: float = 0.4,
                 pad_idx: int = 0, max_len: int = 40):
        super().__init__()
        self.cfg = dict(vocab_size=vocab_size, embed_dim=embed_dim,
                        hidden_dim=hidden_dim, num_layers=num_layers,
                        num_classes=num_classes, dropout=dropout,
                        pad_idx=pad_idx, max_len=max_len)
        self.pad_idx = pad_idx
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0,
                            bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * hidden_dim, num_classes)

    def forward(self, x: torch.Tensor):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        mask = (x != self.pad_idx).unsqueeze(-1).float()
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


# --- routing rules (identical to notebook 03) --------------------------------
_SMALLTALK_RE = re.compile(
    r"\b(hi|hello|hey|good\s*(morning|afternoon|evening)|howdy|greetings"
    r"|goodbye|bye\b|good\s*night|see\s*you|thanks|thank\s*you|thx|ty"
    r"|cheers|appreciate|grateful|have a good one)\b", re.IGNORECASE)
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
    s = clean(text)
    return bool(_SMALLTALK_RE.search(s)) and not any(
        k in s for k in ("order", "refund", "invoice", "deliver", "account",
                         "password", "payment", "cancel", "shipping", "address",
                         "complaint", "return", "track"))


def is_escalation_request(text: str) -> bool:
    return any(p.search(clean(text)) for p in _ESCALATION_PATTERNS)


# --- small canned replies, keyed by detected language ------------------------
_SMALLTALK = {
    "en": ["Hi! I'm the customer support assistant. How can I help you today?",
           "You're welcome! Is there anything else I can do for you?",
           "Goodbye! Thanks for contacting us. Have a great day!"],
    "es": ["¡Hola! Soy el asistente de atención al cliente. ¿En qué puedo ayudarte hoy?"],
    "fr": ["Bonjour ! Je suis l'assistant du service client. Comment puis-je vous aider ?"],
    "de": ["Hallo! Ich bin der Kundenservice-Assistent. Wie kann ich Ihnen heute helfen?"],
    "pt": ["Olá! Sou o assistente de atendimento ao cliente. Como posso ajudar hoje?"],
    "it": ["Ciao! Sono l'assistente del servizio clienti. Come posso aiutarti oggi?"],
    "zh": ["你好！我是客户支持助理，今天有什么可以帮您的吗？"],
    "ar": ["مرحباً! أنا مساعد دعم العملاء. كيف يمكنني مساعدتك اليوم؟"],
}
KB_SIM_MIN = 0.45
_PLACEHOLDER_RE = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}")


def deplaceholder(text: str) -> str:
    return _PLACEHOLDER_RE.sub(
        lambda m: m.group(1).strip().lower().replace("_", " "), text)


# %%
def _require(path, notebook):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing '{path.name}'. Please run notebook {notebook} FIRST, in the "
            f"SAME Colab runtime (models are stored at {ARTIFACT_DIR}).")

# 1) language  (01)
_require(ARTIFACT_DIR / "language_pipeline.pkl", "01 - Language Detection")
with open(ARTIFACT_DIR / "language_pipeline.pkl", "rb") as fh:
    lang_pipe = pickle.load(fh)
lang_report = json.loads((ARTIFACT_DIR / "language_report.json").read_text(encoding="utf-8"))
LANG_NAMES = lang_report.get("languages", {})
print("language model : loaded (classes:", len(lang_pipe.classes_), ")")

# 2) sentiment  (02)
_require(ARTIFACT_DIR / "sentiment_checkpoint.pt", "02 - Sentiment / Emotion")
ckpt = torch.load(ARTIFACT_DIR / "sentiment_checkpoint.pt",
                  map_location="cpu", weights_only=False)
sent_model = EmotionBiLSTM(**ckpt["model_cfg"])
sent_model.load_state_dict(ckpt["model_state"])
sent_model.eval()
SENT_LABELS = ckpt["labels"]
SENT_TONE = ckpt["tone_map"]
SENT_VOCAB = ckpt["vocab"]
print("sentiment model : loaded (labels:", SENT_LABELS, ")")

# 3) intent  (03)
_require(ARTIFACT_DIR / "intent_pipeline.pkl", "03 - Intent classifier")
with open(ARTIFACT_DIR / "intent_pipeline.pkl", "rb") as fh:
    intent_pipe = pickle.load(fh)
route_map = json.loads((ARTIFACT_DIR / "route_map.json").read_text(encoding="utf-8"))
ROUTE_POLICY = route_map["route_policy"]
print("intent model    : loaded (classes:", list(intent_pipe.classes_), ")")

# %%
def detect_language(text):
    s = clean(text)
    proba = lang_pipe.predict_proba([s])[0]
    k = int(proba.argmax())
    iso = str(lang_pipe.classes_[k])
    return {"language": iso, "lang_conf": float(proba[k]),
            "lang_name": LANG_NAMES.get(iso, iso)}


def detect_sentiment(text):
    model = sent_model
    pad_idx = model.cfg["pad_idx"]
    max_len = model.cfg["max_len"]
    unk = SENT_VOCAB.get("<unk>", 1)
    toks = re.findall(r"[^\W_]+", clean(text).lower())[:max_len]
    ids = [SENT_VOCAB.get(w, unk) for w in toks]
    X = torch.full((1, max_len), pad_idx, dtype=torch.long)
    X[0, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=-1)[0]
    k = int(probs.argmax().item())
    emotion = SENT_LABELS[k]
    return {"emotion": emotion, "emotion_conf": float(probs[k]),
            "tone": SENT_TONE.get(emotion, "neutral")}


def route_of(text):
    s = clean(text)
    if is_small_talk(s):
        return "small_talk", 1.0
    if is_escalation_request(s):
        return "contact_support", 0.99
    proba = intent_pipe.predict_proba([s])[0]
    k = int(proba.argmax())
    return str(intent_pipe.classes_[k]), float(proba[k])


def analyze(message):
    out = {"message": message}
    out.update(detect_language(message))
    out.update(detect_sentiment(message))
    route, conf = route_of(message)
    out["route"], out["route_conf"] = route, conf
    out["policy"] = ROUTE_POLICY.get(route, "rag")
    return out

# %%
def bot_answer(message, top_k=TOP_K, use_llm=True):
    t0 = time.time()
    message = clean(message)
    if not message:
        return {"reply": "Please write a message so I can help you.",
                "error": "empty_message", "latency": 0.0}

    a = analyze(message)
    route, policy = a["route"], a["policy"]
    negative = a["tone"] == "negative"
    result = {"reply": "", "route": route, "policy": policy,
              "escalate": False, "priority": negative,
              "sources": [], "used_llm": False, "note": "",
              "latency": 0.0, **a}

    # ---- small talk: no RAG --------------------------------------------
    if route == "small_talk":
        result["reply"] = _SMALLTALK.get(a["language"], _SMALLTALK["en"])[0]
        result["note"] = "handled without retrieval (small talk)"
        result["latency"] = round(time.time() - t0, 3)
        return result

    # ---- complaint / ask-for-human: apologise + escalate ----------------
    if policy == "escalate":
        apology = ("I'm very sorry for the frustration this has caused. "
                   if negative else "")
        if route == "contact_support":
            reply = apology + ("Of course — I've flagged your request and a human "
                               "agent will join you shortly. While you wait, is "
                               "there anything about your order I can look up?")
        else:
            reply = apology + ("I've escalated your case to our support team for "
                               "priority handling — a human agent will respond to "
                               "you as soon as possible.")
        result["reply"] = reply
        result["escalate"] = True
        result["priority"] = True
        result["note"] = "priority path (complaint / human request): no auto-generated policy answer"
        result["latency"] = round(time.time() - t0, 3)
        return result

    # ---- RAG path -------------------------------------------------------
    hits = retrieve(message, top_k=top_k)
    result["sources"] = [{"score": sc, **m} for sc, m in hits]
    if not hits or hits[0][0] < KB_SIM_MIN:
        empathy = ("I'm sorry this is frustrating. " if negative else "")
        result["reply"] = (empathy + "I couldn't find an answer for that in our "
                           "knowledge base. Would you like me to connect you with "
                           "a human agent?")
        result["note"] = "no confident retrieval -> honest out-of-scope reply"
        result["latency"] = round(time.time() - t0, 3)
        return result

    system, user = build_generation_prompt(
        message, [m for _, m in hits[:top_k]], a["lang_name"], a["tone"], route,
        escalate=False)

    if use_llm and client.available:
        gen = client.generate(system, user)
        if gen["ok"]:
            reply = gen["text"]
            if negative and not reply.lower().startswith(
                    ("i'm sorry", "i am sorry", "sorry", "i understand")):
                reply = "I'm sorry to hear that. " + reply
            result["reply"] = reply
            result["used_llm"] = True
            result["note"] = f"grounded answer via {gen['model']}"
            result["latency"] = round(time.time() - t0, 3)
            return result

    # retrieval-only fallback (no key / LLM error) -----------------------
    best = hits[0][1]
    fallback = deplaceholder(best["response"])
    if negative:
        fallback = "I'm sorry this is frustrating — let me help. " + fallback
    result["reply"] = fallback
    result["note"] = "retrieval-only reply (no LLM available)"
    result["latency"] = round(time.time() - t0, 3)
    return result

print("pipeline ready.")

# %% [markdown]
# ### End-to-end demos
# Small talk (no RAG), a negative order-status message (apology + retrieval),
# order management, and a **Spanish** message (language detection matters).

# %%
demos = [
    "hi there, good morning!",                                              # small talk
    "I still haven't received my order and nobody answers. This is a joke.",  # negative
    "I want to cancel my order #8234 please",                              # order_management
    "¿Cuándo llegará mi pedido?",                                          # Spanish
]

for msg in demos:
    print("\n" + "#" * 90)
    print("USER:", msg)
    out = bot_answer(msg)
    print(f"lang    : {out.get('language')} ({out.get('lang_name')})  [{out.get('lang_conf', 0):.2f}]")
    print(f"tone    : {out.get('tone')} ({out.get('emotion')}) [{out.get('emotion_conf', 0):.2f}]")
    print(f"route   : {out.get('route')}  policy={out.get('policy')}  escalate={out.get('escalate')}")
    srcs = out.get("sources", [])
    if srcs:
        print("sources :", [(round(s["score"], 2), s["intent"], s["category"]) for s in srcs[:2]])
    print("BOT     :", out.get("reply", "")[:400])

# %% [markdown]
# ### Escalation + honesty demos
# A **complaint** must route to escalation rather than an auto-answer; an
# **out-of-knowledge** question must be answered honestly (no invented policy).

# %%
for msg in ["You charged me twice and now you ignore my emails. I want a manager.",
            "Do you sponsor a football club in Brazil?"]:
    print("\n" + "#" * 90)
    print("USER:", msg)
    out = bot_answer(msg)
    print(f"route   : {out.get('route')}  policy={out.get('policy')}  escalate={out.get('escalate')}")
    print("BOT     :", out.get("reply", "")[:300])

# %% [markdown]
# ## Summary
#
# * Stage-4 module complete:
#   1. **KB index** — Bitext `instruction→response` pairs embedded with
#      `all-MiniLM-L6-v2` into a FAISS cosine index (`faiss.index` + `kb_meta.pkl`).
#   2. **Grounded generation** — Groq `gpt-oss` prompted strictly from the
#      retrieved responses, with an honest "not covered → offer human" fallback —
#      or retrieval-only answers if no key is set.
#   3. **Full integration** — language (01) + sentiment (02) + intent (03) + RAG
#      behind one `bot_answer(message)`, including small-talk, escalation,
#      empathy and out-of-scope handling — all run from notebooks that were
#      uploaded and run on Colab with **no project files**.
