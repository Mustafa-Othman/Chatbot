# %% [markdown]
# # 03 — Intent classifier  (RAG e-commerce support chatbot) — **standalone / Colab**
#
# Stage 3 of the pipeline. Routing: *what does the customer actually want?* The
# answer decides the whole response policy:
#
# | behaviour                      | route(s)                                          |
# |--------------------------------|---------------------------------------------------|
# | small talk → **no RAG**        | greeting / goodbye / gratitude (keyword rule)    |
# | **RAG** grounded answer        | order_status, order_management, billing_and_refunds, account_management, shipping_address |
# | **escalate / priority**        | complaint, contact_support                        |
# | honest "can't help" → **no RAG**| out-of-scope (low confidence / no retrieval)      |
#
# **Dataset:** the Bitext customer-support corpus (gold `intent` labels for **27
# fine-grained intents**) — downloaded straight from HuggingFace, so this notebook
# needs no local files. The brief asks to condense those 27 intents into a small
# routing set; the mapping below is defined inline.
#
# > **Two routes from the brief have no training rows** in Bitext (`greeting` and
# > `out_of_scope`) — we handle them *outside* the model: small talk by keyword
# > rule, out-of-scope by a low-confidence / no-retrieval gate. The data instead
# > has meaningful **shipping-address** and **contact/human-agent** intents, which
# > get dedicated routes.
#
# **Method:** traditional ML — a `FeatureUnion` of word + character n-gram TF-IDF
# feeding a linear classifier. `MultinomialNB` vs `SGD(log_loss)` are compared;
# the better one (macro-F1) is kept.
#
# **Deliverable:** `intent_pipeline.pkl` + `route_map.json`.

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

default_dir = "/content/chatbot_artifacts" if IN_COLAB else "chatbot_artifacts"
ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", default_dir))
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
print("artifact dir:", ARTIFACT_DIR)

# ---- RUN SETTINGS -------------------------------------------------------
SMOKE = False
if os.environ.get("PROJECT_SMOKE") == "1":
    SMOKE = True

SEED = 42
print("SMOKE mode :", SMOKE)

# %% [markdown]
# ## Shared constants & pre-processing
#
# The 27-intent → route mapping, route list, policy and routing rules (small talk
# + "I want a human") are all inline (no `support_bot` package here).

# %%
import json
import pickle
import random
import re

import numpy as np
import pandas as pd

random.seed(SEED)

BICTEXT_HF = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"

FINE_TO_ROUTE = {
    # order_status — "where is my parcel / when will it arrive / options"
    "track_order": "order_status",
    "delivery_options": "order_status",
    "delivery_period": "order_status",
    # order_management — actions on an order
    "cancel_order": "order_management",
    "change_order": "order_management",
    "place_order": "order_management",
    # billing_and_refunds — money, invoices, payments, fees
    "check_invoice": "billing_and_refunds",
    "get_invoice": "billing_and_refunds",
    "get_refund": "billing_and_refunds",
    "track_refund": "billing_and_refunds",
    "check_refund_policy": "billing_and_refunds",
    "check_cancellation_fee": "billing_and_refunds",
    "check_payment_methods": "billing_and_refunds",
    "payment_issue": "billing_and_refunds",
    # account_management — profiles & preferences
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
    # complaint — priority / escalation regardless of RAG
    "complaint": "complaint",
    "review": "complaint",
    # contact_support — request to talk to a person
    "contact_customer_service": "contact_support",
    "contact_human_agent": "contact_support",
}

ROUTES = ["order_status", "order_management", "billing_and_refunds",
          "account_management", "shipping_address", "complaint",
          "contact_support"]

ROUTE_POLICY = {
    "small_talk": "no_rag",
    "order_status": "rag",
    "order_management": "rag",
    "billing_and_refunds": "rag",
    "account_management": "rag",
    "shipping_address": "rag",
    "complaint": "escalate",
    "contact_support": "escalate",
    "out_of_scope": "no_rag",
}

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


# Small-talk (greeting/goodbye/gratitude) keyword rule — runs BEFORE the model.
_SMALLTALK_RE = re.compile(
    r"\b(hi|hello|hey|good\s*(morning|afternoon|evening)|howdy|greetings"
    r"|goodbye|bye\b|good\s*night|see\s*you|thanks|thank\s*you|thx|ty"
    r"|cheers|appreciate|grateful|have a good one)\b", re.IGNORECASE)

# Strong "I want a person" signals — decisive for routing (escalation).
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


print(f"{len(FINE_TO_ROUTE)} fine intents -> {len(ROUTES)} routes")

# %% [markdown]
# ## 1. Load the Bitext knowledge base (from HuggingFace)
#
# Columns: `instruction` (customer message), `response` (agent answer),
# `intent` (gold fine label), `category`, `flags`. No local files needed.

# %%
from datasets import load_dataset

df = load_dataset(BICTEXT_HF, split="train").to_pandas()
print("shape:", df.shape, "| columns:", list(df.columns))
df = df.dropna(subset=["instruction", "response", "intent"])
print(df[["instruction", "intent", "category"]].head(3).to_string())

# %% [markdown]
# ## 2. The 27 fine intents → routing set
#
# Every fine intent maps to exactly one route; we check all are covered, then
# show route sizes and the mapping table.

# %%
fine_intents = sorted(df["intent"].unique())
unmapped = [i for i in fine_intents if i not in FINE_TO_ROUTE]
print(f"{len(fine_intents)} fine intents found; unmapped: {unmapped or 'none'}")

df["route"] = df["intent"].map(FINE_TO_ROUTE)
print("\nroute distribution:")
print(df["route"].value_counts().to_string())

map_tbl = pd.DataFrame(
    [{"route": rt, "fine intents": ", ".join(
        [f for f in fine_intents if FINE_TO_ROUTE[f] == rt])}
     for rt in ROUTES])
print("\nmapping summary:\n", map_tbl.to_string(index=False))

# %% [markdown]
# ## 3. Class balance & sanity checks
#
# Routes are fairly balanced (~1k each). Small talk should be ~absent from the
# corpus (it is) — that is why it is rule-handled instead of a trained class.

# %%
df["clean_inst"] = df["instruction"].apply(clean)
print(df["clean_inst"].str.split().str.len().describe().to_string())

st = df["clean_inst"].apply(is_small_talk).sum()
print(f"\nrule-detected small-talk rows in corpus: {st} (expected ~0)")

# %% [markdown]
# ## 4. Feature representation & model comparison
#
# Two complementary TF-IDF views are joined with a `FeatureUnion`:
# * **word (1–2) n-grams** — topical vocabulary ("cancel", "refund", "track");
# * **character (2–4) n-grams** — robustness to the corpus's deliberate typos.
#
# We compare two traditional-ML classifiers on a held-out slice and choose the
# better by macro-F1. Both produce probabilities (confidence gate downstream).

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline

FEATURES = FeatureUnion([
    ("words", TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                              sublinear_tf=True, min_df=2, max_features=200_000)),
    ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                              sublinear_tf=True, min_df=2, max_features=300_000)),
])


def make_pipe(clf):
    return Pipeline([("feats", FEATURES), ("clf", clf)])


X = df["clean_inst"].tolist()
y = df["route"].tolist()

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.1, random_state=SEED, stratify=y)

if SMOKE:                          # stratified samples keep the run fast
    X_tr, _, y_tr, _ = train_test_split(X_tr, y_tr, train_size=4000,
                                        random_state=SEED, stratify=y_tr)
    X_te, _, y_te, _ = train_test_split(X_te, y_te, train_size=1200,
                                        random_state=SEED, stratify=y_te)

print(f"train: {len(X_tr):5d}  | test: {len(X_te):5d}")

candidates = {
    "MultinomialNB": MultinomialNB(alpha=0.3),
    "SGD(log_loss)": SGDClassifier(loss="log_loss", alpha=1e-4,
                                   max_iter=1500, random_state=SEED),
}
results = []
for name, clf in candidates.items():
    pipe = make_pipe(clf)
    pipe.fit(X_tr, y_tr)
    p = pipe.predict(X_te)
    acc = accuracy_score(y_te, p)
    mf1 = f1_score(y_te, p, average="macro", zero_division=0)
    results.append((name, acc, mf1))
    print(f"{name:16s} acc={acc:.4f}  macro-F1={mf1:.4f}")

best_name = max(results, key=lambda r: r[2])[0]
print("\nchosen classifier:", best_name)

# %% [markdown]
# ## 5. Train the chosen model & evaluate on the held-out test set
#
# `complaint` and `contact_support` are the routes where a mis-route matters most
# (they change the response *policy*); we keep an eye on their recall.

# %%
from sklearn.metrics import classification_report, confusion_matrix

chosen = candidates[best_name]
final_pipe = make_pipe(chosen)
print("training final model on the held-out training split ...")
final_pipe.fit(X_tr, y_tr)                 # test set never seen during training
pred = final_pipe.predict(X_te)

print(f"\ntest accuracy    : {accuracy_score(y_te, pred):.4f}")
print(f"test macro-F1    : {f1_score(y_te, pred, average='macro', zero_division=0):.4f}")
print(f"test weighted-F1 : {f1_score(y_te, pred, average='weighted', zero_division=0):.4f}")
print("\n", classification_report(y_te, pred, digits=3, zero_division=0))

labels_r = sorted(set(y_te))
cm = confusion_matrix(y_te, pred, labels=labels_r)
print("confusion matrix (true rows / predicted cols):")
print(pd.DataFrame(cm, index=labels_r, columns=labels_r).to_string())

# %% [markdown]
# ## 6. Save the trained model + mapping
#
# The whole pipeline is pickled; the mapping (fine→route, routes, policy) is
# stored as JSON so notebook 04 reconstructs identical routing behaviour.

# %%
with open(ARTIFACT_DIR / "intent_pipeline.pkl", "wb") as fh:
    pickle.dump(final_pipe, fh)

mapping_dump = {"fine_to_route": FINE_TO_ROUTE,
                "routes": ROUTES,
                "route_policy": ROUTE_POLICY,
                "classes": list(final_pipe.classes_)}
(ARTIFACT_DIR / "route_map.json").write_text(
    json.dumps(mapping_dump, indent=2), encoding="utf-8")
print("saved:", ARTIFACT_DIR / "intent_pipeline.pkl")
print("saved:", ARTIFACT_DIR / "route_map.json")

# %% [markdown]
# ## 7. Live demo
#
# `route_of` mirrors the serving call: small-talk is caught by the rule, an
# explicit "I want a human" is forced to `contact_support`, everything else goes
# through the trained classifier (probability shown).

# %%
def route_of(text, pipe=None):
    s = clean(text)
    if is_small_talk(s):
        return "small_talk", 1.0
    if is_escalation_request(s):
        return "contact_support", 0.99
    pipe = pipe or final_pipe
    proba = pipe.predict_proba([s])[0]
    k = int(proba.argmax())
    return pipe.classes_[k], float(proba[k])


demos = [
    "hi there, good morning",
    "thanks a lot for your help!",
    "where is my package? it should have arrived yesterday",
    "i want to cancel my order please",
    "how do i get a refund for a damaged item?",
    "i forgot my password, can you help me reset it",
    "please change the delivery address on my order",
    "you people are useless, i want to speak to a manager",
    "can i talk to a human agent instead of the bot?",
    "do you also sell books on the website?",
]
for d in demos:
    route, conf = route_of(d)
    print(f"[{route:20s} conf={conf:.2f}] {d}")

# %% [markdown]
# ## Summary
#
# * Stage-3 module: a **traditional-ML route classifier** trained on the gold
#   27-intent labels condensed into 7 routing classes (word + char TF-IDF,
#   `MultinomialNB` vs `SGD(log_loss)` compared, better kept).
# * `small_talk` handled by keyword rules; `out_of_scope` by the pipeline's
#   low-confidence / no-retrieval gate.
# * Saves `intent_pipeline.pkl` + `route_map.json` into the shared artifact folder.
#
# Next: notebook **04 — Q&A RAG pipeline** (retrieval + Groq generation + full
# four-stage integration, loading the models 01–03 saved).
