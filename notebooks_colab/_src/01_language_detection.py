# %% [markdown]
# # 01 — Language Detection  (RAG e-commerce support chatbot)  — **standalone / Colab**
#
# Stage 1 of the pipeline. Every incoming customer message is classified into one
# of **20 languages** with a *traditional NLP* model (character TF-IDF + linear
# SGD classifier).
#
# This notebook is **fully standalone**: no project files, no config module, no
# Drive folder needed. It downloads its dataset from HuggingFace and saves the
# trained model into a shared **artifact folder** (`/content/chatbot_artifacts`
# on Colab) that notebooks 01→04 all use.
#
# **Dataset:** [`papluca/language-identification`](https://huggingface.co/datasets/papluca/language-identification)
# — ~90k short texts across 20 languages, pre-split into `train/validation/test`.
#
# **Method:** character n-gram (`char_wb`, 2–4) TF-IDF + `SGDClassifier(log_loss)`
# — character n-grams capture script/orthography for unseen words and work across
# very different writing systems. `log_loss` gives calibrated probabilities used
# downstream as a confidence gate.
#
# > **Colab run order:** run notebooks **01 → 02 → 03 → 04** in one Colab session,
# > keeping the runtime alive. Each saves into the same artifact folder; notebook
# > 04 loads models 01–03 from it.
#
# **Deliverable:** `language_pipeline.pkl` + `language_report.json`.

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

# Shared artifact folder (same path in notebooks 01-04). On Colab this lives in
# the (ephemeral) runtime, so run 01 -> 04 in ONE session. Override anytime with
# the ARTIFACT_DIR environment variable.
default_dir = "/content/chatbot_artifacts" if IN_COLAB else "chatbot_artifacts"
ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", default_dir))
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
print("artifact dir:", ARTIFACT_DIR)

# ---- RUN SETTINGS -------------------------------------------------------
# SMOKE = True -> tiny sample so the whole notebook runs in < 1 minute.
SMOKE = False
if os.environ.get("PROJECT_SMOKE") == "1":
    SMOKE = True

SEED = 42
print("SMOKE mode :", SMOKE)

# %% [markdown]
# ## Shared constants & pre-processing
#
# The language-to-name map and the single `clean()` function used across all four
# modules are defined inline below (this notebook has no `support_bot` package).

# %%
import json
import random

import numpy as np

random.seed(SEED)

DATASET_LANG = "papluca/language-identification"

LANG_NAMES = {
    "ar": "Arabic", "bg": "Bulgarian", "de": "German", "el": "Greek",
    "en": "English", "es": "Spanish", "fr": "French", "hi": "Hindi",
    "it": "Italian", "ja": "Japanese", "nl": "Dutch", "pl": "Polish",
    "pt": "Portuguese", "ru": "Russian", "sw": "Swahili", "th": "Thai",
    "tr": "Turkish", "ur": "Urdu", "vi": "Vietnamese", "zh": "Chinese",
}

import re

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_WS_RE = re.compile(r"\s+")


def clean(text: str) -> str:
    """Shared pre-processing (identical in every notebook). Punctuation and
    accents are kept on purpose: they carry signal for character n-grams and
    separate languages that share a script."""
    if text is None:
        return ""
    s = str(text).lower()
    s = _URL_RE.sub(" ", s)
    s = _EMAIL_RE.sub(" ", s)
    s = s.replace("\r", " ").replace("\n", " ")
    s = _WS_RE.sub(" ", s)
    return s.strip()


print("n languages:", len(LANG_NAMES))

# %% [markdown]
# ## 1. Load the dataset
#
# The dataset ships pre-split. `labels` are ISO 639-1 codes (`en`, `fr`, `zh`…)
# and `text` is the message. In SMOKE mode we stream only a few thousand rows so
# nothing big is downloaded.

# %%
from datasets import load_dataset

if SMOKE:
    from datasets import Dataset

    def smoke_split(split_name, n):
        it = load_dataset(DATASET_LANG, split=split_name, streaming=True)
        return Dataset.from_list(list(it.take(n)))

    splits = {"train": smoke_split("train", 3000),
              "validation": smoke_split("validation", 800),
              "test": smoke_split("test", 1500)}
else:
    splits = {k: load_dataset(DATASET_LANG, split=k)
              for k in ("train", "validation", "test")}

for k, ds in splits.items():
    print(f"{k:10s} {len(ds):6d} rows")

# %% [markdown]
# ## 2. Exploratory check
#
# Confirm the label set, balance, and that all 20 expected languages are present.

# %%
train_df = splits["train"].to_pandas()
print("distinct languages:", sorted(train_df["labels"].unique()))
print("\nrows per language (train):")
print(train_df["labels"].value_counts().to_string())

missing = set(train_df["labels"].unique()) - set(LANG_NAMES)
print("\nlanguages without a friendly name:", missing or "none")

# %% [markdown]
# ## 3. Pre-processing & feature representation
#
# Clean text with the shared `clean()`, then build character n-gram TF-IDF
# (`ngram_range=(2, 4)`, `sublinear_tf`, `min_df=2`).

# %%
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

train_df["text_clean"] = train_df["text"].apply(clean)
print(train_df[["text", "text_clean"]].head(3).to_string())

X_tr = train_df["text_clean"].tolist()
y_tr = train_df["labels"].tolist()

X_va = splits["validation"].to_pandas()["text"].apply(clean).tolist()
y_va = splits["validation"].to_pandas()["labels"].tolist()
X_te = splits["test"].to_pandas()["text"].apply(clean).tolist()
y_te = splits["test"].to_pandas()["labels"].tolist()

tfidf = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(2, 4),
    min_df=2,
    max_features=1_000_000,
    sublinear_tf=True,
)

# loss='log_loss' -> linear model that also outputs predict_proba (the
# probabilities are used downstream as a confidence gate).
clf = SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=1200,
                    random_state=SEED)

pipeline = Pipeline([("tfidf", tfidf), ("clf", clf)])
print(pipeline)

# %% [markdown]
# ## 4. Train & evaluate
#
# Fit on `train`, then report held-out accuracy plus a full per-language report.

# %%
from sklearn.metrics import accuracy_score

print("training ...")
pipeline.fit(X_tr, y_tr)

va_acc = accuracy_score(y_va, pipeline.predict(X_va))
te_acc = accuracy_score(y_te, pipeline.predict(X_te))
print(f"validation accuracy : {va_acc:.4f}")
print(f"test accuracy       : {te_acc:.4f}")

# %% [markdown]
# ### Per-language diagnostics
#
# Remaining errors are almost always *close pairs* (Dutch/German, Spanish/Portuguese)
# and short/code-switched messages.

# %%
from sklearn.metrics import classification_report

labels_sorted = sorted(set(y_te))
pred = pipeline.predict(X_te)

conf = np.zeros((len(labels_sorted), len(labels_sorted)), dtype=int)
idx = {l: i for i, l in enumerate(labels_sorted)}
for t, p in zip(y_te, pred):
    conf[idx[t], idx[p]] += 1

print(classification_report(y_te, pred, digits=3, zero_division=0))

print("\nTop confusing pairs (true -> predicted):")
errs = []
for i, l1 in enumerate(labels_sorted):
    for j, l2 in enumerate(labels_sorted):
        if i != j and conf[i, j] > 0:
            errs.append((conf[i, j], l1, l2))
for n, l1, l2 in sorted(errs, reverse=True)[:8]:
    print(f"  {n:5d}  {l1:>3} -> {l2:<3}   ({LANG_NAMES[l1]} -> {LANG_NAMES[l2]})")

# %% [markdown]
# ## 5. Save the trained model
#
# The whole `Pipeline` (vectorizer **and** classifier) is pickled into the shared
# artifact folder, so tokenisation can never drift between training and the final
# pipeline in notebook 04.

# %%
import pickle

LANG_PIPELINE = ARTIFACT_DIR / "language_pipeline.pkl"
with open(LANG_PIPELINE, "wb") as fh:
    pickle.dump(pipeline, fh)

report = {"test_accuracy": float(te_acc), "validation_accuracy": float(va_acc),
          "n_languages": len(labels_sorted),
          "languages": {l: LANG_NAMES[l] for l in labels_sorted}}
(ARTIFACT_DIR / "language_report.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8")
print("saved:", LANG_PIPELINE)

# %% [markdown]
# ## 6. Quick live demo

# %%
def detect_language(text):
    s = clean(text)
    proba = pipeline.predict_proba([s])[0]
    order = np.argsort(proba)[::-1]
    top = [(pipeline.classes_[i], float(proba[i]), LANG_NAMES[pipeline.classes_[i]])
           for i in order[:3]]
    best = top[0]
    return {"language": best[0], "name": best[2],
            "confidence": round(best[1], 4), "top3": top}


samples = [
    "Hi, where is my order? I ordered a phone last week.",
    "¿Dónde está mi pedido? Llevo una semana esperando.",
    "Où est ma commande ? Je l'ai passée il y a une semaine.",
    "أين طلبي؟ لقد طلبت هاتفاً الأسبوع الماضي",
    "我的订单在哪里？我上周订了一部手机。",
    "Wo ist meine Bestellung? Ich habe letzte Woche ein Telefon bestellt.",
    "Gdzie jest moje zamówienie? Zamówiłem telefon w zeszłym tygodniu.",
]
for s in samples:
    r = detect_language(s)
    print(f"{s[:46]:<48} -> {r['name']:10s} conf={r['confidence']:.3f}")

# %% [markdown]
# ## Summary
#
# * Stage-1 module: a **20-class** language identifier, character n-gram TF-IDF +
#   linear SGD, ~**0.99** held-out accuracy on full data.
# * Saves `language_pipeline.pkl` + `language_report.json` into the shared
#   artifact folder, consumed by notebook **04**.
#
# Next: notebook **02 — Sentiment / Emotion** (a from-scratch RNN).
