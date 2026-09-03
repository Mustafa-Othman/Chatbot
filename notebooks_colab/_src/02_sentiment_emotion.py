# %% [markdown]
# # 02 — Sentiment / Emotion classifier  (RAG e-commerce support chatbot) — **standalone / Colab**
#
# Stage 2 of the pipeline. Before answering, the system must know *how* the
# customer feels — a frustrated customer gets a more apologetic, priority-flagged
# handling than a neutral one.
#
# **Dataset:** [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion)
# — ~20k English messages labelled with **6 emotions**:
# `sadness, joy, love, anger, fear, surprise`.
#
# **Model:** a small from-scratch **bidirectional LSTM (RNN)** over a trainable
# word-embedding layer, trained on the 6 raw emotions then collapsed onto the
# **3 tone buckets** routing needs:
# `negative` (sadness/anger/fear) · `neutral` (surprise) · `positive` (joy/love).
#
# This notebook is **fully standalone**: no project files, no `support_bot`
# package, no Drive folder. It downloads its dataset from HuggingFace and saves a
# portable **checkpoint** (`state_dict` + vocab + config) into the shared
# artifact folder that notebook 04 loads to rebuild the exact same model.
#
# > **Domain-shift note (important for the viva):** the emotion corpus is
# > *Twitter* text, not customer support. We therefore end with a **qualitative
# > check** on hand-written support-style messages, exactly as the brief suggests.
#
# **Deliverable:** `sentiment_checkpoint.pt` + `sentiment_meta.json`.

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
# The emotion-label order and the emotion → tone map (defined inline, no package).

# %%
import json
import math
import random
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

DATASET_EMOTION = "dair-ai/emotion"
EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
EMOTION_TO_TONE = {
    "sadness": "negative", "anger": "negative", "fear": "negative",
    "joy": "positive", "love": "positive", "surprise": "neutral",
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


print("tone map:", EMOTION_TO_TONE)

# %% [markdown]
# ## 1. Load & align the dataset
#
# `dair-ai/emotion` exposes two configs; we take the pre-split one
# (`train/validation/test`). Labels are ids in corpus class order.

# %%
from datasets import Dataset, load_dataset

if SMOKE:                          # streaming -> fetch only what we need
    def smoke_split(name, n):
        it = load_dataset(DATASET_EMOTION, "split", split=name, streaming=True)
        rows = list(it.take(n))
        return Dataset.from_dict({"text": [r["text"] for r in rows],
                                  "label": [r["label"] for r in rows]})

    ds = {"train": smoke_split("train", 2500),
          "validation": smoke_split("validation", 500),
          "test": smoke_split("test", 1000)}
else:
    try:
        dd = load_dataset(DATASET_EMOTION, "split")
    except Exception:
        dd = load_dataset(DATASET_EMOTION)          # fall back to default config
    if "validation" in dd:
        ds = {k: dd[k] for k in ("train", "validation", "test")}
    elif "test" in dd:
        ds = {"train": dd["train"], "validation": dd["test"], "test": dd["test"]}
    else:
        ds = {"train": dd["train"], "validation": dd["train"], "test": dd["train"]}

for k, v in ds.items():
    print(f"{k:10s} {len(v):6d} rows")

feature_labels = getattr(ds["train"].features.get("label"), "names", None)
print("corpus label order:", feature_labels or EMOTION_LABELS)
labels = (EMOTION_LABELS if (feature_labels is None or feature_labels == EMOTION_LABELS)
          else feature_labels)
label_id = {l: i for i, l in enumerate(labels)}

# %% [markdown]
# ## 2. Class & tone balance
#
# `love` and `surprise` are rare — exactly why the robust 3-bucket tone signal
# (used for routing) is preferable to trusting the rarest emotion classes.

# %%
tr = ds["train"].to_pandas()
tr["label_name"] = tr["label"].map(lambda i: labels[i])
tr["tone"] = tr["label_name"].map(lambda e: EMOTION_TO_TONE[e])
print(tr[["text", "label_name", "tone"]].head(4).to_string())
print("\nemotion distribution:\n", tr["label_name"].value_counts().to_string())
print("\ntone distribution:\n", tr["tone"].value_counts().to_string())

# %% [markdown]
# ## 3. Tokenise & build vocabulary
#
# Words are lower-cased alphanumeric runs, truncated to `MAX_LEN`. The vocabulary
# is built **only from the training set** with `<unk>` and `<pad>`.

# %%
PAD, UNK = "<pad>", "<unk>"
VOCAB_SIZE = 20000
MAX_LEN = 40


def build_vocab(texts, max_vocab: int = VOCAB_SIZE):
    from collections import Counter
    counter = Counter()
    for t in texts:
        counter.update(re.findall(r"[^\W_]+", str(t).lower()))
    words = [w for w, _ in counter.most_common(max_vocab)]
    vocab = {PAD: 0, UNK: 1}
    vocab.update({w: i for i, w in enumerate(words, start=2)})
    return vocab


def encode(text: str, vocab: dict, max_len: int):
    unk = vocab.get(UNK, 1)
    toks = re.findall(r"[^\W_]+", str(text).lower())[:max_len]
    return [vocab.get(w, unk) for w in toks]


train_texts = [clean(t) for t in ds["train"]["text"]]
vocab = build_vocab(train_texts)
print("vocab size:", len(vocab), "| top tokens:", list(vocab.items())[:5], "…")


def make_dataset(hf):
    xs = [encode(clean(t), vocab, MAX_LEN) for t in hf["text"]]
    ys = [label_id[labels[i]] for i in hf["label"]]
    return list(zip(xs, ys))


train_data = make_dataset(ds["train"])
valid_data = make_dataset(ds["validation"])
test_data = make_dataset(ds["test"])
print("example (token ids):", train_data[0][0][:12],
      "| label", labels[train_data[0][1]])

# %% [markdown]
# ## 4. DataLoader + collate
#
# Each batch is padded to the batch's own max length with the `<pad>` id (0).
# Padding is masked later in the mean-pooling step.

# %%
PAD_ID = vocab["<pad>"]


def collate(batch):
    xs, ys = zip(*batch)
    maxlen = max(len(x) for x in xs)
    X = torch.full((len(batch), maxlen), PAD_ID, dtype=torch.long)
    for i, x in enumerate(xs):
        X[i, :len(x)] = torch.tensor(x, dtype=torch.long)
    return X, torch.tensor(ys, dtype=torch.long)


BATCH_SIZE = 64
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE,
                          shuffle=True, collate_fn=collate)
valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE,
                          shuffle=False, collate_fn=collate)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE,
                         shuffle=False, collate_fn=collate)
print("train batches:", len(train_loader))

# %% [markdown]
# ## 5. Model architecture
#
# ```text
# input ids (B,T)
#   -> Embedding(vocab, 128, pad_idx=0)      trainable word vectors
#   -> BiLSTM(128 -> 128, layers=2)          both directions
#   -> masked mean pooling over time         (variable length)
#   -> Dropout(0.4) -> Linear(256 -> 6)      emotion logits
# ```
#
# * **Bidirectional** — a sentiment word's meaning often depends on words after
#   it ("not happy").
# * **Masked mean pooling** — robust to variable-length padded inputs; padding
#   positions contribute nothing.

# %%
class EmotionBiLSTM(nn.Module):
    """2-layer BiLSTM over learned word embeddings + masked mean-pooling."""

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
        emb = self.embedding(x)                       # (B, T, E)
        out, _ = self.lstm(emb)                       # (B, T, 2H)
        mask = (x != self.pad_idx).unsqueeze(-1).float()
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)                # (B, C) logits


model = EmotionBiLSTM(vocab_size=len(vocab), embed_dim=128, hidden_dim=128,
                      num_layers=2, num_classes=len(labels), dropout=0.4,
                      pad_idx=PAD_ID, max_len=MAX_LEN)
print(model)
print("\ntrainable parameters:", sum(p.numel() for p in model.parameters()))

# %% [markdown]
# ## 6. Training loop
#
# Cross-entropy + Adam. After each epoch we evaluate on validation and keep the
# **best** weights; the final model is evaluated once on the held-out test set.

# %%
EPOCHS = 2 if SMOKE else 6
LR = 1e-3

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


def run_epoch(model, loader, train=True):
    model.train() if train else model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for Xb, yb in loader:
            logits = model(Xb)
            loss = criterion(logits, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            n += len(yb)
    return total_loss / n, correct / n


best_acc, best_state = 0.0, None
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = run_epoch(model, train_loader, train=True)
    va_loss, va_acc = run_epoch(model, valid_loader, train=False)
    print(f"epoch {epoch:2d}/{EPOCHS} | train loss {tr_loss:.4f} acc {tr_acc:.4f}"
          f" | val loss {va_loss:.4f} acc {va_acc:.4f}")
    if va_acc > best_acc:
        best_acc = va_acc
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

if best_state is not None:
    model.load_state_dict(best_state)
print(f"best validation accuracy: {best_acc:.4f}")

# %% [markdown]
# ## 7. Held-out evaluation
#
# Report three numbers: 6-class accuracy (raw emotions), **3-tone accuracy** (the
# signal actually consumed downstream), and the per-tone report (is `negative`
# — the most safety-critical — caught reliably?).

# %%
model.eval()
pred_emotions, true_labels = [], []
with torch.no_grad():
    for Xb, yb in test_loader:
        logits = model(Xb)
        pred_emotions += [labels[i] for i in logits.argmax(1).tolist()]
        true_labels += [labels[i] for i in yb.tolist()]

te_acc6 = float(np.mean([p == t for p, t in zip(pred_emotions, true_labels)]))
true_tones = [EMOTION_TO_TONE[t] for t in true_labels]
pred_tones = [EMOTION_TO_TONE[p] for p in pred_emotions]
te_acc3 = float(np.mean([a == b for a, b in zip(pred_tones, true_tones)]))
print(f"test accuracy (6 emotions): {te_acc6:.4f}")
print(f"test accuracy (3 tones)   : {te_acc3:.4f}")

print("\n--- per-tone report (what routing sees) ---")
print(classification_report(true_tones, pred_tones, digits=3, zero_division=0))

print("\n--- confusion matrix (emotions, true rows) ---")
cm = confusion_matrix(true_labels, pred_emotions, labels=labels)
print(pd.DataFrame(cm, index=labels, columns=labels).to_string())

# %% [markdown]
# ## 8. Save a portable checkpoint
#
# Unlike a whole-model pickle, we save the **state dict + config + vocab** so
# notebook 04 can rebuild the identical architecture and load the weights — even
# if it runs in a fresh kernel of the same Colab runtime.

# %%
ckpt = {
    "model_state": {k: v.cpu().clone() for k, v in model.state_dict().items()},
    "model_cfg": dict(model.cfg),
    "labels": list(labels),
    "tone_map": dict(EMOTION_TO_TONE),
    "vocab": dict(vocab),
}
torch.save(ckpt, ARTIFACT_DIR / "sentiment_checkpoint.pt")

meta = {"labels": list(labels), "tone_map": dict(EMOTION_TO_TONE),
        "model_cfg": dict(model.cfg), "test_acc6": te_acc6, "test_acc3": te_acc3}
(ARTIFACT_DIR / "sentiment_meta.json").write_text(
    json.dumps(meta, indent=2), encoding="utf-8")
print("saved:", ARTIFACT_DIR / "sentiment_checkpoint.pt")
print("saved:", ARTIFACT_DIR / "sentiment_meta.json")

# %% [markdown]
# ## 9. Qualitative check on *customer-support* text
#
# The domain-shift reality check the brief asks for: the model was trained on
# Twitter; here we eyeball its predictions on realistic e-commerce messages.

# %%
def predict_emotion_batch(model, messages, batch_size=64):
    """Predict emotion name + confidence for a list of messages."""
    model.eval()
    pad_idx = model.cfg["pad_idx"]
    max_len = model.cfg["max_len"]
    out_names, out_probs = [], []
    with torch.no_grad():
        for i in range(0, len(messages), batch_size):
            batch = messages[i:i + batch_size]
            X = torch.full((len(batch), max_len), pad_idx, dtype=torch.long)
            for j, msg in enumerate(batch):
                ids = encode(clean(msg), vocab, MAX_LEN)
                X[j, :len(ids)] = torch.tensor(ids[:max_len], dtype=torch.long)
            logits = model(X)
            probs = torch.softmax(logits, dim=-1)
            for p in probs:
                k = int(p.argmax().item())
                out_names.append(labels[k])
                out_probs.append(float(p[k].item()))
    return out_names, out_probs


support_samples = [
    "I still have not received my order and nobody is answering, this is ridiculous!!!",
    "Just got my refund, thank you so much for the quick help :)",
    "Could you please tell me which delivery options are available for my area?",
    "I want to cancel my order, but the website keeps crashing and I am really stressed.",
    "The new shoes finally arrived and they are perfect, very happy!",
    "Can you tell me what my account number is please?",
]

emotions, confs = predict_emotion_batch(model, support_samples)
for s, e, cf in zip(support_samples, emotions, confs):
    tone = EMOTION_TO_TONE[e]
    print(f"[{tone:8s} | {e:7s} {cf:.2f}] {s[:70]}")

# %% [markdown]
# ## Summary
#
# * Stage-2 module: a **from-scratch BiLSTM** emotion classifier trained on the 6
#   raw emotion labels and collapsed to the **3 tone buckets** used for
#   tone-aware routing.
# * Saves a portable `sentiment_checkpoint.pt` (+ `sentiment_meta.json`) into the
#   shared artifact folder; notebook **04** rebuilds this exact model.
# * Tone-aware behaviour: a `negative` tone triggers an apology + priority
#   handling before the RAG answer.
#
# Next: notebook **03 — Intent classifier** (traditional ML on gold Bitext intents).
