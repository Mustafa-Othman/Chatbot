"""Sentiment module: model class + tokeniser + save/load/predict.

The RNN is a small from-scratch BiLSTM over a trainable embedding layer.
It is defined HERE (not only inside a notebook) so that
  1) the notebook trains the *same* object that gets pickled, and
  2) the Flask app can `torch.load` the saved model without any
     re-definition drift between training and serving.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from torch import nn

from .config import EMOTION_LABELS, EMOTION_TO_TONE, SENT_DIR, SENT_META, SENT_MODEL
from .textproc import clean

PAD, UNK = "<pad>", "<unk>"


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
class EmotionBiLSTM(nn.Module):
    """2-layer BiLSTM over learned word embeddings + mean-pooled sequence
    representation fed to a linear classifier.

    Mean pooling over *masked* timesteps (rather than just taking the last
    hidden state) makes the model robust to the variable-length, padded
    batches produced by the emotion data.
    """

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
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
                            bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        # bidirectional -> 2 * hidden_dim
        self.classifier = nn.Linear(2 * hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None):
        emb = self.embedding(x)                       # (B, T, E)
        out, _ = self.lstm(emb)                       # (B, T, 2H)
        mask = (x != self.pad_idx).unsqueeze(-1).float()
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)                # (B, C) logits


# ---------------------------------------------------------------------------
# Vocabulary helpers
# ---------------------------------------------------------------------------
def build_vocab(texts, max_vocab: int = 20000):
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
    ids = [vocab.get(w, unk) for w in toks]
    return ids


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_sentiment(model, vocab, labels, tone_map, dest_dir: Path | None = None):
    """Save the full model object + a small meta json (vocab/labels/map)."""
    dest_dir = dest_dir or SENT_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model, dest_dir / SENT_MODEL.name)      # full object, incl. cfg
    meta = {
        "vocab": vocab,
        "labels": labels,                              # class names in output order
        "tone_map": tone_map,
        "model_cfg": model.cfg,
        "model_class": "EmotionBiLSTM",
    }
    (dest_dir / SENT_META.name).write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


def load_sentiment(dest_dir: Path | None = None):
    dest_dir = dest_dir or SENT_DIR
    model = torch.load(dest_dir / SENT_MODEL.name, map_location="cpu", weights_only=False)
    meta = json.loads((dest_dir / SENT_META.name).read_text(encoding="utf-8"))
    return model, meta


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def predict_emotion(model, meta, messages, batch_size: int = 64):
    """Batch predict emotion name (and confidence) for a list of messages."""
    model.eval()
    vocab, labels, max_len = meta["vocab"], meta["labels"], model.cfg["max_len"]
    pad_idx = model.cfg["pad_idx"]
    out_names, out_probs = [], []
    with torch.no_grad():
        for i in range(0, len(messages), batch_size):
            batch = messages[i:i + batch_size]
            X = torch.full((len(batch), max_len), pad_idx, dtype=torch.long)
            for j, msg in enumerate(batch):
                ids = encode(clean(msg), vocab, max_len)
                X[j, :len(ids)] = torch.tensor(ids[:max_len], dtype=torch.long)
            logits = model(X)
            probs = torch.softmax(logits, dim=-1)
            for p in probs:
                k = int(p.argmax().item())
                out_names.append(labels[k])
                out_probs.append(float(p[k].item()))
    return out_names, out_probs


def tone_of_emotion(emotion: str, tone_map: dict | None = None) -> str:
    """Map a 6-class emotion onto the 3 tone buckets used for routing."""
    tone_map = tone_map or EMOTION_TO_TONE
    return tone_map.get(emotion, "neutral")
