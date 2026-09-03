"""Light-weight text pre-processing shared by all four modules.

One small, predictable clean() is better than four bespoke pipelines --
the same function is used at training time (in the notebooks) and at
serving time (in the app), which is what keeps the deployed behaviour
identical to the evaluated one.
"""
from __future__ import annotations

import re

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_HANDLE_RE = re.compile(r"@\w+")
_WS_RE = re.compile(r"\s+")
_NUM_RE = re.compile(r"\b\d[\d,.]*\b")  # prices/ids/order numbers


def clean(text: str,
          lowercase: bool = True,
          strip_urls: bool = True,
          strip_handles: bool = False,
          strip_numbers: bool = False) -> str:
    """Normalise a raw customer message.

    We deliberately do NOT remove punctuation/accents globally: several
    modules use character n-grams where punctuation carries signal, and
    accents separate languages (e.g. Spanish "información" vs a typo).
    """
    if text is None:
        return ""
    s = str(text)
    if lowercase:
        s = s.lower()
    if strip_urls:
        s = _URL_RE.sub(" ", s)
        s = _EMAIL_RE.sub(" ", s)
    if strip_handles:
        s = _HANDLE_RE.sub(" ", s)
    if strip_numbers:
        s = _NUM_RE.sub(" ", s)
    s = s.replace("\r", " ").replace("\n", " ")
    s = _WS_RE.sub(" ", s)
    return s.strip()


def tokenize_words(text: str, max_len: int | None = None) -> list[str]:
    """Word tokenizer used by the RNN module (letter/digit runs)."""
    toks = re.findall(r"[^\W_]+", text.lower())
    if max_len is not None:
        toks = toks[:max_len]
    return toks
