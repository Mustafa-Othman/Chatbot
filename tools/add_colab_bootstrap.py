"""One-off helper: insert a Colab self-bootstrap cell at the top of each
notebook source (before its first code cell), then re-build the .ipynb files.

The bootstrap is a no-op when IN_COLAB is False, so local behaviour is
unchanged. Run:  python tools/add_colab_bootstrap.py
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "_src"
NAMES = ["01_language_detection", "02_sentiment_emotion",
         "03_intent_classifier", "04_rag_pipeline"]

BOOTSTRAP = r'''# ===== Colab bootstrap (no-op when running locally) =====================
import os
import subprocess
import sys

IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    # 1) install any libraries Colab is missing (torch/sklearn/pandas ship with it)
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

    # 2) mount Google Drive (where the project files live) + Groq secret
    try:
        from google.colab import drive, userdata  # noqa: F401
        if not os.path.isdir("/content/drive"):
            drive.mount("/content/drive")
        try:
            os.environ.setdefault("GROQ_API_KEY", userdata.get("GROQ_API_KEY"))
        except Exception:
            pass
    except Exception:
        pass

    # 3) locate the project root = folder containing support_bot/ and the task .md
    def _find_proj():
        for _base in ("/content", "/content/drive/MyDrive"):
            if not os.path.isdir(_base):
                continue
            for _dp, _dns, _fns in os.walk(_base):
                if _dp[len(_base):].count(os.sep) > 7:
                    _dns[:] = []
                if "support_bot" in _dns and "Customer_Support_Chatbot_Task.md" in _fns:
                    return _dp
        return None

    _proj = _find_proj()
    if _proj is None:
        # manual fallback: put the whole Chatbot folder at this Drive path
        _proj = "/content/drive/MyDrive/Chatbot"
        print("! Could not auto-detect the project folder -> using", _proj)
    os.chdir(_proj)
    print("Colab project root :", _proj)
'''


def add_to(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if "# ==== Colab bootstrap" in text:
        print(f"already present  {path.name}")
        return False
    lines = text.splitlines(keepends=True)
    # insert a fresh code cell right before the first plain '# %%' code marker
    for i, ln in enumerate(lines):
        if ln.rstrip("\n") == "# %%" and not ln.rstrip("\n").startswith("# %% ["):
            lines.insert(i, "# %%\n")
            lines.insert(i + 1, BOOTSTRAP + "\n")
            break
    else:
        raise RuntimeError(f"no code-cell marker found in {path.name}")
    path.write_text("".join(lines), encoding="utf-8")
    print(f"added bootstrap   {path.name}")
    return True


if __name__ == "__main__":
    for name in NAMES:
        add_to(SRC / f"{name}.py")
    print("\nnow run:  python tools/build_notebooks.py")
