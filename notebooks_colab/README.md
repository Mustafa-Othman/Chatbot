# Standalone Colab notebooks

These **four notebooks** are self-contained versions of the four modules (01
language → 02 sentiment → 03 intent → 04 RAG pipeline). They were written so you
can run them on **Google Colab by uploading only the `.ipynb` files** — no
project folder, no `support_bot` package, no Drive mount, no config file.

## How to run on Colab

1. **Download / copy these four files** to your machine:
   `01_language_detection.ipynb`, `02_sentiment_emotion.ipynb`,
   `03_intent_classifier.ipynb`, `04_rag_pipeline.ipynb`.
2. Go to [colab.research.google.com](https://colab.research.google.com) →
   **File → Upload notebook** → pick `01_language_detection.ipynb`.
3. Press **Runtime → Run all**. The first cell auto-installs any missing packages
   and prints the artifact folder it uses.
4. **Run the other three in order, 01 → 02 → 03 → 04**, each with *Run all*.

### ⚠️ The one rule: run 01 → 04 in the same Colab session

Each notebook saves its trained model into a shared **artifact folder** on the
Colab runtime: `/content/chatbot_artifacts`. Notebook 04 **loads the models that
01–03 saved** from that folder. The folder lives on the *ephemeral runtime disk*,
so:

- Keep the Colab runtime alive and connected between the four runs;
- If a runtime is reset, or you open notebook 04 in a fresh session, notebook 04
  stops with a clear message telling you to run 01 (then 02, 03) first;
- Notebooks 01–03 can each be re-run to retrain and overwrite the artifact.

### Optional (recommended) — Groq key for notebook 04

Notebook 04 calls **Groq** to generate final answers **only if** `GROQ_API_KEY`
is present. Easiest way in Colab: left sidebar **🔑 Secrets** → **New secret** →
name it `GROQ_API_KEY`, paste your `gsk_…`, and toggle "Notebook access" on. The
setup cell reads it automatically. Without a key notebook 04 still works — it
returns the best matching knowledge-base answer instead of a generated one.

## What each notebook downloads from HuggingFace (first run)

| notebook | dataset / model | note |
|---|---|---|
| 01 | `papluca/language-identification` | ~90k, 20 languages |
| 02 | `dair-ai/emotion` | ~20k, 6 emotions |
| 03 | `bitext/Bitext-customer-support-llm-chatbot-training-dataset` | gold 27 intents |
| 04 | same Bitext KB + `sentence-transformers/all-MiniLM-L6-v2` embedder | + Groq LLM |

## Quick test vs full run

Every notebook has a `SMOKE` switch. It defaults to `False` (the real full run —
takes a few minutes per notebook and produces the final models). To sanity-check
everything quickly, set `SMOKE = True` in the second code cell (or re-run with
the shell env var `PROJECT_SMOKE=1`) — each notebook then finishes in under a
minute. When you're done testing, set it back to `False` and re-run 01→04 for
the final artifacts.

## Source files

Like the main repo, these notebooks are built from plain-text sources in
`_src/` by `tools/build_colab_notebooks.py` (run `python tools/build_colab_notebooks.py`).
