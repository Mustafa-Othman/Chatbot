"""Import trained models produced by the STANDALONE (Colab) notebooks into the
layout the local Flask app (deploy/app.py + support_bot) expects.

The Colab notebooks save flat files into one artifact folder with a portable
sentiment *checkpoint* (state_dict). The local SupportBot expects:
  models/language/language_pipeline.pkl       + report.json
  models/sentiment/model.pt (whole model)     + meta.json
  models/intent/intent_pipeline.pkl           + route_map.json
  kb_index/faiss.index + meta.pkl + count.txt

This script:
  * finds the artifact folder (supports Colab's nested download layout),
  * copies the language/intent/KB files over the local ones,
  * converts the sentiment checkpoint into a whole-model model.pt (+ meta.json)
    using the local support_bot.sentiment.EmotionBiLSTM class so torch.load
    unpickles it cleanly at serving time.

Run:  python tools/import_colab_artifacts.py  [path-to-artifact-dir]
Default artifact dir: chatbot_artifacts  (or chatbot_artifacts/chatbot_artifacts)
"""
from __future__ import annotations

import json
import pickle
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:      # tools/ is on sys.path, not the project root
    sys.path.insert(0, str(ROOT))

# where SupportBot expects the files
LANG_DIR = ROOT / "models" / "language"
SENT_DIR = ROOT / "models" / "sentiment"
INTENT_DIR = ROOT / "models" / "intent"
KB_DIR = ROOT / "kb_index"

# dataset-id -> (source file, target dir, target name) for plain copies
COPIES = [
    # language
    ("language_pipeline.pkl", LANG_DIR, "language_pipeline.pkl"),
    ("language_report.json", LANG_DIR, "report.json"),
    # intent
    ("intent_pipeline.pkl", INTENT_DIR, "intent_pipeline.pkl"),
    ("route_map.json", INTENT_DIR, "route_map.json"),
    # knowledge-base index
    ("faiss.index", KB_DIR, "faiss.index"),
    ("kb_meta.pkl", KB_DIR, "meta.pkl"),
]


def find_artifact_dir(hint: str | None) -> Path:
    candidates = []
    if hint:
        candidates.append(Path(hint))
    candidates += [ROOT / "chatbot_artifacts" / "chatbot_artifacts",
                   ROOT / "chatbot_artifacts"]
    for c in candidates:
        if (c / "language_pipeline.pkl").exists():
            return c
    raise SystemExit(
        f"Could not find trained artifacts. Looked in: "
        + "; ".join(str(c) for c in candidates))


def main() -> None:
    src = find_artifact_dir(sys.argv[1] if len(sys.argv) > 1 else None)
    print("artifact source:", src)

    # 1) plain copies ------------------------------------------------------
    for fname, target_dir, target_name in COPIES:
        source = src / fname
        if not source.exists():
            raise SystemExit(f"missing {source}  (did notebook 01-04 all run?)")
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target_dir / target_name)
        size_mb = target_dir.joinpath(target_name).stat().st_size / 1e6
        print(f"  copied {fname:24s} -> {target_name:20s} ({size_mb:.1f} MB)")

    # 2) sentiment checkpoint -> whole model.pt -----------------------------
    import torch
    from support_bot.sentiment import EmotionBiLSTM, PAD, UNK

    ckpt = torch.load(src / "sentiment_checkpoint.pt",
                      map_location="cpu", weights_only=False)
    model = EmotionBiLSTM(**ckpt["model_cfg"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    SENT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model, SENT_DIR / "model.pt")            # whole object
    meta = {
        "vocab": ckpt["vocab"],
        "labels": ckpt["labels"],
        "tone_map": ckpt["tone_map"],
        "model_cfg": ckpt["model_cfg"],
        "model_class": "EmotionBiLSTM",
        "PAD": PAD, "UNK": UNK,
    }
    (SENT_DIR / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print(f"  converted sentiment_checkpoint.pt -> model.pt "
          f"(vocab={len(ckpt['vocab'])}, labels={ckpt['labels']})")

    # 3) KB chunk count ------------------------------------------------------
    try:
        with open(KB_DIR / "meta.pkl", "rb") as fh:
            n = len(pickle.load(fh))
        (KB_DIR / "count.txt").write_text(str(n))
        print(f"  kb chunks: {n}")
    except Exception as e:  # noqa: BLE001
        print("  (could not read kb count:", e, ")")

    print("\nDone. Start the API with:\n"
          "    set GROQ_API_KEY=gsk_...\n"
          "    python deploy/app.py\n"
          "  then open http://127.0.0.1:8000")


if __name__ == "__main__":
    main()