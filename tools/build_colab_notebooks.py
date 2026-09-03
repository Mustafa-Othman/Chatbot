"""Build the STANDALONE (Colab) notebooks from their `# %%` sources.

Sources live in notebooks_colab/_src/*.py and are converted to .ipynb in
notebooks_colab/. Unlike notebooks/_src, these notebooks must NOT import the
local `support_bot` package — they are meant to be uploaded to Google Colab and
run there with only the .ipynb file present.

Run:  python tools/build_colab_notebooks.py [name ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks_colab" / "_src"
OUT = ROOT / "notebooks_colab"


def parse_source(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    cells: list[tuple[str, str]] = []
    cur_lines: list[str] = []
    cur_type = None

    def flush():
        nonlocal cur_lines
        if cur_type is None:
            return
        if cur_type == "markdown":
            body = []
            for ln in cur_lines:
                if ln.startswith("# "):
                    body.append(ln[2:])
                elif ln == "#":
                    body.append("")
                else:
                    body.append(ln)
            src = "\n".join(body).strip()
        else:
            src = "\n".join(cur_lines).strip("\n")
        if src.strip():
            cells.append((cur_type, src))
        cur_lines = []

    for ln in lines:
        if ln.startswith("# %%"):
            flush()
            marker = ln[4:].strip()
            cur_type = "markdown" if marker.startswith("[markdown]") else "code"
            continue
        cur_lines.append(ln)
    flush()
    return cells


def build_one(name: str):
    src = SRC / f"{name}.py"
    cells = parse_source(src)
    nb = nbf.v4.new_notebook()
    nb.cells = []
    for ctype, src_ in cells:
        if ctype == "markdown":
            nb.cells.append(nbf.v4.new_markdown_cell(src_))
        else:
            nb.cells.append(nbf.v4.new_code_cell(src_))
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.14"}
    out = OUT / f"{name}.ipynb"
    # write through an explicit UTF-8 handle (Windows locale-safe)
    with open(out, "w", encoding="utf-8") as fh:
        nbf.write(nb, fh)
    print(f"wrote {out.relative_to(ROOT)}  ({len(nb.cells)} cells)")


if __name__ == "__main__":
    names = sys.argv[1:] or [
        "01_language_detection", "02_sentiment_emotion",
        "03_intent_classifier", "04_rag_pipeline",
    ]
    for n in names:
        build_one(n)
