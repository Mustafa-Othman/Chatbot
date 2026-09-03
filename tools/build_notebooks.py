"""Build .ipynb notebooks from light-weight `# %%` source files.

Each source file under notebooks/_src/*.py uses:
    # %%              -> start a code cell
    # %% [markdown]   -> start a markdown cell (following '# ' lines = content)

Run:  python tools/build_notebooks.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "_src"
OUT = ROOT / "notebooks"


def parse_source(path: Path):
    """Return list of (type, source_string)."""
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
                    body.append(ln)  # keep any non-comment line as-is
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
    # write through an explicit UTF-8 handle: nbformat.write() with a str path
    # uses the OS locale encoding on Windows and can mangle non-ASCII source.
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
