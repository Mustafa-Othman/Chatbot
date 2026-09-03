"""Retrieval: FAISS (flat, cosine) index over MiniLM embeddings of the
knowledge base, plus the metadata needed to ground answers.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from .config import EMBEDDER_ID, KB_INDEX, KB_META


def _get_embedder(model_id: str | None = None):
    """Lazy-load the sentence-transformer model (cached after first use)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_id or EMBEDDER_ID)


def embed_texts(model, texts: list[str], batch_size: int = 64):
    """Return a (n, d) float32 matrix of L2-normalised embeddings."""
    vecs = model.encode(list(texts), batch_size=batch_size,
                        show_progress_bar=True, normalize_embeddings=True,
                        convert_to_numpy=True)
    return np.asarray(vecs, dtype=np.float32)


def faiss_cosine_index(vectors: np.ndarray, dimension: int | None = None):
    """IndexFlatIP over L2-normalised vectors == cosine similarity."""
    import faiss
    dim = vectors.shape[1] if dimension is None else dimension
    index = faiss.IndexFlatIP(dim)
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    return index


def save_index(index, metas: list[dict], kb_dir: Path | None = None):
    kb_dir = kb_dir or KB_INDEX.parent
    kb_dir.mkdir(parents=True, exist_ok=True)
    import faiss
    faiss.write_index(index, str(kb_dir / KB_INDEX.name))
    with open(kb_dir / KB_META.name, "wb") as fh:
        pickle.dump(metas, fh)


def load_index(kb_dir: Path | None = None):
    kb_dir = kb_dir or KB_INDEX.parent
    import faiss
    index = faiss.read_index(str(kb_dir / KB_INDEX.name))
    with open(kb_dir / KB_META.name, "rb") as fh:
        metas = pickle.load(fh)
    return index, metas


def search(index, metas: list[dict], query_vec: np.ndarray, top_k: int = 3):
    """Return top_k [(score, meta)] using inner product on normalised vectors."""
    if query_vec.ndim == 1:
        query_vec = query_vec.reshape(1, -1)
    scores, idxs = index.search(np.ascontiguousarray(query_vec, dtype=np.float32), top_k)
    results = []
    for sc, i in zip(scores[0], idxs[0]):
        if i < 0 or i >= len(metas):
            continue
        results.append((float(sc), metas[int(i)]))
    return results
