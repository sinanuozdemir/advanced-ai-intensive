"""Retrieval primitives used across notebooks 1-5.

- `dense_search` — Chroma + MiniLM (the simple-RAG baseline).
- `bm25_search` — sparse retrieval over the same corpus.
- `HybridRetriever` — RRF-fused dense + sparse retrieval.
- `CrossEncoderReranker` — second-stage reranking on top of any candidate list.
- `filter_by_source` — metadata-aware retrieval helper.

Notebooks generally do:

    chroma = load_chroma(); bm25 = load_bm25()
    retriever = HybridRetriever(chroma, bm25, k=10)
    docs = retriever.search("what is RAG?")
    docs = CrossEncoderReranker().rerank(query, docs, top_k=4)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Dense search (Chroma + MiniLM)
# ---------------------------------------------------------------------------

def dense_search(
    chroma: Any,
    query: str,
    *,
    k: int = 10,
    where: dict[str, Any] | None = None,
) -> list[Document]:
    """Top-k dense retrieval against a Chroma vector store.

    `where` is a Chroma metadata filter, e.g. ``{"source": "wikipedia"}``.
    """
    return chroma.similarity_search(query, k=k, filter=where)


# ---------------------------------------------------------------------------
# Sparse search (BM25 over the same corpus)
# ---------------------------------------------------------------------------

def bm25_search(
    bm25_payload: dict,
    query: str,
    *,
    k: int = 10,
    where: dict[str, Any] | None = None,
) -> list[Document]:
    """Top-k BM25 retrieval. `bm25_payload` is what `corpus.load_bm25()` returns."""
    bm25 = bm25_payload["bm25"]
    documents: list[Document] = bm25_payload["documents"]

    tokens = _tokenize(query)
    scores = bm25.get_scores(tokens)

    pairs: list[tuple[float, Document]] = list(zip(scores, documents))
    if where:
        pairs = [(s, d) for s, d in pairs if _metadata_match(d.metadata, where)]
    pairs.sort(key=lambda pair: pair[0], reverse=True)
    return [d for _, d in pairs[:k]]


def _tokenize(text: str) -> list[str]:
    return [t for t in text.lower().split() if t.isalnum() or any(c.isalnum() for c in t)]


def _metadata_match(meta: dict[str, Any], where: dict[str, Any]) -> bool:
    return all(meta.get(k) == v for k, v in where.items())


def filter_by_source(docs: Iterable[Document], source: str) -> list[Document]:
    """Keep only documents whose ``metadata['source']`` matches."""
    return [d for d in docs if d.metadata.get("source") == source]


# ---------------------------------------------------------------------------
# Hybrid retrieval (RRF fusion)
# ---------------------------------------------------------------------------

@dataclass
class HybridRetriever:
    """RRF-fused dense + BM25 retrieval over the same multi-source corpus.

    Both backends contribute candidates; we fuse their rank lists using
    Reciprocal Rank Fusion (Cormack et al., 2009). RRF is parameter-light and
    works well when the two retrievers disagree strongly on absolute scores.
    """

    chroma: Any
    bm25_payload: dict
    k: int = 10
    rrf_k: int = 60  # standard RRF constant

    def search(
        self,
        query: str,
        *,
        k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[Document]:
        k = k or self.k
        # over-fetch from each side then fuse
        pool = max(k * 3, 20)
        dense_hits = dense_search(self.chroma, query, k=pool, where=where)
        sparse_hits = bm25_search(self.bm25_payload, query, k=pool, where=where)
        return rrf_fuse([dense_hits, sparse_hits], k=k, rrf_k=self.rrf_k)


def rrf_fuse(
    rank_lists: Sequence[Sequence[Document]],
    *,
    k: int = 10,
    rrf_k: int = 60,
) -> list[Document]:
    """Reciprocal Rank Fusion of multiple ranked candidate lists."""
    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}
    for ranks in rank_lists:
        for rank, doc in enumerate(ranks):
            key = _doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            docs.setdefault(key, doc)
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [docs[k_] for k_, _ in ordered[:k]]


def _doc_key(d: Document) -> str:
    cid = d.metadata.get("chunk_id")
    if cid:
        return str(cid)
    return f"{d.metadata.get('source','?')}::{d.metadata.get('title','?')}::{hash(d.page_content) & 0xffffffff:x}"


# ---------------------------------------------------------------------------
# Cross-encoder reranking
# ---------------------------------------------------------------------------

@dataclass
class CrossEncoderReranker:
    """Second-stage reranker. Loads a sentence-transformers cross-encoder once
    and scores ``(query, doc)`` pairs. Default model is the lightweight
    MS MARCO MiniLM, which is fast enough for live demos."""

    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    _model: Any = field(default=None, init=False, repr=False)

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: Sequence[Document],
        *,
        top_k: int | None = None,
    ) -> list[Document]:
        if not candidates:
            return []
        model = self._load()
        pairs = [(query, d.page_content) for d in candidates]
        scores = model.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda p: float(p[0]), reverse=True)
        out = [d for _, d in ranked]
        return out[:top_k] if top_k else out


# ---------------------------------------------------------------------------
# Convenience: build a Chroma index from a list of Documents (used by nb 0)
# ---------------------------------------------------------------------------

def build_chroma_index(
    documents: Sequence[Document],
    *,
    persist_dir,
    collection_name: str,
    embeddings=None,
    reset: bool = True,
):
    """One-shot Chroma builder with persistence. Used by nb 0.

    By default, wipes ``persist_dir`` first so each build is hermetic. This
    avoids the SQLite ``readonly database`` (code 1032) error that occurs
    when a previous Python process / kernel still has a file handle open
    on the same directory.
    """
    import shutil
    from pathlib import Path
    from langchain_chroma import Chroma

    if embeddings is None:
        from corpus import get_embeddings
        embeddings = get_embeddings()

    p = Path(str(persist_dir))
    if reset and p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

    return Chroma.from_documents(
        documents=list(documents),
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(p),
    )


__all__ = [
    "dense_search",
    "bm25_search",
    "HybridRetriever",
    "CrossEncoderReranker",
    "rrf_fuse",
    "filter_by_source",
    "build_chroma_index",
]
