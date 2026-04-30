"""Corpus + index + gold-set loaders.

Notebook 0 builds the artifacts; notebooks 1-5 load them in two lines:

    from corpus import load_chroma, load_bm25, load_gold_set
    chroma = load_chroma()
    bm25 = load_bm25()
    gold = load_gold_set()
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).parent / "data"
CACHE_DIR = DATA_DIR / "corpus_cache"
CHROMA_DIR = DATA_DIR / "chroma_db"
BM25_PATH = DATA_DIR / "bm25_index.pkl"
GOLD_PATH = DATA_DIR / "gold_set.jsonl"

CHROMA_COLLECTION = "advanced_agentic_week1"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Source(str, Enum):
    BEEHIIV = "beehiiv"
    WIKIPEDIA = "wikipedia"
    HOTPOT = "hotpot"


@dataclass
class GoldQuestion:
    id: str
    question: str
    reference_answer: str
    required_sources: list[str]
    required_evidence_ids: list[str] = field(default_factory=list)
    difficulty: str = "medium"
    hop_count: int = 1

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GoldQuestion":
        return cls(
            id=d["id"],
            question=d["question"],
            reference_answer=d["reference_answer"],
            required_sources=list(d.get("required_sources", [])),
            required_evidence_ids=list(d.get("required_evidence_ids", [])),
            difficulty=d.get("difficulty", "medium"),
            hop_count=int(d.get("hop_count", 1)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "reference_answer": self.reference_answer,
            "required_sources": self.required_sources,
            "required_evidence_ids": self.required_evidence_ids,
            "difficulty": self.difficulty,
            "hop_count": self.hop_count,
        }


def get_embeddings():
    """Standard MiniLM embeddings used everywhere — no API key required."""
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def load_chroma():
    """Load the prebuilt Chroma vector store. Run notebook 0 first."""
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"Chroma index not found at {CHROMA_DIR}. "
            "Run notebooks/week1/0_build_corpus.ipynb first."
        )
    from langchain_chroma import Chroma
    return Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


def load_bm25():
    """Load the prebuilt BM25 index. Returns a `BM25Retriever`-compatible dict
    with `bm25` (the rank_bm25 model) and `documents` (list of LangChain Documents)."""
    if not BM25_PATH.exists():
        raise FileNotFoundError(
            f"BM25 index not found at {BM25_PATH}. "
            "Run notebooks/week1/0_build_corpus.ipynb first."
        )
    with open(BM25_PATH, "rb") as f:
        return pickle.load(f)


def load_gold_set() -> list[GoldQuestion]:
    """Load the gold-set questions written by notebook 0."""
    if not GOLD_PATH.exists():
        raise FileNotFoundError(
            f"Gold set not found at {GOLD_PATH}. "
            "Run notebooks/week1/0_build_corpus.ipynb first."
        )
    out: list[GoldQuestion] = []
    with open(GOLD_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(GoldQuestion.from_dict(json.loads(line)))
    return out


def corpus_exists() -> bool:
    """True if all artifacts are present."""
    return CHROMA_DIR.exists() and BM25_PATH.exists() and GOLD_PATH.exists()


def save_bm25(bm25_obj: Any, documents: list[Any]) -> None:
    """Pickle the BM25 model + the parallel documents list (used by nb 0)."""
    BM25_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25_obj, "documents": documents}, f)


def write_gold_set(questions: list[GoldQuestion]) -> None:
    """Write gold set to JSONL (used by nb 0)."""
    GOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GOLD_PATH, "w") as f:
        for q in questions:
            f.write(json.dumps(q.to_dict()) + "\n")


__all__ = [
    "Source",
    "GoldQuestion",
    "DATA_DIR",
    "CACHE_DIR",
    "CHROMA_DIR",
    "BM25_PATH",
    "GOLD_PATH",
    "CHROMA_COLLECTION",
    "EMBEDDING_MODEL",
    "get_embeddings",
    "load_chroma",
    "load_bm25",
    "load_gold_set",
    "corpus_exists",
    "save_bm25",
    "write_gold_set",
]
