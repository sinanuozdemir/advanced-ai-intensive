"""Build / refresh Forge's codebase RAG index.

Persists two artefacts under ``<repo>/.forge/rag_index/``:

- ``bm25.pkl``       — pickle of ``{"bm25": rank_bm25_obj, "documents": [Document, ...]}``
  (same shape as ``notebooks/week1/corpus.load_bm25()`` output).
- ``chroma/``        — Chroma persistent collection with sentence-transformers
  embeddings.

Invoked via the CLI: ``forge index [--force]``.
"""
from __future__ import annotations

import fnmatch
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import RepoRagConfig
from .paths import ForgePaths


# Source extensions we index. Add to this list to widen the corpus —
# binaries / images / lockfiles are intentionally omitted.
TEXT_EXTS = {
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    ".rb", ".go", ".rs", ".java", ".kt", ".swift", ".c", ".h",
    ".cpp", ".hpp", ".cc", ".cs",
    ".html", ".css", ".scss", ".vue", ".svelte",
    ".md", ".rst", ".txt", ".tex",
    ".toml", ".yaml", ".yml", ".json", ".jsonl", ".ini", ".cfg",
    ".sh", ".bash", ".zsh", ".fish",
    ".sql", ".graphql", ".proto",
    ".tsx", ".dockerfile",
}


@dataclass
class Chunk:
    """One chunked snippet of a source file."""

    chunk_id: str
    rel_path: str
    start_line: int
    end_line: int
    text: str


def _excluded(rel: Path, patterns: list[str]) -> bool:
    s = rel.as_posix()
    for pat in patterns:
        if fnmatch.fnmatch(s, pat) or fnmatch.fnmatch(rel.name, pat):
            return True
        # leading directory match
        parts = s.split("/")
        if any(fnmatch.fnmatch(p, pat) for p in parts):
            return True
    return False


def _iter_text_files(root: Path, excludes: list[str]) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        # Filter dirs in-place so os.walk doesn't descend.
        dirnames[:] = [
            d for d in dirnames
            if not _excluded(Path(dirpath).relative_to(root) / d, excludes)
        ]
        for name in filenames:
            p = Path(dirpath) / name
            rel = p.relative_to(root)
            if _excluded(rel, excludes):
                continue
            ext = p.suffix.lower()
            if ext not in TEXT_EXTS and not (
                name.lower() in {"makefile", "dockerfile", "readme"}
            ):
                continue
            try:
                if p.stat().st_size > 1_000_000:  # skip huge files
                    continue
            except OSError:
                continue
            yield p


def _chunk_file(p: Path, rel: Path, chunk_size: int, overlap: int) -> list[Chunk]:
    try:
        text = p.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    lines = text.splitlines()
    if not lines:
        return []
    chunks: list[Chunk] = []
    # Chunk by character size, but track line ranges for citation.
    cur: list[str] = []
    cur_chars = 0
    start_line = 1

    def _emit(end_line: int) -> None:
        """Append a chunk, deduplicating against the last one. JSON-y files
        with a single huge line can otherwise produce identical IDs when
        the rollback can't escape the giant line; see the chroma duplicate-id
        regression that motivated this guard."""
        chunk_id = f"{rel.as_posix()}#L{start_line}-L{end_line}"
        if chunks and chunks[-1].chunk_id == chunk_id:
            # Replace with the wider chunk so we keep the most context.
            chunks[-1] = Chunk(
                chunk_id=chunk_id, rel_path=rel.as_posix(),
                start_line=start_line, end_line=end_line, text="\n".join(cur),
            )
            return
        chunks.append(Chunk(
            chunk_id=chunk_id, rel_path=rel.as_posix(),
            start_line=start_line, end_line=end_line, text="\n".join(cur),
        ))

    for i, line in enumerate(lines, start=1):
        cur.append(line)
        cur_chars += len(line) + 1
        if cur_chars >= chunk_size:
            end_line = i
            _emit(end_line)
            # Roll back ``overlap`` chars of context for the next chunk —
            # but never enough to leave us at or over the threshold. If the
            # last line is itself ≥ chunk_size we keep zero context and
            # start fresh; otherwise progress would stall and we'd emit a
            # chunk per subsequent line.
            if overlap > 0 and cur:
                back_chars = 0
                back: list[str] = []
                budget = max(0, chunk_size - 1)
                for back_line in reversed(cur):
                    line_size = len(back_line) + 1
                    if back_chars + line_size > budget:
                        # Adding this line would defeat the forward-progress
                        # guarantee. Stop here even if we haven't hit overlap.
                        break
                    back.insert(0, back_line)
                    back_chars += line_size
                    if back_chars >= overlap:
                        break
                cur = back
                cur_chars = back_chars
                start_line = end_line - len(cur) + 1 if cur else end_line + 1
            else:
                cur = []
                cur_chars = 0
                start_line = end_line + 1
    if cur:
        end_line = len(lines)
        # Suppress the trailing emit when it'd duplicate the last in-loop
        # chunk (small files where the threshold fired on the final line).
        if not chunks or chunks[-1].end_line != end_line or chunks[-1].start_line != start_line:
            _emit(end_line)
    return chunks


def _tokenize(text: str) -> list[str]:
    """Same tokenizer as ``notebooks/week1/retrievers.py:_tokenize``."""
    return [
        t for t in text.lower().split()
        if t.isalnum() or any(c.isalnum() for c in t)
    ]


def build_index(*, paths: ForgePaths, cfg: RepoRagConfig, force: bool = False) -> int:
    """Walk the repo, chunk text files, build BM25 + Chroma. Returns chunk count."""
    paths.ensure()
    index_dir = paths.rag_index_dir
    bm25_path = index_dir / "bm25.pkl"
    chroma_dir = index_dir / "chroma"

    if force:
        import shutil
        if chroma_dir.exists():
            shutil.rmtree(chroma_dir)
        if bm25_path.exists():
            bm25_path.unlink()

    from langchain_core.documents import Document
    from rank_bm25 import BM25Okapi

    chunks: list[Chunk] = []
    for path in _iter_text_files(paths.repo_root, cfg.index_excludes):
        rel = path.relative_to(paths.repo_root)
        chunks.extend(_chunk_file(path, rel, cfg.chunk_size, cfg.chunk_overlap))

    # Defensive dedup. _chunk_file already guards against most paths to
    # duplicate IDs, but a future chunker change shouldn't be able to crash
    # Chroma's add() — drop later duplicates and keep the first occurrence.
    seen: set[str] = set()
    deduped: list[Chunk] = []
    for c in chunks:
        if c.chunk_id in seen:
            continue
        seen.add(c.chunk_id)
        deduped.append(c)
    chunks = deduped

    if not chunks:
        # Still write an empty BM25 + Chroma so the server boots cleanly.
        documents: list[Document] = []
        with bm25_path.open("wb") as fh:
            pickle.dump({"bm25": None, "documents": documents}, fh)
        return 0

    documents = [
        Document(
            page_content=c.text,
            metadata={
                "chunk_id": c.chunk_id,
                "rel_path": c.rel_path,
                "start_line": c.start_line,
                "end_line": c.end_line,
            },
        )
        for c in chunks
    ]

    # BM25
    tokenized = [_tokenize(d.page_content) for d in documents]
    bm25 = BM25Okapi(tokenized)
    with bm25_path.open("wb") as fh:
        pickle.dump({"bm25": bm25, "documents": documents}, fh)

    # Chroma
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    client = chromadb.PersistentClient(path=str(chroma_dir))
    coll = client.get_or_create_collection(
        name="repo_rag",
        embedding_function=SentenceTransformerEmbeddingFunction(
            model_name=cfg.embedding_model
        ),
    )
    # Reset on rebuild so we don't accumulate stale chunks.
    try:
        existing_ids = coll.get()["ids"]
        if existing_ids:
            coll.delete(ids=existing_ids)
    except Exception:  # noqa: BLE001
        pass

    batch = 256
    for i in range(0, len(documents), batch):
        ids = [d.metadata["chunk_id"] for d in documents[i : i + batch]]
        docs = [d.page_content for d in documents[i : i + batch]]
        metas = [d.metadata for d in documents[i : i + batch]]
        coll.add(ids=ids, documents=docs, metadatas=metas)

    return len(documents)


def load_index(paths: ForgePaths):
    """Return ``(bm25_payload, chroma_collection, embedding_model)``. Raises
    if no index has been built yet."""
    bm25_path = paths.rag_index_dir / "bm25.pkl"
    chroma_dir = paths.rag_index_dir / "chroma"
    if not bm25_path.is_file() or not chroma_dir.is_dir():
        raise FileNotFoundError(
            f"No Forge RAG index at {paths.rag_index_dir}. Run: forge index"
        )
    with bm25_path.open("rb") as fh:
        bm25_payload = pickle.load(fh)
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    # Read the embedding model from the config at load time so a different
    # process can re-open the collection consistently.
    from .config import load_config
    cfg = load_config(paths)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    coll = client.get_or_create_collection(
        name="repo_rag",
        embedding_function=SentenceTransformerEmbeddingFunction(
            model_name=cfg.repo_rag.embedding_model
        ),
    )
    return bm25_payload, coll, cfg.repo_rag


__all__ = ["build_index", "load_index", "Chunk"]
