# Skill: Searching the Heterogeneous Corpus

You have a shell. You can write Python files and run them. Use that to
answer the user's question by searching a pre-built corpus.

## What's on disk

A heterogeneous text corpus has been chunked and indexed two ways. Both
indexes were built by `notebooks/week1/0_build_corpus.ipynb` and live
under `notebooks/week1/data/`.

The corpus pulls from three sources, tagged in each chunk's metadata
under the key `source`:

- `beehiiv` — newsletter posts (RAG / agent topics).
- `wikipedia` — encyclopedia articles.
- `hotpot` — HotpotQA paragraphs.

### 1. Dense index — Chroma

- Persist directory: `notebooks/week1/data/chroma_db/`
- Collection name: `advanced_agentic_week1`
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (384-dim).

A vector database. You embed your query with the same MiniLM model and
ask Chroma for the top-k nearest chunks by cosine similarity. Strong on
paraphrase / semantic match, weak on rare keywords.

To use:

```python
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
chroma = Chroma(
    collection_name="advanced_agentic_week1",
    embedding_function=emb,
    persist_directory="notebooks/week1/data/chroma_db",
)
docs = chroma.similarity_search("your query", k=10)   # list[Document]
```

### 2. Sparse index — BM25

- Pickle path: `notebooks/week1/data/bm25_index.pkl`
- Loads to: `{"bm25": <rank_bm25.BM25Okapi>, "documents": list[Document]}`

A classic lexical (term-frequency) ranker. Fast, no neural model.
Strong on rare keywords and exact phrasing, weak on paraphrase.

To use:

```python
import pickle
with open("notebooks/week1/data/bm25_index.pkl", "rb") as f:
    payload = pickle.load(f)

bm25 = payload["bm25"]
documents = payload["documents"]

tokens = "your query".lower().split()
scores = bm25.get_scores(tokens)
top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]
bm_hits = [documents[i] for i in top_idx]   # list[Document]
```

### 3. Cross-encoder reranker

A second-stage reranker that scores `(query, doc)` pairs jointly. Use
it to rerank the union of BM25 + dense candidates before answering.

- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`

To use:

```python
from sentence_transformers import CrossEncoder
ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
pairs = [("your query", d.page_content) for d in candidates]
scores = ce.predict(pairs)
ranked = [d for _, d in sorted(zip(scores, candidates), key=lambda p: float(p[0]), reverse=True)]
top5 = ranked[:5]
```

## Standard recipe

1. BM25 search for the user's query (top 10).
2. Dense search for the same query (top 10).
3. Concatenate; dedupe if you like.
4. Cross-encoder rerank the union; keep top 5.
5. Read the top 5 chunk texts.
6. Answer in chat with citations to `metadata["source"]` / `metadata["title"]`.

## Constraints

- The shell's working directory is your scratch dir. The repo root is
available; refer to artifacts via the relative paths above.
- You have at most 12 shell commands. Don't waste them.
- If a script errors, fix it and rerun — don't paste the same code
twice unchanged.
- When you have enough evidence, stop running shell commands and
answer the user.

