"""FastMCP server exposing the three retrieval primitives.

Run as a standalone process:

    python -m mcp_demo.server                  # stdio transport (default)
    python -m mcp_demo.server --transport sse  # HTTP/SSE transport on :8765

The `via_mcp` client connects to this process to demonstrate the same
three primitives reachable over an MCP stdio session — the "MCP" column
in segment 1's four-quadrant slide.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from mcp_demo import tools as _tools  # noqa: E402

mcp_server = FastMCP("week2-retriever")


@mcp_server.tool()
def bm25_search(query: str, k: int = 10) -> list[dict]:
    """Lexical BM25 search over the corpus.

    Args:
        query: A focused, self-contained query.
        k: Number of hits to return (default 10).

    Returns:
        List of hit dicts: {id, source, title, text, score}.
    """
    return _tools.bm25_search(query, k=k)


@mcp_server.tool()
def dense_search(query: str, k: int = 10) -> list[dict]:
    """Dense (MiniLM-embedded) vector search over the corpus.

    Args:
        query: A focused, self-contained query.
        k: Number of hits to return (default 10).

    Returns:
        List of hit dicts: {id, source, title, text, score}.
    """
    return _tools.dense_search(query, k=k)


@mcp_server.tool()
def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """Cross-encoder rerank a candidate list (deduped by id).

    Args:
        query: The original query.
        candidates: Hits returned by bm25_search/dense_search (concatenated).
        top_k: Number of reranked hits to keep (default 5).

    Returns:
        The top_k hits sorted by cross-encoder score.
    """
    return _tools.rerank(query, candidates, top_k=top_k)


@mcp_server.tool()
def health() -> dict:
    """Tiny health-check tool. Useful for client smoke tests."""
    return {"ok": True, "tools": ["bm25_search", "dense_search", "rerank", "health"]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="MCP transport. stdio = subprocess pipe (default); sse = HTTP/SSE on :8765",
    )
    args = p.parse_args()
    if args.transport == "sse":
        mcp_server.run(transport="sse")
    else:
        mcp_server.run()


if __name__ == "__main__":
    main()
