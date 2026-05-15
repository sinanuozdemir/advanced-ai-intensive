"""Real-internet research tools.

Two LangChain ``@tool`` callables that the research agent uses to actually
hit the web:

* ``serpapi_search`` — Google results via https://serpapi.com (requires
  ``SERPAPI_API_KEY``).
* ``firecrawl_scrape`` — clean markdown extraction via
  https://api.firecrawl.dev (requires ``FIRECRAWL_API_KEY``). Falls back to
  a naive ``httpx`` + HTML-strip when the Firecrawl key is missing so the
  workflow still produces something useful in dev.

Both tools return strings (the agent loop is happier with strings than
dicts), and they never raise — failure modes are encoded in the returned
text so the LLM can recover ("ERROR: ..." prefix).
"""

from __future__ import annotations

import os
import re
from typing import Any

import httpx
from langchain_core.tools import tool


SERPAPI_URL = "https://serpapi.com/search.json"
FIRECRAWL_URL = "https://api.firecrawl.dev/v1/scrape"

# Cap per-fetch payload so a giant page can't blow up the agent's context.
_MAX_BODY_CHARS = 8000
_DEFAULT_TIMEOUT_S = 30.0


# ---------------------------------------------------------------------------
# SerpAPI
# ---------------------------------------------------------------------------


@tool
def serpapi_search(query: str, k: int = 5) -> str:
    """Search Google via SerpAPI and return the top-k organic results.

    Each result is one line ``[url] title — snippet``. The agent should
    follow up with ``firecrawl_scrape`` on the URLs that look most
    promising before quoting them.

    Args:
        query: A focused, self-contained search query.
        k: Maximum number of results (1-10).
    """
    api_key = os.environ.get("SERPAPI_API_KEY", "").strip()
    if not api_key:
        return (
            "ERROR: SERPAPI_API_KEY is not set on the server. "
            "Tell the user that the deep-research API is mis-configured "
            "and stop trying to call this tool."
        )
    k = max(1, min(int(k or 5), 10))
    params = {
        "engine": "google",
        "q": query,
        "num": k,
        "api_key": api_key,
    }
    try:
        with httpx.Client(timeout=_DEFAULT_TIMEOUT_S) as client:
            resp = client.get(SERPAPI_URL, params=params)
    except httpx.HTTPError as exc:
        return f"ERROR: serpapi request failed: {type(exc).__name__}: {exc}"
    if resp.status_code != 200:
        return f"ERROR: serpapi HTTP {resp.status_code}: {resp.text[:300]}"
    try:
        data: dict[str, Any] = resp.json()
    except ValueError:
        return f"ERROR: serpapi returned non-JSON: {resp.text[:300]}"
    if "error" in data:
        return f"ERROR: serpapi error: {data['error']}"
    organic = data.get("organic_results") or []
    if not organic:
        return "No organic results."
    lines: list[str] = []
    for i, hit in enumerate(organic[:k], 1):
        url = hit.get("link") or ""
        title = (hit.get("title") or "").strip()
        snippet = (hit.get("snippet") or "").replace("\n", " ").strip()
        if not url:
            continue
        lines.append(f"{i}. [{url}] {title} — {snippet}")
    return "\n".join(lines) if lines else "No organic results."


# ---------------------------------------------------------------------------
# Firecrawl (with naive httpx fallback)
# ---------------------------------------------------------------------------


@tool
def firecrawl_scrape(url: str) -> str:
    """Fetch a URL and return its main content as readable text.

    Uses Firecrawl when ``FIRECRAWL_API_KEY`` is set (better extraction,
    handles JS-rendered pages); falls back to a naive httpx GET + HTML
    strip otherwise. Content is capped at ~8k chars so one giant page
    doesn't dominate the context window.

    Args:
        url: The URL to fetch. Should be one returned by ``serpapi_search``.
    """
    url = (url or "").strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        return f"ERROR: refusing to fetch non-http(s) url: {url!r}"

    api_key = os.environ.get("FIRECRAWL_API_KEY", "").strip()
    if api_key:
        text = _firecrawl_fetch(url, api_key)
    else:
        text = _naive_fetch(url)
    return _truncate(text, _MAX_BODY_CHARS)


def _firecrawl_fetch(url: str, api_key: str) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "url": url,
        "formats": ["markdown"],
        "onlyMainContent": True,
    }
    try:
        with httpx.Client(timeout=_DEFAULT_TIMEOUT_S) as client:
            resp = client.post(FIRECRAWL_URL, headers=headers, json=body)
    except httpx.HTTPError as exc:
        return f"ERROR: firecrawl request failed: {type(exc).__name__}: {exc}"
    if resp.status_code != 200:
        return f"ERROR: firecrawl HTTP {resp.status_code}: {resp.text[:300]}"
    try:
        payload = resp.json()
    except ValueError:
        return f"ERROR: firecrawl returned non-JSON: {resp.text[:300]}"
    if not payload.get("success", False):
        return f"ERROR: firecrawl unsuccessful: {payload}"
    data = payload.get("data") or {}
    md = data.get("markdown") or data.get("content") or ""
    title = ((data.get("metadata") or {}).get("title") or "").strip()
    if not md.strip():
        return f"ERROR: firecrawl returned empty content for {url}"
    head = f"{title}\n\n" if title else ""
    return head + md


def _naive_fetch(url: str) -> str:
    """No-API-key fallback. Strips HTML tags and collapses whitespace.

    Good enough to extract paragraphs from static pages; will return very
    little useful text from JS-heavy sites.
    """
    try:
        with httpx.Client(
            timeout=_DEFAULT_TIMEOUT_S,
            follow_redirects=True,
            headers={"User-Agent": "agent_api/0.2 (+research-workflow)"},
        ) as client:
            resp = client.get(url)
    except httpx.HTTPError as exc:
        return f"ERROR: naive fetch failed: {type(exc).__name__}: {exc}"
    if resp.status_code != 200:
        return f"ERROR: naive fetch HTTP {resp.status_code} for {url}"
    ct = resp.headers.get("content-type", "")
    if "html" not in ct and "text" not in ct:
        return f"ERROR: non-text content-type {ct!r} for {url}"
    html = resp.text
    # Drop script/style/nav/footer blocks before tag-stripping.
    html = re.sub(r"(?is)<(script|style|nav|footer|header|aside)[^>]*>.*?</\1>", " ", html)
    text = re.sub(r"(?is)<[^>]+>", " ", html)
    text = re.sub(r"&(nbsp|#160);", " ", text)
    text = re.sub(r"&(amp|#38);", "&", text)
    text = re.sub(r"&(lt|#60);", "<", text)
    text = re.sub(r"&(gt|#62);", ">", text)
    text = re.sub(r"&(quot|#34);", '"', text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return f"ERROR: naive fetch produced no text from {url}"
    return text


def _truncate(text: str, n: int) -> str:
    if len(text) <= n:
        return text
    return text[: n - 3].rstrip() + "..."


RESEARCH_TOOLS = [serpapi_search, firecrawl_scrape]


__all__ = [
    "RESEARCH_TOOLS",
    "serpapi_search",
    "firecrawl_scrape",
]
