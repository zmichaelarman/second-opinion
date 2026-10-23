"""Fetch the Token Factory docs pages the checker covers (CLAUDE.md §10.1).

Raw pages are cached under runs/docs-cache/<date>/. A page already cached that day is reused,
never overwritten, so the cache stays append-only like the rest of runs/.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import httpx

DOCS_INDEX = "https://docs.tokenfactory.nebius.com/llms.txt"
OPENAPI_URL = "https://api.tokenfactory.nebius.com/openapi.json"
PAGE_PATTERNS = [
    r"/quickstart\.md$",
    r"/switch\.md$",
    r"/ai-models-inference/(overview|function-calling|json|rate-limits)\.md$",
    r"/cached-tokens\.md$",
    r"/api-reference/examples/.*\.md$",
    r"/api-reference/inference/create-chat-completion\.md$",
]
_URL = re.compile(r"https?://[^\s)\]]+\.md")


@dataclass
class Page:
    url: str
    title: str
    text: str
    path: Path


def select_urls(index_text: str) -> list[str]:
    urls = dict.fromkeys(_URL.findall(index_text))  # dedupe, keep order
    return [u for u in urls if any(re.search(p, u) for p in PAGE_PATTERNS)]


def cache_name(url: str) -> str:
    return urlparse(url).path.strip("/").replace("/", "__") or "index"


def page_title(text: str, url: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return cache_name(url)


async def _get(http: httpx.AsyncClient, url: str, path: Path) -> str:
    if path.exists():
        return path.read_text()
    resp = await http.get(url)
    resp.raise_for_status()
    path.write_text(resp.text)
    return resp.text


async def fetch_docs(
    runs_dir: Path, day: str | None = None, transport: httpx.AsyncBaseTransport | None = None
) -> tuple[Path, list[Page], dict]:
    """Return (cache dir, pages, openapi spec)."""
    cache = runs_dir / "docs-cache" / (day or date.today().isoformat())
    cache.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=30, follow_redirects=True, transport=transport) as http:
        index = await _get(http, DOCS_INDEX, cache / "llms.txt")
        openapi = json.loads(await _get(http, OPENAPI_URL, cache / "openapi.json"))
        pages = []
        for url in select_urls(index):
            path = cache / cache_name(url)
            text = await _get(http, url, path)
            pages.append(Page(url, page_title(text, url), text, path))
    return cache, pages, openapi
