from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any

import httpx

from second_opinion.config import AppConfig, REPO_ROOT

DOC_URL_PATTERNS = [
    re.compile(r"/quickstart\.md$"),
    re.compile(r"/switch\.md$"),
    re.compile(r"/ai-models-inference/(overview|function-calling|json|rate-limits)\.md$"),
    re.compile(r"/cached-tokens\.md$"),
    re.compile(r"/api-reference/examples/.*\.md$"),
    re.compile(r"/api-reference/inference/create-chat-completion\.md$"),
]


def is_matching_doc_url(url: str) -> bool:
    clean = url.strip()
    return any(p.search(clean) for p in DOC_URL_PATTERNS)


def extract_urls_from_llms_txt(llms_txt: str, base_host: str = "https://docs.tokenfactory.nebius.com") -> list[tuple[str, str]]:
    """Return list of (title, url) from llms.txt markdown links and bare URLs."""
    results: list[tuple[str, str]] = []
    seen: set[str] = set()
    # Match markdown links: [Title](url)
    for m in re.finditer(r"\[([^\]]+)\]\(([^)\s]+)\)", llms_txt):
        title, raw_url = m.group(1).strip(), m.group(2).strip()
        url = raw_url if raw_url.startswith("http") else f"{base_host.rstrip('/')}/{raw_url.lstrip('/')}"
        if is_matching_doc_url(url) and url not in seen:
            seen.add(url)
            results.append((title, url))

    # Also match bare URLs
    for m in re.finditer(r"(https?://[^\s)]+\.md)", llms_txt):
        url = m.group(1).strip()
        if is_matching_doc_url(url) and url not in seen:
            seen.add(url)
            results.append((url.rsplit("/", 1)[-1], url))

    return results


def _slugify_url(url: str) -> str:
    path = re.sub(r"^https?://[^/]+/", "", url)
    return re.sub(r"[^A-Za-z0-9._-]+", "__", path)


async def fetch_docs(
    config: AppConfig,
    cache_date: str | None = None,
    runs_root: Path | None = None,
    client: httpx.AsyncClient | None = None,
) -> list[dict[str, Any]]:
    """Fetch llms.txt and matching .md pages, caching under runs/docs-cache/<date>/."""
    root = runs_root or (REPO_ROOT / "runs")
    day_str = cache_date or date.today().isoformat()
    cache_dir = root / "docs-cache" / day_str
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        pages: list[dict[str, Any]] = []
        for entry in manifest.get("pages", []):
            file_path = cache_dir / entry["filename"]
            if file_path.exists():
                pages.append(
                    {
                        "title": entry["title"],
                        "url": entry["url"],
                        "path": file_path,
                        "content": file_path.read_text(encoding="utf-8"),
                    }
                )
        if pages:
            return pages

    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)

    try:
        resp = await client.get(config.docs_index_url)
        resp.raise_for_status()
        llms_txt = resp.text
        (cache_dir / "llms.txt").write_text(llms_txt, encoding="utf-8")

        matched = extract_urls_from_llms_txt(llms_txt)
        pages_meta: list[dict[str, str]] = []
        pages_out: list[dict[str, Any]] = []

        for title, url in matched:
            r_page = await client.get(url)
            if r_page.status_code != 200:
                continue
            fname = _slugify_url(url)
            fpath = cache_dir / fname
            fpath.write_text(r_page.text, encoding="utf-8")
            pages_meta.append({"title": title, "url": url, "filename": fname})
            pages_out.append(
                {
                    "title": title,
                    "url": url,
                    "path": fpath,
                    "content": r_page.text,
                }
            )

        manifest_path.write_text(
            json.dumps({"date": day_str, "pages": pages_meta}, indent=2),
            encoding="utf-8",
        )
        return pages_out
    finally:
        if owns_client and client is not None:
            await client.aclose()

