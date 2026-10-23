from __future__ import annotations

from typing import Any

import httpx

from second_opinion.config import AppConfig, get_nebius_api_key

EXCLUDED_SUBSTRINGS = ("vl", "omni", "safety", "embed", "rerank")

FALLBACK_NEMOTRON_MODELS = [
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
    "nvidia/Nemotron-3_5-Lightning",
    "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/Nemotron-3-Ultra-550b-a55b",
]


def is_eligible_nemotron_model(model_id: str) -> bool:
    """Discovery rule (§3): include IDs containing 'nemotron' (case-insensitive);
    exclude IDs containing VL, Omni, Safety, embed or rerank.
    """
    lower = model_id.lower()
    if "nemotron" not in lower:
        return False
    for excl in EXCLUDED_SUBSTRINGS:
        if excl in lower:
            return False
    return True


def filter_nemotron_models(model_ids: list[str]) -> list[str]:
    return sorted([m for m in model_ids if is_eligible_nemotron_model(m)], key=lambda s: s.lower())


async def fetch_all_models(
    config: AppConfig,
    api_key: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> list[dict[str, Any]]:
    key = api_key or get_nebius_api_key()
    url = f"{config.nebius_base_url.rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {key}"}
    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=config.connect_timeout_s, read=30.0, write=10.0, pool=10.0)
        )
    try:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        body = resp.json()
        data = body.get("data") if isinstance(body, dict) else []
        return [m for m in (data or []) if isinstance(m, dict) and "id" in m]
    finally:
        if owns_client and client is not None:
            await client.aclose()


async def discover_nemotron_models(
    config: AppConfig,
    api_key: str | None = None,
    client: httpx.AsyncClient | None = None,
    use_fallback_on_error: bool = False,
) -> tuple[list[str], list[str]]:
    """Returns (eligible_nemotron_ids, all_listed_model_ids)."""
    try:
        models_raw = await fetch_all_models(config, api_key=api_key, client=client)
        all_ids = [str(m["id"]) for m in models_raw]
        eligible = filter_nemotron_models(all_ids)
        return eligible, all_ids
    except Exception:
        if use_fallback_on_error:
            return list(FALLBACK_NEMOTRON_MODELS), list(FALLBACK_NEMOTRON_MODELS)
        raise
