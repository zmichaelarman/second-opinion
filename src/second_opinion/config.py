from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "pyproject.toml").exists() or (parent / "CLAUDE.md").exists():
            return parent
    return Path.cwd()


REPO_ROOT = find_repo_root()


@dataclass
class ModelPrice:
    role: str
    input_per_1m: float
    output_per_1m: float


@dataclass
class AppConfig:
    nebius_base_url: str = "https://api.tokenfactory.nebius.com/v1"
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    docs_index_url: str = "https://docs.tokenfactory.nebius.com/llms.txt"
    openapi_url: str = "https://api.tokenfactory.nebius.com/openapi.json"
    rpm: int = 40
    nvidia_rpm: int = 20
    concurrency: int = 6
    n_screen: int = 50
    n_escalate: int = 300
    n_reference: int = 30
    n_rerun: int = 30
    alpha: float = 0.01
    budget_usd: float = 40.0
    cluster_budget_usd: float = 3.0
    temperature: float = 1.0
    top_p: float = 0.95
    max_tokens_thinking_off: int = 1024
    max_tokens_thinking_on: int = 4096
    connect_timeout_s: float = 10.0
    read_timeout_s: float = 600.0
    hypothesis_model: str = "nvidia/Nemotron-3-Ultra-550b-a55b"
    code_model: str = "nvidia/nemotron-3-super-120b-a12b"
    prices_source: str = ""
    prices_as_of: str = ""
    default_price: ModelPrice = field(
        default_factory=lambda: ModelPrice("Default", 0.30, 0.90)
    )
    model_prices: dict[str, ModelPrice] = field(default_factory=dict)

    def get_price(self, model_id: str) -> ModelPrice:
        if model_id in self.model_prices:
            return self.model_prices[model_id]
        lower = model_id.lower()
        for k, v in self.model_prices.items():
            if k.lower() == lower:
                return v
        # Partial match fallback for Nano/Lightning/Super/Ultra
        if "nano" in lower or "lightning" in lower:
            return ModelPrice("Nemotron Nano/Lightning", 0.06, 0.24)
        if "super" in lower:
            return ModelPrice("Nemotron 3 Super", 0.30, 0.90)
        if "ultra" in lower:
            return ModelPrice("Nemotron 3 Ultra", 1.00, 3.00)
        return self.default_price

    def estimate_cost(self, model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
        p = self.get_price(model_id)
        return (prompt_tokens * p.input_per_1m + completion_tokens * p.output_per_1m) / 1_000_000.0


def load_config(repo_root: Path | None = None) -> AppConfig:
    root = repo_root or REPO_ROOT
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
    else:
        load_dotenv(override=False)

    defaults_path = root / "config" / "defaults.yaml"
    prices_path = root / "config" / "prices.yaml"

    defaults_data: dict[str, Any] = {}
    if defaults_path.exists():
        defaults_data = yaml.safe_load(defaults_path.read_text(encoding="utf-8")) or {}

    prices_data: dict[str, Any] = {}
    if prices_path.exists():
        prices_data = yaml.safe_load(prices_path.read_text(encoding="utf-8")) or {}

    budget_env = os.environ.get("SECOND_OPINION_BUDGET_USD")
    budget_usd = float(budget_env) if budget_env else float(defaults_data.get("budget_usd", 40.0))

    def_p = prices_data.get("default", {})
    default_price = ModelPrice(
        role="Default",
        input_per_1m=float(def_p.get("input_per_1m", 0.30)),
        output_per_1m=float(def_p.get("output_per_1m", 0.90)),
    )

    model_prices: dict[str, ModelPrice] = {}
    for mid, mdata in (prices_data.get("models") or {}).items():
        model_prices[mid] = ModelPrice(
            role=str(mdata.get("role", mid)),
            input_per_1m=float(mdata.get("input_per_1m", default_price.input_per_1m)),
            output_per_1m=float(mdata.get("output_per_1m", default_price.output_per_1m)),
        )

    return AppConfig(
        nebius_base_url=str(defaults_data.get("nebius_base_url", "https://api.tokenfactory.nebius.com/v1")),
        nvidia_base_url=str(defaults_data.get("nvidia_base_url", "https://integrate.api.nvidia.com/v1")),
        docs_index_url=str(defaults_data.get("docs_index_url", "https://docs.tokenfactory.nebius.com/llms.txt")),
        openapi_url=str(defaults_data.get("openapi_url", "https://api.tokenfactory.nebius.com/openapi.json")),
        rpm=int(defaults_data.get("rpm", 40)),
        nvidia_rpm=int(defaults_data.get("nvidia_rpm", 20)),
        concurrency=int(defaults_data.get("concurrency", 6)),
        n_screen=int(defaults_data.get("n_screen", 50)),
        n_escalate=int(defaults_data.get("n_escalate", 300)),
        n_reference=int(defaults_data.get("n_reference", 30)),
        n_rerun=int(defaults_data.get("n_rerun", 30)),
        alpha=float(defaults_data.get("alpha", 0.01)),
        budget_usd=budget_usd,
        cluster_budget_usd=float(defaults_data.get("cluster_budget_usd", 3.0)),
        temperature=float(defaults_data.get("temperature", 1.0)),
        top_p=float(defaults_data.get("top_p", 0.95)),
        max_tokens_thinking_off=int(defaults_data.get("max_tokens_thinking_off", 1024)),
        max_tokens_thinking_on=int(defaults_data.get("max_tokens_thinking_on", 4096)),
        connect_timeout_s=float(defaults_data.get("connect_timeout_s", 10.0)),
        read_timeout_s=float(defaults_data.get("read_timeout_s", 600.0)),
        hypothesis_model=str(defaults_data.get("hypothesis_model", "nvidia/Nemotron-3-Ultra-550b-a55b")),
        code_model=str(defaults_data.get("code_model", "nvidia/nemotron-3-super-120b-a12b")),
        prices_source=str(prices_data.get("source", "")),
        prices_as_of=str(prices_data.get("as_of", "")),
        default_price=default_price,
        model_prices=model_prices,
    )


def get_nebius_api_key(repo_root: Path | None = None) -> str:
    root = repo_root or REPO_ROOT
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
    key = os.environ.get("NEBIUS_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "NEBIUS_API_KEY is not set. Add it to .env (see .env.example) before making live Token Factory calls."
        )
    return key


def get_nvidia_api_key(repo_root: Path | None = None) -> str:
    root = repo_root or REPO_ROOT
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
    key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "NVIDIA_API_KEY is not set. Add it to .env (see .env.example) before running reference replay."
        )
    return key

