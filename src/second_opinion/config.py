"""Settings from config/defaults.yaml, config/prices.yaml and .env.

API keys stay in the environment. Nothing here prints, logs or stores them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


class ConfigError(RuntimeError):
    pass


def find_root() -> Path:
    """$SECOND_OPINION_ROOT, else the nearest directory (cwd upwards) holding config/defaults.yaml."""
    env = os.environ.get("SECOND_OPINION_ROOT")
    if env:
        return Path(env).resolve()
    cwd = Path.cwd().resolve()
    for d in (cwd, *cwd.parents):
        if (d / "config" / "defaults.yaml").is_file():
            return d
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Price:
    model: str
    input: float  # USD per 1M prompt tokens
    output: float  # USD per 1M completion tokens
    source: str
    as_of: str
    role: str | None = None

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens * self.input + completion_tokens * self.output) / 1_000_000


@dataclass
class Settings:
    root: Path
    defaults: dict[str, Any]
    prices: dict[str, Price]
    budget_usd: float

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def ledger_path(self) -> Path:
        return self.runs_dir / "ledger.jsonl"

    def provider(self, name: str) -> dict[str, Any]:
        return self.defaults["providers"][name]

    def price_for(self, model: str) -> Price | None:
        """Exact ID match first, then case-insensitive (model ID casing is not guaranteed)."""
        if model in self.prices:
            return self.prices[model]
        return {k.lower(): v for k, v in self.prices.items()}.get(model.lower())


def load_settings(root: Path | None = None) -> Settings:
    root = root or find_root()
    load_dotenv(root / ".env", override=False)
    defaults = yaml.safe_load((root / "config" / "defaults.yaml").read_text())
    raw = yaml.safe_load((root / "config" / "prices.yaml").read_text()) or {}
    prices = {
        model: Price(
            model=model,
            input=float(row["input"]),
            output=float(row["output"]),
            source=str(row["source"]),
            as_of=str(row["as_of"]),
            role=row.get("role"),
        )
        for model, row in (raw.get("models") or {}).items()
    }
    budget = os.environ.get("SECOND_OPINION_BUDGET_USD") or defaults.get("budget_usd", 40)
    return Settings(root=root, defaults=defaults, prices=prices, budget_usd=float(budget))


def api_key(env_name: str) -> str:
    key = os.environ.get(env_name, "").strip()
    if not key:
        raise ConfigError(f"{env_name} is not set. Add it to .env (see .env.example).")
    return key
