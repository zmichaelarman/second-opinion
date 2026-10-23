"""Pydantic models and the JSON schemas generated from them (T5)."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

SCHEMAS_DIR = Path(__file__).parent
SCHEMA_FILE = {"enum_temptation": "flat_basic"}  # variants that share another variant's schema


@cache
def _load(name: str) -> str:
    return (SCHEMAS_DIR / f"{name}.json").read_text()


def load_schema(variant: str) -> dict[str, Any]:
    """A fresh copy of the JSON schema for a T5 variant."""
    return json.loads(_load(SCHEMA_FILE.get(variant, variant)))
