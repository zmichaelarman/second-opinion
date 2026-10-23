"""Pydantic models for the T5 schemas (CLAUDE.md §7). `python -m second_opinion.fixtures.schemas.models`
regenerates the JSON files next to this module; both the .py and the .json are committed."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel

HERE = Path(__file__).parent


class Film(BaseModel):
    title: str
    year: int
    genre: Literal["drama", "thriller", "sci-fi", "comedy", "horror", "fantasy"]


class Customer(BaseModel):
    name: str
    email: str


class LineItem(BaseModel):
    sku: str
    qty: int


class Order(BaseModel):
    customer: Customer
    items: list[LineItem]


class Person(BaseModel):
    name: str
    nickname: Optional[str]
    age: Optional[int]


def generated() -> dict[str, dict[str, Any]]:
    film = Film.model_json_schema()
    return {
        "flat_basic": film,
        "no_extra_props": {**film, "additionalProperties": False},
        "defs_ref": Order.model_json_schema(),
        "anyof_optional": Person.model_json_schema(),
    }


def write_all() -> list[Path]:
    paths = []
    for name, schema in generated().items():
        path = HERE / f"{name}.json"
        path.write_text(json.dumps(schema, indent=2) + "\n")
        paths.append(path)
    return paths


if __name__ == "__main__":
    for p in write_all():
        print(p)
