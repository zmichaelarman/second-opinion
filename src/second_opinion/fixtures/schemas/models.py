from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict


class Film(BaseModel):
    title: str
    year: int
    genre: Literal["drama", "thriller", "sci-fi", "comedy", "horror", "fantasy"]


class FilmNoExtraProps(Film):
    model_config = ConfigDict(extra="forbid")


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
    nickname: Optional[str] = None
    age: Optional[int] = None


def generate_pydantic_schemas() -> dict[str, dict]:
    return {
        "flat_basic": Film.model_json_schema(),
        "no_extra_props": FilmNoExtraProps.model_json_schema(),
        "defs_ref": Order.model_json_schema(),
        "anyof_optional": Person.model_json_schema(),
    }


def write_all_schemas(out_dir: Path | None = None) -> None:
    target = out_dir or Path(__file__).resolve().parent
    target.mkdir(parents=True, exist_ok=True)
    for name, schema in generate_pydantic_schemas().items():
        (target / f"{name}.json").write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    write_all_schemas()
