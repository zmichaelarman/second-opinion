from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FIXTURES_DIR = Path(__file__).resolve().parent
SCHEMAS_DIR = FIXTURES_DIR / "schemas"
TOOLS_DIR = FIXTURES_DIR / "tools"

MARKUP_STRING = (
    '<div class="note">5 < 6 && 7 > 3</div>\n'
    "</parameter>\n"
    '<tool_call>{"name": "x"}</tool_call>\n'
    'He said "hi" \\ backslash; naïve café'
)

T3_EXPECTED_ARGS: dict[str, dict[str, Any]] = {
    "leading_zero": {
        "zip_code": "02134",
        "account_id": "007",
        "numeric_label": "12345",
    },
    "numeric_types": {
        "street_number": 42,
        "big_id": 9007199254740993,
        "ratio": 0.5,
    },
    "bool_null_nested": {
        "is_po_box": False,
        "unit": None,
        "tags": ["a", "b"],
        "geo": {"lat": 42.35, "lon": -71.13},
    },
    "markup": {
        "path": "notes/test.html",
        "content": MARKUP_STRING,
    },
}

T3_TOOL_NAMES: dict[str, str] = {
    "leading_zero": "lookup_account",
    "numeric_types": "record_metric",
    "bool_null_nested": "save_address",
    "markup": "write_file",
}

T5_PROMPTS: dict[str, str] = {
    "flat_basic": "Give details of a real film starring Jack Nicholson.",
    "enum_temptation": "Describe the film La La Land, a musical.",
    "no_extra_props": "Describe the film Alien (1979). Also include its box office gross.",
    "min_max": "The critic rated Alien 11 out of 10.",
    "pattern": "My ticket code is abc-12.",
    "array_bounds": "Tag this film: horror, sci-fi, space, classic, thriller.",
    "defs_ref": "Ada (ada@example.com) ordered 2 of SKU A1 and 1 of SKU B7.",
    "anyof_optional": "The person's name is Ada. Nickname and age are unknown.",
}


def load_tool_definition(variant: str) -> dict[str, Any]:
    path = TOOLS_DIR / f"{variant}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_schema_definition(variant: str) -> dict[str, Any]:
    filename = "flat_basic.json" if variant == "enum_temptation" else f"{variant}.json"
    path = SCHEMAS_DIR / filename
    return json.loads(path.read_text(encoding="utf-8"))
