"""T3 fixtures: expected tool-call arguments, the markup string and the exact-copy prompt (CLAUDE.md §7)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).parent / "tools"

MARKUP_STRING = "\n".join(
    [
        '<div class="note">5 < 6 && 7 > 3</div>',
        "</parameter>",
        '<tool_call>{"name": "x"}</tool_call>',
        'He said "hi" \\ backslash; naïve café',
    ]
)

EXPECTED_ARGS: dict[str, dict[str, Any]] = {
    "leading_zero": {"zip_code": "02134", "account_id": "007", "numeric_label": "12345"},
    "numeric_types": {"street_number": 42, "big_id": 9007199254740993, "ratio": 0.5},
    "bool_null_nested": {"is_po_box": False, "unit": None, "tags": ["a", "b"], "geo": {"lat": 42.35, "lon": -71.13}},
    "markup": {"path": "notes/test.html", "content": MARKUP_STRING},
}

T3_VARIANTS = tuple(EXPECTED_ARGS)


def load_tool(variant: str) -> dict[str, Any]:
    """The OpenAI-format tool definition for a T3 variant (fixtures/tools/<variant>.json)."""
    return json.loads((TOOLS_DIR / f"{variant}.json").read_text())


def tool_name(tool: dict[str, Any]) -> str:
    return tool["function"]["name"]


def tool_prompt(name: str, args: dict[str, Any]) -> str:
    return (
        f"Call `{name}` exactly once. Use exactly these arguments. "
        "Copy every character and keep every type exactly as given:\n\n"
        f"```json\n{json.dumps(args, indent=2, ensure_ascii=False)}\n```"
    )


def forced_tool_choice(name: str) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name}}
