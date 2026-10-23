import asyncio
import json
import random

from jsonschema import Draft202012Validator
from rich.console import Console

from second_opinion.byo import argument_sets, check, generate, reliability_rows

from fake_tf import FakeTokenFactory

TOOL = {"type": "function", "function": {"name": "create_ticket", "parameters": {
    "type": "object",
    "properties": {
        "customer_zip": {"type": "string"},
        "priority": {"enum": ["low", "high"]},
        "count": {"type": "integer", "minimum": 0},
        "ratio": {"type": "number"},
        "urgent": {"type": "boolean"},
        "note": {"type": ["string", "null"]},
        "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3},
        "code": {"type": "string", "pattern": "^[0-9]+$"},
    },
    "required": ["customer_zip", "priority", "count", "ratio", "urgent", "note", "tags", "code"],
    "additionalProperties": False,
}}}


def test_generated_values_are_valid_and_tricky():
    params = TOOL["function"]["parameters"]
    validator = Draft202012Validator(params)
    rng = random.Random(0)
    values = [generate(params, rng) for _ in range(50)]
    assert all(validator.is_valid(v) for v in values)
    zips = {v["customer_zip"] for v in values}
    assert {"02134", "007"} & zips and any("<" in z for z in zips)
    assert 9007199254740993 in {v["count"] for v in values}
    assert all(v["code"].isdigit() for v in values)


def test_argument_sets_are_distinct_valid_and_seeded():
    sets = argument_sets(TOOL)
    assert len(sets) == 5 and len({json.dumps(s, sort_keys=True) for s in sets}) == 5
    assert sets == argument_sets(TOOL)


def test_check_end_to_end(settings, tmp_path):
    tools = tmp_path / "tools.json"
    tools.write_text(json.dumps([TOOL]))
    schemas = tmp_path / "schemas.json"
    schemas.write_text(json.dumps({"order": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}}))
    fake = FakeTokenFactory(coerce_digit_strings=True)
    console = Console(record=True, width=200)
    code = asyncio.run(check(settings, console, tools, schemas, models_spec="nvidia/nemotron-3-super-120b-a12b", n=2,
                             yes=True, transport=fake.transport()))
    assert code == 0
    [run] = list(settings.runs_dir.glob("*-check"))
    summary = json.loads((run / "summary.json").read_text())
    rows = {r["target"]: r for r in summary["rows"]}
    assert rows["tool create_ticket"]["n"] == 10 and rows["schema order"]["n"] == 2
    assert rows["tool create_ticket"]["k"] > 0  # the fake turns digit strings into integers
    assert "reliability per model" in console.export_text()


def test_reliability_rows_pool_argument_sets():
    recs = [{"contract_id": f"BYO.tool.t.{i}", "model": "m", "outcome": o, "final": True}
            for i, o in enumerate(["pass", "fail", "pass", "error"])]
    [row] = reliability_rows(recs)
    assert (row["target"], row["n"], row["k"]) == ("tool t", 3, 1)
