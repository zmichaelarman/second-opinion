import json
from collections import Counter

import pytest

from second_opinion.contracts import registry, t5_schemas
from second_opinion.contracts.base import Condition
from second_opinion.planner import registry as build_registry
from second_opinion.planner import registry_hash

DEFAULTS = {"n_screen": 50, "n_escalate": 300, "alpha": 0.01}


def test_totals_match_the_spec():
    contracts = registry.all_contracts()
    assert len(contracts) == 17
    per_tier = Counter()
    for c in contracts:
        per_tier[c.tier] += len(c.conditions)
    assert dict(per_tier) == {1: 4, 2: 3, 3: 12, 5: 19}
    assert sum(per_tier.values()) * 4 == 152  # cells across 4 models
    assert len({c.id for c in contracts}) == 17


def test_core_conditions():
    core = {(c.id, cond) for c in registry.all_contracts() for cond in c.core_conditions}
    assert core == {
        ("T3.leading_zero", Condition(True, False, variant="leading_zero")),
        ("T3.markup", Condition(True, False, variant="markup")),
        ("T5.flat_basic", Condition(False, variant="flat_basic")),
        ("T5.defs_ref", Condition(False, variant="defs_ref")),
        ("T5.anyof_optional", Condition(False, variant="anyof_optional")),
    }


def req(contract_id, **cond):
    c = registry.by_id()[contract_id]
    match = [x for x in c.conditions if all(getattr(x, k) == v for k, v in cond.items())]
    assert len(match) == 1, match
    return c.build_request("m", match[0], 0)


def test_shared_request_rules():
    off = req("T2.off")
    assert off["chat_template_kwargs"] == {"enable_thinking": False} and off["max_tokens"] == 1024
    assert (off["temperature"], off["top_p"]) == (1.0, 0.95)
    assert off["messages"][0] == {"role": "system", "content": "You are a component in an automated test. Follow the user's instructions exactly."}
    default = req("T2.default")
    assert "chat_template_kwargs" not in default and default["max_tokens"] == 4096
    assert req("T1.json_truncation", max_tokens=128)["max_tokens"] == 128


def test_t3_requests_force_the_tool_and_stream_with_usage():
    body = req("T3.markup", thinking=True, stream=True)
    assert body["tool_choice"] == {"type": "function", "function": {"name": "write_file"}}
    assert body["stream_options"] == {"include_usage": True}
    assert "naïve café" in body["messages"][1]["content"] and "Copy every character" in body["messages"][1]["content"]
    assert body["tools"][0]["function"]["parameters"]["additionalProperties"] is False


def test_t5_request_shapes():
    wrapped = req("T5.flat_basic", thinking=False, shape="wrapped", stream=False)["response_format"]
    assert wrapped["type"] == "json_schema" and set(wrapped["json_schema"]) == {"name", "schema", "strict"}
    bare = req("T5.flat_basic", shape="bare")["response_format"]
    assert bare["json_schema"]["properties"]["genre"]["enum"][2] == "sci-fi"
    assert req("T5.flat_basic", shape="json_object")["response_format"] == {"type": "json_object"}
    assert "$defs" in req("T5.defs_ref", thinking=False)["response_format"]["json_schema"]["schema"]


def test_registry_hash_is_stable_and_sensitive(monkeypatch):
    first = registry_hash(build_registry(DEFAULTS))
    assert first == registry_hash(build_registry(DEFAULTS))
    json.dumps(build_registry(DEFAULTS))  # serializable
    monkeypatch.setitem(t5_schemas.PROMPTS, "pattern", "My ticket code is ABC-1234.")
    assert registry_hash(build_registry(DEFAULTS)) != first
    assert registry_hash(build_registry({**DEFAULTS, "alpha": 0.05})) != first


def test_select():
    assert [c.id for c in registry.select("T1")] == ["T1.json_truncation", "T1.separation"]
    assert [c.id for c in registry.select("T3.markup,T2.off")] == ["T2.off", "T3.markup"]
    with pytest.raises(ValueError):
        registry.select("T9")
