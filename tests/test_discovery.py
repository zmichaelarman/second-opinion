import pytest

from second_opinion.discovery import classify, discover, runnable
from second_opinion.smoke import pick_models, plan_steps

from fake_tf import MODEL_IDS

INCLUDE = ["nemotron"]
EXCLUDE = ["VL", "Omni", "Safety", "embed", "rerank"]


@pytest.mark.parametrize(
    "model_id,included",
    [
        ("nvidia/nemotron-3-super-120b-a12b", True),
        ("nvidia/Nemotron-3_5-Lightning", True),
        ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B", True),
        ("nvidia/Nemotron-Nano-VL-12B-V2", False),
        ("nvidia/nemotron-3.5-super-vl-120b-a12b", False),
        ("nvidia/nemotron-3-nano-omni-30b-a3b-reasoning", False),
        ("nvidia/nemotron-3.5-content-safety", False),
        ("nvidia/llama-nemotron-embed-1b-v2", False),
        ("nvidia/llama-3.2-nv-rerankqa-1b-v2", False),
        ("meta-llama/Llama-3.3-70B-Instruct", False),
    ],
)
def test_classify(model_id, included):
    assert classify(model_id, INCLUDE, EXCLUDE)[0] is included


def test_discover_matches_prices_case_insensitively(settings):
    body = {"data": [{"id": "nvidia/nvidia-nemotron-3-nano-30b-a3b"}, {"id": "nvidia/Nemotron-New-1B"}]}
    found = {m.id: m for m in discover(body, settings)}
    assert found["nvidia/nvidia-nemotron-3-nano-30b-a3b"].price.role == "nano"
    assert found["nvidia/Nemotron-New-1B"].included and found["nvidia/Nemotron-New-1B"].price is None


def test_smoke_plan_stays_within_15_calls(settings):
    models = discover({"data": [{"id": m} for m in MODEL_IDS]}, settings)
    assert len(runnable(models)) == 4
    picked = pick_models(models)
    assert [m.role for m in picked] == ["nano", "lightning", "super", "ultra"]
    steps = plan_steps(picked)
    assert len(steps) == 15
    assert sum(s.via == "openai-sdk" for s in steps) == 1
    cheap = {s.model.id for s in steps if s.name.startswith("banana_off_via_")}
    assert cheap == {"nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"}
