from __future__ import annotations

from pathlib import Path

from second_opinion.docs_check.execute import is_block_eligible_for_execution
from second_opinion.docs_check.extract import extract_code_blocks
from second_opinion.docs_check.repair import make_unified_diff
from second_opinion.docs_check.static import check_code_block_static

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def test_json_md_excerpt_regression() -> None:
    """Offline regression test (§10.7):
    Assert that the checker reports PY_SYNTAX_ERROR, CURL_JSON_INVALID and UNKNOWN_PARAM:response format
    on the real Python and cURL examples copied from cached json.md.
    """
    excerpt_path = FIXTURES_DIR / "json_md_excerpt.md"
    markdown = excerpt_path.read_text(encoding="utf-8")
    blocks = extract_code_blocks(markdown, page_url="https://docs.tokenfactory.nebius.com/ai-models-inference/json.md")
    assert len(blocks) == 2

    all_codes: list[str] = []
    for blk in blocks:
        res = check_code_block_static(blk)
        all_codes.extend(res.codes)

    assert "PY_SYNTAX_ERROR" in all_codes
    assert "CURL_JSON_INVALID" in all_codes
    assert "UNKNOWN_PARAM:response format" in all_codes


def test_model_not_listed_and_execution_allowlist() -> None:
    md = (
        "```python\n"
        "from openai import OpenAI\n"
        "client = OpenAI()\n"
        'client.chat.completions.create(model="meta-llama/Meta-Llama-3.1-8B-Instruct-fast", messages=[])\n'
        "```\n"
        "```python\n"
        "from openai import OpenAI\n"
        "client = OpenAI()\n"
        'client.fine_tuning.jobs.create(model="nvidia/Nemotron-3_5-Lightning")\n'
        "```\n"
    )
    blocks = extract_code_blocks(md, page_url="https://docs.tokenfactory.nebius.com/ai-models-inference/function-calling.md")
    live_models = {"nvidia/Nemotron-3_5-Lightning", "nvidia/nemotron-3-super-120b-a12b"}

    res0 = check_code_block_static(blocks[0], known_models=live_models)
    assert "MODEL_NOT_LISTED" in res0.codes
    assert is_block_eligible_for_execution(blocks[0], [])

    # Denylisted fine_tuning block is not eligible for execution
    assert not is_block_eligible_for_execution(blocks[1], [])

    diff = make_unified_diff("a = 1\n", "a = 2\n", filename="test.py")
    assert "-a = 1" in diff
    assert "+a = 2" in diff

