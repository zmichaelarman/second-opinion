import asyncio
import json
from pathlib import Path

import httpx
from rich.console import Console

from second_opinion.docs_check.execute import eligible, exception_type, run_snippet
from second_opinion.docs_check.extract import CodeBlock, extract_blocks
from second_opinion.docs_check.pipeline import probe_verdict, run_docs
from second_opinion.docs_check.repair import extract_fenced, repair_block
from second_opinion.docs_check.static import (
    CHAT_PARAMS_FALLBACK,
    check_block,
    curl_body,
    request_params,
    top_level_keys,
)

from fake_tf import FakeTokenFactory

FIXTURE = Path(__file__).parent / "fixtures" / "json_md_excerpt.md"
PARAMS = {"/v1/chat/completions": CHAT_PARAMS_FALLBACK, "/v1/completions": frozenset({"model", "prompt", "max_tokens", "temperature", "stream"})}


def block(code, lang="python", raw="python"):
    return CodeBlock("https://docs.example/page.md", 0, lang, raw, "", 1, code)


def test_regression_json_md_defects_are_reported():
    """CLAUDE.md §10.7: the real json.md Python and cURL examples are broken."""
    blocks = extract_blocks("https://docs.tokenfactory.nebius.com/ai-models-inference/json.md", FIXTURE.read_text())
    assert [b.lang for b in blocks] == ["python", "bash"]
    codes = [c for b in blocks for c in check_block(b, PARAMS, None).codes]
    assert "PY_SYNTAX_ERROR" in codes
    assert "CURL_JSON_INVALID" in codes
    assert "UNKNOWN_PARAM:response format" in codes


def test_extract_handles_titles_attributes_and_indented_fences():
    text = "intro\n```python Python expandable theme={null}\nprint(1)\n```\n* item\n  ```json theme={null}\n  {\"a\": 1}\n  ```\n"
    a, b = extract_blocks("u", text)
    assert (a.lang, a.title, a.line, a.code) == ("python", "Python", 2, "print(1)")
    assert (b.lang, b.code, b.line) == ("json", '{"a": 1}', 6)


def test_curl_body_replaces_shell_splices():
    code = """curl https://api.tokenfactory.nebius.com/v1/chat/completions --data-binary '{"model": "m", "x": '"$SCHEMA"', "y": "'"${NAME}"'"}'"""
    assert json.loads(curl_body(code)) == {"model": "m", "x": {}, "y": "{}"}


def test_top_level_keys_tolerates_missing_commas_and_nesting():
    text = '{"model": "m"\n "messages": [{"role": "user"}], "response format": {"type": "json_schema" "json_schema": {}}}'
    assert top_level_keys(text) == ["model", "messages", "response format"]


def test_endpoint_specific_params_and_auth_header():
    code = """curl 'https://api.tokenfactory.nebius.com/v1/completions' -H "Authorization: $NEBIUS_API_KEY" -d '{"model": "m", "prompt": "p"}'"""
    r = check_block(block(code, "bash", "bash"), PARAMS, None)
    assert r.kind == "curl" and r.endpoint == "/v1/completions" and r.codes == ["CURL_AUTH_HEADER"]
    ok = code.replace("Authorization: $", "Authorization: Bearer $")
    assert check_block(block(ok, "bash", "bash"), PARAMS, None).codes == []


def test_model_check_only_for_token_factory_blocks():
    openai_example = 'from openai import OpenAI\nclient = OpenAI()\nclient.responses.create(model="gpt-5.2", input="x")'
    assert check_block(block(openai_example), PARAMS, {"nvidia/a"}).codes == []
    tf_example = 'import os\nkey = os.environ["NEBIUS_API_KEY"]\nmodel="meta-llama/Old-Model"\nplaceholder = {"model": "<model-name>"}'
    r = check_block(block(tf_example), PARAMS, {"nvidia/a"})
    assert r.codes == ["MODEL_NOT_LISTED"] and r.models == ["meta-llama/Old-Model"]


def test_request_params_from_openapi():
    spec = {"paths": {"/v1/embeddings": {"post": {"requestBody": {"content": {"application/json": {
        "schema": {"$ref": "#/components/schemas/E"}}}}}}}, "components": {"schemas": {"E": {"properties": {"model": {}, "input": {}}}}}}
    params = request_params(spec)
    assert params["/v1/embeddings"] == {"model", "input"} and params["/v1/chat/completions"] == CHAT_PARAMS_FALLBACK


def test_eligibility_allow_and_deny_lists():
    ok = block("client.chat.completions.create(model='m', messages=[])")
    deny = block("client.files.create(file=open('x'))\nclient.chat.completions.create()")
    from second_opinion.docs_check.static import StaticResult
    assert eligible(ok, StaticResult("python")) and not eligible(deny, StaticResult("python"))
    assert not eligible(ok, StaticResult("python", codes=["PY_SYNTAX_ERROR"]))


def test_run_snippet_outcomes_and_key_scrubbing():
    assert run_snippet("print('hi')", "k-secret").ok
    bad = run_snippet("import os\nprint(os.environ['NEBIUS_API_KEY'])\nraise ValueError('boom')", "k-secret")
    assert (bad.ok, bad.code) == (False, "PY_RUNTIME_ERROR:ValueError")
    assert "k-secret" not in bad.stdout_tail + bad.stderr_tail and "[REDACTED]" in bad.stdout_tail
    env = run_snippet("import os\nassert set(os.environ) <= {'NEBIUS_API_KEY', 'PATH', 'PWD', 'LC_CTYPE', '__CF_USER_TEXT_ENCODING'}, sorted(os.environ)", "k")
    assert env.ok, env.stderr_tail
    slow = run_snippet("import time\ntime.sleep(5)", "k", timeout=1)
    assert slow.code == "PY_RUNTIME_ERROR:TimeoutExpired"


def test_exception_type_parsing():
    assert exception_type("Traceback...\nopenai.NotFoundError: Error code: 404") == "NotFoundError"
    assert exception_type("nothing useful") == "UnknownError"


def test_repair_retries_until_the_checker_passes():
    broken = block("print('a'")
    replies = iter(["no code here", "```python\nprint('a'\n```", "```python\nprint('a')\n```"])
    sent = []

    async def complete(body, attempt):
        sent.append(body)
        return next(replies)

    result = asyncio.run(repair_block(broken, "Page", "PY_SYNTAX_ERROR", ["nvidia/a"], model="nvidia/super",
                                      complete=complete, recheck=lambda b: check_block(b, PARAMS, None), execute=None))
    assert result.fixed and result.attempts == 3 and result.code == "print('a')"
    assert "+print('a')" in result.patch and "-print('a'" in result.patch
    assert sent[-1]["chat_template_kwargs"] == {"enable_thinking": True} and len(sent[-1]["messages"]) == 6
    assert extract_fenced("x\n```js\nconsole.log(1)\n```\ny") == "console.log(1)"


def test_probe_verdicts():
    assert probe_verdict("response format", {"type": "json_object"}, 200, "hello") == "silently ignored"
    assert probe_verdict("guided_json", {"type": "object"}, 200, '{"answer": "hello"}') == "accepted and honored"
    assert probe_verdict("top_k", 50, 200, "hello") == "accepted (effect not verifiable)"
    assert probe_verdict("top_k", 50, 400, None) == "rejected"


def test_docs_pipeline_static_only_offline(settings):
    page = "https://docs.tokenfactory.nebius.com/ai-models-inference/json.md"
    docs = {
        "https://docs.tokenfactory.nebius.com/llms.txt": f"- [JSON]({page})\n- [Other](https://docs.tokenfactory.nebius.com/other.md)\n",
        "https://api.tokenfactory.nebius.com/openapi.json": json.dumps({"paths": {}}),
        page: "# Structured output\n\n" + FIXTURE.read_text(),
    }

    def docs_handler(request):
        return httpx.Response(200, text=docs[str(request.url)])

    console = Console(record=True, width=200)
    code = asyncio.run(run_docs(settings, console, transport=FakeTokenFactory().transport(),
                                docs_transport=httpx.MockTransport(docs_handler)))
    assert code == 0
    [run_dir] = list(settings.runs_dir.glob("*-docs"))
    findings = [json.loads(x) for x in (run_dir / "docs_findings.jsonl").read_text().splitlines()]
    assert {c for f in findings for c in f["codes"]} >= {"PY_SYNTAX_ERROR", "CURL_JSON_INVALID", "UNKNOWN_PARAM:response format",
                                                         "MODEL_NOT_LISTED"}
    assert all(f["severity"] == "S4_DOCS" and f["owner"] == "nebius_docs" for f in findings)
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["pages"] == 1 and summary["examples_checked"] == 2 and summary["examples_passing"] == 0
