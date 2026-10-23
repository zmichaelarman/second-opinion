# Nebius Token Factory & NVIDIA Nemotron — Developer Feedback (Second Opinion)

## Top 5 Prioritized Issues

1. **[S4_DOCS] Docs defect in `https://docs.tokenfactory.nebius.com/ai-models-inference/json.md` (block #1)** — Codes: `PY_SYNTAX_ERROR` (SyntaxError at line 31: invalid syntax)
2. **[S2_LOUD] Docs defect in `https://docs.tokenfactory.nebius.com/ai-models-inference/json.md` (block #2)** — Codes: `CURL_JSON_INVALID, UNKNOWN_PARAM:response format` (Invalid JSON in cURL data body: Expecting ',' delimiter: line 17 column 4 (char 454)
Unknown chat completion request parameter: 'response format')
3. **[S4_DOCS] Docs defect in `https://docs.tokenfactory.nebius.com/api-reference/examples/text-generation.md` (block #1)** — Codes: `CURL_JSON_INVALID` (Invalid JSON in cURL data body: Expecting property name enclosed in double quotes: line 31 column 54 (char 879))
4. **[S4_DOCS] Docs defect in `https://docs.tokenfactory.nebius.com/api-reference/examples/text-generation.md` (block #2)** — Codes: `JS_SYNTAX_ERROR` (C:\Users\STUDEN~1\AppData\Local\Temp\tmppja6w60l\snippet.mjs:24
    "guided_json": {"type": "object", "properties": {...},}
                                                        ^

SyntaxError: Unexpected token '}'
    at checkSyntax (node:internal/main/check_syntax:84:5)

Node.js v24.21.0)
5. **[S4_DOCS] Docs defect in `https://docs.tokenfactory.nebius.com/api-reference/examples/vision-capabilities.md` (block #2)** — Codes: `JS_SYNTAX_ERROR` (C:\Users\STUDEN~1\AppData\Local\Temp\tmpix77zi_8\snippet.mjs:24
    "guided_json": {"type": "object", "properties": {...},}
                                                        ^

SyntaxError: Unexpected token '}'
    at checkSyntax (node:internal/main/check_syntax:84:5)

Node.js v24.21.0)

---

## 1. Nebius Token Factory

### What we used it for
- Served as the primary OpenAI-compatible inference platform (`/v1/models`, `/v1/chat/completions` in batch and SSE streaming modes) for all contract verification (`T1`–`T5`), documentation repair, and root-cause factorial experiments.

### What worked (with numbers)
- Standard OpenAI-compatible chat completions and dynamic rate-limit headers (`x-ratelimit-*`) allowed reliable automated pacing across 4 Nemotron models.
- Static & runtime verification checked 6 documentation issues and produced automated unified diffs.

### What needs work (linked findings)
- **Structured output documentation (`ai-models-inference/json.md`):** Fix the Python `SyntaxError` (missing commas), fix the cURL `"response format"` typo and invalid JSON body, and document whether both bare (`"json_schema": <schema>`) and OpenAI-wrapped (`"json_schema": {"name", "schema", "strict"}`) shapes are officially supported.
- **Reasoning control documentation:** Explicitly document `chat_template_kwargs: {"enable_thinking": bool}` and the response field (`reasoning_content` / `reasoning`) in the Token Factory inference docs.

### Onboarding from zero to hello world (from `FRICTION_LOG.md`)

# Friction Log

Timestamped record of developer experience friction encountered while building and running Second Opinion against Nebius Token Factory and NVIDIA Nemotron endpoints.

| Time (ET) | Step | Expected | What happened | Minutes lost | Severity |
|---|---|---|---|---|---|
| 2026-10-23 11:20 | Docs inspection (`ai-models-inference/json.md`) | Copy-pasteable Python & cURL structured output examples | Python example is missing commas in/after `response_format` (`SyntaxError` at line 31); cURL example uses `"response format"` (with a space) and misses a comma (`CURL_JSON_INVALID`); bare vs wrapped `json_schema` shapes are mixed across examples | 15 | S2 |
| 2026-10-23 12:02 | Docs static check (`api-reference/examples/text-generation.md`) | Valid cURL JSON body and JS snippet | cURL snippet has invalid JSON (`Expecting property name enclosed in double quotes` at line 31); JS snippet contains literal `{"properties": {...},}` ellipsis syntax causing `node --check` `SyntaxError: Unexpected token '}'` | 10 | S4 |
| 2026-10-23 12:02 | Docs static check (`api-reference/examples/vision-capabilities.md`) | Valid JS snippet | JS snippet contains literal `{"properties": {...},}` ellipsis syntax causing `node --check` `SyntaxError: Unexpected token '}'` | 5 | S4 |
| 2026-10-23 12:02 | Docs static check (`api-reference/examples/text-generation-for-code-autocomplete.md`) | Valid cURL JSON body | cURL snippet is missing a comma after line 2 in the JSON payload (`Expecting ',' delimiter: line 3 column 5`) | 5 | S4 |

### Would we build with it again?
- **Yes.** Having Ultra, Super, Lightning, and Nano available behind a single OpenAI-compatible endpoint with rich rate-limit headers makes multi-tier agent architectures straightforward, provided conversion-layer edge cases and copy-paste doc snippets are tightened.

---

## 2. Nebius AI Cloud

### What we used it for
- Project workspace, key management, console model catalog & pricing lookup, and serverless endpoint routing configuration.

### What worked
- Rapid API key provisioning and unified billing across the Nemotron model family.

### What needs work
- Exposing machine-readable per-model pricing in `/v1/models` (or a billing metadata endpoint) and surfacing the exact backend serving engine version (vLLM / SGLang parser revision) in response headers would help developers diagnose conversion-layer behavior across `Global` serverless regions.

### Would we build with it again?
- **Yes** — low-friction setup and competitive token pricing for open-weights reasoning models.

---

## 3. NVIDIA Nemotron Models

### `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`
- **What we used it for:** Cost-efficient contract test subject across `T1`–`T5`.
- **What worked:** Strong instruction adherence on structured schemas and multi-step reasoning when given sufficient `max_tokens` budget.
- **What needs work:** See capability matrix in `docs/index.html` for per-contract confidence intervals.
- **Would we build with it again?** Yes — pairing Ultra for diagnosis and Super for code generation while guarding conversion-layer edge cases with deterministic schemas proved highly effective.

### `nvidia/Nemotron-3_5-Lightning`
- **What we used it for:** High-speed screening & pilot contract test subject across `T1`–`T5`.
- **What worked:** Strong instruction adherence on structured schemas and multi-step reasoning when given sufficient `max_tokens` budget.
- **What needs work:** See capability matrix in `docs/index.html` for per-contract confidence intervals.
- **Would we build with it again?** Yes — pairing Ultra for diagnosis and Super for code generation while guarding conversion-layer edge cases with deterministic schemas proved highly effective.

### `nvidia/nemotron-3-super-120b-a12b`
- **What we used it for:** Agentic code generation (`investigator/experiment.py` for pytest experiments and `docs_check/repair.py` for repairing broken docs snippets), plus contract test subject.
- **What worked:** Strong instruction adherence on structured schemas and multi-step reasoning when given sufficient `max_tokens` budget.
- **What needs work:** See capability matrix in `docs/index.html` for per-contract confidence intervals.
- **Would we build with it again?** Yes — pairing Ultra for diagnosis and Super for code generation while guarding conversion-layer edge cases with deterministic schemas proved highly effective.

### `nvidia/Nemotron-3-Ultra-550b-a55b`
- **What we used it for:** Diagnostic root-cause hypothesis generation (`investigator/hypothesize.py`) using structured outputs + reasoning, and contract test subject.
- **What worked:** Strong instruction adherence on structured schemas and multi-step reasoning when given sufficient `max_tokens` budget.
- **What needs work:** See capability matrix in `docs/index.html` for per-contract confidence intervals.
- **Would we build with it again?** Yes — pairing Ultra for diagnosis and Super for code generation while guarding conversion-layer edge cases with deterministic schemas proved highly effective.
