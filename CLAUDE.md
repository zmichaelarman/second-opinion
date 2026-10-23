# CLAUDE.md: Second Opinion

Build spec for Claude Code. Put this file at the root of an empty repo named `second-opinion`. Claude Code loads it at the start of every session, so it is also the project's memory: keep §3 and §13 up to date.

---

## 0. Session protocol (read first, every session)

1. Read **§13 Status**. Work only on the current milestone, in order.
2. **Spending rule.** Every command that calls a paid API prints the planned calls, models and estimated cost, then exits unless `--yes` is passed. **Never pass `--yes` yourself.** Show the estimate to the user and wait for an explicit OK in chat. Exceptions: offline unit tests, and `second-opinion smoke` (15 calls or fewer).
3. Never read, print, log or commit API keys. Keys live only in `.env`, which is gitignored. Request bodies stored in `runs/` never include headers with auth.
4. Never exceed provider rate limits. No stress or load testing. The Token Factory terms prohibit exceeding limits.
5. Pass/fail is decided by deterministic code (oracles). An LLM never grades a test.
6. `runs/` is append-only. Never overwrite or delete raw data.
7. When a fact in §3 is marked **UNVERIFIED**, confirm it with a live call before relying on it, then update §3.
8. Any friction you hit with Token Factory's API or docs while building (confusing errors, doc mismatches, missing info) gets a timestamped row in `FRICTION_LOG.md`. That log feeds the hackathon feedback.
9. Finish each milestone by running the offline tests, ticking it in §13, and committing with the message `M<n>: <summary>`.

---

## 1. What we're building

**Second Opinion** is a CLI that checks whether Nebius Token Factory keeps its promises to developers building agents on NVIDIA Nemotron models: that JSON, tool calls, reasoning and streaming come back the way the docs say.

The name comes from the reference replay: before a finding is blamed on anyone, it gets a second opinion from NVIDIA's own hosted endpoint for the same model.

It targets the **conversion layer**. Nemotron emits raw text (reasoning in think tags, tool calls as XML-style markup), and the server converts that into OpenAI-shaped fields. That step is where bugs cluster, and only the platform can fix it.

Three parts, plus a report:

1. **Contract suite.** Deterministic tests of the conversion layer, run at scale with confidence intervals.
2. **Docs checker.** Extracts every code example from the Token Factory inference docs and checks or executes it. An agent repairs broken ones and verifies each fix by running it.
3. **Investigator.** An agent that root-causes failure clusters in four steps:
   1. Shrink the failing request (an algorithm).
   2. Propose a cause (LLM).
   3. Write and run a pytest experiment (LLM writes the code).
   4. Accept or reject the cause with Fisher's exact test.
4. **Report.** A static HTML report on GitHub Pages, plus a drafted feedback write-up.

**Hackathon context.** Nebius x NVIDIA Global AI Hackathon (Devpost).
- Track: Coding and Agentic Engineering.
- Deadline: Fri 2026-10-30, 10:00 PT.
- Target prize: Most Valuable Feedback, judged on the completeness, viability and impact of the feedback. The same project also competes for the track award.

**Non-goals:**
- Benchmarking model intelligence.
- Load or stress testing.
- Headline latency numbers. Latency is recorded but not headlined.
- Any UI beyond the static HTML report.

**Model roles inside Second Opinion:**
- Nemotron 3 Ultra proposes causes in the Investigator (few calls, hardest reasoning).
- Nemotron 3 Super writes experiment code and docs fixes.
- Every listed Nemotron model is also a **test subject**.

---

## 2. Hackathon requirements this repo must satisfy

- [ ] Makes runtime calls to the Token Factory inference API using NVIDIA Nemotron models (true by design).
- [ ] Public GitHub repo with an `LICENSE` file (Apache-2.0) so the license shows in the repo's About box.
- [ ] `README.md` with setup and run instructions, and a section titled **"How Second Opinion uses NVIDIA Nemotron and Nebius Token Factory"**.
- [ ] Working demo URL: the static report served by GitHub Pages from `/docs`.
- [ ] Feedback text drafted in `FEEDBACK_DRAFT.md`. The human edits it and pastes it into Devpost.
- [ ] No secrets in the repo. Check with `git grep -nE "nvapi-[A-Za-z0-9_-]{20,}|Bearer [A-Za-z0-9._-]{20,}"` before every push. (The original pattern, `nvapi-|Bearer [A-Za-z0-9]`, matched its own line in this file, so it could never come back clean.)

---

## 3. Facts (checked 2026-10-23 from public docs and listings; update as you verify)

### Token Factory
- Base URL is `https://api.tokenfactory.nebius.com/v1/`. The API is OpenAI-compatible, with header `Authorization: Bearer $NEBIUS_API_KEY`.
- The docs index is `https://docs.tokenfactory.nebius.com/llms.txt`. Every docs page is available as raw `.md`.
- The OpenAPI spec is at `https://api.tokenfactory.nebius.com/openapi.json`. Checked 2026-10-23 (spec version `20261022-1d60f32ca`):
  - Chat-completions request fields: `extra_body`, `frequency_penalty`, `logit_bias`, `logprobs`, `max_completion_tokens`, `max_tokens`, `messages`, `model`, `n`, `presence_penalty`, `reasoning_effort`, `response_format`, `service_tier`, `stop`, `store`, `stream`, `stream_options`, `temperature`, `tool_choice`, `tools`, `top_logprobs`, `top_p`, `user`.
  - There's no `chat_template_kwargs`. Instead the spec lists `extra_body` as a body field ("To provide extra parameters"), even though in the OpenAI SDK `extra_body` is a client-side option that gets merged into the top level.
  - `reasoning_effort` takes `none | minimal | low | medium | high | xhigh | max`. It may be an alternative thinking switch (**UNVERIFIED** on Nemotron).
  - `tool_choice` takes `"none"`, `"auto"`, `"required"`, a named function, or null. The prose docs list only `"auto"` and named.
  - `max_completion_tokens` is documented as including reasoning tokens.
  - The response message and the stream delta both define `reasoning_content` and `reasoning`.
- **Structured output.** Uses `response_format` with `{"type":"json_schema", ...}` or `{"type":"json_object"}`. The docs show two different `json_schema` shapes:
  - Bare: `"json_schema": <schema>`
  - Wrapped: `"json_schema": {"name", "schema", "strict"}`
- **Function calling.** OpenAI format. The docs list only two `tool_choice` values: `"auto"` and a named function.
- **Rate limits.** Dynamic. The docs' example baseline is 60 RPM and 400k TPM; the real default is shown in the console.
  - Exceeding the limit returns HTTP 429 with `Retry-After`.
  - Response headers: `x-ratelimit-limit-requests`, `x-ratelimit-limit-tokens`, `x-ratelimit-remaining-requests`, `x-ratelimit-remaining-tokens`, `x-ratelimit-reset-requests`, `x-ratelimit-reset-tokens`, `x-ratelimit-dynamic-scale-requests`, `x-ratelimit-dynamic-scale-tokens`, `x-ratelimit-dynamic-period-remaining`, `x-ratelimit-dynamic-period-usage-requests`, `x-ratelimit-dynamic-period-usage-tokens`, `x-ratelimit-over-limit`.
  - `x-ratelimit-over-limit: yes` means the request was processed over the limit. Slow down when you see it.
- **Public serverless endpoints** report region "Global", and the processing location can change without notice. Mention this under limitations.
- **Sandboxes** (Token Factory's environment for agents to run code) is in beta, needs an access request, and reports a per-operation `cost`. Optional only.

### Models
Runtime discovery via `GET /v1/models` is authoritative. The table below is a hint and fallback only.

| Role | Token Factory model ID | Price in / out, $ per 1M tokens (third-party listing, 2026-10-20) |
|---|---|---|
| Nemotron 3 Nano | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` (**UNVERIFIED** casing) | ~0.06 / 0.24 (**UNVERIFIED**) |
| Nemotron 3.5 Lightning | `nvidia/Nemotron-3_5-Lightning` | 0.06 / 0.24 |
| Nemotron 3 Super | `nvidia/nemotron-3-super-120b-a12b` | 0.30 / 0.90 |
| Nemotron 3 Ultra | `nvidia/Nemotron-3-Ultra-550b-a55b` | 1.00 / 3.00 |

- Discovery rule: include IDs that contain `nemotron` (case-insensitive). Exclude IDs containing `VL`, `Omni`, `Safety`, `embed` or `rerank`.
- Prices live in `config/prices.yaml` with a `source` and `as_of` field. The user confirms them in the console.

### Nemotron behavior (from NVIDIA, vLLM and SGLang docs)
- Reasoning is **on by default**. It's controlled through the top-level request field `chat_template_kwargs`, which the OpenAI SDK sends via `extra_body`:
  - `enable_thinking` (bool)
  - `low_effort` (bool)
  - **UNVERIFIED on Token Factory**, whose OpenAPI spec doesn't list it (see above). The smoke test must find out which switch works: top-level `chat_template_kwargs`, `chat_template_kwargs` nested inside `extra_body`, or `reasoning_effort: "none"`.
- Raw output format:
  - Reasoning appears inside `<think>...</think>`.
  - Tool calls use Qwen3-Coder-style XML (`<tool_call>`, `<function=...>`, `<parameter=...>`) and are converted to `tool_calls` server-side.
- The reference parser has a special case: if `</think>` never closes (max_tokens is hit), and thinking is disabled or `force_nonempty_content` is set, the reasoning text is returned as `content`.
- Reasoning text arrives in `message.reasoning_content` or `message.reasoning`, depending on server version. Check both and record which one appears. **UNVERIFIED on Token Factory.**
- NVIDIA recommends sampling at **temperature 1.0, top_p 0.95** for all tasks. Use these defaults unless a test says otherwise.

### NVIDIA reference API (attribution only; free tier)
- Base URL is `https://integrate.api.nvidia.com/v1`, with key `NVIDIA_API_KEY` from build.nvidia.com.
- IDs, checked against the public `GET /v1/models` on 2026-10-23: `nvidia/nemotron-3-super-120b-a12b`, `nvidia/nemotron-3-ultra-550b-a55b`, and Lightning is `nvidia/nemotron-3.5-lightning-30b-a3b`. **Nano isn't listed**: the only Nano is `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`, an Omni variant. Nano findings therefore get no reference replay and stay owner `UNKNOWN`.
- Rate limits are unknown. Default to 20 RPM and small sample sizes.

### Known docs defects (use as regression fixtures; copy the real text from the cached page, don't invent it)
- `ai-models-inference/json.md`, Python example: missing commas inside the `response_format` dict and after it, so it raises a SyntaxError.
- `ai-models-inference/json.md`, cURL example: the key is spelled `"response format"` (with a space), and a comma is missing, so the JSON body is invalid.
- `ai-models-inference/json.md`: the bare and wrapped schema shapes are used inconsistently across examples.
- `ai-models-inference/function-calling.md`: the examples use `meta-llama/Meta-Llama-3.1-8B-Instruct-fast`. Check whether it's still listed.

---

## 4. Tech stack

- **Python ≥ 3.11.**
- **Naming:** repo `second-opinion`, import package `second_opinion`, CLI command `second-opinion` (console script `second-opinion = "second_opinion.cli:app"` in `pyproject.toml`).
- **Dependencies:** `httpx`, `openai`, `typer`, `rich`, `pydantic>=2`, `jsonschema`, `scipy`, `numpy`, `jinja2`, `pyyaml`, `python-dotenv`, `pytest`, `pytest-asyncio`.
- **Optional:** `node` on PATH, used for `node --check` on JS snippets.
- Use raw **`httpx`** for every measured call, to capture the exact JSON and headers. The OpenAI SDK hides unknown fields. Use the `openai` SDK only in the smoke test's "SDK view" check.
- Tests that hit the network are marked `@pytest.mark.live` and excluded by default (`addopts = -m "not live"`).

---

## 5. Repo layout

```
second-opinion/
  CLAUDE.md
  README.md
  LICENSE                      # Apache-2.0
  TEST_PLAN.md                 # generated, committed + tagged prereg-v1 before the screening stage
  FRICTION_LOG.md
  FEEDBACK_DRAFT.md            # generated, then hand-edited
  .env.example                 # NEBIUS_API_KEY= / NVIDIA_API_KEY= / SECOND_OPINION_BUDGET_USD=40
  .gitignore                   # .env, runs/, .venv/, __pycache__/, *.pyc, experiments/*.py.tmp
  pyproject.toml
  config/
    prices.yaml
    defaults.yaml              # rpm, concurrency, n_screen=50, n_escalate=300, alpha=0.01
  src/second_opinion/
    __init__.py
    cli.py                     # typer app
    config.py
    client.py                  # async httpx client (TF + NVIDIA), SSE parsing, raw capture
    pacer.py                   # header-aware rate limiting
    budget.py                  # spend ledger + hard cap
    store.py                   # run dirs, JSONL append, resume
    discovery.py               # GET /models + filtering
    parse.py                   # ParsedResponse from JSON or SSE chunks
    oracles.py                 # deterministic checks, typed_equal, failure codes
    stats.py
    planner.py                 # cells, cost estimates, prereg hash
    runner.py                  # executes stages
    contracts/
      base.py                  # Contract, Condition, Severity, Owner, OracleResult
      registry.py
      t1_reasoning_leak.py
      t2_thinking_switch.py
      t3_tool_args.py          # includes the streaming (T4) conditions
      t5_schemas.py
    fixtures/
      schemas/models.py        # Pydantic models -> generated JSON schemas
      schemas/*.json
      tools/*.json
      payloads.py              # expected argument sets, markup string
    docs_check/
      fetch.py  extract.py  static.py  execute.py  repair.py
    investigator/
      cluster.py  shrink.py  hypothesize.py  experiment.py  decide.py  loop.py
    testkit.py                 # the ONLY API agent-written experiments may import
    reference.py               # NVIDIA replay + owner attribution
    byo.py                     # bring-your-own tools/schemas mode
    report/
      build.py  feedback.py
      templates/report.html.j2
  experiments/                 # agent-written pytest files (committed)
  tests/                       # offline unit tests + recorded fixtures
  runs/                        # gitignored raw data
  docs/                        # GitHub Pages output (index.html, findings.md, data/)
```

---

## 6. Core data model

```python
class Outcome(str, Enum): PASS = "pass"; FAIL = "fail"; ERROR = "error"
class Severity(int, Enum): S1_SILENT = 1; S2_LOUD = 2; S3_GAP = 3; S4_DOCS = 4
class Owner(str, Enum): NEBIUS_PLATFORM = "nebius_platform"; NEBIUS_DOCS = "nebius_docs"; NVIDIA_MODEL = "nvidia_model"; UPSTREAM_SERVING = "upstream_serving"; UNKNOWN = "unknown"

@dataclass(frozen=True)
class Condition:
    thinking: bool | None          # None = omit chat_template_kwargs
    stream: bool = False
    max_tokens: int | None = None
    variant: str = "default"       # payload / schema variant
    shape: str = "wrapped"         # T5 request shape: wrapped | bare | json_object

class Contract(Protocol):
    id: str; tier: int; title: str; hypothesis: str
    severity: Severity; core_conditions: set[Condition]; conditions: list[Condition]
    def build_request(self, model: str, cond: Condition, trial: int) -> dict: ...
    def check(self, req: dict, resp: "ParsedResponse") -> "OracleResult": ...
```

**Cell** = (contract_id, model, condition). `cell_id` = a stable short hash of those three.

**ParsedResponse** fields:
- `status`, `headers` (the rate-limit subset, `retry-after`, any request-id header)
- `content`, `reasoning`, `reasoning_field` (`"reasoning_content"`, `"reasoning"` or None)
- `tool_calls` (a list of `{index, id, name, arguments_raw}`), `finish_reason`, `usage`, `refusal`
- `n_chunks`, `ttft_s`, `total_s`, `raw` (the response JSON, or the list of SSE `data:` payloads, truncated at 200 KB)

**CallRecord** (one JSONL line per attempt):
- Identity: `run_id`, `stage`, `cell_id`, `contract_id`, `model`, `condition`, `trial`, `attempt`
- Timing and exchange: `ts_start`, `request` (body only), `status`, `headers`, `raw`, `parsed`
- Results: `outcome`, `codes`, `detail`, `usage`, `cost_usd`

**Failure codes.** Use these exact strings. Codes are either fatal (they make the trial FAIL) or soft (recorded and reported, but don't fail the trial).

| Code | Fatal | Severity | Meaning |
|---|---|---|---|
| `HTTP_4XX` | yes | S3 (S2 if the request shape is the docs' own) | request rejected |
| `HTTP_5XX`, `TIMEOUT` | n/a | n/a | outcome ERROR, never FAIL |
| `EMPTY_CONTENT` | yes | S2 | finish_reason=stop but content empty |
| `JSON_PARSE` | yes | S2 | JSON requested, content isn't valid JSON |
| `SCHEMA_INVALID:<kw>` | yes | S2 | parses, but violates the schema; `<kw>` = jsonschema validator name |
| `REASONING_IN_CONTENT` | yes | S2 | `<think>`/`</think>` in content, or prose where JSON/answer expected |
| `EXTRA_CONTENT` | yes | S2 | exact-answer task: normalized content ≠ expected |
| `THINKING_NOT_DISABLED` | yes | S3 | enable_thinking=false but reasoning present or completion_tokens > 64 |
| `NO_REASONING_FIELD` | yes | S3 | enable_thinking=true but no separate reasoning field |
| `NO_TOOL_CALLS` | yes | S2 | forced tool_choice, `tool_calls` empty |
| `TOOL_MARKUP_IN_CONTENT` | yes | S2 | content contains `<tool_call>`, `<function=` or `<parameter=` |
| `WRONG_TOOL` | yes | S2 | tool name mismatch |
| `ARGS_JSON_INVALID` | yes | S2 | `arguments` string isn't valid JSON |
| `TYPE_COERCED` | yes | **S1** | same value, wrong JSON type (`"02134"` to 2134, 42 to `"42"`, int to float, big-int precision loss) |
| `VALUE_TRUNCATED` | yes | **S1** | string argument cut short (e.g. ends right before `</parameter>`) |
| `VALUE_MISMATCH` | yes | S2 | different value (owner hint: model copy error) |
| `STREAM_FINISH_MISSING` | yes | S2 | no chunk carried a finish_reason |
| `STREAM_USAGE_MISSING` | soft | S3 | include_usage requested, no usage chunk |
| `FINISH_REASON_WRONG` | soft | S3 | e.g. tool call present but finish_reason ≠ `tool_calls` |
| `TRUNCATED` | soft | n/a | finish_reason=length (informational) |
| `REFUSAL` | soft | n/a | `refusal` non-null |

Outcome rule: any ERROR-class condition gives ERROR. Otherwise any fatal code gives FAIL. Otherwise PASS.

---

## 7. Contract catalog (Tier 1)

### Shared request rules
- **Defaults:** `temperature: 1.0`, `top_p: 0.95`.
- **max_tokens:** 1024 when thinking is off; 4096 when thinking is on or omitted, unless the contract sets it.
- **Thinking switch:** when `cond.thinking` is not None, send top-level `"chat_template_kwargs": {"enable_thinking": <bool>}`.
- **System message** for every fixture: `You are a component in an automated test. Follow the user's instructions exactly.`
- **Answer normalization** for exact-answer tasks: `content.strip().lower().strip(".!\"' ")`.

### T1: Reasoning leaks under token limits (severity S2)
- **`T1.json_truncation`**
  - Conditions: thinking=True, `max_tokens` ∈ {128, 512, 2048}.
  - Request: `response_format` wrapped with the `flat_basic` schema. User message: `Return a JSON object for the film Alien (1979) with title, year and genre. Genre must be one of the allowed values.`
  - Oracle:
    - Content empty and finish=length: PASS, with soft `TRUNCATED`.
    - Content empty and finish=stop: `EMPTY_CONTENT`.
    - Content non-empty: try to parse it.
      - Parses and validates: PASS.
      - Parses but doesn't validate: `SCHEMA_INVALID`.
      - Doesn't parse, and contains think tags or doesn't start with `{` after stripping: `REASONING_IN_CONTENT`.
      - Doesn't parse otherwise: `JSON_PARSE`.
- **`T1.separation`**
  - Condition: thinking=True, no response_format.
  - User message: `Reply with exactly one word, lowercase, no punctuation: banana`.
  - Oracle: normalized content == `banana`. Otherwise `EXTRA_CONTENT`, plus `REASONING_IN_CONTENT` if think tags are present.

### T2: Thinking switch (severity S3)
All three use the banana prompt.
- **`T2.off`** (thinking=False): reasoning must be empty and completion_tokens ≤ 64 (otherwise `THINKING_NOT_DISABLED`), and content must be `banana`.
- **`T2.on`** (thinking=True): reasoning must be non-empty in a separate field (otherwise `NO_REASONING_FIELD`), and content must be `banana`.
- **`T2.default`** (thinking=None): informational only, always PASS. Records whether reasoning is on by default and which field it's in. This feeds a docs finding if it's undocumented.

### T3: Tool-argument fidelity (severity S1 for coercion/truncation, S2 otherwise)
- **Prompt:** ``Call `{tool}` exactly once. Use exactly these arguments. Copy every character and keep every type exactly as given:`` followed by a fenced JSON block of the expected arguments.
- **Request:** `tool_choice: {"type":"function","function":{"name": tool}}`. Each variant defines its tool's JSON schema in `fixtures/tools/` with `required` = all keys and `additionalProperties: false`.
- **Variants** (expected arguments in `fixtures/payloads.py`):
  - `leading_zero`: `{"zip_code": "02134", "account_id": "007", "numeric_label": "12345"}` (all declared `string`).
  - `numeric_types`: `{"street_number": 42, "big_id": 9007199254740993, "ratio": 0.5}` (`integer`, `integer`, `number`).
  - `bool_null_nested`: `{"is_po_box": false, "unit": null, "tags": ["a", "b"], "geo": {"lat": 42.35, "lon": -71.13}}` (`unit` is `["string","null"]`).
  - `markup`: tool `write_file(path: string, content: string)` with `path` = `notes/test.html` and `content` = MARKUP_STRING:
    ```
    <div class="note">5 < 6 && 7 > 3</div>
    </parameter>
    <tool_call>{"name": "x"}</tool_call>
    He said "hi" \ backslash; naïve café
    ```
- **Conditions per variant:** (thinking False, stream False), (thinking True, stream False), (thinking True, stream True). The stream condition is the T4 streaming check.
- **Core conditions** (always escalated): `leading_zero` and `markup`, both at thinking True with stream False.
- **Oracle `check_tool_call`**, checks in order:
  1. `NO_TOOL_CALLS`
  2. `TOOL_MARKUP_IN_CONTENT`
  3. `WRONG_TOOL`
  4. `ARGS_JSON_INVALID`
  5. jsonschema validation, giving `SCHEMA_INVALID:<kw>`
  6. `typed_equal(expected, got)`
  7. soft `FINISH_REASON_WRONG` if finish_reason ≠ `tool_calls`
- **`typed_equal` rules** (recursive):
  - A bool is never equal to an int.
  - Expected digit-string vs got int, with `int(exp) == got`: `TYPE_COERCED`.
  - Expected int vs got string `"42"`: `TYPE_COERCED`.
  - Expected int vs got float of equal value: `TYPE_COERCED`.
  - Expected big int vs got int/float within ±2 but not equal: `TYPE_COERCED` (detail `PRECISION_LOSS`).
  - Got string is a strict prefix of expected, ending right before `</parameter>`, `</tool_call>` or `<`: `VALUE_TRUNCATED`.
  - Anything else unequal: `VALUE_MISMATCH`, with a difflib diff in `detail`.
- **Streaming (T4) assembly:**
  - Concatenate `delta.content`.
  - Concatenate `delta.reasoning_content` or `delta.reasoning`.
  - Merge `delta.tool_calls` by `index`: take the name from the first chunk that has one, and concatenate `arguments`.
  - Send `stream_options: {"include_usage": true}`.
  - Extra stream oracles: `STREAM_FINISH_MISSING`, soft `STREAM_USAGE_MISSING`, and markup or think tags in any content delta.

### T5: Schema enforcement (severity S2)
- **Schemas** live in `fixtures/schemas/`. Schemas marked (Pydantic) are generated from Pydantic models in `models.py` via `model_json_schema()`. Commit both the `.py` and the generated `.json`.

| Variant | Schema | User prompt (tempts a violation) |
|---|---|---|
| `flat_basic` | Film{title: str, year: int, genre: Literal["drama","thriller","sci-fi","comedy","horror","fantasy"]} (Pydantic) | `Give details of a real film starring Jack Nicholson.` |
| `enum_temptation` | same as flat_basic | `Describe the film La La Land, a musical.` |
| `no_extra_props` | flat_basic + `additionalProperties: false` | `Describe the film Alien (1979). Also include its box office gross.` |
| `min_max` | {title: string, rating: number, minimum 0, maximum 10} | `The critic rated Alien 11 out of 10.` |
| `pattern` | {ticket_code: string, pattern `^[A-Z]{3}-[0-9]{4}$`} | `My ticket code is abc-12.` |
| `array_bounds` | {tags: array of string, minItems 2, maxItems 3} | `Tag this film: horror, sci-fi, space, classic, thriller.` |
| `defs_ref` | Order{customer: Customer{name, email}, items: list[LineItem{sku, qty: int}]} (Pydantic, produces `$defs` and `$ref`) | `Ada (ada@example.com) ordered 2 of SKU A1 and 1 of SKU B7.` |
| `anyof_optional` | Person{name: str, nickname: Optional[str], age: Optional[int]} (Pydantic, produces `anyOf` with null) | `The person's name is Ada. Nickname and age are unknown.` |

- **Request shapes:**
  - `wrapped` = `{"type":"json_schema","json_schema":{"name": <variant>, "schema": S, "strict": true}}`
  - `bare` = `{"type":"json_schema","json_schema": S}`
  - `json_object` = `{"type":"json_object"}`
- **Conditions:**
  - Every variant at thinking False and thinking True (wrapped shape): 16 conditions.
  - `flat_basic` with shape=`bare`, thinking False.
  - `flat_basic` with shape=`json_object`, thinking False. Oracle: content parses to a dict.
  - `flat_basic` streaming, thinking False.
  - Total: 19 conditions.
- **Core conditions:** `flat_basic`, `defs_ref` and `anyof_optional`, each at thinking False with the wrapped shape.
- **Oracle:**
  1. 4xx: `HTTP_4XX`. This is fatal, because the docs say these shapes are supported.
  2. Parse the content.
  3. Validate with `Draft202012Validator`. Record all errors, and code the first one as `SCHEMA_INVALID:<validator>`.
  4. A non-null `refusal` gives soft `REFUSAL`.

**Totals:** T1 has 4 conditions, T2 has 3, T3 has 12, T5 has 19, for 38 in all. Across 4 models that's 152 cells.

---

## 8. Runner, pacing, budget

**Stages:**
- **`pilot`:** 1 trial per condition on the cheapest model, plus 3 trials of each T3 variant on Super. About 50 calls. It checks the plumbing and measures median tokens per (contract, thinking) for cost estimates. Excluded from analysis.
- **`screen`:** n=50 per cell. Refuses to run unless the git tag `prereg-v1` exists and `TEST_PLAN.md`'s registry hash matches the code (`--no-prereg-check` for development only).
- **`escalate`:** tops cells up to n=300 total. A cell qualifies if it had at least one fatal failure in screening, or is a core condition.
  - `--max-n MODEL=N` caps cost on a model (e.g. `--max-n nvidia/Nemotron-3-Ultra-550b-a55b=150`).

**Client:**
- Async `httpx`, with a 10 s connect timeout and a 600 s read timeout (Ultra reasoning can be slow).
- Concurrency semaphore, default 6.

**Pacer:**
- Token-bucket rate = min(`--rpm`, default 40, and 0.8 × the latest `x-ratelimit-limit-requests`).
- If `x-ratelimit-remaining-tokens` is below 2× the next request's estimated tokens, wait.
- If `x-ratelimit-over-limit: yes` appears, halve the rate for 120 s.

**Retries:**
- 429: sleep for `Retry-After` (30 s if absent), up to 3 attempts.
- 5xx or timeout: retry once after 5 s.
- Every attempt is its own CallRecord. The final ERROR stays an ERROR.

**Budget:**
- `BudgetGuard` uses a cap from `SECOND_OPINION_BUDGET_USD` (default 40).
- Spend = usage × `prices.yaml`, persisted in `runs/ledger.jsonl` (append-only) across all runs.
- Stop scheduling when spent + in-flight reserve would exceed the cap. Let in-flight calls finish, write the summary, and exit with code 3.

**Planner:**
- Before any paid stage, print a table: model × contract × calls × estimated $ and the total, then exit unless `--yes`.
- Token estimates come from pilot medians. Fallbacks: thinking off ≈ 400 in / 300 out; thinking on ≈ 400 in / 2000 out.

**Storage:**
- Each run gets `runs/<YYYYMMDD-HHMM>-<stage>/` containing `plan.json` (git hash, tag, config, model list), `calls.jsonl` and `summary.json`.
- Runs are resumable: skip (cell_id, trial) pairs that already have a final record.

---

## 9. Statistics and pre-registration

`stats.py`, with offline unit tests against known values:
- `wilson_ci(k, n, z=1.959964)`
- `clopper_pearson(k, n)`
- `rule_of_three(n) = 3/n`
- `fisher_two_sided(f1, n1, f2, n2)` via `scipy.stats.fisher_exact`
- `holm(pvals)`
- `bootstrap_quantile_ci(x, q, B=10000, seed=0)`, used for latency (reported, not headlined)

**Pre-registered comparison families.** Holm correction within each family, adjusted α = 0.01.
- **F1:** streaming vs non-streaming. T3 variants at thinking True, per model.
- **F2:** thinking on vs off. T3 and T5 variants, per model.
- **F3:** wrapped vs bare shape. `T5.flat_basic`, per model.

**Trust matrix.**
- Rows are models. Columns are capabilities:
  - Structured output (T5, thinking off)
  - Structured output with reasoning (T5, thinking on)
  - Tool arguments (T3, non-stream)
  - Tool arguments, streaming (T3 stream)
  - Reasoning control (T2)
  - Reasoning leak (T1)
- Cell color is set by the worst cell in that column:
  - **Red:** any cell whose failure-rate Wilson lower bound exceeds 1%.
  - **Amber:** failures observed, but not red.
  - **Green:** zero fatal failures. Show the bound "failure rate < `rule_of_three(n_pooled)`".

**Pre-registration.** `second-opinion plan --write` generates `TEST_PLAN.md` from the registry: contracts, conditions, n, α, families, color rules, and a SHA-256 of the serialized registry. The **user** commits it and runs `git tag prereg-v1` before the screening stage. Ask them to.

---

## 10. Docs checker (`second-opinion docs`)

1. **Fetch.**
   - Download `llms.txt`.
   - Keep URLs matching `/quickstart\.md$`, `/switch\.md$`, `/ai-models-inference/(overview|function-calling|json|rate-limits)\.md$`, `/cached-tokens\.md$`, `/api-reference/examples/.*\.md$` and `/api-reference/inference/create-chat-completion\.md$`.
   - Cache the raw pages in `runs/docs-cache/<date>/`.
2. **Extract.** Pull fenced code blocks, recording page, block index, language and line. Normalize aliases:
   - `py` to python
   - `sh`, `shell`, `bash` to bash
   - `js`, `javascript`, `ts` to js
3. **Static checks** (free):
   - **Python:** `ast.parse`, giving `PY_SYNTAX_ERROR`.
   - **cURL blocks:**
     - Take the single-quoted body after `--data-binary`, `--data` or `-d`.
     - Replace shell splices like `'"$VAR"'` with `{}`, then run `json.loads`, giving `CURL_JSON_INVALID`.
     - If it parses, compare its top-level keys to the chat-completions request params (from `openapi.json`, with a static fallback list). Unknown keys give `UNKNOWN_PARAM:<key>`.
   - **JS:** if `node` exists, run `node --check` on a temp `.mjs`, giving `JS_SYNTAX_ERROR`.
   - **Model IDs:** match the regex `model\s*[=:]\s*["']([^"']+)["']`. IDs missing from the live model list give `MODEL_NOT_LISTED`.
4. **Execute** (`--execute`; a paid stage, so it follows the spending rule):
   - Eligible blocks are Python only. They must pass static checks, match the allowlist `chat\.completions\.create|models\.list|embeddings\.create`, and not match the denylist `fine_tuning|endpoints|files\.|batches|sandbox|contree|images\.generate`.
   - Run each in a subprocess in a temp directory, with env limited to `NEBIUS_API_KEY` and `PATH`, and a 90 s timeout.
   - A failure gives `PY_RUNTIME_ERROR:<ExceptionType>`.
5. **Repair agent** (`--repair`): this is part of the track requirement that agents write, run and test code.
   - For each failing block, send Nemotron 3 Super (thinking on) the page title, the code, the error and the current model list.
   - Ask for the **minimal** corrected snippet that keeps the original intent, as one fenced block.
   - Re-run the static checks, and execution if the block is eligible. Iterate up to 3 times.
   - Write `docs_findings.jsonl` with: `page_url`, `block_idx`, `lang`, `codes`, `error`, `fixed`, `attempts`, `patch` (unified diff).
6. All docs findings have Severity S4 and Owner `NEBIUS_DOCS`. One exception: an `UNKNOWN_PARAM` that the API silently ignores is S2.
7. **Offline regression tests.** Copy the real Python and cURL examples from the cached `json.md` into `tests/fixtures/json_md_excerpt.md`. Assert that the checker reports `PY_SYNTAX_ERROR`, `CURL_JSON_INVALID` and `UNKNOWN_PARAM:response format`.

---

## 11. Investigator (`second-opinion investigate`) and reference replay

1. **Cluster.**
   - Group fatal failures by signature: (contract_id, sorted fatal codes).
   - Keep clusters with at least 3 failures, or at least 1 if any code is S1.
   - Rank by severity, then by (number of models affected × failure rate).
2. **Shrink** (deterministic, no LLM).
   - **Reducible features:**
     - T3: drop argument keys (in both the tool schema and the expected arguments); halve the markup string (ddmin on lines, then characters); stream to false; thinking to false.
     - T5: drop properties not involved in the first error; drop keywords not involved; inline `$ref`; shorten the prompt.
   - Score each candidate with k=12 trials. Keep a reduction if failures ≥ max(1, ceil(0.5 × k × p0)), where p0 is the cluster's failure rate. Failures are probabilistic, so never decide from one trial.
   - Limits: at most 8 candidates (about 100 calls) per cluster.
   - Output: the minimal request and its measured failure rate.
3. **Hypothesize** (Nemotron 3 Ultra, thinking on, `response_format` json_schema).
   - Input: the minimal request, 3 failing and 3 passing parsed responses, and the **factor menu**.
   - Output: up to 3 `{factor, level_a, level_b, rationale}`. Drop any factor not on the menu.
   - **Factor menu** (implemented in `testkit.py`):
     - `thinking` {true, false}
     - `stream` {true, false}
     - `max_tokens` {int}
     - `shape` {wrapped, bare}
     - `append_literal` {string}: appends to the markup argument
     - `remove_schema_keyword` {keyword}
     - `inline_refs` {true, false}
     - `arg_type` {arg, string|integer}
     - `temperature` {0, 1.0}
4. **Experiment.** Nemotron 3 Super writes `experiments/<cluster>_<h>.py` using only `second_opinion.testkit`:
   ```python
   from second_opinion.testkit import base_request, arm, compare
   BASE = base_request("<cluster_id>")
   def test_hypothesis():
       a = arm(BASE, factor="stream", level=True, n=100)
       b = arm(BASE, factor="stream", level=False, n=100)
       compare(a, b, hypothesis_id="<id>")
   ```
   - **Safety gate** (AST walk) before running:
     - The only import allowed is `second_opinion.testkit`.
     - These names are banned: `open`, `exec`, `eval`, `compile`, `__import__`, `getattr`, `os`, `sys`, `subprocess`, `socket`.
     - No attributes starting with `_`.
     - 60 lines at most.
   - If the gate fails, ask for a rewrite (2 tries at most). After that, run the arms directly and log the fallback.
   - Run with `pytest -q <file> -p no:cacheprovider` in a subprocess. `testkit` enforces the budget and writes results to `runs/<id>/experiments/`.
5. **Decide.**
   - Use the two-sided Fisher test with Holm correction across the cluster's hypotheses.
   - **Accept** if adjusted p < 0.01 and |rate_a − rate_b| ≥ 0.05.
   - If every hypothesis is rejected, run one more round with the rejected ones given as context.
   - Write a **finding** with:
     - Title (LLM-drafted, human-edited), severity, models affected
     - Minimal repro as curl and as Python
     - Per-arm failure rates with Wilson CIs, p-value, the accepted factor
     - Workaround: a factor level that avoids the failure
     - Owner = UNKNOWN until replay
   - Budget: $3 per cluster by default.
6. **Reference replay** (`second-opinion reference`).
   - Run each finding's minimal request n=30 on the NVIDIA API at 20 RPM. If the NVIDIA API rejects a Token Factory-only parameter, drop it and note that.
   - **Owner rules:**
     - Token Factory fails while NVIDIA fails significantly less (Fisher p < 0.05): `NEBIUS_PLATFORM`.
     - Both fail at similar rates: `NVIDIA_MODEL` for `VALUE_MISMATCH`-type codes, `UPSTREAM_SERVING` otherwise.
     - No mapping available: `UNKNOWN`.
   - State in limitations that the NVIDIA API is a different serving stack, used as a reference, not ground truth.

---

## 12. Report, feedback draft, README

**`second-opinion report --runs <dirs...> --out docs/`** writes:
- **`docs/index.html`.** One self-contained file: inline CSS and SVG, no external requests. It handles light and dark mode via `prefers-color-scheme` and is readable at 380 px wide. Sections:
  1. Header: date, git hash, prereg tag.
  2. "What to trust": the trust matrix (§9).
  3. Findings: cards with severity, owner, per-arm rates with CIs, and the repro inside `<details>`.
  4. Docs health: X of Y examples pass, Z repaired, with diffs.
  5. Methods and limitations. Cover sampling settings, light load only, global routing, dates, model IDs, n, thresholds, and what wasn't tested.
  6. Data: `docs/data/*.jsonl.gz`.
- **`docs/findings.md`.**
- **`FEEDBACK_DRAFT.md`** (repo root), following the structure Nebius asked for:
  - Top 5 issues, prioritized, at the top.
  - Then, for each of Token Factory, AI Cloud, and each Nemotron model:
    - what we used it for
    - what worked (with numbers)
    - what needs work (linked findings)
    - onboarding from zero to hello world (pulled from `FRICTION_LOG.md`)
    - whether we'd build with it again, and why

**README sections:**
1. One-line pitch and the demo link
2. What it found (top 3; fill in after the runs)
3. Quickstart: `pip install -e .`, `.env`, `second-opinion smoke`, `second-opinion run --stage pilot`
4. How it works (the 3 parts)
5. **How Second Opinion uses NVIDIA Nemotron and Nebius Token Factory**
   - Ultra: diagnosis
   - Super: writes experiment code and docs fixes
   - All Nemotron models: test subjects
   - Everything runs through the Token Factory API
6. Methodology and pre-registration
7. Limitations
8. License (Apache-2.0)

**`FRICTION_LOG.md` template:**

| Time (ET) | Step | Expected | What happened | Minutes lost | Severity |
|---|---|---|---|---|---|

---

## 13. Milestones and status

Dates are targets. Must tier = M0–M4, Should = M5. If time collapses, M0–M4 alone is a complete submission.

- [ ] **M0: Scaffold and smoke test (Fri 10/23)**
  - [x] `pyproject.toml`, package skeleton, `LICENSE` (Apache-2.0), `.gitignore`, `.env.example`, `FRICTION_LOG.md`, `config/prices.yaml`, `config/defaults.yaml`
  - [ ] `second-opinion models`: discovery and filtering, printed as a table.
  - [ ] `second-opinion smoke` (15 calls or fewer).
    1. List the models.
    2. For each Nemotron model: one banana call with thinking=false and one with thinking=true. Record where reasoning appears.
    3. For each Nemotron model: one forced `leading_zero` tool call.
    4. One call through the `openai` SDK, printing which fields the SDK exposes (`model_extra` keys).
  - [ ] Update every **UNVERIFIED** item in §3 with the smoke results.
  - Acceptance: smoke runs end to end, writes `runs/<ts>-smoke/`, and the secret grep is clean.
- [ ] **M1: Engine (Sat 10/24)**
  - [ ] Client, SSE parser, pacer, budget guard, store/resume, contract base, oracles including `typed_equal`, stats, planner
  - [ ] Offline unit tests for oracles, `typed_equal`, SSE assembly (from recorded smoke fixtures), stats and planner.
  - [ ] `second-opinion run --stage pilot` (needs the user's OK)
- [ ] **M2: Tier 1 contracts and screening (Sat to Sun)**
  - [ ] T1, T2, T3 (with the T4 stream conditions) and T5, with fixtures and generated schemas
  - [ ] `second-opinion plan --write`. The user commits it and tags `prereg-v1`.
  - [ ] `screen` stage (needs the user's OK)
- [ ] **M3: Docs checker (Sun 10/25)**
  - [ ] fetch, extract, static, execute, repair, and the regression tests from §10.7
- [ ] **M4: Escalation, report and Pages (Mon 10/26). Must-tier done.**
  - [ ] `escalate` stage (needs the user's OK).
  - [ ] `second-opinion report`, with GitHub Pages serving `/docs`.
- [ ] **M5: Investigator, reference replay and bring-your-own mode (Tue 10/27)**
  - [ ] cluster, shrink, hypothesize, experiment (with the safety gate), decide
  - [ ] `second-opinion reference`
  - [ ] `second-opinion check --tools tools.json [--schemas schemas.json]` (§14)
- [ ] **M6: Packaging (Wed 10/28)**
  - [ ] README, `FEEDBACK_DRAFT.md`, findings polish.
  - [ ] `second-opinion rerun`: re-runs every finding's repro at n=30 and marks each still-failing or fixed
  - [ ] Screenshots for the demo video
- [ ] **M7: Submit (Thu 10/29)**
  - [ ] Morning `second-opinion rerun`, final report publish, secret grep, Devpost submission (done by the user)


---

## 14. CLI reference

```
second-opinion models
second-opinion smoke
second-opinion plan --write
second-opinion run --stage {pilot,screen,escalate} [--models auto|ID,...] [--contracts T1,T3,...]
                   [--rpm 40] [--concurrency 6] [--max-n MODEL=N] [--yes]
second-opinion docs [--execute] [--repair] [--yes]
second-opinion investigate --runs DIR... [--hypothesis-model ID] [--code-model ID] [--yes]
second-opinion reference --findings FILE [--yes]
second-opinion check --tools tools.json [--schemas schemas.json] [--models ...] [--n 50] [--yes]
second-opinion report --runs DIR... --out docs/
second-opinion rerun --findings FILE [--yes]
```

**Bring-your-own mode** (`second-opinion check`):
- **Tools:** for each user tool definition, generate 5 seeded argument sets from its schema. Use tricky but valid values: leading-zero strings, digit-only strings, markup, unicode, quotes, ints near 2^53, booleans, null where allowed, nested arrays. Run the T3 oracle with the exact-copy prompt.
- **Schemas:** for each user schema, use the prompt `Produce one valid example object.` and the T5 oracle.
- **Output:** a per-tool, per-model reliability table in the report.

---

## 15. Tier 2 and 3 (only after M5, if time remains)

- `tool_choice: "required"` and `"none"` behavior (undocumented)
- Stop sequences with thinking on. Does max_tokens count reasoning tokens?
- Seed determinism at temperature 0: 20 identical calls, compare outputs
- `usage.prompt_tokens_details.cached_tokens` on a repeated long prefix
- Rate-limit headers present on normal responses. Observe only; never exceed limits.
- Response JSON validated against the chat-completion response schema in `openapi.json`
- One context-length probe just over the advertised limit, on Lightning or Nano only: check the status code and how clear the error is
- Does `low_effort` reduce reasoning tokens? (Mann-Whitney U)
- Optional: run Investigator experiments in Token Factory Sandboxes (`--executor sandbox`), only if beta access is approved and the cost is confirmed covered. Keep an `Executor` interface with a local-subprocess default.
