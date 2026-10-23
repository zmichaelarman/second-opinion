# Second Opinion — Pre-Registered Test Plan (`prereg-v1`)

- **Registry SHA-256:** `f1f832eafedb3de9477c2c1ccdd9a5cbd1e7b617b3b3160e9c9d69af1f73d912`
- **Screening sample size (`n_screen`):** `50` trials per cell
- **Escalation sample size (`n_escalate`):** `300` total trials per cell
- **Family-wise significance threshold (`alpha`):** `0.01` (two-sided Fisher exact test with Holm-Bonferroni correction)
- **Sampling defaults:** `temperature=1.0`, `top_p=0.95`

---

## 1. Pre-Registered Comparison Families

- **F1 (Streaming vs Non-Streaming):** `T3` variants (`leading_zero`, `numeric_types`, `bool_null_nested`, `markup`) at `thinking=True`, `stream=True` vs `stream=False`, per model. Holm correction within family ($\alpha = 0.01$).
- **F2 (Thinking On vs Off):** `T3` variants (`stream=False`) and `T5` variants (`shape=wrapped`, `stream=False`), `thinking=True` vs `thinking=False`, per model. Holm correction within family ($\alpha = 0.01$).
- **F3 (Wrapped vs Bare Schema Shape):** `T5.flat_basic` at `thinking=False`, `shape=wrapped` vs `shape=bare`, per model. Holm correction within family ($\alpha = 0.01$).

---

## 2. Trust Matrix Color Rules

Each model row is evaluated across 6 capability columns (`T5 thinking=False`, `T5 thinking=True`, `T3 non-stream`, `T3 stream`, `T2`, `T1`). Cell color is determined by the worst constituent cell in that column:

- **Red:** Any constituent cell whose fatal failure-rate 95% Wilson confidence interval lower bound exceeds `1.0%` (`> 0.01`).
- **Amber:** One or more fatal failures observed (`k >= 1`), but all constituent cells have Wilson lower bound $\le 1.0\%$.
- **Green:** Zero fatal failures observed (`k = 0`). Displays upper bound `failure rate < 3 / n_pooled` (Rule of Three).

---

## 3. Contract Registry (Tier 1)

### `T1` — Reasoning leaks under token limits and separation (Severity: `S2_LOUD`)

**Hypothesis:** When max_tokens truncates output mid-reasoning or thinking is enabled, raw <think> blocks or reasoning prose may leak into message.content.

| # | Variant | Thinking | Stream | Max Tokens | Shape | Core Escalation |
|---|---|---|---|---|---|---|
| 1 | `json_truncation` | `True` | `False` | `128` | `wrapped` | `no` |
| 2 | `json_truncation` | `True` | `False` | `512` | `wrapped` | `no` |
| 3 | `json_truncation` | `True` | `False` | `2048` | `wrapped` | `no` |
| 4 | `separation` | `True` | `False` | `None` | `wrapped` | `no` |

### `T2` — Thinking switch (chat_template_kwargs.enable_thinking) (Severity: `S3_GAP`)

**Hypothesis:** Setting enable_thinking=False disables reasoning and keeps completion_tokens <= 64, while enable_thinking=True populates a dedicated reasoning field.

| # | Variant | Thinking | Stream | Max Tokens | Shape | Core Escalation |
|---|---|---|---|---|---|---|
| 1 | `off` | `False` | `False` | `None` | `wrapped` | `no` |
| 2 | `on` | `True` | `False` | `None` | `wrapped` | `no` |
| 3 | `default` | `None` | `False` | `None` | `wrapped` | `no` |

### `T3` — Tool-argument type and markup fidelity (batch and streaming) (Severity: `S1_SILENT`)

**Hypothesis:** Server-side conversion from Nemotron's XML-style tool markup (<tool_call>, <function=...>, <parameter=...>) into OpenAI tool_calls JSON may coerce leading-zero strings to integers, lose 64-bit integer precision, or truncate strings containing XML/HTML delimiter tags.

| # | Variant | Thinking | Stream | Max Tokens | Shape | Core Escalation |
|---|---|---|---|---|---|---|
| 1 | `leading_zero` | `False` | `False` | `None` | `wrapped` | `no` |
| 2 | `leading_zero` | `True` | `False` | `None` | `wrapped` | `yes` |
| 3 | `leading_zero` | `True` | `True` | `None` | `wrapped` | `no` |
| 4 | `numeric_types` | `False` | `False` | `None` | `wrapped` | `no` |
| 5 | `numeric_types` | `True` | `False` | `None` | `wrapped` | `no` |
| 6 | `numeric_types` | `True` | `True` | `None` | `wrapped` | `no` |
| 7 | `bool_null_nested` | `False` | `False` | `None` | `wrapped` | `no` |
| 8 | `bool_null_nested` | `True` | `False` | `None` | `wrapped` | `no` |
| 9 | `bool_null_nested` | `True` | `True` | `None` | `wrapped` | `no` |
| 10 | `markup` | `False` | `False` | `None` | `wrapped` | `no` |
| 11 | `markup` | `True` | `False` | `None` | `wrapped` | `yes` |
| 12 | `markup` | `True` | `True` | `None` | `wrapped` | `no` |

### `T5` — Structured output JSON schema enforcement and request shapes (Severity: `S2_LOUD`)

**Hypothesis:** Structured output requests across wrapped, bare, and json_object shapes must return valid JSON conforming to Draft 2020-12 schema constraints (enums, additionalProperties: false, numeric/array bounds, regex patterns, $defs/$ref, and anyOf nullable fields) even when prompts tempt violations.

| # | Variant | Thinking | Stream | Max Tokens | Shape | Core Escalation |
|---|---|---|---|---|---|---|
| 1 | `flat_basic` | `False` | `False` | `None` | `wrapped` | `yes` |
| 2 | `flat_basic` | `True` | `False` | `None` | `wrapped` | `no` |
| 3 | `enum_temptation` | `False` | `False` | `None` | `wrapped` | `no` |
| 4 | `enum_temptation` | `True` | `False` | `None` | `wrapped` | `no` |
| 5 | `no_extra_props` | `False` | `False` | `None` | `wrapped` | `no` |
| 6 | `no_extra_props` | `True` | `False` | `None` | `wrapped` | `no` |
| 7 | `min_max` | `False` | `False` | `None` | `wrapped` | `no` |
| 8 | `min_max` | `True` | `False` | `None` | `wrapped` | `no` |
| 9 | `pattern` | `False` | `False` | `None` | `wrapped` | `no` |
| 10 | `pattern` | `True` | `False` | `None` | `wrapped` | `no` |
| 11 | `array_bounds` | `False` | `False` | `None` | `wrapped` | `no` |
| 12 | `array_bounds` | `True` | `False` | `None` | `wrapped` | `no` |
| 13 | `defs_ref` | `False` | `False` | `None` | `wrapped` | `yes` |
| 14 | `defs_ref` | `True` | `False` | `None` | `wrapped` | `no` |
| 15 | `anyof_optional` | `False` | `False` | `None` | `wrapped` | `yes` |
| 16 | `anyof_optional` | `True` | `False` | `None` | `wrapped` | `no` |
| 17 | `flat_basic` | `False` | `False` | `None` | `bare` | `no` |
| 18 | `flat_basic` | `False` | `False` | `None` | `json_object` | `no` |
| 19 | `flat_basic` | `False` | `True` | `None` | `wrapped` | `no` |

**Total Conditions per Model:** `38`
