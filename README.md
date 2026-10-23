# Second Opinion

**Second Opinion** is a CLI that checks whether Nebius Token Factory keeps its promises to developers building agents on NVIDIA Nemotron models: that structured JSON, tool calls, reasoning separation, and streaming come back the way the docs say — and gets a "second opinion" from NVIDIA's reference API before attributing any defect.

- **Static HTML Demo Report:** [`docs/index.html`](docs/index.html) (served via GitHub Pages from `/docs`)

---

## What It Found

1. **Documentation & Request-Shape Defects (`NEBIUS_DOCS` / S4–S2):** In `ai-models-inference/json.md`, the Python structured output snippet contains syntax errors (missing commas inside and after `response_format`), the cURL snippet uses `"response format"` (with a space) and invalid JSON syntax, and examples mix bare (`"json_schema": <schema>`) and OpenAI-wrapped (`"json_schema": {"name", "schema", "strict"}`) shapes without documenting compatibility semantics.
2. **Conversion-Layer Tool & Reasoning Findings (Pending Live Run):** Screening (`n=50`) and escalation (`n=300`) findings across `T1` (reasoning leaks under token limits), `T2` (thinking switch), `T3`/`T4` (tool-argument fidelity & streaming assembly), and `T5` (JSON schema enforcement) are populated automatically by `second-opinion report`.
3. **Reference Replay Attribution (Pending Live Run):** Confirmed failure clusters are replayed at `n=30` against NVIDIA's hosted endpoint (`integrate.api.nvidia.com`) and tested with two-sided Fisher's exact test to separate platform conversion bugs (`NEBIUS_PLATFORM`) from upstream serving (`UPSTREAM_SERVING`) and model copy errors (`NVIDIA_MODEL`).

---

## Quickstart

```bash
# 1. Install in editable mode
pip install -e .

# 2. Configure API keys and budget cap
cp .env.example .env
# Edit .env to set NEBIUS_API_KEY, NVIDIA_API_KEY, and SECOND_OPINION_BUDGET_USD=40

# 3. Discover listed Nemotron models
second-opinion models

# 4. Run the <=15-call smoke check
second-opinion smoke

# 5. Preview and run the pilot stage (prints cost estimate first; pass --yes after approval)
second-opinion run --stage pilot
```

---

## How It Works

Second Opinion targets the **conversion layer** where raw Nemotron tokens (`<think>...</think>` blocks and XML-style `<tool_call><function=...><parameter=...>` markup) are parsed into OpenAI-compatible fields on the server:

1. **Contract Suite (`T1`–`T5`):** Deterministic tests of reasoning separation (`T1`), `chat_template_kwargs.enable_thinking` control (`T2`), tool-argument type/markup fidelity in batch and SSE streaming modes (`T3`/`T4`), and JSON Schema enforcement (`T5`), evaluated with Wilson confidence intervals, Clopper-Pearson intervals, and Rule-of-Three upper bounds.
2. **Docs Checker (`second-opinion docs`):** Fetches Token Factory markdown docs via `llms.txt`, extracts Python/cURL/JS snippets, runs static AST/JSON/OpenAPI parameter/model-ID checks, optionally executes safe snippets in isolated subprocesses (`--execute`), and uses an agent loop on Nemotron 3 Super to repair broken examples (`--repair`).
3. **Investigator (`second-opinion investigate`) & Reference Replay (`second-opinion reference`):**
   - **Shrink:** Deterministically minimizes failing requests (delta-debugging on markup lines/chars, dropping uninvolved schema fields/arguments) using probabilistic multi-trial scoring (`k=12`).
   - **Hypothesize:** Uses Nemotron 3 Ultra (with structured JSON output and reasoning enabled) to propose controlled factor experiments from a pre-defined factor menu.
   - **Experiment:** Uses Nemotron 3 Super to author pytest experiment files (`experiments/<cluster>_<h>.py`) restricted by an AST safety gate to `second_opinion.testkit`.
   - **Decide & Replay:** Evaluates arms with two-sided Fisher's exact test + Holm-Bonferroni correction ($\alpha = 0.01$, $|\Delta p| \ge 0.05$) and replays minimal repros against NVIDIA's API.

---

## How Second Opinion uses NVIDIA Nemotron and Nebius Token Factory

- **NVIDIA Nemotron 3 Ultra (`nvidia/Nemotron-3-Ultra-550b-a55b`):** Serves as the diagnostic reasoning engine in the Investigator (`hypothesize.py`), analyzing minimal failing requests alongside passing/failing response traces to propose testable root-cause hypotheses.
- **NVIDIA Nemotron 3 Super (`nvidia/nemotron-3-super-120b-a12b`):** Powers code generation in two agentic loops — authoring `pytest` factorial experiments (`experiment.py`) against `second_opinion.testkit` and repairing broken documentation code examples (`docs_check/repair.py`) verified by static and runtime checks.
- **All Listed NVIDIA Nemotron Models:** Every discovered text/chat Nemotron model on Token Factory (`Nemotron 3 Nano`, `Nemotron 3.5 Lightning`, `Nemotron 3 Super`, `Nemotron 3 Ultra`) is evaluated as a **test subject** across 38 conditions (152 cells).
- **Nebius Token Factory API (`https://api.tokenfactory.nebius.com/v1/`):** All model discovery, contract evaluation, doc repair, and investigator calls run through Token Factory's OpenAI-compatible inference endpoints with header-aware rate pacing (`x-ratelimit-*`) and persistent spend tracking (`runs/ledger.json`).

---

## Methodology and Pre-Registration

- **Deterministic Oracles Only:** Pass/fail outcomes and failure codes (`TYPE_COERCED`, `VALUE_TRUNCATED`, `REASONING_IN_CONTENT`, `SCHEMA_INVALID:<kw>`, etc.) are computed purely by deterministic Python oracles (`oracles.py`). An LLM never grades a test.
- **Pre-Registration (`TEST_PLAN.md`):** Before running the `screen` stage (`n=50` per cell), `second-opinion plan --write` serializes all contracts, conditions, sample sizes, $\alpha = 0.01$, Holm-corrected comparison families (`F1` streaming vs non-streaming, `F2` thinking on vs off, `F3` wrapped vs bare schema shape), and a SHA-256 registry hash into `TEST_PLAN.md`, tagged `prereg-v1`.
- **Two-Stage Sampling:** Screening runs `n=50` per cell; any cell with $\ge 1$ fatal failure or designated as a core condition escalates to `n=300` total trials.

---

## Limitations

- **Light-Load Correctness Only:** Pacing stays strictly within provider rate limits (default 40 RPM, header-adaptive). We do not load-test or stress-test Token Factory.
- **Global Serverless Routing:** Token Factory public serverless endpoints report region `"Global"`, and underlying serving instances can change without notice.
- **Sampling Defaults:** Unless a factor experiment varies temperature, requests use NVIDIA's recommended `temperature=1.0, top_p=0.95`.
- **Reference Stack Differences:** NVIDIA's `integrate.api.nvidia.com` endpoint uses a separate serving stack and is treated as a diagnostic reference for attribution, not ground truth.

---

## License

Licensed under the [Apache License, Version 2.0](LICENSE).

