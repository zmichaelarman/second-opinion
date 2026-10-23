# Second Opinion

A CLI that checks whether Nebius Token Factory keeps its promises to developers building agents on NVIDIA Nemotron models: that JSON, tool calls, reasoning and streaming come back the way the docs say.

**Report:** <https://zmichaelarman.github.io/second-opinion/> (published from [`docs/`](docs/) by GitHub Pages)

Before a finding is blamed on anyone, it gets a second opinion: the same minimal request replayed on NVIDIA's own hosted endpoint for the same model.

## What it found

*Filled in after the screening and escalation runs. See [`docs/findings.md`](docs/findings.md) and [`FEEDBACK_DRAFT.md`](FEEDBACK_DRAFT.md).*

## Quickstart

Requires Python 3.11 or newer. `node` on PATH is optional (it enables syntax checks for the JavaScript docs examples).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env              # set NEBIUS_API_KEY; NVIDIA_API_KEY is only needed for `reference`
second-opinion models             # free: which Nemotron models will be tested
second-opinion smoke              # 15 calls or fewer: checks the plumbing end to end
second-opinion run --stage pilot  # prints the plan and estimated cost; add --yes to run it
pytest                            # offline tests (tests marked `live` are skipped by default)
```

Every command that calls a paid API prints the planned calls, models and estimated cost, then exits unless `--yes` is passed. Spend is recorded in an append-only ledger (`runs/ledger.jsonl`) and capped by `SECOND_OPINION_BUDGET_USD` (default $40).

## How it works

Nemotron emits raw text: reasoning inside `<think>` tags, and tool calls as Qwen3-Coder-style XML (`<tool_call>`, `<function=...>`, `<parameter=...>`). The serving platform converts that into OpenAI-shaped fields (`content`, `reasoning_content`, `tool_calls`). That conversion layer is where bugs cluster, and only the platform can fix it. Second Opinion tests it in three parts:

1. **Contract suite.** 17 deterministic contracts (38 conditions per model) cover reasoning leaks under token limits, the thinking switch, tool-argument fidelity (leading zeros, 2^53 + 1, `null`, nested objects, markup like `</parameter>` inside an argument), streaming assembly, and JSON Schema enforcement. They run at scale with confidence intervals: `screen` at n=50 per cell, then `escalate` to n=300 where failures appear. Pass/fail is decided by code (oracles), never by an LLM.
2. **Docs checker.** Extracts every code example from the Token Factory inference docs, checks it statically (Python syntax, cURL JSON bodies, unknown request parameters against `openapi.json`, model IDs against the live model list), executes eligible Python examples, and probes whether the API rejects or silently ignores unknown parameters. A repair agent fixes broken examples and proves each fix by re-running the checks.
3. **Investigator.** Root-causes each failure cluster in four steps:
   1. Shrink the failing request to a minimal one (deterministic: 12 trials per candidate).
   2. Propose up to three causes from a fixed factor menu (LLM).
   3. Write a pytest experiment against a small `testkit` API and run it (LLM-written code, behind an AST safety gate).
   4. Accept or reject each cause with Fisher's exact test (Holm-corrected).

   Accepted causes then go through the NVIDIA reference replay, which assigns an owner: Nebius platform, NVIDIA model, upstream serving stack, or unknown.

The **report** (`second-opinion report`) is a single static HTML page: a trust matrix per model and capability, findings with per-arm failure rates and minimal repros, docs health, and the raw data. It also drafts `FEEDBACK_DRAFT.md`.

## How Second Opinion uses NVIDIA Nemotron and Nebius Token Factory

- **Nemotron 3 Ultra: diagnosis.** In the Investigator, Ultra (thinking on, structured output) reads the minimal failing request with failing and passing responses, and proposes testable causes and a finding title.
- **Nemotron 3 Super: writes code.** Super writes the Investigator's pytest experiments, which run against `second_opinion.testkit`, and the minimal fixes for broken docs examples. Both are verified by running them.
- **Every listed Nemotron model is a test subject.** Nano, Lightning, Super and Ultra are discovered at run time from `GET /v1/models` and run through the full contract suite.
- **Everything runs through the Token Factory API.** All measured calls, and all of Second Opinion's own agent calls, go to `https://api.tokenfactory.nebius.com/v1/`. The only exception is the reference replay, which uses NVIDIA's hosted API to get the second opinion.

## Commands

| Command | What it does | Cost |
|---|---|---|
| `second-opinion models` | Discovery and filtering, as a table | free |
| `second-opinion smoke` | 15 calls or fewer: reasoning switch, reasoning field, one forced tool call per model, SDK view | cents |
| `second-opinion plan --write` | Writes `TEST_PLAN.md` with the registry's SHA-256, for pre-registration | free |
| `second-opinion run --stage {pilot,screen,escalate}` | Runs a stage (`--models`, `--contracts`, `--rpm`, `--concurrency`, `--max-n MODEL=N`, `--resume DIR`) | paid, needs `--yes` |
| `second-opinion docs [--execute] [--repair]` | Docs checker: static checks are free; executing and repairing are paid | free / paid |
| `second-opinion investigate` | Investigator on the most severe clusters (`--max-clusters`, `--cluster-budget`) | paid, needs `--yes` |
| `second-opinion reference` | NVIDIA reference replay and owner attribution | free tier, needs `--yes` |
| `second-opinion check --tools tools.json [--schemas schemas.json]` | Bring your own tools and schemas: per-tool, per-model reliability | paid, needs `--yes` |
| `second-opinion report --runs DIR... --out docs/` | Builds the static report, `docs/findings.md` and `FEEDBACK_DRAFT.md` | free |
| `second-opinion rerun` | Re-runs every finding's repro at n=30 and marks it still failing or fixed | paid, needs `--yes` |

## Methodology and pre-registration

- **Sampling:** temperature 1.0 and top_p 0.95 (NVIDIA's recommendation for Nemotron). max_tokens is 1024 with thinking off and 4096 with thinking on, unless a condition sets it.
- **Pre-registration:** `second-opinion plan --write` generates [`TEST_PLAN.md`](TEST_PLAN.md) from the contract registry: every contract, condition and exact request body, n, alpha, the comparison families and the trust-matrix rules, plus a SHA-256 of all of it. The plan is committed and tagged `prereg-v1` before screening, and the `screen` stage refuses to run if the code no longer matches the tagged hash.
- **Statistics:** Wilson intervals for every rate. Three pre-registered comparison families (streaming vs not, thinking on vs off, wrapped vs bare schema shape) use two-sided Fisher exact tests, Holm-corrected within each family at alpha = 0.01. A matrix cell is red when any condition's Wilson lower bound exceeds 1%. A cell with zero failures reports the rule-of-three bound.
- **Raw data:** every attempt is one JSONL record (request body, response headers, raw JSON or SSE chunks, verdict) under `runs/`, which is append-only. The report publishes it gzipped under `docs/data/`. No credentials are ever stored.

## Limitations

- **Light load only:** at most 40 requests per minute, slowing down when the platform's rate-limit headers say so. There is no load or stress testing, and latency is recorded but not reported as a result.
- **Global routing:** public serverless endpoints report region "Global", and processing location can change without notice.
- **The reference is not ground truth.** NVIDIA's hosted API is a different serving stack. Plain Nemotron 3 Nano is not listed there, so Nano findings keep owner "unknown".
- **Prices** come from `config/prices.yaml`, confirmed in the Token Factory console. A model without a price row is never run.
- **Not tested:** model intelligence, `tool_choice: "required"`, seeds, caching and other Tier 2/3 behaviors (CLAUDE.md §15).

## License

Apache-2.0. See [LICENSE](LICENSE).
