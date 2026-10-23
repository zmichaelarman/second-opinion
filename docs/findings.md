# Second Opinion — Findings Summary

- **Generated:** 2026-10-23 16:04 UTC
- **Git Hash:** `uncommitted`
- **Registry SHA-256:** `f1f832eafedb3de9477c2c1ccdd9a5cbd1e7b617b3b3160e9c9d69af1f73d912`

## 1. Trust Matrix

| Model | Structured output (T5, thinking off) | Structured output with reasoning (T5, thinking on) | Tool arguments (T3, non-stream) | Tool arguments, streaming (T3 stream) | Reasoning control (T2) | Reasoning leak (T1) |
|---|---|---|---|---|---|---|
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` | **GRAY** (no data) | **GRAY** (no data) | **GRAY** (no data) | **GRAY** (no data) | **GRAY** (no data) | **GRAY** (no data) |
| `nvidia/Nemotron-3_5-Lightning` | **GRAY** (no data) | **GRAY** (no data) | **GRAY** (no data) | **GRAY** (no data) | **GRAY** (no data) | **GRAY** (no data) |
| `nvidia/nemotron-3-super-120b-a12b` | **GRAY** (no data) | **GRAY** (no data) | **GRAY** (no data) | **GRAY** (no data) | **GRAY** (no data) | **GRAY** (no data) |
| `nvidia/Nemotron-3-Ultra-550b-a55b` | **GRAY** (no data) | **GRAY** (no data) | **GRAY** (no data) | **GRAY** (no data) | **GRAY** (no data) | **GRAY** (no data) |

## 2. Confirmed Findings

_No failure clusters recorded yet._

## 3. Documentation Findings

- `https://docs.tokenfactory.nebius.com/ai-models-inference/json.md` (block #1, python): `PY_SYNTAX_ERROR` — Repaired: `False`
- `https://docs.tokenfactory.nebius.com/ai-models-inference/json.md` (block #2, bash): `CURL_JSON_INVALID, UNKNOWN_PARAM:response format` — Repaired: `False`
- `https://docs.tokenfactory.nebius.com/api-reference/examples/text-generation.md` (block #1, bash): `CURL_JSON_INVALID` — Repaired: `False`
- `https://docs.tokenfactory.nebius.com/api-reference/examples/text-generation.md` (block #2, js): `JS_SYNTAX_ERROR` — Repaired: `False`
- `https://docs.tokenfactory.nebius.com/api-reference/examples/vision-capabilities.md` (block #2, js): `JS_SYNTAX_ERROR` — Repaired: `False`
- `https://docs.tokenfactory.nebius.com/api-reference/examples/text-generation-for-code-autocomplete.md` (block #1, bash): `CURL_JSON_INVALID` — Repaired: `False`
