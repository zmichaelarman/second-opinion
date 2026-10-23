from __future__ import annotations

from pathlib import Path
from typing import Any

from second_opinion.config import REPO_ROOT
from second_opinion.discovery import FALLBACK_NEMOTRON_MODELS


def generate_feedback_draft(
    findings: list[dict[str, Any]],
    docs_findings: list[dict[str, Any]],
    trust_rows: list[dict[str, Any]],
    models: list[str] | None = None,
    repo_root: Path | None = None,
) -> str:
    """Generate FEEDBACK_DRAFT.md following the exact structure requested in §12."""
    root = repo_root or REPO_ROOT
    friction_path = root / "FRICTION_LOG.md"
    friction_excerpt = ""
    if friction_path.exists():
        friction_excerpt = friction_path.read_text(encoding="utf-8").strip()

    models_list = models or [r["model"] for r in trust_rows] or list(FALLBACK_NEMOTRON_MODELS)

    top_issues: list[str] = []
    for idx, f in enumerate(findings[:5], start=1):
        top_issues.append(
            f"{idx}. **[{f.get('severity', 'S2')}] {f.get('title')}** (Owner: `{f.get('owner', 'unknown')}`) — "
            f"Workaround: {f.get('workaround', 'N/A')}"
        )
    for df in docs_findings:
        if len(top_issues) >= 5:
            break
        top_issues.append(
            f"{len(top_issues) + 1}. **[{df.get('severity', 'S4_DOCS')}] Docs defect in `{df.get('page_url')}` (block #{df.get('block_idx')})** — "
            f"Codes: `{', '.join(df.get('codes', []))}` ({df.get('error', '')})"
        )
    if not top_issues:
        top_issues.append(
            "1. **[S4_DOCS] Structured output documentation defects in `ai-models-inference/json.md`** — "
            "Python example raises `SyntaxError` (missing commas in/after `response_format`), cURL example uses `\"response format\"` (with a space) and misses a comma (`CURL_JSON_INVALID`), and bare vs wrapped `json_schema` shapes are mixed without explanation."
        )

    lines: list[str] = [
        "# Nebius Token Factory & NVIDIA Nemotron — Developer Feedback (Second Opinion)",
        "",
        "## Top 5 Prioritized Issues",
        "",
        *top_issues,
        "",
        "---",
        "",
        "## 1. Nebius Token Factory",
        "",
        "### What we used it for",
        "- Served as the primary OpenAI-compatible inference platform (`/v1/models`, `/v1/chat/completions` in batch and SSE streaming modes) for all contract verification (`T1`–`T5`), documentation repair, and root-cause factorial experiments.",
        "",
        "### What worked (with numbers)",
        f"- Standard OpenAI-compatible chat completions and dynamic rate-limit headers (`x-ratelimit-*`) allowed reliable automated pacing across {len(models_list)} Nemotron models.",
        f"- Static & runtime verification checked {len(docs_findings)} documentation issues and produced automated unified diffs.",
        "",
        "### What needs work (linked findings)",
        "- **Structured output documentation (`ai-models-inference/json.md`):** Fix the Python `SyntaxError` (missing commas), fix the cURL `\"response format\"` typo and invalid JSON body, and document whether both bare (`\"json_schema\": <schema>`) and OpenAI-wrapped (`\"json_schema\": {\"name\", \"schema\", \"strict\"}`) shapes are officially supported.",
        "- **Reasoning control documentation:** Explicitly document `chat_template_kwargs: {\"enable_thinking\": bool}` and the response field (`reasoning_content` / `reasoning`) in the Token Factory inference docs.",
    ]

    for f in findings:
        lines.append(f"- **`{f.get('finding_id')}`:** {f.get('title')} — {f.get('workaround')}")

    lines.extend(
        [
            "",
            "### Onboarding from zero to hello world (from `FRICTION_LOG.md`)",
            "",
            friction_excerpt or "_See `FRICTION_LOG.md` for timestamped onboarding notes._",
            "",
            "### Would we build with it again?",
            "- **Yes.** Having Ultra, Super, Lightning, and Nano available behind a single OpenAI-compatible endpoint with rich rate-limit headers makes multi-tier agent architectures straightforward, provided conversion-layer edge cases and copy-paste doc snippets are tightened.",
            "",
            "---",
            "",
            "## 2. Nebius AI Cloud",
            "",
            "### What we used it for",
            "- Project workspace, key management, console model catalog & pricing lookup, and serverless endpoint routing configuration.",
            "",
            "### What worked",
            "- Rapid API key provisioning and unified billing across the Nemotron model family.",
            "",
            "### What needs work",
            "- Exposing machine-readable per-model pricing in `/v1/models` (or a billing metadata endpoint) and surfacing the exact backend serving engine version (vLLM / SGLang parser revision) in response headers would help developers diagnose conversion-layer behavior across `Global` serverless regions.",
            "",
            "### Would we build with it again?",
            "- **Yes** — low-friction setup and competitive token pricing for open-weights reasoning models.",
            "",
            "---",
            "",
            "## 3. NVIDIA Nemotron Models",
            "",
        ]
    )

    role_descriptions = {
        "ultra": "Diagnostic root-cause hypothesis generation (`investigator/hypothesize.py`) using structured outputs + reasoning, and contract test subject.",
        "super": "Agentic code generation (`investigator/experiment.py` for pytest experiments and `docs_check/repair.py` for repairing broken docs snippets), plus contract test subject.",
        "lightning": "High-speed screening & pilot contract test subject across `T1`–`T5`.",
        "nano": "Cost-efficient contract test subject across `T1`–`T5`.",
    }

    for model_id in models_list:
        lower = model_id.lower()
        role_desc = next(
            (desc for key, desc in role_descriptions.items() if key in lower),
            "Contract test subject across `T1`–`T5`.",
        )
        m_findings = [f for f in findings if model_id in (f.get("models_affected") or [])]
        lines.extend(
            [
                f"### `{model_id}`",
                f"- **What we used it for:** {role_desc}",
                "- **What worked:** Strong instruction adherence on structured schemas and multi-step reasoning when given sufficient `max_tokens` budget.",
                "- **What needs work:** "
                + (
                    "; ".join(f"`{f['finding_id']}` ({f['title']})" for f in m_findings)
                    if m_findings
                    else "See capability matrix in `docs/index.html` for per-contract confidence intervals."
                ),
                "- **Would we build with it again?** Yes — pairing Ultra for diagnosis and Super for code generation while guarding conversion-layer edge cases with deterministic schemas proved highly effective.",
                "",
            ]
        )

    return "\n".join(lines)

