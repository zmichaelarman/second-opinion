from __future__ import annotations

import difflib
import json
import re
from pathlib import Path
from typing import Any

from second_opinion.budget import BudgetGuard
from second_opinion.client import InferenceClient
from second_opinion.config import AppConfig
from second_opinion.contracts.base import Owner, Severity
from second_opinion.docs_check.execute import (
    execute_python_block,
    is_block_eligible_for_execution,
)
from second_opinion.docs_check.extract import CodeBlock
from second_opinion.docs_check.static import check_code_block_static


def _extract_first_fenced_block(text: str) -> str:
    m = re.search(r"```[a-zA-Z0-9_-]*\n(.*?)```", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip("\r\n")
    return text.strip()


def make_unified_diff(original: str, fixed: str, filename: str = "snippet") -> str:
    orig_lines = original.splitlines(keepends=True)
    fixed_lines = fixed.splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(
            orig_lines,
            fixed_lines,
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
        )
    )


async def repair_failing_block(
    block: CodeBlock,
    initial_codes: list[str],
    initial_error: str,
    known_models: list[str],
    config: AppConfig,
    client: InferenceClient,
    budget: BudgetGuard,
    run_id: str,
    execute_enabled: bool = False,
    max_attempts: int = 3,
) -> dict[str, Any]:
    """Repair a failing doc snippet using Nemotron 3 Super (thinking on), up to 3 attempts (§10.5)."""
    known_set = set(known_models)
    cur_code = block.code
    cur_error = initial_error
    cur_codes = list(initial_codes)
    attempts = 0
    fixed = False

    sample_models = known_models[:25]

    for attempt in range(1, max_attempts + 1):
        attempts = attempt
        prompt = (
            f"Page title: {block.page_title} ({block.page_url})\n"
            f"Language: {block.lang}\n"
            f"Detected error codes: {', '.join(cur_codes)}\n"
            f"Error details:\n{cur_error}\n\n"
            f"Currently listed models (sample): {', '.join(sample_models)}\n\n"
            f"Broken snippet:\n```{block.lang}\n{cur_code}\n```\n\n"
            "Return the MINIMAL corrected snippet that keeps the original intent, "
            "formatted as a single fenced code block."
        )
        req = {
            "model": config.code_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert developer documentation repair agent. "
                        "Produce a minimal fix for the broken code snippet. "
                        "Output exactly one fenced code block containing the corrected snippet."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "top_p": 0.95,
            "max_tokens": 2048,
            "chat_template_kwargs": {"enable_thinking": True},
        }

        est_cost = config.estimate_cost(config.code_model, 600, 1200)
        await budget.reserve(est_cost)
        resp = await client.chat_completion(req, estimated_tokens=1800)
        actual_cost = budget.compute_call_cost(config.code_model, resp.usage, 600, 1200)
        await budget.commit(est_cost, actual_cost, config.code_model, "docs_repair", run_id, resp.usage)

        candidate_code = _extract_first_fenced_block(resp.content)
        if not candidate_code:
            continue

        candidate_block = CodeBlock(
            page_url=block.page_url,
            page_title=block.page_title,
            block_idx=block.block_idx,
            lang=block.lang,
            raw_lang=block.raw_lang,
            line=block.line,
            code=candidate_code,
        )
        static_res = check_code_block_static(candidate_block, known_models=known_set)
        next_codes = list(static_res.codes)
        next_errors = list(static_res.errors)

        if not next_codes and execute_enabled and is_block_eligible_for_execution(candidate_block, next_codes):
            ok, exec_code, exec_err = execute_python_block(candidate_code)
            if not ok and exec_code:
                next_codes.append(exec_code)
                next_errors.append(exec_err)

        cur_code = candidate_code
        if not next_codes:
            fixed = True
            break
        cur_codes = next_codes
        cur_error = "\n".join(next_errors)

    patch = make_unified_diff(block.code, cur_code, filename=f"block_{block.block_idx}.{block.lang}")
    has_unknown_param = any(c.startswith("UNKNOWN_PARAM:") for c in initial_codes)
    severity = Severity.S2_LOUD.name if has_unknown_param else Severity.S4_DOCS.name

    return {
        "page_url": block.page_url,
        "page_title": block.page_title,
        "block_idx": block.block_idx,
        "lang": block.lang,
        "codes": initial_codes,
        "error": initial_error,
        "fixed": fixed,
        "attempts": attempts,
        "patch": patch,
        "severity": severity,
        "owner": Owner.NEBIUS_DOCS.value,
    }


def write_docs_findings(findings: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in findings:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
