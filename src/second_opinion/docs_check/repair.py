"""Repair agent (CLAUDE.md §10.5). Part of the track requirement that agents write, run and test code.

For each failing block, Nemotron 3 Super (thinking on) gets the page title, the code, the
error and the current model list, and is asked for the minimal corrected snippet as one
fenced block. Each proposal is re-checked statically, and executed when eligible. Up to 3
attempts; each retry tells the model why the previous proposal failed.
"""

from __future__ import annotations

import dataclasses
import difflib
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from ..contracts.base import THINKING_SWITCH, TOP_P, TEMPERATURE, apply_thinking
from .execute import Execution, eligible
from .extract import CodeBlock
from .static import StaticResult

MAX_ATTEMPTS = 3
REPAIR_MAX_TOKENS = 8192
SYSTEM = (
    "You repair code examples from API documentation. Make the smallest change that fixes the reported "
    "problem and keeps the example's intent, style and comments. Reply with exactly one fenced code block."
)
_FENCED = re.compile(r"```[^\n`]*\n(.*?)\n?```", re.S)


@dataclass
class RepairResult:
    fixed: bool
    attempts: int
    code: str | None
    patch: str | None
    last_error: str | None
    history: list[dict[str, Any]] = field(default_factory=list)


def repair_prompt(page_title: str, page_url: str, block: CodeBlock, error: str, models: list[str]) -> str:
    listed = "\n".join(f"- {m}" for m in models) if models else "(unavailable)"
    return (
        f"Page: {page_title} ({page_url})\n"
        f"Language: {block.raw_lang or block.lang}\n\n"
        f"Problem found by our checker:\n{error}\n\n"
        f"Models currently listed by GET /v1/models:\n{listed}\n\n"
        f"Original example:\n```{block.raw_lang}\n{block.code}\n```\n\n"
        "Return the minimal corrected example that keeps the original intent. If a model is no longer listed, "
        "substitute a listed model of the same kind. Reply with exactly one fenced code block and nothing else."
    )


def extract_fenced(text: str | None) -> str | None:
    if not text:
        return None
    m = _FENCED.search(text)
    return m.group(1) if m else None


def unified_patch(block: CodeBlock, fixed: str) -> str:
    return "".join(difflib.unified_diff(
        block.code.splitlines(keepends=True) if block.code.endswith("\n") else (block.code + "\n").splitlines(keepends=True),
        fixed.splitlines(keepends=True) if fixed.endswith("\n") else (fixed + "\n").splitlines(keepends=True),
        fromfile=f"{block.page_url}#block{block.block_idx}", tofile="repaired",
    ))


def repair_request(model: str, messages: list[dict[str, str]]) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": REPAIR_MAX_TOKENS,
    }
    return apply_thinking(body, True, THINKING_SWITCH)


async def repair_block(
    block: CodeBlock,
    page_title: str,
    error: str,
    models: list[str],
    *,
    model: str,
    complete: Callable[[dict[str, Any], int], Awaitable[str | None]],
    recheck: Callable[[CodeBlock], StaticResult],
    execute: Callable[[str], Execution] | None,
) -> RepairResult:
    """`complete(body, attempt)` sends one chat request and returns the reply content."""
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": repair_prompt(page_title, block.page_url, block, error, models)},
    ]
    history: list[dict[str, Any]] = []
    problem = error
    for attempt in range(1, MAX_ATTEMPTS + 1):
        reply = await complete(repair_request(model, messages), attempt)
        candidate = extract_fenced(reply)
        if candidate is None:
            problem = "The reply did not contain a fenced code block."
        else:
            static = recheck(dataclasses.replace(block, code=candidate))
            if static.codes:
                problem = "; ".join(static.errors)
            elif execute and eligible(dataclasses.replace(block, code=candidate), static):
                run = execute(candidate)
                problem = None if run.ok else f"{run.code}: {run.error}"
            else:
                problem = None
        history.append({"attempt": attempt, "problem": problem, "candidate": candidate})
        if problem is None and candidate is not None:
            return RepairResult(True, attempt, candidate, unified_patch(block, candidate), None, history)
        messages += [
            {"role": "assistant", "content": reply or ""},
            {"role": "user", "content": f"That still fails our checker: {problem}\nTry again: one fenced code block only."},
        ]
    return RepairResult(False, MAX_ATTEMPTS, None, None, problem, history)
