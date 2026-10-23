"""Experiments (CLAUDE.md §11.4). Nemotron 3 Super writes experiments/<cluster>_<h>.py using
only second_opinion.testkit. An AST safety gate checks the file before it runs; a failing
file gets up to 2 rewrites, after which the arms run directly and the fallback is logged.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable

from ..contracts.base import THINKING_SWITCH, TEMPERATURE, TOP_P, apply_thinking

ALLOWED_MODULE = "second_opinion.testkit"
# The spec's list, plus a few builtins that reach the same places (stricter is fine: experiments only need testkit).
BANNED_NAMES = frozenset({"open", "exec", "eval", "compile", "__import__", "getattr", "os", "sys", "subprocess", "socket",
                          "globals", "locals", "vars", "setattr", "delattr", "breakpoint", "input"})
MAX_LINES = 60
MAX_REWRITES = 2
RUN_TIMEOUT_S = 3600
MAX_TOKENS = 8192
_FENCED = re.compile(r"```(?:python|py)?[^\n`]*\n(.*?)\n?```", re.S)

TEMPLATE = '''from second_opinion.testkit import base_request, arm, compare

BASE = base_request("{cluster_id}")


def test_hypothesis():
    a = arm(BASE, factor="{factor}", level={level_a!r}, n=100)
    b = arm(BASE, factor="{factor}", level={level_b!r}, n=100)
    compare(a, b, hypothesis_id="{hypothesis_id}")
'''


def gate(source: str) -> list[str]:
    """Problems that stop an agent-written file from running. Empty list = allowed."""
    problems: list[str] = []
    if len(source.splitlines()) > MAX_LINES:
        problems.append(f"more than {MAX_LINES} lines")
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return problems + [f"syntax error: {e.msg} (line {e.lineno})"]
    has_test = calls_arm = calls_compare = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name != ALLOWED_MODULE:
                    problems.append(f"import {alias.name} is not allowed (only {ALLOWED_MODULE})")
        elif isinstance(node, ast.ImportFrom):
            if node.module != ALLOWED_MODULE or node.level != 0:
                problems.append(f"from {'.' * node.level}{node.module or ''} import ... is not allowed")
            for alias in node.names:
                if alias.name == "*" or alias.name.startswith("_"):
                    problems.append(f"importing {alias.name!r} is not allowed")
        elif isinstance(node, ast.Name):
            if node.id in BANNED_NAMES or node.id.startswith("__"):
                problems.append(f"name {node.id!r} is not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                problems.append(f"attribute {node.attr!r} is not allowed")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            has_test = True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            calls_arm |= node.func.id == "arm"
            calls_compare |= node.func.id == "compare"
    if not has_test:
        problems.append("no test_ function")
    if not (calls_arm and calls_compare):
        problems.append("must call arm() and compare()")
    return sorted(set(problems))


def code_prompt(cluster_id: str, hypothesis_id: str, hypothesis: dict[str, Any]) -> str:
    return (
        "Write a pytest file that tests one hypothesis about a failure cluster.\n\n"
        f"Hypothesis: changing `{hypothesis['factor']}` from {json.dumps(hypothesis['level_a'])} to "
        f"{json.dumps(hypothesis['level_b'])} changes the failure rate. Rationale: {hypothesis['rationale']}\n\n"
        "Rules: import only from second_opinion.testkit (base_request, arm, compare); no other imports; at most "
        f"{MAX_LINES} lines; no file, network or OS access; no names starting with an underscore.\n"
        "API: base_request(cluster_id) -> Base; arm(base, factor=..., level=..., n=100) runs n trials with one "
        "factor set; compare(a, b, hypothesis_id=...) records both arms and the Fisher p-value.\n\n"
        "Follow this shape exactly:\n\n```python\n"
        + TEMPLATE.format(cluster_id=cluster_id, hypothesis_id=hypothesis_id, factor=hypothesis["factor"],
                          level_a=hypothesis["level_a"], level_b=hypothesis["level_b"])
        + "```\n\nReply with exactly one fenced python code block."
    )


def code_request(model: str, messages: list[dict[str, str]]) -> dict[str, Any]:
    body: dict[str, Any] = {"model": model, "messages": messages, "temperature": TEMPERATURE, "top_p": TOP_P,
                            "max_tokens": MAX_TOKENS}
    return apply_thinking(body, True, THINKING_SWITCH)


def extract_code(reply: str | None) -> str | None:
    if not reply:
        return None
    m = _FENCED.search(reply)
    return m.group(1).strip() + "\n" if m else None


async def write_experiment(
    complete: Callable[[dict[str, Any]], Awaitable[str | None]],
    model: str,
    cluster_id: str,
    hypothesis_id: str,
    hypothesis: dict[str, Any],
    path: Path,
) -> dict[str, Any]:
    """Ask the code model for the file; gate it; ask for rewrites. Returns {ok, attempts, problems}."""
    messages = [
        {"role": "system", "content": "You write small, exact Python test files. You follow API rules precisely."},
        {"role": "user", "content": code_prompt(cluster_id, hypothesis_id, hypothesis)},
    ]
    problems: list[str] = []
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, MAX_REWRITES + 2):
        reply = await complete(code_request(model, messages))
        source = extract_code(reply)
        problems = ["no fenced python block in the reply"] if source is None else gate(source)
        draft = path.with_suffix(".py.tmp")
        if source is not None:
            draft.write_text(source)
        if not problems and source is not None:
            path.write_text(source)
            draft.unlink(missing_ok=True)
            return {"ok": True, "attempts": attempt, "problems": [], "path": str(path)}
        messages += [{"role": "assistant", "content": reply or ""},
                     {"role": "user", "content": "The safety gate rejected that file: " + "; ".join(problems)
                      + ". Rewrite it following the rules exactly; one fenced python block only."}]
    return {"ok": False, "attempts": MAX_REWRITES + 1, "problems": problems, "path": None}


def run_experiment(path: Path, root: Path, env_extra: dict[str, str], timeout: float = RUN_TIMEOUT_S) -> dict[str, Any]:
    """Run the gated file with pytest in a subprocess."""
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": os.environ.get("HOME", "/tmp"),
           "SECOND_OPINION_ROOT": str(root), **env_extra}
    for key in ("NEBIUS_API_KEY",):
        if os.environ.get(key):
            env[key] = os.environ[key]
    try:
        proc = subprocess.run([sys.executable, "-m", "pytest", "-q", str(path), "-p", "no:cacheprovider"],
                              cwd=root, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"returncode": None, "error": f"timed out after {timeout:.0f} s", "stdout": "", "stderr": ""}
    secret = os.environ.get("NEBIUS_API_KEY", "")
    scrub = (lambda t: t.replace(secret, "[REDACTED]")) if secret else (lambda t: t)
    return {"returncode": proc.returncode, "stdout": scrub(proc.stdout)[-4000:], "stderr": scrub(proc.stderr)[-4000:]}
