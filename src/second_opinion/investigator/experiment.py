from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from second_opinion.budget import BudgetGuard
from second_opinion.client import InferenceClient
from second_opinion.config import AppConfig, REPO_ROOT
from second_opinion.testkit import arm, base_request, compare

BANNED_NAMES = {
    "open",
    "exec",
    "eval",
    "compile",
    "__import__",
    "getattr",
    "os",
    "sys",
    "subprocess",
    "socket",
}
ALLOWED_IMPORT_MODULE = "second_opinion.testkit"
MAX_EXPERIMENT_LINES = 60


def check_experiment_safety_gate(code: str) -> tuple[bool, list[str]]:
    """AST safety gate for agent-written pytest experiment files (§11.4).

    Rules:
    1. At most 60 lines.
    2. Syntax must parse via `ast.parse`.
    3. The ONLY import allowed is `second_opinion.testkit` (`from second_opinion.testkit import ...` or `import second_opinion.testkit`).
    4. Banned names: `open`, `exec`, `eval`, `compile`, `__import__`, `getattr`, `os`, `sys`, `subprocess`, `socket`.
    5. No attributes starting with `_`.
    """
    violations: list[str] = []
    lines = code.splitlines()
    if len(lines) > MAX_EXPERIMENT_LINES:
        violations.append(f"Exceeds {MAX_EXPERIMENT_LINES} lines (got {len(lines)} lines).")

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return (False, [f"SyntaxError at line {exc.lineno}: {exc.msg}"])

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name != ALLOWED_IMPORT_MODULE:
                    violations.append(
                        f"Disallowed import `{alias.name}`. Only `{ALLOWED_IMPORT_MODULE}` is permitted."
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module != ALLOWED_IMPORT_MODULE or node.level != 0:
                violations.append(
                    f"Disallowed import from `{node.module}`. Only `from {ALLOWED_IMPORT_MODULE} import ...` is permitted."
                )
        elif isinstance(node, ast.Name):
            if node.id in BANNED_NAMES:
                violations.append(f"Banned identifier `{node.id}` is not allowed.")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                violations.append(f"Attribute `{node.attr}` starting with `_` is not allowed.")
            if node.attr in BANNED_NAMES:
                violations.append(f"Banned attribute `{node.attr}` is not allowed.")

    return (len(violations) == 0, violations)


def render_canonical_experiment_code(
    cluster_id: str,
    hypothesis_id: str,
    factor: str,
    level_a: Any,
    level_b: Any,
    n: int = 30,
) -> str:
    return (
        "from second_opinion.testkit import base_request, arm, compare\n\n"
        f"BASE = base_request({cluster_id!r})\n\n\n"
        "def test_hypothesis():\n"
        f"    a = arm(BASE, factor={factor!r}, level={level_a!r}, n={n})\n"
        f"    b = arm(BASE, factor={factor!r}, level={level_b!r}, n={n})\n"
        f"    compare(a, b, hypothesis_id={hypothesis_id!r})\n"
    )


def _extract_python_code(text: str) -> str:
    m = re.search(r"```(?:python|py)?\n(.*?)```", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip() + "\n"
    return text.strip() + "\n"


async def generate_and_run_experiment(
    cluster_id: str,
    h_idx: int,
    hypothesis: dict[str, Any],
    config: AppConfig,
    client: InferenceClient,
    budget: BudgetGuard,
    run_id: str,
    context_path: Path,
    experiments_out_dir: Path,
    code_model: str | None = None,
    n_per_arm: int = 30,
) -> dict[str, Any]:
    """Have Nemotron 3 Super write `experiments/<cluster>_<h>.py`, validate with the AST safety gate
    (up to 2 rewrites), and execute with pytest in a subprocess (§11.4).
    """
    model = code_model or config.code_model
    hypothesis_id = f"{cluster_id}_h{h_idx}"
    exp_dir = REPO_ROOT / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    exp_file = exp_dir / f"{cluster_id}_h{h_idx}.py"

    factor = hypothesis["factor"]
    level_a = hypothesis["level_a"]
    level_b = hypothesis["level_b"]

    base_prompt = (
        f"Write a pytest file for cluster {cluster_id!r}, hypothesis {hypothesis_id!r}.\n"
        f"Factor: {factor!r}, level_a={level_a!r}, level_b={level_b!r}, n={n_per_arm}.\n"
        "You MUST only import from `second_opinion.testkit` (`base_request`, `arm`, `compare`).\n"
        "Do NOT import any other module, do not use any `_` attributes, and keep it under 60 lines.\n"
        "Template:\n"
        "```python\n"
        "from second_opinion.testkit import base_request, arm, compare\n"
        f'BASE = base_request("{cluster_id}")\n'
        "def test_hypothesis():\n"
        f'    a = arm(BASE, factor="{factor}", level={level_a!r}, n={n_per_arm})\n'
        f'    b = arm(BASE, factor="{factor}", level={level_b!r}, n={n_per_arm})\n'
        f'    compare(a, b, hypothesis_id="{hypothesis_id}")\n'
        "```"
    )

    gate_passed = False
    chosen_code = ""
    gate_errors: list[str] = []

    # Initial attempt + up to 2 rewrites if gate fails (§11.4)
    for try_idx in range(3):
        user_msg = base_prompt
        if gate_errors:
            user_msg += f"\n\nYour previous code failed the AST safety gate with: {gate_errors}. Fix all violations."

        req = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You write concise pytest experiment files using only second_opinion.testkit.",
                },
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.2,
            "top_p": 0.95,
            "max_tokens": 1024,
            "chat_template_kwargs": {"enable_thinking": True},
        }
        est_cost = config.estimate_cost(model, 400, 600)
        await budget.reserve(est_cost)
        resp = await client.chat_completion(req, estimated_tokens=1000)
        actual_cost = budget.compute_call_cost(model, resp.usage, 400, 600)
        await budget.commit(est_cost, actual_cost, model, "experiment_codegen", run_id, resp.usage)

        candidate = _extract_python_code(resp.content)
        ok, errs = check_experiment_safety_gate(candidate)
        if ok:
            gate_passed = True
            chosen_code = candidate
            break
        gate_errors = errs

    used_fallback = False
    if not gate_passed:
        used_fallback = True
        chosen_code = render_canonical_experiment_code(
            cluster_id, hypothesis_id, factor, level_a, level_b, n=n_per_arm
        )

    exp_file.write_text(chosen_code, encoding="utf-8")

    # Run with `pytest -q <file> -p no:cacheprovider` in a subprocess
    env = dict(os.environ)
    env["SECOND_OPINION_EXPERIMENT_CONTEXT"] = str(context_path)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(exp_file), "-p", "no:cacheprovider"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )

    result_file = experiments_out_dir / f"{hypothesis_id}.json"
    if result_file.exists():
        res_data = json.loads(result_file.read_text(encoding="utf-8"))
        res_data["experiment_file"] = str(exp_file.relative_to(REPO_ROOT))
        res_data["used_fallback"] = used_fallback
        res_data["pytest_returncode"] = proc.returncode
        res_data["rationale"] = hypothesis.get("rationale", "")
        return res_data

    # Direct fallback execution if pytest subprocess didn't produce output
    base_req = base_request(cluster_id)
    a_res = arm(base_req, factor=factor, level=level_a, n=n_per_arm)
    b_res = arm(base_req, factor=factor, level=level_b, n=n_per_arm)
    comp = compare(a_res, b_res, hypothesis_id=hypothesis_id)
    comp["experiment_file"] = str(exp_file.relative_to(REPO_ROOT))
    comp["used_fallback"] = True
    comp["rationale"] = hypothesis.get("rationale", "")
    return comp
