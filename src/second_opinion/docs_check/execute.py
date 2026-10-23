from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from second_opinion.config import get_nebius_api_key
from second_opinion.docs_check.extract import CodeBlock

ALLOWLIST_RE = re.compile(r"chat\.completions\.create|models\.list|embeddings\.create")
DENYLIST_RE = re.compile(r"fine_tuning|endpoints|files\.|batches|sandbox|contree|images\.generate")


def is_block_eligible_for_execution(block: CodeBlock, static_codes: list[str]) -> bool:
    """Eligible blocks are Python only, pass static checks (no PY_SYNTAX_ERROR),
    match allowlist, and do not match denylist (§10.4).
    """
    if block.lang != "python":
        return False
    if "PY_SYNTAX_ERROR" in static_codes:
        return False
    if not ALLOWLIST_RE.search(block.code):
        return False
    if DENYLIST_RE.search(block.code):
        return False
    return True


def execute_python_block(
    code: str,
    api_key: str | None = None,
    timeout_s: float = 90.0,
) -> tuple[bool, str | None, str]:
    """Run Python snippet in a subprocess in a temp directory with restricted env (§10.4).
    Returns (passed, failure_code, error_detail).
    """
    key = api_key or get_nebius_api_key()
    env: dict[str, str] = {
        "NEBIUS_API_KEY": key,
        "PATH": os.environ.get("PATH", ""),
    }
    # Preserve minimal OS variables needed by Python's socket/SSL initialization on Windows
    for win_var in ("SYSTEMROOT", "WINDIR", "USERPROFILE", "TEMP", "TMP", "HOMEDRIVE", "HOMEPATH"):
        if win_var in os.environ:
            env[win_var] = os.environ[win_var]

    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "snippet.py"
        script_path.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=tmpdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            if proc.returncode == 0:
                return (True, None, "")
            stderr = (proc.stderr or proc.stdout or "").strip()
            # Never leak API key in stderr
            if key and key in stderr:
                stderr = stderr.replace(key, "<REDACTED>")
            exc_type = _extract_exception_type(stderr)
            return (False, f"PY_RUNTIME_ERROR:{exc_type}", stderr[-1000:])
        except subprocess.TimeoutExpired:
            return (False, "PY_RUNTIME_ERROR:TimeoutExpired", f"Execution timed out after {timeout_s}s")


def _extract_exception_type(stderr: str) -> str:
    lines = [ln.strip() for ln in stderr.splitlines() if ln.strip()]
    for ln in reversed(lines):
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Warning))\b", ln)
        if m:
            return m.group(1).split(".")[-1]
    return "RuntimeError"
