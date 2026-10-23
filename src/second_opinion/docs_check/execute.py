"""Execute eligible Python examples (CLAUDE.md §10.4). Paid, so it follows the spending rule.

Eligible: Python that passes the static checks, matches the allowlist and not the denylist.
Each runs in a subprocess in a temp directory with only NEBIUS_API_KEY and PATH in its
environment and a 90 s timeout. Captured output is scrubbed of the key before it is stored.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .extract import CodeBlock
from .static import StaticResult

ALLOW = re.compile(r"chat\.completions\.create|models\.list|embeddings\.create")
DENY = re.compile(r"fine_tuning|endpoints|files\.|batches|sandbox|contree|images\.generate")
TIMEOUT_S = 90
EST_COST_PER_RUN_USD = 0.005  # snippets call arbitrary models; usage is not visible, so charge a flat estimate
_EXC = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt|Failure|Warning))\b")


@dataclass
class Execution:
    ok: bool
    code: str | None  # PY_RUNTIME_ERROR:<ExceptionType>
    error: str | None
    returncode: int | None
    seconds: float
    stdout_tail: str
    stderr_tail: str

    def as_dict(self) -> dict:
        return asdict(self)


def eligible(block: CodeBlock, static: StaticResult) -> bool:
    return (
        static.kind == "python"
        and not static.codes
        and bool(ALLOW.search(block.code))
        and not DENY.search(block.code)
    )


def exception_type(stderr: str) -> str:
    for line in reversed(stderr.strip().splitlines()):
        m = _EXC.match(line.strip())
        if m:
            return m.group(1).rsplit(".", 1)[-1]
    return "UnknownError"


def _scrub(text: str, secret: str) -> str:
    return text.replace(secret, "[REDACTED]") if secret else text


def run_snippet(code: str, api_key: str, timeout: float = TIMEOUT_S, python: str = sys.executable) -> Execution:
    env = {"NEBIUS_API_KEY": api_key, "PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "snippet.py").write_text(code)
        try:
            proc = subprocess.run([python, "snippet.py"], cwd=tmp, env=env, capture_output=True, text=True,
                                  timeout=timeout)
        except subprocess.TimeoutExpired as e:
            out = _scrub((e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or ""), api_key)
            return Execution(False, "PY_RUNTIME_ERROR:TimeoutExpired", f"no exit after {timeout:.0f} s", None,
                             time.monotonic() - t0, out[-2000:], "")
    stdout, stderr = _scrub(proc.stdout, api_key), _scrub(proc.stderr, api_key)
    seconds = time.monotonic() - t0
    if proc.returncode == 0:
        return Execution(True, None, None, 0, seconds, stdout[-2000:], stderr[-2000:])
    exc = exception_type(stderr)
    last = next((ln for ln in reversed(stderr.strip().splitlines()) if ln.strip()), "")
    return Execution(False, f"PY_RUNTIME_ERROR:{exc}", last[:500], proc.returncode, seconds, stdout[-2000:], stderr[-2000:])
