"""Run directories: runs/<YYYYMMDD-HHMM>-<stage>/ with plan.json, calls.jsonl and summary.json.

Raw data is append-only: calls.jsonl is only ever appended to and plan.json is written once
(a resumed run writes plan-resume-<ts>.json next to it). summary.json is derived from
calls.jsonl, so a resumed run may rewrite it.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def git_info(root: Path) -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            r = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return r.stdout.strip() if r.returncode == 0 else None

    status = git("status", "--porcelain")
    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(status) if status is not None else None,
        "prereg_tag_commit": git("rev-parse", "-q", "--verify", "refs/tags/prereg-v1^{commit}"),
    }


def call_record(
    *,
    run_id: str,
    stage: str,
    cell_id: str,
    contract_id: str,
    model: str,
    condition: dict[str, Any] | None,
    trial: int,
    attempt: int,
    ts_start: str,
    request: dict[str, Any],
    parsed: Any,
    result: Any,
    cost_usd: float,
    final: bool = True,
    **extra: Any,
) -> dict[str, Any]:
    """One CallRecord (CLAUDE.md §6): one JSONL line per attempt. `request` is the body only."""
    return {
        "run_id": run_id,
        "stage": stage,
        "cell_id": cell_id,
        "contract_id": contract_id,
        "model": model,
        "condition": condition,
        "trial": trial,
        "attempt": attempt,
        "final": final,
        "ts_start": ts_start,
        "request": request,
        "status": parsed.status,
        "headers": parsed.headers,
        "raw": parsed.raw,
        "parsed": parsed.summary(),
        "outcome": result.outcome.value,
        "codes": result.codes,
        "detail": result.detail,
        "usage": parsed.usage,
        "cost_usd": round(cost_usd, 8),
        **extra,
    }


class RunStore:
    def __init__(self, run_dir: Path):
        self.dir = run_dir
        self.run_id = run_dir.name

    @classmethod
    def create(cls, runs_dir: Path, stage: str, now: datetime | None = None) -> "RunStore":
        stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M")
        run_dir, n = runs_dir / f"{stamp}-{stage}", 2
        while run_dir.exists():
            run_dir, n = runs_dir / f"{stamp}-{stage}-{n}", n + 1
        run_dir.mkdir(parents=True)
        return cls(run_dir)

    @classmethod
    def open(cls, run_dir: Path) -> "RunStore":
        if not (run_dir / "calls.jsonl").exists() and not (run_dir / "plan.json").exists():
            raise FileNotFoundError(f"{run_dir} is not a run directory")
        return cls(run_dir)

    @property
    def calls_path(self) -> Path:
        return self.dir / "calls.jsonl"

    def write_plan(self, plan: dict[str, Any]) -> Path:
        path = self.dir / "plan.json"
        if path.exists():
            path = self.dir / f"plan-resume-{datetime.now():%Y%m%d-%H%M%S}.json"
        path.write_text(json.dumps(plan, indent=2, ensure_ascii=False, default=str) + "\n")
        return path

    def read_plan(self) -> dict[str, Any]:
        return json.loads((self.dir / "plan.json").read_text())

    def append_call(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self.calls_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

    def iter_calls(self) -> Iterator[dict[str, Any]]:
        if not self.calls_path.exists():
            return
        with self.calls_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except ValueError:
                    continue  # a torn last line from a crash; the trial reruns on resume

    def final_keys(self) -> set[tuple[str, int]]:
        """(cell_id, trial) pairs that already have a final record."""
        return {(r["cell_id"], r["trial"]) for r in self.iter_calls() if r.get("final")}

    def write_summary(self, summary: dict[str, Any]) -> Path:
        path = self.dir / "summary.json"
        path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n")
        return path
