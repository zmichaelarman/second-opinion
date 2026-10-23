from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from second_opinion.config import REPO_ROOT


def get_git_info(repo_root: Path | None = None) -> dict[str, str | None]:
    root = repo_root or REPO_ROOT
    git_hash: str | None = None
    git_tag: str | None = None
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode == 0:
            git_hash = res.stdout.strip()
    except Exception:
        pass
    try:
        res_tag = subprocess.run(
            ["git", "tag", "--points-at", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res_tag.returncode == 0 and res_tag.stdout.strip():
            git_tag = res_tag.stdout.strip().splitlines()[0]
        else:
            res_all = subprocess.run(
                ["git", "tag", "-l", "prereg-v1"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if res_all.returncode == 0 and "prereg-v1" in res_all.stdout.splitlines():
                git_tag = "prereg-v1"
    except Exception:
        pass
    return {"git_hash": git_hash, "git_tag": git_tag}


@dataclass
class CallRecord:
    run_id: str
    stage: str
    cell_id: str
    contract_id: str
    model: str
    condition: dict[str, Any]
    trial: int
    attempt: int
    ts_start: str
    request: dict[str, Any]
    status: int
    headers: dict[str, str]
    raw: Any
    parsed: dict[str, Any]
    outcome: str
    codes: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] | None = None
    cost_usd: float = 0.0
    is_final: bool = True

    def to_json_line(self) -> str:
        # Ensure request body never includes auth headers
        req_clean = dict(self.request)
        req_clean.pop("headers", None)
        req_clean.pop("api_key", None)
        d = asdict(self)
        d["request"] = req_clean
        return json.dumps(d, ensure_ascii=False)


class RunStore:
    """Append-only run storage in runs/<YYYYMMDD-HHMM>-<stage>/ with resume support (§6, §8)."""

    def __init__(self, run_dir: Path, run_id: str, stage: str) -> None:
        self.run_dir = run_dir
        self.run_id = run_id
        self.stage = stage
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.plan_path = self.run_dir / "plan.json"
        self.calls_path = self.run_dir / "calls.jsonl"
        self.summary_path = self.run_dir / "summary.json"

    @classmethod
    def create_or_resume(
        cls,
        stage: str,
        runs_root: Path | None = None,
        resume_dir: Path | None = None,
    ) -> RunStore:
        root = runs_root or (REPO_ROOT / "runs")
        root.mkdir(parents=True, exist_ok=True)
        if resume_dir is not None:
            run_id = resume_dir.name
            return cls(resume_dir, run_id=run_id, stage=stage)
        ts = datetime.now().strftime("%Y%m%d-%H%M")
        base_name = f"{ts}-{stage}"
        candidate = root / base_name
        suffix = 1
        while candidate.exists() and (candidate / "summary.json").exists():
            suffix += 1
            candidate = root / f"{base_name}-{suffix}"
        return cls(candidate, run_id=candidate.name, stage=stage)

    def write_plan(self, plan_data: dict[str, Any]) -> None:
        if self.plan_path.exists():
            return
        git_info = get_git_info()
        payload = {
            "run_id": self.run_id,
            "stage": self.stage,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **git_info,
            **plan_data,
        }
        self.plan_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def append_call(self, record: CallRecord) -> None:
        with self.calls_path.open("a", encoding="utf-8") as f:
            f.write(record.to_json_line() + "\n")

    def load_calls(self, final_only: bool = False) -> list[dict[str, Any]]:
        if not self.calls_path.exists():
            return []
        records: list[dict[str, Any]] = []
        with self.calls_path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    rec = json.loads(s)
                except json.JSONDecodeError:
                    continue
                if final_only and not rec.get("is_final", True):
                    continue
                records.append(rec)
        return records

    def completed_trials(self) -> set[tuple[str, int]]:
        """Return set of (cell_id, trial) pairs that already have a final CallRecord."""
        done: set[tuple[str, int]] = set()
        for rec in self.load_calls(final_only=True):
            cid = rec.get("cell_id")
            trial = rec.get("trial")
            if cid is not None and trial is not None:
                done.add((str(cid), int(trial)))
        return done

    def write_summary(self, summary_data: dict[str, Any]) -> None:
        self.summary_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")


def load_records_from_runs(run_dirs: list[Path], final_only: bool = True) -> list[dict[str, Any]]:
    """Load and deduplicate final call records across multiple run directories."""
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    unkeyed: list[dict[str, Any]] = []
    for rdir in run_dirs:
        calls_file = rdir / "calls.jsonl" if rdir.is_dir() else rdir
        if not calls_file.exists():
            continue
        with calls_file.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    rec = json.loads(s)
                except json.JSONDecodeError:
                    continue
                if final_only and not rec.get("is_final", True):
                    continue
                cid = rec.get("cell_id")
                trial = rec.get("trial")
                if cid is not None and trial is not None:
                    by_key[(str(cid), int(trial))] = rec
                else:
                    unkeyed.append(rec)
    return list(by_key.values()) + unkeyed
