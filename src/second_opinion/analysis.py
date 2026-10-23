"""Aggregate CallRecords from run directories into per-cell statistics."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from .oracles import is_fatal
from .stats import wilson_ci


def iter_records(run_dirs: Iterable[Path], final_only: bool = True) -> Iterator[dict[str, Any]]:
    for d in run_dirs:
        path = Path(d) / "calls.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if not final_only or r.get("final", True):
                    yield r


def stage_runs(runs_dir: Path, stage: str) -> list[Path]:
    """Run directories for a stage, oldest first (names start with a timestamp)."""
    if not runs_dir.exists():
        return []
    return sorted(d for d in runs_dir.iterdir() if d.is_dir() and d.name.split("-")[2:3] == [stage])


@dataclass
class CellStats:
    cell_id: str
    contract_id: str
    model: str
    condition: dict[str, Any]
    n: int = 0  # final PASS or FAIL trials (the denominator)
    fails: int = 0  # final trials with at least one fatal code
    errors: int = 0  # final ERROR trials (excluded from the denominator)
    max_trial: int = -1
    codes: Counter = field(default_factory=Counter)
    fatal_codes: Counter = field(default_factory=Counter)
    signatures: Counter = field(default_factory=Counter)  # sorted fatal codes per failing trial
    total_s: list[float] = field(default_factory=list)
    ttft_s: list[float] = field(default_factory=list)
    prompt_tokens: list[int] = field(default_factory=list)
    completion_tokens: list[int] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.fails / self.n if self.n else 0.0

    @property
    def wilson(self) -> tuple[float, float]:
        return wilson_ci(self.fails, self.n)

    def as_dict(self) -> dict[str, Any]:
        lo, hi = self.wilson
        return {
            "cell_id": self.cell_id,
            "contract_id": self.contract_id,
            "model": self.model,
            "condition": self.condition,
            "n": self.n,
            "fails": self.fails,
            "errors": self.errors,
            "rate": round(self.rate, 6),
            "wilson_lo": round(lo, 6),
            "wilson_hi": round(hi, 6),
            "codes": dict(self.codes),
            "fatal_codes": dict(self.fatal_codes),
            "signatures": dict(self.signatures),
            "median_total_s": round(statistics.median(self.total_s), 3) if self.total_s else None,
            "median_completion_tokens": statistics.median(self.completion_tokens) if self.completion_tokens else None,
        }


def aggregate(records: Iterable[dict[str, Any]]) -> dict[str, CellStats]:
    cells: dict[str, CellStats] = {}
    for r in records:
        s = cells.get(r["cell_id"])
        if s is None:
            s = cells[r["cell_id"]] = CellStats(r["cell_id"], r["contract_id"], r["model"], r.get("condition") or {})
        s.max_trial = max(s.max_trial, r.get("trial", -1))
        if r["outcome"] == "error":
            s.errors += 1
            continue
        s.n += 1
        s.codes.update(r["codes"])
        fatal = sorted(c for c in r["codes"] if is_fatal(c))
        if fatal:
            s.fails += 1
            s.fatal_codes.update(fatal)
            s.signatures["+".join(fatal)] += 1
        parsed = r.get("parsed") or {}
        if parsed.get("total_s") is not None:
            s.total_s.append(parsed["total_s"])
        if parsed.get("ttft_s") is not None:
            s.ttft_s.append(parsed["ttft_s"])
        usage = r.get("usage") or {}
        if isinstance(usage.get("prompt_tokens"), int):
            s.prompt_tokens.append(usage["prompt_tokens"])
        if isinstance(usage.get("completion_tokens"), int):
            s.completion_tokens.append(usage["completion_tokens"])
    return cells


THINKING_LABEL = {True: "on", False: "off", None: "default"}


def token_key(contract_id: str, thinking: bool | None) -> str:
    return f"{contract_id}|{THINKING_LABEL[thinking]}"


def token_medians(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Median prompt/completion tokens per (contract, thinking), for cost estimates."""
    buckets: dict[str, tuple[list[int], list[int]]] = {}
    for r in records:
        usage = r.get("usage") or {}
        pt, ct = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if not (isinstance(pt, int) and isinstance(ct, int)):
            continue
        key = token_key(r["contract_id"], (r.get("condition") or {}).get("thinking"))
        p, c = buckets.setdefault(key, ([], []))
        p.append(pt)
        c.append(ct)
    return {k: {"prompt": statistics.median(p), "completion": statistics.median(c), "n": len(p)} for k, (p, c) in buckets.items()}
