"""Cluster fatal failures by signature (CLAUDE.md §11.1).

Signature = (contract_id, sorted fatal codes). Keep clusters with at least 3 failures, or at
least 1 if any code is S1. Rank by severity, then by models affected x failure rate.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

from ..contracts.base import Severity
from ..oracles import is_fatal, severity_of
from ..stats import wilson_ci

MIN_FAILURES = 3
MAX_EXAMPLES = 3

PHRASES = {
    "TYPE_COERCED": "Tool argument changes JSON type",
    "VALUE_TRUNCATED": "Tool argument string is cut short",
    "VALUE_MISMATCH": "Tool argument value changes",
    "NO_TOOL_CALLS": "Forced tool call is missing",
    "TOOL_MARKUP_IN_CONTENT": "Raw tool-call markup leaks into content",
    "ARGS_JSON_INVALID": "Tool arguments are not valid JSON",
    "WRONG_TOOL": "Wrong tool called",
    "SCHEMA_INVALID": "Structured output violates the schema",
    "JSON_PARSE": "Structured output is not valid JSON",
    "REASONING_IN_CONTENT": "Reasoning leaks into content",
    "EMPTY_CONTENT": "Content is empty although finish_reason is stop",
    "EXTRA_CONTENT": "Answer carries extra text",
    "THINKING_NOT_DISABLED": "Thinking switch does not turn reasoning off",
    "NO_REASONING_FIELD": "Reasoning is not returned in a separate field",
    "STREAM_FINISH_MISSING": "Stream never sends a finish_reason",
    "HTTP_4XX": "Request shape is rejected (HTTP 4xx)",
}


def severity_rank(codes: Iterable[str]) -> Severity:
    sevs = [severity_of(c) for c in codes]
    return min((s for s in sevs if s is not None), default=Severity.S3_GAP)


def headline(codes: tuple[str, ...]) -> str:
    """A readable name for a signature, led by its most severe code."""
    primary = min(codes, key=lambda c: (severity_of(c) or Severity.S4_DOCS).value)
    base, _, kw = primary.partition(":")
    phrase = PHRASES.get(base, base)
    return f"{phrase} ({kw})" if kw else phrase


@dataclass
class Cluster:
    id: str
    contract_id: str
    codes: tuple[str, ...]
    severity: Severity
    failures: int = 0
    trials: int = 0
    cells: list[dict[str, Any]] = field(default_factory=list)
    examples: list[dict[str, Any]] = field(default_factory=list)

    @property
    def models(self) -> list[str]:
        return sorted({c["model"] for c in self.cells if c["k"] > 0})

    @property
    def rate(self) -> float:
        return self.failures / self.trials if self.trials else 0.0

    @property
    def title(self) -> str:
        return f"{headline(self.codes)}: {self.contract_id}"

    def as_dict(self) -> dict[str, Any]:
        lo, hi = wilson_ci(self.failures, self.trials)
        return {
            "id": self.id, "contract_id": self.contract_id, "codes": list(self.codes), "title": self.title,
            "severity": self.severity.name, "failures": self.failures, "trials": self.trials,
            "rate": self.rate, "wilson_lo": lo, "wilson_hi": hi, "models": self.models,
            "cells": self.cells, "examples": self.examples,
        }


def cluster_id(contract_id: str, codes: tuple[str, ...]) -> str:
    return hashlib.sha256(f"{contract_id}|{'+'.join(codes)}".encode()).hexdigest()[:10]


def build_clusters(records: Iterable[dict[str, Any]], min_failures: int = MIN_FAILURES) -> list[Cluster]:
    finals = [r for r in records if r.get("final", True) and r.get("outcome") in ("pass", "fail")]
    n_by_cell: dict[str, int] = defaultdict(int)
    cell_meta: dict[str, dict[str, Any]] = {}
    for r in finals:
        n_by_cell[r["cell_id"]] += 1
        cell_meta[r["cell_id"]] = {"model": r["model"], "condition": r.get("condition") or {}}
    clusters: dict[tuple[str, tuple[str, ...]], Cluster] = {}
    hits: dict[tuple[str, tuple[str, ...]], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in finals:
        fatal = tuple(sorted(c for c in r["codes"] if is_fatal(c)))
        if not fatal:
            continue
        key = (r["contract_id"], fatal)
        cl = clusters.get(key)
        if cl is None:
            cl = clusters[key] = Cluster(cluster_id(*key), r["contract_id"], fatal, severity_rank(fatal))
        cl.failures += 1
        hits[key][r["cell_id"]] += 1
        if len(cl.examples) < MAX_EXAMPLES:
            p = r.get("parsed") or {}
            cl.examples.append({
                "run_id": r.get("run_id"), "cell_id": r["cell_id"], "trial": r.get("trial"), "model": r["model"],
                "condition": r.get("condition"), "request": r.get("request"), "codes": r["codes"],
                "detail": r.get("detail"), "content_head": (p.get("content") or "")[:600],
                "reasoning_chars": len(p.get("reasoning") or ""), "tool_calls": p.get("tool_calls"),
                "finish_reason": p.get("finish_reason"), "status": r.get("status"),
            })
    kept = []
    for key, cl in clusters.items():
        if cl.failures < min_failures and cl.severity != Severity.S1_SILENT:
            continue
        for cell, k in sorted(hits[key].items()):
            n = n_by_cell[cell]
            lo, hi = wilson_ci(k, n)
            cl.cells.append({"cell_id": cell, **cell_meta[cell], "n": n, "k": k, "rate": k / n, "wilson_lo": lo, "wilson_hi": hi})
        cl.trials = sum(c["n"] for c in cl.cells)
        kept.append(cl)
    return sorted(kept, key=lambda c: (c.severity.value, -len(c.models) * c.rate, c.contract_id))
