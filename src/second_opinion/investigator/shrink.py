"""Shrink a failing request (CLAUDE.md §11.2). Deterministic; no LLM.

Candidates come from reducers tried in a fixed order. Each candidate is scored with k=12
trials and kept if it still fails at least max(1, ceil(0.5 * k * p0)) times, where p0 is the
cluster's failure rate in the cell being shrunk: failures are probabilistic, so nothing is
decided from one trial. At most 8 candidates (about 100 calls) per cluster.
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .spec import Spec, inline_refs
from .trials import TrialRunner

K = 12
MAX_CANDIDATES = 8
ANNOTATIONS = ("title", "description")


def threshold(p0: float, k: int = K) -> int:
    return max(1, math.ceil(0.5 * k * p0))


def involved_fields(examples: list[dict[str, Any]]) -> set[str]:
    """Top-level argument or property names that the cluster's failures point at."""
    names: set[str] = set()
    for ex in examples:
        detail = ex.get("detail") or {}
        for m in detail.get("mismatches") or []:
            names.add(str(m.get("path", "")).removeprefix("$.").split(".")[0].split("[")[0])
        for info in detail.values():
            if isinstance(info, dict):
                for err in info.get("errors") or []:
                    names.add(str(err.get("path", "")).removeprefix("$.").split(".")[0].split("[")[0])
    return {n for n in names if n and n != "$"}


def _drop_keys(spec: Spec, keep: set[str]) -> Spec | None:
    if spec.kind != "tool" or not spec.expected_args:
        return None
    drop = [k for k in spec.expected_args if k not in keep]
    if not drop or not keep & set(spec.expected_args):
        return None
    s = copy.deepcopy(spec)
    params = s.tool["function"]["parameters"]  # type: ignore[index]
    for k in drop:
        s.expected_args.pop(k)  # type: ignore[union-attr]
        params["properties"].pop(k, None)
    params["required"] = [k for k in params.get("required", []) if k not in drop]
    return s


def _markup_key(spec: Spec) -> str | None:
    if spec.kind != "tool" or not spec.expected_args:
        return None
    strings = {k: v for k, v in spec.expected_args.items() if isinstance(v, str) and len(v) > 1}
    return max(strings, key=lambda k: len(strings[k])) if strings else None


def _with_arg(spec: Spec, key: str, value: str) -> Spec:
    s = copy.deepcopy(spec)
    s.expected_args[key] = value  # type: ignore[index]
    return s


Reducer = Callable[[Spec, set[str]], list[tuple[str, Spec]]]


def r_stream_off(spec: Spec, involved: set[str]) -> list[tuple[str, Spec]]:
    return [("stream off", spec.with_factor("stream", False))] if spec.stream else []


def r_thinking_off(spec: Spec, involved: set[str]) -> list[tuple[str, Spec]]:
    return [("thinking off", spec.with_factor("thinking", False))] if spec.thinking else []


def r_drop_uninvolved_args(spec: Spec, involved: set[str]) -> list[tuple[str, Spec]]:
    s = _drop_keys(spec, involved)
    return [(f"keep only arguments {sorted(involved & set(spec.expected_args or {}))}", s)] if s else []


def r_halve_markup_lines(spec: Spec, involved: set[str]) -> list[tuple[str, Spec]]:
    key = _markup_key(spec)
    if not key:
        return []
    lines = spec.expected_args[key].split("\n")  # type: ignore[index]
    if len(lines) < 2:
        return []
    mid = len(lines) // 2
    return [(f"{key}: first {mid} lines", _with_arg(spec, key, "\n".join(lines[:mid]))),
            (f"{key}: last {len(lines) - mid} lines", _with_arg(spec, key, "\n".join(lines[mid:])))]


def r_halve_markup_chars(spec: Spec, involved: set[str]) -> list[tuple[str, Spec]]:
    key = _markup_key(spec)
    if not key:
        return []
    value = spec.expected_args[key]  # type: ignore[index]
    if "\n" in value or len(value) < 8:
        return []
    mid = len(value) // 2
    return [(f"{key}: first {mid} characters", _with_arg(spec, key, value[:mid])),
            (f"{key}: last {len(value) - mid} characters", _with_arg(spec, key, value[mid:]))]


def r_drop_uninvolved_properties(spec: Spec, involved: set[str]) -> list[tuple[str, Spec]]:
    if spec.kind != "json" or not spec.schema or not involved:
        return []
    props = spec.schema.get("properties", {})
    drop = [k for k in props if k not in involved]
    if not drop or not involved & set(props):
        return []
    s = copy.deepcopy(spec)
    for schema in (s.schema, s.oracle_schema):
        if schema:
            for k in drop:
                schema.get("properties", {}).pop(k, None)
            schema["required"] = [k for k in schema.get("required", []) if k not in drop]
    return [(f"keep only properties {sorted(involved & set(props))}", s)]


def r_drop_annotations(spec: Spec, involved: set[str]) -> list[tuple[str, Spec]]:
    if spec.kind != "json" or not spec.schema:
        return []
    s = copy.deepcopy(spec)
    for kw in ANNOTATIONS:
        s = s.with_factor("remove_schema_keyword", kw)
    return [("drop title/description annotations", s)] if s.schema != spec.schema else []


def r_inline_refs(spec: Spec, involved: set[str]) -> list[tuple[str, Spec]]:
    if spec.kind != "json" or not spec.schema or "$defs" not in spec.schema:
        return []
    s = copy.deepcopy(spec)
    s.schema = inline_refs(s.schema)
    return [("inline $ref", s)]


def r_shorten_prompt(spec: Spec, involved: set[str]) -> list[tuple[str, Spec]]:
    if spec.kind != "json" or not spec.prompt:
        return []
    sentences = [x for x in re.split(r"(?<=[.!?])\s+", spec.prompt.strip()) if x]
    if len(sentences) < 2:
        return []
    s = copy.deepcopy(spec)
    s.prompt = sentences[0]
    return [("first sentence of the prompt only", s)]


REDUCERS: dict[str, list[tuple[Reducer, bool]]] = {  # (reducer, repeat while it keeps succeeding)
    "tool": [(r_drop_uninvolved_args, False), (r_halve_markup_lines, True), (r_halve_markup_chars, True),
             (r_stream_off, False), (r_thinking_off, False)],
    "json": [(r_drop_uninvolved_properties, False), (r_drop_annotations, False), (r_inline_refs, False),
             (r_shorten_prompt, False)],
    "answer": [],
}


@dataclass
class ShrinkResult:
    spec: Spec
    rate: float
    n: int
    k: int
    steps: list[dict[str, Any]] = field(default_factory=list)
    examples: list[dict[str, Any]] = field(default_factory=list)
    stopped: str | None = None


async def shrink(spec: Spec, p0: float, n0: int, k0: int, involved: set[str], runner: TrialRunner,
                 k: int = K, max_candidates: int = MAX_CANDIDATES) -> ShrinkResult:
    thr = threshold(p0, k)
    result = ShrinkResult(spec, p0, n0, k0)
    tried = 0
    for reducer, repeat in REDUCERS[spec.kind]:
        while tried < max_candidates and result.stopped is None:
            accepted = False
            for desc, cand in reducer(result.spec, involved):
                if tried >= max_candidates:
                    break
                batch = await runner.run(cand, k, f"shrink:{desc}")
                tried += 1
                ok = batch.k >= thr and batch.n > 0
                result.steps.append({"candidate": desc, "k": batch.k, "n": batch.n, "threshold": thr, "accepted": ok})
                result.examples += batch.examples
                if batch.stopped:
                    result.stopped = batch.stopped
                    break
                if ok:
                    result.spec, result.rate, result.n, result.k = cand, batch.rate, batch.n, batch.k
                    accepted = True
                    break
            if not (accepted and repeat):
                break
    return result
