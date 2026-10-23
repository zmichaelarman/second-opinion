"""Contract building blocks (CLAUDE.md §6) and the shared request rules (§7)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from ..parse import ParsedResponse


class Outcome(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"


class Severity(int, Enum):
    S1_SILENT = 1
    S2_LOUD = 2
    S3_GAP = 3
    S4_DOCS = 4


class Owner(str, Enum):
    NEBIUS_PLATFORM = "nebius_platform"
    NEBIUS_DOCS = "nebius_docs"
    NVIDIA_MODEL = "nvidia_model"
    UPSTREAM_SERVING = "upstream_serving"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Condition:
    thinking: bool | None  # None = omit the thinking switch
    stream: bool = False
    max_tokens: int | None = None
    variant: str = "default"  # payload / schema variant
    shape: str = "wrapped"  # T5 request shape: wrapped | bare | json_object

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Condition":
        return cls(**d)

    def label(self) -> str:
        parts = [{None: "think=default", True: "think=on", False: "think=off"}[self.thinking]]
        if self.stream:
            parts.append("stream")
        if self.max_tokens is not None:
            parts.append(f"max_tokens={self.max_tokens}")
        if self.shape != "wrapped":
            parts.append(f"shape={self.shape}")
        return " ".join(parts)


def cell_id(contract_id: str, model: str, cond: Condition) -> str:
    """Stable short hash of (contract_id, model, condition)."""
    blob = json.dumps({"contract": contract_id, "model": model, "condition": cond.as_dict()}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


@dataclass
class OracleResult:
    outcome: Outcome
    codes: list[str] = field(default_factory=list)  # fatal and soft codes, in check order
    detail: dict[str, Any] = field(default_factory=dict)


class Contract(Protocol):
    id: str
    tier: int
    title: str
    hypothesis: str
    severity: Severity
    core_conditions: set[Condition]
    conditions: list[Condition]

    def build_request(self, model: str, cond: Condition, trial: int) -> dict: ...

    def check(self, req: dict, resp: "ParsedResponse") -> OracleResult: ...


@dataclass(eq=False)
class BaseContract:
    """Shared plumbing for the Tier 1 contracts. Subclasses implement build_request and check."""

    id: str
    tier: int
    title: str
    hypothesis: str
    severity: Severity
    conditions: list[Condition]
    core_conditions: set[Condition] = field(default_factory=set)

    def build_request(self, model: str, cond: Condition, trial: int) -> dict[str, Any]:
        raise NotImplementedError

    def check(self, req: dict[str, Any], resp: "ParsedResponse") -> OracleResult:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        """The pre-registered definition, including the exact request bodies (hashed into TEST_PLAN.md)."""
        return {
            "id": self.id,
            "tier": self.tier,
            "title": self.title,
            "hypothesis": self.hypothesis,
            "severity": self.severity.name,
            "conditions": [c.as_dict() for c in self.conditions],
            "core_conditions": sorted(
                (c.as_dict() for c in self.core_conditions), key=lambda d: json.dumps(d, sort_keys=True)
            ),
            "requests": [self.build_request("<model>", c, 0) for c in self.conditions],
        }


# Shared request rules (§7). These are part of the pre-registered design, so they live in
# code (and in the registry hash), not in config.
SYSTEM_MESSAGE = "You are a component in an automated test. Follow the user's instructions exactly."
BANANA_PROMPT = "Reply with exactly one word, lowercase, no punctuation: banana"
TEMPERATURE = 1.0  # NVIDIA's recommended sampling for Nemotron
TOP_P = 0.95
MAX_TOKENS_THINKING_OFF = 1024
MAX_TOKENS_THINKING_ON = 4096  # also when thinking is omitted (reasoning is on by default)

# How `Condition.thinking` reaches the server. "chat_template_kwargs" is the top-level field
# that NVIDIA/vLLM document; the alternatives exist because Token Factory documents neither.
THINKING_SWITCHES = ("chat_template_kwargs", "extra_body", "reasoning_effort")
THINKING_SWITCH = "chat_template_kwargs"


def default_max_tokens(thinking: bool | None) -> int:
    return MAX_TOKENS_THINKING_OFF if thinking is False else MAX_TOKENS_THINKING_ON


def apply_thinking(body: dict[str, Any], thinking: bool | None, switch: str = THINKING_SWITCH) -> dict[str, Any]:
    if thinking is None:
        return body
    if switch == "chat_template_kwargs":
        body["chat_template_kwargs"] = {"enable_thinking": thinking}
    elif switch == "extra_body":
        body["extra_body"] = {"chat_template_kwargs": {"enable_thinking": thinking}}
    elif switch == "reasoning_effort":
        if not thinking:
            body["reasoning_effort"] = "none"
    else:
        raise ValueError(f"unknown thinking switch {switch!r}")
    return body


def chat_request(
    model: str,
    user: str,
    cond: Condition,
    *,
    max_tokens: int | None = None,
    switch: str = THINKING_SWITCH,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": user}],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": cond.max_tokens or max_tokens or default_max_tokens(cond.thinking),
    }
    apply_thinking(body, cond.thinking, switch)
    if cond.stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    return body
