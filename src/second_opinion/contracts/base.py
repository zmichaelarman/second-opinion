from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from second_opinion.parse import ParsedResponse


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
    thinking: bool | None  # None = omit chat_template_kwargs
    stream: bool = False
    max_tokens: int | None = None
    variant: str = "default"  # payload / schema variant
    shape: str = "wrapped"  # T5 request shape: wrapped | bare | json_object

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Condition:
        return cls(
            thinking=data.get("thinking"),
            stream=bool(data.get("stream", False)),
            max_tokens=data.get("max_tokens"),
            variant=str(data.get("variant", "default")),
            shape=str(data.get("shape", "wrapped")),
        )

    def short_label(self) -> str:
        parts = [f"var={self.variant}"]
        parts.append(f"think={self.thinking}")
        if self.stream:
            parts.append("stream=True")
        if self.max_tokens is not None:
            parts.append(f"max_tok={self.max_tokens}")
        if self.shape != "wrapped":
            parts.append(f"shape={self.shape}")
        return ", ".join(parts)


@dataclass
class OracleResult:
    outcome: Outcome
    codes: list[str] = field(default_factory=list)
    fatal_codes: list[str] = field(default_factory=list)
    soft_codes: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)
    severity: Severity | None = None


class Contract(Protocol):
    id: str
    tier: int
    title: str
    hypothesis: str
    severity: Severity
    core_conditions: set[Condition]
    conditions: list[Condition]

    def build_request(self, model: str, cond: Condition, trial: int) -> dict[str, Any]: ...
    def check(self, req: dict[str, Any], resp: ParsedResponse) -> OracleResult: ...


SYSTEM_PROMPT = "You are a component in an automated test. Follow the user's instructions exactly."


def compute_cell_id(contract_id: str, model: str, cond: Condition) -> str:
    payload = json.dumps(
        {
            "contract_id": contract_id,
            "model": model,
            "condition": cond.to_dict(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def build_base_chat_request(
    model: str,
    cond: Condition,
    user_content: str,
    system_content: str = SYSTEM_PROMPT,
) -> dict[str, Any]:
    if cond.max_tokens is not None:
        max_tokens = cond.max_tokens
    elif cond.thinking is False:
        max_tokens = 1024
    else:
        max_tokens = 4096

    req: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": max_tokens,
    }
    if cond.thinking is not None:
        req["chat_template_kwargs"] = {"enable_thinking": cond.thinking}
    if cond.stream:
        req["stream"] = True
        req["stream_options"] = {"include_usage": True}
    return req
