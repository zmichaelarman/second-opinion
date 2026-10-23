from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


RATE_LIMIT_PREFIX = "x-ratelimit-"
ALLOWED_HEADER_NAMES = {
    "retry-after",
    "x-request-id",
    "request-id",
    "x-nebius-request-id",
    "cf-ray",
    "content-type",
}
MAX_RAW_BYTES = 200 * 1024


def filter_headers(headers: dict[str, str] | Any) -> dict[str, str]:
    """Keep only rate-limit headers, retry-after, and request IDs. Never store auth headers."""
    out: dict[str, str] = {}
    if not headers:
        return out
    items = headers.items() if hasattr(headers, "items") else []
    for k, v in items:
        kl = str(k).lower()
        if kl.startswith(RATE_LIMIT_PREFIX) or kl in ALLOWED_HEADER_NAMES:
            out[kl] = str(v)
    return out


def truncate_raw(raw_obj: Any, max_bytes: int = MAX_RAW_BYTES) -> Any:
    try:
        serialized = json.dumps(raw_obj, ensure_ascii=False)
    except Exception:
        serialized = str(raw_obj)
    encoded = serialized.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return raw_obj
    clipped = encoded[:max_bytes].decode("utf-8", errors="ignore")
    return {"_truncated": True, "original_bytes": len(encoded), "preview": clipped}


@dataclass
class ParsedResponse:
    status: int
    headers: dict[str, str] = field(default_factory=dict)
    content: str = ""
    reasoning: str = ""
    reasoning_field: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    refusal: str | None = None
    n_chunks: int = 0
    ttft_s: float | None = None
    total_s: float = 0.0
    raw: Any = None
    error_type: str | None = None
    content_deltas: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["raw"] = truncate_raw(self.raw)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParsedResponse:
        return cls(
            status=int(data.get("status", 0)),
            headers=dict(data.get("headers") or {}),
            content=str(data.get("content") or ""),
            reasoning=str(data.get("reasoning") or ""),
            reasoning_field=data.get("reasoning_field"),
            tool_calls=list(data.get("tool_calls") or []),
            finish_reason=data.get("finish_reason"),
            usage=data.get("usage"),
            refusal=data.get("refusal"),
            n_chunks=int(data.get("n_chunks", 0)),
            ttft_s=data.get("ttft_s"),
            total_s=float(data.get("total_s", 0.0)),
            raw=data.get("raw"),
            error_type=data.get("error_type"),
            content_deltas=list(data.get("content_deltas") or []),
        )


def parse_json_response(
    status: int,
    headers: dict[str, str] | Any,
    body: Any,
    total_s: float = 0.0,
    ttft_s: float | None = None,
    error_type: str | None = None,
) -> ParsedResponse:
    clean_headers = filter_headers(headers)
    if not isinstance(body, dict):
        return ParsedResponse(
            status=status,
            headers=clean_headers,
            total_s=total_s,
            ttft_s=ttft_s,
            raw=truncate_raw(body),
            error_type=error_type or ("HTTP_5XX" if status >= 500 else None),
        )

    choices = body.get("choices") or []
    c0 = choices[0] if choices and isinstance(choices[0], dict) else {}
    msg = c0.get("message") if isinstance(c0.get("message"), dict) else {}

    content = str(msg.get("content") or "")

    reasoning = ""
    reasoning_field: str | None = None
    if "reasoning_content" in msg and msg["reasoning_content"] is not None:
        reasoning = str(msg["reasoning_content"])
        reasoning_field = "reasoning_content"
    elif "reasoning" in msg and msg["reasoning"] is not None:
        reasoning = str(msg["reasoning"])
        reasoning_field = "reasoning"

    tool_calls: list[dict[str, Any]] = []
    raw_tcs = msg.get("tool_calls") or []
    if isinstance(raw_tcs, list):
        for idx, tc in enumerate(raw_tcs):
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            tool_calls.append(
                {
                    "index": tc.get("index", idx),
                    "id": str(tc.get("id") or ""),
                    "name": str(fn.get("name") or ""),
                    "arguments_raw": str(fn.get("arguments") if fn.get("arguments") is not None else ""),
                }
            )

    finish_reason = c0.get("finish_reason")
    usage = body.get("usage") if isinstance(body.get("usage"), dict) else None
    refusal = str(msg["refusal"]) if msg.get("refusal") is not None else None

    err = error_type
    if err is None and status >= 500:
        err = "HTTP_5XX"

    return ParsedResponse(
        status=status,
        headers=clean_headers,
        content=content,
        reasoning=reasoning,
        reasoning_field=reasoning_field,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        usage=usage,
        refusal=refusal,
        n_chunks=0,
        ttft_s=ttft_s,
        total_s=total_s,
        raw=truncate_raw(body),
        error_type=err,
    )


def parse_sse_chunks(
    status: int,
    headers: dict[str, str] | Any,
    data_payloads: list[str | dict[str, Any]],
    total_s: float = 0.0,
    ttft_s: float | None = None,
    error_type: str | None = None,
) -> ParsedResponse:
    clean_headers = filter_headers(headers)
    if status >= 400:
        err = error_type or ("HTTP_5XX" if status >= 500 else None)
        return ParsedResponse(
            status=status,
            headers=clean_headers,
            total_s=total_s,
            ttft_s=ttft_s,
            raw=truncate_raw(data_payloads),
            error_type=err,
        )

    content_parts: list[str] = []
    content_deltas: list[str] = []
    reasoning_parts: list[str] = []
    reasoning_field: str | None = None
    refusal_parts: list[str] = []
    merged_tcs: dict[int, dict[str, Any]] = {}
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    n_chunks = 0
    raw_list: list[Any] = []

    for item in data_payloads:
        if isinstance(item, str):
            s = item.strip()
            if not s or s == "[DONE]":
                continue
            try:
                chunk = json.loads(s)
            except json.JSONDecodeError:
                raw_list.append(s)
                continue
        elif isinstance(item, dict):
            chunk = item
        else:
            continue

        raw_list.append(chunk)
        n_chunks += 1

        if isinstance(chunk.get("usage"), dict) and chunk["usage"]:
            usage = chunk["usage"]

        choices = chunk.get("choices") or []
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            if choice.get("finish_reason") is not None:
                finish_reason = str(choice["finish_reason"])
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue

            if delta.get("content") is not None and delta.get("content") != "":
                c_piece = str(delta["content"])
                content_parts.append(c_piece)
                content_deltas.append(c_piece)

            if "reasoning_content" in delta and delta["reasoning_content"] is not None:
                r_piece = str(delta["reasoning_content"])
                if r_piece:
                    reasoning_parts.append(r_piece)
                if reasoning_field is None:
                    reasoning_field = "reasoning_content"
            elif "reasoning" in delta and delta["reasoning"] is not None:
                r_piece = str(delta["reasoning"])
                if r_piece:
                    reasoning_parts.append(r_piece)
                if reasoning_field is None:
                    reasoning_field = "reasoning"

            if delta.get("refusal") is not None and delta.get("refusal") != "":
                refusal_parts.append(str(delta["refusal"]))

            delta_tcs = delta.get("tool_calls") or []
            if isinstance(delta_tcs, list):
                for tc in delta_tcs:
                    if not isinstance(tc, dict):
                        continue
                    idx = int(tc.get("index", 0))
                    if idx not in merged_tcs:
                        merged_tcs[idx] = {
                            "index": idx,
                            "id": "",
                            "name": "",
                            "arguments_raw": "",
                        }
                    entry = merged_tcs[idx]
                    if not entry["id"] and tc.get("id"):
                        entry["id"] = str(tc["id"])
                    fn = tc.get("function")
                    if isinstance(fn, dict):
                        if not entry["name"] and fn.get("name"):
                            entry["name"] = str(fn["name"])
                        if fn.get("arguments") is not None:
                            entry["arguments_raw"] += str(fn["arguments"])

    sorted_tcs = [merged_tcs[k] for k in sorted(merged_tcs.keys())]
    refusal = "".join(refusal_parts) if refusal_parts else None

    return ParsedResponse(
        status=status,
        headers=clean_headers,
        content="".join(content_parts),
        reasoning="".join(reasoning_parts),
        reasoning_field=reasoning_field,
        tool_calls=sorted_tcs,
        finish_reason=finish_reason,
        usage=usage,
        refusal=refusal,
        n_chunks=n_chunks,
        ttft_s=ttft_s,
        total_s=total_s,
        raw=truncate_raw(raw_list),
        error_type=error_type,
        content_deltas=content_deltas,
    )

