from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

LANG_ALIASES = {
    "py": "python",
    "python": "python",
    "python3": "python",
    "sh": "bash",
    "shell": "bash",
    "bash": "bash",
    "zsh": "bash",
    "curl": "bash",
    "js": "js",
    "javascript": "js",
    "ts": "js",
    "typescript": "js",
    "mjs": "js",
}


@dataclass
class CodeBlock:
    page_url: str
    page_title: str
    block_idx: int
    lang: str
    raw_lang: str
    line: int
    code: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_lang(raw_lang: str) -> str:
    token = raw_lang.strip().split()[0].lower() if raw_lang.strip() else ""
    return LANG_ALIASES.get(token, token)


def extract_code_blocks(markdown_text: str, page_url: str, page_title: str = "") -> list[CodeBlock]:
    """Extract fenced code blocks from markdown text, recording page, block index, normalized lang, and start line."""
    lines = markdown_text.splitlines()
    blocks: list[CodeBlock] = []
    in_fence = False
    fence_marker = ""
    raw_lang = ""
    start_line = 0
    buf: list[str] = []
    block_idx = 0

    for idx, line in enumerate(lines, start=1):
        stripped = line.lstrip()
        if not in_fence:
            if stripped.startswith("```") or stripped.startswith("~~~"):
                marker_char = stripped[0]
                marker_len = len(stripped) - len(stripped.lstrip(marker_char))
                fence_marker = marker_char * marker_len
                raw_lang = stripped[marker_len:].strip()
                in_fence = True
                start_line = idx
                buf = []
        else:
            if stripped.startswith(fence_marker) and stripped.rstrip() == fence_marker:
                code = "\n".join(buf)
                lang = normalize_lang(raw_lang)
                # Auto-detect cURL in untagged or bash blocks
                if not lang and "curl " in code:
                    lang = "bash"
                blocks.append(
                    CodeBlock(
                        page_url=page_url,
                        page_title=page_title or page_url,
                        block_idx=block_idx,
                        lang=lang,
                        raw_lang=raw_lang,
                        line=start_line,
                        code=code,
                    )
                )
                block_idx += 1
                in_fence = False
                fence_marker = ""
                raw_lang = ""
                buf = []
            else:
                buf.append(line)

    return blocks

