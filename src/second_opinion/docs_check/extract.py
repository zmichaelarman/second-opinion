"""Extract fenced code blocks from docs markdown (CLAUDE.md §10.2)."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

ALIASES = {
    "py": "python", "python": "python", "python3": "python",
    "sh": "bash", "shell": "bash", "bash": "bash", "zsh": "bash", "console": "bash", "curl": "bash",
    "js": "js", "javascript": "js", "ts": "js", "typescript": "js", "mjs": "js", "jsx": "js",
}
_FENCE = re.compile(r"^(?P<indent>[ \t]*)(?P<fence>`{3,}|~{3,})(?P<info>[^`]*)$")
_FLAGS = {"expandable", "wrap", "lines", "highlight"}


@dataclass
class CodeBlock:
    page_url: str
    block_idx: int
    lang: str  # normalized: python, bash, js, or the raw label
    raw_lang: str
    title: str
    line: int  # 1-based line of the opening fence
    code: str

    def as_dict(self) -> dict:
        return asdict(self)


def normalize_lang(raw: str) -> str:
    return ALIASES.get(raw.lower(), raw.lower())


def extract_blocks(page_url: str, text: str) -> list[CodeBlock]:
    lines = text.splitlines()
    blocks: list[CodeBlock] = []
    i = 0
    while i < len(lines):
        m = _FENCE.match(lines[i])
        if not m:
            i += 1
            continue
        indent, fence, info = m["indent"], m["fence"], m["info"].strip()
        tokens = info.split()
        raw_lang = tokens[0] if tokens else ""
        title = " ".join(t for t in tokens[1:] if "=" not in t and t.lower() not in _FLAGS)
        closing = re.compile(rf"^[ \t]*{re.escape(fence[0])}{{{len(fence)},}}\s*$")
        body: list[str] = []
        j = i + 1
        while j < len(lines) and not closing.match(lines[j]):
            line = lines[j]
            body.append(line[len(indent):] if line.startswith(indent) else line.lstrip())
            j += 1
        blocks.append(CodeBlock(page_url, len(blocks), normalize_lang(raw_lang), raw_lang, title, i + 1, "\n".join(body)))
        i = j + 1
    return blocks
