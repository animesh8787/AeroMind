"""Lightweight retrieval over the project-owned notes in ``docs/knowledge``.

Keyword scoring only: no embeddings, no external service. The notes describe this prototype; they
are not OEM manuals or regulatory guidance, and the prompt says so.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

_STOP = set("the a an of to is in and or for on with why what how does do this that it be as at by from are was".split())


@dataclass(frozen=True)
class Chunk:
    source: str
    heading: str
    text: str


def default_dir() -> Path | None:
    env = os.environ.get("AEROMIND_KNOWLEDGE_DIR")
    for c in ([Path(env)] if env else []) + [Path.cwd() / "docs" / "knowledge",
                                               Path(__file__).resolve().parents[3] / "docs" / "knowledge"]:
        if c.is_dir():
            return c
    return None


def _tokens(s: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9_]+", s.lower()) if t not in _STOP and len(t) > 1]


class KnowledgeBase:
    def __init__(self, directory: str | Path | None = None):
        d = Path(directory) if directory else default_dir()
        self.chunks: list[Chunk] = []
        if d and d.is_dir():
            for f in sorted(d.glob("*.md")):
                if f.name.lower() == "readme.md":
                    continue
                self.chunks.extend(self._split(f))

    @staticmethod
    def _split(f: Path) -> list[Chunk]:
        out, head, buf = [], f.stem, []
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.startswith("#") and buf:
                out.append(Chunk(f.name, head, "\n".join(buf).strip()))
                buf = []
            if line.startswith("#"):
                head = line.lstrip("# ").strip()
            buf.append(line)
        if buf:
            out.append(Chunk(f.name, head, "\n".join(buf).strip()))
        return [c for c in out if len(c.text) > 40]

    def retrieve(self, query: str, k: int = 3, max_chars: int = 700) -> list[Chunk]:
        q = set(_tokens(query))
        if not q or not self.chunks:
            return []
        scored = []
        for c in self.chunks:
            toks = _tokens(c.heading + " " + c.text)
            score = sum(1 for t in toks if t in q) / (len(toks) ** 0.5 or 1)
            if score > 0:
                scored.append((score, c))
        scored.sort(key=lambda x: -x[0])
        return [Chunk(c.source, c.heading, c.text[:max_chars]) for _, c in scored[:k]]
