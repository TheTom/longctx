"""File chunking. PRD §5.4 (line-window fallback for v0.3.0;
tree-sitter parseable splits land in v0.3.1).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from longctx_svc.config import (
    CODE_EXTS,
    CONFIG_EXTS,
    PROSE_EXTS,
    get_config,
)


@dataclass(frozen=True)
class Chunk:
    """Single retrievable chunk."""
    text: str
    file_path: str        # absolute path
    start_line: int       # 1-indexed inclusive
    end_line: int         # 1-indexed inclusive
    file_type: str        # 'code' | 'prose' | 'config' | 'other'

    def display(self) -> str:
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


def _classify(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in CODE_EXTS:
        return "code"
    if ext in PROSE_EXTS:
        return "prose"
    if ext in CONFIG_EXTS:
        return "config"
    return "other"


def _chunk_lines(lines: list[str], lines_per_chunk: int, overlap: int,
                 file_path: str, file_type: str) -> list[Chunk]:
    """Slide a window over `lines` producing Chunks. 1-indexed line numbers."""
    if not lines:
        return []
    n = len(lines)
    if n <= lines_per_chunk:
        return [Chunk(
            text="".join(lines),
            file_path=file_path,
            start_line=1,
            end_line=n,
            file_type=file_type,
        )]
    step = max(1, lines_per_chunk - overlap)
    chunks: list[Chunk] = []
    start = 0
    while start < n:
        end = min(start + lines_per_chunk, n)
        chunks.append(Chunk(
            text="".join(lines[start:end]),
            file_path=file_path,
            start_line=start + 1,
            end_line=end,
            file_type=file_type,
        ))
        if end == n:
            break
        start += step
    return chunks


def _chunk_paragraphs(text: str, file_path: str, max_lines: int,
                      overlap: int) -> list[Chunk]:
    """Prose-aware chunking: split on blank lines, then group paragraphs
    until line budget is hit. Falls back to line-window if a single
    paragraph exceeds the budget.
    """
    paragraphs: list[tuple[int, int, str]] = []  # (start_line, end_line, text)
    lines = text.splitlines(keepends=True)
    cur_start: int | None = None
    cur_text: list[str] = []
    for i, line in enumerate(lines, start=1):
        if line.strip():
            if cur_start is None:
                cur_start = i
            cur_text.append(line)
        else:
            if cur_text:
                paragraphs.append((cur_start, i - 1, "".join(cur_text)))
                cur_start = None
                cur_text = []
    if cur_text:
        paragraphs.append((cur_start or 1, len(lines), "".join(cur_text)))
    if not paragraphs:
        return []

    chunks: list[Chunk] = []
    buf_text = ""
    buf_start: int | None = None
    buf_end: int | None = None
    buf_lines = 0
    for s, e, t in paragraphs:
        plen = e - s + 1
        if plen > max_lines:
            # flush current
            if buf_text:
                chunks.append(Chunk(
                    text=buf_text, file_path=file_path,
                    start_line=buf_start or s, end_line=buf_end or e,
                    file_type="prose",
                ))
                buf_text = ""
                buf_start = buf_end = None
                buf_lines = 0
            # fallback to line-window for this paragraph
            big_lines = lines[s - 1:e]
            chunks.extend(_chunk_lines(
                big_lines, max_lines, overlap, file_path, "prose"
            ))
            continue
        if buf_lines + plen > max_lines and buf_text:
            chunks.append(Chunk(
                text=buf_text, file_path=file_path,
                start_line=buf_start or s, end_line=buf_end or e,
                file_type="prose",
            ))
            buf_text = ""
            buf_start = buf_end = None
            buf_lines = 0
        if buf_start is None:
            buf_start = s
        buf_end = e
        buf_text += ("\n" if buf_text else "") + t
        buf_lines += plen
    if buf_text:
        chunks.append(Chunk(
            text=buf_text, file_path=file_path,
            start_line=buf_start or 1, end_line=buf_end or len(lines),
            file_type="prose",
        ))
    return chunks


def chunk_text(text: str, file_path: str, file_type: str | None = None,
               lines_per_chunk: int | None = None,
               overlap: int | None = None) -> list[Chunk]:
    """Chunk arbitrary text. Useful for tests and ad-hoc use."""
    cfg = get_config()
    L = lines_per_chunk or cfg.limits.chunk_lines_default
    O = overlap if overlap is not None else cfg.limits.chunk_lines_overlap
    if file_type is None:
        file_type = _classify(Path(file_path))
    if file_type == "prose":
        return _chunk_paragraphs(text, file_path, L, O)
    if file_type == "config":
        # Whole-file as one chunk if under cap; else line-window.
        lines = text.splitlines(keepends=True)
        if len(lines) <= L:
            return [Chunk(
                text=text, file_path=file_path,
                start_line=1, end_line=max(1, len(lines)),
                file_type="config",
            )]
        return _chunk_lines(lines, L, O, file_path, "config")
    # code: try tree-sitter first (PRD §6.1 / v0.3.1) when enabled.
    if file_type == "code" and cfg.use_treesitter:
        try:
            from longctx_svc.indexer.treesitter_chunker import (
                chunk_code_treesitter,
            )
            ts_result = chunk_code_treesitter(
                text, file_path, Path(file_path).suffix.lower(), L,
            )
            if ts_result is not None:
                return ts_result
        except Exception:  # noqa: BLE001
            pass  # fall through to line-window
    # code or other: line-window
    lines = text.splitlines(keepends=True)
    return _chunk_lines(lines, L, O, file_path, file_type)


def chunk_file(path: Path,
               lines_per_chunk: int | None = None,
               overlap: int | None = None) -> list[Chunk]:
    """Read a file and chunk it. Returns [] if unreadable or empty."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    if not text:
        return []
    return chunk_text(
        text, str(path), file_type=_classify(path),
        lines_per_chunk=lines_per_chunk, overlap=overlap,
    )
