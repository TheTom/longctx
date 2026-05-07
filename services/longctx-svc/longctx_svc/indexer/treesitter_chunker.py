"""Tree-sitter code chunker. PRD §6.1 / v0.3.1.

Splits code at top-level definitions (functions, classes, methods)
when a language parser is available. Falls back silently to the
line-window chunker on unsupported languages or when tree_sitter_*
modules aren't installed.

Languages supported (when corresponding `tree-sitter-<lang>` module is
on PYTHONPATH): Python, TypeScript, JavaScript, Go, Rust.

Behind opt-in config: `LONGCTX_TS=1` (or `ServiceConfig.use_treesitter`).
Until parity with line-window is proven across testers, line-window
remains the default.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from longctx_svc.indexer.chunker import Chunk


# (lang_module_name, [node_types_to_chunk_at])
_LANG_REGISTRY: dict[str, tuple[str, tuple[str, ...]]] = {
    ".py":  ("tree_sitter_python",
             ("function_definition", "class_definition", "decorated_definition")),
    ".ts":  ("tree_sitter_typescript",
             ("function_declaration", "class_declaration",
              "method_definition", "interface_declaration",
              "export_statement")),
    ".tsx": ("tree_sitter_typescript",
             ("function_declaration", "class_declaration",
              "method_definition", "interface_declaration",
              "export_statement")),
    ".js":  ("tree_sitter_javascript",
             ("function_declaration", "class_declaration",
              "method_definition", "export_statement")),
    ".jsx": ("tree_sitter_javascript",
             ("function_declaration", "class_declaration",
              "method_definition", "export_statement")),
    ".mjs": ("tree_sitter_javascript",
             ("function_declaration", "class_declaration",
              "method_definition", "export_statement")),
    ".cjs": ("tree_sitter_javascript",
             ("function_declaration", "class_declaration",
              "method_definition", "export_statement")),
    ".go":  ("tree_sitter_go",
             ("function_declaration", "method_declaration",
              "type_declaration")),
    ".rs":  ("tree_sitter_rust",
             ("function_item", "impl_item", "struct_item",
              "trait_item", "enum_item")),
}


_PARSER_CACHE: dict[str, Callable] = {}


def _get_parser(module_name: str, ext: str):
    """Return a tree-sitter Parser for `module_name`, or None.

    Cached per-process — Parser/Language are cheap to reuse.
    """
    if module_name in _PARSER_CACHE:
        return _PARSER_CACHE[module_name] or None
    try:
        import importlib
        from tree_sitter import Language, Parser
        mod = importlib.import_module(module_name)
        # tree-sitter-typescript ships two grammars; pick the right one.
        if module_name == "tree_sitter_typescript":
            lang = Language(mod.language_typescript()
                            if ext != ".tsx"
                            else mod.language_tsx())
        else:
            lang = Language(mod.language())
        parser = Parser(lang)
        _PARSER_CACHE[module_name] = parser
        return parser
    except Exception:  # noqa: BLE001
        # Module missing, abi mismatch, etc — fall back gracefully.
        _PARSER_CACHE[module_name] = None  # type: ignore[assignment]
        return None


def has_parser_for(path: Path) -> bool:
    ext = path.suffix.lower()
    if ext not in _LANG_REGISTRY:
        return False
    mod_name, _ = _LANG_REGISTRY[ext]
    return _get_parser(mod_name, ext) is not None


def chunk_code_treesitter(
    text: str, file_path: str, ext: str,
    max_lines: int,
) -> list[Chunk] | None:
    """Return semantically split chunks, or None if unsupported.

    `None` signals the caller to fall back to line-window chunking.
    """
    if ext not in _LANG_REGISTRY:
        return None
    mod_name, node_types = _LANG_REGISTRY[ext]
    parser = _get_parser(mod_name, ext)
    if parser is None:
        return None
    try:
        tree = parser.parse(text.encode("utf-8", errors="replace"))
    except Exception:  # noqa: BLE001
        return None
    lines = text.splitlines(keepends=True)
    n_lines = len(lines)
    if n_lines == 0:
        return []

    chunks: list[Chunk] = []
    cursor = 0  # 0-indexed line cursor; tracks how far we've emitted

    # Top-level walk: only direct children of root_node.
    for node in tree.root_node.children:
        if node.type not in node_types:
            continue
        start_row = node.start_point[0]
        end_row = node.end_point[0]
        # Anything before the def becomes a "preamble" chunk (imports etc).
        if start_row > cursor:
            preamble = "".join(lines[cursor:start_row])
            if preamble.strip():
                chunks.extend(_window(
                    preamble, file_path, max_lines,
                    base_line=cursor + 1,
                ))
        body = "".join(lines[start_row:end_row + 1])
        if (end_row - start_row + 1) <= max_lines:
            chunks.append(Chunk(
                text=body, file_path=file_path,
                start_line=start_row + 1, end_line=end_row + 1,
                file_type="code",
            ))
        else:
            # Big function/class — fall through to line-window inside it.
            chunks.extend(_window(
                body, file_path, max_lines,
                base_line=start_row + 1,
            ))
        cursor = end_row + 1

    # Trailing tail
    if cursor < n_lines:
        tail = "".join(lines[cursor:])
        if tail.strip():
            chunks.extend(_window(
                tail, file_path, max_lines,
                base_line=cursor + 1,
            ))

    if not chunks:
        # Tree had no top-level defs (e.g., a script that's just statements)
        # — fall back to line-window.
        return None
    return chunks


def _window(text: str, file_path: str, max_lines: int,
            *, base_line: int) -> list[Chunk]:
    """Light line-window for fragments. Used inside huge defs and for
    preamble/tail."""
    lines = text.splitlines(keepends=True)
    n = len(lines)
    if n == 0:
        return []
    if n <= max_lines:
        return [Chunk(
            text=text, file_path=file_path,
            start_line=base_line, end_line=base_line + n - 1,
            file_type="code",
        )]
    out: list[Chunk] = []
    overlap = max(0, max_lines // 10)
    step = max(1, max_lines - overlap)
    start = 0
    while start < n:
        end = min(start + max_lines, n)
        out.append(Chunk(
            text="".join(lines[start:end]),
            file_path=file_path,
            start_line=base_line + start, end_line=base_line + end - 1,
            file_type="code",
        ))
        if end == n:
            break
        start += step
    return out
