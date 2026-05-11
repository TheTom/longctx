"""Symbol-aware retrieval augment.

Shared by ``longctx_svc.retrieve.pipeline`` (HTTP) and
``longctx_daemon.searcher`` (in-process). Lives in the ``longctx`` core
package so both retrieval surfaces share one implementation.

Bridges the BM25+dense bias toward docs over source on code-fix queries.
Extracts Python identifiers from the query (CamelCase, `class X` / `def X`,
backtick-quoted, snake_case, qualified `X.y`) and grep's the scope root
for their definition sites. Files containing definitions are added to the
candidate pool. A file-type prior then boosts `.py` and demotes docs when
the query has a code signal (traceback, error type, or symbol-def
mention).

Validated 2026-05-11 against a 10-instance SWE-bench retrieval_miss
audit: recovered 5 of 10 cases (django-10924 via ``FilePathField``,
matplotlib-22835 via ``format_cursor_data``, requests-1963 via
``Session`` + ``resolve_redirects``, requests-2148 via ``content`` +
``iter_content``, xarray-3364 via ``concat``).
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# Symbol extraction
# ---------------------------------------------------------------------------

_CAMELCASE_RE = re.compile(r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b")
_BACKTICK_RE  = re.compile(r"`([A-Za-z_][\w\.]*)`")
_CLASS_DEF_RE = re.compile(r"\bclass\s+([A-Z][\w]+)")
_FUNC_DEF_RE  = re.compile(r"\bdef\s+(\w+)")
_SNAKE_RE     = re.compile(r"\b([a-z]+_[a-z]+(?:_[a-z]+)+)\b")
_QUALIFIED_RE = re.compile(r"\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)\b")

# English stopwords that look like identifiers but aren't worth grepping.
# Kept narrow on purpose — better to over-grep cheap rg calls than miss a
# real definition. Length-4 floor below catches most short English noise.
_SYM_STOPWORDS: frozenset[str] = frozenset({
    "this", "that", "those", "these", "self", "None", "True", "False",
    "from", "import", "class", "def", "return", "test", "tests", "code",
    "file", "model", "models", "field", "value", "type", "types", "user",
    "users", "case", "cases", "data", "name", "names", "time", "add",
    "set", "get", "use", "using", "need", "make",
})


def extract_symbols(text: str) -> set[str]:
    """Return identifiers from `text` worth grepping for `class X` / `def X`.

    Filters: drops length<4 and stopwords, splits qualified `X.y` into
    parts so both `X` and `y` are grepped independently.
    """
    syms: set[str] = set()
    syms |= set(_CAMELCASE_RE.findall(text))
    syms |= set(_BACKTICK_RE.findall(text))
    syms |= set(_CLASS_DEF_RE.findall(text))
    syms |= set(_FUNC_DEF_RE.findall(text))
    syms |= set(_SNAKE_RE.findall(text))
    for cls, attr in _QUALIFIED_RE.findall(text):
        if len(cls) >= 4:
            syms.add(cls)
        if len(attr) >= 4:
            syms.add(attr)
    for s in list(syms):
        if "." in s:
            syms.discard(s)
            for p in s.split("."):
                if len(p) >= 4:
                    syms.add(p)
    return {s for s in syms if len(s) >= 4 and s.lower() not in _SYM_STOPWORDS}


# ---------------------------------------------------------------------------
# Repository grep
# ---------------------------------------------------------------------------


def symbol_grep_repo(symbols: set[str], repo_root: Path,
                     max_per_sym: int = 5) -> list[str]:
    """Return absolute paths defining any of the symbols.

    Greps for ``^\\s*class X`` and ``^\\s*def X`` (indented matches OK so
    methods count). Test files are excluded — the source under test is
    what we want, not the test file itself. rg is required.
    """
    if not symbols or not repo_root.exists():
        return []
    hits: list[str] = []
    seen: set[str] = set()
    for sym in symbols:
        for pat in (rf"^\s*class\s+{re.escape(sym)}\b",
                    rf"^\s*def\s+{re.escape(sym)}\b"):
            try:
                r = subprocess.run(
                    ["rg", "-l", "--type", "py",
                     "--glob", "!**/test_*.py", "--glob", "!**/tests/**",
                     pat, str(repo_root)],
                    capture_output=True, text=True, timeout=8,
                )
            except Exception:
                continue
            if r.returncode != 0:
                continue
            for line in r.stdout.strip().split("\n")[:max_per_sym]:
                if not line or line in seen:
                    continue
                seen.add(line)
                hits.append(line)
    return hits


# ---------------------------------------------------------------------------
# Query feature extraction + file-type prior
# ---------------------------------------------------------------------------

_TRACEBACK_RE = re.compile(r'File "[^"]+"')
_SYMBOL_DEF_RE = re.compile(r"\b(?:class|def)\s+\w+")
_ERROR_TYPE_RE = re.compile(r"\b\w+(?:Error|Exception|Warning)\b")


def query_features(text: str) -> dict:
    """Coarse signals indicating the query is about a code defect."""
    return {
        "n_tracebacks":  len(_TRACEBACK_RE.findall(text)),
        "n_symbol_defs": len(_SYMBOL_DEF_RE.findall(text)),
        "n_error_types": len(_ERROR_TYPE_RE.findall(text)),
    }


def has_code_signal(qf: dict) -> bool:
    return (qf.get("n_tracebacks", 0) > 0
            or qf.get("n_symbol_defs", 0) > 0
            or qf.get("n_error_types", 0) > 0)


_DOC_EXTS = (".rst", ".txt", ".md", ".cff", ".yaml", ".yml")


def file_type_weight(path: str, qf: dict) -> float:
    """Boost .py and demote prose/config for code-signal queries.

    Returns 1.0 when there is no code signal — feature-add queries don't
    get a bias. Returns a stable secondary-sort weight when there is.
    """
    if not has_code_signal(qf):
        return 1.0
    p = path.lower()
    if p.endswith(".py"):
        return 1.5
    if p.endswith(_DOC_EXTS):
        return 0.6
    if "/test" in p or "/tests/" in p:
        return 0.9
    return 1.0
