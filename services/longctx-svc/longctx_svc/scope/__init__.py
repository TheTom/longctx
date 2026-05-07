"""Scope detection + walk + gitignore."""
from longctx_svc.scope.detect import (
    DetectedScope,
    detect_scope,
    extract_paths_from_prefill,
    canonicalize_scope,
    find_sentinel_root,
)
from longctx_svc.scope.walk import (
    walk_hot_scope,
    walk_package_scope,
)

__all__ = [
    "DetectedScope",
    "detect_scope",
    "extract_paths_from_prefill",
    "canonicalize_scope",
    "find_sentinel_root",
    "walk_hot_scope",
    "walk_package_scope",
]
