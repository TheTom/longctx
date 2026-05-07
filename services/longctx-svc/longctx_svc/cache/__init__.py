"""Persistent disk cache for built scope indexes. PRD §4."""
from longctx_svc.cache.disk import (
    cache_dir_for,
    clean_older_than,
    list_cached,
    load_index,
    save_index,
)

__all__ = [
    "cache_dir_for",
    "clean_older_than",
    "list_cached",
    "load_index",
    "save_index",
]
