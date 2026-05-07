"""Session identification + session→scope mapping.

PRD §5.5: Sessions are identified in priority order:
  1. x-session-affinity header (OpenCode default)
  2. x-session-id header
  3. metadata.session_id in OpenAI body
  4. Stateless fallback (no session header → ephemeral)

Indexes are keyed by canonical scope path, NOT by session. Sessions point
to scopes; multiple sessions can share one scope.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from longctx_svc.config import get_config


@dataclass
class SessionEntry:
    session_id: str
    scope_hash: str | None
    last_seen_at: float
    detected_via: str         # "header:x-session-affinity" | "ephemeral" | ...
    # PRD §6.2 / v0.3.2 — track top-K cosine across turns for
    # confidence-driven promotion. Sliding window of the most recent
    # values; promotion fires when the trailing N are all below
    # threshold.
    confidence_window: list[float] = field(default_factory=list)
    confidence_window_size: int = 4
    consecutive_low: int = 0
    # PRD §6.3 / v0.3.3 — sessions accumulate scope memberships so a
    # single conversation can span multiple projects without losing
    # earlier ones. `scope_hash` remains the *primary* (most recent).
    scope_hashes: set[str] = field(default_factory=set)

    def record_confidence(self, score: float) -> None:
        self.confidence_window.append(float(score))
        if len(self.confidence_window) > self.confidence_window_size:
            self.confidence_window.pop(0)


def extract_session_id(headers: dict[str, str] | None,
                       body_metadata: dict | None = None
                       ) -> tuple[str | None, str]:
    """Return (session_id, source) per PRD §5.5 priority.

    `headers` keys are lower-cased on entry; we tolerate both shapes.
    Returns (None, "ephemeral") if nothing identifies the session.
    """
    if headers:
        h = {k.lower(): v for k, v in headers.items()}
        if h.get("x-session-affinity"):
            return h["x-session-affinity"], "header:x-session-affinity"
        if h.get("x-session-id"):
            return h["x-session-id"], "header:x-session-id"
    if body_metadata:
        sid = body_metadata.get("session_id")
        if sid:
            return str(sid), "body:metadata.session_id"
    return None, "ephemeral"


class SessionManager:
    """Thread-safe session tracking. Scoped membership only."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sessions: dict[str, SessionEntry] = {}

    def get(self, session_id: str) -> SessionEntry | None:
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry is not None:
                entry.last_seen_at = time.time()
            return entry

    def bind(self, session_id: str, scope_hash: str,
             detected_via: str = "header:x-session-affinity"
             ) -> SessionEntry:
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry is None:
                entry = SessionEntry(
                    session_id=session_id,
                    scope_hash=scope_hash,
                    last_seen_at=time.time(),
                    detected_via=detected_via,
                )
                entry.scope_hashes.add(scope_hash)
                self._sessions[session_id] = entry
            else:
                entry.scope_hash = scope_hash
                entry.scope_hashes.add(scope_hash)
                entry.last_seen_at = time.time()
                entry.detected_via = detected_via
            return entry

    def evict_idle(self) -> int:
        """Drop sessions idle past the timeout. Returns count dropped."""
        cfg = get_config()
        cutoff = time.time() - cfg.limits.session_idle_timeout_seconds
        with self._lock:
            stale = [sid for sid, e in self._sessions.items()
                     if e.last_seen_at < cutoff]
            for sid in stale:
                self._sessions.pop(sid, None)
            return len(stale)

    def all_sessions(self) -> list[SessionEntry]:
        with self._lock:
            return list(self._sessions.values())

    def __len__(self) -> int:
        return len(self._sessions)
