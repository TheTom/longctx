"""Session manager tests. PRD §5.5."""
from __future__ import annotations

import time

import pytest

from longctx_svc.session.manager import SessionManager, extract_session_id


class TestExtractSessionId:
    def test_x_session_affinity_priority_1(self):
        sid, src = extract_session_id({"x-session-affinity": "abc"}, None)
        assert sid == "abc"
        assert src == "header:x-session-affinity"

    def test_x_session_id_priority_2(self):
        sid, src = extract_session_id({"x-session-id": "def"}, None)
        assert sid == "def"
        assert src == "header:x-session-id"

    def test_x_affinity_wins_over_x_session_id(self):
        sid, src = extract_session_id(
            {"x-session-affinity": "a", "x-session-id": "b"}, None,
        )
        assert sid == "a"

    def test_metadata_priority_3(self):
        sid, src = extract_session_id(None, {"session_id": "ghi"})
        assert sid == "ghi"
        assert src == "body:metadata.session_id"

    def test_ephemeral_fallback(self):
        sid, src = extract_session_id(None, None)
        assert sid is None
        assert src == "ephemeral"

    def test_empty_headers_ephemeral(self):
        sid, src = extract_session_id({}, {})
        assert sid is None
        assert src == "ephemeral"

    def test_case_insensitive_headers(self):
        sid, src = extract_session_id({"X-Session-Affinity": "X"}, None)
        assert sid == "X"
        assert src == "header:x-session-affinity"


class TestSessionManager:
    def test_bind_creates_entry(self):
        m = SessionManager()
        entry = m.bind("s1", "scope-aaa")
        assert entry.session_id == "s1"
        assert entry.scope_hash == "scope-aaa"

    def test_get_returns_entry(self):
        m = SessionManager()
        m.bind("s1", "scope-aaa")
        e = m.get("s1")
        assert e is not None
        assert e.scope_hash == "scope-aaa"

    def test_rebind_updates_scope(self):
        m = SessionManager()
        m.bind("s1", "scope-aaa")
        m.bind("s1", "scope-bbb")
        e = m.get("s1")
        assert e.scope_hash == "scope-bbb"

    def test_two_sessions_can_share_scope(self):
        m = SessionManager()
        m.bind("s1", "shared")
        m.bind("s2", "shared")
        assert m.get("s1").scope_hash == m.get("s2").scope_hash

    def test_two_sessions_can_have_different_scopes(self):
        m = SessionManager()
        m.bind("s1", "scope-foo")
        m.bind("s2", "scope-bar")
        assert m.get("s1").scope_hash != m.get("s2").scope_hash

    def test_evict_idle(self):
        m = SessionManager()
        m.bind("s1", "scope-1")
        # Force last_seen_at into the distant past
        e = m.get("s1")
        e.last_seen_at = 0.0
        n = m.evict_idle()
        assert n == 1
        assert m.get("s1") is None

    def test_evict_keeps_recent(self):
        m = SessionManager()
        m.bind("s1", "scope-1")
        n = m.evict_idle()
        assert n == 0
        assert m.get("s1") is not None

    def test_len(self):
        m = SessionManager()
        m.bind("a", "x")
        m.bind("b", "y")
        assert len(m) == 2
