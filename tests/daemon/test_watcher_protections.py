"""Unit tests for v0.4.1 runaway-CPU protections in
``longctx_daemon.watcher``.

Covers the pure-logic pieces:
  * ``_LongctxWatchFilter`` rejects events under ignored-dir basenames.
  * ``_TokenBucket`` rate-limits acquire() correctly + handles 0=disabled.
  * Module-level ``set_paused`` / ``is_paused`` flag round-trips.

The self-throttle + watchfiles wiring are integration-tested separately
via the full Watcher lifecycle in test_watcher.py (the module imports
psutil opportunistically and degrades to a no-op without it).
"""
from __future__ import annotations

import asyncio
import os
import time

import pytest

from longctx_daemon.watcher import (
    _LongctxWatchFilter,
    _TokenBucket,
    is_paused,
    set_paused,
)


# ----------------------------------------------------- watch filter

@pytest.fixture
def filt():
    return _LongctxWatchFilter(frozenset({".git", "target", "node_modules"}))


def test_filter_accepts_normal_paths(filt):
    assert filt(None, f"/Users/tom/dev/myrepo/src/main.rs") is True
    assert filt(None, f"/home/tom/code/foo/bar/baz.py") is True


def test_filter_rejects_ignored_basenames(filt):
    sep = os.sep
    assert filt(None, sep.join(["", "Users", "tom", "dev", "myrepo", ".git", "objects", "ab", "cd"])) is False
    assert filt(None, sep.join(["", "Users", "tom", "dev", "myrepo", "target", "release", "x"])) is False
    assert filt(None, sep.join(["", "home", "tom", "myproj", "node_modules", "foo", "index.js"])) is False


def test_filter_rejects_nested_ignored(filt):
    sep = os.sep
    # Ignored dir deep in the tree still rejects.
    p = sep.join(["", "a", "b", "c", "target", "release", "x.o"])
    assert filt(None, p) is False


def test_filter_handles_paths_with_no_ignored_segment(filt):
    assert filt(None, "/Users/tom/file.txt") is True
    assert filt(None, "single-segment") is True


def test_filter_empty_ignore_set_accepts_everything():
    f = _LongctxWatchFilter(frozenset())
    assert f(None, "/anything/.git/foo") is True
    assert f(None, "/target/x") is True


# ----------------------------------------------------- token bucket

def test_bucket_disabled_when_rate_zero():
    """rate=0 means rate-limiting is disabled. ``acquire()`` returns
    instantly regardless of how many tokens are requested."""
    bucket = _TokenBucket(tokens_per_second=0.0)

    async def go():
        t0 = time.monotonic()
        await bucket.acquire(1000)
        return time.monotonic() - t0

    elapsed = asyncio.run(go())
    assert elapsed < 0.1, f"disabled bucket should be instant; took {elapsed}s"


def test_bucket_first_acquire_uses_burst():
    """Initial burst lets us draw up to ``burst`` tokens immediately
    without waiting for the refill rate."""
    bucket = _TokenBucket(tokens_per_second=1.0, burst=10.0)

    async def go():
        t0 = time.monotonic()
        await bucket.acquire(5.0)  # well under burst
        return time.monotonic() - t0

    elapsed = asyncio.run(go())
    assert elapsed < 0.1, f"first acquire under burst should be instant; took {elapsed}s"


def test_bucket_rate_limits_when_drained():
    """After the burst is gone, further acquires must wait the
    appropriate refill time."""
    bucket = _TokenBucket(tokens_per_second=10.0, burst=2.0)

    async def go():
        # Drain the burst.
        await bucket.acquire(2.0)
        # Next acquire needs 5 tokens at 10/sec = 0.5s minimum.
        t0 = time.monotonic()
        await bucket.acquire(5.0)
        return time.monotonic() - t0

    elapsed = asyncio.run(go())
    # Allow modest slop for scheduling. The point is we waited
    # roughly the refill interval.
    assert 0.3 < elapsed < 1.2, f"expected ~0.5s; got {elapsed}s"


def test_bucket_set_rate_updates_replenish():
    """Calling set_rate() reconfigures the bucket without resetting
    the token count."""
    bucket = _TokenBucket(tokens_per_second=1.0, burst=1.0)
    bucket.set_rate(100.0)

    async def go():
        await bucket.acquire(1.0)  # uses initial burst token
        t0 = time.monotonic()
        await bucket.acquire(10.0)  # 10/100 = 0.1s
        return time.monotonic() - t0

    elapsed = asyncio.run(go())
    assert elapsed < 0.3, f"set_rate should speed up acquire; took {elapsed}s"


# ----------------------------------------------------- pause flag

def test_pause_flag_default_false():
    # Reset to known state (other tests may have toggled).
    set_paused(False)
    assert is_paused() is False


def test_pause_flag_toggles():
    set_paused(False)
    assert is_paused() is False
    set_paused(True)
    assert is_paused() is True
    set_paused(False)
    assert is_paused() is False


def test_pause_flag_truthy_coercion():
    """The setter coerces to bool so any truthy/falsy value works
    (matches what the SIGUSR2 handler does when toggling)."""
    set_paused(1)
    assert is_paused() is True
    set_paused(0)
    assert is_paused() is False
    set_paused(False)  # leave clean for next test
