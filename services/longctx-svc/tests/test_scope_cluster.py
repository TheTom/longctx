"""Unit tests for path-cluster scope detection (longctx_svc.scope.cluster).

These cover the pure function; the per-session state machine that uses
them lives in test_scope_tracker.py.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from longctx_svc.scope.cluster import (
    ClusterResult,
    all_clusters,
    cluster_paths,
)


class TestClusterPaths:
    def test_empty_returns_none(self):
        assert cluster_paths([], min_files=3) is None

    def test_single_path_below_threshold(self):
        assert cluster_paths([Path("/a/b/c.py")], min_files=3) is None

    def test_three_siblings_same_dir(self):
        paths = [
            Path("/work/project/a.py"),
            Path("/work/project/b.py"),
            Path("/work/project/c.py"),
        ]
        result = cluster_paths(paths, min_files=3)
        assert result is not None
        assert result.ancestor == Path("/work/project")
        assert result.unique_files == 3
        assert result.total_paths == 3

    def test_dedupes_repeated_mentions(self):
        # 5 mentions but only 2 unique files — still below threshold of 3.
        paths = [
            Path("/x/a.py"),
            Path("/x/a.py"),
            Path("/x/a.py"),
            Path("/x/b.py"),
            Path("/x/b.py"),
        ]
        assert cluster_paths(paths, min_files=3) is None

    def test_prefers_broader_when_more_files_captured(self):
        # 3 files in /work/project/sub plus 1 sibling in /work/project. The
        # parent dir captures all 4 — that's the more useful scope (anything
        # narrower fragments the index from the sibling). Deeper dir wins
        # only when the file-count tied.
        paths = [
            Path("/work/project/sub/a.py"),
            Path("/work/project/sub/b.py"),
            Path("/work/project/sub/c.py"),
            Path("/work/project/other.py"),
        ]
        result = cluster_paths(paths, min_files=3)
        assert result is not None
        # /work/project captures 4 distinct files; /work/project/sub only 3.
        # The broader (but-still-qualifying) scope wins.
        assert result.ancestor == Path("/work/project")
        assert result.unique_files == 4

    def test_walks_up_when_subdir_thin(self):
        # Each subdir has only 1 file; the parent has 3. Parent wins.
        paths = [
            Path("/work/proj/a/x.py"),
            Path("/work/proj/b/y.py"),
            Path("/work/proj/c/z.py"),
        ]
        result = cluster_paths(paths, min_files=3)
        assert result is not None
        assert result.ancestor == Path("/work/proj")
        assert result.unique_files == 3

    def test_no_cluster_when_paths_diverge(self):
        # Three paths under different first-level roots — the only common
        # ancestor is "/" which the default ``min_depth=2`` filter rejects.
        # That filter exists precisely to stop grab-bag mentions from
        # producing a useless top-level scope.
        paths = [
            Path("/etc/passwd"),
            Path("/tmp/foo"),
            Path("/var/log/x"),
        ]
        assert cluster_paths(paths, min_files=3) is None

    def test_min_depth_lets_root_through_when_disabled(self):
        # Caller can opt out of the depth filter for tests / debug.
        paths = [
            Path("/etc/passwd"),
            Path("/tmp/foo"),
            Path("/var/log/x"),
        ]
        result = cluster_paths(paths, min_files=3, min_depth=1)
        assert result is not None
        assert result.ancestor == Path("/")

    def test_threshold_not_met(self):
        paths = [Path("/a/b/x.py"), Path("/a/b/y.py")]
        assert cluster_paths(paths, min_files=3) is None

    def test_higher_threshold(self):
        paths = [Path(f"/work/{n}.py") for n in "abcdef"]
        # 6 unique files — meets min_files=5.
        result = cluster_paths(paths, min_files=5)
        assert result is not None
        assert result.ancestor == Path("/work")
        assert result.unique_files == 6

    def test_invalid_min_files_raises(self):
        with pytest.raises(ValueError):
            cluster_paths([Path("/a/b/c.py")], min_files=0)

    def test_stable_under_input_reordering(self):
        paths_a = [
            Path("/work/a.py"),
            Path("/work/b.py"),
            Path("/work/c.py"),
        ]
        paths_b = [
            Path("/work/c.py"),
            Path("/work/a.py"),
            Path("/work/b.py"),
        ]
        ra = cluster_paths(paths_a, min_files=3)
        rb = cluster_paths(paths_b, min_files=3)
        assert ra == rb

    def test_broader_scope_wins_over_two_sub_clusters(self):
        # /a/p1 has 3 files; /a/p2 has 3 files; /a captures all 6. Under
        # the count-first comparator the broader /a wins — it's the scope
        # that captures every observation, and the agent is clearly
        # working across both subdirs.
        paths = [
            Path("/a/p1/x.py"), Path("/a/p1/y.py"), Path("/a/p1/z.py"),
            Path("/a/p2/x.py"), Path("/a/p2/y.py"), Path("/a/p2/z.py"),
        ]
        result = cluster_paths(paths, min_files=3)
        assert result is not None
        assert result.ancestor == Path("/a")
        assert result.unique_files == 6

    def test_deepest_wins_on_strict_count_tie(self):
        # /a/p1 has 3 files; /a has the same 3 (parent of all). On a true
        # cardinality tie, the deeper dir wins — it's the more-specific
        # scope. This guards against the comparator regressing to "always
        # prefer the broadest" when broadness gives no extra coverage.
        paths = [
            Path("/a/p1/x.py"), Path("/a/p1/y.py"), Path("/a/p1/z.py"),
        ]
        result = cluster_paths(paths, min_files=3)
        assert result is not None
        assert result.ancestor == Path("/a/p1")
        assert result.unique_files == 3


class TestAllClusters:
    def test_returns_all_qualifying_ancestors(self):
        paths = [
            Path("/work/proj/sub/a.py"),
            Path("/work/proj/sub/b.py"),
            Path("/work/proj/sub/c.py"),
        ]
        results = all_clusters(paths, min_files=3)
        # Every ancestor (/work/proj/sub, /work/proj, /work, /) contains
        # all 3 paths, so all qualify.
        ancestors = [r.ancestor for r in results]
        assert Path("/work/proj/sub") in ancestors
        assert Path("/work/proj") in ancestors
        assert Path("/work") in ancestors

    def test_first_result_is_deepest(self):
        paths = [
            Path("/work/proj/sub/a.py"),
            Path("/work/proj/sub/b.py"),
            Path("/work/proj/sub/c.py"),
        ]
        results = all_clusters(paths, min_files=3)
        assert results
        assert results[0].ancestor == Path("/work/proj/sub")

    def test_empty_below_threshold(self):
        assert all_clusters([Path("/a/x.py")], min_files=3) == []
