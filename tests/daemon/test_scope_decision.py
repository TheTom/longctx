"""Scope-decision tests — §3.5 routing rules.

Pure logic tests; no storage, no embedder. Each test exercises one
priority tier (or one tiebreaker case from the spec table).
"""
from __future__ import annotations

from longctx_daemon.searcher import (
    COMMON_WORD_NAMES,
    decide_scope,
)


# --------------------------------------------------------------- tier 1

def test_explicit_project_wins():
    """``project=`` argument overrides everything else (tier 1)."""
    decision = decide_scope(
        query="anything goes here",
        cwd="/Users/tom/dev/longctx",
        project="mlx-swift-lm",
        active_project_sticky="other",
        projects_in_index=("longctx", "mlx-swift-lm", "other"),
    )
    assert decision.primary_project == "mlx-swift-lm"
    assert decision.primary_source == "explicit_project"
    assert decision.fanout_projects == ("mlx-swift-lm",)
    assert decision.cross_project_pattern_matched is None


def test_explicit_project_with_unknown_name():
    """Tier 1 trusts the caller — unknown project names are accepted.

    The MCP layer is responsible for validating against ``list_projects``
    before invoking the searcher; mis-spellings surface to the agent
    via empty results, not via the scope decision itself.
    """
    decision = decide_scope(
        query="anything",
        cwd=None,
        project="not-actually-indexed",
        active_project_sticky=None,
        projects_in_index=("longctx",),
    )
    assert decision.primary_source == "explicit_project"
    assert decision.fanout_projects == ("not-actually-indexed",)


# --------------------------------------------------------------- tier 2

def test_distinctive_name_bare_mention():
    """Distinctive names match on bare mention — spec table row 2."""
    decision = decide_scope(
        query="how does mlx-swift-lm handle centroid",
        cwd="/Users/tom/dev/longctx/foo",
        project=None,
        active_project_sticky=None,
        projects_in_index=("longctx", "mlx-swift-lm"),
    )
    assert decision.primary_project == "mlx-swift-lm"
    assert decision.primary_source == "cross_project_pattern"
    assert decision.cross_project_pattern_matched == "mlx-swift-lm"


def test_common_word_no_cue_does_not_override():
    """`auth` (common-word) without cue doesn't fire — spec row 3."""
    decision = decide_scope(
        query="where do I handle auth",
        cwd=None,
        project=None,
        active_project_sticky="myapp",
        projects_in_index=("myapp", "auth"),
    )
    # Falls through to sticky (tier 3), NOT cross-project.
    assert decision.primary_project == "myapp"
    assert decision.primary_source == "active_project_sticky"


def test_common_word_with_cue_overrides():
    """`in the auth project` overrides sticky — spec row 4."""
    decision = decide_scope(
        query="in the auth project, where do we rate-limit",
        cwd=None,
        project=None,
        active_project_sticky="myapp",
        projects_in_index=("myapp", "auth"),
    )
    assert decision.primary_project == "auth"
    assert decision.primary_source == "cross_project_pattern"
    assert decision.cross_project_pattern_matched == "auth"


def test_common_word_with_in_cue():
    """`in auth` is the canonical syntactic cue."""
    decision = decide_scope(
        query="how do we sign tokens in auth",
        cwd=None,
        project=None,
        active_project_sticky=None,
        projects_in_index=("auth", "other"),
    )
    assert decision.primary_project == "auth"
    assert decision.primary_source == "cross_project_pattern"


def test_cross_project_comparison_fans_out():
    """`compare auth in longctx vs mlx-swift-lm` matches both — row 5.

    The spec routes this to both ``longctx`` (cue: ``in <name>``) and
    ``mlx-swift-lm`` (distinctive bare mention). ``auth`` here is a
    content word, not a project — there's no cue adjacent to it.
    """
    decision = decide_scope(
        query="compare auth in longctx vs mlx-swift-lm",
        cwd="/Users/tom",
        project=None,
        active_project_sticky=None,
        projects_in_index=("longctx", "mlx-swift-lm", "auth"),
    )
    assert decision.primary_source == "cross_project_pattern"
    fanout_set = set(decision.fanout_projects)
    assert "longctx" in fanout_set
    assert "mlx-swift-lm" in fanout_set
    # `auth` is a common word and there's no cue adjacent to it
    # ("compare auth ..." is not a cue), so it doesn't get matched.
    assert "auth" not in fanout_set


def test_cross_project_comparison_with_dual_cue():
    """Explicit dual-cue query routes to both projects.

    `"in longctx and in mlx-swift-lm"` (or `"compare X in longctx vs
    in mlx-swift-lm"`) is the cleanest way to fanout deliberately.
    """
    decision = decide_scope(
        query="compare lookups in auth and in mlx-swift-lm",
        cwd=None,
        project=None,
        active_project_sticky=None,
        projects_in_index=("auth", "mlx-swift-lm"),
    )
    assert decision.primary_source == "cross_project_pattern"
    fanout_set = set(decision.fanout_projects)
    assert "auth" in fanout_set
    assert "mlx-swift-lm" in fanout_set


def test_distinctive_name_inside_word_does_not_match():
    """`longctx-other` shouldn't match project `longctx`.

    Word-boundary regex saves us from false positives where the
    project name is a substring of a longer identifier.
    """
    decision = decide_scope(
        query="we ran longctx-other today",
        cwd=None,
        project=None,
        active_project_sticky=None,
        projects_in_index=("longctx",),
    )
    # Without sticky / cwd we land in tier 5.
    assert decision.primary_source == "fanout_no_primary"


def test_common_word_collision_examples_from_spec():
    """Every common-word name behaves the same way: needs a cue."""
    for name in ("core", "lib", "api", "client", "server", "tools",
                 "utils", "platform"):
        # No cue: tier 3 sticky wins.
        no_cue = decide_scope(
            query=f"where is {name} located",
            cwd=None,
            project=None,
            active_project_sticky="myapp",
            projects_in_index=("myapp", name),
        )
        assert no_cue.primary_source == "active_project_sticky", name
        # With cue: tier 2 fires.
        with_cue = decide_scope(
            query=f"in {name} how do we lazy-load",
            cwd=None,
            project=None,
            active_project_sticky="myapp",
            projects_in_index=("myapp", name),
        )
        assert with_cue.primary_source == "cross_project_pattern", name
        assert with_cue.primary_project == name


# --------------------------------------------------------------- tier 3

def test_sticky_session():
    """No project / no cross-project hit / sticky set → sticky wins."""
    decision = decide_scope(
        query="what does the rerank step do",
        cwd=None,
        project=None,
        active_project_sticky="longctx",
        projects_in_index=("longctx", "other"),
    )
    assert decision.primary_project == "longctx"
    assert decision.primary_source == "active_project_sticky"
    assert decision.active_project_sticky == "longctx"


# --------------------------------------------------------------- tier 4

def test_cwd_walk_substring_match():
    """cwd containing project name as a path segment → that project."""
    decision = decide_scope(
        query="search anything",
        cwd="/Users/tom/dev/longctx/src/foo.py",
        project=None,
        active_project_sticky=None,
        projects_in_index=("longctx", "mlx-swift-lm"),
    )
    assert decision.primary_project == "longctx"
    assert decision.primary_source == "cwd_walk_to_sentinel"


def test_cwd_walk_with_explicit_roots():
    """Real walk: project_roots map → deepest matching root wins."""
    decision = decide_scope(
        query="find me something",
        cwd="/Users/tom/dev/myorg/svc-a/internal/foo.py",
        project=None,
        active_project_sticky=None,
        projects_in_index=("myorg", "svc-a"),
        project_roots={
            "myorg": "/Users/tom/dev/myorg",
            "svc-a": "/Users/tom/dev/myorg/svc-a",
        },
    )
    # Deepest root wins — svc-a, not its parent monorepo.
    assert decision.primary_project == "svc-a"
    assert decision.primary_source == "cwd_walk_to_sentinel"


def test_cwd_outside_any_project_falls_through():
    """cwd outside every indexed project → tier 5 fanout."""
    decision = decide_scope(
        query="search whatever",
        cwd="/Users/tom/Downloads/random",
        project=None,
        active_project_sticky=None,
        projects_in_index=("longctx", "mlx-swift-lm", "vllm"),
    )
    assert decision.primary_source == "fanout_no_primary"
    assert decision.primary_project is None
    assert set(decision.fanout_projects) == {"longctx", "mlx-swift-lm", "vllm"}


# --------------------------------------------------------------- tier 5

def test_no_signal_fans_out_globally():
    """No project / no sticky / no cwd → fan out across all indexed."""
    decision = decide_scope(
        query="just a generic question",
        cwd=None,
        project=None,
        active_project_sticky=None,
        projects_in_index=("longctx", "mlx-swift-lm"),
    )
    assert decision.primary_source == "fanout_no_primary"
    assert decision.primary_project is None
    assert set(decision.fanout_projects) == {"longctx", "mlx-swift-lm"}


def test_empty_corpus_handled():
    """Tier 5 with zero indexed projects — empty fanout, no primary."""
    decision = decide_scope(
        query="x",
        cwd=None,
        project=None,
        active_project_sticky=None,
        projects_in_index=(),
    )
    assert decision.primary_source == "fanout_no_primary"
    assert decision.primary_project is None
    assert decision.fanout_projects == ()


# --------------------------------------------------------------- internals

def test_common_word_set_matches_spec():
    """Frozen set of common-word names matches the spec exactly."""
    expected = {"auth", "core", "lib", "api", "client", "server",
                "tools", "utils", "platform"}
    assert set(COMMON_WORD_NAMES) == expected


def test_priority_ordering_explicit_beats_cross_project():
    """``project=`` argument beats cross-project pattern."""
    decision = decide_scope(
        query="how does mlx-swift-lm centroid work",
        cwd=None,
        project="longctx",
        active_project_sticky=None,
        projects_in_index=("longctx", "mlx-swift-lm"),
    )
    assert decision.primary_source == "explicit_project"
    assert decision.primary_project == "longctx"


def test_priority_ordering_cross_project_beats_sticky():
    """Cross-project hit beats sticky session (tier 2 > tier 3)."""
    decision = decide_scope(
        query="check mlx-swift-lm for that bug",
        cwd=None,
        project=None,
        active_project_sticky="longctx",
        projects_in_index=("longctx", "mlx-swift-lm"),
    )
    assert decision.primary_source == "cross_project_pattern"
    assert decision.primary_project == "mlx-swift-lm"


def test_priority_ordering_sticky_beats_cwd():
    """Sticky session beats cwd-walk (tier 3 > tier 4)."""
    decision = decide_scope(
        query="something neutral",
        cwd="/Users/tom/dev/longctx/foo.py",
        project=None,
        active_project_sticky="other-project",
        projects_in_index=("longctx", "other-project"),
    )
    assert decision.primary_source == "active_project_sticky"
    assert decision.primary_project == "other-project"


def test_active_project_sticky_carried_through_on_other_tiers():
    """Sticky context is included even when a higher tier wins."""
    decision = decide_scope(
        query="something with mlx-swift-lm",
        cwd=None,
        project=None,
        active_project_sticky="longctx",
        projects_in_index=("longctx", "mlx-swift-lm"),
    )
    # Cross-project tier won but sticky is still surfaced.
    assert decision.primary_source == "cross_project_pattern"
    assert decision.active_project_sticky == "longctx"
