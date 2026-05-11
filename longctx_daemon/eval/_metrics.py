"""Recall@K helpers for the daemon eval harness.

Two-step API:

    per_sample = recall_at_k(rank=needle_rank, k_values=(1, 3, 5, 8))
    # → {1: 0, 3: 1, 5: 1, 8: 1}  if needle was at rank 3

    aggregated = aggregate_recall(all_per_sample_dicts)
    # → {1: 0.40, 3: 0.80, 5: 0.95, 8: 1.00}

The split lets the caller decide how to bin samples (by context length,
by needle count, etc.) before aggregation. All-binary indicators per
sample, fractional aggregates across samples — standard IR retrieval
recall.

A ``rank`` of ``None`` means the needle was missed entirely (didn't
appear in the result list at any depth). A ``rank`` greater than every
``k`` value yields all-zero indicators. A ``rank`` of 0 is invalid (we
use 1-indexed ranks throughout the daemon); 0 is treated as a miss for
defensive parity with ``None``.

TODO(2.x): mean reciprocal rank (MRR) and nDCG would be the next two
metrics to add when the eval suite grows beyond simple top-K recall.
"""
from __future__ import annotations

from typing import Optional, Sequence


def recall_at_k(
    rank: Optional[int],
    k_values: Sequence[int],
) -> dict[int, int]:
    """Per-sample binary recall indicators.

    Args:
        rank: 1-indexed rank of the needle in the result list, or
            ``None`` if the needle was missed entirely. Values <= 0
            are treated as a miss.
        k_values: cutoffs to evaluate at. e.g. ``(1, 3, 5, 8)`` →
            R@1, R@3, R@5, R@8.

    Returns:
        Dict mapping each ``k`` to ``1`` (needle in top-k) or ``0``
        (not in top-k). Aggregated across samples by
        :func:`aggregate_recall`.

    Examples:
        >>> recall_at_k(rank=1, k_values=(1, 3, 5, 8))
        {1: 1, 3: 1, 5: 1, 8: 1}
        >>> recall_at_k(rank=None, k_values=(1, 3, 5))
        {1: 0, 3: 0, 5: 0}
        >>> recall_at_k(rank=4, k_values=(1, 3, 5))
        {1: 0, 3: 0, 5: 1}
    """
    out: dict[int, int] = {}
    if rank is None or rank <= 0:
        for k in k_values:
            out[int(k)] = 0
        return out
    for k in k_values:
        out[int(k)] = 1 if rank <= k else 0
    return out


def aggregate_recall(per_sample: list[dict[int, int]]) -> dict[int, float]:
    """Average per-sample binary indicators into fractional recall.

    Args:
        per_sample: list of ``recall_at_k`` outputs, one per sample.
            All dicts must share the same keys (k values).

    Returns:
        Dict mapping each ``k`` to the fractional recall (mean of the
        indicators). Empty input → empty dict (not zero — caller can
        tell "no samples" apart from "all misses").

    Examples:
        >>> aggregate_recall([
        ...     {1: 1, 3: 1},
        ...     {1: 0, 3: 1},
        ...     {1: 1, 3: 1},
        ... ])
        {1: 0.6666666666666666, 3: 1.0}
        >>> aggregate_recall([])
        {}
    """
    if not per_sample:
        return {}
    # Use the first sample's keys as the template; if later samples
    # diverge the caller has a bug worth surfacing.
    keys = list(per_sample[0].keys())
    n = len(per_sample)
    out: dict[int, float] = {}
    for k in keys:
        total = 0
        for d in per_sample:
            # ``.get`` with default 0 is intentionally permissive —
            # missing key means that sample didn't track that depth,
            # which is equivalent to a miss. Tests can opt into strict
            # parity by asserting key sets directly.
            total += int(d.get(k, 0))
        out[int(k)] = total / n
    return out
