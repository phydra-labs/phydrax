#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Query allowance accounting for represented-source support proofs."""

from __future__ import annotations


def spend_support_work(work: int, amount: int, message: str, /) -> int:
    """Return the remaining query allowance or refuse the proposed charge.

    An over-limit proposal has no returned balance; callers retain their last
    admitted quota. Numerical work performed before a refusal remains charged
    to the separate CoordinateEnclosureBudget, rather than this query counter.
    """
    remaining = work - amount
    if remaining < 0:
        raise ValueError(message)
    return remaining
