#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Private finite-index data-plane mechanics shared across PhydraX domains."""

from ._epoch import IndexEpochPlan
from ._ordering import EPOCH_ORDER_ALGORITHM, StatelessIndexPermutation
from ._prefetch import BoundedPrefetchIterator


__all__ = [
    "BoundedPrefetchIterator",
    "EPOCH_ORDER_ALGORITHM",
    "IndexEpochPlan",
    "StatelessIndexPermutation",
]
