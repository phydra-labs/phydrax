#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Private finite-index data-plane mechanics shared across PhydraX domains."""

from ._distributed import (
    DistributedIndexEpochPlan,
    epoch_shard,
    ExecutionGroupRank,
    make_global_array_from_process_local_data,
    ProcessLocalBatch,
)
from ._epoch import IndexEpochPlan
from ._ordering import EPOCH_ORDER_ALGORITHM, StatelessIndexPermutation
from ._prefetch import BoundedPrefetchIterator


__all__ = [
    "BoundedPrefetchIterator",
    "DistributedIndexEpochPlan",
    "EPOCH_ORDER_ALGORITHM",
    "ExecutionGroupRank",
    "epoch_shard",
    "IndexEpochPlan",
    "ProcessLocalBatch",
    "StatelessIndexPermutation",
    "make_global_array_from_process_local_data",
]
