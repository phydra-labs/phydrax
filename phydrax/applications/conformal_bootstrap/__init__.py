#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite scalar-block and crossing-cone reference calculations."""

from ._blocks import (
    prepare_scalar_blocks,
    PreparedScalarBlocks,
    ScalarBlockEvidence,
    ScalarBlockPlan,
)
from ._cone import (
    assemble_scalar_crossing_cone,
    compare_known_gap_bound,
    CrossingConePlan,
    CrossingConicEvidence,
    CrossingExclusionEvidence,
    exclude_scalar_gap,
    KnownBoundEvidence,
    prepare_crossing_cone,
    PreparedCrossingCone,
    solve_crossing_cone,
)


__all__ = [
    "CrossingConePlan",
    "CrossingConicEvidence",
    "CrossingExclusionEvidence",
    "KnownBoundEvidence",
    "PreparedCrossingCone",
    "PreparedScalarBlocks",
    "ScalarBlockEvidence",
    "ScalarBlockPlan",
    "assemble_scalar_crossing_cone",
    "compare_known_gap_bound",
    "exclude_scalar_gap",
    "prepare_crossing_cone",
    "prepare_scalar_blocks",
    "solve_crossing_cone",
]
