#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Admitted secondary free-energy profiles and native event-exact CTMC workflows.

Only exhaustive bounded ordered-planar, linear, physically labelled strands are
compiled. Caller-supplied parameters govern DNA/RNA/hybrid chemistry; no external
parameter tables or experimentally calibrated kinetic prefactors are bundled.
"""

from ._compile import (
    CompiledSecondaryTarget,
    prepare_secondary_kinetics,
    PreparedSecondaryKinetics,
    SecondaryJumpProcess,
)
from ._model import AssociationConvention, SecondaryEnergyModel, SecondaryRateLaw
from ._state import SecondaryMove, SecondaryStructureState, StrandComplexPartition


__all__ = [
    "AssociationConvention",
    "CompiledSecondaryTarget",
    "PreparedSecondaryKinetics",
    "SecondaryEnergyModel",
    "SecondaryJumpProcess",
    "SecondaryMove",
    "SecondaryRateLaw",
    "SecondaryStructureState",
    "StrandComplexPartition",
    "prepare_secondary_kinetics",
]
