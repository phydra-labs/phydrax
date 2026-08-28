#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from ..._fingerprint import canonical_fingerprint
from ..._strict import AbstractAttribute, StrictModule
from ..._trainable import NonTrainableState


class AbstractSPHDensityPlan(StrictModule, NonTrainableState):
    """Static density-state semantics for an SPH method."""

    density_evolved: AbstractAttribute[bool]
    plan_id: AbstractAttribute[str]


class SummationDensityPlan(AbstractSPHDensityPlan):
    """Recompute density algebraically from current positions."""

    density_evolved: bool = False
    plan_id: str = canonical_fingerprint(
        {"kind": "sph-density-plan", "formulation": "summation"}
    )


class ContinuityDensityPlan(AbstractSPHDensityPlan):
    """Evolve density through the pairwise SPH continuity equation."""

    density_evolved: bool = True
    plan_id: str = canonical_fingerprint(
        {"kind": "sph-density-plan", "formulation": "continuity"}
    )


__all__ = [
    "AbstractSPHDensityPlan",
    "ContinuityDensityPlan",
    "SummationDensityPlan",
]
