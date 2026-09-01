#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx

from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....geometry import MeshRegion
from ....linalg import LinearCapabilityError


class DisplacementDiscontinuityCapability3D(StrictModule, NonTrainableState):
    """Explicit negative capability for an unverified open-sheet kernel."""

    supported: bool = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    pde: str = eqx.field(static=True)
    geometry: str = eqx.field(static=True)
    formulation: str = eqx.field(static=True)
    provider: str = eqx.field(static=True)
    precision: str = eqx.field(static=True)
    resource_evidence: str = eqx.field(static=True)
    error_evidence: str = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    non_goals: tuple[str, ...] = eqx.field(static=True)


_CAPABILITY = DisplacementDiscontinuityCapability3D(
    supported=False,
    ambient_dimension=3,
    pde="static homogeneous isotropic Navier-Cauchy elasticity",
    geometry="constant triangular elements on an oriented open sheet",
    formulation="displacement discontinuity across the sheet",
    provider="none: no primary-formula implementation is approved",
    precision="none: no numerical action is exposed",
    resource_evidence="zero resident/action workspace because preparation is rejected",
    error_evidence=(
        "fail-closed rejection; no kernel values, quadrature estimates, or continuum claims"
    ),
    reason=(
        "A clean-room constant-triangle open-sheet displacement-discontinuity formula, "
        "including edge limits and sign conventions, is not yet source-verified."
    ),
    non_goals=(
        "point-source Kelvin layers presented as displacement-discontinuity elements",
        "closed-surface DP0 elasticity aliases",
        "fracture propagation, contact, nonlinear elasticity, or dynamics",
        "placeholder or approximate fallback kernels",
    ),
)


def displacement_discontinuity_capability_3d() -> DisplacementDiscontinuityCapability3D:
    """Return immutable evidence explaining why this route is unavailable."""
    return _CAPABILITY


def prepare_displacement_discontinuity_dp0_3d(
    region: MeshRegion,
    /,
) -> None:
    """Reject the unverified constant-triangle open-sheet capability."""
    if not isinstance(region, MeshRegion):
        raise TypeError("region must be a MeshRegion.")
    raise LinearCapabilityError(_CAPABILITY.reason)


__all__ = [
    "DisplacementDiscontinuityCapability3D",
    "displacement_discontinuity_capability_3d",
    "prepare_displacement_discontinuity_dp0_3d",
]
