#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..operators.integral.layer_potential._periodic_vector3d import (
    PeriodicMaxwellElectricFieldAction3D,
)


class PeriodicVectorBoundarySolveUnsupportedError(NotImplementedError):
    """A field-only periodic vector provider was requested as a boundary solve."""


class PeriodicVectorBoundarySupport3D(StrictModule, NonTrainableState):
    """Fail-closed capability evidence for a periodic vector field product."""

    ambient_dimension: int = eqx.field(static=True)
    pde: str = eqx.field(static=True)
    geometry: str = eqx.field(static=True)
    formulation: str = eqx.field(static=True)
    provider: str = eqx.field(static=True)
    precision: str = eqx.field(static=True)
    resource_evidence: str = eqx.field(static=True)
    error_evidence: str = eqx.field(static=True)
    non_goals: tuple[str, ...] = eqx.field(static=True)
    action_id: str = eqx.field(static=True)
    action_support_id: str = eqx.field(static=True)
    off_surface_field_action_supported: bool = eqx.field(static=True)
    boundary_trace_supported: bool = eqx.field(static=True)
    boundary_self_action_supported: bool = eqx.field(static=True)
    boundary_solve_supported: bool = eqx.field(static=True)
    continuum_certified: bool = eqx.field(static=True)
    support_id: str = eqx.field(static=True)


def periodic_vector_boundary_support_3d(
    action: PeriodicMaxwellElectricFieldAction3D, /
) -> PeriodicVectorBoundarySupport3D:
    """Return explicit capabilities without promoting a field action to a solve."""

    if not isinstance(action, PeriodicMaxwellElectricFieldAction3D):
        raise TypeError("action must be PeriodicMaxwellElectricFieldAction3D.")
    field_support = action.support
    if (
        not field_support.off_surface_field_action_supported
        or field_support.boundary_self_action_supported
        or field_support.boundary_solve_supported
    ):
        raise ValueError("Periodic Maxwell field support evidence is inconsistent.")
    support_id = canonical_fingerprint(
        {
            "kind": "periodic-vector-boundary-support-3d-v1",
            "action": action.action_id,
            "action_support": field_support.support_id,
            "boundary_trace": False,
            "boundary_self_action": False,
            "boundary_solve": False,
        }
    )
    return PeriodicVectorBoundarySupport3D(
        ambient_dimension=3,
        pde=field_support.pde,
        geometry=field_support.geometry,
        formulation=(
            "capability gate over the prepared off-surface electric Green-dyadic "
            "action; no boundary-limit operator is constructed"
        ),
        provider="PHYDRA periodic vector fail-closed boundary capability gate",
        precision=field_support.precision,
        resource_evidence=field_support.resource_evidence,
        error_evidence=field_support.error_evidence,
        non_goals=(
            "no on-surface tangential trace or jump relation",
            "no periodic EFIE, MFIE, CFIE, or Calderon self operator",
            "no right-hand-side assembly",
            "no linear boundary solve",
            "no inference of solve support from transpose or adjoint actions",
            "no continuum certification",
        ),
        action_id=action.action_id,
        action_support_id=field_support.support_id,
        off_surface_field_action_supported=True,
        boundary_trace_supported=False,
        boundary_self_action_supported=False,
        boundary_solve_supported=False,
        continuum_certified=False,
        support_id=support_id,
    )


def require_periodic_vector_boundary_solve_3d(
    action: PeriodicMaxwellElectricFieldAction3D, /
) -> None:
    """Reject use of the field-only product as a periodic boundary solver."""

    support = periodic_vector_boundary_support_3d(action)
    raise PeriodicVectorBoundarySolveUnsupportedError(
        "Periodic vector support provides only a guarded off-surface Maxwell "
        f"field action ({support.action_id}); boundary self action and boundary "
        "solve support are explicitly absent."
    )


__all__ = [
    "PeriodicVectorBoundarySolveUnsupportedError",
    "PeriodicVectorBoundarySupport3D",
    "periodic_vector_boundary_support_3d",
    "require_periodic_vector_boundary_solve_3d",
]
