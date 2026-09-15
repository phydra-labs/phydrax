#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._beam import AcceleratorBunch


class SpaceChargeKickPlan(StrictModule, NonTrainableState):
    source_plan_id: str = eqx.field(static=True)
    frame_transform_id: str = eqx.field(static=True)
    boundary_condition_id: str = eqx.field(static=True)
    maximum_residual: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        source_plan_id: str,
        frame_transform_id: str,
        boundary_condition_id: str,
        maximum_residual: float,
    ):
        source = str(source_plan_id).strip()
        frame = str(frame_transform_id).strip()
        boundary = str(boundary_condition_id).strip()
        residual = float(maximum_residual)
        if not source or not frame or not boundary or residual < 0.0:
            raise ValueError(
                "Space-charge source, frame, boundary, and residual policy are required."
            )
        self.source_plan_id = source
        self.frame_transform_id = frame
        self.boundary_condition_id = boundary
        self.maximum_residual = residual
        self.plan_id = canonical_fingerprint(
            {
                "kind": "accelerator-space-charge-kick",
                "source": source,
                "frame": frame,
                "boundary": boundary,
                "maximum_residual": residual,
            }
        )


class SpaceChargeKickResult(StrictModule, NonTrainableState):
    bunch: AcceleratorBunch
    momentum_kick: Array
    residual: Array
    accepted: Array
    derivative_valid: Array
    plan_id: str = eqx.field(static=True)


def apply_space_charge_kick(
    plan: SpaceChargeKickPlan,
    bunch: AcceleratorBunch,
    transverse_momentum_kick: ArrayLike,
    residual: ArrayLike,
    /,
) -> SpaceChargeKickResult:
    """Apply a qualified PIC/provider kick without pretending to own its field solve."""
    if not isinstance(plan, SpaceChargeKickPlan) or not isinstance(
        bunch, AcceleratorBunch
    ):
        raise TypeError("plan and bunch must use accelerator types.")
    kick = jnp.asarray(transverse_momentum_kick, dtype=bunch.coordinates.dtype)
    residual_ = jnp.asarray(residual, dtype=bunch.coordinates.dtype).reshape(())
    if kick.shape != (bunch.capacity, 2):
        raise ValueError("transverse_momentum_kick must have shape (bunch.capacity, 2).")
    accepted = (
        jnp.isfinite(residual_)
        & (residual_ <= plan.maximum_residual)
        & jnp.all(jnp.where(bunch.active[:, None], jnp.isfinite(kick), True))
    )
    candidate = bunch.coordinates.at[:, 1].add(jnp.where(bunch.active, kick[:, 0], 0.0))
    candidate = candidate.at[:, 3].add(jnp.where(bunch.active, kick[:, 1], 0.0))
    coordinates = jnp.where(accepted, candidate, bunch.coordinates)
    result = AcceleratorBunch(
        coordinates,
        bunch.weights,
        bunch.particle_ids,
        active=bunch.active,
        reference_rest_energy=float(bunch.reference_rest_energy),
        reference_momentum=float(bunch.reference_momentum),
        reference_charge=float(bunch.reference_charge),
        convention=bunch.convention,
        bunch_id=f"{bunch.bunch_id}:{plan.plan_id}",
    )
    return SpaceChargeKickResult(
        result,
        kick,
        residual_,
        accepted,
        jnp.broadcast_to(accepted, bunch.active.shape) & bunch.active,
        plan.plan_id,
    )


__all__ = [
    "SpaceChargeKickPlan",
    "SpaceChargeKickResult",
    "apply_space_charge_kick",
]
