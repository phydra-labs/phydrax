#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ....discretization.vortex._interfaces import VortexFieldRequest
from ....discretization.vortex._ring_sheet import VortexRingSheetState
from ....discretization.vortex._source import VortexTargetState
from ._filament3d import regularized_filament_velocity_3d


class RingSheetFieldEvaluation(StrictModule):
    velocity: Array | None
    velocity_gradient: Array | None
    vorticity: Array | None
    active_segment_count: Array
    finite: Array
    successful: Array
    evaluation_id: str = eqx.field(static=True)


class PreparedRingSheetField3D(StrictModule):
    state: VortexRingSheetState
    evaluator_id: str = eqx.field(static=True)

    def __init__(self, state: VortexRingSheetState, /):
        if not isinstance(state, VortexRingSheetState):
            raise TypeError("state must be VortexRingSheetState.")
        self.state = state
        self.evaluator_id = canonical_fingerprint(
            {
                "kind": "prepared-ring-sheet-field-3d",
                "topology": state.topology.topology_id,
            }
        )

    def _velocity(self, target: Array, /) -> Array:
        start, end, circulation = self.state.edge_geometry()
        return regularized_filament_velocity_3d(
            target, start, end, circulation, self.state.edge_core_radius
        )

    def evaluate(
        self,
        target: VortexTargetState,
        /,
        *,
        request: VortexFieldRequest = VortexFieldRequest(),
    ) -> RingSheetFieldEvaluation:
        if not isinstance(target, VortexTargetState) or target.dimension != 3:
            raise ValueError("Ring-sheet field targets must be three-dimensional.")
        velocity_all = self._velocity(target.positions)
        gradient_all = None
        if request.velocity_gradient or request.vorticity:
            gradient_all = jax.vmap(
                jax.jacfwd(lambda point: self._velocity(point[None, :])[0])
            )(target.positions)
        if request.vorticity:
            vorticity_all = jnp.stack(
                (
                    gradient_all[:, 2, 1] - gradient_all[:, 1, 2],
                    gradient_all[:, 0, 2] - gradient_all[:, 2, 0],
                    gradient_all[:, 1, 0] - gradient_all[:, 0, 1],
                ),
                axis=-1,
            )
        else:
            vorticity_all = None
        finite = jnp.all(jnp.isfinite(velocity_all))
        if gradient_all is not None:
            finite = finite & jnp.all(jnp.isfinite(gradient_all))
        return RingSheetFieldEvaluation(
            velocity_all if request.velocity else None,
            gradient_all if request.velocity_gradient else None,
            vorticity_all,
            jnp.sum(self.state.topology.edge_active, dtype=jnp.int32),
            finite,
            finite,
            canonical_fingerprint(
                {
                    "kind": "ring-sheet-field-evaluation",
                    "evaluator": self.evaluator_id,
                    "target_count": target.capacity,
                    "request": request.request_id,
                }
            ),
        )


__all__ = ["PreparedRingSheetField3D", "RingSheetFieldEvaluation"]
