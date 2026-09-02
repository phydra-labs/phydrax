#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._transfer import PreparedPICParticleCochainTransfer
from ._types import PICTransferState


class PICParticleResponseState(StrictModule):
    routes: PICTransferState
    magnetic: Array
    alpha: Array
    rotated_velocity: Array
    coefficient: Array
    active: Array


class PICParticleResponseResult(StrictModule):
    current: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class PICParticleResponsePlan(StrictModule, NonTrainableState):
    """Matrix-free ECSIM gather/rotate/scatter response."""

    transfer: PreparedPICParticleCochainTransfer
    plan_id: str = eqx.field(static=True)

    def __init__(self, transfer: PreparedPICParticleCochainTransfer, /):
        if not isinstance(transfer, PreparedPICParticleCochainTransfer):
            raise TypeError("transfer must be PreparedPICParticleCochainTransfer.")
        if transfer.bridge.dimension != 3:
            raise ValueError("Particle response currently requires three dimensions.")
        self.transfer = transfer
        self.plan_id = canonical_fingerprint(
            {"kind": "pic-particle-response", "transfer": transfer.prepared_id}
        )

    def prepare_state(
        self,
        position: ArrayLike,
        velocity: ArrayLike,
        magnetic_cochain: ArrayLike,
        macrocharge: ArrayLike,
        mass: ArrayLike,
        active_mask: ArrayLike,
        step_size: ArrayLike,
        /,
    ) -> PICParticleResponseState:
        routes = self.transfer.build(position, active_mask=active_mask)
        gathered = self.transfer.gather_magnetic(routes, magnetic_cochain)
        velocity_ = jnp.asarray(velocity, dtype=gathered.values.dtype)
        charge = jnp.asarray(macrocharge, dtype=velocity_.dtype)
        masses = jnp.asarray(mass, dtype=velocity_.dtype)
        active = jnp.asarray(active_mask, dtype=bool)
        dt = jnp.asarray(step_size, dtype=velocity_.dtype).reshape(())
        beta = jnp.where(active, charge * dt / (2.0 * jnp.maximum(masses, 1.0e-30)), 0.0)
        magnetic = gathered.values
        cross_matrix = jnp.zeros((velocity_.shape[0], 3, 3), dtype=velocity_.dtype)
        bx, by, bz = magnetic[:, 0], magnetic[:, 1], magnetic[:, 2]
        cross_matrix = cross_matrix.at[:, 0, 1].set(-bz)
        cross_matrix = cross_matrix.at[:, 0, 2].set(by)
        cross_matrix = cross_matrix.at[:, 1, 0].set(bz)
        cross_matrix = cross_matrix.at[:, 1, 2].set(-bx)
        cross_matrix = cross_matrix.at[:, 2, 0].set(-by)
        cross_matrix = cross_matrix.at[:, 2, 1].set(bx)
        identity = jnp.broadcast_to(jnp.eye(3, dtype=velocity_.dtype), cross_matrix.shape)
        outer = magnetic[:, :, None] * magnetic[:, None, :]
        denominator = 1.0 + beta**2 * jnp.sum(magnetic**2, axis=-1)
        alpha = (
            identity
            - beta[:, None, None] * cross_matrix
            + beta[:, None, None] ** 2 * outer
        ) / denominator[:, None, None]
        rotated = contract("pij,pj->pi", alpha, velocity_)
        coefficient = jnp.where(
            active,
            charge**2 * dt / (2.0 * jnp.maximum(masses, 1.0e-30)),
            0.0,
        )
        return PICParticleResponseState(
            routes, magnetic, alpha, rotated, coefficient, active
        )

    def scatter_vector(
        self,
        state: PICParticleResponseState,
        vector: ArrayLike,
        /,
    ) -> PICParticleResponseResult:
        values = jnp.asarray(vector, dtype=state.rotated_velocity.dtype)
        if values.shape != state.rotated_velocity.shape:
            raise ValueError("Particle response vector must have shape (capacity,3).")
        components = []
        successful = jnp.asarray(True)
        for axis, (transfer, route) in enumerate(
            zip(self.transfer.electric, state.routes.electric, strict=True)
        ):
            deposited = transfer.deposit_content(
                route, jnp.where(state.active, values[:, axis], 0.0)
            )
            components.append(deposited.density)
            successful = successful & deposited.successful
        current = self.transfer.bridge.pack_edge_circulation(tuple(components))
        finite = jnp.all(jnp.isfinite(current))
        return PICParticleResponseResult(
            current, finite, successful & finite, self.plan_id
        )

    def known_current(
        self, state: PICParticleResponseState, macrocharge: ArrayLike, /
    ) -> PICParticleResponseResult:
        charge = jnp.asarray(macrocharge, dtype=state.rotated_velocity.dtype)
        return self.scatter_vector(state, charge[:, None] * state.rotated_velocity)

    def apply(
        self,
        state: PICParticleResponseState,
        electric_cochain: ArrayLike,
        /,
    ) -> PICParticleResponseResult:
        gathered = self.transfer.gather_electric(state.routes, electric_cochain)
        rotated = contract("pij,pj->pi", state.alpha, gathered.values)
        return self.scatter_vector(state, state.coefficient[:, None] * rotated)


__all__ = [
    "PICParticleResponsePlan",
    "PICParticleResponseResult",
    "PICParticleResponseState",
]
