#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._core import DiscretizationCapability, DiscretizationKey, PreparationReport
from ._cell_list import CellListParticleNeighborhoodPlan
from ._core import ParticleDiscretization
from ._neighborhood import (
    AbstractParticleNeighborhoodPlan,
    AbstractPreparedParticleNeighborhood,
    ParticleNeighborhoodState,
)
from ._pairwise import ParticleBox
from ._precision import ParticleRealization


class ParticleVerletState(StrictModule, NonTrainableState):
    neighborhood: ParticleNeighborhoodState
    reference_position: Array
    epoch: Array
    rebuilt: Array
    rebuild_count: Array
    maximum_reference_displacement: Array
    certificate_margin: Array
    successful: Array
    prepared_verlet_id: str = eqx.field(static=True)


class VerletParticleNeighborhoodPlan(AbstractParticleNeighborhoodPlan):
    """Certificate-based relation cache composed with one authority neighborhood."""

    base: AbstractParticleNeighborhoodPlan
    interaction_radius: float = eqx.field(static=True)
    skin: float = eqx.field(static=True)
    box: ParticleBox | None
    backend: ParticleRealization = eqx.field(static=True)
    key: DiscretizationKey
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        base: AbstractParticleNeighborhoodPlan,
        interaction_radius: float,
        skin: float,
        /,
        *,
        name: str = "verlet-particle-neighborhood",
        plan_id: str | None = None,
    ):
        if not isinstance(base, AbstractParticleNeighborhoodPlan):
            raise TypeError("base must be an AbstractParticleNeighborhoodPlan.")
        interaction = float(interaction_radius)
        skin_ = float(skin)
        if not np.isfinite(interaction) or interaction <= 0.0:
            raise ValueError("interaction_radius must be finite and positive.")
        if not np.isfinite(skin_) or skin_ <= 0.0:
            raise ValueError("skin must be finite and positive.")
        if isinstance(base, CellListParticleNeighborhoodPlan) and base.search_radius < (
            interaction + skin_
        ):
            raise ValueError(
                "Cell-list search radius must cover interaction_radius plus skin."
            )
        key = DiscretizationKey(
            name,
            base.key.role,
            domain_labels=base.key.domain_labels + ("verlet_cache",),
        )
        identifier = canonical_fingerprint(
            {
                "kind": "verlet-particle-neighborhood-plan",
                "base": base.plan_id,
                "interaction_radius": interaction,
                "skin": skin_,
                "key": key.key_id,
            }
        )
        self.base = base
        self.interaction_radius = interaction
        self.skin = skin_
        self.box = base.box
        self.backend = base.backend
        self.key = key
        self.plan_id = identifier if plan_id is None else str(plan_id)
        if not self.plan_id:
            raise ValueError("plan_id must be nonempty.")

    def prepare(
        self, particles: ParticleDiscretization, /
    ) -> PreparedVerletParticleNeighborhood:
        return PreparedVerletParticleNeighborhood(self, particles)


class PreparedVerletParticleNeighborhood(AbstractPreparedParticleNeighborhood):
    plan: VerletParticleNeighborhoodPlan
    base: AbstractPreparedParticleNeighborhood
    key: DiscretizationKey
    box: ParticleBox | None
    backend: ParticleRealization = eqx.field(static=True)
    pair_capacity: int = eqx.field(static=True)
    particle_capacity: int = eqx.field(static=True)
    particle_discretization_id: str = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)
    artifact_kind: str = eqx.field(static=True)
    preparation: PreparationReport
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self, plan: VerletParticleNeighborhoodPlan, particles: ParticleDiscretization, /
    ):
        if not isinstance(plan, VerletParticleNeighborhoodPlan):
            raise TypeError("plan must be a VerletParticleNeighborhoodPlan.")
        if not isinstance(particles, ParticleDiscretization):
            raise TypeError("particles must be a ParticleDiscretization.")
        base = plan.base.prepare(particles)
        preparation = PreparationReport(
            capabilities=tuple(
                set(base.preparation.capabilities)
                | {DiscretizationCapability.TOPOLOGY_REFRESH_FIXED_CAPACITY}
            ),
            diagnostics=base.preparation.diagnostics
            + (
                "candidate routes cached under a displacement certificate",
                "relation rebuild threshold is skin/2",
                "history remapping is required only after a rebuild",
            ),
            resource_counts={
                **dict(base.preparation.resource_counts),
                "reference_position_values": (
                    particles.capacity * particles.ambient_dimension
                ),
            },
        )
        self.plan = plan
        self.base = base
        self.key = plan.key
        self.box = plan.box
        self.backend = plan.backend
        self.pair_capacity = base.pair_capacity
        self.particle_capacity = particles.capacity
        self.particle_discretization_id = particles.prepared_id
        self.numeric_version = particles.numeric_version
        self.artifact_kind = "verlet-particle-neighborhood"
        self.preparation = preparation
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-verlet-particle-neighborhood",
                "plan": plan.plan_id,
                "base": base.prepared_id,
                "particles": particles.prepared_id,
                "preparation": preparation.report_id,
            }
        )

    def build(self, positions: ArrayLike, /) -> ParticleNeighborhoodState:
        """Build authority routes without cache reuse."""
        return self.base.build(positions)

    def initialize(self, positions: ArrayLike, /) -> ParticleVerletState:
        value = self._positions(positions)
        neighborhood = self.base.build(value)
        successful = neighborhood.successful & jnp.all(jnp.isfinite(value))
        return ParticleVerletState(
            neighborhood,
            value,
            jnp.zeros((), dtype=jnp.int32),
            jnp.asarray(True),
            jnp.asarray(1, dtype=jnp.int32),
            jnp.zeros((), dtype=value.dtype),
            jnp.asarray(0.5 * self.plan.skin, dtype=value.dtype),
            successful,
            self.prepared_id,
        )

    def update(
        self, positions: ArrayLike, previous: ParticleVerletState, /
    ) -> ParticleVerletState:
        if not isinstance(previous, ParticleVerletState):
            raise TypeError("previous must be a ParticleVerletState.")
        if previous.prepared_verlet_id != self.prepared_id:
            raise ValueError("Verlet state belongs to another prepared neighborhood.")
        value = self._positions(positions)
        displacement = value - previous.reference_position
        if self.box is not None:
            displacement = self.box.minimum_image(displacement)
        distance = jnp.sqrt(jnp.sum(displacement * displacement, axis=-1))
        maximum = jnp.max(distance)
        threshold = jnp.asarray(0.5 * self.plan.skin, dtype=value.dtype)
        finite = jnp.all(jnp.isfinite(value)) & jnp.isfinite(maximum)
        rebuild = (~previous.successful) | (~finite) | (maximum > threshold)

        def rebuild_routes(_):
            neighborhood = self.base.build(value)
            successful = neighborhood.successful & finite
            return ParticleVerletState(
                neighborhood,
                value,
                previous.epoch + jnp.asarray(1, dtype=jnp.int32),
                jnp.asarray(True),
                previous.rebuild_count + jnp.asarray(1, dtype=jnp.int32),
                maximum,
                threshold,
                successful,
                self.prepared_id,
            )

        def reuse_routes(_):
            return ParticleVerletState(
                previous.neighborhood,
                previous.reference_position,
                previous.epoch,
                jnp.asarray(False),
                previous.rebuild_count,
                maximum,
                threshold - maximum,
                previous.successful & finite,
                self.prepared_id,
            )

        return jax.lax.cond(rebuild, rebuild_routes, reuse_routes, operand=None)

    def _positions(self, positions: ArrayLike, /) -> Array:
        value = jnp.asarray(positions)
        expected = (
            (self.particle_capacity, self.base.plan.box.ambient_dimension)
            if self.base.plan.box is not None
            else None
        )
        if value.ndim != 2 or value.shape[0] != self.particle_capacity:
            raise ValueError(
                "Verlet positions must have shape (particle_capacity, dimension)."
            )
        if expected is not None and value.shape != expected:
            raise ValueError(f"Verlet positions must have shape {expected}.")
        return value


__all__ = [
    "ParticleVerletState",
    "PreparedVerletParticleNeighborhood",
    "VerletParticleNeighborhoodPlan",
]
