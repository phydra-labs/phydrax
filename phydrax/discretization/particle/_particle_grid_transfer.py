#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._core import ParticleDiscretization
from ._pairwise import ParticleBox


class ParticleGridRelation(StrictModule, NonTrainableState):
    cell_indices: Array
    weights: Array
    valid: Array
    support_count: Array
    partition_residual: Array
    capacity_overflow: Array
    successful: Array
    transfer_id: str = eqx.field(static=True)


class ConservativeParticleGridTransferPlan(StrictModule, NonTrainableState):
    cell_centers: Array
    cell_volumes: Array
    support_radius: float = eqx.field(static=True)
    maximum_cells_per_particle: int = eqx.field(static=True)
    box: ParticleBox | None
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell_centers: ArrayLike,
        cell_volumes: ArrayLike,
        support_radius: float,
        maximum_cells_per_particle: int,
        /,
        *,
        box: ParticleBox | None = None,
        plan_id: str | None = None,
    ):
        centers = np.asarray(cell_centers)
        volumes = np.asarray(cell_volumes)
        radius = float(support_radius)
        capacity = int(maximum_cells_per_particle)
        if centers.ndim != 2 or centers.shape[0] == 0:
            raise ValueError("cell_centers must have shape (cells,dimension).")
        if volumes.shape != (centers.shape[0],):
            raise ValueError("cell_volumes must have cell shape.")
        if (
            np.any(~np.isfinite(centers))
            or np.any(~np.isfinite(volumes))
            or np.any(volumes <= 0.0)
            or not np.isfinite(radius)
            or radius <= 0.0
            or capacity <= 0
            or capacity > centers.shape[0]
        ):
            raise ValueError("Particle-grid transfer geometry/capacities are invalid.")
        if box is not None and (
            not isinstance(box, ParticleBox) or box.ambient_dimension != centers.shape[1]
        ):
            raise ValueError("Particle-grid box must match grid dimension.")
        generated = canonical_fingerprint(
            {
                "kind": "conservative-particle-grid-transfer-plan",
                "grid": array_tree_fingerprint({"centers": centers, "volumes": volumes}),
                "support_radius": radius,
                "maximum_cells_per_particle": capacity,
                "box": None if box is None else box.box_id,
            }
        )
        self.cell_centers = jnp.asarray(centers)
        self.cell_volumes = jnp.asarray(volumes)
        self.support_radius = radius
        self.maximum_cells_per_particle = capacity
        self.box = box
        self.plan_id = generated if plan_id is None else str(plan_id)
        if not self.plan_id:
            raise ValueError("plan_id must be nonempty.")

    def prepare(
        self, particles: ParticleDiscretization, /
    ) -> PreparedParticleGridTransfer:
        return PreparedParticleGridTransfer(self, particles)


class PreparedParticleGridTransfer(StrictModule, NonTrainableState):
    plan: ConservativeParticleGridTransferPlan
    particles: ParticleDiscretization
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: ConservativeParticleGridTransferPlan,
        particles: ParticleDiscretization,
        /,
    ):
        if not isinstance(plan, ConservativeParticleGridTransferPlan):
            raise TypeError("plan must be a ConservativeParticleGridTransferPlan.")
        if not isinstance(particles, ParticleDiscretization):
            raise TypeError("particles must be a ParticleDiscretization.")
        if particles.ambient_dimension != plan.cell_centers.shape[1]:
            raise ValueError("Particle and grid dimensions do not match.")
        self.plan = plan
        self.particles = particles
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-particle-grid-transfer",
                "plan": plan.plan_id,
                "particles": particles.prepared_id,
            }
        )

    @property
    def cell_count(self) -> int:
        return int(self.plan.cell_centers.shape[0])

    def relation(self, positions: ArrayLike, /) -> ParticleGridRelation:
        position = jnp.asarray(positions, dtype=self.plan.cell_centers.dtype)
        if position.shape != (
            self.particles.capacity,
            self.particles.ambient_dimension,
        ):
            raise ValueError("Particle positions do not match transfer support.")
        displacement = position[:, None, :] - self.plan.cell_centers[None, :, :]
        if self.plan.box is not None:
            displacement = self.plan.box.minimum_image(displacement)
        distance = jnp.sqrt(jnp.sum(displacement * displacement, axis=-1))
        q = distance / self.plan.support_radius
        raw = jnp.where(q < 1.0, (1.0 - q) ** 4 * (1.0 + 4.0 * q), 0.0)
        positive_count = jnp.sum(raw > 0.0, axis=-1, dtype=jnp.int32)
        values, indices = jax.lax.top_k(raw, self.plan.maximum_cells_per_particle)
        valid = values > 0.0
        weight_sum = jnp.sum(jnp.where(valid, values, 0.0), axis=-1)
        normalizable = weight_sum > 0.0
        weights = jnp.where(
            valid,
            values / jnp.where(normalizable, weight_sum, 1.0)[:, None],
            0.0,
        )
        active = self.particles.active_mask
        valid = valid & active[:, None]
        weights = jnp.where(valid, weights, 0.0)
        overflow = active & (positive_count > self.plan.maximum_cells_per_particle)
        partition = jnp.sum(weights, axis=-1)
        residual = jnp.where(active, jnp.abs(partition - 1.0), 0.0)
        successful = (
            ~jnp.any(overflow)
            & jnp.all(~active | normalizable)
            & jnp.all(jnp.isfinite(weights))
            & (jnp.max(residual) <= 1.0e-12)
        )
        return ParticleGridRelation(
            indices.astype(jnp.int32),
            weights,
            valid,
            positive_count,
            residual,
            overflow,
            successful,
            self.prepared_id,
        )

    def gather(
        self,
        relation: ParticleGridRelation,
        cell_values: ArrayLike,
        /,
    ) -> Array:
        self._validate_relation(relation)
        values = jnp.asarray(cell_values)
        if values.shape[0] != self.cell_count:
            raise ValueError("cell_values must begin with cell count.")
        gathered = values[relation.cell_indices]
        mask = relation.weights.reshape(
            relation.weights.shape + (1,) * (gathered.ndim - 2)
        )
        result = jnp.sum(mask * gathered, axis=1)
        active_mask = self.particles.active_mask.reshape(
            (self.particles.capacity,) + (1,) * (result.ndim - 1)
        )
        return jnp.where(active_mask, result, 0.0)

    def deposit_particle_content(
        self,
        relation: ParticleGridRelation,
        particle_content: ArrayLike,
        /,
    ) -> Array:
        self._validate_relation(relation)
        content = jnp.asarray(particle_content)
        if content.shape[0] != self.particles.capacity:
            raise ValueError("particle_content must begin with particle capacity.")
        output_shape = (self.cell_count,) + content.shape[1:]
        output = jnp.zeros(output_shape, dtype=content.dtype)
        payload = (
            relation.weights.reshape(relation.weights.shape + (1,) * (content.ndim - 1))
            * content[:, None, ...]
        )
        payload = jnp.where(
            relation.valid.reshape(relation.valid.shape + (1,) * (content.ndim - 1)),
            payload,
            0.0,
        )
        return output.at[relation.cell_indices.reshape(-1)].add(
            payload.reshape((-1,) + content.shape[1:])
        )

    def _validate_relation(self, relation: ParticleGridRelation, /) -> None:
        if not isinstance(relation, ParticleGridRelation):
            raise TypeError("relation must be a ParticleGridRelation.")
        if relation.transfer_id != self.prepared_id:
            raise ValueError("Particle-grid relation belongs to another transfer.")


__all__ = [
    "ConservativeParticleGridTransferPlan",
    "ParticleGridRelation",
    "PreparedParticleGridTransfer",
]
