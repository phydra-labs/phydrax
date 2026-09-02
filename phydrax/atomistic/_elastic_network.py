#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Reference-derived fixed-capacity elastic networks for atomistic supports."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._system import PreparedAtomisticSystem


class ElasticNetworkPreparationEvidence(StrictModule, NonTrainableState):
    """Host-established network capacity and reference provenance."""

    candidate_count: int = eqx.field(static=True)
    edge_count: int = eqx.field(static=True)
    edge_capacity: int = eqx.field(static=True)
    reference_id: str = eqx.field(static=True)
    successful: bool = eqx.field(static=True)


class ElasticNetworkEvaluation(StrictModule):
    """Conservative spring energy, forces, and fixed-capacity edge ledger."""

    energy: Array
    forces: Array
    edge_energies: Array
    finite: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)


class ElasticNetworkPlan(StrictModule, NonTrainableState):
    """Cutoff and resource policy for a reference-derived harmonic network."""

    cutoff: float = eqx.field(static=True)
    stiffness: float = eqx.field(static=True)
    edge_capacity: int = eqx.field(static=True)
    minimum_particle_id_separation: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        cutoff: float,
        stiffness: float,
        edge_capacity: int,
        /,
        *,
        minimum_particle_id_separation: int = 0,
    ):
        cutoff_ = float(cutoff)
        stiffness_ = float(stiffness)
        if not isinstance(edge_capacity, (int, np.integer)) or isinstance(
            edge_capacity, bool
        ):
            raise TypeError("edge_capacity must be an integer.")
        if not isinstance(
            minimum_particle_id_separation, (int, np.integer)
        ) or isinstance(minimum_particle_id_separation, bool):
            raise TypeError("minimum_particle_id_separation must be an integer.")
        capacity = int(edge_capacity)
        separation = int(minimum_particle_id_separation)
        if (
            not np.isfinite(cutoff_)
            or not np.isfinite(stiffness_)
            or cutoff_ <= 0.0
            or stiffness_ <= 0.0
        ):
            raise ValueError(
                "Elastic-network cutoff and stiffness must be finite and positive."
            )
        if capacity <= 0:
            raise ValueError("Elastic-network edge_capacity must be positive.")
        if separation < 0:
            raise ValueError("minimum_particle_id_separation must be non-negative.")
        self.cutoff = cutoff_
        self.stiffness = stiffness_
        self.edge_capacity = capacity
        self.minimum_particle_id_separation = separation
        self.plan_id = canonical_fingerprint(
            {
                "kind": "elastic-network-plan",
                "cutoff": cutoff_.hex(),
                "stiffness": stiffness_.hex(),
                "edge_capacity": capacity,
                "minimum_particle_id_separation": separation,
            }
        )

    def prepare(
        self,
        system: PreparedAtomisticSystem,
        reference_positions: ArrayLike,
        /,
        *,
        reference_id: str | None = None,
    ) -> "PreparedElasticNetwork":
        return PreparedElasticNetwork(
            self, system, reference_positions, reference_id=reference_id
        )


class PreparedElasticNetwork(StrictModule, NonTrainableState):
    """Stable edge slots and equilibrium lengths bound to one atomistic system."""

    plan: ElasticNetworkPlan
    system: PreparedAtomisticSystem
    edge_indices: Array
    edge_particle_ids: Array
    equilibrium_lengths: Array
    valid: Array
    preparation: ElasticNetworkPreparationEvidence
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: ElasticNetworkPlan,
        system: PreparedAtomisticSystem,
        reference_positions: ArrayLike,
        /,
        *,
        reference_id: str | None = None,
    ):
        if not isinstance(plan, ElasticNetworkPlan):
            raise TypeError("plan must be an ElasticNetworkPlan.")
        if not isinstance(system, PreparedAtomisticSystem):
            raise TypeError("system must be a PreparedAtomisticSystem.")
        reference = np.asarray(reference_positions)
        if reference.shape != (system.capacity, 3):
            raise ValueError(
                "Reference positions must match atomistic capacity with shape (capacity, 3)."
            )
        if reference.dtype.kind != "f":
            raise TypeError("Elastic-network reference positions must be floating point.")
        active = np.asarray(system.active_mask, dtype=bool)
        if np.any(~np.isfinite(reference[active])):
            raise ValueError("Active elastic-network reference positions must be finite.")
        reference = np.where(active[:, None], reference, 0.0)
        particle_ids = np.asarray(system.plan.particle_ids, dtype=np.int64)
        candidates: list[tuple[int, int, int, int, float]] = []
        for left in range(system.capacity):
            if not active[left]:
                continue
            for right in range(left + 1, system.capacity):
                if not active[right]:
                    continue
                left_id, right_id = int(particle_ids[left]), int(particle_ids[right])
                if abs(left_id - right_id) < plan.minimum_particle_id_separation:
                    continue
                displacement = jnp.asarray(reference[left] - reference[right])
                if system.cell is not None:
                    displacement = system.cell.minimum_image(displacement)
                distance = float(jnp.sqrt(jnp.sum(displacement * displacement)))
                if distance <= plan.cutoff:
                    if distance <= 0.0:
                        raise ValueError(
                            "Elastic-network reference edges require distinct positions."
                        )
                    low_id, high_id = sorted((left_id, right_id))
                    candidates.append((low_id, high_id, left, right, distance))
        candidates.sort(key=lambda value: (value[0], value[1]))
        required = len(candidates)
        if required > plan.edge_capacity:
            raise ValueError(
                f"Elastic-network edge capacity {plan.edge_capacity} is smaller than required count {required}."
            )
        indices = np.zeros((plan.edge_capacity, 2), dtype=np.int32)
        ids = np.zeros((plan.edge_capacity, 2), dtype=np.int64)
        distances = np.ones((plan.edge_capacity,), dtype=reference.dtype)
        valid = np.zeros((plan.edge_capacity,), dtype=bool)
        for slot, (left_id, right_id, left, right, distance) in enumerate(candidates):
            indices[slot] = (left, right)
            ids[slot] = (left_id, right_id)
            distances[slot] = distance
            valid[slot] = True
        generated_reference = canonical_fingerprint(
            {
                "kind": "elastic-network-reference",
                "system": system.prepared_id,
                "positions": array_tree_fingerprint(reference),
            }
        )
        resolved_reference = (
            generated_reference if reference_id is None else str(reference_id).strip()
        )
        if not resolved_reference:
            raise ValueError("reference_id must be non-empty.")
        preparation = ElasticNetworkPreparationEvidence(
            required,
            required,
            plan.edge_capacity,
            resolved_reference,
            True,
        )
        self.plan = plan
        self.system = system
        self.edge_indices = jnp.asarray(indices)
        self.edge_particle_ids = jnp.asarray(ids)
        self.equilibrium_lengths = jnp.asarray(distances)
        self.valid = jnp.asarray(valid)
        self.preparation = preparation
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-elastic-network",
                "plan": plan.plan_id,
                "system": system.prepared_id,
                "reference": resolved_reference,
                "edges": array_tree_fingerprint(
                    {"particle_ids": ids, "lengths": distances, "valid": valid}
                ),
            }
        )

    def evaluate(self, positions: ArrayLike, /) -> ElasticNetworkEvaluation:
        coordinate = jnp.asarray(positions)
        if coordinate.shape != (self.system.capacity, 3):
            raise ValueError("Elastic-network positions must match atomistic capacity.")
        if coordinate.dtype.kind != "f":
            raise TypeError("Elastic-network positions must be floating point.")
        left = self.edge_indices[:, 0]
        right = self.edge_indices[:, 1]
        displacement = jnp.where(
            self.valid[:, None], coordinate[left] - coordinate[right], 0.0
        )
        if self.system.cell is not None:
            displacement = self.system.cell.minimum_image(displacement)
        squared = jnp.sum(displacement * displacement, axis=-1)
        safe_squared = jnp.where(self.valid, squared, 1.0)
        distance = jnp.sqrt(safe_squared)
        extension = distance - self.equilibrium_lengths
        edge_energy = jnp.where(
            self.valid, 0.5 * self.plan.stiffness * extension * extension, 0.0
        )
        safe_distance = jnp.where(distance > 0.0, distance, 1.0)
        radial_force = jnp.where(
            self.valid,
            -self.plan.stiffness * extension / safe_distance,
            0.0,
        )
        edge_force = radial_force[:, None] * displacement
        forces = jnp.zeros_like(coordinate)
        forces = forces.at[left].add(edge_force)
        forces = forces.at[right].add(-edge_force)
        energy = jnp.sum(edge_energy)
        active_finite = jnp.all(
            jnp.where(
                self.system.active_mask[:, None],
                jnp.isfinite(coordinate),
                True,
            )
        )
        finite = (
            active_finite
            & jnp.isfinite(energy)
            & jnp.all(jnp.isfinite(edge_energy))
            & jnp.all(jnp.isfinite(forces))
        )
        geometry_valid = jnp.all(~self.valid | (squared > 0.0))
        return ElasticNetworkEvaluation(
            energy,
            forces,
            edge_energy,
            finite,
            finite & geometry_valid,
            self.prepared_id,
        )


__all__ = [
    "ElasticNetworkEvaluation",
    "ElasticNetworkPlan",
    "ElasticNetworkPreparationEvidence",
    "PreparedElasticNetwork",
]
