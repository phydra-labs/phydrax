#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...solver._particle_gravity import (
    BarnesHutGravityPlan,
    ParticleOctreePlan3D,
    PreparedParticleOctree3D,
)
from ._status import AstrodynamicsStatus


CollisionMode: TypeAlias = Literal["fail", "merge", "bounce"]


class CloseEncounterPolicy(StrictModule, NonTrainableState):
    encounter_distance: Array
    collision_distance: Array
    collision_mode: CollisionMode = eqx.field(static=True)
    restitution: Array

    def __init__(
        self,
        encounter_distance,
        collision_distance,
        /,
        *,
        collision_mode="fail",
        restitution=1.0,
    ):
        if collision_mode not in ("fail", "merge", "bounce"):
            raise ValueError("Unknown collision mode.")
        self.encounter_distance = jnp.asarray(encounter_distance).reshape(())
        self.collision_distance = jnp.asarray(collision_distance).reshape(())
        self.collision_mode = collision_mode
        self.restitution = jnp.asarray(restitution).reshape(())
        if (
            float(self.encounter_distance) < float(self.collision_distance)
            or float(self.collision_distance) < 0.0
        ):
            raise ValueError("Encounter distances are inconsistent.")


class EncounterEvaluation(StrictModule):
    minimum_distance: Array
    pair: Array
    encountered: Array
    collided: Array
    regularization_prepared: Array
    status: Array


def detect_close_encounter(
    positions: ArrayLike,
    policy: CloseEncounterPolicy,
    /,
    *,
    regularization_prepared: ArrayLike = False,
) -> EncounterEvaluation:
    values = jnp.asarray(positions)
    count = values.shape[0]
    displacement = values[:, None, :] - values[None, :, :]
    distance = jnp.sqrt(jnp.sum(displacement * displacement, axis=-1))
    distance = jnp.where(jnp.eye(count, dtype=bool), jnp.inf, distance)
    flat = jnp.argmin(distance)
    pair = jnp.asarray((flat // count, flat % count), dtype=jnp.int32)
    minimum = jnp.min(distance)
    collided = minimum <= policy.collision_distance
    encountered = minimum <= policy.encounter_distance
    regularized = (
        encountered
        & ~collided
        & jnp.asarray(regularization_prepared, dtype=bool).reshape(())
    )
    status = jnp.where(
        collided,
        int(AstrodynamicsStatus.COLLISION),
        jnp.where(
            encountered & ~regularized,
            int(AstrodynamicsStatus.UNSUPPORTED_REGIME),
            int(AstrodynamicsStatus.SUCCESS),
        ),
    ).astype(jnp.int32)
    return EncounterEvaluation(minimum, pair, encountered, collided, regularized, status)


class PreparedOctree3D(StrictModule, NonTrainableState):
    """Astrodynamics coordinates bound to the core runtime Morton octree plan."""

    origin: Array
    box_size: tuple[float, float, float] = eqx.field(static=True)
    masses: Array
    plan: ParticleOctreePlan3D
    prepared: PreparedParticleOctree3D
    tree_id: str = eqx.field(static=True)

    def __init__(
        self,
        reference_positions: ArrayLike,
        masses: ArrayLike,
        /,
        *,
        leaf_capacity=8,
        maximum_depth=24,
    ):
        positions = np.asarray(reference_positions, dtype=float)
        mass_values = np.asarray(masses, dtype=float)
        if (
            positions.ndim != 2
            or positions.shape[1] != 3
            or mass_values.shape != (positions.shape[0],)
        ):
            raise ValueError("Octree positions/masses have incompatible shapes.")
        capacity = max(int(leaf_capacity), 1)
        requested_depth = int(
            np.ceil(np.log(max(positions.shape[0] / capacity, 1.0)) / np.log(8.0))
        )
        depth = min(max(requested_depth, 1), min(int(maximum_depth), 10))
        minimum = np.min(positions, axis=0)
        maximum = np.max(positions, axis=0)
        extent = max(float(np.max(maximum - minimum)), np.finfo(float).eps)
        padding = 16.0 * np.finfo(float).eps * max(abs(extent), 1.0)
        origin = minimum - padding
        box = (float(extent + 2.0 * padding),) * 3
        local = positions - origin[None, :]
        plan = ParticleOctreePlan3D(box, depth)
        prepared = plan.prepare(local, mass_values)
        self.origin = jnp.asarray(origin)
        self.box_size = box
        self.masses = jnp.asarray(mass_values)
        self.plan = plan
        self.prepared = prepared
        self.tree_id = canonical_fingerprint(
            {
                "kind": "astrodynamics-core-octree-adapter",
                "plan": plan.plan_id,
                "particles": positions.shape[0],
            }
        )

    def prepare(self, positions: ArrayLike, /) -> PreparedParticleOctree3D:
        values = jnp.asarray(positions)
        return self.plan.prepare(values - self.origin[None, :], self.masses)


class HierarchicalGravityResult(StrictModule):
    acceleration: Array
    potential: Array
    valid: Array
    status: Array
    plan_id: str = eqx.field(static=True)


class BarnesHutGravityPlan3D(StrictModule, NonTrainableState):
    """Astrodynamics adapter over the core Barnes-Hut force engine."""

    tree: PreparedOctree3D
    masses: Array
    core: BarnesHutGravityPlan
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        tree,
        masses,
        /,
        *,
        gravitational_constant=1.0,
        opening_angle=0.5,
        softening=1.0e-15,
    ):
        if not isinstance(tree, PreparedOctree3D):
            raise TypeError("tree must be PreparedOctree3D.")
        mass = jnp.asarray(masses)
        if mass.shape != tree.masses.shape:
            raise ValueError("Barnes-Hut masses must match the prepared tree.")
        self.tree = tree
        self.masses = mass
        self.core = BarnesHutGravityPlan(
            gravitational_constant,
            softening=softening,
            opening_angle=opening_angle,
            use_quadrupole=True,
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "astrodynamics-barnes-hut-adapter",
                "tree": tree.tree_id,
                "core": self.core.plan_id,
            }
        )

    def evaluate(self, positions: ArrayLike, /) -> HierarchicalGravityResult:
        prepared = self.tree.prepare(positions)
        result = self.core.evaluate(prepared)
        status = jnp.where(
            result.successful,
            int(AstrodynamicsStatus.SUCCESS),
            int(AstrodynamicsStatus.NONFINITE_INPUT),
        ).astype(jnp.int32)
        return HierarchicalGravityResult(
            result.acceleration,
            jnp.zeros((prepared.positions.shape[0],), dtype=prepared.positions.dtype),
            result.successful,
            status,
            self.plan_id,
        )


__all__ = [
    "BarnesHutGravityPlan3D",
    "CloseEncounterPolicy",
    "EncounterEvaluation",
    "HierarchicalGravityResult",
    "PreparedOctree3D",
    "detect_close_encounter",
]
