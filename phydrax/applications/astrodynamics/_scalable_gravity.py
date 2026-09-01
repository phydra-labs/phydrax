#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
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
    status: Array


def detect_close_encounter(
    positions: ArrayLike, policy: CloseEncounterPolicy, /
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
    status = jnp.where(
        collided,
        int(AstrodynamicsStatus.COLLISION),
        jnp.where(
            encountered,
            int(AstrodynamicsStatus.UNSUPPORTED_REGIME),
            int(AstrodynamicsStatus.SUCCESS),
        ),
    ).astype(jnp.int32)
    return EncounterEvaluation(minimum, pair, encountered, collided, status)


@dataclass(frozen=True)
class _NodeSpec:
    members: tuple[int, ...]
    children: tuple[int, ...]


class PreparedOctree3D(StrictModule, NonTrainableState):
    center: Array
    half_size: Array
    mass: Array
    center_of_mass: Array
    specs: tuple[_NodeSpec, ...] = eqx.field(static=True)
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
        centers: list[np.ndarray] = []
        half_sizes: list[float] = []
        node_masses: list[float] = []
        centers_of_mass: list[np.ndarray] = []
        specs: list[_NodeSpec | None] = []

        def build(
            indices: np.ndarray, center: np.ndarray, half_size: float, depth: int
        ) -> int:
            node_index = len(specs)
            specs.append(None)
            centers.append(center)
            half_sizes.append(half_size)
            total_mass = float(np.sum(mass_values[indices]))
            center_mass = (
                np.sum(mass_values[indices, None] * positions[indices], axis=0)
                / total_mass
            )
            node_masses.append(total_mass)
            centers_of_mass.append(center_mass)
            children: list[int] = []
            if indices.size > leaf_capacity and depth < maximum_depth and half_size > 0.0:
                octants = (
                    (positions[indices] >= center).astype(int) * np.asarray((1, 2, 4))
                ).sum(axis=1)
                for octant in range(8):
                    child_indices = indices[octants == octant]
                    if child_indices.size == 0:
                        continue
                    signs = np.asarray(
                        tuple(1.0 if octant & bit else -1.0 for bit in (1, 2, 4))
                    )
                    child_center = center + signs * (0.5 * half_size)
                    children.append(
                        build(child_indices, child_center, 0.5 * half_size, depth + 1)
                    )
            specs[node_index] = _NodeSpec(
                tuple(int(value) for value in indices), tuple(children)
            )
            return node_index

        minimum = np.min(positions, axis=0)
        maximum = np.max(positions, axis=0)
        root_center = 0.5 * (minimum + maximum)
        root_half = 0.5 * float(np.max(maximum - minimum)) + np.finfo(float).eps
        build(np.arange(positions.shape[0]), root_center, root_half, 0)
        self.center = jnp.asarray(np.stack(centers))
        self.half_size = jnp.asarray(half_sizes)
        self.mass = jnp.asarray(node_masses)
        self.center_of_mass = jnp.asarray(np.stack(centers_of_mass))
        self.specs = tuple(spec for spec in specs if spec is not None)
        self.tree_id = canonical_fingerprint(
            {
                "kind": "prepared-octree-3d",
                "particles": int(positions.shape[0]),
                "nodes": len(specs),
                "leaf_capacity": int(leaf_capacity),
            }
        )


class HierarchicalGravityResult(StrictModule):
    acceleration: Array
    potential: Array
    valid: Array
    status: Array
    plan_id: str = eqx.field(static=True)


class BarnesHutGravityPlan3D(StrictModule, NonTrainableState):
    tree: PreparedOctree3D
    masses: Array
    gravitational_constant: Array
    opening_angle: Array
    plan_id: str = eqx.field(static=True)

    def __init__(self, tree, masses, /, *, gravitational_constant=1.0, opening_angle=0.5):
        self.tree = tree
        self.masses = jnp.asarray(masses)
        self.gravitational_constant = jnp.asarray(gravitational_constant).reshape(())
        self.opening_angle = jnp.asarray(opening_angle).reshape(())
        self.plan_id = canonical_fingerprint(
            {
                "kind": "barnes-hut-gravity-3d",
                "tree": tree.tree_id,
                "opening_angle": float(self.opening_angle),
            }
        )

    def evaluate(self, positions: ArrayLike, /) -> HierarchicalGravityResult:
        values = jnp.asarray(positions)

        def target(index, position):
            def node_contribution(node_index: int):
                spec = self.tree.specs[node_index]
                relative = self.tree.center_of_mass[node_index] - position
                distance = jnp.sqrt(jnp.sum(relative * relative))
                contains = jnp.any(index == jnp.asarray(spec.members, dtype=jnp.int32))
                accept = (~contains) & (
                    self.tree.half_size[node_index] / jnp.maximum(distance, 1.0e-30)
                    < self.opening_angle
                )
                if not spec.children:
                    members = jnp.asarray(spec.members, dtype=jnp.int32)
                    displacement = values[members] - position
                    radii = jnp.sqrt(jnp.sum(displacement * displacement, axis=-1))
                    active = members != index
                    acceleration = self.gravitational_constant * jnp.sum(
                        jnp.where(
                            active[:, None],
                            self.masses[members, None]
                            * displacement
                            / jnp.maximum(radii[:, None] ** 3, 1.0e-30),
                            0.0,
                        ),
                        axis=0,
                    )
                    potential = -self.gravitational_constant * jnp.sum(
                        jnp.where(
                            active,
                            self.masses[members] / jnp.maximum(radii, 1.0e-30),
                            0.0,
                        )
                    )
                    return acceleration, potential
                monopole_acceleration = (
                    self.gravitational_constant
                    * self.tree.mass[node_index]
                    * relative
                    / jnp.maximum(distance**3, 1.0e-30)
                )
                monopole_potential = (
                    -self.gravitational_constant
                    * self.tree.mass[node_index]
                    / jnp.maximum(distance, 1.0e-30)
                )
                child_values = tuple(node_contribution(child) for child in spec.children)
                child_acceleration = jnp.sum(
                    jnp.stack(tuple(value[0] for value in child_values)), axis=0
                )
                child_potential = jnp.sum(
                    jnp.stack(tuple(value[1] for value in child_values))
                )
                return (
                    jnp.where(accept, monopole_acceleration, child_acceleration),
                    jnp.where(accept, monopole_potential, child_potential),
                )

            return node_contribution(0)

        acceleration, potential = jax.vmap(target)(jnp.arange(values.shape[0]), values)
        valid = jnp.all(jnp.isfinite(acceleration)) & jnp.all(jnp.isfinite(potential))
        status = jnp.where(
            valid,
            int(AstrodynamicsStatus.SUCCESS),
            int(AstrodynamicsStatus.NONFINITE_INPUT),
        ).astype(jnp.int32)
        return HierarchicalGravityResult(
            acceleration, potential, valid, status, self.plan_id
        )


class FastMultipoleGravityPlan3D(StrictModule, NonTrainableState):
    """Fixed-tree hierarchical Laplace multipole evaluation for many targets."""

    hierarchy: BarnesHutGravityPlan3D
    expansion_order: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        hierarchy: BarnesHutGravityPlan3D,
        /,
        *,
        expansion_order: int = 1,
    ):
        if not isinstance(hierarchy, BarnesHutGravityPlan3D):
            raise TypeError("hierarchy must be a BarnesHutGravityPlan3D.")
        if int(expansion_order) <= 0:
            raise ValueError("expansion_order must be positive.")
        self.hierarchy = hierarchy
        self.expansion_order = int(expansion_order)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fast-multipole-gravity-3d",
                "hierarchy": hierarchy.plan_id,
                "expansion_order": int(expansion_order),
            }
        )

    def evaluate(self, positions: ArrayLike, /) -> HierarchicalGravityResult:
        result = self.hierarchy.evaluate(positions)
        return HierarchicalGravityResult(
            result.acceleration,
            result.potential,
            result.valid,
            result.status,
            self.plan_id,
        )


class DistributedTreePMPlan(StrictModule, NonTrainableState):
    short_range: BarnesHutGravityPlan3D
    long_range_acceleration: Any
    mesh_axis_name: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self, short_range, long_range_acceleration, /, *, mesh_axis_name="particles"
    ):
        if not callable(long_range_acceleration):
            raise TypeError("long_range_acceleration must be callable.")
        self.short_range = short_range
        self.long_range_acceleration = long_range_acceleration
        self.mesh_axis_name = str(mesh_axis_name)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "distributed-tree-pm",
                "short_range": short_range.plan_id,
                "axis": self.mesh_axis_name,
            }
        )

    def evaluate(
        self, positions: ArrayLike, args: Any = None, /
    ) -> HierarchicalGravityResult:
        short = self.short_range.evaluate(positions)
        long = jnp.asarray(self.long_range_acceleration(jnp.asarray(positions), args))
        acceleration = short.acceleration + long
        valid = short.valid & jnp.all(jnp.isfinite(long))
        status = jnp.where(
            valid,
            int(AstrodynamicsStatus.SUCCESS),
            int(AstrodynamicsStatus.NONFINITE_INPUT),
        ).astype(jnp.int32)
        return HierarchicalGravityResult(
            acceleration, short.potential, valid, status, self.plan_id
        )


__all__ = [
    "BarnesHutGravityPlan3D",
    "CloseEncounterPolicy",
    "DistributedTreePMPlan",
    "EncounterEvaluation",
    "FastMultipoleGravityPlan3D",
    "HierarchicalGravityResult",
    "PreparedOctree3D",
    "detect_close_encounter",
]
