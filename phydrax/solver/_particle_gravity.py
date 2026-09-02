#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from itertools import product
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.spatial import (
    MortonAddressPlan,
    MortonPointHierarchyPlan,
    MortonPointHierarchyState,
    SparseLevelOctreePlan,
)


class NewtonianPairKernel(StrictModule, NonTrainableState):
    gravitational_constant: float = eqx.field(static=True)
    softening: float = eqx.field(static=True)
    cutoff: float | None = eqx.field(static=True)
    kernel_id: str = eqx.field(static=True)

    def __init__(
        self,
        gravitational_constant: float,
        /,
        *,
        softening: float,
        cutoff: float | None = None,
    ):
        gravity = float(gravitational_constant)
        epsilon = float(softening)
        cutoff_ = None if cutoff is None else float(cutoff)
        if (
            not np.isfinite(gravity)
            or gravity <= 0.0
            or not np.isfinite(epsilon)
            or epsilon <= 0.0
            or (cutoff_ is not None and (not np.isfinite(cutoff_) or cutoff_ <= 0.0))
        ):
            raise ValueError("Newtonian pair kernel is invalid.")
        self.gravitational_constant = gravity
        self.softening = epsilon
        self.cutoff = cutoff_
        self.kernel_id = canonical_fingerprint(
            {
                "kind": "newtonian-pair-kernel",
                "gravitational_constant": gravity,
                "softening": epsilon,
                "cutoff": cutoff_,
            }
        )

    def acceleration(
        self,
        target_positions: ArrayLike,
        source_positions: ArrayLike,
        source_masses: ArrayLike,
        /,
        *,
        exclude_diagonal: bool = False,
    ) -> Array:
        targets = jnp.asarray(target_positions)
        sources = jnp.asarray(source_positions, dtype=targets.dtype)
        masses = jnp.asarray(source_masses, dtype=targets.dtype)
        if (
            targets.ndim != 2
            or sources.ndim != 2
            or targets.shape[1] != sources.shape[1]
            or masses.shape != (sources.shape[0],)
        ):
            raise ValueError("Pair-kernel source/target shapes are invalid.")
        displacement = sources[None, :, :] - targets[:, None, :]
        radius_squared = jnp.sum(displacement**2, axis=-1) + self.softening**2
        mask = jnp.ones(radius_squared.shape, dtype=bool)
        if exclude_diagonal:
            if targets.shape[0] != sources.shape[0]:
                raise ValueError(
                    "Diagonal exclusion requires equal source/target counts."
                )
            mask = mask & ~jnp.eye(targets.shape[0], dtype=bool)
        if self.cutoff is not None:
            mask = mask & (radius_squared <= self.cutoff**2 + self.softening**2)
        contribution = (
            self.gravitational_constant
            * masses[None, :, None]
            * displacement
            / radius_squared[..., None] ** 1.5
        )
        return jnp.sum(jnp.where(mask[..., None], contribution, 0.0), axis=1)


class ParticleGravityEvidence(StrictModule):
    net_force: Array
    maximum_acceleration: Array
    interaction_count: Array
    approximation_error: Array
    finite: Array
    successful: Array


class DirectParticleGravityPlan(StrictModule, NonTrainableState):
    kernel: NewtonianPairKernel

    def __init__(self, kernel: NewtonianPairKernel, /):
        self.kernel = kernel

    def evaluate(
        self,
        positions: ArrayLike,
        masses: ArrayLike,
        active_mask: ArrayLike | None = None,
        /,
    ) -> tuple[Array, ParticleGravityEvidence]:
        position = jnp.asarray(positions)
        mass = jnp.asarray(masses, dtype=position.dtype)
        active = (
            jnp.ones((position.shape[0],), dtype=bool)
            if active_mask is None
            else jnp.asarray(active_mask, dtype=bool)
        )
        acceleration = self.kernel.acceleration(
            position,
            position,
            jnp.where(active, mass, 0.0),
            exclude_diagonal=True,
        )
        acceleration = jnp.where(active[:, None], acceleration, 0.0)
        finite = jnp.all(jnp.isfinite(acceleration))
        evidence = ParticleGravityEvidence(
            jnp.sum(mass[:, None] * acceleration, axis=0),
            jnp.max(jnp.sqrt(jnp.sum(acceleration**2, axis=-1))),
            jnp.sum(active) * jnp.maximum(jnp.sum(active) - 1, 0),
            jnp.asarray(0.0, dtype=position.dtype),
            finite,
            finite,
        )
        return acceleration, evidence


class DistributedParticleLayout(StrictModule, NonTrainableState):
    device_count: int = eqx.field(static=True)
    capacity_per_device: int = eqx.field(static=True)
    key_boundaries: Array
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        device_count: int,
        capacity_per_device: int,
        key_boundaries: ArrayLike,
        /,
    ):
        devices = int(device_count)
        capacity = int(capacity_per_device)
        boundaries = jnp.asarray(key_boundaries, dtype=jnp.uint32)
        if devices <= 0 or capacity <= 0 or boundaries.shape != (devices + 1,):
            raise ValueError("Distributed particle layout is invalid.")
        boundaries = eqx.error_if(
            boundaries,
            jnp.any(boundaries[1:] < boundaries[:-1]),
            "Distributed Morton boundaries must be increasing.",
        )
        self.device_count = devices
        self.capacity_per_device = capacity
        self.key_boundaries = boundaries
        self.layout_id = canonical_fingerprint(
            {
                "kind": "distributed-particle-layout",
                "device_count": devices,
                "capacity_per_device": capacity,
                "boundaries": np.asarray(boundaries).tolist(),
            }
        )

    def owners(self, morton_keys: ArrayLike, /) -> Array:
        keys = jnp.asarray(morton_keys, dtype=jnp.uint32)
        return jnp.clip(
            jnp.searchsorted(self.key_boundaries[1:], keys, side="right"),
            0,
            self.device_count - 1,
        )


class PreparedParticleOctree3D(StrictModule):
    """Particle payloads, sparse octree topology, and node multipoles."""

    positions: Array
    masses: Array
    active_mask: Array
    morton_keys: Array
    permutation: Array
    leaf_indices: Array
    leaf_mass: Array
    leaf_center_of_mass: Array
    leaf_quadrupole: Array
    leaf_centers: Array
    leaf_half_size: Array
    hierarchy: MortonPointHierarchyState
    box_size: tuple[float, float, float] = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    target_leaf_occupancy: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class ParticleOctreePlan3D(StrictModule, NonTrainableState):
    """Prepare a sparse occupied octree without a dense finest-level lattice."""

    address_plan: MortonAddressPlan
    box_size: tuple[float, float, float] = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    leaf_count: int = eqx.field(static=True)
    target_leaf_occupancy: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        box_size: tuple[float, float, float],
        depth: int,
        /,
        *,
        target_leaf_occupancy: int = 4,
    ):
        lengths = tuple(float(value) for value in box_size)
        depth_ = int(depth)
        target = int(target_leaf_occupancy)
        if len(lengths) != 3 or any(
            not np.isfinite(value) or value <= 0.0 for value in lengths
        ):
            raise ValueError("Particle octree requires a finite positive 3-D box.")
        if depth_ < 1 or depth_ > 10:
            raise ValueError("Particle octree depth must lie in [1,10].")
        if target < 1:
            raise ValueError("target_leaf_occupancy must be positive.")
        self.address_plan = MortonAddressPlan((0.0, 0.0, 0.0), lengths, depth_)
        self.box_size = lengths
        self.depth = depth_
        self.leaf_count = (1 << depth_) ** 3
        self.target_leaf_occupancy = target
        self.plan_id = canonical_fingerprint(
            {
                "kind": "sparse-particle-octree-3d",
                "box_size": list(lengths),
                "depth": depth_,
                "target_leaf_occupancy": target,
            }
        )

    def prepare(
        self,
        positions: ArrayLike,
        masses: ArrayLike,
        active_mask: ArrayLike | None = None,
        /,
    ) -> PreparedParticleOctree3D:
        position = jnp.asarray(positions)
        mass = jnp.asarray(masses, dtype=position.dtype)
        active = (
            jnp.ones((position.shape[0],), dtype=bool)
            if active_mask is None
            else jnp.asarray(active_mask, dtype=bool)
        )
        if (
            position.ndim != 2
            or position.shape[1] != 3
            or mass.shape != active.shape
            or mass.shape != (position.shape[0],)
        ):
            raise ValueError(
                "Particle octree positions, masses, and active mask disagree."
            )
        invalid_payload = (
            jnp.any(jnp.where(active[:, None], ~jnp.isfinite(position), False))
            | jnp.any(jnp.where(active, ~jnp.isfinite(mass), False))
            | jnp.any(jnp.where(active, mass <= 0.0, False))
        )
        safe_position = jnp.where(active[:, None], position, 0.0)
        hierarchy_plan = MortonPointHierarchyPlan(
            self.address_plan,
            position.shape[0],
            node_capacity=(self.depth + 1) * position.shape[0],
            target_leaf_occupancy=self.target_leaf_occupancy,
        )
        hierarchy = hierarchy_plan.build(
            safe_position,
            active_mask=active,
            stable_ids=jnp.arange(position.shape[0], dtype=jnp.int64),
        )
        position = eqx.error_if(
            safe_position,
            invalid_payload | ~hierarchy.evidence.successful,
            "Active octree particles must be finite, positive-mass, and inside the box.",
        )
        sorted_position = position[hierarchy.storage_to_logical]
        sorted_mass = mass[hierarchy.storage_to_logical]
        sorted_active = hierarchy.sorted_active
        node_capacity = hierarchy.node_active.size
        point_capacity = position.shape[0]
        node_slots = jnp.arange(node_capacity, dtype=jnp.int32)
        leaf_slots = jnp.nonzero(
            hierarchy.node_is_leaf,
            size=node_capacity,
            fill_value=node_capacity,
        )[0].astype(jnp.int32)
        leaf_valid = node_slots < hierarchy.evidence.active_leaves
        safe_leaf_slots = jnp.minimum(leaf_slots, node_capacity - 1)
        leaf_starts = jnp.where(
            leaf_valid,
            hierarchy.node_item_starts[safe_leaf_slots],
            point_capacity,
        )
        leaf_order = jnp.argsort(leaf_starts, stable=True)
        ordered_leaf_slots = leaf_slots[leaf_order]
        ordered_leaf_starts = leaf_starts[leaf_order]
        storage_slots = jnp.arange(point_capacity, dtype=jnp.int32)
        leaf_rank = jnp.searchsorted(ordered_leaf_starts, storage_slots, side="right") - 1
        safe_leaf_rank = jnp.maximum(leaf_rank, 0)
        sorted_leaf_indices = jnp.where(
            sorted_active,
            ordered_leaf_slots[safe_leaf_rank],
            -1,
        ).astype(jnp.int32)
        safe_point_leaf = jnp.maximum(sorted_leaf_indices, 0)
        safe_mass = jnp.where(sorted_active, sorted_mass, 0.0)
        node_mass = (
            jnp.zeros((node_capacity,), dtype=mass.dtype)
            .at[safe_point_leaf]
            .add(safe_mass)
        )
        weighted_position = (
            jnp.zeros((node_capacity, 3), dtype=position.dtype)
            .at[safe_point_leaf]
            .add(safe_mass[:, None] * sorted_position)
        )
        safe_node_mass = jnp.where(node_mass > 0.0, node_mass, 1.0)
        node_center = weighted_position / safe_node_mass[:, None]
        centered = sorted_position - node_center[safe_point_leaf]
        outer = centered[:, :, None] * centered[:, None, :]
        radius_squared = jnp.sum(centered**2, axis=-1)
        particle_quadrupole = safe_mass[:, None, None] * (
            3.0 * outer - radius_squared[:, None, None] * jnp.eye(3, dtype=position.dtype)
        )
        node_quadrupole = (
            jnp.zeros((node_capacity, 3, 3), dtype=position.dtype)
            .at[safe_point_leaf]
            .add(particle_quadrupole)
        )
        identity = jnp.eye(3, dtype=position.dtype)
        for level in range(self.depth - 1, -1, -1):
            internal = (
                hierarchy.node_active
                & ~hierarchy.node_is_leaf
                & (hierarchy.node_levels == level)
            )
            children = hierarchy.node_children
            child_valid = children >= 0
            safe_children = jnp.maximum(children, 0)
            child_mass = jnp.where(child_valid, node_mass[safe_children], 0.0)
            parent_mass = jnp.sum(child_mass, axis=1)
            child_center = node_center[safe_children]
            parent_center = (
                jnp.sum(child_mass[..., None] * child_center, axis=1)
                / jnp.where(parent_mass > 0.0, parent_mass, 1.0)[:, None]
            )
            displacement = child_center - parent_center[:, None, :]
            displacement_outer = displacement[..., :, None] * displacement[..., None, :]
            displacement_squared = jnp.sum(displacement**2, axis=-1)
            translation = child_mass[..., None, None] * (
                3.0 * displacement_outer
                - displacement_squared[..., None, None] * identity
            )
            child_quadrupole = jnp.where(
                child_valid[..., None, None],
                node_quadrupole[safe_children],
                0.0,
            )
            parent_quadrupole = jnp.sum(child_quadrupole + translation, axis=1)
            node_mass = jnp.where(internal, parent_mass, node_mass)
            node_center = jnp.where(internal[:, None], parent_center, node_center)
            node_quadrupole = jnp.where(
                internal[:, None, None], parent_quadrupole, node_quadrupole
            )
        leaf_indices = (
            jnp.full((point_capacity,), -1, dtype=jnp.int32)
            .at[hierarchy.storage_to_logical]
            .set(sorted_leaf_indices)
        )
        morton_keys = hierarchy.sorted_codes[hierarchy.logical_to_storage]
        return PreparedParticleOctree3D(
            positions=position,
            masses=mass,
            active_mask=active,
            morton_keys=morton_keys,
            permutation=hierarchy.storage_to_logical,
            leaf_indices=leaf_indices,
            leaf_mass=node_mass,
            leaf_center_of_mass=node_center,
            leaf_quadrupole=node_quadrupole,
            leaf_centers=hierarchy.node_centers.astype(position.dtype),
            leaf_half_size=hierarchy.node_half_widths.astype(position.dtype),
            hierarchy=hierarchy,
            box_size=self.box_size,
            depth=self.depth,
            target_leaf_occupancy=self.target_leaf_occupancy,
            prepared_id=canonical_fingerprint(
                {
                    "kind": "prepared-sparse-particle-octree",
                    "plan": self.plan_id,
                    "capacity": position.shape[0],
                }
            ),
        )


class TreeGravityEvidence(StrictModule):
    net_force: Array
    maximum_acceleration: Array
    accepted_leaf_interactions: Array
    direct_particle_interactions: Array
    maximum_opening_indicator: Array
    traversal_complete: Array
    active_nodes: Array
    finite: Array
    successful: Array


class TreeGravityResult(StrictModule):
    acceleration: Array
    evidence: TreeGravityEvidence
    successful: Array


class BarnesHutGravityPlan(StrictModule, NonTrainableState):
    gravitational_constant: float = eqx.field(static=True)
    softening: float = eqx.field(static=True)
    opening_angle: float = eqx.field(static=True)
    use_quadrupole: bool = eqx.field(static=True)
    direct_chunk_size: int = eqx.field(static=True)
    target_batch_size: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        gravitational_constant: float,
        /,
        *,
        softening: float,
        opening_angle: float = 0.5,
        use_quadrupole: bool = True,
        direct_chunk_size: int = 32,
        target_batch_size: int = 32,
    ):
        gravity = float(gravitational_constant)
        epsilon = float(softening)
        theta = float(opening_angle)
        chunk = int(direct_chunk_size)
        target_batch = int(target_batch_size)
        if (
            not np.isfinite(gravity)
            or gravity <= 0.0
            or not np.isfinite(epsilon)
            or epsilon <= 0.0
            or not np.isfinite(theta)
            or theta < 0.0
            or theta >= 1.0
            or chunk <= 0
            or target_batch <= 0
        ):
            raise ValueError("Barnes-Hut policy is invalid.")
        self.gravitational_constant = gravity
        self.softening = epsilon
        self.opening_angle = theta
        self.use_quadrupole = bool(use_quadrupole)
        self.direct_chunk_size = chunk
        self.target_batch_size = target_batch
        self.plan_id = canonical_fingerprint(
            {
                "kind": "barnes-hut-gravity",
                "gravitational_constant": gravity,
                "softening": epsilon,
                "opening_angle": theta,
                "use_quadrupole": bool(use_quadrupole),
                "direct_chunk_size": chunk,
                "target_batch_size": target_batch,
            }
        )

    def _evaluate_branchless(
        self,
        tree: PreparedParticleOctree3D,
        /,
        *,
        short_range_scale: float | None,
        cutoff: float | None,
    ) -> TreeGravityResult:
        position = tree.positions
        hierarchy = tree.hierarchy
        point_capacity = position.shape[0]
        node_capacity = hierarchy.node_active.size
        sorted_logical = hierarchy.storage_to_logical
        sorted_position = position[sorted_logical]
        sorted_mass = tree.masses[sorted_logical]
        sorted_active = hierarchy.sorted_active
        sorted_leaf = tree.leaf_indices[sorted_logical]
        scale = (
            None
            if short_range_scale is None
            else jnp.asarray(short_range_scale, dtype=position.dtype)
        )
        cutoff_value = (
            None if cutoff is None else jnp.asarray(cutoff, dtype=position.dtype)
        )

        def radial_kernel(distance_squared, distance):
            kernel = distance_squared ** (-1.5)
            if scale is not None:
                argument = distance / (2.0 * scale)
                kernel = kernel * (
                    jax.scipy.special.erfc(argument)
                    + distance / (scale * jnp.sqrt(jnp.pi)) * jnp.exp(-(argument**2))
                )
            return kernel

        def evaluate_target(inputs):
            target_position, target_storage, target_active = inputs
            node_displacement = tree.leaf_center_of_mass - target_position
            node_distance_squared = (
                jnp.sum(node_displacement**2, axis=-1) + self.softening**2
            )
            node_distance = jnp.sqrt(node_distance_squared)
            node_radius = jnp.sqrt(jnp.sum(hierarchy.node_half_widths**2, axis=-1))
            node_size = 2.0 * jnp.max(hierarchy.node_half_widths, axis=-1)
            opening_indicator = node_size / node_distance
            contains_target = (target_storage >= hierarchy.node_item_starts) & (
                target_storage < hierarchy.node_item_starts + hierarchy.node_item_counts
            )
            node_valid = hierarchy.node_active & (tree.leaf_mass > 0.0)
            outside_cutoff = jnp.zeros((node_capacity,), dtype=bool)
            fully_inside_cutoff = jnp.ones((node_capacity,), dtype=bool)
            if cutoff_value is not None:
                outside_cutoff = node_distance - node_radius > cutoff_value
                fully_inside_cutoff = node_distance + node_radius <= cutoff_value
            accept = (
                node_valid
                & ~hierarchy.node_is_leaf
                & ~contains_target
                & (opening_indicator < self.opening_angle)
                & fully_inside_cutoff
            )
            terminal = accept | (node_valid & outside_cutoff)

            def propagate_blocked(node, blocked):
                parent = hierarchy.node_parents[node]
                safe_parent = jnp.maximum(parent, 0)
                value = (parent >= 0) & (blocked[safe_parent] | terminal[safe_parent])
                return blocked.at[node].set(value)

            blocked = jax.lax.fori_loop(
                0,
                node_capacity,
                propagate_blocked,
                jnp.zeros((node_capacity,), dtype=bool),
            )
            selected_far = target_active & accept & ~blocked
            selected_leaf = (
                target_active
                & node_valid
                & hierarchy.node_is_leaf
                & ~outside_cutoff
                & ~blocked
            )
            far_kernel = radial_kernel(node_distance_squared, node_distance)
            far_contribution = (
                self.gravitational_constant
                * tree.leaf_mass[:, None]
                * node_displacement
                * far_kernel[:, None]
            )
            if self.use_quadrupole:
                q_r = contract(
                    "nij,nj->ni",
                    tree.leaf_quadrupole,
                    node_displacement,
                )
                r_q_r = jnp.sum(node_displacement * q_r, axis=-1)
                far_contribution = far_contribution + self.gravitational_constant * (
                    2.5 * r_q_r[:, None] * node_displacement / node_distance[:, None] ** 7
                    - q_r / node_distance[:, None] ** 5
                )
            far_acceleration = jnp.sum(
                jnp.where(selected_far[:, None], far_contribution, 0.0),
                axis=0,
            )
            safe_sorted_leaf = jnp.maximum(sorted_leaf, 0)
            direct_mask = (
                target_active
                & sorted_active
                & selected_leaf[safe_sorted_leaf]
                & (jnp.arange(point_capacity, dtype=jnp.int32) != target_storage)
            )
            source_displacement = sorted_position - target_position
            source_distance_squared = (
                jnp.sum(source_displacement**2, axis=-1) + self.softening**2
            )
            source_distance = jnp.sqrt(source_distance_squared)
            if cutoff_value is not None:
                direct_mask = direct_mask & (source_distance <= cutoff_value)
            direct_contribution = (
                self.gravitational_constant
                * sorted_mass[:, None]
                * source_displacement
                * radial_kernel(source_distance_squared, source_distance)[:, None]
            )
            direct_acceleration = jnp.sum(
                jnp.where(direct_mask[:, None], direct_contribution, 0.0),
                axis=0,
            )
            return (
                far_acceleration + direct_acceleration,
                jnp.sum(selected_far, dtype=jnp.int32),
                jnp.sum(direct_mask, dtype=jnp.int32),
                jnp.max(
                    jnp.where(selected_far, opening_indicator, 0.0),
                    initial=0.0,
                ),
                hierarchy.evidence.successful,
            )

        (
            sorted_acceleration,
            accepted,
            direct,
            indicator,
            complete,
        ) = jax.lax.map(
            evaluate_target,
            (
                sorted_position,
                jnp.arange(point_capacity, dtype=jnp.int32),
                sorted_active,
            ),
            batch_size=min(self.target_batch_size, point_capacity),
        )
        acceleration = (
            jnp.zeros_like(position).at[sorted_logical].set(sorted_acceleration)
        )
        acceleration = jnp.where(tree.active_mask[:, None], acceleration, 0.0)
        finite = jnp.all(jnp.isfinite(acceleration))
        traversal_complete = jnp.all(complete)
        successful = hierarchy.evidence.successful & traversal_complete & finite
        evidence = TreeGravityEvidence(
            net_force=jnp.sum(
                jnp.where(tree.active_mask, tree.masses, 0.0)[:, None] * acceleration,
                axis=0,
            ),
            maximum_acceleration=jnp.max(
                jnp.sqrt(jnp.sum(acceleration**2, axis=-1)),
                initial=0.0,
            ),
            accepted_leaf_interactions=jnp.sum(accepted, dtype=jnp.int32),
            direct_particle_interactions=jnp.sum(direct, dtype=jnp.int32),
            maximum_opening_indicator=jnp.max(indicator, initial=0.0),
            traversal_complete=traversal_complete,
            active_nodes=hierarchy.evidence.active_nodes,
            finite=finite,
            successful=successful,
        )
        return TreeGravityResult(acceleration, evidence, successful)

    def _evaluate_impl(
        self,
        tree: PreparedParticleOctree3D,
        /,
        *,
        short_range_scale: float | None,
        cutoff: float | None,
        fixed_iterations: bool,
    ) -> TreeGravityResult:
        if tree.positions.shape[0] <= 4096:
            return self._evaluate_branchless(
                tree,
                short_range_scale=short_range_scale,
                cutoff=cutoff,
            )
        position = tree.positions
        hierarchy = tree.hierarchy
        point_capacity = position.shape[0]
        node_capacity = hierarchy.node_active.size
        stack_capacity = 1 + 7 * tree.depth
        sorted_logical = hierarchy.storage_to_logical
        sorted_position = position[sorted_logical]
        sorted_mass = tree.masses[sorted_logical]
        sorted_active = hierarchy.sorted_active
        chunk_offsets = jnp.arange(self.direct_chunk_size, dtype=jnp.int32)
        scale = (
            None
            if short_range_scale is None
            else jnp.asarray(short_range_scale, dtype=position.dtype)
        )
        cutoff_value = (
            None if cutoff is None else jnp.asarray(cutoff, dtype=position.dtype)
        )

        def radial_kernel(distance_squared, distance):
            kernel = distance_squared ** (-1.5)
            if scale is not None:
                argument = distance / (2.0 * scale)
                kernel = kernel * (
                    jax.scipy.special.erfc(argument)
                    + distance / (scale * jnp.sqrt(jnp.pi)) * jnp.exp(-(argument**2))
                )
            return kernel

        def evaluate_target(inputs):
            target_position, target_storage, target_active = inputs
            stack = jnp.zeros((stack_capacity,), dtype=jnp.int32)
            has_root = target_active & (hierarchy.root_slot >= 0)
            stack = stack.at[0].set(jnp.maximum(hierarchy.root_slot, 0))
            initial = (
                stack,
                has_root.astype(jnp.int32),
                jnp.zeros((3,), dtype=position.dtype),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0.0, dtype=position.dtype),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(False),
            )

            def traversal_body(state):
                (
                    current_stack,
                    top,
                    acceleration,
                    accepted_count,
                    direct_count,
                    maximum_indicator,
                    visits,
                    overflow,
                ) = state
                next_top = top - 1
                node = current_stack[next_top]
                node_mass = tree.leaf_mass[node]
                displacement = tree.leaf_center_of_mass[node] - target_position
                distance_squared = jnp.sum(displacement**2) + self.softening**2
                distance = jnp.sqrt(distance_squared)
                node_radius = jnp.sqrt(jnp.sum(hierarchy.node_half_widths[node] ** 2))
                node_size = 2.0 * jnp.max(hierarchy.node_half_widths[node])
                opening_indicator = node_size / distance
                contains_target = (target_storage >= hierarchy.node_item_starts[node]) & (
                    target_storage
                    < hierarchy.node_item_starts[node] + hierarchy.node_item_counts[node]
                )
                node_valid = hierarchy.node_active[node] & (node_mass > 0.0)
                outside_cutoff = jnp.asarray(False)
                fully_inside_cutoff = jnp.asarray(True)
                if cutoff_value is not None:
                    outside_cutoff = distance - node_radius > cutoff_value
                    fully_inside_cutoff = distance + node_radius <= cutoff_value
                accept = (
                    node_valid
                    & ~hierarchy.node_is_leaf[node]
                    & ~contains_target
                    & (opening_indicator < self.opening_angle)
                    & fully_inside_cutoff
                )
                far_kernel = radial_kernel(distance_squared, distance)
                far_contribution = (
                    self.gravitational_constant * node_mass * displacement * far_kernel
                )
                if self.use_quadrupole:
                    q_r = tree.leaf_quadrupole[node] @ displacement
                    r_q_r = jnp.sum(displacement * q_r)
                    far_contribution = far_contribution + self.gravitational_constant * (
                        2.5 * r_q_r * displacement / distance**7 - q_r / distance**5
                    )
                acceleration = acceleration + jnp.where(accept, far_contribution, 0.0)
                accepted_count = accepted_count + accept.astype(jnp.int32)
                maximum_indicator = jnp.maximum(
                    maximum_indicator,
                    jnp.where(accept, opening_indicator, 0.0),
                )

                evaluate_leaf = (
                    node_valid & hierarchy.node_is_leaf[node] & ~outside_cutoff
                )

                def direct_leaf(direct_state):
                    offset, direct_acceleration, interactions = direct_state
                    source_storage = (
                        hierarchy.node_item_starts[node] + offset + chunk_offsets
                    )
                    source_in_leaf = (
                        source_storage
                        < hierarchy.node_item_starts[node]
                        + hierarchy.node_item_counts[node]
                    ) & (source_storage < point_capacity)
                    safe_storage = jnp.minimum(source_storage, point_capacity - 1)
                    source_displacement = sorted_position[safe_storage] - target_position
                    source_distance_squared = (
                        jnp.sum(source_displacement**2, axis=-1) + self.softening**2
                    )
                    source_distance = jnp.sqrt(source_distance_squared)
                    source_valid = (
                        source_in_leaf
                        & sorted_active[safe_storage]
                        & (source_storage != target_storage)
                    )
                    if cutoff_value is not None:
                        source_valid = source_valid & (source_distance <= cutoff_value)
                    contribution = (
                        self.gravitational_constant
                        * sorted_mass[safe_storage, None]
                        * source_displacement
                        * radial_kernel(source_distance_squared, source_distance)[:, None]
                    )
                    direct_acceleration = direct_acceleration + jnp.sum(
                        jnp.where(source_valid[:, None], contribution, 0.0),
                        axis=0,
                    )
                    return (
                        offset + self.direct_chunk_size,
                        direct_acceleration,
                        interactions + jnp.sum(source_valid, dtype=jnp.int32),
                    )

                direct_initial = (
                    jnp.asarray(0, dtype=jnp.int32),
                    jnp.zeros((3,), dtype=position.dtype),
                    jnp.asarray(0, dtype=jnp.int32),
                )

                def evaluate_direct_leaf(_):
                    if not fixed_iterations:
                        _, leaf_acceleration, leaf_interactions = jax.lax.while_loop(
                            lambda direct_state: (
                                direct_state[0] < hierarchy.node_item_counts[node]
                            ),
                            direct_leaf,
                            direct_initial,
                        )
                        return leaf_acceleration, leaf_interactions

                    def run_chunks(chunk_count):
                        _, leaf_acceleration, leaf_interactions = jax.lax.fori_loop(
                            0,
                            chunk_count,
                            lambda _, direct_state: direct_leaf(direct_state),
                            direct_initial,
                        )
                        return leaf_acceleration, leaf_interactions

                    return jax.lax.cond(
                        hierarchy.node_item_counts[node] > tree.target_leaf_occupancy,
                        lambda: run_chunks(
                            (point_capacity + self.direct_chunk_size - 1)
                            // self.direct_chunk_size
                        ),
                        lambda: run_chunks(
                            (tree.target_leaf_occupancy + self.direct_chunk_size - 1)
                            // self.direct_chunk_size
                        ),
                    )

                leaf_acceleration, leaf_interactions = jax.lax.cond(
                    evaluate_leaf,
                    evaluate_direct_leaf,
                    lambda _: (
                        jnp.zeros((3,), dtype=position.dtype),
                        jnp.asarray(0, dtype=jnp.int32),
                    ),
                    operand=None,
                )
                acceleration = acceleration + leaf_acceleration
                direct_count = direct_count + leaf_interactions

                descend = (
                    node_valid & ~hierarchy.node_is_leaf[node] & ~accept & ~outside_cutoff
                )
                children = hierarchy.node_children[node]
                for child_index in range(children.shape[0]):
                    child = children[child_index]
                    push = descend & (child >= 0)
                    has_capacity = next_top < stack_capacity
                    write = push & has_capacity
                    safe_top = jnp.minimum(next_top, stack_capacity - 1)
                    current_stack = current_stack.at[safe_top].set(
                        jnp.where(write, child, current_stack[safe_top])
                    )
                    next_top = next_top + write.astype(jnp.int32)
                    overflow = overflow | (push & ~has_capacity)
                return (
                    current_stack,
                    next_top,
                    acceleration,
                    accepted_count,
                    direct_count,
                    maximum_indicator,
                    visits + 1,
                    overflow,
                )

            if fixed_iterations:

                def traversal_iteration(_, state):
                    active_step = (state[1] > 0) & ~state[7]
                    return jax.lax.cond(
                        active_step,
                        traversal_body,
                        lambda current: current,
                        state,
                    )

                final = jax.lax.fori_loop(
                    0,
                    node_capacity,
                    traversal_iteration,
                    initial,
                )
            else:
                final = jax.lax.while_loop(
                    lambda state: (state[1] > 0) & (state[6] < node_capacity) & ~state[7],
                    traversal_body,
                    initial,
                )
            _, remaining, acceleration, accepted, direct, indicator, visits, overflow = (
                final
            )
            complete = (remaining == 0) & ~overflow & (visits <= node_capacity)
            return acceleration, accepted, direct, indicator, complete

        (
            sorted_acceleration,
            accepted,
            direct,
            indicator,
            complete,
        ) = jax.lax.map(
            evaluate_target,
            (
                sorted_position,
                jnp.arange(point_capacity, dtype=jnp.int32),
                sorted_active,
            ),
            batch_size=min(self.target_batch_size, point_capacity),
        )
        acceleration = (
            jnp.zeros_like(position).at[sorted_logical].set(sorted_acceleration)
        )
        acceleration = jnp.where(tree.active_mask[:, None], acceleration, 0.0)
        finite = jnp.all(jnp.isfinite(acceleration))
        traversal_complete = jnp.all(complete)
        successful = hierarchy.evidence.successful & traversal_complete & finite
        evidence = TreeGravityEvidence(
            net_force=jnp.sum(
                jnp.where(tree.active_mask, tree.masses, 0.0)[:, None] * acceleration,
                axis=0,
            ),
            maximum_acceleration=jnp.max(
                jnp.sqrt(jnp.sum(acceleration**2, axis=-1)),
                initial=0.0,
            ),
            accepted_leaf_interactions=jnp.sum(accepted, dtype=jnp.int32),
            direct_particle_interactions=jnp.sum(direct, dtype=jnp.int32),
            maximum_opening_indicator=jnp.max(indicator, initial=0.0),
            traversal_complete=traversal_complete,
            active_nodes=hierarchy.evidence.active_nodes,
            finite=finite,
            successful=successful,
        )
        return TreeGravityResult(acceleration, evidence, successful)

    def evaluate(
        self,
        tree: PreparedParticleOctree3D,
        /,
        *,
        short_range_scale: float | None = None,
        cutoff: float | None = None,
    ) -> TreeGravityResult:
        plan = self

        @jax.custom_vjp
        def run(current_tree):
            return plan._evaluate_impl(
                current_tree,
                short_range_scale=short_range_scale,
                cutoff=cutoff,
                fixed_iterations=False,
            )

        def forward(current_tree):
            result = plan._evaluate_impl(
                current_tree,
                short_range_scale=short_range_scale,
                cutoff=cutoff,
                fixed_iterations=False,
            )
            return result, current_tree

        def backward(current_tree, cotangent):
            _, pullback = jax.vjp(
                lambda value: plan._evaluate_impl(
                    value,
                    short_range_scale=short_range_scale,
                    cutoff=cutoff,
                    fixed_iterations=True,
                ),
                current_tree,
            )
            return (pullback(cotangent)[0],)

        run.defvjp(forward, backward)
        return run(tree)


class CartesianExpansionSpace(StrictModule, NonTrainableState):
    order: int = eqx.field(static=True)
    exponents: tuple[tuple[int, int, int], ...] = eqx.field(static=True)
    coefficient_count: int = eqx.field(static=True)

    def __init__(self, order: int, /):
        order_ = int(order)
        if order_ < 1 or order_ > 6:
            raise ValueError("Cartesian FMM order must lie in [1,6].")
        exponents = tuple(
            (i, j, k)
            for total in range(order_ + 1)
            for i in range(total + 1)
            for j in range(total - i + 1)
            for k in (total - i - j,)
        )
        self.order = order_
        self.exponents = exponents
        self.coefficient_count = len(exponents)


class CartesianFMMOperators(StrictModule, NonTrainableState):
    """First-order Cartesian P2M/M2M/M2L/L2L/L2P/P2P operators."""

    expansion: CartesianExpansionSpace
    gravitational_constant: float = eqx.field(static=True)
    softening: float = eqx.field(static=True)

    def __init__(
        self,
        expansion: CartesianExpansionSpace,
        gravitational_constant: float,
        softening: float,
        /,
    ):
        if expansion.order != 1:
            raise ValueError(
                "Current qualified Cartesian FMM operators require order one."
            )
        gravity = float(gravitational_constant)
        epsilon = float(softening)
        if (
            not np.isfinite(gravity)
            or gravity <= 0.0
            or not np.isfinite(epsilon)
            or epsilon <= 0.0
        ):
            raise ValueError("Cartesian FMM operator constants are invalid.")
        self.expansion = expansion
        self.gravitational_constant = gravity
        self.softening = epsilon

    def p2m(
        self,
        positions: ArrayLike,
        masses: ArrayLike,
        center: ArrayLike,
        /,
    ) -> Array:
        position = jnp.asarray(positions)
        mass = jnp.asarray(masses, dtype=position.dtype)
        center_ = jnp.asarray(center, dtype=position.dtype)
        relative = position - center_
        coefficients = []
        for i, j, k in self.expansion.exponents:
            coefficients.append(
                jnp.sum(
                    mass * relative[:, 0] ** i * relative[:, 1] ** j * relative[:, 2] ** k
                )
            )
        return jnp.stack(coefficients)

    def m2m(self, coefficients: ArrayLike, shift: ArrayLike, /) -> Array:
        values = jnp.asarray(coefficients)
        shift_ = jnp.asarray(shift, dtype=values.dtype)
        output = []
        for alpha in self.expansion.exponents:
            total = jnp.asarray(0.0, dtype=values.dtype)
            for index, beta in enumerate(self.expansion.exponents):
                if all(beta[axis] <= alpha[axis] for axis in range(3)):
                    multiplier = 1.0
                    for axis in range(3):
                        if alpha[axis] - beta[axis] == 1:
                            multiplier = multiplier * shift_[axis]
                    total = total + multiplier * values[index]
            output.append(total)
        return jnp.stack(output)

    def m2l(
        self,
        multipole: ArrayLike,
        source_center: ArrayLike,
        target_center: ArrayLike,
        /,
    ) -> Array:
        values = jnp.asarray(multipole)
        source = jnp.asarray(source_center, dtype=values.dtype)
        target = jnp.asarray(target_center, dtype=values.dtype)
        displacement = target - source
        radius_squared = jnp.sum(displacement**2) + self.softening**2
        radius = jnp.sqrt(radius_squared)
        mass_index = self.expansion.exponents.index((0, 0, 0))
        dipole = jnp.stack(
            tuple(
                values[self.expansion.exponents.index(exponent)]
                for exponent in ((1, 0, 0), (0, 1, 0), (0, 0, 1))
            )
        )
        mass = values[mass_index]
        potential = -self.gravitational_constant * (
            mass / radius + jnp.dot(dipole, displacement) / radius**3
        )
        gradient = self.gravitational_constant * (
            mass * displacement / radius**3
            + dipole / radius**3
            - 3.0 * jnp.dot(dipole, displacement) * displacement / radius**5
        )
        local = jnp.zeros_like(values).at[mass_index].set(potential)
        for axis, exponent in enumerate(((1, 0, 0), (0, 1, 0), (0, 0, 1))):
            local = local.at[self.expansion.exponents.index(exponent)].set(gradient[axis])
        return local

    def l2l(self, local: ArrayLike, shift: ArrayLike, /) -> Array:
        values = jnp.asarray(local)
        shift_ = jnp.asarray(shift, dtype=values.dtype)
        mass_index = self.expansion.exponents.index((0, 0, 0))
        gradient = jnp.stack(
            tuple(
                values[self.expansion.exponents.index(exponent)]
                for exponent in ((1, 0, 0), (0, 1, 0), (0, 0, 1))
            )
        )
        return values.at[mass_index].set(values[mass_index] + jnp.dot(gradient, shift_))

    def l2p(self, local: ArrayLike, displacement: ArrayLike, /) -> tuple[Array, Array]:
        values = jnp.asarray(local)
        offset = jnp.asarray(displacement, dtype=values.dtype)
        mass_index = self.expansion.exponents.index((0, 0, 0))
        gradient = jnp.stack(
            tuple(
                values[self.expansion.exponents.index(exponent)]
                for exponent in ((1, 0, 0), (0, 1, 0), (0, 0, 1))
            )
        )
        return values[mass_index] + jnp.dot(gradient, offset), -gradient

    def p2p(
        self,
        target: ArrayLike,
        source_positions: ArrayLike,
        source_masses: ArrayLike,
        /,
    ) -> Array:
        target_ = jnp.asarray(target)
        source = jnp.asarray(source_positions, dtype=target_.dtype)
        mass = jnp.asarray(source_masses, dtype=target_.dtype)
        displacement = source - target_
        radius_squared = jnp.sum(displacement**2, axis=-1) + self.softening**2
        return jnp.sum(
            self.gravitational_constant
            * mass[:, None]
            * displacement
            / radius_squared[:, None] ** 1.5,
            axis=0,
        )


class UniformFMMPlan(StrictModule, NonTrainableState):
    """Sparse occupied-level first-order Cartesian fast multipole method."""

    gravitational_constant: float = eqx.field(static=True)
    softening: float = eqx.field(static=True)
    expansion: CartesianExpansionSpace
    maximum_far_interactions: int | None = eqx.field(static=True)
    maximum_near_interactions: int | None = eqx.field(static=True)
    direct_chunk_size: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        gravitational_constant: float,
        expansion: CartesianExpansionSpace,
        /,
        *,
        softening: float,
        maximum_far_interactions: int | None = None,
        maximum_near_interactions: int | None = None,
        direct_chunk_size: int = 32,
    ):
        gravity = float(gravitational_constant)
        epsilon = float(softening)
        far_capacity = (
            None if maximum_far_interactions is None else int(maximum_far_interactions)
        )
        near_capacity = (
            None if maximum_near_interactions is None else int(maximum_near_interactions)
        )
        chunk = int(direct_chunk_size)
        if (
            not np.isfinite(gravity)
            or gravity <= 0.0
            or not np.isfinite(epsilon)
            or epsilon <= 0.0
            or expansion.order != 1
            or (far_capacity is not None and far_capacity <= 0)
            or (near_capacity is not None and near_capacity <= 0)
            or chunk <= 0
        ):
            raise ValueError(
                "Sparse Cartesian FMM requires positive constants, order one, "
                "and positive capacities."
            )
        self.gravitational_constant = gravity
        self.softening = epsilon
        self.expansion = expansion
        self.maximum_far_interactions = far_capacity
        self.maximum_near_interactions = near_capacity
        self.direct_chunk_size = chunk
        self.plan_id = canonical_fingerprint(
            {
                "kind": "sparse-cartesian-fmm",
                "gravitational_constant": gravity,
                "softening": epsilon,
                "order": expansion.order,
                "maximum_far_interactions": far_capacity,
                "maximum_near_interactions": near_capacity,
                "direct_chunk_size": chunk,
            }
        )

    def evaluate(self, tree: PreparedParticleOctree3D, /) -> TreeGravityResult:
        point_capacity = tree.positions.shape[0]
        parent_stencil = 27
        far_stencil = 189
        far_capacity = (
            point_capacity * max(tree.depth - 1, 1) * min(far_stencil, point_capacity)
            if self.maximum_far_interactions is None
            else self.maximum_far_interactions
        )
        near_capacity = (
            point_capacity * min(parent_stencil, point_capacity)
            if self.maximum_near_interactions is None
            else self.maximum_near_interactions
        )
        level_tree = SparseLevelOctreePlan(
            MortonAddressPlan((0.0, 0.0, 0.0), tree.box_size, tree.depth),
            point_capacity,
            far_interaction_capacity=far_capacity,
            near_interaction_capacity=near_capacity,
        ).prepare(
            tree.positions,
            active_mask=tree.active_mask,
            stable_ids=jnp.arange(point_capacity, dtype=jnp.int64),
        )
        hierarchy = level_tree.hierarchy
        operators = CartesianFMMOperators(
            self.expansion, self.gravitational_constant, self.softening
        )
        node_capacity = hierarchy.node_active.size
        node_slots = jnp.arange(node_capacity, dtype=jnp.int32)
        sorted_logical = hierarchy.storage_to_logical
        sorted_position = tree.positions[sorted_logical]
        sorted_mass = tree.masses[sorted_logical]
        sorted_active = hierarchy.sorted_active
        leaf_slots = jnp.nonzero(
            hierarchy.node_is_leaf,
            size=node_capacity,
            fill_value=node_capacity,
        )[0].astype(jnp.int32)
        leaf_valid = node_slots < hierarchy.evidence.active_leaves
        safe_leaf_slots = jnp.minimum(leaf_slots, node_capacity - 1)
        leaf_starts = jnp.where(
            leaf_valid,
            hierarchy.node_item_starts[safe_leaf_slots],
            point_capacity,
        )
        leaf_order = jnp.argsort(leaf_starts, stable=True)
        ordered_leaf_slots = leaf_slots[leaf_order]
        ordered_leaf_starts = leaf_starts[leaf_order]
        storage_slots = jnp.arange(point_capacity, dtype=jnp.int32)
        leaf_rank = jnp.searchsorted(ordered_leaf_starts, storage_slots, side="right") - 1
        point_leaf = jnp.where(
            sorted_active,
            ordered_leaf_slots[jnp.maximum(leaf_rank, 0)],
            -1,
        ).astype(jnp.int32)
        safe_point_leaf = jnp.maximum(point_leaf, 0)
        relative = sorted_position - hierarchy.node_centers[safe_point_leaf]
        safe_mass = jnp.where(sorted_active, sorted_mass, 0.0)
        multipole = jnp.zeros(
            (node_capacity, self.expansion.coefficient_count),
            dtype=tree.positions.dtype,
        )
        for coefficient, exponent in enumerate(self.expansion.exponents):
            particle_coefficient = safe_mass
            for axis in range(3):
                particle_coefficient = (
                    particle_coefficient * relative[:, axis] ** exponent[axis]
                )
            multipole = multipole.at[safe_point_leaf, coefficient].add(
                particle_coefficient
            )
        for level in range(tree.depth - 1, -1, -1):
            at_level = (
                hierarchy.node_active
                & ~hierarchy.node_is_leaf
                & (hierarchy.node_levels == level)
            )
            children = hierarchy.node_children
            child_valid = children >= 0
            safe_children = jnp.maximum(children, 0)
            child_values = multipole[safe_children]
            shifts = (
                hierarchy.node_centers[safe_children] - hierarchy.node_centers[:, None, :]
            )
            translated = jax.vmap(jax.vmap(operators.m2m))(child_values, shifts)
            parent_values = jnp.sum(
                jnp.where(child_valid[..., None], translated, 0.0), axis=1
            )
            multipole = jnp.where(at_level[:, None], parent_values, multipole)

        safe_far_targets = jnp.maximum(level_tree.far_targets, 0)
        safe_far_sources = jnp.maximum(level_tree.far_sources, 0)
        far_local = jax.vmap(operators.m2l)(
            multipole[safe_far_sources],
            hierarchy.node_centers[safe_far_sources],
            hierarchy.node_centers[safe_far_targets],
        )
        far_local = jnp.where(level_tree.far_active[:, None], far_local, 0.0)
        local = jnp.zeros_like(multipole).at[safe_far_targets].add(far_local)
        for level in range(1, tree.depth + 1):
            at_level = hierarchy.node_active & (hierarchy.node_levels == level)
            parents = jnp.maximum(hierarchy.node_parents, 0)
            shifts = hierarchy.node_centers - hierarchy.node_centers[parents]
            inherited = jax.vmap(operators.l2l)(local[parents], shifts)
            local = local + jnp.where(at_level[:, None], inherited, 0.0)
        _, local_acceleration = jax.vmap(operators.l2p)(
            local[safe_point_leaf],
            relative,
        )
        local_acceleration = jnp.where(sorted_active[:, None], local_acceleration, 0.0)

        chunk_offsets = jnp.arange(self.direct_chunk_size, dtype=jnp.int32)

        def near_relation_body(relation, acceleration):
            target_node = jnp.maximum(level_tree.near_targets[relation], 0)
            source_node = jnp.maximum(level_tree.near_sources[relation], 0)
            relation_active = level_tree.near_active[relation]
            target_start = hierarchy.node_item_starts[target_node]
            target_count = jnp.where(
                relation_active, hierarchy.node_item_counts[target_node], 0
            )
            source_start = hierarchy.node_item_starts[source_node]
            source_count = jnp.where(
                relation_active, hierarchy.node_item_counts[source_node], 0
            )

            def target_body(state):
                target_offset, current, interaction_count = state
                target_storage = target_start + target_offset
                target_position = sorted_position[target_storage]

                def source_body(source_state):
                    source_offset, contribution, count = source_state
                    source_storage = source_start + source_offset + chunk_offsets
                    source_valid = (source_storage < source_start + source_count) & (
                        source_storage < point_capacity
                    )
                    safe_source = jnp.minimum(source_storage, point_capacity - 1)
                    source_valid = (
                        source_valid
                        & sorted_active[safe_source]
                        & (safe_source != target_storage)
                    )
                    displacement = sorted_position[safe_source] - target_position
                    distance_squared = (
                        jnp.sum(displacement**2, axis=-1) + self.softening**2
                    )
                    value = (
                        self.gravitational_constant
                        * sorted_mass[safe_source, None]
                        * displacement
                        / distance_squared[:, None] ** 1.5
                    )
                    contribution = contribution + jnp.sum(
                        jnp.where(source_valid[:, None], value, 0.0), axis=0
                    )
                    return (
                        source_offset + self.direct_chunk_size,
                        contribution,
                        count + jnp.sum(source_valid, dtype=jnp.int32),
                    )

                _, contribution, count = jax.lax.fori_loop(
                    0,
                    (point_capacity + self.direct_chunk_size - 1)
                    // self.direct_chunk_size,
                    lambda _, source_state: source_body(source_state),
                    (
                        jnp.asarray(0, dtype=jnp.int32),
                        jnp.zeros((3,), dtype=tree.positions.dtype),
                        jnp.asarray(0, dtype=jnp.int32),
                    ),
                )
                current = current.at[target_storage].add(contribution)
                return target_offset + 1, current, interaction_count + count

            target_initial = (
                jnp.asarray(0, dtype=jnp.int32),
                acceleration,
                jnp.asarray(0, dtype=jnp.int32),
            )

            def evaluate_relation(initial):
                def target_iteration(_, state):
                    return jax.lax.cond(
                        state[0] < target_count,
                        target_body,
                        lambda current: current,
                        state,
                    )

                return jax.lax.fori_loop(
                    0,
                    point_capacity,
                    target_iteration,
                    initial,
                )

            _, updated, count = jax.lax.cond(
                relation_active,
                evaluate_relation,
                lambda initial: initial,
                target_initial,
            )
            return updated, count

        def accumulate_relation(relation, state):
            acceleration, count = state
            updated, relation_count = near_relation_body(relation, acceleration)
            return updated, count + relation_count

        near_acceleration, direct_count = jax.lax.fori_loop(
            0,
            level_tree.near_active.size,
            accumulate_relation,
            (
                jnp.zeros_like(local_acceleration),
                jnp.asarray(0, dtype=jnp.int32),
            ),
        )
        sorted_acceleration = local_acceleration + near_acceleration
        acceleration = (
            jnp.zeros_like(tree.positions).at[sorted_logical].set(sorted_acceleration)
        )
        acceleration = jnp.where(tree.active_mask[:, None], acceleration, 0.0)
        safe_far_distance = jnp.sqrt(
            jnp.sum(
                (
                    hierarchy.node_centers[safe_far_sources]
                    - hierarchy.node_centers[safe_far_targets]
                )
                ** 2,
                axis=-1,
            )
            + self.softening**2
        )
        far_size = 2.0 * jnp.max(hierarchy.node_half_widths[safe_far_sources], axis=-1)
        error_indicator = jnp.max(
            jnp.where(
                level_tree.far_active,
                (far_size / safe_far_distance) ** 2,
                0.0,
            ),
            initial=0.0,
        )
        finite = jnp.all(jnp.isfinite(acceleration))
        successful = level_tree.evidence.successful & finite
        evidence = TreeGravityEvidence(
            net_force=jnp.sum(
                jnp.where(tree.active_mask, tree.masses, 0.0)[:, None] * acceleration,
                axis=0,
            ),
            maximum_acceleration=jnp.max(
                jnp.sqrt(jnp.sum(acceleration**2, axis=-1)), initial=0.0
            ),
            accepted_leaf_interactions=jnp.sum(level_tree.far_active, dtype=jnp.int32),
            direct_particle_interactions=direct_count,
            maximum_opening_indicator=error_indicator,
            traversal_complete=level_tree.evidence.successful,
            active_nodes=level_tree.evidence.active_nodes,
            finite=finite,
            successful=successful,
        )
        return TreeGravityResult(acceleration, evidence, successful)


class PeriodicEwaldEvidence(StrictModule):
    real_space_acceleration: Array
    reciprocal_acceleration: Array
    net_force: Array
    finite: Array
    successful: Array


class PeriodicEwaldResult(StrictModule):
    acceleration: Array
    evidence: PeriodicEwaldEvidence
    successful: Array


class PeriodicEwaldForcePlan(StrictModule, NonTrainableState):
    """Small-N softened-neutral periodic Ewald acceleration reference."""

    box_size: tuple[float, ...] = eqx.field(static=True)
    gravitational_constant: float = eqx.field(static=True)
    softening: float = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    real_offsets: Array
    wavevectors: Array
    volume: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        box_size: tuple[float, ...],
        gravitational_constant: float,
        /,
        *,
        softening: float,
        alpha: float,
        real_shells: int = 2,
        reciprocal_modes: int = 4,
    ):
        lengths = tuple(float(value) for value in box_size)
        gravity = float(gravitational_constant)
        epsilon = float(softening)
        alpha_ = float(alpha)
        real = int(real_shells)
        reciprocal = int(reciprocal_modes)
        if (
            not lengths
            or any(not np.isfinite(value) or value <= 0.0 for value in lengths)
            or not np.isfinite(gravity)
            or gravity <= 0.0
            or not np.isfinite(epsilon)
            or epsilon <= 0.0
            or not np.isfinite(alpha_)
            or alpha_ <= 0.0
            or real < 0
            or reciprocal < 1
        ):
            raise ValueError("Periodic Ewald policy is invalid.")
        dimension = len(lengths)
        integer_offsets = np.asarray(
            tuple(product(range(-real, real + 1), repeat=dimension)), dtype=float
        )
        reciprocal_indices = np.asarray(
            tuple(
                index
                for index in product(range(-reciprocal, reciprocal + 1), repeat=dimension)
                if any(value != 0 for value in index)
            ),
            dtype=float,
        )
        wavevectors = 2.0 * np.pi * reciprocal_indices / np.asarray(lengths)[None, :]
        self.box_size = lengths
        self.gravitational_constant = gravity
        self.softening = epsilon
        self.alpha = alpha_
        self.real_offsets = jnp.asarray(integer_offsets * np.asarray(lengths)[None, :])
        self.wavevectors = jnp.asarray(wavevectors)
        self.volume = float(np.prod(lengths))
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-ewald-force",
                "box_size": list(lengths),
                "gravitational_constant": gravity,
                "softening": epsilon,
                "alpha": alpha_,
                "real_shells": real,
                "reciprocal_modes": reciprocal,
            }
        )

    def evaluate(self, positions: ArrayLike, masses: ArrayLike, /) -> PeriodicEwaldResult:
        position = jnp.asarray(positions)
        mass = jnp.asarray(masses, dtype=position.dtype)
        if (
            position.ndim != 2
            or position.shape[1] != len(self.box_size)
            or mass.shape != (position.shape[0],)
        ):
            raise ValueError("Periodic Ewald positions/masses have incompatible shapes.")
        position = eqx.error_if(
            position,
            jnp.any(~jnp.isfinite(position))
            | jnp.any(~jnp.isfinite(mass))
            | jnp.any(mass <= 0.0),
            "Periodic Ewald inputs must be finite with positive masses.",
        )
        target = position[:, None, None, :]
        source = position[None, :, None, :] + self.real_offsets[None, None, :, :]
        displacement = source - target
        distance_squared = jnp.sum(displacement**2, axis=-1) + self.softening**2
        distance = jnp.sqrt(distance_squared)
        zero_offset = jnp.all(self.real_offsets == 0.0, axis=-1)
        self_pair = (
            jnp.eye(position.shape[0], dtype=bool)[:, :, None]
            & zero_offset[None, None, :]
        )
        screening = jax.scipy.special.erfc(self.alpha * distance) + (
            2.0
            * self.alpha
            * distance
            / jnp.sqrt(jnp.pi)
            * jnp.exp(-((self.alpha * distance) ** 2))
        )
        inverse_cube = jnp.where(self_pair, 0.0, screening / distance**3)
        real_acceleration = jnp.sum(
            self.gravitational_constant
            * mass[None, :, None, None]
            * displacement
            * inverse_cube[..., None],
            axis=(1, 2),
        )
        k = self.wavevectors.astype(position.dtype)
        k_squared = jnp.sum(k**2, axis=-1)
        source_phase = contract("kd,nd->kn", k, position)
        density_real = contract("n,kn->k", mass, jnp.cos(source_phase))
        density_imag = -contract("n,kn->k", mass, jnp.sin(source_phase))
        target_phase = source_phase.T
        real_product = -density_real[None, :] * jnp.sin(target_phase) - density_imag[
            None, :
        ] * jnp.cos(target_phase)
        coefficient = (
            4.0
            * jnp.pi
            * self.gravitational_constant
            / self.volume
            * jnp.exp(-k_squared / (4.0 * self.alpha**2))
            / k_squared
        )
        reciprocal_acceleration = contract("k,nk,kd->nd", coefficient, real_product, k)
        acceleration = real_acceleration + reciprocal_acceleration
        net_force = jnp.sum(mass[:, None] * acceleration, axis=0)
        finite = jnp.all(jnp.isfinite(acceleration))
        evidence = PeriodicEwaldEvidence(
            real_acceleration,
            reciprocal_acceleration,
            net_force,
            finite,
            finite,
        )
        return PeriodicEwaldResult(acceleration, evidence, finite)


class PeriodicBarnesHutPlan(StrictModule, NonTrainableState):
    """Barnes-Hut plus an exact small-N Ewald-minus-direct periodic correction."""

    barnes_hut: BarnesHutGravityPlan
    ewald: Any

    def __init__(self, barnes_hut: BarnesHutGravityPlan, ewald: Any, /):
        if (
            barnes_hut.gravitational_constant != ewald.gravitational_constant
            or barnes_hut.softening != ewald.softening
        ):
            raise ValueError("Barnes-Hut and Ewald kernels must match.")
        self.barnes_hut = barnes_hut
        self.ewald = ewald

    def evaluate(self, tree: PreparedParticleOctree3D, /) -> TreeGravityResult:
        approximate = self.barnes_hut.evaluate(tree)
        position = tree.positions
        displacement = position[None, :, :] - position[:, None, :]
        squared = jnp.sum(displacement**2, axis=-1) + self.barnes_hut.softening**2
        direct = jnp.sum(
            jnp.where(
                (tree.active_mask[None, :] & ~jnp.eye(position.shape[0], dtype=bool))[
                    ..., None
                ],
                self.barnes_hut.gravitational_constant
                * tree.masses[None, :, None]
                * displacement
                / squared[..., None] ** 1.5,
                0.0,
            ),
            axis=1,
        )
        periodic = self.ewald.evaluate(position, tree.masses)
        acceleration = approximate.acceleration + periodic.acceleration - direct
        finite = (
            approximate.successful
            & periodic.successful
            & jnp.all(jnp.isfinite(acceleration))
        )
        evidence = TreeGravityEvidence(
            net_force=jnp.sum(tree.masses[:, None] * acceleration, axis=0),
            maximum_acceleration=jnp.max(
                jnp.sqrt(jnp.sum(acceleration**2, axis=-1)), initial=0.0
            ),
            accepted_leaf_interactions=(approximate.evidence.accepted_leaf_interactions),
            direct_particle_interactions=(
                approximate.evidence.direct_particle_interactions
            ),
            maximum_opening_indicator=(approximate.evidence.maximum_opening_indicator),
            traversal_complete=approximate.evidence.traversal_complete,
            active_nodes=approximate.evidence.active_nodes,
            finite=finite,
            successful=finite,
        )
        return TreeGravityResult(acceleration, evidence, finite)


class MeshComplementCalibrationEvidence(StrictModule):
    maximum_absolute_residual: Array
    rms_residual: Array
    tolerance_met: Array
    finite: Array
    successful: Array


class MeshComplementCalibrationPlan(StrictModule, NonTrainableState):
    tolerance: float = eqx.field(static=True)

    def __init__(self, tolerance: float, /):
        value = float(tolerance)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("Mesh-complement tolerance must be finite and positive.")
        self.tolerance = value

    def qualify(
        self,
        reference_acceleration: ArrayLike,
        long_range_acceleration: ArrayLike,
        short_range_acceleration: ArrayLike,
        /,
    ) -> MeshComplementCalibrationEvidence:
        reference = jnp.asarray(reference_acceleration)
        long_range = jnp.asarray(long_range_acceleration, dtype=reference.dtype)
        short_range = jnp.asarray(short_range_acceleration, dtype=reference.dtype)
        if long_range.shape != reference.shape or short_range.shape != reference.shape:
            raise ValueError("Mesh-complement accelerations must have equal shapes.")
        residual = long_range + short_range - reference
        norm = jnp.sqrt(jnp.sum(residual**2, axis=-1))
        maximum = jnp.max(norm)
        rms = jnp.sqrt(jnp.mean(norm**2))
        finite = jnp.all(jnp.isfinite(residual))
        tolerance_met = maximum <= self.tolerance
        return MeshComplementCalibrationEvidence(
            maximum, rms, tolerance_met, finite, finite & tolerance_met
        )


class TreePMSplitPolicy(StrictModule, NonTrainableState):
    split_scale: float = eqx.field(static=True)
    cutoff: float = eqx.field(static=True)
    compensation_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(self, split_scale: float, cutoff: float, compensation_id: str, /):
        split = float(split_scale)
        cutoff_ = float(cutoff)
        compensation = str(compensation_id).strip()

        if (
            not np.isfinite(split)
            or split <= 0.0
            or not np.isfinite(cutoff_)
            or cutoff_ <= split
            or not compensation
        ):
            raise ValueError("TreePM split policy is invalid.")
        self.split_scale = split
        self.cutoff = cutoff_
        self.compensation_id = compensation
        self.policy_id = canonical_fingerprint(
            {
                "kind": "treepm-split-policy",
                "split_scale": split,
                "cutoff": cutoff_,
                "compensation_id": compensation,
            }
        )


class TreePMResult(StrictModule):
    long_range_acceleration: Array
    short_range_acceleration: Array
    total_acceleration: Array
    short_evidence: TreeGravityEvidence
    finite: Array
    successful: Array


class TreePMPlan(StrictModule, NonTrainableState):
    short_range: BarnesHutGravityPlan
    split: TreePMSplitPolicy
    plan_id: str = eqx.field(static=True)

    def __init__(self, short_range: BarnesHutGravityPlan, split: TreePMSplitPolicy, /):
        self.short_range = short_range
        self.split = split
        self.plan_id = canonical_fingerprint(
            {
                "kind": "single-device-treepm",
                "short_range": short_range.plan_id,
                "split": split.policy_id,
            }
        )

    def evaluate(
        self,
        tree: PreparedParticleOctree3D,
        long_range_acceleration: ArrayLike,
        /,
    ) -> TreePMResult:
        long_range = jnp.asarray(long_range_acceleration, dtype=tree.positions.dtype)
        if long_range.shape != tree.positions.shape:
            raise ValueError("TreePM long-range acceleration must match particles.")
        short = self.short_range.evaluate(
            tree,
            short_range_scale=self.split.split_scale,
            cutoff=self.split.cutoff,
        )
        total = long_range + short.acceleration
        finite = jnp.all(jnp.isfinite(total))
        return TreePMResult(
            long_range,
            short.acceleration,
            total,
            short.evidence,
            finite,
            finite & short.successful,
        )


__all__ = [
    "BarnesHutGravityPlan",
    "CartesianExpansionSpace",
    "CartesianFMMOperators",
    "DirectParticleGravityPlan",
    "DistributedParticleLayout",
    "MeshComplementCalibrationEvidence",
    "MeshComplementCalibrationPlan",
    "NewtonianPairKernel",
    "ParticleGravityEvidence",
    "ParticleOctreePlan3D",
    "PeriodicBarnesHutPlan",
    "PeriodicEwaldEvidence",
    "PeriodicEwaldForcePlan",
    "PeriodicEwaldResult",
    "PreparedParticleOctree3D",
    "TreeGravityEvidence",
    "TreeGravityResult",
    "TreePMPlan",
    "TreePMResult",
    "TreePMSplitPolicy",
    "UniformFMMPlan",
]
