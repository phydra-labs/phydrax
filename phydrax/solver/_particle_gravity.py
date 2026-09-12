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
from ..backends.spatial import spatial_pair_acceleration
from ..discretization.spatial import (
    MortonAddressPlan,
    MortonPointHierarchyPlan,
    MortonPointHierarchyState,
)
from ..discretization.spatial._plane_interactions import MortonPlaneInteractionPlan
from ..discretization.spatial._plane_schedule import MortonPlaneSchedulePlan
from ..operators.integral.multipole._cartesian_radial import (
    monomial,
    multi_binomial,
    multi_index_factorial,
    plummer_scaled_cartesian_derivatives,
)
from ..sparse import EdgeRelation, RelationAccumulation, RelationExecutionPlan


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
        sorted_leaf_indices = hierarchy.sorted_point_leaf_slots
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
        leaf_indices = hierarchy.logical_point_leaf_slots
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


class CartesianFMMResourceEvidence(StrictModule):
    expansion_order: Array
    opening_angle: Array
    required_nodes: Array
    node_capacity: Array
    required_queue: Array
    queue_capacity: Array
    required_far: Array
    far_capacity: Array
    required_near: Array
    near_capacity: Array
    minimum_scale_exponent: Array
    maximum_scale_exponent: Array
    p2m_count: Array
    m2m_count: Array
    m2l_count: Array
    l2l_count: Array
    l2p_count: Array
    p2p_count: Array
    successful: Array
    accumulation: RelationAccumulation = eqx.field(static=True)


class TreeGravityResult(StrictModule):
    acceleration: Array
    evidence: TreeGravityEvidence
    successful: Array
    fmm_evidence: CartesianFMMResourceEvidence | None = None


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
    """Graded three-dimensional Cartesian multi-index coefficient layout."""

    order: int = eqx.field(static=True)
    exponents: tuple[tuple[int, int, int], ...] = eqx.field(static=True)
    degrees: tuple[int, ...] = eqx.field(static=True)
    factorials: tuple[int, ...] = eqx.field(static=True)
    coefficient_count: int = eqx.field(static=True)

    def __init__(self, order: int, /):
        order_ = int(order)
        if order_ < 1 or order_ > 7:
            raise ValueError("Cartesian FMM order must lie in [1,7].")
        exponents = tuple(
            (i, j, k)
            for total in range(order_ + 1)
            for i in range(total + 1)
            for j in range(total - i + 1)
            for k in (total - i - j,)
        )
        self.order = order_
        self.exponents = exponents
        self.degrees = tuple(sum(exponent) for exponent in exponents)
        self.factorials = tuple(multi_index_factorial(exponent) for exponent in exponents)
        self.coefficient_count = len(exponents)


class CartesianFMMOperators(StrictModule, NonTrainableState):
    """Scale-normalized Cartesian P2M/M2M/M2L/L2L/L2P/P2P operators."""

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
        *,
        scale: ArrayLike = 1.0,
    ) -> Array:
        position = jnp.asarray(positions)
        mass = jnp.asarray(masses, dtype=position.dtype)
        center_ = jnp.asarray(center, dtype=position.dtype)
        scale_ = jnp.asarray(scale, dtype=position.dtype)
        relative = (position - center_) / scale_
        coefficients = [
            jnp.sum(
                mass
                * jax.vmap(lambda value, exponent=exponent: monomial(value, exponent))(
                    relative
                )
            )
            for exponent in self.expansion.exponents
        ]
        return jnp.stack(coefficients)

    def m2m(
        self,
        coefficients: ArrayLike,
        shift: ArrayLike,
        /,
        *,
        source_scale: ArrayLike = 1.0,
        target_scale: ArrayLike = 1.0,
    ) -> Array:
        values = jnp.asarray(coefficients)
        shift_ = jnp.asarray(shift, dtype=values.dtype)
        source_scale_ = jnp.asarray(source_scale, dtype=values.dtype)
        target_scale_ = jnp.asarray(target_scale, dtype=values.dtype)
        normalized_shift = shift_ / target_scale_
        scale_ratio = source_scale_ / target_scale_
        output = []
        for alpha in self.expansion.exponents:
            total = jnp.asarray(0.0, dtype=values.dtype)
            for index, beta in enumerate(self.expansion.exponents):
                if all(beta[axis] <= alpha[axis] for axis in range(3)):
                    difference = tuple(alpha[axis] - beta[axis] for axis in range(3))
                    total = (
                        total
                        + multi_binomial(alpha, beta)
                        * monomial(normalized_shift, difference)
                        * scale_ratio ** sum(beta)
                        * values[index]
                    )
            output.append(total)
        return jnp.stack(output)

    def m2l(
        self,
        multipole: ArrayLike,
        source_center: ArrayLike,
        target_center: ArrayLike,
        /,
        *,
        source_scale: ArrayLike = 1.0,
        target_scale: ArrayLike = 1.0,
    ) -> Array:
        values = jnp.asarray(multipole)
        source = jnp.asarray(source_center, dtype=values.dtype)
        target = jnp.asarray(target_center, dtype=values.dtype)
        source_scale_ = jnp.asarray(source_scale, dtype=values.dtype)
        target_scale_ = jnp.asarray(target_scale, dtype=values.dtype)
        displacement = target - source
        distance = jnp.sqrt(jnp.sum(displacement * displacement))
        common_extent = jnp.maximum(
            jnp.maximum(source_scale_, target_scale_),
            jnp.maximum(distance, jnp.asarray(self.softening, dtype=values.dtype)),
        )
        common_scale = jnp.exp2(jnp.ceil(jnp.log2(common_extent)))
        derivatives = plummer_scaled_cartesian_derivatives(
            self.expansion.exponents,
            displacement,
            self.softening,
            self.gravitational_constant,
            common_scale,
        )
        source_ratio = source_scale_ / common_scale
        target_ratio = target_scale_ / common_scale
        output = []
        for beta_index, beta in enumerate(self.expansion.exponents):
            beta_degree = self.expansion.degrees[beta_index]
            total = jnp.asarray(0.0, dtype=values.dtype)
            for alpha_index, alpha in enumerate(self.expansion.exponents):
                alpha_degree = self.expansion.degrees[alpha_index]
                if alpha_degree + beta_degree <= self.expansion.order:
                    derivative_exponent = tuple(
                        alpha[axis] + beta[axis] for axis in range(3)
                    )
                    derivative_index = self.expansion.exponents.index(derivative_exponent)
                    total = total + (-1) ** alpha_degree * values[
                        alpha_index
                    ] * derivatives[
                        derivative_index
                    ] * source_ratio**alpha_degree * target_ratio**beta_degree / (
                        common_scale
                        * self.expansion.factorials[alpha_index]
                        * self.expansion.factorials[beta_index]
                    )
            output.append(total)
        return jnp.stack(output)

    def l2l(
        self,
        local: ArrayLike,
        shift: ArrayLike,
        /,
        *,
        source_scale: ArrayLike = 1.0,
        target_scale: ArrayLike = 1.0,
    ) -> Array:
        values = jnp.asarray(local)
        shift_ = jnp.asarray(shift, dtype=values.dtype)
        source_scale_ = jnp.asarray(source_scale, dtype=values.dtype)
        target_scale_ = jnp.asarray(target_scale, dtype=values.dtype)
        normalized_shift = shift_ / source_scale_
        scale_ratio = target_scale_ / source_scale_
        output = []
        for beta in self.expansion.exponents:
            total = jnp.asarray(0.0, dtype=values.dtype)
            for alpha_index, alpha in enumerate(self.expansion.exponents):
                if all(beta[axis] <= alpha[axis] for axis in range(3)):
                    difference = tuple(alpha[axis] - beta[axis] for axis in range(3))
                    total = (
                        total
                        + multi_binomial(alpha, beta)
                        * monomial(normalized_shift, difference)
                        * scale_ratio ** sum(beta)
                        * values[alpha_index]
                    )
            output.append(total)
        return jnp.stack(output)

    def l2p(
        self,
        local: ArrayLike,
        displacement: ArrayLike,
        /,
        *,
        scale: ArrayLike = 1.0,
    ) -> tuple[Array, Array]:
        values = jnp.asarray(local)
        offset = jnp.asarray(displacement, dtype=values.dtype)
        scale_ = jnp.asarray(scale, dtype=values.dtype)
        normalized = offset / scale_
        potential = jnp.asarray(0.0, dtype=values.dtype)
        gradient = jnp.zeros((3,), dtype=values.dtype)
        for index, exponent in enumerate(self.expansion.exponents):
            potential = potential + values[index] * monomial(normalized, exponent)
            for axis in range(3):
                if exponent[axis] > 0:
                    reduced = list(exponent)
                    reduced[axis] -= 1
                    gradient = gradient.at[axis].add(
                        values[index]
                        * exponent[axis]
                        * monomial(normalized, tuple(reduced))
                        / scale_
                    )
        return potential, -gradient

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
    """Scale-normalized Cartesian FMM over compact Morton execution planes."""

    gravitational_constant: float = eqx.field(static=True)
    softening: float = eqx.field(static=True)
    expansion: CartesianExpansionSpace
    opening_angle: float = eqx.field(static=True)
    maximum_nodes: int | None = eqx.field(static=True)
    maximum_queue_interactions: int | None = eqx.field(static=True)
    maximum_far_interactions: int | None = eqx.field(static=True)
    maximum_near_interactions: int | None = eqx.field(static=True)
    maximum_leaf_occupancy: int = eqx.field(static=True)
    coarsening_factor: int = eqx.field(static=True)
    target_top_nodes: int = eqx.field(static=True)
    accumulation: RelationAccumulation = eqx.field(static=True)
    execution_backend: str = eqx.field(static=True)
    pallas_interpret: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        gravitational_constant: float,
        expansion: CartesianExpansionSpace,
        /,
        *,
        softening: float,
        opening_angle: float = 0.6,
        maximum_nodes: int | None = None,
        maximum_queue_interactions: int | None = None,
        maximum_far_interactions: int | None = None,
        maximum_near_interactions: int | None = None,
        maximum_leaf_occupancy: int = 16,
        coarsening_factor: int = 8,
        target_top_nodes: int = 32,
        accumulation: RelationAccumulation = "deterministic",
        execution_backend: str = "jax",
        pallas_interpret: bool = False,
    ):
        gravity = float(gravitational_constant)
        epsilon = float(softening)
        theta = float(opening_angle)
        node_capacity = None if maximum_nodes is None else int(maximum_nodes)
        queue_capacity = (
            None
            if maximum_queue_interactions is None
            else int(maximum_queue_interactions)
        )
        far_capacity = (
            None if maximum_far_interactions is None else int(maximum_far_interactions)
        )
        near_capacity = (
            None if maximum_near_interactions is None else int(maximum_near_interactions)
        )
        leaf_occupancy = int(maximum_leaf_occupancy)
        coarse = int(coarsening_factor)
        top_nodes = int(target_top_nodes)
        if (
            not np.isfinite(gravity)
            or gravity <= 0.0
            or not np.isfinite(epsilon)
            or epsilon <= 0.0
            or not 0.0 < theta < 1.0
            or (node_capacity is not None and node_capacity <= 0)
            or (queue_capacity is not None and queue_capacity <= 0)
            or (far_capacity is not None and far_capacity <= 0)
            or (near_capacity is not None and near_capacity <= 0)
            or leaf_occupancy <= 0
            or coarse < 2
            or top_nodes <= 0
            or accumulation not in ("fast", "deterministic", "compensated")
            or execution_backend not in ("jax", "pallas")
        ):
            raise ValueError("Cartesian FMM policy is invalid.")
        self.gravitational_constant = gravity
        self.softening = epsilon
        self.expansion = expansion
        self.opening_angle = theta
        self.maximum_nodes = node_capacity
        self.maximum_queue_interactions = queue_capacity
        self.maximum_far_interactions = far_capacity
        self.maximum_near_interactions = near_capacity
        self.maximum_leaf_occupancy = leaf_occupancy
        self.coarsening_factor = coarse
        self.target_top_nodes = top_nodes
        self.accumulation = accumulation
        self.execution_backend = execution_backend
        self.pallas_interpret = bool(pallas_interpret)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "plane-cartesian-fmm",
                "gravitational_constant": gravity,
                "softening": epsilon,
                "order": expansion.order,
                "opening_angle": theta,
                "maximum_nodes": node_capacity,
                "maximum_queue_interactions": queue_capacity,
                "maximum_far_interactions": far_capacity,
                "maximum_near_interactions": near_capacity,
                "maximum_leaf_occupancy": leaf_occupancy,
                "coarsening_factor": coarse,
                "target_top_nodes": top_nodes,
                "accumulation": accumulation,
                "execution_backend": execution_backend,
                "pallas_interpret": bool(pallas_interpret),
            }
        )

    def _evaluate_impl(self, tree: PreparedParticleOctree3D, /) -> TreeGravityResult:
        point_capacity = int(tree.positions.shape[0])
        address = MortonAddressPlan(
            (0.0, 0.0, 0.0),
            tree.box_size,
            tree.depth,
        )
        schedule_plan = MortonPlaneSchedulePlan(
            address,
            point_capacity,
            node_capacity=self.maximum_nodes,
            maximum_leaf_occupancy=self.maximum_leaf_occupancy,
            coarsening_factor=self.coarsening_factor,
            target_top_nodes=self.target_top_nodes,
        )
        schedule = schedule_plan.build(
            tree.positions,
            active_mask=tree.active_mask,
            stable_ids=jnp.arange(point_capacity, dtype=jnp.int64),
        )
        leaf_capacity = schedule_plan.plane_capacities[0]
        maximum_leaf_pairs = max(leaf_capacity * leaf_capacity, 1)
        queue_capacity = (
            maximum_leaf_pairs
            if self.maximum_queue_interactions is None
            else self.maximum_queue_interactions
        )
        far_capacity = (
            maximum_leaf_pairs
            if self.maximum_far_interactions is None
            else self.maximum_far_interactions
        )
        near_capacity = (
            maximum_leaf_pairs
            if self.maximum_near_interactions is None
            else self.maximum_near_interactions
        )
        interaction_plan = MortonPlaneInteractionPlan(
            schedule_plan,
            opening_angle=self.opening_angle,
            queue_capacity=queue_capacity,
            far_capacity=far_capacity,
            near_capacity=near_capacity,
        )
        interactions = interaction_plan.build(schedule)
        operators = CartesianFMMOperators(
            self.expansion,
            self.gravitational_constant,
            self.softening,
        )
        node_capacity = schedule_plan.node_capacity
        sorted_logical = schedule.point_order.storage_to_logical
        sorted_position = schedule.point_order.encoding.coordinates[sorted_logical]
        sorted_mass = tree.masses[sorted_logical]
        sorted_active = schedule.point_order.sorted_active
        point_leaf = schedule.sorted_point_leaf_slots
        safe_point_leaf = jnp.maximum(point_leaf, 0)
        relative = sorted_position - schedule.node_centers[safe_point_leaf]
        normalized_relative = relative / schedule.node_scales[safe_point_leaf, None]
        safe_mass = jnp.where(sorted_active, sorted_mass, 0.0)
        multipole = jnp.zeros(
            (node_capacity, self.expansion.coefficient_count),
            dtype=tree.positions.dtype,
        )
        for coefficient, exponent in enumerate(self.expansion.exponents):
            particle_coefficient = safe_mass
            for axis in range(3):
                particle_coefficient = (
                    particle_coefficient * normalized_relative[:, axis] ** exponent[axis]
                )
            multipole = multipole.at[safe_point_leaf, coefficient].add(
                particle_coefficient
            )

        child_offsets = jnp.arange(schedule_plan.coarsening_factor, dtype=jnp.int32)
        m2m_count = jnp.asarray(0, dtype=jnp.int32)
        for plane in range(1, schedule_plan.plane_count):
            at_plane = schedule.node_active & (schedule.node_planes == plane)
            children = schedule.node_child_starts[:, None] + child_offsets[None, :]
            child_valid = at_plane[:, None] & (
                child_offsets[None, :] < schedule.node_child_counts[:, None]
            )
            safe_children = jnp.clip(children, 0, node_capacity - 1)
            child_values = multipole[safe_children]
            shifts = (
                schedule.node_centers[safe_children] - schedule.node_centers[:, None, :]
            )
            source_scales = schedule.node_scales[safe_children]
            target_scales = jnp.broadcast_to(
                schedule.node_scales[:, None], source_scales.shape
            )
            translated = jax.vmap(
                jax.vmap(
                    lambda value, shift, source_scale, target_scale: operators.m2m(
                        value,
                        shift,
                        source_scale=source_scale,
                        target_scale=target_scale,
                    )
                )
            )(child_values, shifts, source_scales, target_scales)
            parent_values = jnp.sum(
                jnp.where(child_valid[..., None], translated, 0.0),
                axis=1,
            )
            multipole = jnp.where(at_plane[:, None], parent_values, multipole)
            m2m_count = m2m_count + jnp.sum(child_valid, dtype=jnp.int32)

        far_sources = interactions.far.source_indices
        far_targets = interactions.far.target_indices
        far_local = jax.vmap(
            lambda multipole_, source, target, source_scale, target_scale: operators.m2l(
                multipole_,
                source,
                target,
                source_scale=source_scale,
                target_scale=target_scale,
            )
        )(
            multipole[far_sources],
            schedule.node_centers[far_sources],
            schedule.node_centers[far_targets],
            schedule.node_scales[far_sources],
            schedule.node_scales[far_targets],
        )
        far_local = jnp.where(interactions.far.valid[:, None], far_local, 0.0)
        far_execution = RelationExecutionPlan(
            maximum_active_targets=node_capacity
        ).prepare(
            interactions.far,
            stable_route_ids=jnp.arange(interactions.far.capacity, dtype=jnp.int64),
        )
        local, far_reduction = far_execution.reduce(
            far_local,
            accumulation=self.accumulation,
        )

        l2l_count = jnp.asarray(0, dtype=jnp.int32)
        for plane in range(schedule_plan.plane_count - 2, -1, -1):
            at_plane = schedule.node_active & (schedule.node_planes == plane)
            parents = jnp.maximum(schedule.node_parents, 0)
            inherited = jax.vmap(
                lambda value, shift, source_scale, target_scale: operators.l2l(
                    value,
                    shift,
                    source_scale=source_scale,
                    target_scale=target_scale,
                )
            )(
                local[parents],
                schedule.node_centers - schedule.node_centers[parents],
                schedule.node_scales[parents],
                schedule.node_scales,
            )
            local = local + jnp.where(at_plane[:, None], inherited, 0.0)
            l2l_count = l2l_count + jnp.sum(at_plane, dtype=jnp.int32)

        _, local_acceleration = jax.vmap(
            lambda value, offset, scale: operators.l2p(
                value,
                offset,
                scale=scale,
            )
        )(
            local[safe_point_leaf],
            relative,
            schedule.node_scales[safe_point_leaf],
        )
        local_acceleration = jnp.where(sorted_active[:, None], local_acceleration, 0.0)

        near_sources = interactions.near.source_indices
        near_targets = interactions.near.target_indices
        leaf_offsets = jnp.arange(schedule_plan.maximum_leaf_occupancy, dtype=jnp.int32)
        target_storage = (
            schedule.node_item_starts[near_targets, None] + leaf_offsets[None, :]
        )
        source_storage = (
            schedule.node_item_starts[near_sources, None] + leaf_offsets[None, :]
        )
        target_valid = interactions.near.valid[:, None] & (
            leaf_offsets[None, :] < schedule.node_item_counts[near_targets, None]
        )
        source_valid = interactions.near.valid[:, None] & (
            leaf_offsets[None, :] < schedule.node_item_counts[near_sources, None]
        )
        safe_targets = jnp.clip(target_storage, 0, point_capacity - 1)
        safe_sources = jnp.clip(source_storage, 0, point_capacity - 1)
        displacement = (
            sorted_position[safe_sources][:, None, :, :]
            - sorted_position[safe_targets][:, :, None, :]
        )
        pair_valid = (
            target_valid[:, :, None]
            & source_valid[:, None, :]
            & sorted_active[safe_targets][:, :, None]
            & sorted_active[safe_sources][:, None, :]
            & (safe_targets[:, :, None] != safe_sources[:, None, :])
        )
        pair_value = spatial_pair_acceleration(
            displacement,
            jnp.broadcast_to(
                sorted_mass[safe_sources][:, None, :],
                pair_valid.shape,
            ),
            pair_valid,
            softening=self.softening,
            coefficient=self.gravitational_constant,
            backend=self.execution_backend,
            pallas_interpret=self.pallas_interpret,
        )
        near_route_values = jnp.sum(pair_value, axis=2)
        point_route_valid = target_valid.reshape((-1,))
        point_targets = safe_targets.reshape((-1,))
        point_relation = EdgeRelation(
            jnp.zeros(point_targets.shape, dtype=jnp.int32),
            jnp.where(point_route_valid, point_targets, 0),
            source_size=1,
            target_size=point_capacity,
            valid=point_route_valid,
        )
        point_execution = RelationExecutionPlan(
            maximum_active_targets=point_capacity
        ).prepare(
            point_relation,
            stable_route_ids=jnp.arange(point_relation.capacity, dtype=jnp.int64),
        )
        near_acceleration, near_reduction = point_execution.reduce(
            near_route_values.reshape((-1, 3)),
            accumulation=self.accumulation,
        )
        sorted_acceleration = local_acceleration + near_acceleration
        acceleration = (
            jnp.zeros_like(tree.positions).at[sorted_logical].set(sorted_acceleration)
        )
        acceleration = jnp.where(tree.active_mask[:, None], acceleration, 0.0)
        finite = (
            jnp.all(jnp.isfinite(acceleration))
            & far_reduction.finite
            & near_reduction.finite
        )
        successful = (
            schedule.evidence.successful
            & interactions.evidence.successful
            & far_reduction.successful
            & near_reduction.successful
            & finite
        )
        direct_count = jnp.sum(pair_valid, dtype=jnp.int32)
        evidence = TreeGravityEvidence(
            net_force=jnp.sum(
                jnp.where(tree.active_mask, tree.masses, 0.0)[:, None] * acceleration,
                axis=0,
            ),
            maximum_acceleration=jnp.max(
                jnp.sqrt(jnp.sum(acceleration**2, axis=-1)),
                initial=0.0,
            ),
            accepted_leaf_interactions=interactions.evidence.required_far,
            direct_particle_interactions=direct_count,
            maximum_opening_indicator=interactions.evidence.maximum_accepted_ratio,
            traversal_complete=interactions.evidence.complete,
            active_nodes=schedule.evidence.active_nodes,
            finite=finite,
            successful=successful,
        )
        fmm_evidence = CartesianFMMResourceEvidence(
            expansion_order=jnp.asarray(self.expansion.order, dtype=jnp.int32),
            opening_angle=jnp.asarray(self.opening_angle, dtype=tree.positions.dtype),
            required_nodes=schedule.evidence.required_nodes,
            node_capacity=schedule.evidence.node_capacity,
            required_queue=interactions.evidence.required_queue,
            queue_capacity=interactions.evidence.queue_capacity,
            required_far=interactions.evidence.required_far,
            far_capacity=interactions.evidence.far_capacity,
            required_near=interactions.evidence.required_near,
            near_capacity=interactions.evidence.near_capacity,
            minimum_scale_exponent=schedule.evidence.minimum_scale_exponent,
            maximum_scale_exponent=schedule.evidence.maximum_scale_exponent,
            p2m_count=schedule.evidence.active_points,
            m2m_count=m2m_count,
            m2l_count=interactions.evidence.required_far,
            l2l_count=l2l_count,
            l2p_count=schedule.evidence.active_points,
            p2p_count=direct_count,
            successful=successful,
            accumulation=self.accumulation,
        )
        return TreeGravityResult(
            acceleration,
            evidence,
            successful,
            fmm_evidence,
        )

    def evaluate(self, tree: PreparedParticleOctree3D, /) -> TreeGravityResult:
        """Evaluate with a whole-FMM rematerializing reverse rule."""
        plan = self

        @jax.custom_vjp
        def run(current_tree):
            return plan._evaluate_impl(current_tree)

        def forward(current_tree):
            result = plan._evaluate_impl(current_tree)
            return result, current_tree

        def backward(current_tree, cotangent):
            _, pullback = jax.vjp(
                lambda value: plan._evaluate_impl(value).acceleration,
                current_tree,
            )
            return (pullback(cotangent.acceleration)[0],)

        run.defvjp(forward, backward)
        return run(tree)


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
    "CartesianFMMResourceEvidence",
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
