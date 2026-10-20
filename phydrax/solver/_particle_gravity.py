#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from itertools import product
from typing import Any, final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from phydrax.ein import contract

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..backends.spatial import spatial_leaf_pair_fragment
from ..discretization.spatial import (
    MortonAddressPlan,
    MortonPointHierarchyPlan,
    MortonPointHierarchyState,
    MortonRadiusRelationPlan,
    MortonSourcePlan,
    SpatialDistanceBackend,
)
from ..discretization.spatial._plane_interactions import (
    MortonPlaneInteractionPlan,
    MortonPlaneInteractionState,
)
from ..discretization.spatial._plane_schedule import (
    MortonPlaneSchedulePlan,
    MortonPlaneScheduleState,
)
from ..operators.integral.multipole._cartesian_radial import (
    monomial,
    multi_binomial,
    multi_index_factorial,
    plummer_scaled_cartesian_derivatives,
    treepm_scaled_cartesian_derivatives,
)
from ..sparse import (
    EdgeRelation,
    KeyGroupAccumulation,
    PreparedStreamedRelation,
    RelationAccumulation,
    route_reduce,
    RowRelation,
    StreamedFragment,
    StreamedPayloadSpec,
    StreamedRelationEvidence,
    StreamedRelationPlan,
)
from ..sparse._streamed import _input_link, _poison
from ..typing import checked


_TargetInputs: TypeAlias = tuple[Array, Array, Array]
_TargetOutputs: TypeAlias = tuple[Array, Array, Array, Array, Array]
_TraversalState: TypeAlias = tuple[Array, Array, Array, Array, Array, Array, Array, Array]
_DirectLeafState: TypeAlias = tuple[Array, Array, Array]


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
    ) -> None:
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
        mask = jnp.ones(radius_squared.shape, dtype=jnp.bool_)
        if exclude_diagonal:
            if targets.shape[0] != sources.shape[0]:
                raise ValueError(
                    "Diagonal exclusion requires equal source/target counts."
                )
            mask = mask & ~jnp.eye(targets.shape[0], dtype=jnp.bool_)
        if self.cutoff is not None:
            mask = mask & (radius_squared <= self.cutoff**2 + self.softening**2)
        contribution = (
            self.gravitational_constant
            * masses[None, :, None]
            * displacement
            / radius_squared[..., None] ** 1.5
        )
        return jnp.sum(jnp.where(mask[..., None], contribution, 0.0), axis=1)


class TreePMShortRangeKernel(StrictModule, NonTrainableState):
    """Exact compact-support TreePM short-range Plummer force policy."""

    gravitational_constant: float = eqx.field(static=True)
    softening: float = eqx.field(static=True)
    split_scale: float = eqx.field(static=True)
    cutoff: float = eqx.field(static=True)
    kernel_id: str = eqx.field(static=True)

    def __init__(
        self,
        gravitational_constant: float,
        softening: float,
        split_scale: float,
        cutoff: float,
        /,
    ) -> None:
        gravity = float(gravitational_constant)
        epsilon = float(softening)
        split = float(split_scale)
        cutoff_ = float(cutoff)
        if (
            not np.isfinite(gravity)
            or gravity <= 0.0
            or not np.isfinite(epsilon)
            or epsilon <= 0.0
            or not np.isfinite(split)
            or split <= 0.0
            or not np.isfinite(cutoff_)
            or cutoff_ <= max(split, epsilon)
        ):
            raise ValueError("TreePM short-range kernel is invalid.")
        self.gravitational_constant = gravity
        self.softening = epsilon
        self.split_scale = split
        self.cutoff = cutoff_
        self.kernel_id = canonical_fingerprint(
            {
                "kind": "treepm-short-range-kernel",
                "gravitational_constant": gravity,
                "softening": epsilon,
                "split_scale": split,
                "cutoff": cutoff_,
            }
        )

    def factor(self, distance: ArrayLike, /) -> Array:
        """Return the radial inverse-cube factor before mass and displacement."""
        radius = jnp.asarray(distance)
        argument = radius / (2.0 * self.split_scale)
        return radius**-3 * (
            jax.scipy.special.erfc(argument)
            + radius / (self.split_scale * jnp.sqrt(jnp.pi)) * jnp.exp(-(argument**2))
        )


class ParticleGravityEvidence(StrictModule):
    net_force: Array
    maximum_acceleration: Array
    interaction_count: Array
    approximation_error: Array
    finite: Array
    successful: Array


class DirectParticleGravityPlan(StrictModule, NonTrainableState):
    kernel: NewtonianPairKernel

    def __init__(self, kernel: NewtonianPairKernel, /) -> None:
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
            jnp.ones((position.shape[0],), dtype=jnp.bool_)
            if active_mask is None
            else jnp.asarray(active_mask, dtype=jnp.bool_)
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
    ) -> None:
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
    ) -> None:
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
            jnp.ones((position.shape[0],), dtype=jnp.bool_)
            if active_mask is None
            else jnp.asarray(active_mask, dtype=jnp.bool_)
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
    ) -> None:
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

        def radial_kernel(distance_squared: Array, distance: Array) -> Array:
            kernel = distance_squared ** (-1.5)
            if scale is not None:
                argument = distance / (2.0 * scale)
                kernel = kernel * (
                    jax.scipy.special.erfc(argument)
                    + distance / (scale * jnp.sqrt(jnp.pi)) * jnp.exp(-(argument**2))
                )
            return kernel

        def evaluate_target(inputs: _TargetInputs) -> _TargetOutputs:
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
            outside_cutoff = jnp.zeros((node_capacity,), dtype=jnp.bool_)
            fully_inside_cutoff = jnp.ones((node_capacity,), dtype=jnp.bool_)
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

            def propagate_blocked(node: Array, blocked: Array) -> Array:
                parent = hierarchy.node_parents[node]
                safe_parent = jnp.maximum(parent, 0)
                value = (parent >= 0) & (blocked[safe_parent] | terminal[safe_parent])
                return blocked.at[node].set(value)

            blocked = jax.lax.fori_loop(
                0,
                node_capacity,
                propagate_blocked,
                jnp.zeros((node_capacity,), dtype=jnp.bool_),
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

        def radial_kernel(distance_squared: Array, distance: Array) -> Array:
            kernel = distance_squared ** (-1.5)
            if scale is not None:
                argument = distance / (2.0 * scale)
                kernel = kernel * (
                    jax.scipy.special.erfc(argument)
                    + distance / (scale * jnp.sqrt(jnp.pi)) * jnp.exp(-(argument**2))
                )
            return kernel

        def evaluate_target(inputs: _TargetInputs) -> _TargetOutputs:
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

            def traversal_body(state: _TraversalState) -> _TraversalState:
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

                def direct_leaf(direct_state: _DirectLeafState) -> _DirectLeafState:
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

                def evaluate_direct_leaf(_operand: None) -> tuple[Array, Array]:
                    if not fixed_iterations:
                        _, leaf_acceleration, leaf_interactions = jax.lax.while_loop(
                            lambda direct_state: (
                                direct_state[0] < hierarchy.node_item_counts[node]
                            ),
                            direct_leaf,
                            direct_initial,
                        )
                        return leaf_acceleration, leaf_interactions

                    def run_chunks(chunk_count: int) -> tuple[Array, Array]:
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

                def traversal_iteration(
                    _: Array, state: _TraversalState
                ) -> _TraversalState:
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
        def run(current_tree: PreparedParticleOctree3D) -> TreeGravityResult:
            return plan._evaluate_impl(
                current_tree,
                short_range_scale=short_range_scale,
                cutoff=cutoff,
                fixed_iterations=False,
            )

        def forward(
            current_tree: PreparedParticleOctree3D,
        ) -> tuple[TreeGravityResult, PreparedParticleOctree3D]:
            result = plan._evaluate_impl(
                current_tree,
                short_range_scale=short_range_scale,
                cutoff=cutoff,
                fixed_iterations=False,
            )
            return result, current_tree

        def backward(
            current_tree: PreparedParticleOctree3D, cotangent: TreeGravityResult
        ) -> tuple[PreparedParticleOctree3D]:
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

    def __init__(self, order: int, /) -> None:
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
    short_range_scale: float | None = eqx.field(static=True)
    short_range_cutoff: float | None = eqx.field(static=True)

    def __init__(
        self,
        expansion: CartesianExpansionSpace,
        gravitational_constant: float,
        softening: float,
        /,
        *,
        short_range_scale: float | None = None,
        short_range_cutoff: float | None = None,
    ) -> None:
        gravity = float(gravitational_constant)
        epsilon = float(softening)
        split = None if short_range_scale is None else float(short_range_scale)
        cutoff = None if short_range_cutoff is None else float(short_range_cutoff)
        if (
            not np.isfinite(gravity)
            or gravity <= 0.0
            or not np.isfinite(epsilon)
            or epsilon <= 0.0
            or (split is None) != (cutoff is None)
            or (
                split is not None
                and cutoff is not None
                and (
                    not np.isfinite(split)
                    or split <= 0.0
                    or not np.isfinite(cutoff)
                    or cutoff <= max(split, epsilon)
                )
            )
        ):
            raise ValueError("Cartesian FMM operator constants are invalid.")
        self.expansion = expansion
        self.gravitational_constant = gravity
        self.softening = epsilon
        self.short_range_scale = split
        self.short_range_cutoff = cutoff

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
                    difference = (
                        alpha[0] - beta[0],
                        alpha[1] - beta[1],
                        alpha[2] - beta[2],
                    )
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
        if self.short_range_scale is None:
            derivatives = plummer_scaled_cartesian_derivatives(
                self.expansion.exponents,
                displacement,
                self.softening,
                self.gravitational_constant,
                common_scale,
            )
        else:
            derivatives = treepm_scaled_cartesian_derivatives(
                self.expansion.exponents,
                displacement,
                self.softening,
                self.gravitational_constant,
                common_scale,
                self.short_range_scale,
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
                    difference = (
                        alpha[0] - beta[0],
                        alpha[1] - beta[1],
                        alpha[2] - beta[2],
                    )
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
                        * monomial(normalized, (reduced[0], reduced[1], reduced[2]))
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
        radius = jnp.sqrt(radius_squared)
        factor = radius_squared ** (-1.5)
        split = self.short_range_scale
        cutoff = self.short_range_cutoff
        if split is not None and cutoff is not None:
            argument = radius / (2.0 * split)
            factor = factor * (
                jax.scipy.special.erfc(argument)
                + radius / (split * jnp.sqrt(jnp.pi)) * jnp.exp(-(argument**2))
            )
            factor = jnp.where(
                radius <= cutoff,
                factor,
                0.0,
            )
        return jnp.sum(
            self.gravitational_constant * mass[:, None] * displacement * factor[:, None],
            axis=0,
        )


class PreparedUniformFMMStructure(StrictModule):
    """Position-dependent Cartesian FMM topology reused across source strengths.

    Owns the compact Morton plane schedule and receiver-major bounded sparse
    streams of accepted far node pairs and near leaf pairs. Particle-pair
    working storage is fragment-local, never proportional to near capacity.
    Strengths enter only at evaluation. ``positions`` are box coordinates
    with origin ``(0, 0, 0)`` and inactive rows zeroed.
    """

    positions: Array
    active_mask: Array
    schedule: MortonPlaneScheduleState
    interactions: MortonPlaneInteractionState
    far_execution: PreparedStreamedRelation
    near_execution: PreparedStreamedRelation
    interaction_plan: MortonPlaneInteractionPlan
    schedule_plan: MortonPlaneSchedulePlan
    box_size: tuple[float, float, float] = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)


@final
class UniformFMMStructureTransition(StrictModule):
    """Atomic geometry update; a refused candidate never replaces acceptance."""

    candidate: PreparedUniformFMMStructure
    accepted: PreparedUniformFMMStructure
    successful: Array
    reused_topology: Array
    rebuilt: Array


class LaplaceMonopoleFieldResult(StrictModule):
    """Signed softened Laplace monopole potential and its true gradient.

    ``potential[i] = G * sum_{j != i, active} q[j] / sqrt(|x[i] - x[j]|**2 + eps**2)``
    and ``gradient[i]`` is its derivative with respect to ``x[i]``. Inactive
    rows are zero. ``successful`` is false when any schedule, traversal, or
    reduction capacity is exhausted, an active point lies outside the box, or
    the field is non-finite.
    """

    potential: Array
    gradient: Array
    evidence: CartesianFMMResourceEvidence
    successful: Array


class _UniformFMMField(StrictModule):
    """Logical-order field of one strength vector over a prepared structure."""

    gradient: Array
    potential: Array | None
    m2m_count: Array
    l2l_count: Array
    direct_count: Array
    finite: Array
    successful: Array


def _fmm_epilogue(
    parameters: tuple[()],
    receiver: tuple[Array, ...],
    aggregate: Array,
    /,
) -> Array:
    return aggregate


def _near_epilogue(
    parameters: tuple[()],
    receiver: tuple[Array, ...],
    aggregate: tuple[Array, Array],
    /,
) -> tuple[Array, Array]:
    return aggregate


def _broadcast_owner_rows(
    rows: Array,
    members: Array,
    owned: Array,
    capacity: int,
    /,
) -> Array:
    """Copy each owner row onto its distinct members, leaving others zero.

    ``members[i, k]`` with ``owned[i, k]`` names distinct members, so every
    member receives one nonzero contribution. The adjoint is a gather followed
    by a bounded per-owner sum, not a duplicate-index scatter of member
    cotangents into owners.
    """
    payload = (1,) * (rows.ndim - 1)
    material = jnp.where(
        owned.reshape(owned.shape + payload), rows[:, None], jnp.zeros((), rows.dtype)
    )
    return (
        jnp.zeros((capacity,) + rows.shape[1:], dtype=rows.dtype)
        .at[members]
        .add(material)
    )


@final
class _CartesianFarEvent(StrictModule):
    operators: CartesianFMMOperators

    def __call__(
        self,
        parameters: tuple[()],
        source: tuple[Array, Array, Array],
        target: tuple[Array, Array],
        edge: tuple[()],
        /,
    ) -> Array:
        moment, source_center, source_scale = source
        target_center, target_scale = target
        return self.operators.m2l(
            moment,
            source_center,
            target_center,
            source_scale=source_scale,
            target_scale=target_scale,
        )


@final
class _NearFragmentAggregation(StrictModule):
    softening: float = eqx.field(static=True)
    coefficient: float = eqx.field(static=True)
    short_range_scale: float | None = eqx.field(static=True)
    short_range_cutoff: float | None = eqx.field(static=True)
    backend: SpatialDistanceBackend = eqx.field(static=True)
    pallas_interpret: bool = eqx.field(static=True)

    def __call__(
        self,
        parameters: tuple[()],
        sources: tuple[Array, Array, Array, Array],
        receivers: tuple[Array, Array, Array],
        edges: tuple[()],
        fragment: StreamedFragment,
        seed: tuple[KeyGroupAccumulation, KeyGroupAccumulation],
        /,
    ) -> tuple[KeyGroupAccumulation, KeyGroupAccumulation]:
        field = spatial_leaf_pair_fragment(
            sources,
            receivers,
            fragment,
            seed[0],
            softening=self.softening,
            coefficient=self.coefficient,
            short_range_scale=self.short_range_scale,
            short_range_cutoff=self.short_range_cutoff,
            backend=self.backend,
            pallas_interpret=self.pallas_interpret,
        )
        source_position, _, source_mask, source_ids = sources
        target_position, target_mask, target_ids = receivers
        senders = fragment.lane_sources
        slots = fragment.lane_slots
        pair = (
            fragment.lane_use[:, None, None]
            & target_mask[slots, :, None]
            & source_mask[senders, None, :]
            & (target_ids[slots, :, None] != source_ids[senders, None, :])
        )
        if self.short_range_cutoff is not None:
            delta = (
                source_position[senders, None, :, :] - target_position[slots, :, None, :]
            )
            radius_squared = jnp.sum(delta * delta, axis=-1) + self.softening**2
            pair = pair & (jnp.sqrt(radius_squared) <= self.short_range_cutoff)
        counts = jnp.sum(pair, axis=(1, 2), dtype=jnp.int32)
        return field, fragment.reduce(counts, seed[1])


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
    execution_backend: SpatialDistanceBackend = eqx.field(static=True)
    pallas_interpret: bool = eqx.field(static=True)
    short_range_scale: float | None = eqx.field(static=True)
    short_range_cutoff: float | None = eqx.field(static=True)
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
        execution_backend: SpatialDistanceBackend = "jax",
        pallas_interpret: bool = False,
        short_range_scale: float | None = None,
        short_range_cutoff: float | None = None,
    ) -> None:
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
        split = None if short_range_scale is None else float(short_range_scale)
        cutoff = None if short_range_cutoff is None else float(short_range_cutoff)
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
            or (split is None) != (cutoff is None)
            or (
                split is not None
                and cutoff is not None
                and (
                    not np.isfinite(split)
                    or split <= 0.0
                    or not np.isfinite(cutoff)
                    or cutoff <= max(split, epsilon)
                )
            )
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
        self.short_range_scale = split
        self.short_range_cutoff = cutoff
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
                "short_range_scale": split,
                "short_range_cutoff": cutoff,
                "stream": {
                    "receiver_tile": 8,
                    "edge_tile": 32,
                    "replay": "step",
                    "capacity_bounded": True,
                    "near_message": "leaf-potential-gradient",
                },
            }
        )

    def prepare_structure(
        self,
        positions: ArrayLike,
        /,
        *,
        box_size: tuple[float, float, float],
        depth: int,
        active_mask: ArrayLike | None = None,
    ) -> PreparedUniformFMMStructure:
        """Prepare the position-dependent FMM topology reused across strengths.

        ``positions`` are box coordinates in ``[0, box_size)`` with origin
        ``(0, 0, 0)`` and ``depth`` is the Morton address depth. The compact
        plane schedule and receiver-major far/near streams are built once.
        Node, queue, far, and near capacity exhaustion and out-of-box points
        are reported through the evaluated result's evidence instead of raised.
        """
        position = jnp.asarray(positions)
        if position.ndim != 2 or position.shape[1] != 3 or position.shape[0] < 1:
            raise ValueError("FMM positions must have shape (N, 3) with N >= 1.")
        if position.dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
            raise TypeError("FMM positions must have float32 or float64 dtype.")
        active = (
            jnp.ones((position.shape[0],), dtype=jnp.bool_)
            if active_mask is None
            else jnp.asarray(active_mask, dtype=jnp.bool_)
        )
        if active.shape != (position.shape[0],):
            raise ValueError("FMM active_mask must have one flag per position.")
        if len(box_size) != 3:
            raise ValueError("FMM box_size must contain three lengths.")
        lengths = (float(box_size[0]), float(box_size[1]), float(box_size[2]))
        point_capacity = position.shape[0]
        schedule_plan = MortonPlaneSchedulePlan(
            MortonAddressPlan((0.0, 0.0, 0.0), lengths, depth),
            point_capacity,
            node_capacity=self.maximum_nodes,
            maximum_leaf_occupancy=self.maximum_leaf_occupancy,
            coarsening_factor=self.coarsening_factor,
            target_top_nodes=self.target_top_nodes,
        )
        leaf_capacity = schedule_plan.plane_capacities[0]
        maximum_leaf_pairs = max(leaf_capacity * leaf_capacity, 1)
        interaction_plan = MortonPlaneInteractionPlan(
            schedule_plan,
            opening_angle=self.opening_angle,
            queue_capacity=(
                maximum_leaf_pairs
                if self.maximum_queue_interactions is None
                else self.maximum_queue_interactions
            ),
            far_capacity=(
                maximum_leaf_pairs
                if self.maximum_far_interactions is None
                else self.maximum_far_interactions
            ),
            near_capacity=(
                maximum_leaf_pairs
                if self.maximum_near_interactions is None
                else self.maximum_near_interactions
            ),
            interaction_cutoff=(
                None
                if self.short_range_cutoff is None
                else float(np.sqrt(self.short_range_cutoff**2 - self.softening**2))
            ),
        )
        safe_position = jnp.where(active[:, None], position, 0.0)
        schedule = schedule_plan.build(
            safe_position,
            active_mask=active,
            stable_ids=jnp.arange(point_capacity, dtype=jnp.int64),
        )
        interactions = interaction_plan.build(schedule)
        far_execution, near_execution = self._prepare_streams(
            schedule_plan, interactions, epoch=schedule.epoch
        )
        return PreparedUniformFMMStructure(
            positions=safe_position,
            active_mask=active,
            schedule=schedule,
            interactions=interactions,
            far_execution=far_execution,
            near_execution=near_execution,
            interaction_plan=interaction_plan,
            schedule_plan=schedule_plan,
            box_size=lengths,
            depth=schedule_plan.address_plan.maximum_depth,
            plan_id=self.plan_id,
            structure_id=canonical_fingerprint(
                {
                    "kind": "prepared-uniform-fmm-structure",
                    "plan": self.plan_id,
                    "schedule_plan": schedule_plan.plan_id,
                    "interaction_plan": interaction_plan.plan_id,
                }
            ),
        )

    def _prepare_streams(
        self,
        schedule_plan: MortonPlaneSchedulePlan,
        interactions: MortonPlaneInteractionState,
        /,
        *,
        epoch: Array,
    ) -> tuple[PreparedStreamedRelation, PreparedStreamedRelation]:
        """Bind bounded leaf/node streams with host/JIT-compatible extents."""
        owner = canonical_fingerprint(
            {
                "kind": "uniform-fmm-stream-topology",
                "plan_id": self.plan_id,
                "schedule_plan_id": schedule_plan.plan_id,
            }
        )
        far = StreamedRelationPlan(
            receiver_tile=8,
            edge_tile=32,
            channel_capacity=self.expansion.coefficient_count,
            accumulation=self.accumulation,
        ).prepare(
            interactions.far,
            owner_id=owner,
            epoch=epoch,
            capacity_bounded=True,
        )
        leaf_capacity = schedule_plan.plane_capacities[0]
        near_relation = EdgeRelation(
            interactions.near.source_indices,
            interactions.near.target_indices,
            source_size=leaf_capacity,
            target_size=leaf_capacity,
            valid=interactions.near.valid,
        )
        near = StreamedRelationPlan(
            receiver_tile=8,
            edge_tile=32,
            channel_capacity=4 * schedule_plan.maximum_leaf_occupancy + 1,
            accumulation=self.accumulation,
        ).prepare(
            near_relation,
            owner_id=owner,
            epoch=epoch,
            capacity_bounded=True,
        )
        return far, near

    @checked
    def refresh_structure(
        self,
        previous_structure: PreparedUniformFMMStructure,
        positions: ArrayLike,
        /,
        *,
        box_size: tuple[float, float, float],
        depth: int,
        active_mask: ArrayLike | None = None,
    ) -> UniformFMMStructureTransition:
        """Refit admitted geometry or rebuild; reject updates atomically.

        Omitted activity retains the previous mask. Reuse requires the plane
        owner's unchanged membership/order certificate and current acceptance
        of every retained far pair. Moving cutoff geometry always rebuilds.
        """
        previous = previous_structure
        if previous.plan_id != self.plan_id:
            raise ValueError("The FMM structure belongs to a different plan.")
        if tuple(box_size) != previous.box_size or depth != previous.depth:
            raise ValueError("FMM refresh must retain its box and Morton depth.")
        position = jnp.asarray(positions)
        if position.shape != previous.positions.shape:
            raise ValueError("FMM refresh must retain particle capacity and dimension.")
        if position.dtype != previous.positions.dtype:
            raise TypeError("FMM refresh must retain its floating geometry dtype.")
        active = (
            previous.active_mask
            if active_mask is None
            else jnp.asarray(active_mask, dtype=jnp.bool_)
        )
        if active.shape != previous.active_mask.shape:
            raise ValueError("FMM active_mask must have one flag per position.")
        safe_position = jnp.where(active[:, None], position, 0.0)
        transition = previous.schedule_plan.refresh(
            previous.schedule,
            safe_position,
            active_mask=active,
            stable_ids=previous.schedule.point_order.stable_ids,
        )
        schedule = transition.candidate
        moved = jnp.any(safe_position != previous.positions)
        if self.short_range_cutoff is None:
            evidence = previous.interaction_plan.recheck(
                schedule, interactions=previous.interactions
            )
            reusable = transition.refitted & evidence.successful
        else:
            evidence = previous.interactions.evidence
            reusable = transition.refitted & ~moved & evidence.successful
        retained = MortonPlaneInteractionState(
            far=previous.interactions.far,
            near=previous.interactions.near,
            evidence=evidence,
        )

        def reuse(
            _: None,
        ) -> tuple[
            MortonPlaneInteractionState,
            PreparedStreamedRelation,
            PreparedStreamedRelation,
        ]:
            return retained, previous.far_execution, previous.near_execution

        def rebuild(
            _: None,
        ) -> tuple[
            MortonPlaneInteractionState,
            PreparedStreamedRelation,
            PreparedStreamedRelation,
        ]:
            interactions = previous.interaction_plan.build(schedule)
            far, near = self._prepare_streams(
                previous.schedule_plan, interactions, epoch=schedule.epoch
            )
            return interactions, far, near

        interactions, far, near = jax.lax.cond(reusable, reuse, rebuild, None)
        candidate = eqx.tree_at(
            lambda value: (
                value.positions,
                value.active_mask,
                value.schedule,
                value.interactions,
                value.far_execution,
                value.near_execution,
            ),
            previous,
            (safe_position, active, schedule, interactions, far, near),
        )
        successful = (
            transition.accepted_candidate
            & interactions.evidence.successful
            & far.schedule.successful
            & near.schedule.successful
        )
        accepted = jax.lax.cond(successful, lambda _: candidate, lambda _: previous, None)
        return UniformFMMStructureTransition(
            candidate=candidate,
            accepted=accepted,
            successful=successful,
            reused_topology=successful & reusable,
            rebuilt=~reusable,
        )

    def _upward_pass(
        self,
        structure: PreparedUniformFMMStructure,
        normalized_relative: Array,
        sorted_strength: Array,
        operators: CartesianFMMOperators,
        /,
    ) -> tuple[Array, Array]:
        """P2M into leaf nodes, then M2M up the compact planes."""
        schedule = structure.schedule
        schedule_plan = structure.schedule_plan
        node_capacity = schedule_plan.node_capacity
        if self.accumulation == "fast":
            safe_point_leaf = jnp.maximum(schedule.sorted_point_leaf_slots, 0)
            multipole = jnp.zeros(
                (node_capacity, self.expansion.coefficient_count),
                dtype=structure.positions.dtype,
            )
            for coefficient, exponent in enumerate(self.expansion.exponents):
                particle_coefficient = sorted_strength
                for axis in range(3):
                    particle_coefficient = (
                        particle_coefficient
                        * normalized_relative[:, axis] ** exponent[axis]
                    )
                multipole = multipole.at[safe_point_leaf, coefficient].add(
                    particle_coefficient
                )
        else:
            # Ordered P2M sums each leaf's storage row in slot order through the
            # shared sparse row reducer instead of a duplicate-index scatter.
            safe_storage, mask = self._ordered_leaf_rows(structure)
            retained = mask.shape[0]
            leaf_relative = normalized_relative[safe_storage]
            leaf_strength = jnp.where(mask, sorted_strength[safe_storage], 0.0)
            coefficients = []
            for exponent in self.expansion.exponents:
                particle_coefficient = leaf_strength
                for axis in range(3):
                    particle_coefficient = (
                        particle_coefficient * leaf_relative[..., axis] ** exponent[axis]
                    )
                coefficients.append(particle_coefficient)
            rows = RowRelation(
                safe_storage, source_size=schedule_plan.point_capacity, valid=mask
            )
            leaf_multipole = route_reduce(
                rows,
                jnp.stack(coefficients, axis=-1),
                accumulation=self.accumulation,
            )
            multipole = jnp.concatenate(
                (
                    leaf_multipole,
                    jnp.zeros(
                        (node_capacity - retained, leaf_multipole.shape[1]),
                        dtype=leaf_multipole.dtype,
                    ),
                )
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
        return multipole, m2m_count

    def _downward_pass(
        self,
        structure: PreparedUniformFMMStructure,
        multipole: Array,
        operators: CartesianFMMOperators,
        /,
    ) -> tuple[Array, StreamedRelationEvidence, Array]:
        """M2L over accepted far node pairs, then L2L down the compact planes."""
        schedule = structure.schedule
        shape = jax.ShapeDtypeStruct(
            (self.expansion.coefficient_count,), structure.positions.dtype
        )
        result = structure.far_execution.evaluate(
            StreamedPayloadSpec(shape, shape),
            _CartesianFarEvent(operators),
            _fmm_epilogue,
            (),
            (multipole, schedule.node_centers, schedule.node_scales),
            (schedule.node_centers, schedule.node_scales),
            (),
        )
        local, far_reduction = result.receiver_outputs, result.evidence

        l2l_count = jnp.asarray(0, dtype=jnp.int32)
        for plane in range(structure.schedule_plan.plane_count - 2, -1, -1):
            at_plane = schedule.node_active & (schedule.node_planes == plane)
            parents = jnp.maximum(schedule.node_parents, 0)
            parent_local = (
                local[parents]
                if self.accumulation == "fast"
                else self._ordered_parent_local(structure, local, plane)
            )
            inherited = jax.vmap(
                lambda value, shift, source_scale, target_scale: operators.l2l(
                    value,
                    shift,
                    source_scale=source_scale,
                    target_scale=target_scale,
                )
            )(
                parent_local,
                schedule.node_centers - schedule.node_centers[parents],
                schedule.node_scales[parents],
                schedule.node_scales,
            )
            local = local + jnp.where(at_plane[:, None], inherited, 0.0)
            l2l_count = l2l_count + jnp.sum(at_plane, dtype=jnp.int32)
        return local, far_reduction, l2l_count

    @staticmethod
    def _ordered_parent_local(
        structure: PreparedUniformFMMStructure,
        local: Array,
        plane: int,
        /,
    ) -> Array:
        """Parent local rows of plane nodes, broadcast from contiguous child ranges."""
        schedule = structure.schedule
        plan = structure.schedule_plan
        offsets = jnp.arange(plan.coarsening_factor, dtype=jnp.int32)
        children = jnp.clip(
            schedule.node_child_starts[:, None] + offsets[None, :],
            0,
            plan.node_capacity - 1,
        )
        owns = (schedule.node_active & (schedule.node_planes == plane + 1))[:, None] & (
            offsets[None, :] < schedule.node_child_counts[:, None]
        )
        return _broadcast_owner_rows(local, children, owns, plan.node_capacity)

    @staticmethod
    def _leaf_rows(
        structure: PreparedUniformFMMStructure,
        /,
    ) -> tuple[Array, Array]:
        """Leaf-major storage rows: clipped point slots and live-slot mask."""
        schedule = structure.schedule
        plan = structure.schedule_plan
        offsets = jnp.arange(plan.maximum_leaf_occupancy, dtype=jnp.int32)
        leaf_slots = jnp.arange(plan.plane_capacities[0], dtype=jnp.int32)
        leaf_nodes = jnp.minimum(leaf_slots, plan.node_capacity - 1)
        storage = schedule.node_item_starts[leaf_nodes, None] + offsets[None, :]
        safe_storage = jnp.clip(storage, 0, plan.point_capacity - 1)
        mask = (
            (leaf_slots[:, None] < plan.node_capacity)
            & schedule.node_active[leaf_nodes, None]
            & (offsets[None, :] < schedule.node_item_counts[leaf_nodes, None])
            & (storage < plan.point_capacity)
            & schedule.point_order.sorted_active[safe_storage]
        )
        return safe_storage, mask

    @classmethod
    def _ordered_leaf_rows(
        cls,
        structure: PreparedUniformFMMStructure,
        /,
    ) -> tuple[Array, Array]:
        """Leaf rows owning distinct points: plane-0 nodes below the node capacity.

        Low packed slots past the active leaves hold coarser nodes whose item
        ranges overlap their leaves, so they are masked here.
        """
        plan = structure.schedule_plan
        retained = min(plan.plane_capacities[0], plan.node_capacity)
        safe_storage, mask = cls._leaf_rows(structure)
        leaf = structure.schedule.node_planes[:retained, None] == 0
        return safe_storage[:retained], mask[:retained] & leaf

    def _point_local(
        self,
        structure: PreparedUniformFMMStructure,
        local: Array,
        /,
    ) -> Array:
        """Each storage point's leaf local expansion for L2P."""
        if self.accumulation == "fast":
            return local[jnp.maximum(structure.schedule.sorted_point_leaf_slots, 0)]
        safe_storage, mask = self._ordered_leaf_rows(structure)
        return _broadcast_owner_rows(
            local[: mask.shape[0]],
            safe_storage,
            mask,
            structure.schedule_plan.point_capacity,
        )

    def _near_field(
        self,
        structure: PreparedUniformFMMStructure,
        sorted_position: Array,
        sorted_strength: Array,
        /,
        *,
        with_potential: bool,
    ) -> tuple[Array, Array | None, StreamedRelationEvidence, Array]:
        """Complete exact P2P through bounded receiver-major leaf fragments."""
        schedule = structure.schedule
        plan = structure.schedule_plan
        width = plan.maximum_leaf_occupancy
        safe_storage, mask = self._leaf_rows(structure)
        leaf_position = jnp.where(mask[..., None], sorted_position[safe_storage], 0.0)
        leaf_strength = jnp.where(mask, sorted_strength[safe_storage], 0.0)
        field_shape = jax.ShapeDtypeStruct((width, 4), sorted_position.dtype)
        count_shape = jax.ShapeDtypeStruct((), jnp.int32)
        result = structure.near_execution.evaluate_fragments(
            StreamedPayloadSpec((field_shape, count_shape), (field_shape, count_shape)),
            _NearFragmentAggregation(
                self.softening,
                self.gravitational_constant,
                self.short_range_scale,
                self.short_range_cutoff,
                self.execution_backend,
                self.pallas_interpret,
            ),
            _near_epilogue,
            (),
            (leaf_position, leaf_strength, mask, safe_storage),
            (leaf_position, mask, safe_storage),
            (),
        )
        leaf_field, leaf_count = result.receiver_outputs
        point_leaf = jnp.maximum(schedule.sorted_point_leaf_slots, 0)
        point_offset = (
            jnp.arange(plan.point_capacity, dtype=jnp.int32)
            - schedule.node_item_starts[point_leaf]
        )
        point_field = leaf_field[point_leaf, jnp.clip(point_offset, 0, width - 1)]
        point_field = jnp.where(
            schedule.point_order.sorted_active[:, None], point_field, 0.0
        )
        return (
            point_field[:, :3],
            point_field[:, 3] if with_potential else None,
            result.evidence,
            jnp.sum(leaf_count, dtype=jnp.int32),
        )

    def _field(
        self,
        structure: PreparedUniformFMMStructure,
        strengths: Array,
        /,
        *,
        with_potential: bool,
    ) -> _UniformFMMField:
        """Run P2M/M2M/M2L/L2L/L2P/P2P for one strength vector."""
        operators = CartesianFMMOperators(
            self.expansion,
            self.gravitational_constant,
            self.softening,
            short_range_scale=self.short_range_scale,
            short_range_cutoff=self.short_range_cutoff,
        )
        schedule = structure.schedule
        sorted_logical = schedule.point_order.storage_to_logical
        sorted_position = schedule.point_order.sorted_coordinates
        sorted_active = schedule.point_order.sorted_active
        safe_point_leaf = jnp.maximum(schedule.sorted_point_leaf_slots, 0)
        point_scale = schedule.node_scales[safe_point_leaf]
        relative = sorted_position - schedule.node_centers[safe_point_leaf]
        sorted_strength = jnp.where(sorted_active, strengths[sorted_logical], 0.0)
        multipole, m2m_count = self._upward_pass(
            structure,
            relative / point_scale[:, None],
            sorted_strength,
            operators,
        )
        local, far_reduction, l2l_count = self._downward_pass(
            structure,
            multipole,
            operators,
        )
        point_local = self._point_local(structure, local)
        local_potential, local_gradient = jax.vmap(
            lambda value, offset, scale: operators.l2p(
                value,
                offset,
                scale=scale,
            )
        )(point_local, relative, point_scale)
        local_gradient = jnp.where(sorted_active[:, None], local_gradient, 0.0)
        near_gradient, near_potential, near_reduction, direct_count = self._near_field(
            structure,
            sorted_position,
            sorted_strength,
            with_potential=with_potential,
        )
        active = structure.active_mask
        gradient = (
            jnp.zeros_like(structure.positions)
            .at[sorted_logical]
            .set(local_gradient + near_gradient)
        )
        gradient = jnp.where(active[:, None], gradient, 0.0)
        finite = (
            jnp.all(jnp.isfinite(gradient))
            & jnp.all(jnp.where(active, jnp.isfinite(strengths), True))
            & far_reduction.finite
            & near_reduction.finite
        )
        potential = None
        if near_potential is not None:
            # Locals expand the gravitational potential -G/r, so the far
            # monopole potential is the negated L2P value.
            sorted_potential = (
                jnp.where(sorted_active, -local_potential, 0.0) + near_potential
            )
            potential = (
                jnp.zeros(active.shape, dtype=structure.positions.dtype)
                .at[sorted_logical]
                .set(sorted_potential)
            )
            potential = jnp.where(active, potential, 0.0)
            finite = finite & jnp.all(jnp.isfinite(potential))
        successful = (
            schedule.evidence.successful
            & structure.interactions.evidence.successful
            & far_reduction.successful
            & near_reduction.successful
            & finite
        )
        link = _input_link(
            (structure.positions, strengths),
            (gradient.dtype,) if potential is None else (gradient.dtype, potential.dtype),
        )
        gradient = _poison(gradient, successful, link)
        if potential is not None:
            potential = _poison(potential, successful, link)
        finite = finite & successful
        return _UniformFMMField(
            gradient=gradient,
            potential=potential,
            m2m_count=m2m_count,
            l2l_count=l2l_count,
            direct_count=direct_count,
            finite=finite,
            successful=successful,
        )

    def _resource_evidence(
        self,
        structure: PreparedUniformFMMStructure,
        field: _UniformFMMField,
        /,
    ) -> CartesianFMMResourceEvidence:
        schedule = structure.schedule.evidence
        interactions = structure.interactions.evidence
        return CartesianFMMResourceEvidence(
            expansion_order=jnp.asarray(self.expansion.order, dtype=jnp.int32),
            opening_angle=jnp.asarray(
                self.opening_angle, dtype=structure.positions.dtype
            ),
            required_nodes=schedule.required_nodes,
            node_capacity=schedule.node_capacity,
            required_queue=interactions.required_queue,
            queue_capacity=interactions.queue_capacity,
            required_far=interactions.required_far,
            far_capacity=interactions.far_capacity,
            required_near=interactions.required_near,
            near_capacity=interactions.near_capacity,
            minimum_scale_exponent=schedule.minimum_scale_exponent,
            maximum_scale_exponent=schedule.maximum_scale_exponent,
            p2m_count=schedule.active_points,
            m2m_count=field.m2m_count,
            m2l_count=interactions.required_far,
            l2l_count=field.l2l_count,
            l2p_count=schedule.active_points,
            p2p_count=field.direct_count,
            successful=field.successful,
            accumulation=self.accumulation,
        )

    def _evaluate_impl(self, tree: PreparedParticleOctree3D, /) -> TreeGravityResult:
        structure = self.prepare_structure(
            tree.positions,
            box_size=tree.box_size,
            depth=tree.depth,
            active_mask=tree.active_mask,
        )
        field = self._field(structure, tree.masses, with_potential=False)
        acceleration = field.gradient
        interactions = structure.interactions.evidence
        evidence = TreeGravityEvidence(
            net_force=jnp.sum(
                jnp.where(tree.active_mask, tree.masses, 0.0)[:, None] * acceleration,
                axis=0,
            ),
            maximum_acceleration=jnp.max(
                jnp.sqrt(jnp.sum(acceleration**2, axis=-1)),
                initial=0.0,
            ),
            accepted_leaf_interactions=interactions.required_far,
            direct_particle_interactions=field.direct_count,
            maximum_opening_indicator=interactions.maximum_accepted_ratio,
            traversal_complete=interactions.complete,
            active_nodes=structure.schedule.evidence.active_nodes,
            finite=field.finite,
            successful=field.successful,
        )
        return TreeGravityResult(
            acceleration,
            evidence,
            field.successful,
            self._resource_evidence(structure, field),
        )

    def evaluate(self, tree: PreparedParticleOctree3D, /) -> TreeGravityResult:
        """Evaluate with a whole-FMM rematerializing reverse rule."""
        plan = self

        @jax.custom_vjp
        def run(current_tree: PreparedParticleOctree3D) -> TreeGravityResult:
            return plan._evaluate_impl(current_tree)

        def forward(
            current_tree: PreparedParticleOctree3D,
        ) -> tuple[TreeGravityResult, PreparedParticleOctree3D]:
            result = plan._evaluate_impl(current_tree)
            return result, current_tree

        def backward(
            current_tree: PreparedParticleOctree3D, cotangent: TreeGravityResult
        ) -> tuple[PreparedParticleOctree3D]:
            _, pullback = jax.vjp(
                lambda value: plan._evaluate_impl(value).acceleration,
                current_tree,
            )
            return (pullback(cotangent.acceleration)[0],)

        run.defvjp(forward, backward)
        return run(tree)

    @checked
    def evaluate_monopole(
        self,
        structure: PreparedUniformFMMStructure,
        strengths: ArrayLike,
        /,
    ) -> LaplaceMonopoleFieldResult:
        """Evaluate the signed softened Laplace monopole field on a prepared structure.

        ``potential[i] = G * sum_{j != i} q[j] / sqrt(|x[i] - x[j]|**2 + eps**2)``
        over active sources, with ``G = gravitational_constant`` and
        ``eps = softening``. ``gradient[i]`` is the true derivative of
        ``potential[i]`` with respect to ``x[i]``,
        ``G * sum_j q[j] * (x[j] - x[i]) / (|x[i] - x[j]|**2 + eps**2)**1.5``,
        which is the acceleration ``evaluate`` returns for masses equal to the
        strengths. Strengths are signed and are the only per-call input, so
        one prepared structure serves every strength refresh. Both near
        potential and gradient use the selected bounded fragment backend;
        Pallas derivatives execute the same fragment's JAX reference.
        """
        if self.short_range_scale is not None:
            raise ValueError(
                "The Laplace monopole route requires a UniformFMMPlan without "
                "short_range_scale."
            )
        if structure.plan_id != self.plan_id:
            raise ValueError(
                "The FMM structure was prepared by a different UniformFMMPlan."
            )
        strength = jnp.asarray(strengths, dtype=structure.positions.dtype)
        if strength.shape != structure.active_mask.shape:
            raise ValueError(
                "Monopole strengths must have one value per prepared position."
            )
        field = self._field(structure, strength, with_potential=True)
        if field.potential is None:
            raise RuntimeError("The FMM monopole field did not produce a potential.")
        return LaplaceMonopoleFieldResult(
            potential=field.potential,
            gradient=field.gradient,
            evidence=self._resource_evidence(structure, field),
            successful=field.successful,
        )


class PeriodicEwaldEvidence(StrictModule):
    real_space_acceleration: Array
    reciprocal_acceleration: Array
    net_force: Array
    required_real_pairs: Array
    real_pair_capacity: Array
    real_pair_overflow: Array
    real_cutoff: Array
    finite: Array
    successful: Array
    real_space_execution: Literal["direct_shells", "screened_radius"] = eqx.field(
        static=True
    )


class PeriodicEwaldResult(StrictModule):
    acceleration: Array
    evidence: PeriodicEwaldEvidence
    successful: Array


class PeriodicEwaldForcePlan(StrictModule, NonTrainableState):
    """Periodic Ewald force with explicit finite-shell real-space execution."""

    box_size: tuple[float, ...] = eqx.field(static=True)
    gravitational_constant: float = eqx.field(static=True)
    softening: float = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    real_offsets: Array
    wavevectors: Array
    volume: float = eqx.field(static=True)
    real_space_execution: Literal["direct_shells", "screened_radius"] = eqx.field(
        static=True
    )
    real_cutoff: float | None = eqx.field(static=True)
    maximum_real_pairs: int | None = eqx.field(static=True)
    real_address_lower: tuple[float, ...] = eqx.field(static=True)
    real_address_upper: tuple[float, ...] = eqx.field(static=True)
    zero_offset_index: int = eqx.field(static=True)
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
        real_space_execution: Literal["direct_shells", "screened_radius"] = (
            "direct_shells"
        ),
        real_cutoff: float | None = None,
        maximum_real_pairs: int | None = None,
    ) -> None:
        lengths = tuple(float(value) for value in box_size)
        gravity = float(gravitational_constant)
        epsilon = float(softening)
        alpha_ = float(alpha)
        real = int(real_shells)
        reciprocal = int(reciprocal_modes)
        cutoff = None if real_cutoff is None else float(real_cutoff)
        pair_capacity = None if maximum_real_pairs is None else int(maximum_real_pairs)
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
            or real_space_execution not in ("direct_shells", "screened_radius")
            or (
                real_space_execution == "screened_radius"
                and (cutoff is None or not np.isfinite(cutoff) or cutoff <= 0.0)
            )
            or (
                real_space_execution == "direct_shells"
                and (cutoff is not None or pair_capacity is not None)
            )
            or (pair_capacity is not None and pair_capacity <= 0)
        ):
            raise ValueError("Periodic Ewald policy is invalid.")
        dimension = len(lengths)
        integer_offsets = np.asarray(
            tuple(product(range(-real, real + 1), repeat=dimension)),
            dtype=np.float64,
        )
        offset_vectors = integer_offsets * np.asarray(lengths)[None, :]
        reciprocal_indices = np.asarray(
            tuple(
                index
                for index in product(range(-reciprocal, reciprocal + 1), repeat=dimension)
                if any(value != 0 for value in index)
            ),
            dtype=np.float64,
        )
        wavevectors = 2.0 * np.pi * reciprocal_indices / np.asarray(lengths)[None, :]
        lower = np.min(offset_vectors, axis=0)
        upper = np.max(offset_vectors, axis=0) + np.asarray(lengths)
        zero_index = int(np.nonzero(np.all(integer_offsets == 0.0, axis=1))[0][0])
        self.box_size = lengths
        self.gravitational_constant = gravity
        self.softening = epsilon
        self.alpha = alpha_
        self.real_offsets = jnp.asarray(offset_vectors)
        self.wavevectors = jnp.asarray(wavevectors)
        self.volume = float(np.prod(lengths))
        self.real_space_execution = real_space_execution
        self.real_cutoff = cutoff
        self.maximum_real_pairs = pair_capacity
        self.real_address_lower = tuple(float(value) for value in lower)
        self.real_address_upper = tuple(float(value) for value in upper)
        self.zero_offset_index = zero_index
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-ewald-force",
                "box_size": list(lengths),
                "gravitational_constant": gravity,
                "softening": epsilon,
                "alpha": alpha_,
                "real_shells": real,
                "reciprocal_modes": reciprocal,
                "real_space_execution": real_space_execution,
                "real_cutoff": cutoff,
                "maximum_real_pairs": pair_capacity,
            }
        )

    def _screening(self, distance: Array, /) -> Array:
        return jax.scipy.special.erfc(self.alpha * distance) + (
            2.0
            * self.alpha
            * distance
            / jnp.sqrt(jnp.pi)
            * jnp.exp(-((self.alpha * distance) ** 2))
        )

    def _direct_real_space(
        self,
        position: Array,
        mass: Array,
        active: Array,
    ) -> tuple[Array, Array, Array, Array]:
        target = position[:, None, None, :]
        source = position[None, :, None, :] + self.real_offsets[None, None, :, :]
        displacement = source - target
        distance_squared = jnp.sum(displacement**2, axis=-1) + self.softening**2
        distance = jnp.sqrt(distance_squared)
        zero_offset = (
            jnp.arange(self.real_offsets.shape[0], dtype=jnp.int32)
            == self.zero_offset_index
        )
        active_pair = active[:, None, None] & active[None, :, None]
        self_pair = (
            jnp.eye(position.shape[0], dtype=jnp.bool_)[:, :, None]
            & zero_offset[None, None, :]
        )
        excluded = ~active_pair | self_pair
        inverse_cube = jnp.where(
            excluded,
            0.0,
            self._screening(distance) / distance**3,
        )
        acceleration = jnp.sum(
            self.gravitational_constant
            * mass[None, :, None, None]
            * displacement
            * inverse_cube[..., None],
            axis=(1, 2),
        )
        required = jnp.sum(~excluded, dtype=jnp.int32)
        active_count = jnp.sum(active, dtype=jnp.int32)
        capacity = (active_count * active_count * self.real_offsets.shape[0]).astype(
            jnp.int32
        )
        return acceleration, required, capacity, jnp.asarray(False)

    def _screened_radius_real_space(
        self,
        position: Array,
        mass: Array,
        active: Array,
    ) -> tuple[Array, Array, Array, Array]:
        real_cutoff = self.real_cutoff
        # __init__ requires real_cutoff for screened_radius execution.
        if not (real_cutoff is not None):
            raise RuntimeError("Internal invariant failed: real_cutoff is not None.")
        count = position.shape[0]
        capacity = (
            count * count * self.real_offsets.shape[0]
            if self.maximum_real_pairs is None
            else self.maximum_real_pairs
        )
        per_offset_capacity = min(capacity, count * count)
        source_plan = MortonSourcePlan(
            MortonAddressPlan(
                self.real_address_lower,
                self.real_address_upper,
                21,
            ),
            count,
        )
        relation_plan = MortonRadiusRelationPlan(
            source_plan,
            count,
            per_offset_capacity,
            inclusive=True,
        )
        lengths = jnp.asarray(self.box_size, dtype=position.dtype)
        wrapped = jnp.mod(position, lengths)
        identifiers = jnp.arange(count, dtype=jnp.int64)
        acceleration = jnp.zeros_like(position)
        required = jnp.asarray(0, dtype=jnp.int32)
        route_success = jnp.asarray(True)
        for offset_index in range(self.real_offsets.shape[0]):
            offset = self.real_offsets[offset_index].astype(position.dtype)
            sources = source_plan.prepare(
                wrapped + offset, active_mask=active, stable_ids=identifiers
            )
            relation_result = relation_plan.query(
                sources,
                wrapped,
                real_cutoff,
                target_mask=active,
                target_stable_ids=identifiers,
                exclude_self=offset_index == self.zero_offset_index,
            )
            relation = relation_result.relation
            source_indices = relation.source_indices
            target_indices = relation.target_indices
            displacement = wrapped[source_indices] + offset - wrapped[target_indices]
            distance = jnp.sqrt(
                jnp.sum(displacement * displacement, axis=-1) + self.softening**2
            )
            contribution = (
                self.gravitational_constant
                * mass[source_indices, None]
                * displacement
                * (self._screening(distance) / distance**3)[:, None]
            )
            acceleration = acceleration.at[target_indices].add(
                jnp.where(relation.valid[:, None], contribution, 0.0)
            )
            required = required + relation_result.evidence.required_pairs
            route_success = route_success & relation_result.evidence.successful
        overflow = (required > capacity) | ~route_success
        acceleration = jnp.where(overflow, jnp.zeros_like(acceleration), acceleration)
        return (
            acceleration,
            required,
            jnp.asarray(capacity, dtype=jnp.int32),
            overflow,
        )

    def evaluate(
        self,
        positions: ArrayLike,
        masses: ArrayLike,
        /,
        *,
        active_mask: ArrayLike | None = None,
    ) -> PeriodicEwaldResult:
        position = jnp.asarray(positions)
        mass = jnp.asarray(masses, dtype=position.dtype)
        if (
            position.ndim != 2
            or position.shape[1] != len(self.box_size)
            or mass.shape != (position.shape[0],)
        ):
            raise ValueError("Periodic Ewald positions/masses have incompatible shapes.")
        active = (
            jnp.ones(mass.shape, dtype=jnp.bool_)
            if active_mask is None
            else jnp.asarray(active_mask, dtype=jnp.bool_)
        )
        if active.shape != mass.shape:
            raise ValueError("Periodic Ewald active_mask must match masses.")
        invalid = jnp.any(
            jnp.where(active[:, None], ~jnp.isfinite(position), False)
        ) | jnp.any(jnp.where(active, ~jnp.isfinite(mass) | (mass <= 0.0), False))
        position = eqx.error_if(
            position,
            invalid,
            "Active periodic Ewald inputs must be finite with positive masses.",
        )
        position = jnp.where(active[:, None], position, 0.0)
        mass = jnp.where(active, mass, 0.0)
        if self.real_space_execution == "direct_shells":
            (
                real_acceleration,
                required_pairs,
                pair_capacity,
                pair_overflow,
            ) = self._direct_real_space(position, mass, active)
            reciprocal_position = position
            real_cutoff = jnp.asarray(jnp.inf, dtype=position.dtype)
        else:
            (
                real_acceleration,
                required_pairs,
                pair_capacity,
                pair_overflow,
            ) = self._screened_radius_real_space(position, mass, active)
            reciprocal_position = jnp.mod(
                position,
                jnp.asarray(self.box_size, dtype=position.dtype),
            )
            real_cutoff = jnp.asarray(self.real_cutoff, dtype=position.dtype)
        k = self.wavevectors.astype(position.dtype)
        k_squared = jnp.sum(k**2, axis=-1)
        source_phase = contract("kd,nd->kn", k, reciprocal_position)
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
        reciprocal_acceleration = contract(
            "k,nk,kd->nd",
            coefficient,
            real_product,
            k,
        )
        reciprocal_acceleration = jnp.where(active[:, None], reciprocal_acceleration, 0.0)
        acceleration = real_acceleration + reciprocal_acceleration
        acceleration = jnp.where(
            pair_overflow,
            jnp.zeros_like(acceleration),
            acceleration,
        )
        net_force = jnp.sum(mass[:, None] * acceleration, axis=0)
        finite = jnp.all(jnp.isfinite(acceleration))
        successful = finite & ~pair_overflow
        evidence = PeriodicEwaldEvidence(
            real_acceleration,
            reciprocal_acceleration,
            net_force,
            required_pairs,
            pair_capacity,
            pair_overflow,
            real_cutoff,
            finite,
            successful,
            self.real_space_execution,
        )
        return PeriodicEwaldResult(acceleration, evidence, successful)


class PeriodicBarnesHutPlan(StrictModule, NonTrainableState):
    """Barnes-Hut plus an exact small-N Ewald-minus-direct periodic correction."""

    barnes_hut: BarnesHutGravityPlan
    ewald: Any

    def __init__(self, barnes_hut: BarnesHutGravityPlan, ewald: Any, /) -> None:
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
                (
                    tree.active_mask[:, None]
                    & tree.active_mask[None, :]
                    & ~jnp.eye(position.shape[0], dtype=jnp.bool_)
                )[..., None],
                self.barnes_hut.gravitational_constant
                * tree.masses[None, :, None]
                * displacement
                / squared[..., None] ** 1.5,
                0.0,
            ),
            axis=1,
        )
        periodic = self.ewald.evaluate(
            position, tree.masses, active_mask=tree.active_mask
        )
        acceleration = jnp.where(
            tree.active_mask[:, None],
            approximate.acceleration + periodic.acceleration - direct,
            0.0,
        )
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

    def __init__(self, tolerance: float, /) -> None:
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

    def __init__(
        self, split_scale: float, cutoff: float, compensation_id: str, /
    ) -> None:
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
    """Calibrated mesh complement with Barnes--Hut or Cartesian FMM short range."""

    short_range: BarnesHutGravityPlan | UniformFMMPlan
    split: TreePMSplitPolicy
    short_range_kernel: TreePMShortRangeKernel
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        short_range: BarnesHutGravityPlan | UniformFMMPlan,
        split: TreePMSplitPolicy,
        /,
    ) -> None:
        if not isinstance(short_range, (BarnesHutGravityPlan, UniformFMMPlan)):
            raise TypeError(
                "short_range must be a BarnesHutGravityPlan or UniformFMMPlan."
            )
        kernel = TreePMShortRangeKernel(
            short_range.gravitational_constant,
            short_range.softening,
            split.split_scale,
            split.cutoff,
        )
        if isinstance(short_range, UniformFMMPlan) and (
            short_range.short_range_scale != split.split_scale
            or short_range.short_range_cutoff != split.cutoff
        ):
            raise ValueError(
                "UniformFMMPlan must be prepared with this TreePM split scale and cutoff."
            )
        self.short_range = short_range
        self.split = split
        self.short_range_kernel = kernel
        self.plan_id = canonical_fingerprint(
            {
                "kind": "single-device-treepm",
                "short_range": short_range.plan_id,
                "split": split.policy_id,
                "kernel": kernel.kernel_id,
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
        if isinstance(self.short_range, BarnesHutGravityPlan):
            short = self.short_range.evaluate(
                tree,
                short_range_scale=self.split.split_scale,
                cutoff=self.split.cutoff,
            )
        else:
            short = self.short_range.evaluate(tree)
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
    "LaplaceMonopoleFieldResult",
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
    "PreparedUniformFMMStructure",
    "TreeGravityEvidence",
    "TreeGravityResult",
    "TreePMPlan",
    "TreePMResult",
    "TreePMSplitPolicy",
    "TreePMShortRangeKernel",
    "UniformFMMPlan",
    "UniformFMMStructureTransition",
]
