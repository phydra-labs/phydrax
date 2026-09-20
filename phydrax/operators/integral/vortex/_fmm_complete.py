#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ....discretization.spatial import MortonAddressPlan, SparseLevelOctreePlan
from ....discretization.spatial._plane_interactions import (
    MortonPlaneInteractionPlan,
    MortonPlaneInteractionState,
)
from ....discretization.spatial._plane_schedule import (
    MortonPlaneSchedulePlan,
    MortonPlaneScheduleState,
)
from ....discretization.vortex._capabilities import VortexVelocityCapabilities
from ....discretization.vortex._compatibility import (
    request_fields,
    validate_vortex_velocity_evaluation,
    VortexVelocityCompatibility,
)
from ....discretization.vortex._interfaces import (
    AbstractPreparedVortexVelocity,
    AbstractVortexVelocityPlan,
    DEFAULT_VORTEX_FIELD_REQUEST,
    VortexFieldRequest,
    VortexVelocityDiagnostics,
    VortexVelocityEvaluation,
)
from ....discretization.vortex._precision import VortexPrecisionPolicy
from ....discretization.vortex._source import VortexSourceState, VortexTargetState
from ....sparse import EdgeRelation, RelationExecutionPlan
from ._gaussian2d import gaussian_vortex_kernel_2d
from ._gaussian3d import GaussianErfVortexKernel3D


VortexFMMExecution = Literal["level_octree", "plane_dual"]


class VortexFMMEvidence(StrictModule):
    p2m_count: Array
    m2m_count: Array
    m2l_count: Array
    l2l_count: Array
    near_pair_count: Array
    expansion_order: int = eqx.field(static=True)
    geometric_tail_bound: Array
    maximum_reference_displacement: Array
    stale_topology: Array
    source_overflow: Array
    finite: Array
    execution: VortexFMMExecution = eqx.field(static=True)


class VortexFMMPlan(AbstractVortexVelocityPlan):
    """Reference-envelope vortex FMM with level-octree or plane execution."""

    reference_position: Array
    reference_target: Array
    lower: tuple[float, ...] = eqx.field(static=True)
    upper: tuple[float, ...] = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    expansion_order: int = eqx.field(static=True)
    leaf_capacity: int = eqx.field(static=True)
    target_leaf_capacity: int = eqx.field(static=True)
    maximum_reference_displacement: float = eqx.field(static=True)
    source_capacity: int = eqx.field(static=True)
    target_reference_capacity: int = eqx.field(static=True)
    reference_target_topology: Literal["same-support", "arbitrary-targets"] = eqx.field(
        static=True
    )
    dimension: int = eqx.field(static=True)
    execution: VortexFMMExecution = eqx.field(static=True)
    plane_coarsening_factor: int = eqx.field(static=True)
    plane_target_top_nodes: int = eqx.field(static=True)
    plane_opening_angle: float = eqx.field(static=True)
    maximum_queue_interactions: int | None = eqx.field(static=True)
    maximum_far_interactions: int | None = eqx.field(static=True)
    maximum_near_interactions: int | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    capabilities: VortexVelocityCapabilities

    def __init__(
        self,
        reference_position: ArrayLike,
        lower: ArrayLike,
        upper: ArrayLike,
        /,
        *,
        reference_targets: ArrayLike | None = None,
        depth: int = 3,
        expansion_order: int = 1,
        leaf_capacity: int = 64,
        target_leaf_capacity: int | None = None,
        maximum_reference_displacement: float = 0.05,
        execution: VortexFMMExecution = "level_octree",
        plane_coarsening_factor: int = 8,
        plane_target_top_nodes: int = 32,
        plane_opening_angle: float = 0.6,
        maximum_queue_interactions: int | None = None,
        maximum_far_interactions: int | None = None,
        maximum_near_interactions: int | None = None,
        precision: VortexPrecisionPolicy | None = None,
    ):
        reference = np.asarray(reference_position, dtype=np.float64)
        lower_array = np.asarray(lower, dtype=np.float64)
        upper_array = np.asarray(upper, dtype=np.float64)
        dimension = reference.shape[1] if reference.ndim == 2 else -1
        targets = (
            reference
            if reference_targets is None
            else np.asarray(
                reference_targets,
                dtype=np.float64,
            )
        )
        target_topology = (
            "same-support" if reference_targets is None else "arbitrary-targets"
        )
        depth_value = int(depth)
        order = int(expansion_order)
        source_leaf = int(leaf_capacity)
        target_leaf = (
            source_leaf if target_leaf_capacity is None else int(target_leaf_capacity)
        )
        displacement = float(maximum_reference_displacement)
        coarse = int(plane_coarsening_factor)
        top_nodes = int(plane_target_top_nodes)
        opening = float(plane_opening_angle)
        queue = (
            None
            if maximum_queue_interactions is None
            else int(maximum_queue_interactions)
        )
        far = None if maximum_far_interactions is None else int(maximum_far_interactions)
        near = (
            None if maximum_near_interactions is None else int(maximum_near_interactions)
        )
        if (
            reference.ndim != 2
            or reference.shape[0] == 0
            or dimension not in (2, 3)
            or targets.ndim != 2
            or targets.shape[0] == 0
            or targets.shape[1] != dimension
            or lower_array.shape != (dimension,)
            or upper_array.shape != lower_array.shape
            or np.any(~np.isfinite(reference))
            or np.any(~np.isfinite(targets))
            or np.any(~np.isfinite(lower_array))
            or np.any(~np.isfinite(upper_array))
            or np.any(upper_array <= lower_array)
            or np.any(reference < lower_array)
            or np.any(reference >= upper_array)
            or np.any(targets < lower_array)
            or np.any(targets >= upper_array)
            or depth_value < 1
            or order not in (0, 1)
            or source_leaf <= 0
            or target_leaf <= 0
            or not np.isfinite(displacement)
            or displacement <= 0.0
            or execution not in ("level_octree", "plane_dual")
            or coarse < 2
            or top_nodes <= 0
            or (queue is not None and queue <= 0)
            or (far is not None and far <= 0)
            or (near is not None and near <= 0)
            or (target_topology == "same-support" and source_leaf != target_leaf)
            or not 0.0 < opening < 1.0
        ):
            raise ValueError(
                "Vortex FMM geometry, execution, or capacity controls are invalid."
            )
        lower_tuple = tuple(float(value) for value in lower_array)
        upper_tuple = tuple(float(value) for value in upper_array)
        precision_value = VortexPrecisionPolicy() if precision is None else precision
        self.reference_position = jnp.asarray(reference)
        self.reference_target = jnp.asarray(targets)
        self.lower = lower_tuple
        self.upper = upper_tuple
        self.depth = depth_value
        self.expansion_order = order
        self.leaf_capacity = source_leaf
        self.target_leaf_capacity = target_leaf
        self.maximum_reference_displacement = displacement
        self.source_capacity = reference.shape[0]
        self.target_reference_capacity = targets.shape[0]
        self.reference_target_topology = target_topology
        self.dimension = dimension
        self.execution = execution
        self.plane_coarsening_factor = coarse
        self.plane_target_top_nodes = top_nodes
        self.plane_opening_angle = opening
        self.maximum_queue_interactions = queue
        self.maximum_far_interactions = far
        self.maximum_near_interactions = near
        self.capabilities = VortexVelocityCapabilities(
            dimension,
            required_source_fields=(
                "positions",
                "strength",
                "active_mask",
                "core_radius",
            ),
            supported_fields=("velocity", "velocity_gradient", "vorticity"),
            domain="free-space",
            precision=precision_value,
            derivatives=(
                "source-position",
                "source-strength",
                "source-core-radius",
                "target-position",
            ),
            target_topologies=("same-support", "arbitrary-targets"),
            acceleration="fmm",
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "sparse-vortex-fmm-plan",
                "reference": array_tree_fingerprint(reference),
                "reference_targets": array_tree_fingerprint(targets),
                "lower": list(lower_tuple),
                "upper": list(upper_tuple),
                "depth": depth_value,
                "order": order,
                "source_leaf_capacity": source_leaf,
                "target_leaf_capacity": target_leaf,
                "maximum_reference_displacement": displacement,
                "execution": execution,
                "plane_coarsening_factor": coarse,
                "plane_target_top_nodes": top_nodes,
                "plane_opening_angle": opening,
                "maximum_queue_interactions": queue,
                "maximum_far_interactions": far,
                "maximum_near_interactions": near,
            }
        )

    def prepare(
        self,
        /,
        *,
        source_capacity: int,
        target_capacity: int | None = None,
        source_kind: str = "particle",
        target_topology: str = "same-support",
        request: VortexFieldRequest = DEFAULT_VORTEX_FIELD_REQUEST,
    ) -> PreparedVortexFMM:
        targets = (
            int(source_capacity) if target_capacity is None else int(target_capacity)
        )
        if int(source_capacity) != self.source_capacity:
            raise ValueError("Vortex FMM source capacity differs from reference tree.")
        if self.execution == "plane_dual" and (
            targets != self.target_reference_capacity
            or target_topology != self.reference_target_topology
        ):
            raise ValueError(
                "Plane vortex FMM target capacity/topology differs from its reference envelope."
            )
        compatibility = VortexVelocityCompatibility(
            self.capabilities,
            source_capacity=self.source_capacity,
            target_capacity=targets,
            source_kind=source_kind,
            target_topology=target_topology,
            requested_fields=request_fields(request),
        )
        return PreparedVortexFMM(self, compatibility)


class PreparedVortexFMM(AbstractPreparedVortexVelocity):
    plan: VortexFMMPlan
    compatibility: VortexVelocityCompatibility
    dimension: int = eqx.field(static=True)
    source_capacity: int = eqx.field(static=True)
    target_capacity: int = eqx.field(static=True)
    backend_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    capabilities: VortexVelocityCapabilities
    source_plane_plan: MortonPlaneSchedulePlan | None = eqx.field(static=True)
    target_plane_plan: MortonPlaneSchedulePlan | None = eqx.field(static=True)
    source_plane: MortonPlaneScheduleState | None
    target_plane: MortonPlaneScheduleState | None
    plane_interactions: MortonPlaneInteractionState | None

    def __init__(
        self,
        plan: VortexFMMPlan,
        compatibility: VortexVelocityCompatibility,
        /,
    ):
        self.plan = plan
        self.compatibility = compatibility
        self.dimension = plan.dimension
        self.source_capacity = compatibility.source_capacity
        self.target_capacity = compatibility.target_capacity
        self.backend_id = plan.plan_id
        self.capabilities = plan.capabilities
        source_plane_plan = None
        target_plane_plan = None
        source_plane = None
        target_plane = None
        plane_interactions = None
        if plan.execution == "plane_dual":
            address = MortonAddressPlan(plan.lower, plan.upper, plan.depth)
            source_plane_plan = MortonPlaneSchedulePlan(
                address,
                self.source_capacity,
                maximum_leaf_occupancy=plan.leaf_capacity,
                coarsening_factor=plan.plane_coarsening_factor,
                target_top_nodes=plan.plane_target_top_nodes,
            )
            source_plane = source_plane_plan.build(
                plan.reference_position,
                bounding_padding=plan.maximum_reference_displacement,
                stable_ids=jnp.arange(self.source_capacity, dtype=jnp.int64),
            )
            if plan.reference_target_topology == "same-support":
                target_plane_plan = source_plane_plan
                target_plane = source_plane
            else:
                target_plane_plan = MortonPlaneSchedulePlan(
                    address,
                    self.target_capacity,
                    maximum_leaf_occupancy=plan.target_leaf_capacity,
                    coarsening_factor=plan.plane_coarsening_factor,
                    target_top_nodes=plan.plane_target_top_nodes,
                )
                target_plane = target_plane_plan.build(
                    plan.reference_target,
                    bounding_padding=plan.maximum_reference_displacement,
                    stable_ids=jnp.arange(self.target_capacity, dtype=jnp.int64),
                )
            source_leaves = (
                self.source_capacity + plan.leaf_capacity - 1
            ) // plan.leaf_capacity
            target_leaves = (
                self.target_capacity + plan.target_leaf_capacity - 1
            ) // plan.target_leaf_capacity
            route_capacity = max(source_leaves * target_leaves, 1)
            queue_capacity = (
                route_capacity
                if plan.maximum_queue_interactions is None
                else plan.maximum_queue_interactions
            )
            far_capacity = (
                route_capacity
                if plan.maximum_far_interactions is None
                else plan.maximum_far_interactions
            )
            near_capacity = (
                route_capacity
                if plan.maximum_near_interactions is None
                else plan.maximum_near_interactions
            )
            plane_interactions = MortonPlaneInteractionPlan(
                source_plane_plan,
                target_plane_plan,
                opening_angle=plan.plane_opening_angle,
                queue_capacity=queue_capacity,
                far_capacity=far_capacity,
                near_capacity=near_capacity,
            ).build(
                source_plane,
                target_plane,
                same_support=plan.reference_target_topology == "same-support",
            )
            if not bool(
                source_plane.evidence.successful
                & target_plane.evidence.successful
                & plane_interactions.evidence.successful
            ):
                raise ValueError(
                    "Plane vortex schedule or interaction capacity is exhausted."
                )
        self.source_plane_plan = source_plane_plan
        self.target_plane_plan = target_plane_plan
        self.source_plane = source_plane
        self.target_plane = target_plane
        self.plane_interactions = plane_interactions
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-sparse-vortex-fmm",
                "plan": plan.plan_id,
                "compatibility": compatibility.compatibility_id,
                "source_plane": (
                    None if source_plane_plan is None else source_plane_plan.plan_id
                ),
                "target_plane": (
                    None if target_plane_plan is None else target_plane_plan.plan_id
                ),
            }
        )

    def _kernel(self, strength: Array, displacement: Array, /) -> Array:
        if self.dimension == 2:
            squared = jnp.sum(displacement**2)
            return (
                strength
                * jnp.asarray((-displacement[1], displacement[0]))
                / (2.0 * jnp.pi * squared)
            )
        squared = jnp.sum(displacement**2)
        return jnp.cross(strength, displacement) / (
            4.0 * jnp.pi * squared * jnp.sqrt(squared)
        )

    def _multipole_velocity(
        self, displacement: Array, monopole: Array, first_moment: Array, /
    ) -> Array:
        value = self._kernel(monopole, displacement)
        if self.plan.expansion_order == 0:
            return value
        if self.dimension == 2:
            jacobian = jax.jacfwd(
                lambda point: self._kernel(jnp.asarray(1.0, dtype=point.dtype), point)
            )(displacement)
            return value - jacobian @ first_moment
        correction = jnp.zeros((3,), dtype=displacement.dtype)
        basis = jnp.eye(3, dtype=displacement.dtype)
        for component in range(3):
            basis_vector = basis[component]
            jacobian = jax.jacfwd(
                lambda point, vector=basis_vector: self._kernel(vector, point)
            )(displacement)
            correction = correction + jacobian @ first_moment[component]
        return value - correction

    def _moments(self, source, hierarchy, point_leaf, sorted_logical):
        node_capacity = hierarchy.node_active.size
        combined_capacity = sorted_logical.shape[0]
        is_source = hierarchy.sorted_active & (sorted_logical < self.source_capacity)
        source_index = jnp.clip(sorted_logical, 0, self.source_capacity - 1)
        source_strength = source.safe_strength()[source_index]
        sorted_position = jnp.concatenate(
            (source.safe_positions(), jnp.zeros((self.target_capacity, self.dimension))),
            axis=0,
        )[sorted_logical]
        safe_leaf = jnp.maximum(point_leaf, 0)
        relative = sorted_position - hierarchy.node_centers[safe_leaf]
        if self.dimension == 2:
            strength = jnp.where(is_source, source_strength, 0.0)
            monopole = (
                jnp.zeros((node_capacity,), dtype=source.positions.dtype)
                .at[safe_leaf]
                .add(strength)
            )
            first = (
                jnp.zeros((node_capacity, self.dimension), dtype=source.positions.dtype)
                .at[safe_leaf]
                .add(strength[:, None] * relative)
            )
        else:
            strength = jnp.where(is_source[:, None], source_strength, 0.0)
            monopole = (
                jnp.zeros((node_capacity, 3), dtype=source.positions.dtype)
                .at[safe_leaf]
                .add(strength)
            )
            first = (
                jnp.zeros(
                    (node_capacity, 3, self.dimension), dtype=source.positions.dtype
                )
                .at[safe_leaf]
                .add(strength[..., None] * relative[:, None, :])
            )
        del combined_capacity
        for level in range(self.plan.depth - 1, -1, -1):
            at_level = (
                hierarchy.node_active
                & ~hierarchy.node_is_leaf
                & (hierarchy.node_levels == level)
            )
            children = hierarchy.node_children
            child_valid = children >= 0
            safe_children = jnp.maximum(children, 0)
            child_monopole = monopole[safe_children]
            shift = (
                hierarchy.node_centers[safe_children] - hierarchy.node_centers[:, None, :]
            )
            if self.dimension == 2:
                child_monopole = jnp.where(child_valid, child_monopole, 0.0)
                parent_monopole = jnp.sum(child_monopole, axis=1)
                translated_first = (
                    first[safe_children] + child_monopole[..., None] * shift
                )
                parent_first = jnp.sum(
                    jnp.where(child_valid[..., None], translated_first, 0.0),
                    axis=1,
                )
                monopole = jnp.where(at_level, parent_monopole, monopole)
                first = jnp.where(at_level[:, None], parent_first, first)
            else:
                child_monopole = jnp.where(child_valid[..., None], child_monopole, 0.0)
                parent_monopole = jnp.sum(child_monopole, axis=1)
                translated_first = (
                    first[safe_children]
                    + child_monopole[..., None] * shift[:, :, None, :]
                )
                parent_first = jnp.sum(
                    jnp.where(child_valid[..., None, None], translated_first, 0.0),
                    axis=1,
                )
                monopole = jnp.where(at_level[:, None], parent_monopole, monopole)
                first = jnp.where(at_level[:, None, None], parent_first, first)
        return monopole, first

    def _plane_stale(
        self,
        source: VortexSourceState,
        target: VortexTargetState,
    ) -> tuple[Array, Array]:
        if self.source_plane is None or self.target_plane is None:
            raise RuntimeError("Plane vortex topology is not prepared.")
        source_position = source.safe_positions()
        target_position = target.positions
        source_displacement = jnp.sqrt(
            jnp.sum(
                (source_position - self.plan.reference_position) ** 2,
                axis=-1,
            )
        )
        target_displacement = jnp.sqrt(
            jnp.sum(
                (target_position - self.plan.reference_target) ** 2,
                axis=-1,
            )
        )
        maximum_displacement = jnp.maximum(
            jnp.max(
                jnp.where(source.active_mask, source_displacement, 0.0),
                initial=0.0,
            ),
            jnp.max(target_displacement, initial=0.0),
        )
        source_leaf = jnp.maximum(
            self.source_plane.logical_point_leaf_slots,
            0,
        )
        target_leaf = jnp.maximum(
            self.target_plane.logical_point_leaf_slots,
            0,
        )
        source_lower = (
            self.source_plane.node_centers[source_leaf]
            - self.source_plane.node_half_widths[source_leaf]
        )
        source_upper = (
            self.source_plane.node_centers[source_leaf]
            + self.source_plane.node_half_widths[source_leaf]
        )
        target_lower = (
            self.target_plane.node_centers[target_leaf]
            - self.target_plane.node_half_widths[target_leaf]
        )
        target_upper = (
            self.target_plane.node_centers[target_leaf]
            + self.target_plane.node_half_widths[target_leaf]
        )
        source_tolerance = (
            8.0
            * jnp.finfo(source_position.dtype).eps
            * (1.0 + jnp.abs(source_lower) + jnp.abs(source_upper))
        )
        target_tolerance = (
            8.0
            * jnp.finfo(target_position.dtype).eps
            * (1.0 + jnp.abs(target_lower) + jnp.abs(target_upper))
        )
        source_inside = jnp.all(
            jnp.where(
                source.active_mask[:, None],
                (source_position >= source_lower - source_tolerance)
                & (source_position <= source_upper + source_tolerance),
                True,
            )
        )
        target_inside = jnp.all(
            (target_position >= target_lower - target_tolerance)
            & (target_position <= target_upper + target_tolerance)
        )
        finite = jnp.all(jnp.isfinite(source_position)) & jnp.all(
            jnp.isfinite(target_position)
        )
        stale = (
            (maximum_displacement > self.plan.maximum_reference_displacement)
            | ~source_inside
            | ~target_inside
            | ~finite
        )
        return maximum_displacement, stale

    def _plane_moments(
        self,
        source: VortexSourceState,
    ) -> tuple[Array, Array]:
        if self.source_plane is None or self.source_plane_plan is None:
            raise RuntimeError("Plane vortex topology is not prepared.")
        schedule = self.source_plane
        node_capacity = self.source_plane_plan.node_capacity
        leaf_capacity = self.source_plane_plan.plane_capacities[0]
        leaf_width = self.source_plane_plan.maximum_leaf_occupancy
        logical = schedule.point_order.storage_to_logical
        sorted_position = source.safe_positions()[logical]
        sorted_strength = source.safe_strength()[logical]
        sorted_active = schedule.point_order.sorted_active & source.active_mask[logical]
        offsets = jnp.arange(leaf_width, dtype=jnp.int32)
        storage = schedule.node_item_starts[:leaf_capacity, None] + offsets[None, :]
        safe_storage = jnp.clip(storage, 0, self.source_capacity - 1)
        valid = (
            offsets[None, :] < schedule.node_item_counts[:leaf_capacity, None]
        ) & sorted_active[safe_storage]
        positions = sorted_position[safe_storage]
        centers = schedule.node_centers[:leaf_capacity]
        relative = positions - centers[:, None, :]
        if self.dimension == 2:
            strength = jnp.where(valid, sorted_strength[safe_storage], 0.0)
            leaf_monopole = jnp.sum(strength, axis=1)
            leaf_first = jnp.sum(strength[..., None] * relative, axis=1)
            monopole = (
                jnp.zeros((node_capacity,), dtype=positions.dtype)
                .at[:leaf_capacity]
                .set(leaf_monopole)
            )
            first = (
                jnp.zeros((node_capacity, self.dimension), dtype=positions.dtype)
                .at[:leaf_capacity]
                .set(leaf_first)
            )
        else:
            strength = jnp.where(
                valid[..., None],
                sorted_strength[safe_storage],
                0.0,
            )
            leaf_monopole = jnp.sum(strength, axis=1)
            leaf_first = jnp.sum(strength[..., None] * relative[:, :, None, :], axis=1)
            monopole = (
                jnp.zeros((node_capacity, 3), dtype=positions.dtype)
                .at[:leaf_capacity]
                .set(leaf_monopole)
            )
            first = (
                jnp.zeros(
                    (node_capacity, 3, self.dimension),
                    dtype=positions.dtype,
                )
                .at[:leaf_capacity]
                .set(leaf_first)
            )
        child_offsets = jnp.arange(self.source_plane_plan.coarsening_factor)
        for plane in range(1, self.source_plane_plan.plane_count):
            at_plane = schedule.node_active & (schedule.node_planes == plane)
            children = schedule.node_child_starts[:, None] + child_offsets[None, :]
            child_valid = at_plane[:, None] & (
                child_offsets[None, :] < schedule.node_child_counts[:, None]
            )
            safe_children = jnp.clip(children, 0, node_capacity - 1)
            child_monopole = monopole[safe_children]
            shift = (
                schedule.node_centers[safe_children] - schedule.node_centers[:, None, :]
            )
            if self.dimension == 2:
                child_monopole = jnp.where(child_valid, child_monopole, 0.0)
                parent_monopole = jnp.sum(child_monopole, axis=1)
                translated_first = (
                    first[safe_children] + child_monopole[..., None] * shift
                )
                parent_first = jnp.sum(
                    jnp.where(child_valid[..., None], translated_first, 0.0),
                    axis=1,
                )
                monopole = jnp.where(at_plane, parent_monopole, monopole)
                first = jnp.where(at_plane[:, None], parent_first, first)
            else:
                child_monopole = jnp.where(
                    child_valid[..., None],
                    child_monopole,
                    0.0,
                )
                parent_monopole = jnp.sum(child_monopole, axis=1)
                translated_first = (
                    first[safe_children]
                    + child_monopole[..., None] * shift[:, :, None, :]
                )
                parent_first = jnp.sum(
                    jnp.where(
                        child_valid[..., None, None],
                        translated_first,
                        0.0,
                    ),
                    axis=1,
                )
                monopole = jnp.where(at_plane[:, None], parent_monopole, monopole)
                first = jnp.where(at_plane[:, None, None], parent_first, first)
        return monopole, first

    def _evaluate_plane(
        self,
        source: VortexSourceState,
        target: VortexTargetState,
        request: VortexFieldRequest,
    ) -> VortexVelocityEvaluation:
        if (
            self.source_plane is None
            or self.target_plane is None
            or self.source_plane_plan is None
            or self.target_plane_plan is None
            or self.plane_interactions is None
        ):
            raise RuntimeError("Plane vortex topology is not prepared.")
        displacement_from_reference, stale = self._plane_stale(source, target)
        source_schedule = self.source_plane
        target_schedule = self.target_plane
        interactions = self.plane_interactions
        source_monopole, source_first = self._plane_moments(source)
        far = interactions.far
        far_source = far.source_indices
        far_target = far.target_indices
        far_displacement = (
            target_schedule.node_centers[far_target]
            - source_schedule.node_centers[far_source]
        )
        far_velocity_route = jax.vmap(self._multipole_velocity)(
            far_displacement,
            source_monopole[far_source],
            source_first[far_source],
        )

        def far_gradient(displacement, monopole, first):
            return jax.jacfwd(
                lambda value: self._multipole_velocity(value, monopole, first)
            )(displacement)

        far_gradient_route = jax.vmap(far_gradient)(
            far_displacement,
            source_monopole[far_source],
            source_first[far_source],
        )
        far_velocity_route = jnp.where(
            far.valid[:, None],
            far_velocity_route,
            0.0,
        )
        far_gradient_route = jnp.where(
            far.valid[:, None, None],
            far_gradient_route,
            0.0,
        )
        target_node_capacity = self.target_plane_plan.node_capacity
        far_execution = RelationExecutionPlan(
            maximum_active_targets=target_node_capacity
        ).prepare(far)
        local_velocity, _velocity_evidence = far_execution.reduce(
            far_velocity_route,
            accumulation="deterministic",
        )
        local_gradient, _gradient_evidence = far_execution.reduce(
            far_gradient_route,
            accumulation="deterministic",
        )
        for plane in range(self.target_plane_plan.plane_count - 2, -1, -1):
            at_plane = target_schedule.node_active & (
                target_schedule.node_planes == plane
            )
            parent = jnp.maximum(target_schedule.node_parents, 0)
            shift = target_schedule.node_centers - target_schedule.node_centers[parent]
            inherited_velocity = local_velocity[parent] + contract(
                "nij,nj->ni",
                local_gradient[parent],
                shift,
            )
            local_velocity = local_velocity + jnp.where(
                at_plane[:, None],
                inherited_velocity,
                0.0,
            )
            local_gradient = local_gradient + jnp.where(
                at_plane[:, None, None],
                local_gradient[parent],
                0.0,
            )
        target_leaf = jnp.maximum(
            target_schedule.logical_point_leaf_slots,
            0,
        )
        target_delta = target.positions - target_schedule.node_centers[target_leaf]
        far_velocity = local_velocity[target_leaf] + contract(
            "tij,tj->ti",
            local_gradient[target_leaf],
            target_delta,
        )
        far_gradient_value = local_gradient[target_leaf]

        near = interactions.near
        target_offsets = jnp.arange(
            self.target_plane_plan.maximum_leaf_occupancy,
            dtype=jnp.int32,
        )
        source_offsets = jnp.arange(
            self.source_plane_plan.maximum_leaf_occupancy,
            dtype=jnp.int32,
        )
        target_storage = (
            target_schedule.node_item_starts[near.target_indices, None]
            + target_offsets[None, :]
        )
        source_storage = (
            source_schedule.node_item_starts[near.source_indices, None]
            + source_offsets[None, :]
        )
        target_valid = near.valid[:, None] & (
            target_offsets[None, :]
            < target_schedule.node_item_counts[near.target_indices, None]
        )
        source_valid = near.valid[:, None] & (
            source_offsets[None, :]
            < source_schedule.node_item_counts[near.source_indices, None]
        )
        safe_target_storage = jnp.clip(
            target_storage,
            0,
            self.target_capacity - 1,
        )
        safe_source_storage = jnp.clip(
            source_storage,
            0,
            self.source_capacity - 1,
        )
        target_logical = target_schedule.point_order.storage_to_logical[
            safe_target_storage
        ]
        source_logical = source_schedule.point_order.storage_to_logical[
            safe_source_storage
        ]
        displacement = (
            target.positions[target_logical][:, :, None, :]
            - source.safe_positions()[source_logical][:, None, :, :]
        )
        target_identity = (
            jnp.full((self.target_capacity,), -1, dtype=jnp.int32)
            if target.source_indices is None
            else target.source_indices
        )
        self_pair = (
            target_identity[target_logical][:, :, None] == source_logical[:, None, :]
        )
        pair_valid = (
            target_valid[:, :, None]
            & source_valid[:, None, :]
            & source.active_mask[source_logical][:, None, :]
            & ~self_pair
        )
        if self.dimension == 2:
            unit = gaussian_vortex_kernel_2d(
                displacement,
                jnp.broadcast_to(
                    source.safe_core_radius()[source_logical][:, None, :],
                    displacement.shape[:-1],
                ),
            )
            source_strength = jnp.broadcast_to(
                source.safe_strength()[source_logical][:, None, :],
                displacement.shape[:-1],
            )
            pair_velocity = source_strength[..., None] * unit.velocity
            pair_gradient = source_strength[..., None, None] * unit.velocity_gradient
        else:
            kernel = GaussianErfVortexKernel3D().evaluate(
                displacement,
                jnp.broadcast_to(
                    source.safe_strength()[source_logical][:, None, :, :],
                    displacement.shape,
                ),
                jnp.broadcast_to(
                    source.safe_core_radius()[source_logical][:, None, :],
                    displacement.shape[:-1],
                ),
            )
            pair_velocity = kernel.velocity
            pair_gradient = kernel.velocity_gradient
        near_velocity_route = jnp.sum(
            jnp.where(pair_valid[..., None], pair_velocity, 0.0),
            axis=2,
        )
        near_gradient_route = jnp.sum(
            jnp.where(pair_valid[..., None, None], pair_gradient, 0.0),
            axis=2,
        )
        route_valid = target_valid.reshape((-1,))
        route_targets = target_logical.reshape((-1,))
        point_relation = EdgeRelation(
            jnp.zeros(route_targets.shape, dtype=jnp.int32),
            jnp.where(route_valid, route_targets, 0),
            source_size=1,
            target_size=self.target_capacity,
            valid=route_valid,
        )
        point_execution = RelationExecutionPlan(
            maximum_active_targets=self.target_capacity
        ).prepare(point_relation)
        near_velocity, _near_velocity_evidence = point_execution.reduce(
            near_velocity_route.reshape((-1, self.dimension)),
            accumulation="deterministic",
        )
        near_gradient, _near_gradient_evidence = point_execution.reduce(
            near_gradient_route.reshape((-1, self.dimension, self.dimension)),
            accumulation="deterministic",
        )
        velocity = far_velocity + near_velocity
        gradient = far_gradient_value + near_gradient
        if request.vorticity:
            if self.dimension == 2:
                vorticity = gradient[:, 1, 0] - gradient[:, 0, 1]
            else:
                vorticity = jnp.stack(
                    (
                        gradient[:, 2, 1] - gradient[:, 1, 2],
                        gradient[:, 0, 2] - gradient[:, 2, 0],
                        gradient[:, 1, 0] - gradient[:, 0, 1],
                    ),
                    axis=-1,
                )
        else:
            vorticity = None
        route_monopole = source_monopole[far_source]
        monopole_norm = (
            jnp.abs(route_monopole)
            if self.dimension == 2
            else jnp.sqrt(jnp.sum(route_monopole**2, axis=-1))
        )
        source_radius = jnp.sqrt(
            jnp.sum(
                source_schedule.node_half_widths[far_source] ** 2,
                axis=-1,
            )
        )
        target_radius = jnp.sqrt(
            jnp.sum(
                target_schedule.node_half_widths[far_target] ** 2,
                axis=-1,
            )
        )
        route_distance = jnp.sqrt(jnp.sum(far_displacement**2, axis=-1))
        ratio = (source_radius + target_radius) / jnp.maximum(
            route_distance,
            jnp.finfo(source.positions.dtype).tiny,
        )
        tail = jnp.sum(
            jnp.where(
                far.valid,
                monopole_norm * ratio ** (self.plan.expansion_order + 1),
                0.0,
            )
        )
        finite = jnp.all(jnp.isfinite(velocity)) & jnp.all(jnp.isfinite(gradient))
        source_overflow = ~interactions.evidence.successful
        successful = finite & ~stale & ~source_overflow
        evidence = VortexFMMEvidence(
            p2m_count=jnp.sum(source.active_mask, dtype=jnp.int32),
            m2m_count=jnp.sum(
                source_schedule.node_active & (source_schedule.node_planes > 0),
                dtype=jnp.int32,
            ),
            m2l_count=jnp.sum(far.valid, dtype=jnp.int32),
            l2l_count=jnp.maximum(
                target_schedule.evidence.active_nodes - 1,
                0,
            ),
            near_pair_count=jnp.sum(pair_valid, dtype=jnp.int32),
            expansion_order=self.plan.expansion_order,
            geometric_tail_bound=tail,
            maximum_reference_displacement=displacement_from_reference,
            stale_topology=stale,
            source_overflow=source_overflow,
            finite=finite,
            execution="plane_dual",
        )
        diagnostics = VortexVelocityDiagnostics(
            jnp.asarray(source.capacity, dtype=jnp.int32),
            jnp.asarray(target.capacity, dtype=jnp.int32),
            jnp.sum(far.valid, dtype=jnp.int32) + jnp.sum(pair_valid, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.min(source.safe_core_radius()),
            jnp.asarray(True),
            finite,
            ~stale,
            successful,
            evidence,
        )
        return VortexVelocityEvaluation(
            velocity if request.velocity else None,
            gradient if request.velocity_gradient else None,
            vorticity,
            successful,
            self.backend_id,
            canonical_fingerprint(
                {
                    "kind": "plane-vortex-fmm-evaluation",
                    "prepared": self.prepared_id,
                    "request": request.request_id,
                }
            ),
            diagnostics,
        )

    def evaluate(
        self,
        source: VortexSourceState,
        target: VortexTargetState,
        /,
        *,
        request: VortexFieldRequest = DEFAULT_VORTEX_FIELD_REQUEST,
    ) -> VortexVelocityEvaluation:
        source, target = validate_vortex_velocity_evaluation(
            self.capabilities, self.compatibility, source, target, request
        )
        lower = jnp.asarray(self.plan.lower, dtype=target.positions.dtype)
        upper = jnp.asarray(self.plan.upper, dtype=target.positions.dtype)
        target_position = eqx.error_if(
            target.positions,
            jnp.any((target.positions < lower) | (target.positions >= upper)),
            "Vortex FMM targets must lie inside the prepared tree bounds.",
        )
        target = eqx.tree_at(lambda value: value.positions, target, target_position)
        if self.plan.execution == "plane_dual":
            return self._evaluate_plane(source, target, request)
        displacement_from_reference = jnp.max(
            jnp.where(
                source.active_mask,
                jnp.sqrt(
                    jnp.sum(
                        (source.safe_positions() - self.plan.reference_position) ** 2,
                        axis=-1,
                    )
                ),
                0.0,
            ),
            initial=0.0,
        )
        stale = displacement_from_reference > self.plan.maximum_reference_displacement
        combined_position = jnp.concatenate(
            (source.safe_positions(), target.positions), axis=0
        )
        combined_active = jnp.concatenate(
            (
                source.active_mask,
                jnp.ones((target.capacity,), dtype=jnp.bool_),
            )
        )
        combined_capacity = self.source_capacity + self.target_capacity
        branching = 1 << self.dimension
        stencil = 3**self.dimension
        far_stencil = stencil * (branching - 1)
        far_capacity = (
            combined_capacity
            * max(self.plan.depth - 1, 1)
            * min(far_stencil, combined_capacity)
        )
        near_capacity = combined_capacity * min(stencil, combined_capacity)
        level_tree = SparseLevelOctreePlan(
            MortonAddressPlan(self.plan.lower, self.plan.upper, self.plan.depth),
            combined_capacity,
            far_interaction_capacity=far_capacity,
            near_interaction_capacity=near_capacity,
        ).prepare(
            combined_position,
            active_mask=combined_active,
            stable_ids=jnp.arange(combined_capacity, dtype=jnp.int64),
        )
        hierarchy = level_tree.hierarchy
        point_leaf = hierarchy.sorted_point_leaf_slots
        sorted_logical = hierarchy.storage_to_logical
        monopole, first_moment = self._moments(
            source, hierarchy, point_leaf, sorted_logical
        )
        safe_far_source = jnp.maximum(level_tree.far_sources, 0)
        safe_far_target = jnp.maximum(level_tree.far_targets, 0)
        source_center = hierarchy.node_centers[safe_far_source]
        target_center = hierarchy.node_centers[safe_far_target]
        route_displacement = jnp.where(
            level_tree.far_active[:, None],
            target_center - source_center,
            jnp.ones_like(target_center),
        )
        route_monopole = monopole[safe_far_source]
        route_first = first_moment[safe_far_source]
        route_value = jax.vmap(self._multipole_velocity)(
            route_displacement, route_monopole, route_first
        )

        def route_gradient(displacement, source_monopole, source_first):
            return jax.jacfwd(
                lambda value: self._multipole_velocity(
                    value, source_monopole, source_first
                )
            )(displacement)

        route_gradient_value = jax.vmap(route_gradient)(
            route_displacement, route_monopole, route_first
        )
        route_value = jnp.where(level_tree.far_active[:, None], route_value, 0.0)
        route_gradient_value = jnp.where(
            level_tree.far_active[:, None, None], route_gradient_value, 0.0
        )
        local_value = (
            jnp.zeros(
                (hierarchy.node_active.size, self.dimension), dtype=source.positions.dtype
            )
            .at[safe_far_target]
            .add(route_value)
        )
        local_gradient = (
            jnp.zeros(
                (hierarchy.node_active.size, self.dimension, self.dimension),
                dtype=source.positions.dtype,
            )
            .at[safe_far_target]
            .add(route_gradient_value)
        )
        for level in range(1, self.plan.depth + 1):
            at_level = hierarchy.node_active & (hierarchy.node_levels == level)
            parent = jnp.maximum(hierarchy.node_parents, 0)
            shift = hierarchy.node_centers - hierarchy.node_centers[parent]
            inherited_value = local_value[parent] + contract(
                "nij,nj->ni", local_gradient[parent], shift
            )
            local_value = local_value + jnp.where(at_level[:, None], inherited_value, 0.0)
            local_gradient = local_gradient + jnp.where(
                at_level[:, None, None], local_gradient[parent], 0.0
            )
        target_logical = self.source_capacity + jnp.arange(
            self.target_capacity, dtype=jnp.int32
        )
        target_storage = hierarchy.logical_to_storage[target_logical]
        target_leaf = point_leaf[target_storage]
        safe_target_leaf = jnp.maximum(target_leaf, 0)
        delta = target.positions - hierarchy.node_centers[safe_target_leaf]
        far_velocity = local_value[safe_target_leaf] + contract(
            "tij,tj->ti", local_gradient[safe_target_leaf], delta
        )
        near_velocity = jnp.zeros_like(far_velocity)
        near_gradient = jnp.zeros(
            (target.capacity, self.dimension, self.dimension), dtype=far_velocity.dtype
        )
        near_count = jnp.asarray(0, dtype=jnp.int32)
        source_offsets = jnp.arange(self.plan.leaf_capacity, dtype=jnp.int32)
        target_identity = target.source_indices

        def near_route_body(route, state):
            velocity, gradient, count = state
            target_node = jnp.maximum(level_tree.near_targets[route], 0)
            source_node = jnp.maximum(level_tree.near_sources[route], 0)
            route_active = level_tree.near_active[route]
            source_start = hierarchy.node_item_starts[source_node]
            source_count = jnp.where(
                route_active, hierarchy.node_item_counts[source_node], 0
            )
            target_mask = target_leaf == target_node

            def source_body(source_state):
                offset, current_velocity, current_gradient, current_count = source_state
                source_storage = source_start + offset + source_offsets
                source_in_leaf = (source_storage < source_start + source_count) & (
                    source_storage < combined_capacity
                )
                safe_storage = jnp.minimum(source_storage, combined_capacity - 1)
                logical = sorted_logical[safe_storage]
                source_index = jnp.clip(logical, 0, self.source_capacity - 1)
                source_valid = (
                    source_in_leaf
                    & (logical < self.source_capacity)
                    & source.active_mask[source_index]
                )
                displacement = (
                    target.positions[:, None, :]
                    - source.safe_positions()[source_index][None, :, :]
                )
                self_mask = (
                    jnp.zeros((target.capacity, self.plan.leaf_capacity), dtype=jnp.bool_)
                    if target_identity is None
                    else target_identity[:, None] == source_index[None, :]
                )
                if self.dimension == 2:
                    unit_kernel = gaussian_vortex_kernel_2d(
                        displacement,
                        jnp.broadcast_to(
                            source.safe_core_radius()[source_index][None, :],
                            displacement.shape[:-1],
                        ),
                    )
                    pair_strength = jnp.broadcast_to(
                        source.safe_strength()[source_index][None, :],
                        displacement.shape[:-1],
                    )
                    pair_velocity = pair_strength[..., None] * unit_kernel.velocity
                    pair_gradient = (
                        pair_strength[..., None, None] * unit_kernel.velocity_gradient
                    )
                else:
                    kernel = GaussianErfVortexKernel3D().evaluate(
                        displacement,
                        jnp.broadcast_to(
                            source.safe_strength()[source_index][None, :, :],
                            displacement.shape,
                        ),
                        jnp.broadcast_to(
                            source.safe_core_radius()[source_index][None, :],
                            displacement.shape[:-1],
                        ),
                    )
                    pair_velocity = kernel.velocity
                    pair_gradient = kernel.velocity_gradient
                pair_mask = target_mask[:, None] & source_valid[None, :] & ~self_mask
                current_velocity = current_velocity + jnp.sum(
                    jnp.where(pair_mask[..., None], pair_velocity, 0.0), axis=1
                )
                current_gradient = current_gradient + jnp.sum(
                    jnp.where(pair_mask[..., None, None], pair_gradient, 0.0),
                    axis=1,
                )
                return (
                    offset + self.plan.leaf_capacity,
                    current_velocity,
                    current_gradient,
                    current_count + jnp.sum(pair_mask, dtype=jnp.int32),
                )

            source_initial = (
                jnp.asarray(0, dtype=jnp.int32),
                velocity,
                gradient,
                jnp.asarray(0, dtype=jnp.int32),
            )

            def evaluate_route(initial):
                return jax.lax.fori_loop(
                    0,
                    (combined_capacity + self.plan.leaf_capacity - 1)
                    // self.plan.leaf_capacity,
                    lambda _, source_state: source_body(source_state),
                    initial,
                )

            _, velocity, gradient, route_count = jax.lax.cond(
                route_active,
                evaluate_route,
                lambda initial: initial,
                source_initial,
            )
            return velocity, gradient, count + route_count

        near_velocity, near_gradient, near_count = jax.lax.fori_loop(
            0,
            level_tree.near_active.size,
            near_route_body,
            (near_velocity, near_gradient, near_count),
        )
        velocity_all = far_velocity + near_velocity
        gradient_all = local_gradient[safe_target_leaf] + near_gradient
        if request.vorticity:
            if self.dimension == 2:
                vorticity = gradient_all[:, 1, 0] - gradient_all[:, 0, 1]
            else:
                vorticity = jnp.stack(
                    (
                        gradient_all[:, 2, 1] - gradient_all[:, 1, 2],
                        gradient_all[:, 0, 2] - gradient_all[:, 2, 0],
                        gradient_all[:, 1, 0] - gradient_all[:, 0, 1],
                    ),
                    axis=-1,
                )
        else:
            vorticity = None
        monopole_norm = (
            jnp.abs(route_monopole)
            if self.dimension == 2
            else jnp.sqrt(jnp.sum(route_monopole**2, axis=-1))
        )
        route_distance = jnp.sqrt(jnp.sum(route_displacement**2, axis=-1))
        ratio = jnp.max(
            hierarchy.node_half_widths[safe_far_source], axis=-1
        ) / jnp.maximum(
            route_distance,
            jnp.finfo(source.positions.dtype).tiny,
        )
        tail = jnp.sum(
            jnp.where(
                level_tree.far_active,
                monopole_norm * ratio ** (self.plan.expansion_order + 1),
                0.0,
            )
        )
        finite = jnp.all(jnp.isfinite(velocity_all)) & jnp.all(jnp.isfinite(gradient_all))
        source_overflow = ~level_tree.evidence.successful
        successful = finite & ~stale & ~source_overflow
        evidence = VortexFMMEvidence(
            p2m_count=jnp.sum(source.active_mask, dtype=jnp.int32),
            m2m_count=jnp.sum(
                hierarchy.node_active & ~hierarchy.node_is_leaf,
                dtype=jnp.int32,
            ),
            m2l_count=jnp.sum(level_tree.far_active, dtype=jnp.int32),
            l2l_count=jnp.maximum(level_tree.evidence.active_nodes - 1, 0),
            near_pair_count=near_count,
            expansion_order=self.plan.expansion_order,
            geometric_tail_bound=tail,
            maximum_reference_displacement=displacement_from_reference,
            stale_topology=stale,
            source_overflow=source_overflow,
            finite=finite,
            execution="level_octree",
        )
        diagnostics = VortexVelocityDiagnostics(
            jnp.asarray(source.capacity, dtype=jnp.int32),
            jnp.asarray(target.capacity, dtype=jnp.int32),
            jnp.sum(level_tree.far_active, dtype=jnp.int32) + near_count,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.min(source.safe_core_radius()),
            jnp.asarray(True),
            finite,
            ~stale,
            successful,
            evidence,
        )
        return VortexVelocityEvaluation(
            velocity_all if request.velocity else None,
            gradient_all if request.velocity_gradient else None,
            vorticity,
            successful,
            self.backend_id,
            canonical_fingerprint(
                {
                    "kind": "sparse-vortex-fmm-evaluation",
                    "prepared": self.prepared_id,
                    "request": request.request_id,
                }
            ),
            diagnostics,
        )


__all__ = [
    "PreparedVortexFMM",
    "VortexFMMEvidence",
    "VortexFMMExecution",
    "VortexFMMPlan",
]
