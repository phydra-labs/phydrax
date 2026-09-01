#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import itertools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
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
from ._gaussian2d import gaussian_vortex_kernel_2d
from ._gaussian3d import GaussianErfVortexKernel3D


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


class VortexFMMPlan(AbstractVortexVelocityPlan):
    reference_position: Array
    lower: Array
    upper: Array
    depth: int = eqx.field(static=True)
    expansion_order: int = eqx.field(static=True)
    leaf_capacity: int = eqx.field(static=True)
    maximum_reference_displacement: float = eqx.field(static=True)
    level_offsets: tuple[int, ...] = eqx.field(static=True)
    level_counts: tuple[int, ...] = eqx.field(static=True)
    centers: Array
    half_width: Array
    parent: Array
    children: Array
    node_level: Array
    leaf_sources: Array
    leaf_source_valid: Array
    m2l_source: Array
    m2l_target: Array
    m2l_valid: Array
    near_source_leaf: Array
    near_target_leaf: Array
    near_valid: Array
    source_capacity: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    capabilities: VortexVelocityCapabilities

    def __init__(
        self,
        reference_position: ArrayLike,
        lower: ArrayLike,
        upper: ArrayLike,
        /,
        *,
        depth: int = 3,
        expansion_order: int = 1,
        leaf_capacity: int = 64,
        maximum_reference_displacement: float = 0.05,
        precision: VortexPrecisionPolicy | None = None,
    ):
        reference, lower_, upper_ = (
            np.asarray(reference_position, dtype=float),
            np.asarray(lower, dtype=float),
            np.asarray(upper, dtype=float),
        )
        dimension, depth_, order, leaf_capacity_ = (
            int(reference.shape[1]) if reference.ndim == 2 else -1,
            int(depth),
            int(expansion_order),
            int(leaf_capacity),
        )
        if (
            reference.ndim != 2
            or reference.shape[0] == 0
            or dimension not in (2, 3)
            or lower_.shape != (dimension,)
            or upper_.shape != lower_.shape
            or np.any(upper_ <= lower_)
            or np.any(reference < lower_)
            or np.any(reference >= upper_)
            or depth_ < 1
            or order not in (0, 1)
            or leaf_capacity_ <= 0
            or maximum_reference_displacement <= 0.0
        ):
            raise ValueError(
                "Vortex FMM geometry/depth/order/capacity controls are invalid."
            )
        branching = 2**dimension
        level_counts = tuple(branching**level for level in range(depth_ + 1))
        offsets = tuple(sum(level_counts[:level]) for level in range(depth_ + 1))
        node_count = sum(level_counts)
        centers, half_widths, parents, node_levels = [], [], [], []
        children = -np.ones((node_count, branching), dtype=np.int32)
        cell_indices_by_level = []
        for level in range(depth_ + 1):
            cells_per_axis = 2**level
            width = (upper_ - lower_) / cells_per_axis
            indices = np.asarray(
                tuple(itertools.product(range(cells_per_axis), repeat=dimension)),
                dtype=np.int32,
            )
            cell_indices_by_level.append(indices)
            for local, index in enumerate(indices):
                global_index = offsets[level] + local
                centers.append(lower_ + (index + 0.5) * width)
                half_widths.append(0.5 * width)
                node_levels.append(level)
                if level == 0:
                    parents.append(-1)
                else:
                    parent_index = tuple((index // 2).tolist())
                    parent_local = np.ravel_multi_index(
                        parent_index, (2 ** (level - 1),) * dimension
                    )
                    parent_global = offsets[level - 1] + parent_local
                    parents.append(parent_global)
                    child_bits = tuple((index % 2).tolist())
                    child_slot = np.ravel_multi_index(child_bits, (2,) * dimension)
                    children[parent_global, child_slot] = global_index
        leaf_cells = cell_indices_by_level[-1]
        leaf_count = level_counts[-1]
        normalized = (reference - lower_) / (upper_ - lower_)
        source_cell = np.minimum(
            (normalized * (2**depth_)).astype(np.int32), 2**depth_ - 1
        )
        source_leaf_local = np.ravel_multi_index(source_cell.T, (2**depth_,) * dimension)
        leaf_sources = -np.ones((leaf_count, leaf_capacity_), dtype=np.int32)
        leaf_valid = np.zeros_like(leaf_sources, dtype=bool)
        overflow = False
        for source_index, leaf in enumerate(source_leaf_local):
            slot = int(np.sum(leaf_valid[leaf]))
            if slot >= leaf_capacity_:
                overflow = True
            else:
                leaf_sources[leaf, slot], leaf_valid[leaf, slot] = source_index, True
        m2l_pairs = []
        for level in range(2, depth_ + 1):
            cells = cell_indices_by_level[level]
            cells_per_axis = 2**level
            for target_local, target_cell in enumerate(cells):
                target_parent = target_cell // 2
                for parent_offset in itertools.product((-1, 0, 1), repeat=dimension):
                    source_parent = target_parent + np.asarray(parent_offset)
                    if np.any(source_parent < 0) or np.any(
                        source_parent >= 2 ** (level - 1)
                    ):
                        continue
                    source_parent_local = np.ravel_multi_index(
                        tuple(source_parent), (2 ** (level - 1),) * dimension
                    )
                    source_parent_global = offsets[level - 1] + source_parent_local
                    for source_global in children[source_parent_global]:
                        if source_global < 0:
                            continue
                        source_local = source_global - offsets[level]
                        source_cell_index = cells[source_local]
                        if np.all(np.abs(source_cell_index - target_cell) <= 1):
                            continue
                        m2l_pairs.append((offsets[level] + target_local, source_global))
        near_pairs = []
        for target_leaf, target_cell in enumerate(leaf_cells):
            for offset in itertools.product((-1, 0, 1), repeat=dimension):
                source_cell_index = target_cell + np.asarray(offset)
                if np.any(source_cell_index < 0) or np.any(
                    source_cell_index >= 2**depth_
                ):
                    continue
                source_leaf = np.ravel_multi_index(
                    tuple(source_cell_index), (2**depth_,) * dimension
                )
                near_pairs.append((target_leaf, source_leaf))
        m2l_capacity = max(len(m2l_pairs), 1)
        near_capacity = max(len(near_pairs), 1)
        self.reference_position, self.lower, self.upper = (
            jnp.asarray(reference),
            jnp.asarray(lower_),
            jnp.asarray(upper_),
        )
        (
            self.depth,
            self.expansion_order,
            self.leaf_capacity,
            self.maximum_reference_displacement,
        ) = depth_, order, leaf_capacity_, float(maximum_reference_displacement)
        self.level_offsets, self.level_counts = offsets, level_counts
        self.centers, self.half_width = jnp.asarray(centers), jnp.asarray(half_widths)
        self.parent, self.children, self.node_level = (
            jnp.asarray(parents, dtype=jnp.int32),
            jnp.asarray(children),
            jnp.asarray(node_levels, dtype=jnp.int8),
        )
        self.leaf_sources, self.leaf_source_valid = (
            jnp.asarray(leaf_sources),
            jnp.asarray(leaf_valid),
        )
        self.m2l_target = jnp.asarray(
            [pair[0] for pair in m2l_pairs] + [0] * (m2l_capacity - len(m2l_pairs)),
            dtype=jnp.int32,
        )
        self.m2l_source = jnp.asarray(
            [pair[1] for pair in m2l_pairs] + [0] * (m2l_capacity - len(m2l_pairs)),
            dtype=jnp.int32,
        )
        self.m2l_valid = jnp.arange(m2l_capacity) < len(m2l_pairs)
        self.near_target_leaf = jnp.asarray(
            [pair[0] for pair in near_pairs] + [0] * (near_capacity - len(near_pairs)),
            dtype=jnp.int32,
        )
        self.near_source_leaf = jnp.asarray(
            [pair[1] for pair in near_pairs] + [0] * (near_capacity - len(near_pairs)),
            dtype=jnp.int32,
        )
        self.near_valid = jnp.arange(near_capacity) < len(near_pairs)
        self.source_capacity, self.dimension = int(reference.shape[0]), dimension
        precision_ = VortexPrecisionPolicy() if precision is None else precision
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
            precision=precision_,
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
                "kind": "vortex-fmm-plan",
                "reference": array_tree_fingerprint(reference),
                "lower": lower_.tolist(),
                "upper": upper_.tolist(),
                "depth": depth_,
                "order": order,
                "leaf_capacity": leaf_capacity_,
                "overflow": overflow,
            }
        )
        if overflow:
            raise ValueError(
                "Vortex FMM reference source occupancy exceeds leaf capacity."
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

    def __init__(
        self, plan: VortexFMMPlan, compatibility: VortexVelocityCompatibility, /
    ):
        self.plan, self.compatibility = plan, compatibility
        self.dimension, self.source_capacity, self.target_capacity = (
            plan.dimension,
            compatibility.source_capacity,
            compatibility.target_capacity,
        )
        self.backend_id, self.capabilities = plan.plan_id, plan.capabilities
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-vortex-fmm",
                "plan": plan.plan_id,
                "compatibility": compatibility.compatibility_id,
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
            jacobian = jax.jacfwd(lambda point: self._kernel(basis_vector, point))(
                displacement
            )
            correction = correction + jacobian @ first_moment[component]
        return value - correction

    def _moments(self, source: VortexSourceState, /) -> tuple[Array, Array]:
        node_count = int(self.plan.centers.shape[0])
        monopole_shape = (node_count,) if self.dimension == 2 else (node_count, 3)
        first_shape = (
            (node_count, self.dimension)
            if self.dimension == 2
            else (node_count, 3, self.dimension)
        )
        monopole = jnp.zeros(monopole_shape, dtype=source.positions.dtype)
        first = jnp.zeros(first_shape, dtype=source.positions.dtype)
        leaf_offset = self.plan.level_offsets[-1]
        safe_indices = jnp.where(self.plan.leaf_source_valid, self.plan.leaf_sources, 0)
        source_strength = source.safe_strength()[safe_indices]
        source_position = source.safe_positions()[safe_indices]
        source_strength = jnp.where(
            self.plan.leaf_source_valid
            if self.dimension == 2
            else self.plan.leaf_source_valid[..., None],
            source_strength,
            0.0,
        )
        relative = source_position - self.plan.centers[leaf_offset:, None, :]
        leaf_monopole = jnp.sum(source_strength, axis=1)
        if self.dimension == 2:
            leaf_first = jnp.sum(source_strength[..., None] * relative, axis=1)
        else:
            leaf_first = jnp.sum(
                source_strength[..., :, None] * relative[..., None, :], axis=1
            )
        monopole = monopole.at[leaf_offset:].set(leaf_monopole)
        first = first.at[leaf_offset:].set(leaf_first)
        for level in range(self.plan.depth - 1, -1, -1):
            start, count = self.plan.level_offsets[level], self.plan.level_counts[level]
            for local in range(count):
                node = start + local
                child = self.plan.children[node]
                valid = child >= 0
                safe = jnp.where(valid, child, 0)
                child_monopole = jnp.where(
                    valid if self.dimension == 2 else valid[:, None], monopole[safe], 0.0
                )
                monopole_value = jnp.sum(child_monopole, axis=0)
                shift = self.plan.centers[safe] - self.plan.centers[node]
                if self.dimension == 2:
                    first_value = jnp.sum(
                        jnp.where(
                            valid[:, None],
                            first[safe] + child_monopole[:, None] * shift,
                            0.0,
                        ),
                        axis=0,
                    )
                else:
                    first_value = jnp.sum(
                        jnp.where(
                            valid[:, None, None],
                            first[safe] + child_monopole[..., None] * shift[:, None, :],
                            0.0,
                        ),
                        axis=0,
                    )
                monopole = monopole.at[node].set(monopole_value)
                first = first.at[node].set(first_value)
        return monopole, first

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
        target_position = eqx.error_if(
            target.positions,
            jnp.any(
                (target.positions < self.plan.lower)
                | (target.positions >= self.plan.upper)
            ),
            "Vortex FMM targets must lie inside the prepared tree bounds.",
        )
        target = eqx.tree_at(
            lambda value: value.positions,
            target,
            target_position,
        )
        displacement_from_reference = jnp.max(
            jnp.linalg.norm(
                source.safe_positions() - self.plan.reference_position, axis=-1
            )
        )
        stale = displacement_from_reference > self.plan.maximum_reference_displacement
        monopole, first_moment = self._moments(source)
        node_count = int(self.plan.centers.shape[0])
        local_value = jnp.zeros(
            (node_count, self.dimension), dtype=source.positions.dtype
        )
        local_gradient = jnp.zeros(
            (node_count, self.dimension, self.dimension), dtype=source.positions.dtype
        )
        tail = jnp.asarray(0.0, dtype=source.positions.dtype)
        for route in range(int(self.plan.m2l_source.size)):
            source_node, target_node, valid = (
                self.plan.m2l_source[route],
                self.plan.m2l_target[route],
                self.plan.m2l_valid[route],
            )
            source_center = self.plan.centers[source_node]
            target_center = self.plan.centers[target_node]
            source_monopole = monopole[source_node]
            source_first_moment = first_moment[source_node]
            displacement = target_center - source_center
            value = self._multipole_velocity(
                displacement,
                source_monopole,
                source_first_moment,
            )
            gradient = jax.jacfwd(
                lambda point: self._multipole_velocity(
                    point - source_center,
                    source_monopole,
                    source_first_moment,
                )
            )(target_center)
            local_value = local_value.at[target_node].add(jnp.where(valid, value, 0.0))
            local_gradient = local_gradient.at[target_node].add(
                jnp.where(valid, gradient, 0.0)
            )
            ratio = jnp.max(self.plan.half_width[source_node]) / jnp.maximum(
                jnp.linalg.norm(displacement), jnp.finfo(source.positions.dtype).tiny
            )
            tail = tail + jnp.where(
                valid,
                jnp.linalg.norm(monopole[source_node])
                * ratio ** (self.plan.expansion_order + 1),
                0.0,
            )
        for level in range(1, self.plan.depth + 1):
            start, count = self.plan.level_offsets[level], self.plan.level_counts[level]
            for local in range(count):
                node = start + local
                parent = self.plan.parent[node]
                shift = self.plan.centers[node] - self.plan.centers[parent]
                local_value = local_value.at[node].add(
                    local_value[parent] + local_gradient[parent] @ shift
                )
                local_gradient = local_gradient.at[node].add(local_gradient[parent])
        cells_per_axis = 2**self.plan.depth
        normalized = (target.positions - self.plan.lower) / (
            self.plan.upper - self.plan.lower
        )
        cell = jnp.clip(
            jnp.floor(normalized * cells_per_axis).astype(jnp.int32),
            0,
            cells_per_axis - 1,
        )
        multiplier = jnp.asarray(
            tuple(cells_per_axis**power for power in range(self.dimension - 1, -1, -1)),
            dtype=jnp.int32,
        )
        leaf_local = jnp.sum(cell * multiplier, axis=-1)
        leaf_node = self.plan.level_offsets[-1] + leaf_local
        delta = target.positions - self.plan.centers[leaf_node]
        far_velocity = local_value[leaf_node] + contract(
            "tij,tj->ti",
            local_gradient[leaf_node],
            delta,
        )
        near_velocity = jnp.zeros_like(far_velocity)
        near_gradient = jnp.zeros(
            (target.capacity, self.dimension, self.dimension), dtype=far_velocity.dtype
        )
        near_count = jnp.asarray(0, dtype=jnp.int32)
        target_identity = target.source_indices
        for route in range(int(self.plan.near_source_leaf.size)):
            target_leaf, source_leaf, valid_route = (
                self.plan.near_target_leaf[route],
                self.plan.near_source_leaf[route],
                self.plan.near_valid[route],
            )
            target_mask = leaf_local == target_leaf
            indices = self.plan.leaf_sources[source_leaf]
            valid_source = self.plan.leaf_source_valid[source_leaf]
            safe = jnp.where(valid_source, indices, 0)
            displacement = (
                target.positions[:, None, :] - source.safe_positions()[safe][None, :, :]
            )
            source_active = valid_source & source.active_mask[safe]
            self_mask = (
                jnp.zeros((target.capacity, self.plan.leaf_capacity), dtype=bool)
                if target_identity is None
                else target_identity[:, None] == safe[None, :]
            )
            if self.dimension == 2:
                unit_kernel = gaussian_vortex_kernel_2d(
                    displacement,
                    jnp.broadcast_to(
                        source.safe_core_radius()[safe][None, :],
                        displacement.shape[:-1],
                    ),
                )
                pair_strength = jnp.broadcast_to(
                    source.safe_strength()[safe][None, :],
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
                        source.safe_strength()[safe][None, :, :],
                        displacement.shape,
                    ),
                    jnp.broadcast_to(
                        source.safe_core_radius()[safe][None, :],
                        displacement.shape[:-1],
                    ),
                )
                pair_velocity = kernel.velocity
                pair_gradient = kernel.velocity_gradient
            pair_mask = (
                valid_route & target_mask[:, None] & source_active[None, :] & ~self_mask
            )
            near_velocity = near_velocity + jnp.sum(
                jnp.where(pair_mask[..., None], pair_velocity, 0.0), axis=1
            )
            near_gradient = near_gradient + jnp.sum(
                jnp.where(
                    pair_mask[..., None, None],
                    pair_gradient,
                    0.0,
                ),
                axis=1,
            )
            near_count = near_count + jnp.sum(pair_mask, dtype=jnp.int32)
        velocity_all = far_velocity + near_velocity
        gradient_all = local_gradient[leaf_node] + near_gradient
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
        finite = jnp.all(jnp.isfinite(velocity_all)) & jnp.all(jnp.isfinite(gradient_all))
        successful = finite & ~stale
        evidence = VortexFMMEvidence(
            jnp.asarray(source.capacity, dtype=jnp.int32),
            jnp.asarray(sum(self.plan.level_counts[:-1]), dtype=jnp.int32),
            jnp.sum(self.plan.m2l_valid, dtype=jnp.int32),
            jnp.asarray(node_count - 1, dtype=jnp.int32),
            near_count,
            self.plan.expansion_order,
            tail,
            displacement_from_reference,
            stale,
            jnp.asarray(False),
            finite,
        )
        diagnostics = VortexVelocityDiagnostics(
            jnp.asarray(source.capacity, dtype=jnp.int32),
            jnp.asarray(target.capacity, dtype=jnp.int32),
            jnp.sum(self.plan.m2l_valid, dtype=jnp.int32) + near_count,
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
                    "kind": "vortex-fmm-evaluation",
                    "prepared": self.prepared_id,
                    "request": request.request_id,
                }
            ),
            diagnostics,
        )


__all__ = ["PreparedVortexFMM", "VortexFMMEvidence", "VortexFMMPlan"]
