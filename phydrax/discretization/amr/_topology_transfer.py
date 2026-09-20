#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Sparse conservative cell transfer between immutable AMR topology epochs."""

from __future__ import annotations

from collections.abc import Sequence
from math import prod

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import adjoint, ArraySpace, DiagonalPairing, transpose
from ...sparse import EdgeRelation, SparseCoordinateOperator
from .._spaces import DiscreteFieldSpace, EntityDofLayout
from .._topology_epoch import TopologyEpochTransition
from .._transfer import FieldTransfer, TransferProperties
from ._core import BlockHierarchyState, BlockHierarchyTopology, BlockLevelState


class BlockFieldTopologyTransitionResult(StrictModule):
    """Transitioned hierarchy payload with componentwise conservation evidence."""

    state: BlockHierarchyState
    source_content: Array
    target_content: Array
    conservation_residual: Array
    successful: Array
    result_id: str = eqx.field(static=True)


class _CellReference(tuple):
    __slots__ = ()

    @property
    def level(self) -> int:
        return self[0]

    @property
    def slot(self) -> int:
        return self[1]

    @property
    def local_flat(self) -> int:
        return self[2]

    @property
    def start(self) -> tuple[int, ...]:
        return self[3]

    @property
    def stop(self) -> tuple[int, ...]:
        return self[4]


class BlockCellOverlapRoutes(StrictModule, NonTrainableState):
    """Exact fixed-capacity overlap relation from source cells to target averages."""

    relation: EdgeRelation
    weights: Array
    maximum_target_row_width: int = eqx.field(static=True)
    route_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_indices: Sequence[int],
        target_indices: Sequence[int],
        target_weights: Sequence[float],
        source_count: int,
        target_count: int,
        /,
    ):
        source = np.asarray(source_indices, dtype=np.int32)
        target = np.asarray(target_indices, dtype=np.int32)
        weights = np.asarray(target_weights, dtype=np.float64)
        source_count_ = int(source_count)
        target_count_ = int(target_count)
        if (
            source_count_ <= 0
            or target_count_ <= 0
            or source.ndim != 1
            or target.shape != source.shape
            or weights.shape != source.shape
            or source.size == 0
            or np.any(source < 0)
            or np.any(source >= source_count_)
            or np.any(target < 0)
            or np.any(target >= target_count_)
            or np.any(~np.isfinite(weights))
            or np.any(weights < 0.0)
        ):
            raise ValueError("Sparse AMR overlap routes are invalid.")
        row_width = np.bincount(target, minlength=target_count_)
        if np.any(row_width == 0):
            raise ValueError("Every target AMR cell requires at least one source route.")
        self.relation = EdgeRelation(
            source,
            target,
            source_size=source_count_,
            target_size=target_count_,
        )
        self.weights = jnp.asarray(weights)
        self.maximum_target_row_width = int(np.max(row_width))
        self.route_id = canonical_fingerprint(
            {
                "kind": "block-amr-cell-overlap-routes",
                "source_count": source_count_,
                "target_count": target_count_,
                "relation": {
                    "source_indices": array_tree_fingerprint(source),
                    "target_indices": array_tree_fingerprint(target),
                },
                "weights": array_tree_fingerprint(weights),
            }
        )


def _finest_scales(topology: BlockHierarchyTopology) -> tuple[int, ...]:
    levels = topology.plan.levels
    result = [1] * len(levels)
    for level in reversed(range(len(levels) - 1)):
        result[level] = result[level + 1] * levels[level].refinement_ratio
    return tuple(result)


def _active_boxes(
    topology: BlockHierarchyTopology,
    level: int,
    /,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    plan = topology.plan
    level_plan = plan.levels[level]
    metadata = topology.levels[level]
    count = int(np.count_nonzero(np.asarray(metadata.active)))
    logical = np.asarray(metadata.logical_indices, dtype=np.int32)
    return tuple(
        (
            tuple(
                int(value) * size
                for value, size in zip(logical[slot], level_plan.block_shape, strict=True)
            ),
            tuple(
                (int(value) + 1) * size
                for value, size in zip(logical[slot], level_plan.block_shape, strict=True)
            ),
        )
        for slot in range(count)
    )


def _child_region_is_covered(
    fine_boxes: Sequence[tuple[tuple[int, ...], tuple[int, ...]]],
    start: tuple[int, ...],
    stop: tuple[int, ...],
    /,
) -> bool:
    for coordinate in np.ndindex(
        tuple(upper - lower for lower, upper in zip(start, stop, strict=True))
    ):
        point = tuple(
            lower + offset for lower, offset in zip(start, coordinate, strict=True)
        )
        if not any(
            all(
                lower <= value < upper
                for value, lower, upper in zip(point, box_lower, box_upper, strict=True)
            )
            for box_lower, box_upper in fine_boxes
        ):
            return False
    return True


def _cell_references(
    topology: BlockHierarchyTopology,
    *,
    leaves_only: bool,
) -> tuple[_CellReference, ...]:
    plan = topology.plan
    scales = _finest_scales(topology)
    result: list[_CellReference] = []
    fine_boxes_by_level = tuple(
        () if level + 1 == len(plan.levels) else _active_boxes(topology, level + 1)
        for level in range(len(plan.levels))
    )
    for level, (level_plan, metadata) in enumerate(
        zip(plan.levels, topology.levels, strict=True)
    ):
        count = int(np.count_nonzero(np.asarray(metadata.active)))
        logical = np.asarray(metadata.logical_indices, dtype=np.int32)
        ratio = level_plan.refinement_ratio
        scale = scales[level]
        for slot in range(count):
            block_origin = tuple(
                int(index) * size
                for index, size in zip(logical[slot], level_plan.block_shape, strict=True)
            )
            for local in np.ndindex(level_plan.block_shape):
                global_cell = tuple(
                    origin + index
                    for origin, index in zip(block_origin, local, strict=True)
                )
                covered = level + 1 < len(plan.levels) and _child_region_is_covered(
                    fine_boxes_by_level[level],
                    tuple(index * ratio for index in global_cell),
                    tuple((index + 1) * ratio for index in global_cell),
                )
                if leaves_only and covered:
                    continue
                start = tuple(index * scale for index in global_cell)
                stop = tuple(value + scale for value in start)
                result.append(
                    _CellReference(
                        (
                            level,
                            slot,
                            int(
                                np.ravel_multi_index(
                                    local,
                                    level_plan.block_shape,
                                )
                            ),
                            start,
                            stop,
                        )
                    )
                )
    return tuple(result)


def _overlap(left: _CellReference, right: _CellReference, /) -> int:
    volume = 1
    for left_start, left_stop, right_start, right_stop in zip(
        left.start,
        left.stop,
        right.start,
        right.stop,
        strict=True,
    ):
        width = min(left_stop, right_stop) - max(left_start, right_start)
        if width <= 0:
            return 0
        volume *= width
    return volume


def _reference_volume(cell: _CellReference, /) -> int:
    return prod(stop - start for start, stop in zip(cell.start, cell.stop, strict=True))


def _overlap_routes(
    source: Sequence[_CellReference],
    target: Sequence[_CellReference],
    scales: Sequence[int],
    /,
) -> BlockCellOverlapRoutes:
    """Build nested overlap routes by ancestry, never source-by-target scanning."""
    if not source or not target:
        raise ValueError("AMR overlap routes require non-empty source and target cells.")
    scales_ = tuple(scales)
    dimension = len(source[0].start)
    if (
        len(scales_)
        <= max(
            max(cell.level for cell in source),
            max(cell.level for cell in target),
        )
        or any(value <= 0 for value in scales_)
        or any(scales_[level] % scales_[level + 1] for level in range(len(scales_) - 1))
    ):
        raise ValueError("AMR overlap routes require a nested integer scale hierarchy.")
    source_by_cell = {
        (cell.level, cell.start): index for index, cell in enumerate(source)
    }
    if len(source_by_cell) != len(source):
        raise ValueError("AMR source leaf-cell keys must be unique.")

    def descendants(level: int, start: tuple[int, ...]) -> tuple[int, ...]:
        direct = source_by_cell.get((level, start))
        if direct is not None:
            return (direct,)
        if level + 1 == len(scales_):
            raise ValueError(
                "AMR source leaves do not provide complete nested target coverage."
            )
        ratio = scales_[level] // scales_[level + 1]
        child_scale = scales_[level + 1]
        return tuple(
            index
            for child in np.ndindex((ratio,) * dimension)
            for index in descendants(
                level + 1,
                tuple(
                    value + child_axis * child_scale
                    for value, child_axis in zip(start, child, strict=True)
                ),
            )
        )

    source_indices: list[int] = []
    target_indices: list[int] = []
    target_weights: list[float] = []
    for target_index, target_cell in enumerate(target):
        target_volume = _reference_volume(target_cell)
        ancestor = None
        for level in range(target_cell.level, -1, -1):
            scale = scales_[level]
            start = tuple(value // scale * scale for value in target_cell.start)
            source_index = source_by_cell.get((level, start))
            if source_index is not None:
                ancestor = source_index
                break
        contributors = (
            (ancestor,)
            if ancestor is not None
            else descendants(target_cell.level, target_cell.start)
        )
        row_sum = 0.0
        for source_index in contributors:
            source_cell = source[source_index]
            overlap = _overlap(target_cell, source_cell)
            if overlap == 0:
                continue
            source_indices.append(source_index)
            target_indices.append(target_index)
            target_weights.append(overlap / target_volume)
            row_sum += overlap / target_volume
        if not np.isclose(row_sum, 1.0, rtol=0.0, atol=1.0e-14):
            raise ValueError(
                "AMR topology cells do not provide complete nested coverage."
            )
    return BlockCellOverlapRoutes(
        source_indices,
        target_indices,
        target_weights,
        len(source),
        len(target),
    )


def _leaf_measures(
    topology: BlockHierarchyTopology,
    cells: Sequence[_CellReference],
) -> np.ndarray:
    base_volume = float(
        np.prod(
            [
                float(np.asarray(axis.interval_widths)[0])
                for axis in topology.plan.grid.structured_axes
            ]
        )
    )
    dimension = len(topology.plan.grid.shape)
    cumulative = [1]
    for level in range(1, len(topology.plan.levels)):
        cumulative.append(
            cumulative[-1] * topology.plan.levels[level - 1].refinement_ratio
        )
    return np.asarray(
        [base_volume / cumulative[cell.level] ** dimension for cell in cells],
        dtype=np.float64,
    )


def _field_space(
    topology: BlockHierarchyTopology,
    field_name: str,
    count: int,
    measures: np.ndarray,
    dtype,
) -> DiscreteFieldSpace:
    entity_id = canonical_fingerprint(
        {
            "kind": "amr-leaf-cell-set",
            "topology": topology.topology_id,
            "count": count,
        }
    )
    layout = EntityDofLayout(entity_id, count, count)
    return DiscreteFieldSpace(
        field_name,
        topology.epoch.epoch_id,
        layout,
        ArraySpace(
            (count,),
            dtype=dtype,
            pairing=DiagonalPairing(jnp.asarray(measures, dtype=dtype)),
        ),
        representation="cell_average",
        conformity="discontinuous",
    )


def _route_operator(
    routes: BlockCellOverlapRoutes,
    source,
    target,
    /,
) -> SparseCoordinateOperator:
    return SparseCoordinateOperator(
        routes.relation,
        routes.weights,
        source=source,
        target=target,
        operator_id=canonical_fingerprint(
            {
                "kind": "amr-sparse-overlap-operator",
                "routes": routes.route_id,
                "source": source.space_id,
                "target": target.space_id,
            }
        ),
    )


class BlockFieldTopologyTransition(StrictModule, NonTrainableState):
    """Conservative componentwise cell-field transfer between block epochs.

    All overlap actions use fixed route arrays.  No source-by-target dense matrix is
    allocated, and the algebraic dual pullback remains distinct from the Hilbert
    adjoint under physical cell-volume pairings.
    """

    source_topology: BlockHierarchyTopology
    target_topology: BlockHierarchyTopology
    field_name: str = eqx.field(static=True)
    component_shape: tuple[int, ...] = eqx.field(static=True)
    transfer: FieldTransfer
    transition: TopologyEpochTransition
    leaf_routes: BlockCellOverlapRoutes
    reconstruction_routes: BlockCellOverlapRoutes
    reconstruction_operator: SparseCoordinateOperator
    source_levels: Array
    source_slots: Array
    source_local: Array
    target_levels: Array
    target_slots: Array
    target_local: Array
    transition_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: BlockHierarchyTopology,
        target: BlockHierarchyTopology,
        field_name: str,
        /,
        *,
        component_shape: Sequence[int] = (),
        dtype=jnp.float64,
    ):
        if not isinstance(source, BlockHierarchyTopology) or not isinstance(
            target, BlockHierarchyTopology
        ):
            raise TypeError("Block field transitions require hierarchy topologies.")
        if source.plan.plan_id != target.plan.plan_id:
            raise ValueError("Block field transitions cannot change hierarchy geometry.")
        if target.epoch.index != source.epoch.index + 1:
            raise ValueError(
                "Block field transitions require consecutive topology epochs."
            )
        name = str(field_name).strip()
        components = tuple(component_shape)
        if not name or any(size <= 0 for size in components):
            raise ValueError("Field name and component shape must be valid.")
        source_leaf = _cell_references(source, leaves_only=True)
        target_leaf = _cell_references(target, leaves_only=True)
        target_storage = _cell_references(target, leaves_only=False)
        leaf_routes = _overlap_routes(
            source_leaf,
            target_leaf,
            _finest_scales(source),
        )
        source_measures = _leaf_measures(source, source_leaf)
        target_measures = _leaf_measures(target, target_leaf)
        if not np.allclose(
            np.bincount(
                np.asarray(leaf_routes.relation.source_indices),
                weights=np.asarray(leaf_routes.weights)
                * target_measures[np.asarray(leaf_routes.relation.target_indices)],
                minlength=len(source_leaf),
            ),
            source_measures,
            rtol=1e-12,
            atol=1e-14,
        ):
            raise RuntimeError(
                "AMR leaf-cell transition failed conservation qualification."
            )
        source_space = _field_space(
            source, name, len(source_leaf), source_measures, dtype
        )
        target_space = _field_space(
            target, name, len(target_leaf), target_measures, dtype
        )
        primal = _route_operator(
            leaf_routes,
            source_space.vector_space,
            target_space.vector_space,
        )
        dual = transpose(primal)
        hilbert = adjoint(primal)
        transfer = FieldTransfer(
            source_space,
            target_space,
            primal,
            dual_pullback_operator=dual,
            hilbert_adjoint_operator=hilbert,
            properties=TransferProperties(
                constant_preserving=True,
                conservative=True,
                positivity_preserving=True,
                nested=True,
                adjoint_paired=True,
                differentiable_geometry=False,
                exact_on=("nested-cell-average",),
            ),
        )
        transition = TopologyEpochTransition(
            source.epoch,
            target.epoch,
            transfer,
            source_measures,
            target_measures,
        )
        reconstruction_routes = _overlap_routes(
            target_leaf,
            target_storage,
            _finest_scales(target),
        )
        storage_measures = _leaf_measures(target, target_storage)
        storage_space = ArraySpace(
            (len(target_storage),),
            dtype=dtype,
            pairing=DiagonalPairing(jnp.asarray(storage_measures, dtype=dtype)),
        )
        reconstruction_operator = _route_operator(
            reconstruction_routes,
            target_space.vector_space,
            storage_space,
        )
        self.source_topology = source
        self.target_topology = target
        self.field_name = name
        self.component_shape = components
        self.transfer = transfer
        self.transition = transition
        self.leaf_routes = leaf_routes
        self.reconstruction_routes = reconstruction_routes
        self.reconstruction_operator = reconstruction_operator
        self.source_levels = jnp.asarray(
            [cell.level for cell in source_leaf], dtype=jnp.int32
        )
        self.source_slots = jnp.asarray(
            [cell.slot for cell in source_leaf], dtype=jnp.int32
        )
        self.source_local = jnp.asarray(
            [cell.local_flat for cell in source_leaf], dtype=jnp.int32
        )
        self.target_levels = jnp.asarray(
            [cell.level for cell in target_storage], dtype=jnp.int32
        )
        self.target_slots = jnp.asarray(
            [cell.slot for cell in target_storage], dtype=jnp.int32
        )
        self.target_local = jnp.asarray(
            [cell.local_flat for cell in target_storage], dtype=jnp.int32
        )
        self.transition_id = canonical_fingerprint(
            {
                "kind": "block-field-topology-transition",
                "topology_transition": transition.transition_id,
                "field": name,
                "component_shape": components,
                "leaf_routes": leaf_routes.route_id,
                "reconstruction_routes": reconstruction_routes.route_id,
            }
        )

    def apply(self, state: BlockHierarchyState, /) -> BlockFieldTopologyTransitionResult:
        if not isinstance(state, BlockHierarchyState) or (
            state.topology.epoch.epoch_id != self.source_topology.epoch.epoch_id
        ):
            raise ValueError(
                "AMR field transition state does not match its source epoch."
            )
        if any(
            level.values.shape[1 + len(level.plan.block_shape) :] != self.component_shape
            for level in state.levels
        ):
            raise ValueError(
                "AMR field payload components do not match transition field."
            )
        component_count = prod(self.component_shape) if self.component_shape else 1
        source_values = []
        for level, slot, local in zip(
            np.asarray(self.source_levels),
            np.asarray(self.source_slots),
            np.asarray(self.source_local),
            strict=True,
        ):
            block = state.levels[int(level)].safe_values()[int(slot)]
            source_values.append(block.reshape((-1, component_count))[int(local)])
        packed_source = jnp.stack(source_values, axis=0)
        component_results = tuple(
            self.transition.apply(packed_source[:, component])
            for component in range(component_count)
        )
        packed_target = jnp.stack(
            tuple(result.values for result in component_results), axis=1
        )
        storage_values = jnp.stack(
            tuple(
                self.reconstruction_operator.mv(packed_target[:, component])
                for component in range(component_count)
            ),
            axis=1,
        )
        target_arrays = [
            jnp.zeros(
                (level.maximum_blocks,) + level.block_shape + self.component_shape,
                dtype=storage_values.dtype,
            )
            for level in self.target_topology.plan.levels
        ]
        for row, (level, slot, local) in enumerate(
            zip(
                np.asarray(self.target_levels),
                np.asarray(self.target_slots),
                np.asarray(self.target_local),
                strict=True,
            )
        ):
            level_index = int(level)
            block_shape = self.target_topology.plan.levels[level_index].block_shape
            local_index = np.unravel_index(int(local), block_shape)
            target_arrays[level_index] = (
                target_arrays[level_index]
                .at[(int(slot),) + local_index]
                .set(storage_values[row].reshape(self.component_shape))
            )
        target_levels = tuple(
            BlockLevelState(plan, metadata, values)
            for plan, metadata, values in zip(
                self.target_topology.plan.levels,
                self.target_topology.levels,
                target_arrays,
                strict=True,
            )
        )
        source_content = jnp.stack(
            tuple(result.source_content for result in component_results)
        ).reshape(self.component_shape)
        target_content = jnp.stack(
            tuple(result.target_content for result in component_results)
        ).reshape(self.component_shape)
        residual = jnp.stack(
            tuple(result.conservation_residual for result in component_results)
        ).reshape(self.component_shape)
        successful = jnp.all(
            jnp.stack(tuple(result.successful for result in component_results))
        )
        return BlockFieldTopologyTransitionResult(
            state=BlockHierarchyState(self.target_topology, target_levels),
            source_content=source_content,
            target_content=target_content,
            conservation_residual=residual,
            successful=successful,
            result_id=self.transition_id,
        )


__all__ = [
    "BlockCellOverlapRoutes",
    "BlockFieldTopologyTransition",
    "BlockFieldTopologyTransitionResult",
]
