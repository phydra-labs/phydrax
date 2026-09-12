#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-topology composite cell coordinates for block AMR."""

from __future__ import annotations

from collections.abc import Sequence
from math import prod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PyTree

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import DiagonalPairing, PyTreeSpace
from ._core import BlockHierarchyState, BlockHierarchyTopology


def _checked(value: Array, invalid: Array, message: str, /) -> Array:
    if isinstance(invalid, jax.core.Tracer):
        return eqx.error_if(value, invalid, message)
    if bool(invalid):
        raise ValueError(message)
    return value


class CompositeAMRCellLayout(StrictModule, NonTrainableState):
    """Leaf-cell Hilbert layout over every fixed-capacity block slot.

    Each level value is an array with shape
    ``(maximum_blocks, *block_shape, *component_shape)``.  Physical leaf cells
    carry their Cartesian cell volume in the pairing.  Covered and inactive
    storage remains in the coordinate space with a positive dummy pairing
    weight; consumers must give those identity rows zero right-hand side.
    """

    topology: BlockHierarchyTopology
    space: PyTreeSpace
    leaf_mask: tuple[Array, ...]
    pairing_weights: tuple[Array, ...]
    cell_measures: Array
    flat_leaf_mask: Array
    component_shape: tuple[int, ...] = eqx.field(static=True)
    level_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    cell_offsets: tuple[int, ...] = eqx.field(static=True)
    cell_count: int = eqx.field(static=True)
    component_count: int = eqx.field(static=True)
    physical_cell_count: int = eqx.field(static=True)
    topology_fingerprint: str = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        topology: BlockHierarchyTopology,
        /,
        *,
        component_shape: Sequence[int] = (),
        dtype: Any = np.float64,
        dummy_weight: float = 1.0,
    ):
        if not isinstance(topology, BlockHierarchyTopology):
            raise TypeError("Composite AMR layout requires BlockHierarchyTopology.")
        components = tuple(int(size) for size in component_shape)
        if any(size <= 0 for size in components):
            raise ValueError("Composite AMR component dimensions must be positive.")
        dtype_ = np.dtype(jax.dtypes.canonicalize_dtype(np.dtype(dtype)))
        if not np.issubdtype(dtype_, np.inexact) or np.issubdtype(
            dtype_, np.complexfloating
        ):
            raise TypeError(
                "Composite AMR cell coordinates require a real inexact dtype."
            )
        dummy = float(dummy_weight)
        if not np.isfinite(dummy) or dummy <= 0.0:
            raise ValueError(
                "Composite AMR dummy pairing weight must be positive and finite."
            )

        masks = tuple(
            np.asarray(metadata.active, dtype=bool).reshape(
                (level_plan.maximum_blocks,) + (1,) * len(level_plan.block_shape)
            )
            & ~np.asarray(topology.covered_cells[level], dtype=bool)
            for level, (level_plan, metadata) in enumerate(
                zip(topology.plan.levels, topology.levels, strict=True)
            )
        )
        shapes = tuple(mask.shape + components for mask in masks)
        component_count = prod(components) if components else 1
        offsets: list[int] = []
        next_offset = 0
        cell_weights: list[np.ndarray] = []
        expanded_weights: list[np.ndarray] = []
        for level, (mask, spacing) in enumerate(
            zip(masks, topology.plan.level_spacings, strict=True)
        ):
            offsets.append(next_offset)
            next_offset += mask.size
            volume = float(prod(spacing))
            cell_weight = np.where(mask, volume, dummy).astype(dtype_)
            cell_weights.append(cell_weight.reshape((-1,)))
            expanded = cell_weight.reshape(cell_weight.shape + (1,) * len(components))
            expanded_weights.append(np.broadcast_to(expanded, shapes[level]).copy())
        flat_mask = np.concatenate(tuple(mask.reshape((-1,)) for mask in masks))
        measures = np.concatenate(tuple(cell_weights))
        weight_arrays = tuple(jnp.asarray(value) for value in expanded_weights)
        topology_fingerprint = canonical_fingerprint(
            {
                "kind": "composite-amr-fixed-topology",
                "epoch": topology.epoch.epoch_id,
                "topology": topology.topology_id,
                "partition": topology.partition_id,
            }
        )
        layout_id = canonical_fingerprint(
            {
                "kind": "composite-amr-cell-layout",
                "topology": topology_fingerprint,
                "level_shapes": shapes,
                "component_shape": components,
                "dtype": dtype_.str,
                "leaf_masks": [array_tree_fingerprint(value) for value in masks],
                "pairing_weights": [
                    array_tree_fingerprint(value) for value in expanded_weights
                ],
            }
        )
        pairing = DiagonalPairing(
            weight_arrays,
            pairing_id=canonical_fingerprint(
                {"kind": "composite-amr-volume-pairing", "layout": layout_id}
            ),
        )
        structure = tuple(jax.ShapeDtypeStruct(shape, dtype_) for shape in shapes)
        space = PyTreeSpace(
            structure,
            pairing=pairing,
            space_id=canonical_fingerprint(
                {"kind": "composite-amr-cell-space", "layout": layout_id}
            ),
        )

        self.topology = topology
        self.space = space
        self.leaf_mask = tuple(jnp.asarray(value) for value in masks)
        self.pairing_weights = weight_arrays
        self.cell_measures = jnp.asarray(measures)
        self.flat_leaf_mask = jnp.asarray(flat_mask)
        self.component_shape = components
        self.level_shapes = shapes
        self.cell_offsets = tuple(offsets)
        self.cell_count = next_offset
        self.component_count = component_count
        self.physical_cell_count = int(np.count_nonzero(flat_mask))
        self.topology_fingerprint = topology_fingerprint
        self.layout_id = layout_id

    @property
    def dtype(self) -> np.dtype:
        return self.space.structure_leaves[0].dtype

    def require_topology(self, topology: BlockHierarchyTopology, /) -> None:
        if not isinstance(topology, BlockHierarchyTopology) or (
            topology.epoch.epoch_id != self.topology.epoch.epoch_id
            or topology.topology_id != self.topology.topology_id
            or topology.partition_id != self.topology.partition_id
        ):
            raise ValueError(
                "Composite AMR cell layout requires its exact fixed topology epoch."
            )

    def validate(self, values: PyTree[Any], /) -> tuple[Array, ...]:
        checked = self.space.validate(values)
        if not isinstance(checked, tuple):
            raise RuntimeError("Composite AMR cell space lost its tuple level structure.")
        return checked

    def bind_state(self, state: BlockHierarchyState, /) -> tuple[Array, ...]:
        if not isinstance(state, BlockHierarchyState):
            raise TypeError("Composite AMR layout can bind only BlockHierarchyState.")
        self.require_topology(state.topology)
        return self.validate(tuple(level.values for level in state.levels))

    def flatten_cells(self, values: PyTree[Any], /) -> Array:
        return self.space.flatten(self.validate(values)).reshape(
            (self.cell_count, self.component_count)
        )

    def unflatten_cells(self, values: Array, /) -> tuple[Array, ...]:
        matrix = jnp.asarray(values)
        expected = (self.cell_count, self.component_count)
        if matrix.shape != expected or matrix.dtype != self.dtype:
            raise ValueError(
                f"Composite AMR cell matrix must have shape {expected} and dtype {self.dtype}."
            )
        restored = self.space.unflatten(matrix.reshape((-1,)))
        if not isinstance(restored, tuple):
            raise RuntimeError("Composite AMR cell space lost its tuple level structure.")
        return restored

    def zero_masked(self, values: PyTree[Any], /) -> tuple[Array, ...]:
        matrix = self.flatten_cells(values)
        result = jnp.where(self.flat_leaf_mask[:, None], matrix, 0.0)
        return self.unflatten_cells(result.astype(self.dtype))

    def require_zero_masked(
        self,
        values: PyTree[Any],
        /,
        *,
        name: str = "right-hand side",
    ) -> tuple[Array, ...]:
        checked = self.validate(values)
        matrix = self.flatten_cells(checked)
        invalid = jnp.any(jnp.where(self.flat_leaf_mask[:, None], 0.0, matrix) != 0.0)
        output = list(checked)
        output[0] = _checked(
            output[0],
            invalid,
            f"Composite AMR {name} must be zero on inactive and covered storage.",
        )
        return tuple(output)

    def integral(self, values: PyTree[Any], /) -> Array:
        matrix = self.flatten_cells(values)
        weights = jnp.where(self.flat_leaf_mask, self.cell_measures, 0.0)
        return jnp.sum(weights[:, None] * matrix, axis=0).reshape(self.component_shape)

    def zero_mean(self, values: PyTree[Any], /) -> tuple[Array, ...]:
        matrix = self.flatten_cells(values)
        weights = jnp.where(self.flat_leaf_mask, self.cell_measures, 0.0)
        total = jnp.sum(weights)
        means = jnp.sum(weights[:, None] * matrix, axis=0) / total
        projected = jnp.where(self.flat_leaf_mask[:, None], matrix - means[None, :], 0.0)
        return self.unflatten_cells(projected.astype(self.dtype))

    def constant_mode_coordinates(self, /) -> Array:
        identity = jnp.eye(self.component_count, dtype=self.dtype)
        cells = self.flat_leaf_mask.astype(self.dtype)[:, None, None] * identity[None]
        return cells.reshape((self.space.size, self.component_count))


__all__ = ["CompositeAMRCellLayout"]
