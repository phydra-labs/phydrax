#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native-complex fields on one immutable composite block-AMR topology."""

from __future__ import annotations

from collections.abc import Sequence
from math import prod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PyTree

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    AbstractLinearOperator,
    DiagonalPairing,
    FunctionLinearOperator,
    OperatorProperties,
    PyTreeSpace,
)
from ._composite import CompositeAMRCellLayout
from ._core import BlockHierarchyState, BlockHierarchyTopology
from ._fd_halo import FDAMRFillPatchWorkspace, FillPatchSource
from ._fd_runtime import PreparedFDAMRHierarchy


class ComplexCompositeAMRCellLayout(StrictModule, NonTrainableState):
    """Volume-paired native-complex coordinates over composite AMR leaf cells.

    The existing real composite layout remains the geometry and mask owner. This
    adapter changes only the scalar field algebra; it neither invents an AMR
    topology nor splits a complex wavefunction into independently evolved fields.
    """

    real_layout: CompositeAMRCellLayout
    space: PyTreeSpace
    dtype: np.dtype = eqx.field(static=True)
    real_dtype: np.dtype = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        topology: BlockHierarchyTopology,
        /,
        *,
        component_shape: Sequence[int] = (),
        dtype: Any = np.complex128,
    ):
        dtype_ = np.dtype(jax.dtypes.canonicalize_dtype(np.dtype(dtype)))
        if not np.issubdtype(dtype_, np.complexfloating):
            raise TypeError("Complex AMR field storage requires a complex dtype.")
        real_dtype = np.empty((), dtype=dtype_).real.dtype
        real_layout = CompositeAMRCellLayout(
            topology,
            component_shape=component_shape,
            dtype=real_dtype,
        )
        structure = tuple(
            jax.ShapeDtypeStruct(shape, dtype_) for shape in real_layout.level_shapes
        )
        pairing = DiagonalPairing(
            real_layout.pairing_weights,
            pairing_id=canonical_fingerprint(
                {
                    "kind": "complex-composite-amr-volume-pairing",
                    "real_layout": real_layout.layout_id,
                    "dtype": dtype_.str,
                }
            ),
        )
        layout_id = canonical_fingerprint(
            {
                "kind": "complex-composite-amr-cell-layout",
                "real_layout": real_layout.layout_id,
                "dtype": dtype_.str,
            }
        )
        self.real_layout = real_layout
        self.space = PyTreeSpace(
            structure,
            pairing=pairing,
            space_id=canonical_fingerprint(
                {"kind": "complex-composite-amr-cell-space", "layout": layout_id}
            ),
        )
        self.dtype = dtype_
        self.real_dtype = real_dtype
        self.layout_id = layout_id

    @property
    def topology(self) -> BlockHierarchyTopology:
        return self.real_layout.topology

    @property
    def leaf_mask(self) -> tuple[Array, ...]:
        return self.real_layout.leaf_mask

    @property
    def flat_leaf_mask(self) -> Array:
        return self.real_layout.flat_leaf_mask

    @property
    def cell_measures(self) -> Array:
        return self.real_layout.cell_measures

    @property
    def component_shape(self) -> tuple[int, ...]:
        return self.real_layout.component_shape

    @property
    def component_count(self) -> int:
        return self.real_layout.component_count

    @property
    def cell_count(self) -> int:
        return self.real_layout.cell_count

    @property
    def level_shapes(self) -> tuple[tuple[int, ...], ...]:
        return self.real_layout.level_shapes

    @property
    def topology_fingerprint(self) -> str:
        return self.real_layout.topology_fingerprint

    def require_topology(self, topology: BlockHierarchyTopology, /) -> None:
        self.real_layout.require_topology(topology)

    def validate(self, values: PyTree[Any], /) -> tuple[Array, ...]:
        checked = self.space.validate(values)
        if not isinstance(checked, tuple):
            raise RuntimeError("Complex AMR cell space lost its tuple level structure.")
        return checked

    def bind_state(self, state: BlockHierarchyState, /) -> tuple[Array, ...]:
        if not isinstance(state, BlockHierarchyState):
            raise TypeError("Complex AMR layout can bind only BlockHierarchyState.")
        self.require_topology(state.topology)
        return self.validate(tuple(level.values for level in state.levels))

    def flatten_cells(self, values: PyTree[Any], /) -> Array:
        return self.space.flatten(self.validate(values)).reshape(
            (self.cell_count, self.component_count)
        )

    def unflatten_cells(self, values: Array, /) -> tuple[Array, ...]:
        matrix = jnp.asarray(values)
        expected = (self.cell_count, self.component_count)
        if matrix.shape != expected or np.dtype(matrix.dtype) != self.dtype:
            raise ValueError(
                f"Complex AMR cell matrix must have shape {expected} and dtype {self.dtype}."
            )
        restored = self.space.unflatten(matrix.reshape((-1,)))
        if not isinstance(restored, tuple):
            raise RuntimeError("Complex AMR cell space lost its tuple level structure.")
        return restored

    def zero_masked(self, values: PyTree[Any], /) -> tuple[Array, ...]:
        matrix = self.flatten_cells(values)
        result = jnp.where(self.flat_leaf_mask[:, None], matrix, 0.0)
        return self.unflatten_cells(result.astype(self.dtype))

    def integral(self, values: PyTree[Any], /) -> Array:
        matrix = self.flatten_cells(values)
        weights = jnp.where(self.flat_leaf_mask, self.cell_measures, 0.0)
        return jnp.sum(weights[:, None] * matrix, axis=0).reshape(self.component_shape)

    def probability(self, values: PyTree[Any], /) -> Array:
        matrix = self.flatten_cells(values)
        weights = jnp.where(self.flat_leaf_mask, self.cell_measures, 0.0)
        density = jnp.abs(matrix) ** 2
        return jnp.sum(weights[:, None] * density, axis=0).reshape(self.component_shape)


def _componentwise_real_action(action, values: tuple[Array, ...], /) -> tuple[Array, ...]:
    real = action(tuple(jnp.real(value) for value in values))
    imaginary = action(tuple(jnp.imag(value) for value in values))
    return tuple(
        jax.lax.complex(real_value, imaginary_value).astype(values[index].dtype)
        for index, (real_value, imaginary_value) in enumerate(
            zip(real, imaginary, strict=True)
        )
    )


def complexify_composite_amr_operator(
    operator: AbstractLinearOperator,
    layout: ComplexCompositeAMRCellLayout,
    /,
) -> FunctionLinearOperator:
    """Lift a real composite AMR operator to native complex field coordinates."""
    if not isinstance(operator, AbstractLinearOperator):
        raise TypeError("operator must implement AbstractLinearOperator.")
    if not isinstance(layout, ComplexCompositeAMRCellLayout):
        raise TypeError("layout must be ComplexCompositeAMRCellLayout.")
    real_layout = layout.real_layout
    if (
        operator.source.space_id != real_layout.space.space_id
        or operator.target.space_id != real_layout.space.space_id
    ):
        raise ValueError("Real operator and complex AMR layout geometry must agree.")

    def action(values):
        return _componentwise_real_action(operator.mv, values)

    def transpose_action(values):
        return _componentwise_real_action(operator.transpose_mv, values)

    properties = OperatorProperties(
        diagonal=operator.properties.diagonal,
        triangular=operator.properties.triangular,
        self_adjoint=operator.properties.self_adjoint,
        positive_definite=operator.properties.positive_definite,
        positive_semidefinite=operator.properties.positive_semidefinite,
        block_diagonal=operator.properties.block_diagonal,
        rank=operator.properties.rank,
        evidence=dict(operator.properties.evidence),
    )
    return FunctionLinearOperator(
        action,
        source=layout.space,
        target=layout.space,
        transpose_action=transpose_action,
        properties=properties,
        operator_id=canonical_fingerprint(
            {
                "kind": "complexified-composite-amr-operator",
                "operator": operator.operator_id,
                "layout": layout.layout_id,
            }
        ),
    )


def _u1_linear_prolong(values: Array, ratio: int, dimension: int, /) -> Array:
    """Conservative complex-linear cell prolongation, equivariant under global U(1)."""
    result = values
    for axis in range(dimension):
        previous = jnp.roll(result, 1, axis=axis)
        following = jnp.roll(result, -1, axis=axis)
        lower = [slice(None)] * result.ndim
        upper = [slice(None)] * result.ndim
        lower[axis] = 0
        upper[axis] = result.shape[axis] - 1
        previous = previous.at[tuple(lower)].set(result[tuple(lower)])
        following = following.at[tuple(upper)].set(result[tuple(upper)])
        slope = 0.5 * (following - previous)
        pieces = tuple(
            result + ((child + 0.5) / ratio - 0.5) * slope for child in range(ratio)
        )
        stacked = jnp.stack(pieces, axis=axis + 1)
        shape = stacked.shape
        result = stacked.reshape(
            shape[:axis] + (shape[axis] * shape[axis + 1],) + shape[axis + 2 :]
        )
    return result


class ComplexAMRFillPatchResult(StrictModule):
    """U(1)-equivariant native-complex FillPatch workspaces and completeness."""

    workspaces: tuple[FDAMRFillPatchWorkspace, ...]
    complete: Array
    result_id: str = eqx.field(static=True)

    def require_complete(self, /) -> tuple[FDAMRFillPatchWorkspace, ...]:
        if not bool(self.complete):
            raise ValueError("Complex AMR FillPatch has unresolved cells.")
        return self.workspaces


def complex_amr_fill_patch(
    prepared: PreparedFDAMRHierarchy,
    state: BlockHierarchyState,
    /,
) -> ComplexAMRFillPatchResult:
    """Execute FillPatch with one complex-linear, U(1)-equivariant prolongation."""
    if not isinstance(prepared, PreparedFDAMRHierarchy):
        raise TypeError("prepared must be PreparedFDAMRHierarchy.")
    if not isinstance(state, BlockHierarchyState):
        raise TypeError("state must be BlockHierarchyState.")
    if any(
        not jnp.issubdtype(level.values.dtype, jnp.complexfloating)
        for level in state.levels
    ):
        raise TypeError("Complex AMR FillPatch requires complex hierarchy values.")
    plans = prepared.prepare_fill_patch(state.topology)
    workspaces = []
    complete_levels = []
    for fill, metadata, level_plan in zip(
        plans,
        state.topology.levels,
        state.topology.plan.levels,
        strict=True,
    ):
        same_values = fill._same_level_values(state)
        same_mask = (
            (fill.source_class == int(FillPatchSource.INTERIOR))
            | (fill.source_class == int(FillPatchSource.SAME_LEVEL))
            | (fill.source_class == int(FillPatchSource.PERIODIC))
        )
        values = jnp.where(same_mask, same_values, 0.0)
        valid = same_mask
        if fill.level > 0:
            if fill.transfer is None:
                raise RuntimeError("Complex FillPatch lost its coarse transfer geometry.")
            donors = fill._coarse_donor_values(state)
            dimension = len(state.topology.plan.grid.shape)
            donor_patches = donors.reshape(fill.source_class.shape + (3,) * dimension)
            route_count = prod(fill.source_class.shape)
            flat_patches = donor_patches.reshape((route_count,) + (3,) * dimension)
            prolonged = jax.vmap(
                lambda patch: _u1_linear_prolong(
                    patch,
                    fill.transfer.refinement_ratio,
                    dimension,
                )
            )(flat_patches)
            child = jnp.maximum(
                fill.coarse_child_indices.reshape((route_count, dimension)),
                0,
            )
            ratio = fill.transfer.refinement_ratio
            fine_index = (jnp.arange(route_count, dtype=jnp.int32),) + tuple(
                ratio + child[:, axis] for axis in range(dimension)
            )
            coarse_values = prolonged[fine_index].reshape(fill.source_class.shape)
            coarse_mask = fill.source_class == int(
                FillPatchSource.COARSE_TIME_INTERPOLATED
            )
            values = jnp.where(coarse_mask, coarse_values, values)
            valid = valid | coarse_mask
        active = metadata.active.reshape(
            (level_plan.maximum_blocks,) + (1,) * (valid.ndim - 1)
        )
        valid = valid & active
        values = jnp.where(valid, values, 0.0)
        workspace = FDAMRFillPatchWorkspace(
            values,
            valid,
            fill.source_class,
            canonical_fingerprint(
                {
                    "kind": "u1-equivariant-complex-fill-patch-level",
                    "plan": fill.plan_id,
                }
            ),
        )
        workspaces.append(workspace)
        complete_levels.append(jnp.all(valid | ~active))
    complete = jnp.all(jnp.stack(tuple(complete_levels)))
    return ComplexAMRFillPatchResult(
        tuple(workspaces),
        complete,
        canonical_fingerprint(
            {
                "kind": "u1-equivariant-complex-amr-fill-patch-result",
                "prepared": prepared.prepared_id,
                "epoch": state.topology.epoch.epoch_id,
                "workspaces": [workspace.workspace_id for workspace in workspaces],
            }
        ),
    )


__all__ = [
    "ComplexAMRFillPatchResult",
    "ComplexCompositeAMRCellLayout",
    "complex_amr_fill_patch",
    "complexify_composite_amr_operator",
]
