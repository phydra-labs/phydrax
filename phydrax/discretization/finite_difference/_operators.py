#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ...linalg import (
    AbstractLinearOperator,
    ArraySpace,
    DiagonalPairing,
    EuclideanPairing,
    OperatorCapabilities,
    OperatorProperties,
)
from .._spaces import DiscreteFieldSpace
from .._tensor_support import PreparedTensorGrid
from ._certification import (
    certify_operator_adjoint,
    certify_operator_conservation,
    certify_stencil_consistency,
    FDAdjointReport,
    FDConservationReport,
    FDConsistencyReport,
)
from ._coefficients import StencilCoefficientPlan
from ._request import DerivativeRequest
from ._stencil import (
    BoundaryStencilSet,
    LinearStencil,
    StencilFootprint,
    StencilRowKind,
)


def _interior_width(request: DerivativeRequest, /) -> int:
    width = request.derivative_order + request.accuracy_order - 1
    axis = request.source_location.axis_names.index(request.axis)
    shift = request.target_location.offsets[axis] - request.source_location.offsets[axis]
    if request.bias == "centered" and shift.denominator == 1 and width % 2 == 0:
        width += 1
    return max(width, request.derivative_order + 1)


def _window_start(
    coordinates: np.ndarray,
    evaluation_point: float,
    width: int,
    bias: str,
    /,
) -> int:
    insertion = int(np.searchsorted(coordinates, evaluation_point, side="left"))
    if bias == "forward":
        return insertion
    if bias == "backward":
        return insertion - width
    return insertion - width // 2


def prepare_linear_stencil(
    grid: PreparedTensorGrid,
    request: DerivativeRequest,
    /,
) -> BoundaryStencilSet:
    """Prepare a masked variable-width derivative bank between exact entity layouts."""
    if not isinstance(grid, PreparedTensorGrid) or not isinstance(
        request, DerivativeRequest
    ):
        raise TypeError("grid and request must be prepared tensor/derivative values.")
    axis = grid.axis_names.index(request.axis)
    source_layout = grid.layout_at(request.source_location)
    target_layout = grid.layout_at(request.target_location)
    if any(
        source_layout.shape[index] != target_layout.shape[index]
        for index in range(len(grid.shape))
        if index != axis
    ):
        raise ValueError("Derivative source/target non-differentiated axes must align.")
    source_coordinates = np.asarray(source_layout.coordinates_by_axis[axis], dtype=float)
    target_coordinates = np.asarray(target_layout.coordinates_by_axis[axis], dtype=float)
    source_count = int(source_coordinates.size)
    target_count = int(target_coordinates.size)
    interior_width = _interior_width(request)
    closure_width = max(
        interior_width,
        request.derivative_order + request.accuracy_order,
    )
    capacity = closure_width if request.boundary == "one_sided" else interior_width
    if capacity > source_count:
        raise ValueError(
            f"Derivative stencil capacity {capacity} exceeds source entity count {source_count}."
        )
    indices = np.zeros((target_count, capacity), dtype=np.int32)
    weights = np.full((target_count, capacity), np.nan, dtype=float)
    valid = np.zeros((target_count, capacity), dtype=bool)
    plans: list[StencilCoefficientPlan] = []
    row_kinds: list[StencilRowKind] = []
    maximum_lower = 0
    maximum_upper = 0
    periodic = request.boundary == "periodic"
    periodic_plan: StencilCoefficientPlan | None = None
    if periodic:
        if source_count != target_count:
            raise ValueError(
                "Periodic derivatives require equal source and target counts."
            )
        spacing = np.diff(source_coordinates)
        if not np.allclose(spacing, spacing[0], rtol=1e-10, atol=1e-12):
            raise ValueError("Periodic finite differences require uniform spacing.")
        delta = float(spacing[0])
        location_shift = float(
            request.target_location.offsets[axis] - request.source_location.offsets[axis]
        )
        if request.bias == "forward":
            relative = np.arange(interior_width, dtype=np.int32)
        elif request.bias == "backward":
            relative = np.arange(-interior_width + 1, 1, dtype=np.int32)
        else:
            start = int(np.floor(location_shift - 0.5 * (interior_width - 1)))
            relative = np.arange(start, start + interior_width, dtype=np.int32)
        coefficient_nodes = (relative.astype(float) - location_shift) * delta
        periodic_plan = StencilCoefficientPlan(
            coefficient_nodes,
            0.0,
            request.derivative_order,
            request.accuracy_order,
        )
    for output_index, evaluation_point in enumerate(target_coordinates):
        if periodic:
            point_indices = (output_index + relative) % source_count
            width = interior_width
            coefficient_plan = periodic_plan
            row_kind: StencilRowKind = "interior"
            logical_relative = relative
        else:
            start = _window_start(
                source_coordinates,
                float(evaluation_point),
                interior_width,
                request.bias,
            )
            if 0 <= start and start + interior_width <= source_count:
                width = interior_width
                row_kind = "interior"
            else:
                width = closure_width
                if start < 0:
                    row_kind = "lower_closure"
                    start = 0
                else:
                    row_kind = "upper_closure"
                    start = source_count - width
            point_indices = np.arange(start, start + width, dtype=np.int32)
            coefficient_plan = StencilCoefficientPlan(
                source_coordinates[point_indices],
                float(evaluation_point),
                request.derivative_order,
                request.accuracy_order,
            )
            anchor = int(
                np.clip(
                    np.searchsorted(source_coordinates, evaluation_point),
                    0,
                    source_count - 1,
                )
            )
            logical_relative = point_indices - anchor
        if coefficient_plan is None:
            raise RuntimeError("Periodic coefficient preparation unexpectedly failed.")
        indices[output_index, :width] = point_indices
        weights[output_index, :width] = np.asarray(coefficient_plan.weights)
        valid[output_index, :width] = True
        plans.append(coefficient_plan)
        row_kinds.append(row_kind)
        maximum_lower = max(maximum_lower, int(max(0, -np.min(logical_relative))))
        maximum_upper = max(maximum_upper, int(max(0, np.max(logical_relative))))
    lower = [0] * len(grid.axis_names)
    upper = [0] * len(grid.axis_names)
    lower[axis] = maximum_lower
    upper[axis] = maximum_upper
    footprint = StencilFootprint(grid.axis_names, lower, upper)
    stencil = LinearStencil(
        request,
        axis,
        indices,
        weights,
        plans,
        footprint,
        valid=valid,
        row_kinds=row_kinds,
    )
    return BoundaryStencilSet(stencil, kind=request.boundary)


class PreparedStencilOperator(AbstractLinearOperator):
    """Masked tensor-axis stencil with rectangular transpose and weighted adjoint."""

    source: ArraySpace
    target: ArraySpace
    stencil_set: BoundaryStencilSet
    axis: int = eqx.field(static=True)
    indices: Array
    weights: Array
    valid: Array
    consistency_report: FDConsistencyReport
    adjoint_report: FDAdjointReport
    conservation_report: FDConservationReport

    def __init__(
        self,
        stencil_set: BoundaryStencilSet,
        source: DiscreteFieldSpace,
        target: DiscreteFieldSpace,
        /,
    ):
        if not isinstance(stencil_set, BoundaryStencilSet):
            raise TypeError("stencil_set must be a BoundaryStencilSet.")
        if not isinstance(source, DiscreteFieldSpace) or not isinstance(
            target, DiscreteFieldSpace
        ):
            raise TypeError("source and target must be DiscreteFieldSpace values.")
        if not isinstance(source.vector_space, ArraySpace) or not isinstance(
            target.vector_space, ArraySpace
        ):
            raise TypeError("Prepared stencil operators require ArraySpace fields.")
        axis = stencil_set.stencil.axis_index
        if len(source.vector_space.shape) != len(target.vector_space.shape):
            raise ValueError("Stencil source and target ranks must match.")
        if any(
            source.vector_space.shape[index] != target.vector_space.shape[index]
            for index in range(len(source.vector_space.shape))
            if index != axis
        ):
            raise ValueError("Stencil non-differentiated source/target axes must match.")
        active_indices = np.asarray(stencil_set.stencil.indices)[
            np.asarray(stencil_set.stencil.valid)
        ]
        if np.any(active_indices >= source.vector_space.shape[axis]):
            raise ValueError("Stencil source index exceeds source entity count.")
        if target.vector_space.shape[axis] != stencil_set.stencil.indices.shape[0]:
            raise ValueError("Stencil row count must match target entity count.")
        self.source = source.vector_space
        self.target = target.vector_space
        self.properties = OperatorProperties(evidence={})
        self.capabilities = OperatorCapabilities(
            transpose=True,
            adjoint=True,
            materialize=True,
            diagonal_assembly=False,
        )
        self.batch_shape = ()
        self.operator_id = canonical_fingerprint(
            {
                "kind": "prepared-stencil-operator",
                "stencil": stencil_set.stencil.stencil_id,
                "source": source.field_space_id,
                "target": target.field_space_id,
            }
        )
        self.stencil_set = stencil_set
        self.axis = axis
        self.indices = stencil_set.stencil.indices
        self.weights = stencil_set.stencil.weights
        self.valid = stencil_set.stencil.valid
        self.consistency_report = certify_stencil_consistency(stencil_set)
        self.adjoint_report = certify_operator_adjoint(self)
        self.conservation_report = certify_operator_conservation(
            self,
            periodic=stencil_set.kind == "periodic",
        )

    def mv(self, vector: ArrayLike, /) -> Array:
        value = self.source.validate(jnp.asarray(vector))
        moved = jnp.moveaxis(value, self.axis, 0)
        safe_indices = jnp.where(self.valid, self.indices, 0)
        gathered = moved[safe_indices]
        mask_shape = self.valid.shape + (1,) * (gathered.ndim - 2)
        safe_gathered = jnp.where(
            self.valid.reshape(mask_shape),
            gathered,
            jnp.zeros((), dtype=gathered.dtype),
        )
        safe_weights = jnp.where(self.valid, self.weights, 0.0)
        weight_shape = safe_weights.shape + (1,) * (gathered.ndim - 2)
        result = jnp.sum(safe_weights.reshape(weight_shape) * safe_gathered, axis=1)
        result = jnp.moveaxis(result, 0, self.axis)
        return self.target.validate(result)

    def transpose_mv(self, vector: ArrayLike, /) -> Array:
        value = self.target.validate(jnp.asarray(vector))
        moved = jnp.moveaxis(value, self.axis, 0)
        safe_indices = jnp.where(self.valid, self.indices, 0)
        mask_shape = self.valid.shape + (1,) * (moved.ndim - 1)
        safe_weights = jnp.where(self.valid, self.weights, 0.0)
        contributions = safe_weights.reshape(mask_shape) * moved[:, None, ...]
        contributions = jnp.where(
            self.valid.reshape(mask_shape),
            contributions,
            jnp.zeros((), dtype=contributions.dtype),
        )
        source_axis = self.source.shape[self.axis]
        output_shape = (source_axis,) + moved.shape[1:]
        output = (
            jnp.zeros(output_shape, dtype=contributions.dtype)
            .at[safe_indices]
            .add(contributions)
        )
        return self.source.validate(jnp.moveaxis(output, 0, self.axis))

    def adjoint_mv(self, vector: ArrayLike, /) -> Array:
        value = self.target.validate(jnp.asarray(vector))
        target_pairing = self.target.pairing
        source_pairing = self.source.pairing
        if isinstance(target_pairing, DiagonalPairing):
            value = target_pairing.weights * value
        elif not isinstance(target_pairing, EuclideanPairing):
            raise ValueError("Stencil adjoints require Euclidean or diagonal pairings.")
        result = jnp.conj(self.transpose_mv(jnp.conj(value)))
        if isinstance(source_pairing, DiagonalPairing):
            result = result / source_pairing.weights
        elif not isinstance(source_pairing, EuclideanPairing):
            raise ValueError("Stencil adjoints require Euclidean or diagonal pairings.")
        return result

    def _materialize(self, /) -> Array:
        if self.source.size * self.target.size > 4096**2:
            raise ValueError("Stencil materialization exceeds the explicit size budget.")
        identity = jnp.eye(self.source.size, dtype=self.source.dtype).reshape(
            (self.source.size,) + self.source.shape
        )
        columns = jax.vmap(self.mv)(identity).reshape((self.source.size, -1))
        return columns.T


__all__ = ["PreparedStencilOperator", "prepare_linear_stencil"]
