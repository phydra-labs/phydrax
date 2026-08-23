#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    AbstractLinearOperator,
    AbstractPreconditioner,
    ArraySpace,
    DiagonalPairing,
    DiagonalPreconditioner,
    MultigridCycleKind,
    MultigridCyclePolicy,
    MultigridHierarchy,
    MultigridLevel,
    MultigridPreconditioner,
    OperatorCapabilities,
    OperatorProperties,
    PreconditionerProperties,
)
from .._axis import TensorGridPlan, UniformAxisSpec, UniformCellAxisSpec
from .._tensor_support import PreparedTensorGrid
from ..finite_volume._diffusion import (
    ConservativeDiffusionPlan,
    PreparedConservativeDiffusion,
)
from ._coefficients import fornberg_weights


StructuredMGCompatibility: TypeAlias = Literal["error", "project_rhs"]
StructuredMGGauge: TypeAlias = Literal["zero_mean", "minimum_norm"]

StructuredSmootherKind: TypeAlias = Literal["jacobi", "red_black", "line"]
StructuredCoarsening: TypeAlias = Literal["full", "semi"]


class StructuredTransferReport(StrictModule, NonTrainableState):
    """Constant preservation, conservative restriction, and shape evidence."""

    fine_shape: tuple[int, ...] = eqx.field(static=True)
    coarse_shape: tuple[int, ...] = eqx.field(static=True)
    constant_residual: float = eqx.field(static=True)
    conservation_residual: float | None = eqx.field(static=True)
    passed: bool = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        fine_shape: tuple[int, ...],
        coarse_shape: tuple[int, ...],
        constant_residual: float,
        conservation_residual: float | None,
        transfer_id: str,
    ):
        constant = float(constant_residual)
        conservation = (
            None if conservation_residual is None else float(conservation_residual)
        )
        self.fine_shape = fine_shape
        self.coarse_shape = coarse_shape
        self.constant_residual = constant
        self.conservation_residual = conservation
        self.passed = constant <= 1e-12 and (
            conservation is None or conservation <= 1e-12
        )
        self.report_id = canonical_fingerprint(
            {
                "kind": "structured-transfer-report",
                "transfer": transfer_id,
                "fine_shape": list(fine_shape),
                "coarse_shape": list(coarse_shape),
                "constant_residual": constant,
                "conservation_residual": conservation,
            }
        )


class StructuredTensorTransferOperator(AbstractLinearOperator):
    """Tensor-product structured restriction or prolongation without global matrices."""

    source: ArraySpace
    target: ArraySpace
    axis_matrices: tuple[Array, ...]

    def __init__(
        self,
        source: ArraySpace,
        target: ArraySpace,
        axis_matrices: Sequence[ArrayLike],
        transfer_kind: str,
        /,
    ):
        if not isinstance(source, ArraySpace) or not isinstance(target, ArraySpace):
            raise TypeError("Structured transfers require ArraySpace source and target.")
        matrices = tuple(
            jnp.asarray(value, dtype=source.dtype) for value in axis_matrices
        )
        if len(matrices) != len(source.shape) or len(target.shape) != len(source.shape):
            raise ValueError("Structured transfer matrices must align with tensor rank.")
        if any(
            matrix.shape != (target.shape[axis], source.shape[axis])
            for axis, matrix in enumerate(matrices)
        ):
            raise ValueError("Structured transfer axis matrix has incompatible shape.")
        self.source = source
        self.target = target
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
                "kind": "structured-tensor-transfer",
                "transfer_kind": str(transfer_kind),
                "source": source.space_id,
                "target": target.space_id,
                "matrices": [array_tree_fingerprint(value) for value in matrices],
            }
        )
        self.axis_matrices = matrices

    def mv(self, vector: ArrayLike, /) -> Array:
        value = self.source.validate(jnp.asarray(vector))
        return self.target.validate(_apply_axis_matrices(value, self.axis_matrices))

    def transpose_mv(self, vector: ArrayLike, /) -> Array:
        value = self.target.validate(jnp.asarray(vector))
        matrices = tuple(jnp.swapaxes(value_, 0, 1) for value_ in self.axis_matrices)
        return self.source.validate(_apply_axis_matrices(value, matrices))

    def adjoint_mv(self, vector: ArrayLike, /) -> Array:
        value = self.target.validate(jnp.asarray(vector))
        target_pairing = self.target.pairing
        source_pairing = self.source.pairing
        if isinstance(target_pairing, DiagonalPairing):
            value = target_pairing.weights * value
        result = self.transpose_mv(value)
        if isinstance(source_pairing, DiagonalPairing):
            result = result / source_pairing.weights
        return result

    def _materialize(self, /) -> Array:
        if self.source.size * self.target.size > 4096**2:
            raise ValueError("Transfer materialization exceeds explicit size budget.")
        identity = jnp.eye(self.source.size, dtype=self.source.dtype).reshape(
            (self.source.size,) + self.source.shape
        )
        columns = jax.vmap(self.mv)(identity).reshape((self.source.size, -1))
        return columns.T


def _apply_axis_matrices(
    value: Array,
    matrices: Sequence[Array],
    /,
) -> Array:
    result = value
    for axis, matrix in enumerate(matrices):
        moved = jnp.moveaxis(result, axis, 0)
        moved = jnp.tensordot(matrix, moved, axes=((1,), (0,)))
        result = jnp.moveaxis(moved, 0, axis)
    return result


class StructuredTransferPlan(StrictModule, NonTrainableState):
    """Entity-aware tensor restriction and prolongation between nested grids."""

    fine_grid: PreparedTensorGrid
    coarse_grid: PreparedTensorGrid
    restriction_matrices: tuple[Array, ...]
    prolongation_matrices: tuple[Array, ...]
    report: StructuredTransferReport
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        fine_grid: PreparedTensorGrid,
        coarse_grid: PreparedTensorGrid,
        /,
    ):
        if not isinstance(fine_grid, PreparedTensorGrid) or not isinstance(
            coarse_grid, PreparedTensorGrid
        ):
            raise TypeError("Structured transfer requires two prepared tensor grids.")
        if fine_grid.axis_names != coarse_grid.axis_names or len(fine_grid.shape) != len(
            coarse_grid.shape
        ):
            raise ValueError("Structured transfer grids must share tensor axes.")
        fine_entities = fine_grid.primary_entity_layout.axis_entities
        coarse_entities = coarse_grid.primary_entity_layout.axis_entities
        if fine_entities != coarse_entities:
            raise ValueError("Structured transfer cannot change primary entity kinds.")
        restriction = []
        prolongation = []
        conservative = all(entity == "interval" for entity in fine_entities)
        for axis, entity in enumerate(fine_entities):
            fine_axis = fine_grid.structured_axes[axis]
            coarse_axis = coarse_grid.structured_axes[axis]
            if entity == "interval":
                restriction.append(_cell_restriction(fine_axis, coarse_axis))
                prolongation.append(
                    _point_interpolation(
                        np.asarray(coarse_axis.interval_centers),
                        np.asarray(fine_axis.interval_centers),
                    )
                )
            else:
                restriction.append(
                    _point_injection(
                        np.asarray(fine_axis.point_coordinates),
                        np.asarray(coarse_axis.point_coordinates),
                    )
                )
                prolongation.append(
                    _point_interpolation(
                        np.asarray(coarse_axis.point_coordinates),
                        np.asarray(fine_axis.point_coordinates),
                    )
                )
        restriction_ = tuple(jnp.asarray(value) for value in restriction)
        prolongation_ = tuple(jnp.asarray(value) for value in prolongation)
        constant_residual = max(
            max(
                float(np.max(np.abs(np.asarray(matrix) @ np.ones(matrix.shape[1]) - 1.0)))
                for matrix in restriction_
            ),
            max(
                float(np.max(np.abs(np.asarray(matrix) @ np.ones(matrix.shape[1]) - 1.0)))
                for matrix in prolongation_
            ),
        )
        conservation_residual = None
        if conservative:
            fine_measure = np.asarray(fine_grid.quadrature_weights).reshape((-1,))
            coarse_measure = np.asarray(coarse_grid.quadrature_weights).reshape((-1,))
            global_restriction = np.asarray(restriction_[0])
            for matrix in restriction_[1:]:
                global_restriction = np.kron(global_restriction, np.asarray(matrix))
            conservation_residual = float(
                np.max(np.abs(coarse_measure @ global_restriction - fine_measure))
            )
        identifier = canonical_fingerprint(
            {
                "kind": "structured-transfer-plan",
                "fine": fine_grid.prepared_id,
                "coarse": coarse_grid.prepared_id,
                "restriction_shapes": [list(value.shape) for value in restriction_],
                "prolongation_shapes": [list(value.shape) for value in prolongation_],
            }
        )
        report = StructuredTransferReport(
            fine_shape=fine_grid.shape,
            coarse_shape=coarse_grid.shape,
            constant_residual=constant_residual,
            conservation_residual=conservation_residual,
            transfer_id=identifier,
        )
        if not report.passed:
            raise RuntimeError(
                "Structured transfer failed constant/conservation evidence."
            )
        self.fine_grid = fine_grid
        self.coarse_grid = coarse_grid
        self.restriction_matrices = restriction_
        self.prolongation_matrices = prolongation_
        self.report = report
        self.plan_id = identifier

    def prepare(
        self,
        fine_space: ArraySpace,
        coarse_space: ArraySpace,
        /,
    ) -> tuple[StructuredTensorTransferOperator, StructuredTensorTransferOperator]:
        restriction = StructuredTensorTransferOperator(
            fine_space,
            coarse_space,
            self.restriction_matrices,
            "restriction",
        )
        prolongation = StructuredTensorTransferOperator(
            coarse_space,
            fine_space,
            self.prolongation_matrices,
            "prolongation",
        )
        return restriction, prolongation

    def restrict_array(self, values: ArrayLike, /) -> Array:
        value = jnp.asarray(values)
        if value.shape[: len(self.fine_grid.shape)] != self.fine_grid.shape:
            raise ValueError("Restricted array must begin with the fine-grid shape.")
        return _apply_axis_matrices(value, self.restriction_matrices)


def _cell_edges(axis, /) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray([axis.bounds[0]]),
            np.asarray(axis.bounds[0] + np.cumsum(np.asarray(axis.interval_widths))),
        )
    )


def _cell_restriction(fine_axis, coarse_axis, /) -> np.ndarray:
    fine_edges = _cell_edges(fine_axis)
    coarse_edges = _cell_edges(coarse_axis)
    matrix = np.zeros(
        (coarse_axis.interval_centers.size, fine_axis.interval_centers.size)
    )
    for coarse in range(matrix.shape[0]):
        for fine in range(matrix.shape[1]):
            overlap = max(
                0.0,
                min(coarse_edges[coarse + 1], fine_edges[fine + 1])
                - max(coarse_edges[coarse], fine_edges[fine]),
            )
            matrix[coarse, fine] = overlap / (
                coarse_edges[coarse + 1] - coarse_edges[coarse]
            )
    return matrix


def _point_injection(fine: np.ndarray, coarse: np.ndarray, /) -> np.ndarray:
    matrix = np.zeros((coarse.size, fine.size))
    for row, coordinate in enumerate(coarse):
        index = int(np.argmin(np.abs(fine - coordinate)))
        if abs(fine[index] - coordinate) > 1e-10:
            raise ValueError("Nodal coarse coordinates must be nested in the fine grid.")
        matrix[row, index] = 1.0
    return matrix


def _point_interpolation(source: np.ndarray, target: np.ndarray, /) -> np.ndarray:
    width = min(2, source.size)
    matrix = np.zeros((target.size, source.size))
    for row, coordinate in enumerate(target):
        insertion = int(np.searchsorted(source, coordinate))
        start = int(np.clip(insertion - 1, 0, source.size - width))
        indices = np.arange(start, start + width)
        matrix[row, indices] = fornberg_weights(source[indices], coordinate, 0)
    return matrix


class _DenseCoarsePreconditioner(AbstractPreconditioner):
    inverse: Array

    def __init__(self, operator: AbstractLinearOperator, /):
        matrix = operator._materialize()
        inverse = jnp.linalg.pinv(matrix, rtol=1e-12)
        self.space = operator.source
        self.properties = PreconditionerProperties(
            linear=True,
            stationary=True,
            self_adjoint=False,
            evidence={"linear": "construction", "stationary": "construction"},
        )
        self.preconditioner_id = canonical_fingerprint(
            {
                "kind": "structured-dense-coarse-solve",
                "operator": operator.operator_id,
            }
        )
        self.inverse = inverse

    def apply(
        self,
        residual,
        /,
        *,
        iteration: ArrayLike | None = None,
    ):
        del iteration
        coordinates = self.space.flatten(self.space.validate(residual))
        return self.space.unflatten(self.inverse @ coordinates)


class StructuredMultigridResult(StrictModule, NonTrainableState):
    value: Array
    residual_norms: Array
    converged: Array
    cycles: int = eqx.field(static=True)

    def __init__(self, value: Array, residual_norms: Array, tolerance: float, /):
        self.value = value
        self.residual_norms = residual_norms
        scale = jnp.maximum(1.0, residual_norms[0])
        self.converged = residual_norms[-1] <= float(tolerance) * scale
        self.cycles = int(residual_norms.size - 1)


class _RedBlackPreconditioner(AbstractPreconditioner):
    operator: AbstractLinearOperator
    inverse_diagonal: Array
    color_masks: tuple[Array, Array]
    relaxation: float = eqx.field(static=True)

    def __init__(
        self,
        operator: AbstractLinearOperator,
        diagonal: Array,
        shape: tuple[int, ...],
        relaxation: float,
        /,
    ):
        coordinates = jnp.indices(shape)
        parity = jnp.sum(coordinates, axis=0) % 2
        self.operator = operator
        self.inverse_diagonal = 1.0 / diagonal
        self.color_masks = (parity == 0, parity == 1)
        self.relaxation = float(relaxation)
        self.space = operator.source
        self.properties = PreconditionerProperties(
            linear=True,
            stationary=True,
            evidence={"linear": "construction", "stationary": "construction"},
        )
        self.preconditioner_id = canonical_fingerprint(
            {
                "kind": "structured-red-black-smoother",
                "operator": operator.operator_id,
                "shape": list(shape),
                "relaxation": float(relaxation),
            }
        )

    def apply(
        self,
        residual,
        /,
        *,
        iteration: ArrayLike | None = None,
    ):
        del iteration
        rhs = self.space.validate(residual)
        estimate = jnp.zeros_like(rhs)
        for mask in self.color_masks:
            defect = rhs - self.operator.mv(estimate)
            estimate = estimate + jnp.where(
                mask,
                self.relaxation * self.inverse_diagonal * defect,
                0.0,
            )
        return estimate


class _LinePreconditioner(AbstractPreconditioner):
    lower: Array
    diagonal: Array
    upper: Array
    axis: int = eqx.field(static=True)

    def __init__(
        self,
        operator: AbstractLinearOperator,
        diffusion: PreparedConservativeDiffusion,
        axis: int,
        /,
    ):
        lower, diagonal, upper = _line_coefficients(diffusion, axis)
        self.lower = lower
        self.diagonal = diagonal
        self.upper = upper
        self.axis = int(axis)
        self.space = operator.source
        self.properties = PreconditionerProperties(
            linear=True,
            stationary=True,
            evidence={"linear": "construction", "stationary": "construction"},
        )
        self.preconditioner_id = canonical_fingerprint(
            {
                "kind": "structured-line-smoother",
                "operator": operator.operator_id,
                "axis": int(axis),
            }
        )

    def apply(
        self,
        residual,
        /,
        *,
        iteration: ArrayLike | None = None,
    ):
        del iteration
        rhs = self.space.validate(residual)
        return _tridiagonal_solve(
            self.lower,
            self.diagonal,
            self.upper,
            rhs,
            self.axis,
        )


def _line_coefficients(
    diffusion: PreparedConservativeDiffusion,
    axis: int,
    /,
) -> tuple[Array, Array, Array]:
    grid = diffusion.plan.grid
    structured_axis = grid.structured_axes[axis]
    if structured_axis.periodic:
        raise ValueError("Structured line smoothing currently requires a bounded axis.")
    diagonal = -diffusion.diagonal()
    face = diffusion.plan.interpolation.interpolate(
        diffusion.coefficient[..., axis, axis],
        grid.axis_names[axis],
    )
    centers = structured_axis.interval_centers
    distances = jnp.diff(centers)
    widths = structured_axis.interval_widths
    lower = jnp.zeros(grid.shape, dtype=diagonal.dtype)
    upper = jnp.zeros(grid.shape, dtype=diagonal.dtype)
    if grid.shape[axis] > 1:
        face_interior = jnp.take(
            face,
            jnp.arange(1, face.shape[axis] - 1),
            axis=axis,
        )
        distance_shape = [1] * len(grid.shape)
        distance_shape[axis] = int(distances.size)
        coupling = face_interior / distances.reshape(distance_shape)
        lower_widths = jnp.take(widths, jnp.arange(1, widths.size))
        upper_widths = jnp.take(widths, jnp.arange(widths.size - 1))
        lower_shape = [1] * len(grid.shape)
        upper_shape = [1] * len(grid.shape)
        lower_shape[axis] = int(lower_widths.size)
        upper_shape[axis] = int(upper_widths.size)
        lower_index: list[slice | int] = [slice(None)] * len(grid.shape)
        upper_index: list[slice | int] = [slice(None)] * len(grid.shape)
        lower_index[axis] = slice(1, grid.shape[axis])
        upper_index[axis] = slice(0, grid.shape[axis] - 1)
        lower = lower.at[tuple(lower_index)].set(
            -coupling / lower_widths.reshape(lower_shape)
        )
        upper = upper.at[tuple(upper_index)].set(
            -coupling / upper_widths.reshape(upper_shape)
        )
    return lower, diagonal, upper


def _tridiagonal_solve(
    lower: Array,
    diagonal: Array,
    upper: Array,
    right_hand_side: Array,
    axis: int,
    /,
) -> Array:
    lower_ = jnp.moveaxis(lower, axis, 0).reshape((lower.shape[axis], -1))
    diagonal_ = jnp.moveaxis(diagonal, axis, 0).reshape((diagonal.shape[axis], -1))
    upper_ = jnp.moveaxis(upper, axis, 0).reshape((upper.shape[axis], -1))
    rhs_ = jnp.moveaxis(right_hand_side, axis, 0).reshape(
        (right_hand_side.shape[axis], -1)
    )
    first_c = upper_[0] / diagonal_[0]
    first_d = rhs_[0] / diagonal_[0]

    def forward(carry, values):
        previous_c, previous_d = carry
        lower_value, diagonal_value, upper_value, rhs_value = values
        denominator = diagonal_value - lower_value * previous_c
        current_c = upper_value / denominator
        current_d = (rhs_value - lower_value * previous_d) / denominator
        return (current_c, current_d), (current_c, current_d)

    _, history = jax.lax.scan(
        forward,
        (first_c, first_d),
        (lower_[1:], diagonal_[1:], upper_[1:], rhs_[1:]),
    )
    c_values = jnp.concatenate((first_c[None], history[0]), axis=0)
    d_values = jnp.concatenate((first_d[None], history[1]), axis=0)

    def backward(next_value, values):
        c_value, d_value = values
        current = d_value - c_value * next_value
        return current, current

    _, reversed_solution = jax.lax.scan(
        backward,
        d_values[-1],
        (c_values[:-1][::-1], d_values[:-1][::-1]),
    )
    solution = jnp.concatenate(
        (reversed_solution[::-1], d_values[-1:]),
        axis=0,
    )
    moved_shape = jnp.moveaxis(right_hand_side, axis, 0).shape
    solution = solution.reshape(moved_shape)
    return jnp.moveaxis(solution, 0, axis)


class StructuredMultigridPlan(StrictModule, NonTrainableState):
    """Geometric cell-centered diffusion hierarchy with entity-aware transfers."""

    finest_operator: PreparedConservativeDiffusion
    compatibility: StructuredMGCompatibility = eqx.field(static=True)
    gauge: StructuredMGGauge = eqx.field(static=True)
    minimum_coarse_points: int = eqx.field(static=True)
    maximum_levels: int = eqx.field(static=True)
    coarsening: StructuredCoarsening = eqx.field(static=True)
    relaxation: float = eqx.field(static=True)
    smoother: StructuredSmootherKind = eqx.field(static=True)
    line_axis: str | None = eqx.field(static=True)
    pre_smoothing: int = eqx.field(static=True)
    post_smoothing: int = eqx.field(static=True)
    cycle_kind: MultigridCycleKind = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        finest_operator: PreparedConservativeDiffusion,
        /,
        *,
        minimum_coarse_points: int = 4,
        maximum_levels: int = 10,
        coarsening: StructuredCoarsening = "full",
        relaxation: float = 2.0 / 3.0,
        compatibility: StructuredMGCompatibility = "error",
        gauge: StructuredMGGauge = "zero_mean",
        smoother: StructuredSmootherKind = "jacobi",
        line_axis: str | None = None,
        pre_smoothing: int = 2,
        post_smoothing: int = 2,
        cycle_kind: MultigridCycleKind = "v",
    ):
        if not isinstance(finest_operator, PreparedConservativeDiffusion):
            raise TypeError(
                "Structured multigrid requires PreparedConservativeDiffusion."
            )
        minimum = int(minimum_coarse_points)
        maximum = int(maximum_levels)
        relaxation_ = float(relaxation)
        if (
            minimum < 2
            or maximum < 2
            or coarsening not in ("full", "semi")
            or compatibility not in ("error", "project_rhs")
            or gauge not in ("zero_mean", "minimum_norm")
            or smoother not in ("jacobi", "red_black", "line")
            or not np.isfinite(relaxation_)
            or relaxation_ <= 0.0
            or relaxation_ >= 2.0
            or int(pre_smoothing) < 0
            or int(post_smoothing) < 0
        ):
            raise ValueError("Structured multigrid controls are invalid.")
        line_axis_ = None if line_axis is None else str(line_axis)
        if smoother == "line":
            if line_axis_ not in finest_operator.plan.grid.axis_names:
                raise ValueError("Line smoother requires one bounded grid axis name.")
            line_index = finest_operator.plan.grid.axis_names.index(line_axis_)
            if finest_operator.plan.grid.structured_axes[line_index].periodic:
                raise ValueError("Line smoother currently requires a bounded axis.")
        elif line_axis_ is not None:
            raise ValueError("line_axis is valid only for the line smoother.")
        MultigridCyclePolicy(cycle_kind)
        self.finest_operator = finest_operator
        self.minimum_coarse_points = minimum
        self.maximum_levels = maximum
        self.coarsening = coarsening
        self.relaxation = relaxation_
        self.compatibility = compatibility
        self.gauge = gauge
        self.smoother = smoother
        self.line_axis = line_axis_
        self.pre_smoothing = int(pre_smoothing)
        self.post_smoothing = int(post_smoothing)
        self.cycle_kind = cycle_kind
        self.plan_id = canonical_fingerprint(
            {
                "kind": "structured-multigrid-plan",
                "finest": finest_operator.operator_id,
                "minimum_coarse_points": minimum,
                "maximum_levels": maximum,
                "coarsening": coarsening,
                "relaxation": relaxation_,
                "smoothing": [int(pre_smoothing), int(post_smoothing)],
                "smoother": smoother,
                "line_axis": line_axis_,
                "cycle": cycle_kind,
                "compatibility": compatibility,
                "gauge": gauge,
            }
        )

    def prepare(self, /) -> "PreparedStructuredMultigrid":
        return PreparedStructuredMultigrid(self)


class PreparedStructuredMultigrid(StrictModule, NonTrainableState):
    plan: StructuredMultigridPlan
    grids: tuple[PreparedTensorGrid, ...]
    diffusion_operators: tuple[PreparedConservativeDiffusion, ...]
    level_operators: tuple[AbstractLinearOperator, ...]
    transfers: tuple[StructuredTransferPlan, ...]
    hierarchy: MultigridHierarchy
    preconditioner: MultigridPreconditioner
    nullspace_dimension: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: StructuredMultigridPlan, /):
        if not isinstance(plan, StructuredMultigridPlan):
            raise TypeError("plan must be StructuredMultigridPlan.")
        grids = [plan.finest_operator.plan.grid]
        diffusions = [plan.finest_operator]
        transfers = []
        coefficient = plan.finest_operator.coefficient
        while len(grids) < plan.maximum_levels:
            coarse_grid = _coarsen_grid(
                grids[-1],
                plan.minimum_coarse_points,
                coarsening=plan.coarsening,
            )
            if coarse_grid is None:
                break
            transfer = StructuredTransferPlan(grids[-1], coarse_grid)
            coefficient = transfer.restrict_array(coefficient)
            boundaries = {
                axis: pair
                for axis, pair in zip(
                    coarse_grid.axis_names,
                    diffusions[-1].plan.boundaries,
                    strict=True,
                )
            }
            coarse_diffusion = ConservativeDiffusionPlan(
                coarse_grid,
                boundaries=boundaries,
                interpolation=diffusions[-1].plan.interpolation.kind,
            ).prepare(coefficient)
            grids.append(coarse_grid)
            transfers.append(transfer)
            diffusions.append(coarse_diffusion)
        if len(grids) < 2:
            raise ValueError("Structured multigrid could not construct a coarse level.")
        level_operators = tuple(-1.0 * value for value in diffusions)
        transfer_operators = tuple(
            transfer.prepare(fine.source, coarse.source)
            for transfer, fine, coarse in zip(
                transfers,
                level_operators[:-1],
                level_operators[1:],
                strict=True,
            )
        )
        levels = []
        for index, operator in enumerate(level_operators):
            if index == len(level_operators) - 1:
                smoother = _DenseCoarsePreconditioner(operator)
                restriction = None
                prolongation = None
            else:
                diagonal_array = -diffusions[index].diagonal()
                if plan.smoother == "jacobi":
                    smoother = DiagonalPreconditioner(
                        diagonal_array.reshape((-1,)) / plan.relaxation,
                        space=operator.source,
                        positive_definite=True,
                    )
                elif plan.smoother == "red_black":
                    smoother = _RedBlackPreconditioner(
                        operator,
                        diagonal_array,
                        grids[index].shape,
                        plan.relaxation,
                    )
                else:
                    smoother = _LinePreconditioner(
                        operator,
                        diffusions[index],
                        grids[index].axis_names.index(str(plan.line_axis)),
                    )
                restriction, prolongation = transfer_operators[index]
            levels.append(
                MultigridLevel(
                    operator,
                    smoother,
                    restriction=restriction,
                    prolongation=prolongation,
                    pre_smoothing=plan.pre_smoothing,
                    post_smoothing=plan.post_smoothing,
                )
            )
        hierarchy = MultigridHierarchy(tuple(levels))
        preconditioner = MultigridPreconditioner(
            hierarchy,
            cycle_policy=MultigridCyclePolicy(plan.cycle_kind),
        )
        self.plan = plan
        self.grids = tuple(grids)
        self.diffusion_operators = tuple(diffusions)
        self.level_operators = level_operators
        self.transfers = tuple(transfers)
        self.hierarchy = hierarchy
        self.preconditioner = preconditioner
        nullspace_dimension = int(
            all(
                lower.kind in ("neumann", "periodic")
                and upper.kind in ("neumann", "periodic")
                for lower, upper in plan.finest_operator.plan.boundaries
            )
        )
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-structured-multigrid",
                "plan": plan.plan_id,
                "hierarchy": hierarchy.hierarchy_id,
                "grids": [value.prepared_id for value in grids],
            }
        )
        self.nullspace_dimension = nullspace_dimension

    def apply(self, residual: ArrayLike, /) -> Array:
        return self.preconditioner.apply(residual)

    def solve(
        self,
        right_hand_side: ArrayLike,
        /,
        *,
        cycles: int = 12,
        tolerance: float = 1e-8,
    ) -> StructuredMultigridResult:
        count = int(cycles)
        if count <= 0 or tolerance <= 0.0:
            raise ValueError("Multigrid cycles/tolerance must be positive.")
        operator = self.level_operators[0]
        if not isinstance(operator.source, ArraySpace) or not isinstance(
            operator.target, ArraySpace
        ):
            raise RuntimeError("Structured multigrid lost its array level space.")
        rhs = operator.target.validate(jnp.asarray(right_hand_side))
        if self.nullspace_dimension:
            measure = self.grids[0].quadrature_weights
            mass = jnp.sum(measure)
            incompatible = jnp.sum(measure * rhs)
            scale = jnp.maximum(1.0, jnp.sum(jnp.abs(measure * rhs)))
            if self.plan.compatibility == "error":
                rhs = eqx.error_if(
                    rhs,
                    jnp.abs(incompatible) > tolerance * scale,
                    "Structured multigrid RHS is incompatible with its nullspace.",
                )
            else:
                rhs = rhs - incompatible / mass
        value = jnp.zeros(operator.source.shape, dtype=operator.source.dtype)
        norms = [jnp.linalg.norm(operator.target.flatten(rhs))]
        for _ in range(count):
            residual = rhs - operator.mv(value)
            value = value + self.preconditioner.apply(residual)
            if self.nullspace_dimension:
                if self.plan.gauge == "zero_mean":
                    value = value - jnp.sum(
                        self.grids[0].quadrature_weights * value
                    ) / jnp.sum(self.grids[0].quadrature_weights)
                else:
                    value = value - jnp.mean(value)
            norms.append(
                jnp.linalg.norm(operator.target.flatten(rhs - operator.mv(value)))
            )
        return StructuredMultigridResult(
            value,
            jnp.asarray(norms),
            tolerance,
        )


def _coarsen_grid(
    grid: PreparedTensorGrid,
    minimum: int,
    /,
    *,
    coarsening: StructuredCoarsening,
) -> PreparedTensorGrid | None:
    counts = list(grid.shape)
    candidates = [
        max(minimum, count // 2) if count // 2 >= minimum and count > minimum else count
        for count in counts
    ]
    if coarsening == "semi":
        largest = max(counts)
        candidates = [
            candidate if count == largest else count
            for count, candidate in zip(counts, candidates, strict=True)
        ]
    if tuple(candidates) == tuple(counts):
        return None
    specs = []
    for count, axis in zip(candidates, grid.structured_axes, strict=True):
        if axis.primary_entity == "interval":
            specs.append(UniformCellAxisSpec(count, periodic=axis.periodic))
        else:
            point_count = count if count % 2 == 1 else count + 1
            specs.append(
                UniformAxisSpec(
                    point_count,
                    endpoint=not axis.periodic,
                    periodic=axis.periodic,
                )
            )
    bounds = jnp.stack(tuple(axis.bounds for axis in grid.structured_axes), axis=-1)
    return TensorGridPlan(tuple(specs), axis_names=grid.axis_names).prepare(bounds)


__all__ = [
    "PreparedStructuredMultigrid",
    "StructuredCoarsening",
    "StructuredMultigridPlan",
    "StructuredMGCompatibility",
    "StructuredMGGauge",
    "StructuredMultigridResult",
    "StructuredTensorTransferOperator",
    "StructuredTransferPlan",
    "StructuredTransferReport",
]
