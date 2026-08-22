#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    AbstractLinearOperator,
    ArraySpace,
    DiagonalPairing,
    EuclideanPairing,
    OperatorCapabilities,
    OperatorProperties,
)
from .._tensor_support import PreparedTensorGrid
from ._certification import FDConservationReport, FDStabilityReport


FaceInterpolationKind: TypeAlias = Literal["arithmetic", "harmonic", "upwind", "callable"]
ConservativeBoundaryKind: TypeAlias = Literal["periodic", "dirichlet", "neumann", "robin"]
AdvectionForm: TypeAlias = Literal["advective", "conservative", "skew", "split_energy"]
AdvectionReconstruction: TypeAlias = Literal["arithmetic", "upwind"]


class ConservativeBoundaryCondition(StrictModule, NonTrainableState):
    """One scalar boundary law αu + β∂u/∂x = target."""

    kind: ConservativeBoundaryKind = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    beta: float = eqx.field(static=True)
    condition_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: ConservativeBoundaryKind,
        /,
        *,
        alpha: float | None = None,
        beta: float | None = None,
    ):
        if kind not in ("periodic", "dirichlet", "neumann", "robin"):
            raise ValueError("Unknown conservative boundary kind.")
        alpha_ = (
            1.0
            if kind in ("dirichlet", "robin") and alpha is None
            else float(alpha or 0.0)
        )
        beta_ = (
            1.0 if kind in ("neumann", "robin") and beta is None else float(beta or 0.0)
        )
        if not np.isfinite(alpha_) or not np.isfinite(beta_):
            raise ValueError("Conservative boundary coefficients must be finite.")
        if kind == "dirichlet" and alpha_ == 0.0:
            raise ValueError("Dirichlet alpha must be nonzero.")
        if kind == "neumann" and beta_ == 0.0:
            raise ValueError("Neumann beta must be nonzero.")
        if kind == "robin" and (alpha_ == 0.0 or beta_ == 0.0):
            raise ValueError("Robin alpha and beta must be nonzero.")
        if kind == "periodic" and (alpha_ != 0.0 or beta_ != 0.0):
            raise ValueError("Periodic boundaries cannot carry physical coefficients.")
        self.kind = kind
        self.alpha = alpha_
        self.beta = beta_
        self.condition_id = canonical_fingerprint(
            {
                "kind": "conservative-boundary-condition",
                "boundary_kind": kind,
                "alpha": alpha_,
                "beta": beta_,
            }
        )


def _condition(value: ConservativeBoundaryCondition | ConservativeBoundaryKind, /):
    return (
        value
        if isinstance(value, ConservativeBoundaryCondition)
        else ConservativeBoundaryCondition(value)
    )


def _normalize_boundaries(
    grid: PreparedTensorGrid,
    boundaries: Mapping[
        str,
        tuple[
            ConservativeBoundaryCondition | ConservativeBoundaryKind,
            ConservativeBoundaryCondition | ConservativeBoundaryKind,
        ],
    ]
    | None,
    /,
) -> tuple[tuple[ConservativeBoundaryCondition, ConservativeBoundaryCondition], ...]:
    supplied = {} if boundaries is None else dict(boundaries)
    unknown = set(supplied).difference(grid.axis_names)
    if unknown:
        raise ValueError(
            f"Conservative boundaries reference unknown axes {sorted(unknown)!r}."
        )
    output = []
    for axis_name, axis in zip(grid.axis_names, grid.structured_axes, strict=True):
        default = ("periodic", "periodic") if axis.periodic else ("neumann", "neumann")
        raw = supplied.get(axis_name, default)
        lower, upper = _condition(raw[0]), _condition(raw[1])
        if axis.periodic != (lower.kind == upper.kind == "periodic"):
            raise ValueError(
                "Grid periodicity and conservative boundary semantics must agree."
            )
        if (lower.kind == "periodic") != (upper.kind == "periodic"):
            raise ValueError("Periodicity must be declared on both boundary sides.")
        output.append((lower, upper))
    return tuple(output)


def _boundary_value(
    value: ArrayLike,
    shape: tuple[int, ...],
    axis: int,
    dtype: Any,
    /,
) -> Array:
    target_shape = shape[:axis] + shape[axis + 1 :]
    result = jnp.asarray(value, dtype=dtype)
    if result.shape == ():
        return jnp.broadcast_to(result, target_shape)
    if result.shape != target_shape:
        raise ValueError(f"Boundary value must have shape {target_shape} or be scalar.")
    return result


def _expand_axis(value: Array, axis: int, /) -> Array:
    return jnp.expand_dims(value, axis=axis)


class FaceCoefficientPlan(StrictModule, NonTrainableState):
    """Cell-to-face interpolation with explicit discontinuity semantics."""

    grid: PreparedTensorGrid
    kind: FaceInterpolationKind = eqx.field(static=True)
    function: Callable[[Array, Array, Array | None], Array] | None = eqx.field(
        static=True
    )
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid: PreparedTensorGrid,
        /,
        *,
        kind: FaceInterpolationKind = "harmonic",
        function: Callable[[Array, Array, Array | None], Array] | None = None,
    ):
        if not isinstance(grid, PreparedTensorGrid):
            raise TypeError("Face coefficient plan requires PreparedTensorGrid.")
        if kind not in ("arithmetic", "harmonic", "upwind", "callable"):
            raise ValueError("Unknown face interpolation kind.")
        if (kind == "callable") != (function is not None):
            raise ValueError("Callable face interpolation requires exactly one function.")
        self.grid = grid
        self.kind = kind
        self.function = function
        self.plan_id = canonical_fingerprint(
            {
                "kind": "face-coefficient-plan",
                "grid": grid.prepared_id,
                "interpolation": kind,
                "function": None if function is None else repr(function),
            }
        )

    def _combine(
        self,
        left: Array,
        right: Array,
        velocity: Array | None,
        /,
    ) -> Array:
        if self.kind == "arithmetic":
            return 0.5 * (left + right)
        if self.kind == "harmonic":
            denominator = left + right
            safe = jnp.where(denominator != 0.0, denominator, 1.0)
            return jnp.where(denominator != 0.0, 2.0 * left * right / safe, 0.0)
        if self.kind == "upwind":
            if velocity is None:
                raise ValueError("Upwind face interpolation requires face velocity.")
            return jnp.where(velocity >= 0.0, left, right)
        if self.function is None:
            raise RuntimeError("Callable face interpolation lost its function.")
        return jnp.asarray(self.function(left, right, velocity))

    def interpolate(
        self,
        values: ArrayLike,
        axis: str,
        /,
        *,
        velocity: ArrayLike | None = None,
    ) -> Array:
        value = jnp.asarray(values)
        rank = len(self.grid.shape)
        if value.shape[:rank] != self.grid.shape:
            raise ValueError("Cell coefficient must begin with the primary grid shape.")
        axis_index = self.grid.axis_names.index(axis)
        structured_axis = self.grid.structured_axes[axis_index]
        trailing = value.shape[rank:]
        face_shape = self.grid.faces(axis).shape + trailing
        velocity_ = None if velocity is None else jnp.asarray(velocity)
        if velocity_ is not None and velocity_.shape != face_shape:
            raise ValueError("Face velocity shape does not match interpolated values.")
        if structured_axis.periodic:
            left = jnp.roll(value, 1, axis=axis_index)
            right = value
            result = self._combine(left, right, velocity_)
        else:
            lower = jnp.take(value, jnp.asarray([0]), axis=axis_index)
            upper = jnp.take(value, jnp.asarray([-1]), axis=axis_index)
            left = jnp.take(
                value,
                jnp.arange(self.grid.shape[axis_index] - 1),
                axis=axis_index,
            )
            right = jnp.take(
                value,
                jnp.arange(1, self.grid.shape[axis_index]),
                axis=axis_index,
            )
            velocity_interior = (
                None
                if velocity_ is None
                else jnp.take(
                    velocity_,
                    jnp.arange(1, face_shape[axis_index] - 1),
                    axis=axis_index,
                )
            )
            interior = self._combine(left, right, velocity_interior)
            result = jnp.concatenate((lower, interior, upper), axis=axis_index)
        if result.shape != face_shape:
            raise RuntimeError("Face interpolation produced the wrong entity shape.")
        return result


def _normalize_tensor_coefficient(
    coefficient: ArrayLike,
    grid: PreparedTensorGrid,
    /,
) -> Array:
    dimension = len(grid.shape)
    value = jnp.asarray(coefficient)
    if value.shape == () or value.shape == grid.shape:
        scalar = jnp.broadcast_to(value, grid.shape)
        identity = jnp.eye(dimension, dtype=scalar.dtype)
        return scalar[..., None, None] * identity
    if value.shape == grid.shape + (dimension,):
        identity = jnp.eye(dimension, dtype=value.dtype)
        return value[..., :, None] * identity
    if value.shape == (dimension, dimension):
        return jnp.broadcast_to(value, grid.shape + (dimension, dimension))
    if value.shape == grid.shape + (dimension, dimension):
        return value
    raise ValueError(
        "Diffusion coefficient must be scalar, cell scalar, cell diagonal, or cell tensor."
    )


class ConservativeDiffusionPlan(StrictModule, NonTrainableState):
    """Conservative cell-centered scalar/tensor diffusion preparation."""

    grid: PreparedTensorGrid
    boundaries: tuple[
        tuple[ConservativeBoundaryCondition, ConservativeBoundaryCondition], ...
    ]
    interpolation: FaceCoefficientPlan
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid: PreparedTensorGrid,
        /,
        *,
        boundaries: Mapping[
            str,
            tuple[
                ConservativeBoundaryCondition | ConservativeBoundaryKind,
                ConservativeBoundaryCondition | ConservativeBoundaryKind,
            ],
        ]
        | None = None,
        interpolation: FaceInterpolationKind = "harmonic",
    ):
        if not isinstance(grid, PreparedTensorGrid):
            raise TypeError("Conservative diffusion requires PreparedTensorGrid.")
        if grid.primary_entity_layout.layout_id != grid.cells().layout_id:
            raise ValueError("Conservative diffusion requires an interval-primary grid.")
        boundaries_ = _normalize_boundaries(grid, boundaries)
        face_plan = FaceCoefficientPlan(grid, kind=interpolation)
        self.grid = grid
        self.boundaries = boundaries_
        self.interpolation = face_plan
        self.plan_id = canonical_fingerprint(
            {
                "kind": "conservative-diffusion-plan",
                "grid": grid.prepared_id,
                "boundaries": [
                    [lower.condition_id, upper.condition_id]
                    for lower, upper in boundaries_
                ],
                "interpolation": face_plan.plan_id,
            }
        )

    def prepare(self, coefficient: ArrayLike, /) -> "PreparedConservativeDiffusion":
        return PreparedConservativeDiffusion(self, coefficient)


class PreparedConservativeDiffusion(AbstractLinearOperator):
    """Conservative cell-to-face flux followed by face-to-cell divergence."""

    source: ArraySpace
    target: ArraySpace
    plan: ConservativeDiffusionPlan
    coefficient: Array
    conservation_report: FDConservationReport
    stability_report: FDStabilityReport

    def __init__(
        self,
        plan: ConservativeDiffusionPlan,
        coefficient: ArrayLike,
        /,
    ):
        if not isinstance(plan, ConservativeDiffusionPlan):
            raise TypeError("plan must be ConservativeDiffusionPlan.")
        coefficient_ = _normalize_tensor_coefficient(coefficient, plan.grid)
        host = np.asarray(coefficient_)
        if np.any(~np.isfinite(host)):
            raise ValueError("Diffusion coefficient must be finite.")
        symmetric_residual = float(np.max(np.abs(host - np.swapaxes(host, -1, -2))))
        symmetric = 0.5 * (host + np.swapaxes(host, -1, -2))
        eigenvalues = np.linalg.eigvalsh(symmetric)
        diagonal = np.zeros_like(symmetric)
        diagonal_indices = np.arange(symmetric.shape[-1])
        diagonal[..., diagonal_indices, diagonal_indices] = symmetric[
            ..., diagonal_indices, diagonal_indices
        ]
        diagonal_residual = float(np.max(np.abs(symmetric - diagonal)))
        positive = (
            bool(np.all(eigenvalues > 0.0))
            and symmetric_residual <= 1e-10
            and diagonal_residual <= 1e-12
        )
        field = plan.grid.field_space("conservative_diffusion")
        if not isinstance(field.vector_space, ArraySpace):
            raise TypeError("Conservative diffusion requires an ArraySpace field.")
        self.source = field.vector_space
        self.target = field.vector_space
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
                "kind": "prepared-conservative-diffusion",
                "plan": plan.plan_id,
                "coefficient_shape": list(coefficient_.shape),
            }
        )
        self.plan = plan
        self.coefficient = coefficient_
        constant = jnp.ones(plan.grid.shape, dtype=self.source.dtype)
        constant_residual = float(np.asarray(jnp.max(jnp.abs(self.mv(constant)))))
        global_balance = float(
            np.asarray(jnp.abs(jnp.sum(plan.grid.quadrature_weights * self.mv(constant))))
        )
        self.conservation_report = FDConservationReport(
            constant_state_residual=constant_residual,
            global_balance_residual=global_balance,
            tolerance=1e-10,
            operator_id=self.operator_id,
        )
        self.stability_report = FDStabilityReport(
            "diffusion_seminegativity",
            residual=0.0 if positive else None,
            tolerance=1e-10,
            assumptions=(
                "symmetric positive coefficient",
                "homogeneous declared boundary data",
            ),
            evidence="analytic" if positive else "unknown",
            subject_id=self.operator_id,
        )

    def _boundary_gradient(
        self,
        value: Array,
        axis_index: int,
        side: Literal["lower", "upper"],
        condition: ConservativeBoundaryCondition,
        target: ArrayLike,
        /,
    ) -> Array:
        axis = self.plan.grid.structured_axes[axis_index]
        state_index = 0 if side == "lower" else value.shape[axis_index] - 1
        state = jnp.take(value, state_index, axis=axis_index)
        target_ = _boundary_value(target, value.shape, axis_index, value.dtype)
        distance = (
            axis.interval_centers[0] - axis.bounds[0]
            if side == "lower"
            else axis.bounds[1] - axis.interval_centers[-1]
        )
        if condition.kind == "dirichlet":
            boundary_state = target_ / condition.alpha
            return (
                (state - boundary_state) / distance
                if side == "lower"
                else (boundary_state - state) / distance
            )
        if condition.kind == "neumann":
            return target_ / condition.beta
        denominator = condition.beta + (
            -condition.alpha * distance if side == "lower" else condition.alpha * distance
        )
        denominator = eqx.error_if(
            denominator,
            jnp.abs(denominator) <= np.finfo(float).eps,
            "Conservative Robin boundary is singular at the cell center.",
        )
        return (target_ - condition.alpha * state) / denominator

    def _normal_gradient(
        self,
        value: Array,
        axis_index: int,
        targets: tuple[ArrayLike, ArrayLike],
        /,
    ) -> Array:
        grid = self.plan.grid
        axis = grid.structured_axes[axis_index]
        if axis.periodic:
            previous = jnp.roll(value, 1, axis=axis_index)
            widths = axis.interval_widths
            reshape = [1] * value.ndim
            reshape[axis_index] = int(widths.size)
            return (value - previous) / widths.reshape(reshape)
        lower_condition, upper_condition = self.plan.boundaries[axis_index]
        lower = _expand_axis(
            self._boundary_gradient(
                value,
                axis_index,
                "lower",
                lower_condition,
                targets[0],
            ),
            axis_index,
        )
        upper = _expand_axis(
            self._boundary_gradient(
                value,
                axis_index,
                "upper",
                upper_condition,
                targets[1],
            ),
            axis_index,
        )
        left = jnp.take(
            value,
            jnp.arange(value.shape[axis_index] - 1),
            axis=axis_index,
        )
        right = jnp.take(
            value,
            jnp.arange(1, value.shape[axis_index]),
            axis=axis_index,
        )
        distances = jnp.diff(axis.interval_centers)
        reshape = [1] * value.ndim
        reshape[axis_index] = int(distances.size)
        interior = (right - left) / distances.reshape(reshape)
        return jnp.concatenate((lower, interior, upper), axis=axis_index)

    def _cell_gradient(self, value: Array, axis_index: int, /) -> Array:
        axis = self.plan.grid.structured_axes[axis_index]
        if axis.periodic:
            width = axis.interval_widths
            reshape = [1] * value.ndim
            reshape[axis_index] = int(width.size)
            return (
                jnp.roll(value, -1, axis=axis_index) - jnp.roll(value, 1, axis=axis_index)
            ) / (2.0 * width.reshape(reshape))
        if value.shape[axis_index] < 3:
            raise ValueError("Tangential cell gradient requires at least three cells.")
        lower = (
            jnp.take(value, 1, axis=axis_index) - jnp.take(value, 0, axis=axis_index)
        ) / (axis.interval_centers[1] - axis.interval_centers[0])
        upper = (
            jnp.take(value, -1, axis=axis_index) - jnp.take(value, -2, axis=axis_index)
        ) / (axis.interval_centers[-1] - axis.interval_centers[-2])
        center = jnp.take(
            value,
            jnp.arange(2, value.shape[axis_index]),
            axis=axis_index,
        ) - jnp.take(
            value,
            jnp.arange(value.shape[axis_index] - 2),
            axis=axis_index,
        )
        distances = axis.interval_centers[2:] - axis.interval_centers[:-2]
        reshape = [1] * value.ndim
        reshape[axis_index] = int(distances.size)
        center = center / distances.reshape(reshape)
        return jnp.concatenate(
            (
                _expand_axis(lower, axis_index),
                center,
                _expand_axis(upper, axis_index),
            ),
            axis=axis_index,
        )

    def _coefficient(self, coefficient: ArrayLike, /) -> Array:
        return _normalize_tensor_coefficient(coefficient, self.plan.grid)

    def diagonal_with_coefficient(
        self,
        coefficient: ArrayLike,
        /,
    ) -> Array:
        grid = self.plan.grid
        coefficient_ = self._coefficient(coefficient)
        diagonal = jnp.zeros(grid.shape, dtype=coefficient_.dtype)
        for axis_index, axis_name in enumerate(grid.axis_names):
            structured_axis = grid.structured_axes[axis_index]
            widths = structured_axis.interval_widths
            width_shape = [1] * len(grid.shape)
            width_shape[axis_index] = int(widths.size)
            cell_width = widths.reshape(width_shape)
            face_coefficient = self.plan.interpolation.interpolate(
                coefficient_[..., axis_index, axis_index],
                axis_name,
            )
            if structured_axis.periodic:
                diagonal = (
                    diagonal
                    - (face_coefficient + jnp.roll(face_coefficient, -1, axis=axis_index))
                    / cell_width**2
                )
                continue
            centers = structured_axis.interval_centers
            center_distance = jnp.diff(centers)
            distance_shape = [1] * len(grid.shape)
            distance_shape[axis_index] = int(center_distance.size)
            interior_distance = center_distance.reshape(distance_shape)
            lower_condition, upper_condition = self.plan.boundaries[axis_index]
            lower_distance = centers[0] - structured_axis.bounds[0]
            upper_distance = structured_axis.bounds[1] - centers[-1]

            def boundary_derivative(
                condition: ConservativeBoundaryCondition,
                distance: Array,
                side: Literal["lower", "upper"],
            ) -> Array:
                if condition.kind == "dirichlet":
                    return 1.0 / distance if side == "lower" else -1.0 / distance
                if condition.kind == "neumann":
                    return jnp.asarray(0.0)
                denominator = condition.beta + (
                    -condition.alpha * distance
                    if side == "lower"
                    else condition.alpha * distance
                )
                return -condition.alpha / denominator

            lower_face = jnp.take(face_coefficient, 0, axis=axis_index)
            upper_face = jnp.take(
                face_coefficient,
                face_coefficient.shape[axis_index] - 1,
                axis=axis_index,
            )
            lower_value = (
                -lower_face
                * boundary_derivative(
                    lower_condition,
                    lower_distance,
                    "lower",
                )
                / jnp.take(widths, 0)
            )
            upper_value = (
                upper_face
                * boundary_derivative(
                    upper_condition,
                    upper_distance,
                    "upper",
                )
                / jnp.take(widths, widths.size - 1)
            )
            contribution = jnp.zeros(grid.shape, dtype=coefficient_.dtype)
            lower_index: list[slice | int] = [slice(None)] * len(grid.shape)
            upper_index: list[slice | int] = [slice(None)] * len(grid.shape)
            lower_index[axis_index] = 0
            upper_index[axis_index] = grid.shape[axis_index] - 1
            contribution = contribution.at[tuple(lower_index)].add(lower_value)
            contribution = contribution.at[tuple(upper_index)].add(upper_value)
            if grid.shape[axis_index] > 1:
                lower_interior = (
                    jnp.take(
                        face_coefficient,
                        jnp.arange(1, face_coefficient.shape[axis_index] - 1),
                        axis=axis_index,
                    )
                    / interior_distance
                )
                left_cells = jnp.take(
                    cell_width,
                    jnp.arange(cell_width.shape[axis_index] - 1),
                    axis=axis_index,
                )
                right_cells = jnp.take(
                    cell_width,
                    jnp.arange(1, cell_width.shape[axis_index]),
                    axis=axis_index,
                )
                left_index: list[slice | int] = [slice(None)] * len(grid.shape)
                right_index: list[slice | int] = [slice(None)] * len(grid.shape)
                left_index[axis_index] = slice(0, grid.shape[axis_index] - 1)
                right_index[axis_index] = slice(1, grid.shape[axis_index])
                contribution = contribution.at[tuple(left_index)].add(
                    -lower_interior / left_cells
                )
                contribution = contribution.at[tuple(right_index)].add(
                    -lower_interior / right_cells
                )
            diagonal = diagonal + contribution
        return diagonal

    def diagonal(self, /) -> Array:
        return self.diagonal_with_coefficient(self.coefficient)

    def fluxes(
        self,
        value: Array,
        coefficient: ArrayLike,
        boundary_values: Mapping[str, tuple[ArrayLike, ArrayLike]] | None,
        /,
    ) -> tuple[Array, ...]:
        grid = self.plan.grid
        coefficient_ = self._coefficient(coefficient)
        targets = {} if boundary_values is None else dict(boundary_values)
        unknown = set(targets).difference(grid.axis_names)
        if unknown:
            raise ValueError(
                f"Diffusion boundary data names unknown axes {sorted(unknown)!r}."
            )
        cell_gradients = tuple(
            self._cell_gradient(value, axis) for axis in range(len(grid.shape))
        )
        output = []
        for normal_axis, axis_name in enumerate(grid.axis_names):
            normal = self._normal_gradient(
                value,
                normal_axis,
                targets.get(axis_name, (0.0, 0.0)),
            )
            flux = jnp.zeros_like(normal)
            for derivative_axis in range(len(grid.shape)):
                gradient = (
                    normal
                    if derivative_axis == normal_axis
                    else self.plan.interpolation.interpolate(
                        cell_gradients[derivative_axis],
                        axis_name,
                    )
                )
                face_coefficient = self.plan.interpolation.interpolate(
                    coefficient_[..., normal_axis, derivative_axis],
                    axis_name,
                )
                flux = flux + face_coefficient * gradient
            output.append(flux)
        return tuple(output)

    def divergence(self, fluxes: Sequence[Array], /) -> Array:
        grid = self.plan.grid
        if len(tuple(fluxes)) != len(grid.shape):
            raise ValueError("Diffusion requires one normal flux per axis.")
        result = jnp.zeros(grid.shape, dtype=jnp.result_type(*tuple(fluxes)))
        for axis_index, flux in enumerate(fluxes):
            structured_axis = grid.structured_axes[axis_index]
            widths = structured_axis.interval_widths
            reshape = [1] * len(grid.shape)
            reshape[axis_index] = int(widths.size)
            if structured_axis.periodic:
                difference = jnp.roll(flux, -1, axis=axis_index) - flux
            else:
                upper = jnp.take(
                    flux,
                    jnp.arange(1, flux.shape[axis_index]),
                    axis=axis_index,
                )
                lower = jnp.take(
                    flux,
                    jnp.arange(flux.shape[axis_index] - 1),
                    axis=axis_index,
                )
                difference = upper - lower
            result = result + difference / widths.reshape(reshape)
        return result

    def apply_with_coefficient(
        self,
        vector: ArrayLike,
        coefficient: ArrayLike,
        /,
        *,
        boundary_values: Mapping[str, tuple[ArrayLike, ArrayLike]] | None = None,
    ) -> Array:
        value = self.source.validate(jnp.asarray(vector))
        return self.divergence(self.fluxes(value, coefficient, boundary_values))

    def apply(
        self,
        vector: ArrayLike,
        /,
        *,
        boundary_values: Mapping[str, tuple[ArrayLike, ArrayLike]] | None = None,
    ) -> Array:
        return self.apply_with_coefficient(
            vector,
            self.coefficient,
            boundary_values=boundary_values,
        )

    def mv(self, vector: ArrayLike, /) -> Array:
        return self.apply(vector)

    def transpose_mv(self, vector: ArrayLike, /) -> Array:
        value = self.target.validate(jnp.asarray(vector))
        zero = jnp.zeros(self.source.shape, dtype=self.source.dtype)
        return jax.linear_transpose(self.mv, zero)(value)[0]

    def adjoint_mv(self, vector: ArrayLike, /) -> Array:
        value = self.target.validate(jnp.asarray(vector))
        target_pairing = self.target.pairing
        source_pairing = self.source.pairing
        if isinstance(target_pairing, DiagonalPairing):
            value = target_pairing.weights * value
        elif not isinstance(target_pairing, EuclideanPairing):
            raise ValueError("Diffusion adjoint requires diagonal or Euclidean pairing.")
        result = jnp.conj(self.transpose_mv(jnp.conj(value)))
        if isinstance(source_pairing, DiagonalPairing):
            result = result / source_pairing.weights
        elif not isinstance(source_pairing, EuclideanPairing):
            raise ValueError("Diffusion adjoint requires diagonal or Euclidean pairing.")
        return result

    def _materialize(self, /) -> Array:
        if self.source.size * self.target.size > 4096**2:
            raise ValueError("Diffusion materialization exceeds explicit size budget.")
        identity = jnp.eye(self.source.size, dtype=self.source.dtype).reshape(
            (self.source.size,) + self.source.shape
        )
        columns = jax.vmap(self.mv)(identity).reshape((self.source.size, -1))
        return columns.T


class ConservativeAdvectionPlan(StrictModule, NonTrainableState):
    """Explicit conservative, advective, or split scalar transport form."""

    grid: PreparedTensorGrid
    form: AdvectionForm = eqx.field(static=True)
    reconstruction: AdvectionReconstruction = eqx.field(static=True)
    boundaries: tuple[
        tuple[ConservativeBoundaryCondition, ConservativeBoundaryCondition], ...
    ]
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid: PreparedTensorGrid,
        /,
        *,
        form: AdvectionForm = "conservative",
        reconstruction: AdvectionReconstruction = "upwind",
        boundaries: Mapping[
            str,
            tuple[
                ConservativeBoundaryCondition | ConservativeBoundaryKind,
                ConservativeBoundaryCondition | ConservativeBoundaryKind,
            ],
        ]
        | None = None,
    ):
        if not isinstance(grid, PreparedTensorGrid):
            raise TypeError("Conservative advection requires PreparedTensorGrid.")
        if grid.primary_entity_layout.layout_id != grid.cells().layout_id:
            raise ValueError("Conservative advection requires an interval-primary grid.")
        if form not in ("advective", "conservative", "skew", "split_energy"):
            raise ValueError("Unknown advection form.")
        if reconstruction not in ("arithmetic", "upwind"):
            raise ValueError("Unknown advection reconstruction.")
        boundaries_ = _normalize_boundaries(grid, boundaries)
        self.grid = grid
        self.form = form
        self.reconstruction = reconstruction
        self.boundaries = boundaries_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "conservative-advection-plan",
                "grid": grid.prepared_id,
                "form": form,
                "reconstruction": reconstruction,
                "boundaries": [
                    [lower.condition_id, upper.condition_id]
                    for lower, upper in boundaries_
                ],
            }
        )

    def prepare(self, velocity: ArrayLike | Sequence[ArrayLike], /):
        return PreparedConservativeAdvection(self, velocity)


class PreparedConservativeAdvection(StrictModule):
    """Prepared face velocity and explicit scalar transport form."""

    plan: ConservativeAdvectionPlan
    face_velocity: tuple[Array, ...]
    geometry: PreparedConservativeDiffusion
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: ConservativeAdvectionPlan,
        velocity: ArrayLike | Sequence[ArrayLike],
        /,
    ):
        if not isinstance(plan, ConservativeAdvectionPlan):
            raise TypeError("plan must be ConservativeAdvectionPlan.")
        faces = self._prepare_velocity(plan, velocity)
        geometry = ConservativeDiffusionPlan(
            plan.grid,
            boundaries={
                axis: pair
                for axis, pair in zip(
                    plan.grid.axis_names,
                    plan.boundaries,
                    strict=True,
                )
            },
            interpolation="arithmetic",
        ).prepare(1.0)
        self.plan = plan
        self.face_velocity = faces
        self.geometry = geometry
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-conservative-advection",
                "plan": plan.plan_id,
                "velocity_shapes": [list(value.shape) for value in faces],
            }
        )

    @staticmethod
    def _prepare_velocity(
        plan: ConservativeAdvectionPlan,
        velocity: ArrayLike | Sequence[ArrayLike],
        /,
    ) -> tuple[Array, ...]:
        dimension = len(plan.grid.shape)
        if isinstance(velocity, (tuple, list)):
            faces = tuple(jnp.asarray(value) for value in velocity)
            if len(faces) != dimension or any(
                value.shape != plan.grid.faces(axis).shape
                for value, axis in zip(
                    faces,
                    plan.grid.axis_names,
                    strict=True,
                )
            ):
                raise ValueError(
                    "Face velocity tuple must align with every normal face layout."
                )
            return faces
        cell_velocity = jnp.asarray(velocity)
        if cell_velocity.shape != plan.grid.shape + (dimension,):
            raise ValueError("Cell velocity must have one trailing spatial component.")
        interpolation = FaceCoefficientPlan(plan.grid, kind="arithmetic")
        return tuple(
            interpolation.interpolate(cell_velocity[..., axis], axis_name)
            for axis, axis_name in enumerate(plan.grid.axis_names)
        )

    def _resolve_velocity(
        self,
        velocity: ArrayLike | Sequence[ArrayLike],
        /,
    ) -> tuple[Array, ...]:
        return self._prepare_velocity(self.plan, velocity)

    def _face_state(
        self,
        state: Array,
        axis_index: int,
        targets: tuple[ArrayLike, ArrayLike],
        velocity: Array,
        /,
    ) -> Array:
        grid = self.plan.grid
        axis = grid.structured_axes[axis_index]
        if axis.periodic:
            left = jnp.roll(state, 1, axis=axis_index)
            right = state
            return (
                0.5 * (left + right)
                if self.plan.reconstruction == "arithmetic"
                else jnp.where(velocity >= 0.0, left, right)
            )
        lower_condition, upper_condition = self.plan.boundaries[axis_index]
        lower_state = jnp.take(state, jnp.asarray([0]), axis=axis_index)
        upper_state = jnp.take(state, jnp.asarray([-1]), axis=axis_index)
        lower_target = _expand_axis(
            _boundary_value(targets[0], state.shape, axis_index, state.dtype),
            axis_index,
        )
        upper_target = _expand_axis(
            _boundary_value(targets[1], state.shape, axis_index, state.dtype),
            axis_index,
        )
        lower_external = (
            lower_target / lower_condition.alpha
            if lower_condition.kind == "dirichlet"
            else lower_state
        )
        upper_external = (
            upper_target / upper_condition.alpha
            if upper_condition.kind == "dirichlet"
            else upper_state
        )
        left = jnp.concatenate((lower_external, state), axis=axis_index)
        right = jnp.concatenate((state, upper_external), axis=axis_index)
        if self.plan.reconstruction == "arithmetic":
            return 0.5 * (left + right)
        return jnp.where(velocity >= 0.0, left, right)

    def _flux_divergence(
        self,
        state: Array,
        boundary_values: Mapping[str, tuple[ArrayLike, ArrayLike]],
        face_velocity: tuple[Array, ...],
        /,
    ) -> Array:
        fluxes = tuple(
            velocity
            * self._face_state(
                state,
                axis,
                boundary_values.get(axis_name, (0.0, 0.0)),
                velocity,
            )
            for axis, (axis_name, velocity) in enumerate(
                zip(self.plan.grid.axis_names, face_velocity, strict=True)
            )
        )
        return self.geometry.divergence(fluxes)

    def _advective(
        self,
        state: Array,
        face_velocity: tuple[Array, ...],
        /,
    ) -> Array:
        result = jnp.zeros_like(state)
        for axis, velocity in enumerate(face_velocity):
            gradient = self.geometry._cell_gradient(state, axis)
            cell_velocity = (
                0.5 * (velocity + jnp.roll(velocity, -1, axis=axis))
                if self.plan.grid.structured_axes[axis].periodic
                else 0.5
                * (
                    jnp.take(
                        velocity,
                        jnp.arange(velocity.shape[axis] - 1),
                        axis=axis,
                    )
                    + jnp.take(
                        velocity,
                        jnp.arange(1, velocity.shape[axis]),
                        axis=axis,
                    )
                )
            )
            result = result + cell_velocity * gradient
        return result

    def _apply_faces(
        self,
        state: Array,
        face_velocity: tuple[Array, ...],
        boundary_values: Mapping[str, tuple[ArrayLike, ArrayLike]],
        /,
    ) -> Array:
        conservative = self._flux_divergence(
            state,
            boundary_values,
            face_velocity,
        )
        if self.plan.form == "conservative":
            return conservative
        advective = self._advective(state, face_velocity)
        if self.plan.form == "advective":
            return advective
        return 0.5 * (conservative + advective)

    def apply(
        self,
        state: ArrayLike,
        /,
        *,
        boundary_values: Mapping[str, tuple[ArrayLike, ArrayLike]] | None = None,
    ) -> Array:
        value = jnp.asarray(state)
        if value.shape != self.plan.grid.shape:
            raise ValueError("Advection state must match the cell grid shape.")
        targets = {} if boundary_values is None else dict(boundary_values)
        return self._apply_faces(value, self.face_velocity, targets)

    def apply_with_velocity(
        self,
        state: ArrayLike,
        velocity: ArrayLike | Sequence[ArrayLike],
        /,
        *,
        boundary_values: Mapping[str, tuple[ArrayLike, ArrayLike]] | None = None,
    ) -> Array:
        value = jnp.asarray(state)
        if value.shape != self.plan.grid.shape:
            raise ValueError("Advection state must match the cell grid shape.")
        targets = {} if boundary_values is None else dict(boundary_values)
        return self._apply_faces(
            value,
            self._resolve_velocity(velocity),
            targets,
        )


__all__ = [
    "AdvectionForm",
    "AdvectionReconstruction",
    "ConservativeAdvectionPlan",
    "ConservativeBoundaryCondition",
    "ConservativeBoundaryKind",
    "ConservativeDiffusionPlan",
    "FaceCoefficientPlan",
    "FaceInterpolationKind",
    "PreparedConservativeAdvection",
    "PreparedConservativeDiffusion",
]
