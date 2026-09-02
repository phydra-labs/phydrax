#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from itertools import permutations
from math import factorial

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...linalg import (
    DenseLinearOperator,
    DenseLU,
    FailurePolicy,
    LinearSolvePolicy,
    LinearSystem,
    solve,
)


_LU_POLICY = LinearSolvePolicy(DenseLU(), failure=FailurePolicy("status"))


def _bitmap_axes(bitmap: int, dimension: int, /) -> tuple[int, ...]:
    return tuple(axis for axis in range(dimension) if bitmap & (1 << axis))


def _permutation_sign(values: tuple[int, ...], /) -> int:
    return (
        -1
        if sum(
            values[i] > values[j]
            for i in range(len(values))
            for j in range(i + 1, len(values))
        )
        % 2
        else 1
    )


def _chevalley_vector(metric: Array, axis: int, value: Array, /) -> Array:
    dimension = metric.shape[-1]
    count = 1 << dimension
    output = jnp.zeros_like(value)
    for bitmap in range(count):
        coefficient = value[..., bitmap]
        if not bitmap & (1 << axis):
            lower = bitmap & ((1 << axis) - 1)
            sign = -1 if lower.bit_count() % 2 else 1
            output = output.at[..., bitmap | (1 << axis)].add(sign * coefficient)
        axes = _bitmap_axes(bitmap, dimension)
        for position, contracted_axis in enumerate(axes):
            sign = -1 if position % 2 else 1
            output = output.at[..., bitmap ^ (1 << contracted_axis)].add(
                sign * metric[..., axis, contracted_axis] * coefficient
            )
    return output


class CliffordMetricField(StrictModule):
    metric: Callable[[Array], Array]
    dimension: int = eqx.field(static=True)
    signature: tuple[int, int] = eqx.field(static=True)
    rank_tolerance: float = eqx.field(static=True)
    field_id: str = eqx.field(static=True)

    def __init__(
        self,
        metric: Callable[[Array], Array],
        /,
        *,
        dimension: int,
        signature: tuple[int, int],
        rank_tolerance: float = 1e-8,
        field_id: str,
    ):
        if not callable(metric):
            raise TypeError("metric must be callable.")
        dimension_ = int(dimension)
        signature_ = tuple(int(value) for value in signature)
        if (
            dimension_ < 1
            or dimension_ > 6
            or len(signature_) != 2
            or sum(signature_) != dimension_
        ):
            raise ValueError(
                "Clifford metric field requires fixed dimension <= 6 and a full nondegenerate signature."
            )
        if float(rank_tolerance) <= 0.0 or not field_id:
            raise ValueError("rank_tolerance and field_id must be valid.")
        self.metric = metric
        self.dimension = dimension_
        self.signature = signature_
        self.rank_tolerance = float(rank_tolerance)
        self.field_id = str(field_id)

    def evaluate(self, coordinates: ArrayLike, /) -> tuple[Array, Array]:
        point = jnp.asarray(coordinates)
        metric = jnp.asarray(self.metric(point))
        if metric.shape != point.shape[:-1] + (self.dimension, self.dimension):
            raise ValueError("Clifford metric field returned an incompatible shape.")
        symmetric = jnp.max(jnp.abs(metric - jnp.swapaxes(metric, -1, -2)))
        eigenvalues = jnp.linalg.eigvalsh(metric)
        positive = jnp.sum(eigenvalues > self.rank_tolerance, axis=-1)
        negative = jnp.sum(eigenvalues < -self.rank_tolerance, axis=-1)
        valid = (
            (symmetric <= self.rank_tolerance)
            & jnp.all(positive == self.signature[0])
            & jnp.all(negative == self.signature[1])
        )
        return metric, valid


class PreparedCliffordMetricProduct(StrictModule):
    """Associative Chevalley product on the fixed exterior basis."""

    metric_field: CliffordMetricField
    blade_count: int = eqx.field(static=True)
    permutation_table: tuple[tuple[tuple[int, ...], int], ...] = eqx.field(static=True)

    def __init__(self, metric_field: CliffordMetricField, /):
        if not isinstance(metric_field, CliffordMetricField):
            raise TypeError("metric_field must be a CliffordMetricField.")
        table = []
        for bitmap in range(1 << metric_field.dimension):
            axes = _bitmap_axes(bitmap, metric_field.dimension)
            for order in permutations(axes):
                table.append((tuple(order), _permutation_sign(tuple(order))))
        self.metric_field = metric_field
        self.blade_count = 1 << metric_field.dimension
        self.permutation_table = tuple(table)

    def __call__(
        self, coordinates: ArrayLike, left: ArrayLike, right: ArrayLike, /
    ) -> Array:
        metric, valid = self.metric_field.evaluate(coordinates)
        left_ = jnp.asarray(left)
        right_ = jnp.asarray(right)
        if left_.shape[-1:] != (self.blade_count,) or right_.shape[-1:] != (
            self.blade_count,
        ):
            raise ValueError(
                "Metric Clifford operands must use the full exterior blade layout."
            )
        result = jnp.zeros(
            jnp.broadcast_shapes(left_.shape, right_.shape),
            dtype=jnp.result_type(left_, right_, metric),
        )
        for bitmap in range(self.blade_count):
            axes = _bitmap_axes(bitmap, self.metric_field.dimension)
            quantized = jnp.zeros_like(result)
            for order in permutations(axes):
                term = jnp.broadcast_to(right_, result.shape)
                for axis in reversed(order):
                    term = _chevalley_vector(metric, axis, term)
                quantized = quantized + _permutation_sign(tuple(order)) * term
            quantized = quantized / factorial(len(axes))
            result = (
                result
                + jnp.broadcast_to(left_[..., bitmap, None], result.shape) * quantized
            )
        return eqx.error_if(
            result,
            ~jnp.all(valid),
            "Clifford metric signature/rank left the prepared epoch.",
        )


class CliffordInverseResult(StrictModule):
    value: Array
    left_residual: Array
    right_residual: Array
    valid: Array

    def __init__(
        self,
        value: ArrayLike,
        left_residual: ArrayLike,
        right_residual: ArrayLike,
        valid: ArrayLike,
        /,
    ):
        self.value = jnp.asarray(value)
        self.left_residual = jnp.asarray(left_residual)
        self.right_residual = jnp.asarray(right_residual)
        self.valid = jnp.asarray(valid, dtype=bool)


def invert_multivector(
    product: PreparedCliffordMetricProduct,
    coordinates: ArrayLike,
    value: ArrayLike,
    /,
    *,
    tolerance: float = 1e-8,
) -> CliffordInverseResult:
    multivector = jnp.asarray(value)
    if multivector.shape != (product.blade_count,):
        raise ValueError(
            "Clifford inversion currently requires one unbatched full-layout multivector."
        )
    basis = jnp.eye(product.blade_count, dtype=multivector.dtype)
    left_action = jax.vmap(lambda column: product(coordinates, multivector, column))(
        basis
    ).T
    operator = DenseLinearOperator(
        left_action, operator_id=f"{product.metric_field.field_id}:left-regular"
    )
    identity = jnp.zeros_like(multivector).at[0].set(1)
    solved = solve(
        LinearSystem(operator, problem_id=f"{product.metric_field.field_id}:inverse"),
        identity,
        policy=_LU_POLICY,
    )
    candidate = solved.value
    left = product(coordinates, multivector, candidate)
    right = product(coordinates, candidate, multivector)
    left_residual = jnp.max(jnp.abs(left - identity))
    right_residual = jnp.max(jnp.abs(right - identity))
    valid = (
        solved.diagnostics.converged
        & (left_residual <= tolerance)
        & (right_residual <= tolerance)
    )
    return CliffordInverseResult(
        jnp.where(valid, candidate, jnp.full_like(candidate, jnp.nan)),
        left_residual,
        right_residual,
        valid,
    )


class PinElement(StrictModule):
    value: Array
    parity_residual: Array
    norm_residual: Array
    vector_residual: Array
    metric_residual: Array
    parity: int = eqx.field(static=True)
    valid: Array

    def __init__(
        self,
        product: PreparedCliffordMetricProduct,
        coordinates: ArrayLike,
        value: ArrayLike,
        /,
        *,
        parity: int,
        tolerance: float = 1e-8,
    ):
        parity_ = int(parity)
        if parity_ not in (0, 1):
            raise ValueError("Pin parity must be zero or one.")
        if float(tolerance) <= 0.0:
            raise ValueError("Pin tolerance must be positive.")
        value_ = jnp.asarray(value)
        if value_.shape != (product.blade_count,):
            raise ValueError(
                "Pin membership requires one unbatched full-layout multivector."
            )
        coordinate_values = jnp.asarray(coordinates)
        dimension = product.metric_field.dimension
        metric, metric_valid = product.metric_field.evaluate(coordinate_values)
        if metric.shape != (dimension, dimension):
            raise ValueError("Pin membership requires one unbatched metric evaluation.")
        grades = tuple(bitmap.bit_count() for bitmap in range(product.blade_count))
        forbidden_parity = jnp.asarray(
            tuple(grade % 2 != parity_ for grade in grades), dtype=bool
        )
        parity_residual = jnp.max(jnp.where(forbidden_parity, jnp.abs(value_), 0.0))

        reversion_signs = jnp.asarray(
            tuple((-1) ** (grade * (grade - 1) // 2) for grade in grades),
            dtype=value_.dtype,
        )
        reversed_value = reversion_signs * value_
        left_norm = product(coordinates, value_, reversed_value)
        right_norm = product(coordinates, reversed_value, value_)
        scalar_norm = 0.5 * (left_norm[0] + right_norm[0])
        norm_residual = jnp.max(
            jnp.stack(
                (
                    jnp.max(jnp.abs(left_norm[1:])),
                    jnp.max(jnp.abs(right_norm[1:])),
                    jnp.max(jnp.abs(left_norm - right_norm)),
                    jnp.abs(jnp.abs(scalar_norm) - 1.0),
                )
            )
        )
        safe_scalar_norm = jnp.where(jnp.abs(scalar_norm) > tolerance, scalar_norm, 1.0)
        group_inverse = reversed_value / safe_scalar_norm

        vector_indices = tuple(1 << axis for axis in range(dimension))
        vector_basis = jnp.eye(product.blade_count, dtype=value_.dtype)[
            jnp.asarray(vector_indices)
        ]
        twisted_value = (-1 if parity_ else 1) * value_
        transformed = jax.vmap(
            lambda vector: product(
                coordinates,
                product(coordinates, twisted_value, vector),
                group_inverse,
            )
        )(vector_basis)
        vector_mask = (
            jnp.zeros((product.blade_count,), dtype=bool)
            .at[jnp.asarray(vector_indices)]
            .set(True)
        )
        vector_residual = jnp.max(
            jnp.where(vector_mask[None, :], 0.0, jnp.abs(transformed))
        )
        vector_coordinates = transformed[:, jnp.asarray(vector_indices)]
        metric_residual = jnp.max(
            jnp.abs(vector_coordinates @ metric @ vector_coordinates.T - metric)
        )
        finite = (
            jnp.all(jnp.isfinite(value_))
            & jnp.all(jnp.isfinite(left_norm))
            & jnp.all(jnp.isfinite(right_norm))
            & jnp.all(jnp.isfinite(transformed))
            & jnp.isfinite(metric_residual)
        )
        real_coordinates = (
            jnp.all(jnp.isreal(value_))
            & jnp.all(jnp.isreal(coordinate_values))
            & jnp.all(jnp.isreal(metric))
        )
        valid = (
            metric_valid
            & real_coordinates
            & finite
            & (parity_residual <= tolerance)
            & (norm_residual <= tolerance)
            & (vector_residual <= tolerance)
            & (metric_residual <= tolerance)
        )
        self.value = value_
        self.parity_residual = parity_residual
        self.norm_residual = norm_residual
        self.vector_residual = vector_residual
        self.metric_residual = metric_residual
        self.parity = parity_
        self.valid = valid


class SpinElement(StrictModule):
    value: Array
    parity_residual: Array
    norm_residual: Array
    vector_residual: Array
    metric_residual: Array
    parity: int = eqx.field(static=True)
    valid: Array

    def __init__(
        self,
        product: PreparedCliffordMetricProduct,
        coordinates: ArrayLike,
        value: ArrayLike,
        /,
        *,
        tolerance: float = 1e-8,
    ):
        pin = PinElement(product, coordinates, value, parity=0, tolerance=tolerance)
        self.value = pin.value
        self.parity_residual = pin.parity_residual
        self.norm_residual = pin.norm_residual
        self.vector_residual = pin.vector_residual
        self.metric_residual = pin.metric_residual
        self.parity = 0
        self.valid = pin.valid


class MinimalLeftIdeal(StrictModule):
    idempotent: Array
    basis: Array
    dimension: int = eqx.field(static=True)
    valid: Array

    def __init__(
        self,
        product: PreparedCliffordMetricProduct,
        coordinates: ArrayLike,
        idempotent: ArrayLike,
        /,
        *,
        expected_dimension: int,
        tolerance: float = 1e-8,
    ):
        value = jnp.asarray(idempotent)
        residual = jnp.max(jnp.abs(product(coordinates, value, value) - value))
        basis = jax.vmap(lambda blade: product(coordinates, blade, value))(
            jnp.eye(product.blade_count, dtype=value.dtype)
        ).T
        singular = jnp.linalg.svd(basis, compute_uv=False)
        rank = jnp.sum(singular > tolerance, dtype=jnp.int32)
        valid = (residual <= tolerance) & (rank == int(expected_dimension))
        self.idempotent = value
        self.basis = basis
        self.dimension = int(expected_dimension)
        self.valid = valid


class CliffordCochainProductPlan(StrictModule):
    product: PreparedCliffordMetricProduct
    source_cells: Array
    left_cells: Array
    right_cells: Array
    coefficients: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        product: PreparedCliffordMetricProduct,
        source_cells: ArrayLike,
        left_cells: ArrayLike,
        right_cells: ArrayLike,
        coefficients: ArrayLike,
        /,
        *,
        plan_id: str,
    ):
        arrays = tuple(
            jnp.asarray(value)
            for value in (source_cells, left_cells, right_cells, coefficients)
        )
        if any(value.shape != arrays[0].shape or value.ndim != 1 for value in arrays):
            raise ValueError(
                "Clifford cochain diagonal arrays must be equal rank-1 arrays."
            )
        self.product = product
        self.source_cells = arrays[0].astype(jnp.int32)
        self.left_cells = arrays[1].astype(jnp.int32)
        self.right_cells = arrays[2].astype(jnp.int32)
        self.coefficients = arrays[3]
        self.plan_id = str(plan_id)

    def evaluate(
        self, coordinates: ArrayLike, left: ArrayLike, right: ArrayLike, /
    ) -> Array:
        terms = (
            jax.vmap(lambda a, b: self.product(coordinates, a, b))(
                jnp.asarray(left)[self.left_cells], jnp.asarray(right)[self.right_cells]
            )
            * self.coefficients[:, None]
        )
        count = int(jnp.max(self.source_cells)) + 1
        return (
            jnp.zeros((count, self.product.blade_count), dtype=terms.dtype)
            .at[self.source_cells]
            .add(terms)
        )


class CliffordProjectorEvidence(StrictModule):
    idempotence_residual: Array
    plemelj_residual: Array
    trace_residual: Array
    finite: Array
    valid: Array

    def __init__(
        self,
        *,
        idempotence_residual: ArrayLike,
        plemelj_residual: ArrayLike,
        trace_residual: ArrayLike,
        tolerance: float,
    ):
        self.idempotence_residual = jnp.asarray(idempotence_residual)
        self.plemelj_residual = jnp.asarray(plemelj_residual)
        self.trace_residual = jnp.asarray(trace_residual)
        residuals = jnp.stack(
            (
                self.idempotence_residual,
                self.plemelj_residual,
                self.trace_residual,
            )
        )
        self.finite = jnp.all(jnp.isfinite(residuals))
        self.valid = self.finite & jnp.all(residuals <= tolerance)


class PreparedCauchyCliffordProjector(StrictModule):
    nodes: Array
    weights: Array
    normals: Array
    kernel: Callable[[Array, Array], Array]
    tolerance: float = eqx.field(static=True)
    projector_id: str = eqx.field(static=True)

    def __init__(
        self,
        nodes: ArrayLike,
        weights: ArrayLike,
        normals: ArrayLike,
        kernel: Callable[[Array, Array], Array],
        /,
        *,
        tolerance: float = 1e-6,
        projector_id: str,
    ):
        nodes_ = jnp.asarray(nodes)
        weights_ = jnp.asarray(weights)
        normals_ = jnp.asarray(normals)
        if (
            nodes_.ndim != 2
            or normals_.shape != nodes_.shape
            or weights_.shape != nodes_.shape[:1]
            or not callable(kernel)
        ):
            raise ValueError(
                "Cauchy-Clifford projector boundary arrays are incompatible."
            )
        if float(tolerance) <= 0.0:
            raise ValueError("Projector tolerance must be positive.")
        self.nodes = nodes_
        self.weights = weights_
        self.normals = normals_
        self.kernel = kernel
        self.tolerance = float(tolerance)
        self.projector_id = str(projector_id)

    def apply(self, target: ArrayLike, boundary_values: ArrayLike, /) -> Array:
        values = jnp.asarray(boundary_values)
        if values.shape[0] != self.nodes.shape[0]:
            raise ValueError("Boundary values must match projector quadrature nodes.")
        kernels = jax.vmap(lambda node: self.kernel(jnp.asarray(target), node))(
            self.nodes
        )
        return jnp.sum(
            self.weights.reshape((-1,) + (1,) * (values.ndim - 1)) * kernels * values,
            axis=0,
        )

    def evidence(
        self,
        projector_matrix: ArrayLike,
        /,
        *,
        expected_trace: float,
        plemelj_residual: ArrayLike,
    ) -> CliffordProjectorEvidence:
        matrix = jnp.asarray(projector_matrix)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Projector evidence requires one square matrix.")
        return CliffordProjectorEvidence(
            idempotence_residual=jnp.max(jnp.abs(matrix @ matrix - matrix)),
            plemelj_residual=plemelj_residual,
            trace_residual=jnp.abs(jnp.trace(matrix) - float(expected_trace)),
            tolerance=self.tolerance,
        )


class ConformalCliffordModel(StrictModule):
    dimension: int = eqx.field(static=True)

    def __init__(self, dimension: int, /):
        if int(dimension) < 1:
            raise ValueError("Conformal model dimension must be positive.")
        self.dimension = int(dimension)

    def embed(self, point: ArrayLike, /) -> Array:
        value = jnp.asarray(point)
        if value.shape[-1:] != (self.dimension,):
            raise ValueError("Conformal point has the wrong dimension.")
        squared = jnp.sum(value * value, axis=-1, keepdims=True)
        return jnp.concatenate(
            (value, 0.5 * (1.0 - squared), 0.5 * (1.0 + squared)), axis=-1
        )

    def null_residual(self, embedded: ArrayLike, /) -> Array:
        value = jnp.asarray(embedded)
        return jnp.abs(jnp.sum(value[..., :-1] ** 2, axis=-1) - value[..., -1] ** 2)


class ProjectiveCliffordModel(StrictModule):
    radical_dimension: int = eqx.field(static=True)

    def __init__(self, radical_dimension: int = 1, /):
        if int(radical_dimension) < 1:
            raise ValueError("Projective Clifford model requires a nonzero radical.")
        self.radical_dimension = int(radical_dimension)

    def require_invertible(self, radical_component: ArrayLike, /) -> None:
        if bool(jnp.any(jnp.asarray(radical_component) != 0)):
            raise ValueError("Projective radical components are excluded from inversion.")


__all__ = [
    "CliffordCochainProductPlan",
    "CliffordInverseResult",
    "CliffordMetricField",
    "CliffordProjectorEvidence",
    "ConformalCliffordModel",
    "MinimalLeftIdeal",
    "PinElement",
    "PreparedCauchyCliffordProjector",
    "PreparedCliffordMetricProduct",
    "ProjectiveCliffordModel",
    "SpinElement",
    "invert_multivector",
]
