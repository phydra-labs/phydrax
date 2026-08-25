#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Mapping
from fractions import Fraction
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._doc import DOC_KEY0
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._model import AbstractArrayModel, StructuredDerivativeProvider
from ...metrix.clifford import (
    basis_blade_product,
    CliffordAlgebraSpec,
    CliffordBladeLayout,
)
from ._core import (
    AbstractTrefftzBasis,
    SimilarityNormalization,
    TrefftzResourceBudget,
    TrefftzResourceEvidence,
    TRIAL_SPACE_CERTIFICATE_KEY,
    TRIAL_SPACE_REPRESENTATION_KEY,
    TrialSpaceCertificate,
)
from ._polynomial import _multiindices


def _monogenic_rank(dimension: int, degree: int, blade_count: int, /) -> int:
    return blade_count * math.comb(dimension + degree - 2, degree)


def _rational_nullspace(matrix: np.ndarray, /) -> np.ndarray:
    rows, columns = matrix.shape
    if rows == 0:
        return np.eye(columns, dtype=float)
    values = [
        [Fraction(int(matrix[row, column])) for column in range(columns)]
        for row in range(rows)
    ]
    pivot_columns: list[int] = []
    pivot_row = 0
    for column in range(columns):
        selected = next(
            (row for row in range(pivot_row, rows) if values[row][column]),
            None,
        )
        if selected is None:
            continue
        values[pivot_row], values[selected] = values[selected], values[pivot_row]
        pivot = values[pivot_row][column]
        values[pivot_row] = [value / pivot for value in values[pivot_row]]
        for row in range(rows):
            if row == pivot_row:
                continue
            factor = values[row][column]
            if factor:
                values[row] = [
                    value - factor * source
                    for value, source in zip(values[row], values[pivot_row])
                ]
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == rows:
            break
    free_columns = tuple(
        column for column in range(columns) if column not in set(pivot_columns)
    )
    basis = np.zeros((columns, len(free_columns)), dtype=float)
    for basis_column, free_column in enumerate(free_columns):
        basis[free_column, basis_column] = 1.0
        for row, pivot_column in enumerate(pivot_columns):
            basis[pivot_column, basis_column] = float(-values[row][free_column])
    return basis


def _dirac_block(
    algebra: CliffordAlgebraSpec,
    layout: CliffordBladeLayout,
    degree: int,
    /,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    exponents = _multiindices(degree, algebra.dimension)
    blade_count = layout.blade_count
    column_count = len(exponents) * blade_count
    if degree == 0:
        matrix = np.zeros((0, column_count), dtype=np.int64)
    else:
        lower = _multiindices(degree - 1, algebra.dimension)
        lower_lookup = {value: index for index, value in enumerate(lower)}
        matrix = np.zeros((len(lower) * blade_count, column_count), dtype=np.int64)
        for monomial, alpha in enumerate(exponents):
            for axis, exponent in enumerate(alpha):
                if exponent == 0:
                    continue
                beta = list(alpha)
                beta[axis] -= 1
                lower_position = lower_lookup[tuple(beta)]
                reciprocal_sign = algebra.diagonal[axis]
                for blade_position, bitmap in enumerate(layout.bitmaps):
                    coefficient, output_bitmap = basis_blade_product(
                        algebra,
                        1 << axis,
                        bitmap,
                    )
                    output_position = layout.position(output_bitmap)
                    matrix[
                        lower_position * blade_count + output_position,
                        monomial * blade_count + blade_position,
                    ] += reciprocal_sign * exponent * coefficient
    nullspace = _rational_nullspace(matrix)
    expected_rank = _monogenic_rank(
        algebra.dimension,
        degree,
        blade_count,
    )
    if nullspace.shape != (column_count, expected_rank):
        raise RuntimeError(
            "Exact monogenic nullspace rank disagrees with the Fischer dimension."
        )
    product = matrix @ nullspace
    residual = 0.0 if product.size == 0 else float(np.max(np.abs(product)))
    operator_scale = 1.0 if matrix.size == 0 else float(np.max(np.abs(matrix)))
    basis_scale = max(float(np.max(np.abs(nullspace))), 1.0)
    tolerance = (
        256.0
        * np.finfo(np.float64).eps
        * operator_scale
        * basis_scale
        * max(column_count, expected_rank, 1)
    )
    coefficients = nullspace.reshape((len(exponents), blade_count, expected_rank))
    return (
        np.asarray(exponents, dtype=np.int32),
        np.transpose(coefficients, (0, 2, 1)),
        residual,
        tolerance,
    )


class MonogenicPolynomialBasis(AbstractTrefftzBasis):
    """Canonical full-Clifford polynomial basis in the kernel of the left Dirac operator."""

    algebra: CliffordAlgebraSpec
    layout: CliffordBladeLayout
    normalization: SimilarityNormalization
    maximum_degree: int = eqx.field(static=True)
    exponent_blocks: tuple[Array, ...]
    coefficient_blocks: tuple[Array, ...]
    _rank: int = eqx.field(static=True)
    _basis_id: str = eqx.field(static=True)
    _certificate: TrialSpaceCertificate
    _resource_evidence: TrefftzResourceEvidence

    def __init__(
        self,
        algebra: CliffordAlgebraSpec,
        maximum_degree: int,
        /,
        *,
        normalization: SimilarityNormalization | None = None,
        resources: TrefftzResourceBudget | None = None,
    ):
        if not isinstance(algebra, CliffordAlgebraSpec):
            raise TypeError("algebra must be a CliffordAlgebraSpec.")
        if not algebra.nondegenerate:
            raise ValueError("Monogenic basis requires a nondegenerate Clifford algebra.")
        degree = int(maximum_degree)
        if degree != maximum_degree or degree < 0:
            raise ValueError("maximum_degree must be a nonnegative integer.")
        normalization_ = (
            SimilarityNormalization(jnp.zeros((algebra.dimension,)))
            if normalization is None
            else normalization
        )
        if not isinstance(normalization_, SimilarityNormalization):
            raise TypeError("normalization must be SimilarityNormalization or None.")
        if normalization_.dimension != algebra.dimension:
            raise ValueError(
                "Monogenic basis normalization dimension does not match algebra."
            )
        resources_ = TrefftzResourceBudget() if resources is None else resources
        if not isinstance(resources_, TrefftzResourceBudget):
            raise TypeError("resources must be TrefftzResourceBudget or None.")
        layout = CliffordBladeLayout.full(algebra)
        rank = sum(
            _monogenic_rank(algebra.dimension, value, layout.blade_count)
            for value in range(degree + 1)
        )
        monomial_count = sum(
            len(_multiindices(value, algebra.dimension)) for value in range(degree + 1)
        )
        basis_entries = sum(
            len(_multiindices(value, algebra.dimension))
            * _monogenic_rank(algebra.dimension, value, layout.blade_count)
            * layout.blade_count
            for value in range(degree + 1)
        )
        resource_evidence = resources_.check(
            rank=rank,
            monomials=monomial_count,
            basis_entries=basis_entries,
            basis_bytes=8 * basis_entries + 4 * monomial_count * algebra.dimension,
        )
        blocks = tuple(
            _dirac_block(algebra, layout, value) for value in range(degree + 1)
        )
        exponent_blocks = tuple(jnp.asarray(value[0]) for value in blocks)
        coefficient_blocks = tuple(jnp.asarray(value[1], dtype=float) for value in blocks)
        residual = max(value[2] for value in blocks)
        tolerance = max(value[3] for value in blocks)
        basis_id = canonical_fingerprint(
            {
                "kind": "monogenic-polynomial-basis-v1",
                "algebra": algebra.algebra_id,
                "layout": layout.layout_id,
                "maximum_degree": degree,
                "normalization": normalization_.normalization_id,
                "blocks": array_tree_fingerprint((exponent_blocks, coefficient_blocks)),
                "resources": resource_evidence.evidence_id,
            }
        )
        parameters = {
            "maximum_degree": degree,
            "left_dirac": 1,
            **{f"signature_{axis}": value for axis, value in enumerate(algebra.diagonal)},
        }
        certificate = TrialSpaceCertificate(
            equation_family="dirac",
            ambient_dimension=algebra.dimension,
            field_shape=(layout.blade_count,),
            construction="canonical-left-monogenic-polynomial-nullspace",
            equation_parameters=parameters,
            normalization_id=normalization_.normalization_id,
            basis_id=basis_id,
            representation_id=algebra.algebra_id,
            rank=rank,
            assumptions=(
                "constant diagonal nondegenerate metric",
                "left Dirac convention",
                "full real Clifford coefficient algebra",
            ),
            construction_residual=residual,
            construction_tolerance=tolerance,
        )
        self.algebra = algebra
        self.layout = layout
        self.normalization = normalization_
        self.maximum_degree = degree
        self.exponent_blocks = exponent_blocks
        self.coefficient_blocks = coefficient_blocks
        self._rank = rank
        self._basis_id = basis_id
        self._certificate = certificate
        self._resource_evidence = resource_evidence

    @property
    def dimension(self) -> int:
        return self.algebra.dimension

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def dtype(self) -> jnp.dtype:
        return self.coefficient_blocks[0].dtype

    @property
    def basis_id(self) -> str:
        return self._basis_id

    @property
    def certificate(self) -> TrialSpaceCertificate:
        return self._certificate

    @property
    def resource_evidence(self) -> TrefftzResourceEvidence:
        return self._resource_evidence

    def evaluate(self, point: ArrayLike, /) -> Array:
        normalized = self.normalization(point)
        features = []
        for exponents, coefficients in zip(
            self.exponent_blocks,
            self.coefficient_blocks,
        ):
            monomials = jnp.prod(normalized[None, :] ** exponents, axis=1)
            features.append(jnp.einsum("m,mrb->rb", monomials, coefficients))
        return jnp.concatenate(features, axis=0)

    def evaluate_partial(
        self,
        point: ArrayLike,
        axis: int,
        order: int = 1,
        /,
    ) -> Array:
        normalized = self.normalization(point)
        axis_ = int(axis)
        order_ = int(order)
        if not 0 <= axis_ < self.dimension:
            raise ValueError("Monogenic partial axis is out of range.")
        if order_ < 0:
            raise ValueError("Monogenic partial order must be nonnegative.")
        features = []
        for exponents, coefficients in zip(
            self.exponent_blocks,
            self.coefficient_blocks,
        ):
            powers = exponents[:, axis_]
            factor = jnp.ones(powers.shape, dtype=normalized.dtype)
            for offset in range(order_):
                factor = factor * jnp.maximum(powers - offset, 0)
            reduced = exponents.at[:, axis_].set(jnp.maximum(powers - order_, 0))
            monomials = (
                factor
                * jnp.prod(normalized[None, :] ** reduced, axis=1)
                / self.normalization.scale.astype(normalized.dtype) ** order_
            )
            features.append(jnp.einsum("m,mrb->rb", monomials, coefficients))
        return jnp.concatenate(features, axis=0)


class LinearMonogenicField(AbstractArrayModel, StructuredDerivativeProvider):
    """Trainable real coefficients over a fixed monogenic multivector basis."""

    basis: MonogenicPolynomialBasis
    coefficients: Array
    channels: int = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int | tuple[int, int] = eqx.field(static=True)

    def __init__(
        self,
        basis: MonogenicPolynomialBasis,
        /,
        *,
        channels: int = 1,
        initial_scale: float = 0.0,
        key: Array = DOC_KEY0,
    ):
        if not isinstance(basis, MonogenicPolynomialBasis):
            raise TypeError("basis must be MonogenicPolynomialBasis.")
        channels_ = int(channels)
        scale = float(initial_scale)
        if channels_ <= 0:
            raise ValueError("Monogenic field channels must be positive.")
        if not math.isfinite(scale) or scale < 0.0:
            raise ValueError("initial_scale must be finite and nonnegative.")
        shape = (channels_, basis.rank)
        coefficients = jnp.zeros(shape, dtype=basis.dtype)
        if scale:
            coefficients = (
                scale
                * jr.normal(key, shape, dtype=basis.dtype)
                / math.sqrt(float(basis.rank))
            )
        self.basis = basis
        self.coefficients = coefficients
        self.channels = channels_
        self.in_size = basis.dimension
        self.out_size = (
            basis.layout.blade_count
            if channels_ == 1
            else (channels_, basis.layout.blade_count)
        )

    def _contract(self, features: Array, /) -> Array:
        values = jnp.einsum("cr,rb->cb", self.coefficients, features)
        return values[0] if self.channels == 1 else values

    def __call__(self, point: Array, /, *, key: Any = None) -> Array:
        del key
        return self._contract(self.basis.evaluate(point))

    def model_metadata(self) -> Mapping[str, Any]:
        field_shape = (
            (self.basis.layout.blade_count,)
            if self.channels == 1
            else (self.channels, self.basis.layout.blade_count)
        )
        return {
            TRIAL_SPACE_CERTIFICATE_KEY: self.basis.certificate.for_field_shape(
                field_shape
            ),
            TRIAL_SPACE_REPRESENTATION_KEY: (
                self.basis.algebra,
                self.basis.layout,
            ),
        }

    def try_structured_partial(
        self,
        *,
        deps: tuple[str, ...],
        var: str,
        axis: int,
        order: int,
        args: tuple[Any, ...],
        key: Any,
        kwargs: dict[str, Any],
    ) -> tuple[Any | None, str | None]:
        del key, kwargs
        if deps != (var,) or len(args) != 1:
            return (
                None,
                "monogenic analytic partial requires one packed vector dependency",
            )
        return self._contract(self.basis.evaluate_partial(args[0], axis, order)), None

    def handle_structured_derivative_fallback(self, reason: str, /) -> None:
        del reason


__all__ = ["LinearMonogenicField", "MonogenicPolynomialBasis"]
