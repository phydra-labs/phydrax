#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from fractions import Fraction
from typing import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ._core import (
    AbstractTrefftzBasis,
    SimilarityNormalization,
    TrefftzResourceBudget,
    TrefftzResourceEvidence,
    TrialSpaceCertificate,
)


def _multiindices(total_degree: int, dimension: int) -> tuple[tuple[int, ...], ...]:
    if dimension == 1:
        return ((int(total_degree),),)
    values: list[tuple[int, ...]] = []
    for first in range(int(total_degree) + 1):
        for suffix in _multiindices(int(total_degree) - first, dimension - 1):
            values.append((first, *suffix))
    return tuple(values)


def _harmonic_rank(dimension: int, degree: int) -> int:
    total = math.comb(dimension + degree - 1, degree)
    laplacian_rank = 0 if degree < 2 else math.comb(dimension + degree - 3, degree - 2)
    return total - laplacian_rank


def _harmonic_counts(dimension: int, maximum_degree: int) -> tuple[int, int, int]:
    monomials = sum(
        math.comb(dimension + degree - 1, degree)
        for degree in range(maximum_degree + 1)
    )
    rank = sum(
        _harmonic_rank(dimension, degree) for degree in range(maximum_degree + 1)
    )
    entries = sum(
        math.comb(dimension + degree - 1, degree)
        * _harmonic_rank(dimension, degree)
        for degree in range(maximum_degree + 1)
    )
    return monomials, rank, entries


def _critical_construction_audit(
    residuals: Sequence[float],
    tolerances: Sequence[float],
    labels: Sequence[int],
    /,
    *,
    construction: str,
) -> tuple[float, float]:
    residuals_ = tuple(float(value) for value in residuals)
    tolerances_ = tuple(float(value) for value in tolerances)
    labels_ = tuple(int(value) for value in labels)
    if not residuals_ or not (
        len(residuals_) == len(tolerances_) == len(labels_)
    ):
        raise ValueError("Construction audits must be nonempty and aligned.")
    ratios = []
    for residual, tolerance, label in zip(
        residuals_,
        tolerances_,
        labels_,
        strict=True,
    ):
        if not math.isfinite(residual) or residual < 0.0:
            raise ValueError(
                f"{construction} block {label} has invalid residual {residual!r}."
            )
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(
                f"{construction} block {label} has invalid tolerance {tolerance!r}."
            )
        if residual > tolerance:
            ratio = math.inf if tolerance == 0.0 else residual / tolerance
            raise ValueError(
                f"{construction} block {label} failed its construction audit: "
                f"residual={residual:.3e}, tolerance={tolerance:.3e}, "
                f"ratio={ratio:.3e}."
            )
        ratios.append(
            0.0
            if residual == 0.0 and tolerance == 0.0
            else residual / tolerance
        )
    critical = max(range(len(ratios)), key=ratios.__getitem__)
    return residuals_[critical], tolerances_[critical]


def _add_scaled_expression(
    output: dict[int, Fraction],
    source: dict[int, Fraction],
    scale: Fraction,
) -> None:
    for column, value in source.items():
        candidate = output.get(column, Fraction(0)) + scale * value
        if candidate:
            output[column] = candidate
        elif column in output:
            del output[column]


def _canonical_harmonic_block(
    dimension: int,
    degree: int,
) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
    """Return a canonical exact nullspace basis for one homogeneous degree.

    The coefficient of every monomial with first exponent at least two is a
    lexicographic pivot. Coefficients with first exponent zero or one are free.
    Laplacian equations are then solved by increasing first exponent using exact
    rational arithmetic. No numerical nullspace orientation enters the basis.
    """

    exponents = _multiindices(degree, dimension)
    free = tuple(value for value in exponents if value[0] < 2)
    expected_rank = _harmonic_rank(dimension, degree)
    if len(free) != expected_rank:
        raise RuntimeError("Canonical harmonic free-column count is inconsistent.")

    expressions: dict[tuple[int, ...], dict[int, Fraction]] = {
        value: {column: Fraction(1)} for column, value in enumerate(free)
    }
    if degree >= 2:
        for beta in _multiindices(degree - 2, dimension):
            pivot = (beta[0] + 2, *beta[1:])
            denominator = Fraction((beta[0] + 2) * (beta[0] + 1))
            expression: dict[int, Fraction] = {}
            for axis in range(1, dimension):
                target = list(beta)
                target[axis] += 2
                target_tuple = tuple(target)
                if target_tuple not in expressions:
                    raise RuntimeError(
                        "Canonical harmonic elimination encountered an unresolved coefficient."
                    )
                factor = Fraction((beta[axis] + 2) * (beta[axis] + 1), 1)
                _add_scaled_expression(
                    expression,
                    expressions[target_tuple],
                    -factor / denominator,
                )
            expressions[pivot] = expression

    rational_rows = tuple(expressions[value] for value in exponents)
    block = np.zeros((len(exponents), expected_rank), dtype=np.float64)
    for row, expression in enumerate(rational_rows):
        for column, value in expression.items():
            block[row, column] = float(value)

    if degree < 2:
        laplacian = np.zeros((0, len(exponents)), dtype=np.float64)
    else:
        lower = _multiindices(degree - 2, dimension)
        lower_index = {value: index for index, value in enumerate(lower)}
        laplacian = np.zeros((len(lower), len(exponents)), dtype=np.float64)
        for column, alpha in enumerate(exponents):
            for axis, exponent in enumerate(alpha):
                if exponent < 2:
                    continue
                beta = list(alpha)
                beta[axis] -= 2
                laplacian[lower_index[tuple(beta)], column] += exponent * (exponent - 1)

    product = laplacian @ block
    residual = 0.0 if product.size == 0 else float(np.max(np.abs(product)))
    operator_scale = 1.0 if laplacian.size == 0 else float(np.max(np.abs(laplacian)))
    basis_scale = max(float(np.max(np.abs(block))), 1.0)
    tolerance = (
        256.0
        * np.finfo(np.float64).eps
        * max(operator_scale, 1.0)
        * basis_scale
        * max(len(exponents), expected_rank, 1)
    )
    singular_values = np.linalg.svd(block, compute_uv=False)
    if singular_values.size != expected_rank or singular_values[-1] <= 0.0:
        raise RuntimeError("Canonical harmonic basis is rank deficient.")
    return (
        np.asarray(exponents, dtype=np.int32),
        block,
        residual,
        tolerance,
        singular_values,
    )


class HarmonicPolynomialBasis(AbstractTrefftzBasis):
    """Canonical total-degree harmonic-polynomial basis in Euclidean space."""

    normalization: SimilarityNormalization
    maximum_degree: int = eqx.field(static=True)
    exponent_blocks: tuple[Array, ...]
    coefficient_blocks: tuple[Array, ...]
    singular_value_blocks: tuple[Array, ...]
    construction_residuals: tuple[Array, ...]
    construction_tolerances: tuple[Array, ...]
    _basis_id: str = eqx.field(static=True)
    _certificate: TrialSpaceCertificate
    _resource_evidence: TrefftzResourceEvidence

    def __init__(
        self,
        dimension: int,
        maximum_degree: int,
        /,
        *,
        normalization: SimilarityNormalization | None = None,
        resources: TrefftzResourceBudget | None = None,
    ):
        dimension_ = int(dimension)
        degree_ = int(maximum_degree)
        if dimension_ < 2:
            raise ValueError("HarmonicPolynomialBasis requires dimension >= 2.")
        if degree_ < 0:
            raise ValueError("maximum_degree must be nonnegative.")
        normalization_ = (
            SimilarityNormalization(np.zeros((dimension_,), dtype=float), 1.0)
            if normalization is None
            else normalization
        )
        if not isinstance(normalization_, SimilarityNormalization):
            raise TypeError("normalization must be a SimilarityNormalization or None.")
        if normalization_.dimension != dimension_:
            raise ValueError("Normalization dimension must match the harmonic basis.")
        budget = TrefftzResourceBudget() if resources is None else resources
        if not isinstance(budget, TrefftzResourceBudget):
            raise TypeError("resources must be a TrefftzResourceBudget or None.")
        monomials, rank, entries = _harmonic_counts(dimension_, degree_)
        evidence = budget.check(
            rank=rank,
            monomials=monomials,
            basis_entries=entries,
            basis_bytes=entries * np.dtype(np.float64).itemsize,
        )

        blocks = tuple(
            _canonical_harmonic_block(dimension_, degree)
            for degree in range(degree_ + 1)
        )
        residuals = tuple(block[2] for block in blocks)
        tolerances = tuple(block[3] for block in blocks)
        residual, tolerance = _critical_construction_audit(
            residuals,
            tolerances,
            range(degree_ + 1),
            construction="Harmonic degree",
        )
        exponent_blocks = tuple(jnp.asarray(block[0]) for block in blocks)
        coefficient_blocks = tuple(jnp.asarray(block[1]) for block in blocks)
        singular_blocks = tuple(jnp.asarray(block[4]) for block in blocks)
        basis_id = canonical_fingerprint(
            {
                "kind": "canonical-harmonic-polynomial-basis-v1",
                "dimension": dimension_,
                "maximum_degree": degree_,
                "normalization_id": normalization_.normalization_id,
                "canonical_blocks": array_tree_fingerprint(
                    (exponent_blocks, coefficient_blocks)
                ),
            }
        )
        certificate = TrialSpaceCertificate(
            equation_family="laplace",
            ambient_dimension=dimension_,
            construction="canonical-harmonic-polynomial-nullspace",
            normalization_id=normalization_.normalization_id,
            basis_id=basis_id,
            rank=rank,
            assumptions=(
                "Euclidean Laplacian",
                "fixed scalar similarity normalization",
                "finite total-degree polynomial subspace",
            ),
            construction_residual=residual,
            construction_tolerance=tolerance,
        )
        self.normalization = normalization_
        self.maximum_degree = degree_
        self.exponent_blocks = exponent_blocks
        self.coefficient_blocks = coefficient_blocks
        self.singular_value_blocks = singular_blocks
        self.construction_residuals = tuple(jnp.asarray(value) for value in residuals)
        self.construction_tolerances = tuple(
            jnp.asarray(value) for value in tolerances
        )
        self._basis_id = basis_id
        self._certificate = certificate
        self._resource_evidence = evidence

    @property
    def dimension(self) -> int:
        return self.normalization.dimension

    @property
    def rank(self) -> int:
        return self._resource_evidence.rank

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
            strict=True,
        ):
            monomials = jnp.prod(
                jnp.power(normalized[None, :], exponents),
                axis=-1,
            )
            features.append(monomials @ coefficients.astype(monomials.dtype))
        return jnp.concatenate(tuple(features), axis=0)


class PolyharmonicAlmansiBasis(AbstractTrefftzBasis):
    """Almansi basis whose span satisfies ``Laplacian**order u = 0``."""

    harmonic_bases: tuple[HarmonicPolynomialBasis, ...]
    order: int = eqx.field(static=True)
    maximum_degrees: tuple[int, ...] = eqx.field(static=True)
    constituent_certificate_ids: tuple[str, ...] = eqx.field(static=True)
    construction_residuals: tuple[Array, ...]
    construction_tolerances: tuple[Array, ...]
    _basis_id: str = eqx.field(static=True)
    _certificate: TrialSpaceCertificate
    _resource_evidence: TrefftzResourceEvidence

    def __init__(
        self,
        dimension: int,
        order: int,
        maximum_degree: int | Sequence[int],
        /,
        *,
        normalization: SimilarityNormalization | None = None,
        resources: TrefftzResourceBudget | None = None,
    ):
        dimension_ = int(dimension)
        order_ = int(order)
        if dimension_ < 2:
            raise ValueError("PolyharmonicAlmansiBasis requires dimension >= 2.")
        if order_ <= 0:
            raise ValueError("Polyharmonic order must be positive.")
        if isinstance(maximum_degree, int):
            degrees = (int(maximum_degree),) * order_
        else:
            degrees = tuple(int(value) for value in maximum_degree)
        if len(degrees) != order_ or any(value < 0 for value in degrees):
            raise ValueError(
                "maximum_degree must be nonnegative and provide one value per Almansi block."
            )
        normalization_ = (
            SimilarityNormalization(np.zeros((dimension_,), dtype=float), 1.0)
            if normalization is None
            else normalization
        )
        if not isinstance(normalization_, SimilarityNormalization):
            raise TypeError("normalization must be a SimilarityNormalization or None.")
        if normalization_.dimension != dimension_:
            raise ValueError("Normalization dimension must match the Almansi basis.")
        budget = TrefftzResourceBudget() if resources is None else resources
        if not isinstance(budget, TrefftzResourceBudget):
            raise TypeError("resources must be a TrefftzResourceBudget or None.")
        counts = tuple(_harmonic_counts(dimension_, degree) for degree in degrees)
        monomials = sum(value[0] for value in counts)
        rank = sum(value[1] for value in counts)
        entries = sum(value[2] for value in counts)
        evidence = budget.check(
            rank=rank,
            monomials=monomials,
            basis_entries=entries,
            basis_bytes=entries * np.dtype(np.float64).itemsize,
        )
        bases = tuple(
            HarmonicPolynomialBasis(
                dimension_,
                degree,
                normalization=normalization_,
                resources=budget,
            )
            for degree in degrees
        )
        basis_id = canonical_fingerprint(
            {
                "kind": "polyharmonic-almansi-basis-v1",
                "dimension": dimension_,
                "order": order_,
                "maximum_degrees": list(degrees),
                "normalization_id": normalization_.normalization_id,
                "harmonic_basis_ids": [basis.basis_id for basis in bases],
            }
        )
        constituent_residuals = tuple(
            float(basis.certificate.construction_residual) for basis in bases
        )
        constituent_tolerances = tuple(
            basis.certificate.construction_tolerance for basis in bases
        )
        residual, tolerance = _critical_construction_audit(
            constituent_residuals,
            constituent_tolerances,
            range(order_),
            construction="Almansi constituent",
        )
        certificate = TrialSpaceCertificate(
            equation_family="polyharmonic",
            ambient_dimension=dimension_,
            construction="almansi-harmonic-polynomial-basis",
            equation_parameters={"order": order_},
            normalization_id=normalization_.normalization_id,
            basis_id=basis_id,
            rank=rank,
            assumptions=(
                "Euclidean polyharmonic operator",
                "fixed scalar similarity normalization",
                "finite Almansi polynomial subspace",
                "star-shaped domain about the normalization center for completeness",
            ),
            construction_residual=residual,
            construction_tolerance=tolerance,
        )
        self.harmonic_bases = bases
        self.order = order_
        self.maximum_degrees = degrees
        self.constituent_certificate_ids = tuple(
            basis.certificate.certificate_id for basis in bases
        )
        self.construction_residuals = tuple(
            jnp.asarray(value) for value in constituent_residuals
        )
        self.construction_tolerances = tuple(
            jnp.asarray(value) for value in constituent_tolerances
        )
        self._basis_id = basis_id
        self._certificate = certificate
        self._resource_evidence = evidence

    @property
    def dimension(self) -> int:
        return self.harmonic_bases[0].dimension

    @property
    def rank(self) -> int:
        return self._resource_evidence.rank

    @property
    def dtype(self) -> jnp.dtype:
        return self.harmonic_bases[0].dtype

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
        normalized = self.harmonic_bases[0].normalization(point)
        radius_squared = jnp.sum(normalized * normalized)
        return jnp.concatenate(
            tuple(
                (radius_squared**power) * basis.evaluate(point)
                for power, basis in enumerate(self.harmonic_bases)
            ),
            axis=0,
        )


__all__ = ["HarmonicPolynomialBasis", "PolyharmonicAlmansiBasis"]
