#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from itertools import combinations

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._complex import (
    AlmostComplexStructure,
    validate_almost_complex_structure,
)
from ._curvature import ricci_tensor
from ._forms import DifferentialForm, exterior_derivative
from ._metric import RiemannianMetric
from ._operators import covariant_derivative
from ._symplectic import SymplecticForm, validate_symplectic_form
from ._tensor import TensorType
from ._validation import validate_metric


class _FundamentalFormCoefficients(StrictModule):
    metric: RiemannianMetric
    complex_structure: AlmostComplexStructure
    indices: tuple[tuple[int, int], ...]

    def __init__(
        self,
        metric: RiemannianMetric,
        complex_structure: AlmostComplexStructure,
        /,
    ):
        self.metric = metric
        self.complex_structure = complex_structure
        self.indices = tuple(combinations(range(metric.chart.dimension), 2))

    def __call__(self, coordinates: Array, /) -> Array:
        matrix = self.metric(coordinates) @ self.complex_structure(coordinates)
        return jnp.stack(tuple(matrix[left, right] for left, right in self.indices))


class _RicciFormCoefficients(StrictModule):
    structure: KahlerStructure
    indices: tuple[tuple[int, int], ...]

    def __init__(self, structure: KahlerStructure, /):
        self.structure = structure
        self.indices = tuple(combinations(range(structure.metric.chart.dimension), 2))

    def __call__(self, coordinates: Array, /) -> Array:
        matrix = ricci_tensor(
            self.structure.metric, coordinates
        ) @ self.structure.complex_structure(coordinates)
        return jnp.stack(tuple(matrix[left, right] for left, right in self.indices))


class HermitianStructure(StrictModule):
    """A Riemannian metric paired with an almost-complex structure."""

    metric: RiemannianMetric
    complex_structure: AlmostComplexStructure

    def __init__(
        self,
        metric: RiemannianMetric,
        complex_structure: AlmostComplexStructure,
        /,
    ):
        if not isinstance(metric, RiemannianMetric):
            raise TypeError("HermitianStructure requires a RiemannianMetric.")
        if not isinstance(complex_structure, AlmostComplexStructure):
            raise TypeError("HermitianStructure requires an AlmostComplexStructure.")
        if not metric.chart.compatible_with(complex_structure.chart):
            raise ValueError("Hermitian metric and complex-structure charts must match.")
        self.metric = metric
        self.complex_structure = complex_structure

    @property
    def chart(self):
        return self.metric.chart

    def fundamental_form(self) -> DifferentialForm:
        return DifferentialForm(
            _FundamentalFormCoefficients(self.metric, self.complex_structure),
            chart=self.chart,
            degree=2,
        )


class HermitianValidationReport(StrictModule):
    valid: Array
    metric_valid: Array
    complex_valid: Array
    compatibility_residual: Array
    skew_residual: Array

    def __init__(
        self,
        *,
        valid: ArrayLike,
        metric_valid: ArrayLike,
        complex_valid: ArrayLike,
        compatibility_residual: ArrayLike,
        skew_residual: ArrayLike,
    ):
        self.valid = jnp.asarray(valid, dtype=bool)
        self.metric_valid = jnp.asarray(metric_valid, dtype=bool)
        self.complex_valid = jnp.asarray(complex_valid, dtype=bool)
        self.compatibility_residual = jnp.asarray(compatibility_residual)
        self.skew_residual = jnp.asarray(skew_residual)


def validate_hermitian_structure(
    structure: HermitianStructure,
    points: ArrayLike,
    /,
    *,
    compatibility_tolerance: float = 1e-8,
    raise_on_error: bool = True,
) -> HermitianValidationReport:
    if not isinstance(structure, HermitianStructure):
        raise TypeError("structure must be a HermitianStructure.")
    if compatibility_tolerance < 0.0:
        raise ValueError("compatibility_tolerance must be non-negative.")
    metric_report = validate_metric(structure.metric, points, raise_on_error=False)
    complex_report = validate_almost_complex_structure(
        structure.complex_structure,
        points,
        require_integrable=False,
        raise_on_error=False,
    )
    metric = structure.metric(points)
    complex_matrix = structure.complex_structure(points)
    transformed = jnp.swapaxes(complex_matrix, -1, -2) @ metric @ complex_matrix
    compatibility_residual = jnp.max(jnp.abs(transformed - metric))
    fundamental = metric @ complex_matrix
    skew_residual = jnp.max(jnp.abs(fundamental + jnp.swapaxes(fundamental, -1, -2)))
    valid = (
        metric_report.valid
        & complex_report.valid
        & (compatibility_residual <= compatibility_tolerance)
        & (skew_residual <= compatibility_tolerance)
    )
    report = HermitianValidationReport(
        valid=valid,
        metric_valid=metric_report.valid,
        complex_valid=complex_report.valid,
        compatibility_residual=compatibility_residual,
        skew_residual=skew_residual,
    )
    if raise_on_error and not bool(jax.device_get(valid)):
        raise ValueError(
            "Hermitian validation failed: "
            f"compatibility_residual={float(jax.device_get(compatibility_residual))}, "
            f"skew_residual={float(jax.device_get(skew_residual))}."
        )
    return report


class KahlerStructure(StrictModule):
    """A Hermitian-geometry candidate with Kähler validation semantics."""

    hermitian: HermitianStructure

    def __init__(self, hermitian: HermitianStructure, /):
        if not isinstance(hermitian, HermitianStructure):
            raise TypeError("KahlerStructure requires a HermitianStructure.")
        self.hermitian = hermitian

    @property
    def metric(self) -> RiemannianMetric:
        return self.hermitian.metric

    @property
    def complex_structure(self) -> AlmostComplexStructure:
        return self.hermitian.complex_structure

    @property
    def chart(self):
        return self.hermitian.chart

    def fundamental_form(self) -> DifferentialForm:
        return self.hermitian.fundamental_form()

    def symplectic_form(self) -> SymplecticForm:
        return SymplecticForm(self.fundamental_form())

    def ricci_form(self) -> DifferentialForm:
        return DifferentialForm(
            _RicciFormCoefficients(self),
            chart=self.chart,
            degree=2,
        )


class KahlerValidationReport(StrictModule):
    valid: Array
    hermitian_valid: Array
    integrable: Array
    closed: Array
    compatibility_residual: Array
    nijenhuis_residual: Array
    closure_residual: Array
    covariant_complex_residual: Array
    minimum_symplectic_singular_value: Array

    def __init__(
        self,
        *,
        valid: ArrayLike,
        hermitian_valid: ArrayLike,
        integrable: ArrayLike,
        closed: ArrayLike,
        compatibility_residual: ArrayLike,
        nijenhuis_residual: ArrayLike,
        closure_residual: ArrayLike,
        covariant_complex_residual: ArrayLike,
        minimum_symplectic_singular_value: ArrayLike,
    ):
        self.valid = jnp.asarray(valid, dtype=bool)
        self.hermitian_valid = jnp.asarray(hermitian_valid, dtype=bool)
        self.integrable = jnp.asarray(integrable, dtype=bool)
        self.closed = jnp.asarray(closed, dtype=bool)
        self.compatibility_residual = jnp.asarray(compatibility_residual)
        self.nijenhuis_residual = jnp.asarray(nijenhuis_residual)
        self.closure_residual = jnp.asarray(closure_residual)
        self.covariant_complex_residual = jnp.asarray(covariant_complex_residual)
        self.minimum_symplectic_singular_value = jnp.asarray(
            minimum_symplectic_singular_value
        )


def validate_kahler_structure(
    structure: KahlerStructure,
    points: ArrayLike,
    /,
    *,
    compatibility_tolerance: float = 1e-8,
    integrability_tolerance: float = 1e-8,
    closure_tolerance: float = 1e-8,
    covariant_tolerance: float = 1e-8,
    raise_on_error: bool = True,
) -> KahlerValidationReport:
    if not isinstance(structure, KahlerStructure):
        raise TypeError("structure must be a KahlerStructure.")
    if (
        min(
            compatibility_tolerance,
            integrability_tolerance,
            closure_tolerance,
            covariant_tolerance,
        )
        < 0.0
    ):
        raise ValueError("Kähler tolerances must be non-negative.")
    hermitian = validate_hermitian_structure(
        structure.hermitian,
        points,
        compatibility_tolerance=compatibility_tolerance,
        raise_on_error=False,
    )
    complex_report = validate_almost_complex_structure(
        structure.complex_structure,
        points,
        integrability_tolerance=integrability_tolerance,
        require_integrable=True,
        raise_on_error=False,
    )
    symplectic = validate_symplectic_form(
        structure.symplectic_form(),
        points,
        closure_tolerance=closure_tolerance,
    )
    if structure.chart.dimension == 2:
        closure_residual = jnp.asarray(0.0)
    else:
        closure_residual = jnp.max(
            jnp.abs(exterior_derivative(structure.fundamental_form())(points))
        )
    covariant = covariant_derivative(
        structure.complex_structure,
        structure.metric,
        TensorType(("contravariant", "covariant")),
        points,
    )
    covariant_residual = jnp.max(jnp.abs(covariant))
    valid = (
        hermitian.valid
        & complex_report.integrable
        & symplectic.valid
        & (covariant_residual <= covariant_tolerance)
    )
    report = KahlerValidationReport(
        valid=valid,
        hermitian_valid=hermitian.valid,
        integrable=complex_report.integrable,
        closed=symplectic.closed,
        compatibility_residual=hermitian.compatibility_residual,
        nijenhuis_residual=complex_report.nijenhuis_residual,
        closure_residual=closure_residual,
        covariant_complex_residual=covariant_residual,
        minimum_symplectic_singular_value=symplectic.minimum_singular_value,
    )
    if raise_on_error and not bool(jax.device_get(valid)):
        raise ValueError(
            "Kähler validation failed: "
            f"nijenhuis_residual={float(jax.device_get(complex_report.nijenhuis_residual))}, "
            f"closure_residual={float(jax.device_get(closure_residual))}, "
            f"covariant_complex_residual={float(jax.device_get(covariant_residual))}."
        )
    return report


__all__ = [
    "HermitianStructure",
    "HermitianValidationReport",
    "KahlerStructure",
    "KahlerValidationReport",
    "validate_hermitian_structure",
    "validate_kahler_structure",
]
