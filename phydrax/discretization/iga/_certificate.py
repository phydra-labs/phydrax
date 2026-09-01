#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import combinations, permutations
from math import comb, isfinite

import numpy as np

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ._geometry import NURBSGeometryState
from ._volume import (
    aabbs_strictly_separated,
    BlockComplex,
    exact_planar_facet_halfspaces,
    TensorNURBSVolume,
)


class CertificateDisposition(str, Enum):
    """Closed three-way outcome of a theorem-backed decision procedure."""

    PASS = "pass"
    COUNTEREXAMPLE = "counterexample"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True, slots=True)
class IntervalBound:
    """One outward-rounded closed interval retained as certificate evidence."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        if (
            not isfinite(self.lower)
            or not isfinite(self.upper)
            or self.lower > self.upper
        ):
            raise ValueError("Interval bounds must be finite and ordered.")


@dataclass(frozen=True, slots=True)
class CertificateDiagnostic:
    """Structured fail-closed diagnostic emitted by a certificate producer."""

    code: str
    message: str
    block_id: str | None = None
    cell_index: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not self.code or not self.message:
            raise ValueError("Certificate diagnostics require a code and message.")


@dataclass(frozen=True, slots=True)
class CellJacobianEvidence:
    """Bernstein convex-hull bounds over one exact knot-span sub-box."""

    block_id: str
    span_index: tuple[int, ...]
    subdivision_path: tuple[int, ...]
    parameter_bounds: tuple[tuple[float, float], ...]
    weight: IntervalBound
    jacobian_measure: IntervalBound
    theorem: str


@dataclass(frozen=True, slots=True)
class CertificateCounterexample:
    """Analytic negative evidence, never a sampled or tessellated observation."""

    code: str
    message: str
    block_id: str
    cell: CellJacobianEvidence | None = None
    association_id: str | None = None


@dataclass(frozen=True, slots=True)
class TensorNURBSCertificatePolicy:
    """Finite deterministic budget and strict margins for Bernstein proofs."""

    maximum_subdivision_depth: int = 8
    maximum_subboxes: int = 4096
    positive_margin: float = 0.0
    roundoff_operation_budget: int = 32768
    policy_id: str = ""

    def __post_init__(self) -> None:
        if self.maximum_subdivision_depth < 0:
            raise ValueError("maximum_subdivision_depth must be non-negative.")
        if self.maximum_subboxes < 1:
            raise ValueError("maximum_subboxes must be positive.")
        if not isfinite(self.positive_margin) or self.positive_margin < 0.0:
            raise ValueError("positive_margin must be finite and non-negative.")
        if self.roundoff_operation_budget < 64:
            raise ValueError("roundoff_operation_budget must be at least 64.")
        if not self.policy_id:
            object.__setattr__(
                self,
                "policy_id",
                canonical_fingerprint(
                    {
                        "kind": "tensor-nurbs-certificate-policy",
                        "maximum_subdivision_depth": self.maximum_subdivision_depth,
                        "maximum_subboxes": self.maximum_subboxes,
                        "positive_margin": self.positive_margin,
                        "roundoff_operation_budget": self.roundoff_operation_budget,
                    }
                ),
            )


@dataclass(frozen=True, slots=True)
class LocalGeometryCertificate:
    """CERT-C1 local orientation certificate for one square tensor NURBS map."""

    disposition: CertificateDisposition
    block_id: str
    dimension: int
    cells: tuple[CellJacobianEvidence, ...]
    diagnostics: tuple[CertificateDiagnostic, ...]
    counterexample: CertificateCounterexample | None
    policy_id: str
    certificate_id: str

    @property
    def accepted(self) -> bool:
        return self.disposition is CertificateDisposition.PASS

    @property
    def minimum_weight(self) -> float:
        return min(cell.weight.lower for cell in self.cells)

    @property
    def minimum_jacobian(self) -> float:
        return min(cell.jacobian_measure.lower for cell in self.cells)

    def qualification_payload(self, profile: str = "CERT-C1", /) -> dict[str, object]:
        return {
            "profile": str(profile),
            "certificate_id": self.certificate_id,
            "accepted": self.accepted,
            "disposition": self.disposition.value,
            "block_id": self.block_id,
            "minimum_weight": self.minimum_weight,
            "minimum_jacobian": self.minimum_jacobian,
            "theorem": "positive rational denominator and Bernstein detJ lower bound",
        }


@dataclass(frozen=True, slots=True)
class SurfaceEmbeddingCertificate:
    """Local regular-embedding evidence for a tensor NURBS surface in 3D."""

    disposition: CertificateDisposition
    block_id: str
    cells: tuple[CellJacobianEvidence, ...]
    diagnostics: tuple[CertificateDiagnostic, ...]
    counterexample: CertificateCounterexample | None
    policy_id: str
    certificate_id: str

    @property
    def accepted(self) -> bool:
        return self.disposition is CertificateDisposition.PASS

    def qualification_payload(
        self, profile: str = "CERT-SURFACE", /
    ) -> dict[str, object]:
        return {
            "profile": str(profile),
            "certificate_id": self.certificate_id,
            "accepted": self.accepted,
            "disposition": self.disposition.value,
            "block_id": self.block_id,
            "minimum_gram_determinant": min(
                cell.jacobian_measure.lower for cell in self.cells
            ),
            "theorem": "positive rational denominator and Bernstein Gram-determinant lower bound",
        }


@dataclass(frozen=True, slots=True)
class DeformedJacobianCertificate:
    """DEFORM-C1 reference and deformed orientation evidence under one basis."""

    disposition: CertificateDisposition
    reference: LocalGeometryCertificate
    deformed: LocalGeometryCertificate
    displacement_fingerprint: dict[str, object]
    load_factor: float
    diagnostics: tuple[CertificateDiagnostic, ...]
    certificate_id: str

    @property
    def accepted(self) -> bool:
        return self.disposition is CertificateDisposition.PASS

    def qualification_payload(self, profile: str = "DEFORM-C1", /) -> dict[str, object]:
        return {
            "profile": str(profile),
            "certificate_id": self.certificate_id,
            "accepted": self.accepted,
            "disposition": self.disposition.value,
            "reference_certificate_id": self.reference.certificate_id,
            "deformed_certificate_id": self.deformed.certificate_id,
            "load_factor": self.load_factor,
            "minimum_deformed_jacobian": self.deformed.minimum_jacobian,
        }


@dataclass(frozen=True, slots=True)
class BoundaryInjectivityCertificate:
    """Exact-incidence and convex-hull separation proof for a block boundary."""

    disposition: CertificateDisposition
    incidence_ids: tuple[str, ...]
    separation_pairs: tuple[tuple[str, str], ...]
    diagnostics: tuple[CertificateDiagnostic, ...]
    certificate_id: str

    @property
    def accepted(self) -> bool:
        return self.disposition is CertificateDisposition.PASS


@dataclass(frozen=True, slots=True)
class MappingDegreeCertificate:
    """Degree of a connected, oriented block complex after boundary certification."""

    disposition: CertificateDisposition
    degree: int | None
    boundary_certificate_id: str
    theorem: str
    certificate_id: str

    @property
    def accepted(self) -> bool:
        return self.disposition is CertificateDisposition.PASS and self.degree == 1


@dataclass(frozen=True, slots=True)
class GlobalInjectivityCertificate:
    """F1 global one-to-one certificate for an oriented tensor-block complex."""

    disposition: CertificateDisposition
    local_certificates: tuple[LocalGeometryCertificate, ...]
    boundary: BoundaryInjectivityCertificate
    mapping_degree: MappingDegreeCertificate
    diagnostics: tuple[CertificateDiagnostic, ...]
    counterexample: CertificateCounterexample | None
    complex_id: str
    certificate_id: str

    @property
    def accepted(self) -> bool:
        return (
            self.disposition is CertificateDisposition.PASS
            and self.boundary.accepted
            and self.mapping_degree.accepted
            and all(value.accepted for value in self.local_certificates)
        )

    def qualification_payload(self, profile: str = "F1", /) -> dict[str, object]:
        return {
            "profile": str(profile),
            "certificate_id": self.certificate_id,
            "accepted": self.accepted,
            "disposition": self.disposition.value,
            "complex_id": self.complex_id,
            "local_certificate_ids": [
                value.certificate_id for value in self.local_certificates
            ],
            "boundary_certificate_id": self.boundary.certificate_id,
            "mapping_degree_certificate_id": self.mapping_degree.certificate_id,
            "mapping_degree": self.mapping_degree.degree,
        }


@dataclass(slots=True)
class _BernsteinPolynomial:
    coefficients: np.ndarray

    @property
    def degrees(self) -> tuple[int, ...]:
        return tuple(size - 1 for size in self.coefficients.shape)


def _poly_scale(value: _BernsteinPolynomial, scale: float, /) -> _BernsteinPolynomial:
    return _BernsteinPolynomial(value.coefficients * scale)


def _poly_add(
    first: _BernsteinPolynomial,
    second: _BernsteinPolynomial,
    /,
    *,
    second_scale: float = 1.0,
) -> _BernsteinPolynomial:
    if first.coefficients.shape != second.coefficients.shape:
        raise ValueError("Bernstein polynomial degrees do not agree.")
    return _BernsteinPolynomial(first.coefficients + second_scale * second.coefficients)


def _poly_product(
    first: _BernsteinPolynomial, second: _BernsteinPolynomial, /
) -> _BernsteinPolynomial:
    first_degrees = first.degrees
    second_degrees = second.degrees
    if len(first_degrees) != len(second_degrees):
        raise ValueError("Bernstein polynomial dimensions do not agree.")
    result_degrees = tuple(
        first_degree + second_degree
        for first_degree, second_degree in zip(first_degrees, second_degrees, strict=True)
    )
    result = np.zeros(tuple(degree + 1 for degree in result_degrees), dtype=float)
    first_weights = tuple(
        tuple(comb(degree, index) for index in range(degree + 1))
        for degree in first_degrees
    )
    second_weights = tuple(
        tuple(comb(degree, index) for index in range(degree + 1))
        for degree in second_degrees
    )
    result_weights = tuple(
        tuple(comb(degree, index) for index in range(degree + 1))
        for degree in result_degrees
    )
    for first_index in np.ndindex(first.coefficients.shape):
        first_value = first.coefficients[first_index]
        for second_index in np.ndindex(second.coefficients.shape):
            target = tuple(
                left + right
                for left, right in zip(first_index, second_index, strict=True)
            )
            factor = 1.0
            for axis, (left, right, total) in enumerate(
                zip(first_index, second_index, target, strict=True)
            ):
                factor *= (
                    first_weights[axis][left]
                    * second_weights[axis][right]
                    / result_weights[axis][total]
                )
            result[target] += factor * first_value * second.coefficients[second_index]
    return _BernsteinPolynomial(result)


def _poly_derivative(
    value: _BernsteinPolynomial, axis: int, parameter_width: float, /
) -> _BernsteinPolynomial:
    degree = value.degrees[axis]
    if degree < 1:
        raise ValueError("A constant Bernstein axis has no first derivative.")
    return _BernsteinPolynomial(
        np.diff(value.coefficients, axis=axis) * (degree / parameter_width)
    )


def _determinant(
    polynomials: list[list[_BernsteinPolynomial]], /
) -> _BernsteinPolynomial:
    dimension = len(polynomials)
    result: _BernsteinPolynomial | None = None
    for route in permutations(range(dimension)):
        inversions = sum(
            route[first] > route[second]
            for first in range(dimension)
            for second in range(first + 1, dimension)
        )
        term = polynomials[0][route[0]]
        for row in range(1, dimension):
            term = _poly_product(term, polynomials[row][route[row]])
        if result is None:
            result = _poly_scale(term, -1.0 if inversions % 2 else 1.0)
        else:
            result = _poly_add(result, term, second_scale=-1.0 if inversions % 2 else 1.0)
    if result is None:
        raise ValueError("A polynomial determinant requires positive dimension.")
    return result


def _insert_knot_axis(
    controls: np.ndarray,
    knots: np.ndarray,
    degree: int,
    knot: float,
    axis: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.moveaxis(controls, axis, 0)
    control_count = values.shape[0]
    n = control_count - 1
    multiplicity = int(np.count_nonzero(knots == knot))
    if multiplicity >= degree:
        return controls, knots
    span_candidates = np.flatnonzero((knots[:-1] <= knot) & (knot < knots[1:]))
    if span_candidates.size != 1:
        raise ValueError("Interior knot insertion requires one active containing span.")
    span = int(span_candidates[0])
    inserted = np.empty((control_count + 1, *values.shape[1:]), dtype=float)
    inserted[: span - degree + 1] = values[: span - degree + 1]
    inserted[span - multiplicity + 1 :] = values[span - multiplicity :]
    for index in range(span - degree + 1, span - multiplicity + 1):
        denominator = knots[index + degree] - knots[index]
        if not denominator > 0.0:
            raise ValueError("Knot insertion encountered a zero active denominator.")
        alpha = (knot - knots[index]) / denominator
        inserted[index] = alpha * values[index] + (1.0 - alpha) * values[index - 1]
    new_knots = np.insert(knots, span + 1, knot)
    return np.moveaxis(inserted, 0, axis), new_knots


def _bezier_homogeneous_cells(
    volume: TensorNURBSVolume, /
) -> tuple[tuple[np.ndarray, tuple[tuple[float, float], ...], tuple[int, ...]], ...]:
    points = np.asarray(volume.geometry.control_points, dtype=float)
    weights = np.asarray(volume.geometry.weights, dtype=float)
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(weights)):
        raise ValueError("Certification requires concrete finite NURBS data.")
    homogeneous = np.concatenate(
        (points * weights[..., None], weights[..., None]), axis=-1
    )
    refined = homogeneous
    refined_knots: list[np.ndarray] = []
    for axis_index, axis in enumerate(volume.basis.axes):
        knots = np.asarray(axis.knots, dtype=float).copy()
        lower, upper = axis.parameter_interval
        interiors = np.unique(knots[(knots > lower) & (knots < upper)])
        for knot in interiors:
            while int(np.count_nonzero(knots == knot)) < axis.degree:
                refined, knots = _insert_knot_axis(
                    refined, knots, axis.degree, float(knot), axis_index
                )
        refined_knots.append(knots)
    cells = []
    for span_index in np.ndindex(volume.basis.span_shape):
        selector = tuple(
            slice(index * axis.degree, index * axis.degree + axis.degree + 1)
            for index, axis in zip(span_index, volume.basis.axes, strict=True)
        )
        controls = refined[selector]
        expected = tuple(axis.degree + 1 for axis in volume.basis.axes)
        if controls.shape[:-1] != expected:
            raise RuntimeError(
                "Bezier extraction produced an inconsistent cell control net."
            )
        bounds = tuple(
            tuple(float(value) for value in np.asarray(axis.span_bounds[index]))
            for index, axis in zip(span_index, volume.basis.axes, strict=True)
        )
        cells.append((controls, bounds, tuple(int(value) for value in span_index)))
    return tuple(cells)


def _split_bezier_axis(
    controls: np.ndarray, axis: int, /
) -> tuple[np.ndarray, np.ndarray]:
    values = np.moveaxis(controls, axis, 0)
    degree = values.shape[0] - 1
    left = np.empty_like(values)
    right = np.empty_like(values)
    work = values.copy()
    left[0] = work[0]
    right[-1] = work[-1]
    for level in range(1, degree + 1):
        work = 0.5 * (work[:-1] + work[1:])
        left[level] = work[0]
        right[-level - 1] = work[-1]
    return np.moveaxis(left, 0, axis), np.moveaxis(right, 0, axis)


def _outward_coefficients_bound(
    coefficients: np.ndarray, policy: TensorNURBSCertificatePolicy, /
) -> IntervalBound:
    minimum = float(np.min(coefficients))
    maximum = float(np.max(coefficients))
    scale = max(1.0, float(np.max(np.abs(coefficients))))
    epsilon = np.finfo(float).eps
    operations = policy.roundoff_operation_budget
    gamma = operations * epsilon / (1.0 - operations * epsilon)
    padding = gamma * scale
    return IntervalBound(
        float(np.nextafter(minimum - padding, -np.inf)),
        float(np.nextafter(maximum + padding, np.inf)),
    )


def _positive_power_interval(bound: IntervalBound, exponent: int, /) -> IntervalBound:
    if bound.lower <= 0.0:
        raise ValueError("A positive denominator bound is required.")
    return IntervalBound(bound.lower**exponent, bound.upper**exponent)


def _divide_intervals(
    numerator: IntervalBound, denominator: IntervalBound, /
) -> IntervalBound:
    if denominator.lower <= 0.0:
        raise ValueError("Interval division requires a positive denominator.")
    values = (
        numerator.lower / denominator.lower,
        numerator.lower / denominator.upper,
        numerator.upper / denominator.lower,
        numerator.upper / denominator.upper,
    )
    return IntervalBound(
        float(np.nextafter(min(values), -np.inf)),
        float(np.nextafter(max(values), np.inf)),
    )


def _numerator_jacobian(
    homogeneous: np.ndarray,
    bounds: tuple[tuple[float, float], ...],
    ambient_dimension: int,
) -> tuple[_BernsteinPolynomial, list[list[_BernsteinPolynomial]]]:
    weight = _BernsteinPolynomial(homogeneous[..., -1])
    coordinates = [
        _BernsteinPolynomial(homogeneous[..., component])
        for component in range(ambient_dimension)
    ]
    numerator: list[list[_BernsteinPolynomial]] = [[] for _ in range(ambient_dimension)]
    for component, coordinate in enumerate(coordinates):
        for axis, (lower, upper) in enumerate(bounds):
            derivative_coordinate = _poly_derivative(coordinate, axis, upper - lower)
            derivative_weight = _poly_derivative(weight, axis, upper - lower)
            numerator[component].append(
                _poly_add(
                    _poly_product(weight, derivative_coordinate),
                    _poly_product(coordinate, derivative_weight),
                    second_scale=-1.0,
                )
            )
    return weight, numerator


def _jacobian_measure_polynomial(
    homogeneous: np.ndarray,
    bounds: tuple[tuple[float, float], ...],
    ambient_dimension: int,
    theorem: str,
) -> tuple[_BernsteinPolynomial, _BernsteinPolynomial, int]:
    weight, numerator = _numerator_jacobian(homogeneous, bounds, ambient_dimension)
    dimension = len(bounds)
    if theorem == "determinant":
        if ambient_dimension != dimension:
            raise ValueError("A determinant certificate requires a square geometry map.")
        measure = _determinant(numerator)
        denominator_exponent = 2 * dimension
    elif theorem == "surface-gram":
        if dimension != 2 or ambient_dimension != 3:
            raise ValueError("The surface embedding theorem requires a 2D map into 3D.")
        first = _poly_add(
            _poly_product(numerator[1][0], numerator[2][1]),
            _poly_product(numerator[2][0], numerator[1][1]),
            second_scale=-1.0,
        )
        second = _poly_add(
            _poly_product(numerator[2][0], numerator[0][1]),
            _poly_product(numerator[0][0], numerator[2][1]),
            second_scale=-1.0,
        )
        third = _poly_add(
            _poly_product(numerator[0][0], numerator[1][1]),
            _poly_product(numerator[1][0], numerator[0][1]),
            second_scale=-1.0,
        )
        measure = _poly_add(
            _poly_add(_poly_product(first, first), _poly_product(second, second)),
            _poly_product(third, third),
        )
        denominator_exponent = 8
    else:
        raise ValueError(f"Unknown Jacobian theorem {theorem!r}.")
    return weight, measure, denominator_exponent


def _cell_evidence(
    block_id: str,
    span_index: tuple[int, ...],
    subdivision_path: tuple[int, ...],
    homogeneous: np.ndarray,
    bounds: tuple[tuple[float, float], ...],
    ambient_dimension: int,
    theorem: str,
    policy: TensorNURBSCertificatePolicy,
) -> CellJacobianEvidence:
    weight_polynomial, measure_polynomial, exponent = _jacobian_measure_polynomial(
        homogeneous, bounds, ambient_dimension, theorem
    )
    weight = _outward_coefficients_bound(weight_polynomial.coefficients, policy)
    measure_numerator = _outward_coefficients_bound(
        measure_polynomial.coefficients, policy
    )
    if weight.lower <= 0.0:
        measure = IntervalBound(-np.finfo(float).max, np.finfo(float).max)
    else:
        measure = _divide_intervals(
            measure_numerator, _positive_power_interval(weight, exponent)
        )
    return CellJacobianEvidence(
        block_id,
        span_index,
        subdivision_path,
        bounds,
        weight,
        measure,
        theorem,
    )


def _certificate_id(
    kind: str,
    volume: TensorNURBSVolume,
    policy: TensorNURBSCertificatePolicy,
    disposition: CertificateDisposition,
    cells: tuple[CellJacobianEvidence, ...],
) -> str:
    return canonical_fingerprint(
        {
            "kind": kind,
            "volume": volume.volume_id,
            "policy": policy.policy_id,
            "disposition": disposition.value,
            "cells": [
                {
                    "span": cell.span_index,
                    "path": cell.subdivision_path,
                    "bounds": cell.parameter_bounds,
                    "weight": (cell.weight.lower, cell.weight.upper),
                    "measure": (
                        cell.jacobian_measure.lower,
                        cell.jacobian_measure.upper,
                    ),
                }
                for cell in cells
            ],
        }
    )


def _certify_local(
    volume: TensorNURBSVolume,
    policy: TensorNURBSCertificatePolicy,
    theorem: str,
) -> tuple[
    CertificateDisposition,
    tuple[CellJacobianEvidence, ...],
    tuple[CertificateDiagnostic, ...],
    CertificateCounterexample | None,
]:
    queue: list[
        tuple[
            np.ndarray,
            tuple[tuple[float, float], ...],
            tuple[int, ...],
            tuple[int, ...],
        ]
    ] = [
        (controls, bounds, span, ())
        for controls, bounds, span in _bezier_homogeneous_cells(volume)
    ]
    leaves: list[CellJacobianEvidence] = []
    diagnostics: list[CertificateDiagnostic] = []
    counterexample: CertificateCounterexample | None = None
    processed = 0
    budget_exhausted = False
    while queue:
        homogeneous, bounds, span_index, path = queue.pop(0)
        processed += 1
        evidence = _cell_evidence(
            volume.block_id,
            span_index,
            path,
            homogeneous,
            bounds,
            volume.ambient_dimension,
            theorem,
            policy,
        )
        if evidence.weight.upper <= 0.0:
            leaves.append(evidence)
            counterexample = CertificateCounterexample(
                "nonpositive-weight-function",
                "The Bernstein upper bound proves a nonpositive rational denominator.",
                volume.block_id,
                evidence,
            )
            break
        if evidence.jacobian_measure.upper <= 0.0:
            leaves.append(evidence)
            counterexample = CertificateCounterexample(
                "nonpositive-jacobian-measure",
                "The Bernstein upper bound proves nonpositive Jacobian measure on a parameter box.",
                volume.block_id,
                evidence,
            )
            break
        if (
            evidence.weight.lower > 0.0
            and evidence.jacobian_measure.lower > policy.positive_margin
        ):
            leaves.append(evidence)
            continue
        depth = len(path)
        if (
            depth >= policy.maximum_subdivision_depth
            or processed + len(queue) + 2 > policy.maximum_subboxes
        ):
            leaves.append(evidence)
            budget_exhausted = True
            continue
        widths = np.asarray([upper - lower for lower, upper in bounds])
        axis = int(np.argmax(widths))
        left, right = _split_bezier_axis(homogeneous, axis)
        midpoint = 0.5 * (bounds[axis][0] + bounds[axis][1])
        left_bounds = list(bounds)
        right_bounds = list(bounds)
        left_bounds[axis] = (bounds[axis][0], midpoint)
        right_bounds[axis] = (midpoint, bounds[axis][1])
        queue.append((left, tuple(left_bounds), span_index, (*path, 2 * axis)))
        queue.append((right, tuple(right_bounds), span_index, (*path, 2 * axis + 1)))
    if counterexample is not None:
        disposition = CertificateDisposition.COUNTEREXAMPLE
        diagnostics.append(
            CertificateDiagnostic(
                counterexample.code,
                counterexample.message,
                volume.block_id,
                counterexample.cell.span_index
                if counterexample.cell is not None
                else None,
            )
        )
    elif budget_exhausted:
        disposition = CertificateDisposition.INCONCLUSIVE
        diagnostics.append(
            CertificateDiagnostic(
                "bernstein-budget-exhausted",
                "Strict positivity was not proved within the bounded subdivision budget.",
                volume.block_id,
            )
        )
    else:
        disposition = CertificateDisposition.PASS
    return disposition, tuple(leaves), tuple(diagnostics), counterexample


def certify_tensor_nurbs(
    volume: TensorNURBSVolume,
    /,
    *,
    policy: TensorNURBSCertificatePolicy | None = None,
) -> LocalGeometryCertificate:
    """Certify positive W and detJ on every knot cell without point sampling."""
    if not isinstance(volume, TensorNURBSVolume):
        raise TypeError("volume must be a TensorNURBSVolume.")
    if volume.parametric_dimension != volume.ambient_dimension:
        raise ValueError("CERT-C1 requires equal parametric and ambient dimensions.")
    policy_ = TensorNURBSCertificatePolicy() if policy is None else policy
    if not isinstance(policy_, TensorNURBSCertificatePolicy):
        raise TypeError("policy must be a TensorNURBSCertificatePolicy.")
    disposition, cells, diagnostics, counterexample = _certify_local(
        volume, policy_, "determinant"
    )
    identifier = _certificate_id("cert-c1", volume, policy_, disposition, cells)
    return LocalGeometryCertificate(
        disposition,
        volume.block_id,
        volume.parametric_dimension,
        cells,
        diagnostics,
        counterexample,
        policy_.policy_id,
        identifier,
    )


def certify_surface_embedding(
    surface: TensorNURBSVolume,
    /,
    *,
    policy: TensorNURBSCertificatePolicy | None = None,
) -> SurfaceEmbeddingCertificate:
    """Certify positive surface Gram determinant by rational Bernstein bounds."""
    if not isinstance(surface, TensorNURBSVolume):
        raise TypeError("surface must be a TensorNURBSVolume.")
    if surface.parametric_dimension != 2 or surface.ambient_dimension != 3:
        raise ValueError(
            "Surface embedding certification requires a 2D NURBS map into 3D."
        )
    policy_ = TensorNURBSCertificatePolicy() if policy is None else policy
    disposition, cells, diagnostics, counterexample = _certify_local(
        surface, policy_, "surface-gram"
    )
    identifier = _certificate_id(
        "surface-embedding", surface, policy_, disposition, cells
    )
    return SurfaceEmbeddingCertificate(
        disposition,
        surface.block_id,
        cells,
        diagnostics,
        counterexample,
        policy_.policy_id,
        identifier,
    )


def certify_deformed_tensor_nurbs(
    reference: TensorNURBSVolume,
    displacement_control_points: object,
    /,
    *,
    load_factor: float = 1.0,
    policy: TensorNURBSCertificatePolicy | None = None,
) -> DeformedJacobianCertificate:
    """Certify reference and deformed Jacobians from exact control-net algebra."""
    if not isinstance(reference, TensorNURBSVolume):
        raise TypeError("reference must be a TensorNURBSVolume.")
    factor = float(load_factor)
    if not isfinite(factor):
        raise ValueError("load_factor must be finite.")
    displacement = np.asarray(displacement_control_points, dtype=float)
    reference_points = np.asarray(reference.geometry.control_points, dtype=float)
    if displacement.shape != reference_points.shape:
        raise ValueError(
            "Displacement control points must match the geometry control net."
        )
    if not np.all(np.isfinite(displacement)):
        raise ValueError("Displacement control points must be finite.")
    geometry = NURBSGeometryState(
        reference_points + factor * displacement,
        np.asarray(reference.geometry.weights),
    )
    deformed_volume = TensorNURBSVolume(
        f"{reference.block_id}:deformed",
        reference.basis,
        geometry,
        patch_id=reference.patch_id,
        numeric_revision=canonical_fingerprint(
            {
                "reference": reference.numeric_revision,
                "displacement": array_tree_fingerprint(displacement),
                "load_factor": factor,
            }
        ),
    )
    reference_certificate = certify_tensor_nurbs(reference, policy=policy)
    deformed_certificate = certify_tensor_nurbs(deformed_volume, policy=policy)
    if (
        reference_certificate.disposition is CertificateDisposition.COUNTEREXAMPLE
        or deformed_certificate.disposition is CertificateDisposition.COUNTEREXAMPLE
    ):
        disposition = CertificateDisposition.COUNTEREXAMPLE
    elif reference_certificate.accepted and deformed_certificate.accepted:
        disposition = CertificateDisposition.PASS
    else:
        disposition = CertificateDisposition.INCONCLUSIVE
    diagnostics = (*reference_certificate.diagnostics, *deformed_certificate.diagnostics)
    displacement_fingerprint = array_tree_fingerprint(displacement)
    identifier = canonical_fingerprint(
        {
            "kind": "deform-c1",
            "reference": reference_certificate.certificate_id,
            "deformed": deformed_certificate.certificate_id,
            "displacement": displacement_fingerprint,
            "load_factor": factor,
            "disposition": disposition.value,
        }
    )
    return DeformedJacobianCertificate(
        disposition,
        reference_certificate,
        deformed_certificate,
        displacement_fingerprint,
        factor,
        diagnostics,
        identifier,
    )


def _p_matrix_proved(
    volume: TensorNURBSVolume,
    policy: TensorNURBSCertificatePolicy,
) -> tuple[bool, CertificateDiagnostic | None]:
    """Apply the Gale-Nikaido sufficient condition on every Bezier span."""
    dimension = volume.parametric_dimension
    if dimension != volume.ambient_dimension:
        return False, CertificateDiagnostic(
            "p-matrix-dimension",
            "Gale-Nikaido requires a square map.",
            volume.block_id,
        )
    for homogeneous, bounds, span_index in _bezier_homogeneous_cells(volume):
        _, numerator = _numerator_jacobian(homogeneous, bounds, volume.ambient_dimension)
        for size in range(1, dimension + 1):
            for indices in combinations(range(dimension), size):
                minor = _determinant(
                    [[numerator[row][column] for column in indices] for row in indices]
                )
                bound = _outward_coefficients_bound(minor.coefficients, policy)
                if bound.lower <= policy.positive_margin:
                    return False, CertificateDiagnostic(
                        "p-matrix-unproved",
                        "A principal Jacobian numerator minor lacks a strict Bernstein lower bound.",
                        volume.block_id,
                        span_index,
                    )
    return True, None


def certify_global_injectivity(
    complex_: BlockComplex,
    /,
    *,
    policy: TensorNURBSCertificatePolicy | None = None,
) -> GlobalInjectivityCertificate:
    """Certify F1 using local orientation, univalence, incidence, and separation."""
    if not isinstance(complex_, BlockComplex):
        raise TypeError("complex_ must be a BlockComplex.")
    policy_ = TensorNURBSCertificatePolicy() if policy is None else policy
    local = tuple(
        certify_tensor_nurbs(volume, policy=policy_) for volume in complex_.volumes
    )
    diagnostics: list[CertificateDiagnostic] = [
        diagnostic for certificate in local for diagnostic in certificate.diagnostics
    ]
    counterexample = next(
        (
            certificate.counterexample
            for certificate in local
            if certificate.counterexample is not None
        ),
        None,
    )
    disposition = (
        CertificateDisposition.COUNTEREXAMPLE
        if counterexample is not None
        else CertificateDisposition.PASS
    )
    if disposition is CertificateDisposition.PASS and not all(
        value.accepted for value in local
    ):
        disposition = CertificateDisposition.INCONCLUSIVE
    for volume in complex_.volumes:
        proved, diagnostic = _p_matrix_proved(volume, policy_)
        if not proved and disposition is CertificateDisposition.PASS:
            disposition = CertificateDisposition.INCONCLUSIVE
        if diagnostic is not None:
            diagnostics.append(diagnostic)
    incidence_ids: list[str] = []
    for incidence, check in zip(
        complex_.incidences, complex_.exact_incidence_checks(), strict=True
    ):
        if check.matched:
            incidence_ids.append(check.association_id)
        else:
            diagnostics.append(
                CertificateDiagnostic(
                    "exact-incidence-failed",
                    check.reason,
                    incidence.left.block_id,
                )
            )
            if disposition is CertificateDisposition.PASS:
                disposition = CertificateDisposition.INCONCLUSIVE
        if check.matched:
            separated, reason = exact_planar_facet_halfspaces(
                complex_.volume(incidence.left.block_id),
                complex_.volume(incidence.right.block_id),
                incidence,
            )
            if not separated:
                diagnostics.append(
                    CertificateDiagnostic(
                        "adjacent-halfspace-unproved",
                        reason,
                        incidence.left.block_id,
                    )
                )
                if disposition is CertificateDisposition.PASS:
                    disposition = CertificateDisposition.INCONCLUSIVE
    if not complex_.connected():
        diagnostics.append(
            CertificateDiagnostic(
                "block-complex-disconnected",
                "Mapping degree one is unavailable for a disconnected block complex.",
            )
        )
        if disposition is CertificateDisposition.PASS:
            disposition = CertificateDisposition.INCONCLUSIVE
    adjacent_block_pairs = {
        frozenset((incidence.left.block_id, incidence.right.block_id))
        for incidence in complex_.incidences
    }
    separation_pairs: list[tuple[str, str]] = []
    for first, second in combinations(complex_.volumes, 2):
        pair = frozenset((first.block_id, second.block_id))
        if pair in adjacent_block_pairs:
            continue
        separated, _, _ = aabbs_strictly_separated(
            first.control_aabb(), second.control_aabb()
        )
        if separated:
            separation_pairs.append(tuple(sorted((first.block_id, second.block_id))))
        else:
            diagnostics.append(
                CertificateDiagnostic(
                    "nonadjacent-separation-unproved",
                    "Nonadjacent rational control hulls have no strict coordinate separation.",
                    first.block_id,
                )
            )
            if disposition is CertificateDisposition.PASS:
                disposition = CertificateDisposition.INCONCLUSIVE
    if complex_.permitted_boundary_contacts:
        diagnostics.append(
            CertificateDiagnostic(
                "lower-dimensional-contact-unproved",
                "Permitted lower-dimensional contacts require an exact incidence theorem not supplied here.",
            )
        )
        if disposition is CertificateDisposition.PASS:
            disposition = CertificateDisposition.INCONCLUSIVE
    boundary_disposition = (
        CertificateDisposition.PASS
        if disposition is CertificateDisposition.PASS
        else disposition
    )
    boundary_id = canonical_fingerprint(
        {
            "kind": "boundary-injectivity",
            "complex": complex_.complex_id,
            "disposition": boundary_disposition.value,
            "incidences": incidence_ids,
            "separations": separation_pairs,
        }
    )
    boundary = BoundaryInjectivityCertificate(
        boundary_disposition,
        tuple(incidence_ids),
        tuple(separation_pairs),
        tuple(diagnostics),
        boundary_id,
    )
    degree_disposition = (
        CertificateDisposition.PASS
        if boundary.accepted and all(value.accepted for value in local)
        else boundary_disposition
    )
    degree = 1 if degree_disposition is CertificateDisposition.PASS else None
    degree_id = canonical_fingerprint(
        {
            "kind": "mapping-degree",
            "boundary": boundary_id,
            "degree": degree,
            "disposition": degree_disposition.value,
        }
    )
    mapping_degree = MappingDegreeCertificate(
        degree_disposition,
        degree,
        boundary_id,
        "Gale-Nikaido block univalence plus oriented exact gluing and separated control hulls",
        degree_id,
    )
    identifier = canonical_fingerprint(
        {
            "kind": "f1-global-injectivity",
            "complex": complex_.complex_id,
            "locals": [value.certificate_id for value in local],
            "boundary": boundary_id,
            "degree": degree_id,
            "disposition": disposition.value,
        }
    )
    return GlobalInjectivityCertificate(
        disposition,
        local,
        boundary,
        mapping_degree,
        tuple(diagnostics),
        counterexample,
        complex_.complex_id,
        identifier,
    )


def tensor_nurbs_cell_bounds(
    volume: TensorNURBSVolume,
    /,
    *,
    policy: TensorNURBSCertificatePolicy | None = None,
) -> tuple[CellJacobianEvidence, ...]:
    """Expose one nonadaptive analytic bound pass for barrier parameterization."""
    if not isinstance(volume, TensorNURBSVolume):
        raise TypeError("volume must be a TensorNURBSVolume.")
    policy_ = (
        TensorNURBSCertificatePolicy(
            maximum_subdivision_depth=0,
            maximum_subboxes=max(1, volume.basis.cell_count),
        )
        if policy is None
        else policy
    )
    return tuple(
        _cell_evidence(
            volume.block_id,
            span_index,
            (),
            homogeneous,
            bounds,
            volume.ambient_dimension,
            "determinant",
            policy_,
        )
        for homogeneous, bounds, span_index in _bezier_homogeneous_cells(volume)
    )


__all__ = [
    "BoundaryInjectivityCertificate",
    "CellJacobianEvidence",
    "CertificateCounterexample",
    "CertificateDiagnostic",
    "CertificateDisposition",
    "DeformedJacobianCertificate",
    "GlobalInjectivityCertificate",
    "IntervalBound",
    "LocalGeometryCertificate",
    "MappingDegreeCertificate",
    "SurfaceEmbeddingCertificate",
    "TensorNURBSCertificatePolicy",
    "certify_deformed_tensor_nurbs",
    "certify_global_injectivity",
    "certify_surface_embedding",
    "certify_tensor_nurbs",
    "tensor_nurbs_cell_bounds",
]
