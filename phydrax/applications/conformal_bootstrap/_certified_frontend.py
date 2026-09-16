#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Global-block PMP compilation, half-line SOS certificates, and bound studies."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, localcontext
from itertools import pairwise
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...linalg import certify_interval_psd, DecimalInterval
from ._global_blocks import PreparedGlobalScalarBlocks
from ._pmp import (
    ConformalPolynomialMatrixProgram,
    DampedRationalPrefactor,
    PolynomialMatrixBlock,
)


BootstrapCertificateStatus: TypeAlias = Literal[
    "certified", "numerical-candidate", "inconclusive"
]
NavigatorStatus: TypeAlias = Literal["converged", "inconclusive", "empty"]


class GlobalBlockPMPPlan(StrictModule):
    """Finite polynomial frontend for prepared global-block derivatives."""

    blocks: PreparedGlobalScalarBlocks
    minimum_dimensions: tuple[float, ...] = eqx.field(static=True)
    maximum_dimensions: tuple[float, ...] = eqx.field(static=True)
    polynomial_degree: int = eqx.field(static=True)
    sample_count: int = eqx.field(static=True)
    significant_digits: int = eqx.field(static=True)
    maximum_fit_error: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        blocks: PreparedGlobalScalarBlocks,
        minimum_dimensions: Sequence[float],
        maximum_dimensions: Sequence[float],
        /,
        *,
        polynomial_degree: int,
        sample_count: int,
        significant_digits: int = 17,
        maximum_fit_error: float = 1e-6,
    ):
        if not isinstance(blocks, PreparedGlobalScalarBlocks):
            raise TypeError("blocks must be PreparedGlobalScalarBlocks.")
        minimum = tuple(float(value) for value in minimum_dimensions)
        maximum = tuple(float(value) for value in maximum_dimensions)
        degree = int(polynomial_degree)
        samples = int(sample_count)
        digits = int(significant_digits)
        error = float(maximum_fit_error)
        if len(minimum) != len(blocks.plan.spins) or len(maximum) != len(minimum):
            raise ValueError("One dimension interval is required per prepared spin.")
        if any(
            not math.isfinite(left)
            or not math.isfinite(right)
            or left >= right
            or left <= blocks._unitarity_bound(spin)
            for left, right, spin in zip(minimum, maximum, blocks.plan.spins, strict=True)
        ):
            raise ValueError("PMP dimension intervals must lie above unitarity.")
        if degree < 0 or samples < degree + 2 or digits < 8 or error <= 0.0:
            raise ValueError("Polynomial fitting controls are invalid.")
        content = {
            "kind": "global-block-pmp-plan",
            "blocks": blocks.prepared_id,
            "minimum_dimensions": minimum,
            "maximum_dimensions": maximum,
            "polynomial_degree": degree,
            "sample_count": samples,
            "significant_digits": digits,
            "maximum_fit_error": error,
        }
        self.blocks = blocks
        self.minimum_dimensions = minimum
        self.maximum_dimensions = maximum
        self.polynomial_degree = degree
        self.sample_count = samples
        self.significant_digits = digits
        self.maximum_fit_error = error
        self.plan_id = canonical_fingerprint(content)


class GlobalBlockPMPFitEvidence(StrictModule):
    maximum_relative_errors: Array
    sample_dimensions: Array
    finite: Array
    accepted: Array
    evidence_id: str = eqx.field(static=True)


class CompiledGlobalBlockPMP(StrictModule):
    plan: GlobalBlockPMPPlan
    program: ConformalPolynomialMatrixProgram
    evidence: GlobalBlockPMPFitEvidence
    compiled_id: str = eqx.field(static=True)


def compile_global_blocks_to_pmp(plan: GlobalBlockPMPPlan, /) -> CompiledGlobalBlockPMP:
    """Fit each prepared derivative component on a declared compact gap interval."""

    if not isinstance(plan, GlobalBlockPMPPlan):
        raise TypeError("plan must be GlobalBlockPMPPlan.")
    blocks: list[PolynomialMatrixBlock] = []
    errors: list[float] = []
    dimensions_by_spin: list[np.ndarray] = []
    functional_count = len(plan.blocks.plan.derivative_orders) * int(
        plan.blocks.plan.evaluation_points.shape[0]
    )
    for spin, minimum, maximum in zip(
        plan.blocks.plan.spins,
        plan.minimum_dimensions,
        plan.maximum_dimensions,
        strict=True,
    ):
        chebyshev_nodes = np.cos(
            np.pi * (np.arange(plan.sample_count) + 0.5) / plan.sample_count
        )
        dimensions = minimum + 0.5 * (maximum - minimum) * (1.0 + chebyshev_nodes)
        dimensions.sort()
        coordinate = dimensions - minimum
        table = np.stack(
            [
                np.asarray(plan.blocks.derivative_table(value, spin)).reshape(-1)
                for value in dimensions
            ],
            axis=0,
        )
        if table.shape != (plan.sample_count, functional_count):
            raise ValueError("Prepared block derivative table changed frontend shape.")
        polynomials: list[tuple[str, ...]] = []
        fitted = np.empty_like(table)
        for functional in range(functional_count):
            coefficients = np.polynomial.polynomial.polyfit(
                coordinate,
                table[:, functional],
                plan.polynomial_degree,
            )
            fitted[:, functional] = np.polynomial.polynomial.polyval(
                coordinate,
                coefficients,
            )
            polynomials.append(
                tuple(
                    format(float(value), f".{plan.significant_digits}g")
                    for value in coefficients
                )
            )
        scale = np.maximum(1.0, np.abs(table))
        error = float(np.max(np.abs(table - fitted) / scale))
        errors.append(error)
        dimensions_by_spin.append(dimensions)
        blocks.append(
            PolynomialMatrixBlock(
                DampedRationalPrefactor("1", "1"),
                (((tuple(polynomials)),),),
                sample_points=tuple(
                    format(float(value - minimum), f".{plan.significant_digits}g")
                    for value in dimensions
                ),
                sample_scalings=("1",) * plan.sample_count,
                reduced_sample_scalings=("1",) * plan.sample_count,
            )
        )
    accepted = bool(np.all(np.asarray(errors) <= plan.maximum_fit_error))
    if not accepted:
        raise ValueError("Global-block polynomial fit exceeds maximum_fit_error.")
    objective = ("0",) * functional_count
    normalization = ("1",) + ("0",) * (functional_count - 1)
    program = ConformalPolynomialMatrixProgram(
        objective,
        normalization,
        tuple(blocks),
        frontend_id=plan.plan_id,
        frontend_precision_bits=math.ceil(plan.significant_digits * math.log2(10.0)),
    )
    sample_table = np.stack(dimensions_by_spin, axis=0)
    evidence_id = canonical_fingerprint(
        {
            "kind": "global-block-pmp-fit-evidence",
            "plan": plan.plan_id,
            "errors": errors,
            "sample_dimensions": array_tree_fingerprint(sample_table),
        }
    )
    evidence = GlobalBlockPMPFitEvidence(
        maximum_relative_errors=jnp.asarray(errors),
        sample_dimensions=jnp.asarray(sample_table),
        finite=jnp.asarray(np.all(np.isfinite(sample_table))),
        accepted=jnp.asarray(accepted),
        evidence_id=evidence_id,
    )
    compiled_id = canonical_fingerprint(
        {
            "kind": "compiled-global-block-pmp",
            "plan": plan.plan_id,
            "pmp": program.pmp_id,
            "evidence": evidence_id,
        }
    )
    return CompiledGlobalBlockPMP(
        plan=plan,
        program=program,
        evidence=evidence,
        compiled_id=compiled_id,
    )


@dataclass(frozen=True, slots=True)
class HalfLineSOSWitness:
    """Exact-decimal witness P(x) = VᵀQ₀V + x WᵀQ₁W."""

    constant_gram: tuple[tuple[str, ...], ...]
    linear_gram: tuple[tuple[str, ...], ...]
    matrix_dimension: int
    witness_id: str

    def __init__(
        self,
        constant_gram: Sequence[Sequence[str]],
        linear_gram: Sequence[Sequence[str]],
        /,
        *,
        matrix_dimension: int = 1,
    ):
        constant = tuple(tuple(str(value) for value in row) for row in constant_gram)
        linear = tuple(tuple(str(value) for value in row) for row in linear_gram)
        dimension = int(matrix_dimension)
        if not constant or any(len(row) != len(constant) for row in constant):
            raise ValueError("constant_gram must be non-empty and square.")
        if not linear or any(len(row) != len(linear) for row in linear):
            raise ValueError("linear_gram must be non-empty and square.")
        if dimension < 1 or len(constant) % dimension or len(linear) % dimension:
            raise ValueError("SOS Gram dimensions must contain complete matrix blocks.")
        for matrix in (constant, linear):
            if any(
                Decimal(value) != Decimal(matrix[j][i])
                for i, row in enumerate(matrix)
                for j, value in enumerate(row)
            ):
                raise ValueError("SOS Gram matrices must be exactly symmetric.")
        content = {
            "kind": "half-line-sos-witness",
            "constant_gram": constant,
            "linear_gram": linear,
            "matrix_dimension": dimension,
        }
        object.__setattr__(self, "constant_gram", constant)
        object.__setattr__(self, "linear_gram", linear)
        object.__setattr__(self, "matrix_dimension", dimension)
        object.__setattr__(self, "witness_id", canonical_fingerprint(content))


@dataclass(frozen=True, slots=True)
class PMPContinuumCertificate:
    pmp_id: str
    functional: tuple[str, ...]
    coefficient_residuals: tuple[str, ...]
    gram_certificate_ids: tuple[str, ...]
    continuum_positive: bool
    certificate_id: str


DecimalMatrixPolynomial = tuple[tuple[tuple[Decimal, ...], ...], ...]


def _combined_matrix_polynomial(
    block: PolynomialMatrixBlock,
    functional: tuple[Decimal, ...],
    /,
) -> DecimalMatrixPolynomial:
    dimension = block.matrix_dimension
    degree = max(
        len(polynomial)
        for row in block.polynomials
        for entry in row
        for polynomial in entry
    )
    coefficients = [
        [[Decimal(0) for _ in range(dimension)] for _ in range(dimension)]
        for _ in range(degree)
    ]
    for row in range(dimension):
        for column in range(dimension):
            for weight, polynomial in zip(
                functional,
                block.polynomials[row][column],
                strict=True,
            ):
                for order, value in enumerate(polynomial):
                    coefficients[order][row][column] += weight * Decimal(value)
    return tuple(
        tuple(tuple(value for value in row) for row in matrix) for matrix in coefficients
    )


def _witness_matrix_coefficients(
    witness: HalfLineSOSWitness,
    /,
) -> DecimalMatrixPolynomial:
    dimension = witness.matrix_dimension
    constant = tuple(
        tuple(Decimal(value) for value in row) for row in witness.constant_gram
    )
    linear = tuple(tuple(Decimal(value) for value in row) for row in witness.linear_gram)
    constant_order = len(constant) // dimension
    linear_order = len(linear) // dimension
    degree = max(2 * (constant_order - 1), 2 * (linear_order - 1) + 1)
    coefficients = [
        [[Decimal(0) for _ in range(dimension)] for _ in range(dimension)]
        for _ in range(degree + 1)
    ]
    for left_power in range(constant_order):
        for right_power in range(constant_order):
            for row in range(dimension):
                for column in range(dimension):
                    coefficients[left_power + right_power][row][column] += constant[
                        left_power * dimension + row
                    ][right_power * dimension + column]
    for left_power in range(linear_order):
        for right_power in range(linear_order):
            for row in range(dimension):
                for column in range(dimension):
                    coefficients[left_power + right_power + 1][row][column] += linear[
                        left_power * dimension + row
                    ][right_power * dimension + column]
    while len(coefficients) > 1 and all(
        value == 0 for row in coefficients[-1] for value in row
    ):
        coefficients.pop()
    return tuple(
        tuple(tuple(value for value in row) for row in matrix) for matrix in coefficients
    )


def certify_pmp_continuum_positivity(
    program: ConformalPolynomialMatrixProgram,
    functional: Sequence[str],
    witnesses: Sequence[HalfLineSOSWitness],
    /,
    *,
    decimal_tolerance: str = "1e-40",
    precision: int = 100,
) -> PMPContinuumCertificate:
    """Reconstruct every polynomial matrix and certify its half-line SOS witness."""

    if not isinstance(program, ConformalPolynomialMatrixProgram):
        raise TypeError("program must be ConformalPolynomialMatrixProgram.")
    functional_ = tuple(str(value) for value in functional)
    if len(functional_) != program.functional_count:
        raise ValueError("Functional dimension does not match the PMP.")
    weights = tuple(Decimal(value) for value in functional_)
    witnesses_ = tuple(witnesses)
    if len(witnesses_) != len(program.blocks) or any(
        not isinstance(value, HalfLineSOSWitness) for value in witnesses_
    ):
        raise TypeError("One HalfLineSOSWitness is required per PMP block.")
    tolerance = Decimal(decimal_tolerance)
    if tolerance < 0:
        raise ValueError("decimal_tolerance must be non-negative.")
    residuals: list[str] = []
    gram_ids: list[str] = []
    valid = True
    with localcontext() as context:
        context.prec = int(precision)
        for block, witness in zip(program.blocks, witnesses_, strict=True):
            if witness.matrix_dimension != block.matrix_dimension:
                raise ValueError("SOS witness matrix dimension does not match its block.")
            polynomial = _combined_matrix_polynomial(block, weights)
            reconstructed = _witness_matrix_coefficients(witness)
            count = max(len(polynomial), len(reconstructed))
            difference: list[Decimal] = []
            for order in range(count):
                for row in range(block.matrix_dimension):
                    for column in range(block.matrix_dimension):
                        direct = (
                            polynomial[order][row][column]
                            if order < len(polynomial)
                            else Decimal(0)
                        )
                        witness_value = (
                            reconstructed[order][row][column]
                            if order < len(reconstructed)
                            else Decimal(0)
                        )
                        difference.append(direct - witness_value)
            residual = max((abs(value) for value in difference), default=Decimal(0))
            residuals.append(str(residual))
            constant_certificate = certify_interval_psd(
                tuple(
                    tuple(
                        DecimalInterval.point(value, precision=precision) for value in row
                    )
                    for row in witness.constant_gram
                ),
                precision=precision,
            )
            linear_certificate = certify_interval_psd(
                tuple(
                    tuple(
                        DecimalInterval.point(value, precision=precision) for value in row
                    )
                    for row in witness.linear_gram
                ),
                precision=precision,
            )
            gram_ids.extend(
                (constant_certificate.certificate_id, linear_certificate.certificate_id)
            )
            valid &= (
                residual <= tolerance
                and constant_certificate.positive_semidefinite
                and linear_certificate.positive_semidefinite
            )
    content = {
        "kind": "pmp-continuum-certificate",
        "pmp_id": program.pmp_id,
        "functional": functional_,
        "witnesses": [value.witness_id for value in witnesses_],
        "coefficient_residuals": residuals,
        "gram_certificate_ids": gram_ids,
        "continuum_positive": valid,
        "decimal_tolerance": str(tolerance),
        "precision": int(precision),
    }
    return PMPContinuumCertificate(
        program.pmp_id,
        functional_,
        tuple(residuals),
        tuple(gram_ids),
        valid,
        canonical_fingerprint(content),
    )


@dataclass(frozen=True, slots=True)
class CertifiedBootstrapResult:
    status: BootstrapCertificateStatus
    bound: float
    primal_dual_gap: float
    block_error_bound: float
    spin_tail_certified: bool
    certificate_id: str
    result_id: str


def certify_bootstrap_bound(
    certificate: PMPContinuumCertificate,
    bound: float,
    /,
    *,
    primal_dual_gap: float,
    maximum_primal_dual_gap: float,
    block_error_bound: float,
    maximum_block_error: float,
    spin_tail_certified: bool,
) -> CertifiedBootstrapResult:
    """Promote a numerical candidate only when every independent gate is valid."""

    if not isinstance(certificate, PMPContinuumCertificate):
        raise TypeError("certificate must be PMPContinuumCertificate.")
    values = tuple(
        float(value)
        for value in (
            bound,
            primal_dual_gap,
            maximum_primal_dual_gap,
            block_error_bound,
            maximum_block_error,
        )
    )
    if not all(math.isfinite(value) for value in values) or any(
        value < 0.0 for value in values[1:]
    ):
        raise ValueError(
            "Bootstrap bound and error controls must be finite and non-negative."
        )
    if (
        certificate.continuum_positive
        and primal_dual_gap <= maximum_primal_dual_gap
        and block_error_bound <= maximum_block_error
        and spin_tail_certified
    ):
        status: BootstrapCertificateStatus = "certified"
    elif certificate.continuum_positive:
        status = "numerical-candidate"
    else:
        status = "inconclusive"
    content = {
        "kind": "certified-bootstrap-result",
        "status": status,
        "bound": values[0],
        "primal_dual_gap": values[1],
        "maximum_primal_dual_gap": values[2],
        "block_error_bound": values[3],
        "maximum_block_error": values[4],
        "spin_tail_certified": bool(spin_tail_certified),
        "certificate_id": certificate.certificate_id,
    }
    return CertifiedBootstrapResult(
        status,
        values[0],
        values[1],
        values[3],
        bool(spin_tail_certified),
        certificate.certificate_id,
        canonical_fingerprint(content),
    )


@dataclass(frozen=True, slots=True)
class BootstrapNavigatorLevel:
    derivative_order: int
    spin_cutoff: int
    coordinates: tuple[float, ...]
    navigator_values: tuple[float, ...]
    level_id: str

    def __init__(
        self,
        derivative_order: int,
        spin_cutoff: int,
        coordinates: Sequence[float],
        navigator_values: Sequence[float],
        /,
    ):
        derivative = int(derivative_order)
        spin = int(spin_cutoff)
        points = tuple(float(value) for value in coordinates)
        values = tuple(float(value) for value in navigator_values)
        if derivative < 1 or spin < 0 or len(points) < 2 or len(points) != len(values):
            raise ValueError("Navigator level dimensions are invalid.")
        if tuple(sorted(points)) != points or len(set(points)) != len(points):
            raise ValueError("Navigator coordinates must be unique and increasing.")
        if not all(math.isfinite(value) for value in (*points, *values)):
            raise ValueError("Navigator coordinates and values must be finite.")
        content = {
            "kind": "bootstrap-navigator-level",
            "derivative_order": derivative,
            "spin_cutoff": spin,
            "coordinates": points,
            "navigator_values": values,
        }
        object.__setattr__(self, "derivative_order", derivative)
        object.__setattr__(self, "spin_cutoff", spin)
        object.__setattr__(self, "coordinates", points)
        object.__setattr__(self, "navigator_values", values)
        object.__setattr__(self, "level_id", canonical_fingerprint(content))


@dataclass(frozen=True, slots=True)
class BootstrapNavigatorStudy:
    status: NavigatorStatus
    intervals: tuple[tuple[float, float], ...]
    nested: bool
    final_width: float
    study_id: str


def analyze_bootstrap_navigator(
    levels: Sequence[BootstrapNavigatorLevel],
    /,
    *,
    maximum_final_width: float,
) -> BootstrapNavigatorStudy:
    """Audit nested one-dimensional feasible islands across truncation levels."""

    levels_ = tuple(levels)
    if not levels_ or any(
        not isinstance(value, BootstrapNavigatorLevel) for value in levels_
    ):
        raise TypeError("levels must contain BootstrapNavigatorLevel values.")
    if tuple((value.derivative_order, value.spin_cutoff) for value in levels_) != tuple(
        sorted((value.derivative_order, value.spin_cutoff) for value in levels_)
    ):
        raise ValueError("Navigator levels must be ordered by truncation coordinates.")
    intervals: list[tuple[float, float]] = []
    for level in levels_:
        feasible = tuple(
            coordinate
            for coordinate, value in zip(
                level.coordinates,
                level.navigator_values,
                strict=True,
            )
            if value <= 0.0
        )
        if not feasible:
            content = {
                "kind": "bootstrap-navigator-study",
                "levels": [value.level_id for value in levels_],
                "status": "empty",
            }
            return BootstrapNavigatorStudy(
                "empty",
                tuple(intervals),
                False,
                math.inf,
                canonical_fingerprint(content),
            )
        intervals.append((min(feasible), max(feasible)))
    nested = all(
        current[0] >= previous[0] and current[1] <= previous[1]
        for previous, current in pairwise(intervals)
    )
    final_width = intervals[-1][1] - intervals[-1][0]
    status: NavigatorStatus = (
        "converged"
        if nested and final_width <= float(maximum_final_width)
        else "inconclusive"
    )
    content = {
        "kind": "bootstrap-navigator-study",
        "levels": [value.level_id for value in levels_],
        "status": status,
        "intervals": intervals,
        "nested": nested,
        "final_width": final_width,
        "maximum_final_width": float(maximum_final_width),
    }
    return BootstrapNavigatorStudy(
        status,
        tuple(intervals),
        nested,
        final_width,
        canonical_fingerprint(content),
    )


__all__ = [
    "BootstrapCertificateStatus",
    "BootstrapNavigatorLevel",
    "BootstrapNavigatorStudy",
    "CertifiedBootstrapResult",
    "CompiledGlobalBlockPMP",
    "GlobalBlockPMPFitEvidence",
    "GlobalBlockPMPPlan",
    "HalfLineSOSWitness",
    "NavigatorStatus",
    "PMPContinuumCertificate",
    "analyze_bootstrap_navigator",
    "certify_bootstrap_bound",
    "certify_pmp_continuum_positivity",
    "compile_global_blocks_to_pmp",
]
