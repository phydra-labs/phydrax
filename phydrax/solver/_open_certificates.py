#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite/truncation-scoped open-system refinement and uniqueness certificates."""

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..linalg import (
    DenseLinearOperator,
    FactorizationPolicy,
    factorize,
    HermitianSpectrum,
)


class FiniteRefinementCertificate(StrictModule):
    parameter_values: Array
    observable_values: Array
    successive_differences: Array
    boundary_indicators: Array
    stabilized: Array
    remainder: Array
    valid: Array
    axis: str = eqx.field(static=True)
    estimate_kind: Literal["difference", "bound"] = eqx.field(static=True)
    assumptions: tuple[str, ...] = eqx.field(static=True)
    claim: str = eqx.field(static=True)


def certify_finite_refinement(
    parameter_values: ArrayLike,
    observable_values: ArrayLike,
    boundary_indicators: ArrayLike,
    /,
    *,
    axis: str,
    tolerance: float,
    certified_future_contraction: float | None = None,
) -> FiniteRefinementCertificate:
    """Certify only stabilization over a declared finite nested sequence."""
    parameters = jnp.asarray(parameter_values)
    observables = jnp.asarray(observable_values)
    boundary = jnp.asarray(boundary_indicators)
    if (
        parameters.ndim != 1
        or parameters.shape[0] < 2
        or observables.shape[0] != parameters.shape[0]
        or boundary.shape[0] != parameters.shape[0]
    ):
        raise ValueError(
            "Refinement arrays must share a stage axis with at least two stages."
        )
    if not isinstance(axis, str) or not axis or tolerance < 0.0:
        raise ValueError("axis/tolerance are invalid.")
    differences = jnp.max(
        jnp.abs(observables[1:] - observables[:-1]).reshape(
            (parameters.shape[0] - 1, -1)
        ),
        axis=-1,
    )
    stabilized = differences[-1] <= tolerance
    if certified_future_contraction is None:
        remainder = differences[-1]
        kind = "difference"
        assumptions = ()
    else:
        contraction = float(certified_future_contraction)
        if not np.isfinite(contraction) or not 0.0 <= contraction < 1.0:
            raise ValueError("certified_future_contraction must lie in [0, 1).")
        remainder = differences[-1] * contraction / (1.0 - contraction)
        kind = "bound"
        assumptions = (f"certified-future-contraction<{contraction}",)
    valid = (
        jnp.all(jnp.isfinite(parameters))
        & jnp.all(jnp.isfinite(observables))
        & jnp.all(jnp.isfinite(boundary))
        & jnp.all(jnp.diff(parameters) > 0.0)
    )
    return FiniteRefinementCertificate(
        parameter_values=parameters,
        observable_values=observables,
        successive_differences=differences,
        boundary_indicators=boundary,
        stabilized=stabilized,
        remainder=remainder,
        valid=valid,
        axis=axis,
        estimate_kind=kind,
        assumptions=assumptions,
        claim="stabilized-over-declared-refinements-not-unbounded-convergence",
    )


class FiniteSteadyStateCertificate(StrictModule):
    density: Array
    right_nullity: Array
    left_trace_residual: Array
    stationary_residual: Array
    hermiticity_residual: Array
    minimum_eigenvalue: Array
    trace_residual: Array
    detailed_balance_residual: Array
    certified_gap: Array
    unique: Array
    physical: Array
    valid: Array
    dimension: int = eqx.field(static=True)
    gap_claim: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


def certify_finite_lindblad_steady_state(
    liouvillian: ArrayLike,
    dimension: int,
    /,
    *,
    tolerance: float = 1e-9,
    detailed_balance_symmetrizer: ArrayLike | None = None,
) -> FiniteSteadyStateCertificate:
    """Factor the explicit d² Liouvillian; finite trajectories are not accepted."""
    matrix = jnp.asarray(liouvillian)
    size = int(dimension)
    if size <= 0 or matrix.shape != (size * size, size * size):
        raise ValueError("liouvillian must be the explicit square d² generator.")
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative.")
    decomposition = factorize(DenseLinearOperator(matrix), FactorizationPolicy("svd"))
    rank = decomposition.rank()
    nullity = size * size - rank
    trace_vector = jnp.eye(size, dtype=matrix.dtype).reshape(-1)
    left_trace_residual = jnp.sqrt(
        jnp.sum(jnp.abs(jnp.conj(matrix.T) @ trace_vector) ** 2)
    )
    augmented = jnp.concatenate((matrix, trace_vector[None, :]), axis=0)
    rhs = jnp.concatenate(
        (
            jnp.zeros((size * size,), dtype=matrix.dtype),
            jnp.ones((1,), dtype=matrix.dtype),
        )
    )
    constrained = factorize(
        DenseLinearOperator(augmented), FactorizationPolicy("qr")
    ).solve(rhs)
    density = jnp.asarray(constrained.value).reshape((size, size))
    stationary = jnp.sqrt(jnp.sum(jnp.abs(matrix @ density.reshape(-1)) ** 2))
    hermiticity = jnp.max(jnp.abs(density - jnp.conj(density.T)))
    trace_residual = jnp.abs(jnp.trace(density) - 1.0)
    spectrum = HermitianSpectrum(
        0.5 * (density + jnp.conj(density.T)), tolerance=tolerance
    )
    physical = (
        constrained.successful
        & spectrum.valid
        & (stationary <= tolerance)
        & (hermiticity <= tolerance)
        & (trace_residual <= tolerance)
        & (spectrum.minimum_eigenvalue >= -tolerance)
    )
    unique = (nullity == 1) & (left_trace_residual <= tolerance)
    if detailed_balance_symmetrizer is None:
        balance_residual = jnp.asarray(jnp.nan)
        gap = jnp.asarray(jnp.nan)
        gap_claim = "none-generic-nonnormal-separation-is-not-a-certified-rate"
        balance_valid = jnp.asarray(True)
    else:
        symmetrizer = jnp.asarray(detailed_balance_symmetrizer)
        if symmetrizer.shape != matrix.shape:
            raise ValueError("detailed_balance_symmetrizer must have d² square shape.")
        symmetrizer_scale = jnp.max(jnp.abs(symmetrizer))
        normalization_valid = jnp.isfinite(symmetrizer_scale) & (symmetrizer_scale > 0.0)
        safe_scale = jnp.where(normalization_valid, symmetrizer_scale, 1.0)
        normalized_symmetrizer = jnp.where(
            normalization_valid,
            symmetrizer / safe_scale,
            jnp.eye(matrix.shape[0], dtype=symmetrizer.dtype),
        )
        metric_spectrum = HermitianSpectrum(normalized_symmetrizer, tolerance=tolerance)
        metric_positive = metric_spectrum.valid & (
            metric_spectrum.minimum_eigenvalue > tolerance
        )
        safe_metric_eigenvalues = jnp.where(
            metric_positive,
            metric_spectrum.eigenvalues,
            jnp.ones_like(metric_spectrum.eigenvalues),
        )
        metric_eigenvectors = metric_spectrum.eigenvectors
        metric_eigenvectors_adjoint = jnp.conj(metric_eigenvectors.T)
        metric_root = jnp.sqrt(safe_metric_eigenvalues)
        matrix_in_metric_basis = (
            metric_eigenvectors_adjoint @ matrix @ metric_eigenvectors
        )
        left_action = metric_spectrum.eigenvalues[:, None] * matrix_in_metric_basis
        right_action = (
            jnp.conj(matrix_in_metric_basis.T) * metric_spectrum.eigenvalues[None, :]
        )
        balance_scale = jnp.maximum(
            jnp.maximum(
                jnp.sqrt(jnp.sum(jnp.abs(left_action) ** 2)),
                jnp.sqrt(jnp.sum(jnp.abs(right_action) ** 2)),
            ),
            jnp.finfo(metric_spectrum.eigenvalues.dtype).tiny,
        )
        balance_residual = (
            jnp.sqrt(jnp.sum(jnp.abs(left_action - right_action) ** 2)) / balance_scale
        )
        balanced_generator = -(
            metric_root[:, None]
            * matrix_in_metric_basis
            * jnp.reciprocal(metric_root)[None, :]
        )
        decay_spectrum = HermitianSpectrum(
            balanced_generator,
            tolerance=tolerance,
        )
        decay = decay_spectrum.eigenvalues
        decay_scale = jnp.maximum(jnp.max(jnp.abs(decay)), 1.0)
        nonnegative_decay = decay_spectrum.minimum_eigenvalue >= (
            -tolerance * decay_scale
        )
        zero_mode = jnp.abs(decay[0]) <= tolerance * decay_scale
        if decay.shape[0] > 1:
            gap_candidate = decay[1]
            has_gap = jnp.asarray(True)
        else:
            gap_candidate = jnp.asarray(jnp.nan, dtype=decay.dtype)
            has_gap = jnp.asarray(False)
        balance_valid = (
            normalization_valid
            & metric_positive
            & jnp.isfinite(balance_residual)
            & (balance_residual <= tolerance)
            & decay_spectrum.valid
            & nonnegative_decay
            & zero_mode
            & has_gap
            & (gap_candidate >= 0.0)
            & jnp.isfinite(gap_candidate)
        )
        gap = jnp.where(
            balance_valid,
            gap_candidate,
            jnp.asarray(jnp.nan, dtype=decay.dtype),
        )
        gap_claim = "finite-detailed-balance-similarity-gap"
    valid = unique & physical & balance_valid & jnp.all(jnp.isfinite(matrix))
    return FiniteSteadyStateCertificate(
        density=density,
        right_nullity=nullity,
        left_trace_residual=left_trace_residual,
        stationary_residual=stationary,
        hermiticity_residual=hermiticity,
        minimum_eigenvalue=spectrum.minimum_eigenvalue,
        trace_residual=trace_residual,
        detailed_balance_residual=balance_residual,
        certified_gap=gap,
        unique=unique,
        physical=physical,
        valid=valid,
        dimension=size,
        gap_claim=gap_claim,
        claim="unique-only-in-declared-finite-liouville-space",
    )


class ProcessIdentifiabilityCertificate(StrictModule):
    projected_design: Array
    numerical_rank: Array
    quotient_dimension: Array
    gauge_orthonormality_residual: Array
    identifiable: Array
    valid: Array
    design_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


def certify_process_identifiability(
    design_jacobian: ArrayLike,
    gauge_basis: ArrayLike,
    /,
    *,
    tolerance: float = 1e-9,
    design_id: str,
) -> ProcessIdentifiabilityCertificate:
    """Certify rank only modulo a caller-declared orthonormal finite gauge basis."""
    design, gauge = jnp.asarray(design_jacobian), jnp.asarray(gauge_basis)
    if design.ndim != 2 or gauge.ndim != 2 or gauge.shape[0] != design.shape[1]:
        raise ValueError("design_jacobian/gauge_basis finite parameter axes disagree.")
    if not isinstance(design_id, str) or not design_id:
        raise ValueError("design_id must be nonempty.")
    gram = jnp.conj(gauge.T) @ gauge
    orthonormality = (
        jnp.max(jnp.abs(gram - jnp.eye(gauge.shape[1], dtype=gram.dtype)))
        if gauge.shape[1]
        else jnp.asarray(0.0)
    )
    projector = jnp.eye(design.shape[1], dtype=design.dtype) - gauge @ jnp.conj(gauge.T)
    projected = design @ projector
    decomposition = factorize(DenseLinearOperator(projected), FactorizationPolicy("svd"))
    rank = decomposition.rank()
    quotient_dimension = jnp.asarray(design.shape[1] - gauge.shape[1], dtype=jnp.int32)
    identifiable = rank == quotient_dimension
    valid = jnp.all(jnp.isfinite(projected)) & (orthonormality <= tolerance)
    return ProcessIdentifiabilityCertificate(
        projected_design=projected,
        numerical_rank=rank,
        quotient_dimension=quotient_dimension,
        gauge_orthonormality_residual=orthonormality,
        identifiable=identifiable,
        valid=jnp.asarray(valid) & identifiable,
        design_id=design_id,
        claim="unique-only-in-declared-finite-quotient-and-design",
    )


__all__ = [
    "FiniteRefinementCertificate",
    "FiniteSteadyStateCertificate",
    "ProcessIdentifiabilityCertificate",
    "certify_finite_lindblad_steady_state",
    "certify_finite_refinement",
    "certify_process_identifiability",
]
