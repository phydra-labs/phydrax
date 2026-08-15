#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..._strict import StrictModule
from ._schur import (
    PreparedSchurSolve,
    schur_eigensolve,
    SchurEigenproblem,
    SchurSolvePlan,
    SchurSolvePolicy,
    SchurSolveResult,
    SchurSolveStatus,
)


class SpectralObservableStatus(IntEnum):
    """Aggregate status for observables derived from one full Schur form."""

    SUCCESS = 0
    SOURCE_FAILURE = 1
    NONFINITE = 2


class SpectralStabilityStatus(IntEnum):
    """Continuous- or discrete-time linear spectral stability class."""

    STABLE = 0
    MARGINAL = 1
    UNSTABLE = 2
    UNKNOWN = 3


class SpectralObservableProvenance(StrictModule):
    """Source Schur identities and definitions used by the observable bundle."""

    problem_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    eigenvalue_ordering: str = eqx.field(static=True)
    numerical_abscissa_definition: str = eqx.field(static=True)
    numeric_version: Array


class SchurSpectralObservables(StrictModule):
    """Full-spectrum algebraic, nonnormality, and dynamical-stability observables."""

    spectral_radius: Array
    spectral_abscissa: Array
    minimum_modulus: Array
    numerical_abscissa: Array
    trace: Array
    determinant: Array
    log_absolute_determinant: Array
    determinant_phase: Array
    determinant_finite: Array
    singular: Array
    frobenius_norm: Array
    spectral_centroid: Array
    spectral_variance: Array
    minimum_eigenvalue_separation: Array
    departure_from_normality: Array
    continuous_stability_margin: Array
    discrete_stability_margin: Array
    continuous_time_stability: Array
    discrete_time_stability: Array
    status: Array
    source_status: Array
    finite: Array
    provenance: SpectralObservableProvenance

    @property
    def successful(self) -> Array:
        return self.status == int(SpectralObservableStatus.SUCCESS)


def schur_spectral_observables(
    source: SchurSolveResult | SchurEigenproblem | PreparedSchurSolve,
    /,
    *,
    policy: SchurSolvePolicy | SchurSolvePlan | None = None,
    stability_tolerance: float = 0.0,
) -> SchurSpectralObservables:
    """Evaluate full-spectrum observables from, or together with, a Schur solve."""
    tolerance = float(stability_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("stability_tolerance must be finite and non-negative.")
    if isinstance(source, SchurSolveResult):
        if policy is not None:
            raise ValueError("policy must be omitted when source is a SchurSolveResult.")
        result = source
    elif isinstance(source, (SchurEigenproblem, PreparedSchurSolve)):
        result = schur_eigensolve(source, policy=policy)
    else:
        raise TypeError(
            "source must be a SchurSolveResult, SchurEigenproblem, or PreparedSchurSolve."
        )
    eigenvalues = result.eigenvalues
    moduli = jnp.abs(eigenvalues)
    spectral_radius = jnp.max(moduli)
    spectral_abscissa = jnp.max(jnp.real(eigenvalues))
    minimum_modulus = jnp.min(moduli)
    hermitian_part = 0.5 * (result.schur_form + jnp.conj(result.schur_form.T))
    numerical_abscissa = jnp.max(jnp.linalg.eigvalsh(hermitian_part))
    trace = jnp.sum(eigenvalues)
    determinant = jnp.prod(eigenvalues)
    nonzero = moduli > 0
    log_absolute_determinant = jnp.sum(jnp.log(moduli))
    determinant_phase = jnp.where(
        jnp.all(nonzero),
        jnp.prod(eigenvalues / jnp.where(nonzero, moduli, 1)),
        jnp.asarray(0, dtype=eigenvalues.dtype),
    )
    centroid = trace / eigenvalues.size
    spectral_variance = jnp.mean(jnp.abs(eigenvalues - centroid) ** 2)
    minimum_separation = jnp.min(result.diagnostics.eigenvalue_separation)
    continuous_margin = -spectral_abscissa
    discrete_margin = 1.0 - spectral_radius
    finite = (
        jnp.all(jnp.isfinite(eigenvalues))
        & jnp.isfinite(spectral_radius)
        & jnp.isfinite(spectral_abscissa)
        & jnp.isfinite(numerical_abscissa)
        & jnp.isfinite(trace)
        & jnp.isfinite(spectral_variance)
    )
    source_success = result.status == int(SchurSolveStatus.SUCCESS)
    status = jnp.where(
        ~finite,
        int(SpectralObservableStatus.NONFINITE),
        jnp.where(
            ~source_success,
            int(SpectralObservableStatus.SOURCE_FAILURE),
            int(SpectralObservableStatus.SUCCESS),
        ),
    ).astype(jnp.int32)
    continuous_stability = _continuous_stability(
        spectral_abscissa,
        tolerance,
        finite,
    )
    discrete_stability = _discrete_stability(spectral_radius, tolerance, finite)
    return SchurSpectralObservables(
        spectral_radius=spectral_radius,
        spectral_abscissa=spectral_abscissa,
        minimum_modulus=minimum_modulus,
        numerical_abscissa=numerical_abscissa,
        trace=trace,
        determinant=determinant,
        log_absolute_determinant=log_absolute_determinant,
        determinant_phase=determinant_phase,
        determinant_finite=jnp.isfinite(determinant),
        singular=jnp.any(~nonzero),
        frobenius_norm=jnp.linalg.norm(result.schur_form),
        spectral_centroid=centroid,
        spectral_variance=spectral_variance,
        minimum_eigenvalue_separation=minimum_separation,
        departure_from_normality=result.diagnostics.departure_from_normality,
        continuous_stability_margin=continuous_margin,
        discrete_stability_margin=discrete_margin,
        continuous_time_stability=continuous_stability,
        discrete_time_stability=discrete_stability,
        status=status,
        source_status=result.status,
        finite=finite,
        provenance=SpectralObservableProvenance(
            problem_id=result.provenance.problem_id,
            plan_id=result.provenance.plan_id,
            prepared_id=result.provenance.prepared_id,
            operator_id=result.provenance.operator_id,
            eigenvalue_ordering=result.provenance.ordering,
            numerical_abscissa_definition="max eig((T + T^H) / 2)",
            numeric_version=result.provenance.numeric_version,
        ),
    )


def _continuous_stability(
    spectral_abscissa: Array,
    tolerance: float,
    finite: Array,
    /,
) -> Array:
    classified = jnp.where(
        spectral_abscissa < -tolerance,
        int(SpectralStabilityStatus.STABLE),
        jnp.where(
            spectral_abscissa > tolerance,
            int(SpectralStabilityStatus.UNSTABLE),
            int(SpectralStabilityStatus.MARGINAL),
        ),
    )
    return jnp.where(
        finite,
        classified,
        int(SpectralStabilityStatus.UNKNOWN),
    ).astype(jnp.int32)


def _discrete_stability(
    spectral_radius: Array,
    tolerance: float,
    finite: Array,
    /,
) -> Array:
    classified = jnp.where(
        spectral_radius < 1.0 - tolerance,
        int(SpectralStabilityStatus.STABLE),
        jnp.where(
            spectral_radius > 1.0 + tolerance,
            int(SpectralStabilityStatus.UNSTABLE),
            int(SpectralStabilityStatus.MARGINAL),
        ),
    )
    return jnp.where(
        finite,
        classified,
        int(SpectralStabilityStatus.UNKNOWN),
    ).astype(jnp.int32)


__all__ = [
    "SchurSpectralObservables",
    "SpectralObservableProvenance",
    "SpectralObservableStatus",
    "SpectralStabilityStatus",
    "schur_spectral_observables",
]
