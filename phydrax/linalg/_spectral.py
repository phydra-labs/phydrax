#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, PyTree

from .._strict import StrictModule
from ._operators import AbstractLinearOperator
from ._pairings import DiagonalPairing, EuclideanPairing
from ._policies import LinearSolvePolicy
from ._problems import LinearSystem
from ._runtime import solve
from ._spaces import (
    _coordinate_dtype,
    AbstractVectorSpace,
    ArraySpace,
    BlockSpace,
    DualSpace,
    PyTreeSpace,
)
from .krylov import (
    arnoldi,
    golub_kahan,
    KrylovBreakdownStatus,
    lanczos,
)


class SpectralEstimate(StrictModule):
    value: Array
    lower_bound: Array
    upper_bound: Array
    residual_bound: Array
    converged: Array
    iterations: Array
    matvec_count: Array
    adjoint_matvec_count: Array
    method: str = eqx.field(static=True)
    quantity: str = eqx.field(static=True)


class SpectralBoundsEstimate(StrictModule):
    lower: Array
    upper: Array
    lower_residual_bound: Array
    upper_residual_bound: Array
    converged: Array
    iterations: Array
    matvec_count: Array
    method: str = eqx.field(static=True)


class NumericalRangeEstimate(StrictModule):
    center: Array
    radius: Array
    real_interval: Array
    imaginary_interval: Array
    samples: Array
    matvec_count: Array
    method: str = eqx.field(static=True)


class StochasticProbeStatus(IntEnum):
    """Estimator-local status shared by stochastic probe computations."""

    SUCCESS = 0
    TRUNCATED = 1
    KRYLOV_FAILURE = 2
    SOLVE_FAILURE = 3
    NONFINITE = 4
    NOT_EVALUATED = 5


class StochasticProbeDiagnostics(StrictModule):
    """Per-probe source evidence retained by stochastic estimators."""

    source_statuses: Array
    iterations: Array
    residual_norm: Array
    relative_residual: Array
    finite: Array
    converged: Array
    matvec_count: Array
    adjoint_matvec_count: Array


class StochasticEstimate(StrictModule):
    estimate: Array
    standard_error: Array
    samples: Array
    finite: Array
    converged: Array
    probe_converged: Array
    probe_statuses: Array
    probe_error_estimates: Array
    iterations: Array
    num_probes: int = eqx.field(static=True)
    matvec_count: Array
    adjoint_matvec_count: Array
    diagnostics: StochasticProbeDiagnostics
    method: str = eqx.field(static=True)
    quantity: str = eqx.field(static=True)


def estimate_spectral_radius(
    operator: AbstractLinearOperator,
    /,
    *,
    max_dimension: int = 32,
    initial: PyTree[Any] | None = None,
) -> SpectralEstimate:
    """Estimate spectral radius from a breakdown-safe Arnoldi projection."""
    _validate_endomorphism(operator)
    coordinates = _initial_coordinates(operator, initial)
    dimension = min(_positive_int(max_dimension, "max_dimension"), operator.source.size)
    decomposition = arnoldi(
        _coordinate_action(operator),
        coordinates,
        max_dimension=dimension,
        inner=_coordinate_inner(operator),
        orthogonalization="selective",
    )
    projected = decomposition.projected[:-1]
    eigenvalues = jnp.linalg.eigvals(projected)
    radius = jnp.max(jnp.abs(eigenvalues))
    residual = decomposition.residual_norm
    has_subspace = decomposition.effective_dimension > 0
    radius = jnp.where(has_subspace, radius, jnp.asarray(jnp.nan, radius.dtype))
    converged = _whole_krylov_space_certified(
        decomposition.breakdown_status,
        decomposition.effective_dimension,
        operator.source.size,
        coordinates,
        (radius, residual),
        (decomposition.orthogonality_error,),
    )
    return SpectralEstimate(
        value=radius,
        lower_bound=jnp.where(has_subspace, jnp.maximum(radius - residual, 0), radius),
        upper_bound=jnp.where(has_subspace, radius + residual, radius),
        residual_bound=residual,
        converged=converged,
        iterations=decomposition.effective_dimension,
        matvec_count=decomposition.matvec_count,
        adjoint_matvec_count=jnp.asarray(0, dtype=jnp.int32),
        method="arnoldi",
        quantity="spectral-radius",
    )


def estimate_operator_norm(
    operator: AbstractLinearOperator,
    /,
    *,
    max_dimension: int = 32,
    initial: PyTree[Any] | None = None,
) -> SpectralEstimate:
    """Estimate the largest singular value under declared source/target pairings."""
    if not isinstance(operator, AbstractLinearOperator):
        raise TypeError("operator must be an AbstractLinearOperator.")
    if not operator.capabilities.adjoint or operator.batch_shape:
        raise ValueError("Operator-norm estimation requires an unbatched adjoint action.")
    coordinates = _initial_target_coordinates(operator, initial)
    dimension = min(
        _positive_int(max_dimension, "max_dimension"),
        operator.source.size,
        operator.target.size,
    )
    decomposition = golub_kahan(
        _coordinate_action(operator),
        _coordinate_adjoint_action(operator),
        coordinates,
        max_dimension=dimension,
        left_inner=lambda left, right: operator.target.inner(
            operator.target.unflatten(left), operator.target.unflatten(right)
        ),
        right_inner=_coordinate_inner(operator),
    )
    projected = _golub_kahan_matrix(decomposition)
    singular_values = jnp.linalg.svd(projected, compute_uv=False)
    value = singular_values[0]
    full_dimension = min(operator.source.size, operator.target.size)
    residual = jnp.where(
        decomposition.effective_dimension == full_dimension,
        jnp.asarray(0, dtype=decomposition.superdiagonal.dtype),
        jnp.abs(
            decomposition.superdiagonal[
                jnp.maximum(decomposition.effective_dimension - 1, 0)
            ]
        ),
    )
    has_subspace = decomposition.effective_dimension > 0
    value = jnp.where(has_subspace, value, jnp.asarray(jnp.nan, value.dtype))
    converged = _whole_krylov_space_certified(
        decomposition.breakdown_status,
        decomposition.effective_dimension,
        full_dimension,
        coordinates,
        (value, residual),
        (
            decomposition.left_orthogonality_error,
            decomposition.right_orthogonality_error,
        ),
    )
    return SpectralEstimate(
        value=value,
        lower_bound=jnp.where(has_subspace, jnp.maximum(value - residual, 0), value),
        upper_bound=jnp.where(has_subspace, value + residual, value),
        residual_bound=residual,
        converged=converged,
        iterations=decomposition.effective_dimension,
        matvec_count=decomposition.matvec_count,
        adjoint_matvec_count=decomposition.adjoint_matvec_count,
        method="golub-kahan",
        quantity="operator-norm",
    )


def estimate_spectral_bounds(
    operator: AbstractLinearOperator,
    /,
    *,
    max_dimension: int = 32,
    initial: PyTree[Any] | None = None,
) -> SpectralBoundsEstimate:
    """Estimate extremal eigenvalues of a certified self-adjoint operator."""
    _validate_endomorphism(operator)
    if not operator.properties.certifies("self_adjoint"):
        raise ValueError("Spectral bounds require certified self-adjoint structure.")
    coordinates = _initial_coordinates(operator, initial)
    dimension = min(_positive_int(max_dimension, "max_dimension"), operator.source.size)
    decomposition = lanczos(
        _coordinate_action(operator),
        coordinates,
        max_dimension=dimension,
        inner=_coordinate_inner(operator),
        orthogonalization="selective",
    )
    projected = decomposition.projected[:-1]
    active = jnp.arange(dimension) < decomposition.effective_dimension
    scale = jnp.sum(jnp.abs(projected)) + 1
    lower_matrix = projected + jnp.diag(jnp.where(active, 0, scale))
    upper_matrix = projected - jnp.diag(jnp.where(active, 0, scale))
    lower = jnp.min(jnp.linalg.eigvalsh(lower_matrix))
    upper = jnp.max(jnp.linalg.eigvalsh(upper_matrix))
    residual = decomposition.residual_norm
    has_subspace = decomposition.effective_dimension > 0
    lower = jnp.where(has_subspace, lower, jnp.asarray(jnp.nan, lower.dtype))
    upper = jnp.where(has_subspace, upper, jnp.asarray(jnp.nan, upper.dtype))
    converged = _whole_krylov_space_certified(
        decomposition.breakdown_status,
        decomposition.effective_dimension,
        operator.source.size,
        coordinates,
        (lower, upper, residual),
        (decomposition.orthogonality_error,),
    )
    return SpectralBoundsEstimate(
        lower=lower,
        upper=upper,
        lower_residual_bound=residual,
        upper_residual_bound=residual,
        converged=converged,
        iterations=decomposition.effective_dimension,
        matvec_count=decomposition.matvec_count,
        method="lanczos",
    )


def estimate_condition_number(
    operator: AbstractLinearOperator,
    /,
    *,
    max_dimension: int = 32,
    initial: PyTree[Any] | None = None,
) -> SpectralEstimate:
    """Estimate σ_max/σ_min from one Golub–Kahan bidiagonalization."""
    if not isinstance(operator, AbstractLinearOperator):
        raise TypeError("operator must be an AbstractLinearOperator.")
    if not operator.capabilities.adjoint or operator.batch_shape:
        raise ValueError("Condition estimation requires an unbatched adjoint action.")
    coordinates = _initial_target_coordinates(operator, initial)
    dimension = min(
        _positive_int(max_dimension, "max_dimension"),
        operator.source.size,
        operator.target.size,
    )
    decomposition = golub_kahan(
        _coordinate_action(operator),
        _coordinate_adjoint_action(operator),
        coordinates,
        max_dimension=dimension,
        left_inner=lambda left, right: operator.target.inner(
            operator.target.unflatten(left), operator.target.unflatten(right)
        ),
        right_inner=_coordinate_inner(operator),
    )
    projected = _golub_kahan_matrix(decomposition)
    active = jnp.arange(dimension) < decomposition.effective_dimension
    gram = jnp.conj(projected.T) @ projected
    scale = jnp.sum(jnp.abs(gram)) + 1
    regularized_gram = gram + jnp.diag(jnp.where(active, 0, scale))
    squared_singular_values = jnp.linalg.eigvalsh(regularized_gram)
    largest = jnp.sqrt(jnp.maximum(jnp.max(jnp.linalg.eigvalsh(gram)), 0))
    smallest = jnp.sqrt(jnp.maximum(jnp.min(squared_singular_values), 0))
    has_subspace = decomposition.effective_dimension > 0
    condition = jnp.where(
        has_subspace,
        largest / smallest,
        jnp.asarray(jnp.nan, largest.dtype),
    )
    full_dimension = min(operator.source.size, operator.target.size)
    residual = jnp.where(
        decomposition.effective_dimension == full_dimension,
        jnp.asarray(0, dtype=decomposition.superdiagonal.dtype),
        jnp.abs(
            decomposition.superdiagonal[
                jnp.maximum(decomposition.effective_dimension - 1, 0)
            ]
        ),
    )
    converged = _whole_krylov_space_certified(
        decomposition.breakdown_status,
        decomposition.effective_dimension,
        full_dimension,
        coordinates,
        (condition, residual),
        (
            decomposition.left_orthogonality_error,
            decomposition.right_orthogonality_error,
        ),
    ) & (smallest > residual)
    return SpectralEstimate(
        value=condition,
        lower_bound=jnp.where(
            has_subspace,
            jnp.maximum((largest - residual) / (smallest + residual), 1),
            condition,
        ),
        upper_bound=jnp.where(
            has_subspace,
            (largest + residual) / jnp.maximum(smallest - residual, 0),
            condition,
        ),
        residual_bound=residual,
        converged=converged,
        iterations=decomposition.effective_dimension,
        matvec_count=decomposition.matvec_count,
        adjoint_matvec_count=decomposition.adjoint_matvec_count,
        method="golub-kahan",
        quantity="condition-number",
    )


def estimate_numerical_range(
    operator: AbstractLinearOperator,
    /,
    *,
    key: Array,
    num_probes: int = 32,
) -> NumericalRangeEstimate:
    """Random-probe numerical-range envelope under the source pairing."""
    _validate_endomorphism(operator)
    count = _positive_int(num_probes, "num_probes")
    dtype = _coordinate_dtype(operator.source)
    keys = jr.split(key, count)

    def one(probe_key):
        if np.issubdtype(dtype, np.complexfloating):
            real_key, imaginary_key = jr.split(probe_key)
            real_dtype = np.empty((), dtype=dtype).real.dtype
            vector = (
                jr.normal(real_key, (operator.source.size,), dtype=real_dtype)
                + 1j
                * jr.normal(
                    imaginary_key,
                    (operator.source.size,),
                    dtype=real_dtype,
                )
            ).astype(dtype)
        else:
            vector = jr.normal(probe_key, (operator.source.size,), dtype=dtype)
        physical = operator.source.unflatten(vector)
        norm = jnp.sqrt(jnp.real(operator.source.inner(physical, physical)))
        unit = operator.source.unflatten(vector / jnp.where(norm > 0, norm, 1))
        return operator.source.inner(unit, operator.mv(unit))

    samples = jax.vmap(one)(keys)
    center = jnp.mean(samples)
    radius = jnp.max(jnp.abs(samples - center))
    return NumericalRangeEstimate(
        center=center,
        radius=radius,
        real_interval=jnp.asarray([jnp.min(samples.real), jnp.max(samples.real)]),
        imaginary_interval=jnp.asarray([jnp.min(samples.imag), jnp.max(samples.imag)]),
        samples=samples,
        matvec_count=jnp.asarray(count, dtype=jnp.int32),
        method="random Rayleigh probes",
    )


def stochastic_trace(
    operator: AbstractLinearOperator,
    scalar_function: Callable[[Array], Array] = lambda value: value,
    /,
    *,
    key: Array,
    num_probes: int = 16,
    max_dimension: int = 32,
) -> StochasticEstimate:
    """Stochastic Lanczos quadrature for the algebraic trace of ``f(A)``."""
    _validate_endomorphism(operator)
    if not operator.properties.certifies("self_adjoint"):
        raise ValueError("Stochastic Lanczos quadrature requires self-adjoint structure.")
    if not callable(scalar_function):
        raise TypeError("scalar_function must be callable.")
    _require_square_root_pairing(operator.source)
    count = _positive_int(num_probes, "num_probes")
    dimension = min(_positive_int(max_dimension, "max_dimension"), operator.source.size)
    probes = jax.vmap(lambda probe: _whiten_coordinates(operator.source, probe))(
        _rademacher_probes(
            key,
            count,
            operator.source.size,
            _coordinate_dtype(operator.source),
        )
    )

    def one(coordinates):
        decomposition = lanczos(
            _coordinate_action(operator),
            coordinates,
            max_dimension=dimension,
            inner=_coordinate_inner(operator),
            orthogonalization="selective",
        )
        projected = decomposition.projected[:-1]
        quadrature = _effective_lanczos_quadrature(
            projected,
            decomposition.effective_dimension,
            scalar_function,
        )
        norm_squared = jnp.real(
            operator.source.inner(
                operator.source.unflatten(coordinates),
                operator.source.unflatten(coordinates),
            )
        )
        sample = norm_squared * quadrature
        status_successful = _successful_krylov_status(decomposition.breakdown_status)
        orthogonality_ok = _orthogonality_is_acceptable(
            decomposition.orthogonality_error,
            coordinates.dtype,
        )
        finite = (
            jnp.all(jnp.isfinite(sample))
            & jnp.isfinite(decomposition.residual_norm)
            & jnp.isfinite(decomposition.orthogonality_error)
        )
        usable = (
            finite
            & status_successful
            & orthogonality_ok
            & (decomposition.effective_dimension > 0)
        )
        exact_quadrature = (
            decomposition.breakdown_status == int(KrylovBreakdownStatus.HAPPY)
        ) | (decomposition.effective_dimension == operator.source.size)
        probe_converged = usable & exact_quadrature
        probe_status = jnp.where(
            ~finite,
            int(StochasticProbeStatus.NONFINITE),
            jnp.where(
                probe_converged,
                int(StochasticProbeStatus.SUCCESS),
                jnp.where(
                    status_successful & orthogonality_ok,
                    int(StochasticProbeStatus.TRUNCATED),
                    int(StochasticProbeStatus.KRYLOV_FAILURE),
                ),
            ),
        ).astype(jnp.int32)
        relative_residual = decomposition.residual_norm / jnp.maximum(
            jnp.sqrt(norm_squared),
            jnp.asarray(jnp.finfo(coordinates.real.dtype).tiny),
        )
        return (
            sample,
            usable,
            probe_converged,
            probe_status,
            decomposition.breakdown_status,
            decomposition.effective_dimension,
            decomposition.residual_norm,
            relative_residual,
            finite,
            decomposition.matvec_count,
            decomposition.adjoint_matvec_count,
        )

    (
        raw_samples,
        usable,
        probe_converged,
        probe_statuses,
        source_statuses,
        iterations,
        residual_norms,
        relative_residuals,
        probe_finite,
        matvec_counts,
        adjoint_matvec_counts,
    ) = jax.vmap(one)(probes)
    samples, estimate, standard_error = _masked_statistics(raw_samples, usable)
    finite = jnp.all(usable) & jnp.all(jnp.isfinite(estimate))
    converged = finite & jnp.all(probe_converged)
    diagnostics = StochasticProbeDiagnostics(
        source_statuses=source_statuses,
        iterations=iterations,
        residual_norm=residual_norms,
        relative_residual=relative_residuals,
        finite=probe_finite,
        converged=probe_converged,
        matvec_count=matvec_counts,
        adjoint_matvec_count=adjoint_matvec_counts,
    )
    return StochasticEstimate(
        estimate=estimate,
        standard_error=standard_error,
        samples=samples,
        finite=finite,
        converged=converged,
        probe_converged=probe_converged,
        probe_statuses=probe_statuses,
        probe_error_estimates=residual_norms,
        iterations=jnp.sum(iterations, dtype=jnp.int32),
        num_probes=count,
        matvec_count=jnp.sum(matvec_counts, dtype=jnp.int32),
        adjoint_matvec_count=jnp.sum(adjoint_matvec_counts, dtype=jnp.int32),
        diagnostics=diagnostics,
        method="stochastic-lanczos-quadrature",
        quantity="trace",
    )


def stochastic_log_determinant(
    operator: AbstractLinearOperator,
    /,
    *,
    key: Array,
    num_probes: int = 16,
    max_dimension: int = 32,
) -> StochasticEstimate:
    if not operator.properties.certifies("positive_definite"):
        raise ValueError(
            "Log-determinant estimation requires positive-definite evidence."
        )
    result = stochastic_trace(
        operator,
        jnp.log,
        key=key,
        num_probes=num_probes,
        max_dimension=max_dimension,
    )
    return StochasticEstimate(
        estimate=result.estimate,
        standard_error=result.standard_error,
        samples=result.samples,
        finite=result.finite,
        converged=result.converged,
        probe_converged=result.probe_converged,
        probe_statuses=result.probe_statuses,
        probe_error_estimates=result.probe_error_estimates,
        iterations=result.iterations,
        num_probes=result.num_probes,
        matvec_count=result.matvec_count,
        adjoint_matvec_count=result.adjoint_matvec_count,
        diagnostics=result.diagnostics,
        method=result.method,
        quantity="log-determinant",
    )


def estimate_diagonal(
    operator: AbstractLinearOperator,
    /,
    *,
    key: Array,
    num_probes: int = 32,
) -> StochasticEstimate:
    """Hutchinson diagonal estimate in canonical coordinates."""
    if not isinstance(operator, AbstractLinearOperator) or operator.batch_shape:
        raise TypeError("operator must be an unbatched AbstractLinearOperator.")
    if not operator.source.compatible(operator.target):
        raise ValueError("Diagonal estimation requires an endomorphism.")
    count = _positive_int(num_probes, "num_probes")
    probes = _rademacher_probes(
        key,
        count,
        operator.source.size,
        _coordinate_dtype(operator.source),
    )
    images = jax.vmap(_coordinate_action(operator))(probes)
    raw_samples = jnp.conj(probes) * images
    probe_finite = jnp.all(jnp.isfinite(raw_samples), axis=-1)
    samples, estimate, standard_error = _masked_statistics(
        raw_samples,
        probe_finite,
    )
    probe_converged = probe_finite
    probe_statuses = jnp.where(
        probe_finite,
        int(StochasticProbeStatus.SUCCESS),
        int(StochasticProbeStatus.NONFINITE),
    ).astype(jnp.int32)
    zeros = jnp.zeros((count,), dtype=raw_samples.real.dtype)
    int_zeros = jnp.zeros((count,), dtype=jnp.int32)
    matvec_counts = jnp.ones((count,), dtype=jnp.int32)
    diagnostics = StochasticProbeDiagnostics(
        source_statuses=probe_statuses,
        iterations=int_zeros,
        residual_norm=zeros,
        relative_residual=zeros,
        finite=probe_finite,
        converged=probe_converged,
        matvec_count=matvec_counts,
        adjoint_matvec_count=int_zeros,
    )
    finite = jnp.all(probe_finite) & jnp.all(jnp.isfinite(estimate))
    return StochasticEstimate(
        estimate=estimate,
        standard_error=standard_error,
        samples=samples,
        finite=finite,
        converged=finite,
        probe_converged=probe_converged,
        probe_statuses=probe_statuses,
        probe_error_estimates=zeros,
        iterations=jnp.asarray(0, dtype=jnp.int32),
        num_probes=count,
        matvec_count=jnp.asarray(count, dtype=jnp.int32),
        adjoint_matvec_count=jnp.asarray(0, dtype=jnp.int32),
        diagnostics=diagnostics,
        method="hutchinson",
        quantity="diagonal",
    )


def estimate_inverse_diagonal(
    operator: AbstractLinearOperator,
    /,
    *,
    key: Array,
    num_probes: int = 32,
    solve_policy: LinearSolvePolicy | None = None,
) -> StochasticEstimate:
    """Hutchinson estimate of diag(A⁻¹) with solve-status evidence."""
    _validate_endomorphism(operator)
    count = _positive_int(num_probes, "num_probes")
    probes = _rademacher_probes(
        key,
        count,
        operator.source.size,
        _coordinate_dtype(operator.source),
    )

    def one(probe):
        physical = operator.target.unflatten(probe)
        result = solve(LinearSystem(operator), physical, policy=solve_policy)
        sample = jnp.conj(probe) * operator.source.flatten(result.value)
        diagnostics = result.diagnostics
        sample_finite = jnp.all(jnp.isfinite(sample))
        successful = (
            result.successful & diagnostics.converged & diagnostics.finite & sample_finite
        )
        return (
            sample,
            successful,
            result.status,
            diagnostics.iterations,
            diagnostics.residual_norm,
            diagnostics.relative_residual,
            diagnostics.finite & sample_finite,
            diagnostics.matvec_count,
            diagnostics.adjoint_matvec_count,
        )

    (
        raw_samples,
        probe_converged,
        source_statuses,
        iterations,
        residual_norms,
        relative_residuals,
        probe_finite,
        matvec_counts,
        adjoint_matvec_counts,
    ) = jax.vmap(one)(probes)
    samples, estimate, standard_error = _masked_statistics(
        raw_samples,
        probe_converged,
    )
    probe_statuses = jnp.where(
        ~probe_finite,
        int(StochasticProbeStatus.NONFINITE),
        jnp.where(
            probe_converged,
            int(StochasticProbeStatus.SUCCESS),
            int(StochasticProbeStatus.SOLVE_FAILURE),
        ),
    ).astype(jnp.int32)
    diagnostics = StochasticProbeDiagnostics(
        source_statuses=source_statuses,
        iterations=iterations,
        residual_norm=residual_norms,
        relative_residual=relative_residuals,
        finite=probe_finite,
        converged=probe_converged,
        matvec_count=matvec_counts,
        adjoint_matvec_count=adjoint_matvec_counts,
    )
    finite = jnp.all(probe_converged) & jnp.all(jnp.isfinite(estimate))
    converged = finite & jnp.all(probe_converged)
    return StochasticEstimate(
        estimate=estimate,
        standard_error=standard_error,
        samples=samples,
        finite=finite,
        converged=converged,
        probe_converged=probe_converged,
        probe_statuses=probe_statuses,
        probe_error_estimates=residual_norms,
        iterations=jnp.sum(iterations, dtype=jnp.int32),
        num_probes=count,
        matvec_count=jnp.sum(matvec_counts, dtype=jnp.int32),
        adjoint_matvec_count=jnp.sum(adjoint_matvec_counts, dtype=jnp.int32),
        diagnostics=diagnostics,
        method="hutchinson-solve",
        quantity="inverse-diagonal",
    )


def _validate_endomorphism(operator: AbstractLinearOperator, /) -> None:
    if not isinstance(operator, AbstractLinearOperator):
        raise TypeError("operator must be an AbstractLinearOperator.")
    if operator.batch_shape or not operator.source.compatible(operator.target):
        raise ValueError("Spectral estimation requires an unbatched endomorphism.")


def _positive_int(value: Any, name: str, /) -> int:
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive.")
    return result


def _coordinate_action(operator: AbstractLinearOperator, /):
    def action(coordinates):
        return operator.target.flatten(
            operator.mv(operator.source.unflatten(coordinates))
        )

    return action


def _coordinate_adjoint_action(operator: AbstractLinearOperator, /):
    def action(coordinates):
        return operator.source.flatten(
            operator.adjoint_mv(operator.target.unflatten(coordinates))
        )

    return action


def _coordinate_inner(operator: AbstractLinearOperator, /):
    def inner(left, right):
        return operator.source.inner(
            operator.source.unflatten(left), operator.source.unflatten(right)
        )

    return inner


def _successful_krylov_status(status: Array, /) -> Array:
    return (status == int(KrylovBreakdownStatus.NONE)) | (
        status == int(KrylovBreakdownStatus.HAPPY)
    )


def _orthogonality_is_acceptable(error: Array, dtype: Any, /) -> Array:
    real_dtype = np.empty((), dtype=np.dtype(dtype)).real.dtype
    threshold = jnp.asarray(
        100.0 * np.sqrt(np.finfo(real_dtype).eps),
        dtype=real_dtype,
    )
    return jnp.isfinite(error) & (error <= threshold)


def _whole_krylov_space_certified(
    status: Array,
    effective_dimension: Array,
    full_dimension: int,
    coordinates: Array,
    evidence: tuple[Array, ...],
    orthogonality_errors: tuple[Array, ...],
    /,
) -> Array:
    finite = jnp.all(jnp.isfinite(coordinates))
    for value in evidence:
        finite = finite & jnp.all(jnp.isfinite(value))
    orthogonality_ok = jnp.asarray(True)
    for error in orthogonality_errors:
        orthogonality_ok = orthogonality_ok & _orthogonality_is_acceptable(
            error,
            coordinates.dtype,
        )
    return (
        _successful_krylov_status(status)
        & (effective_dimension == full_dimension)
        & finite
        & orthogonality_ok
    )


def _rademacher_probes(
    key: Array,
    count: int,
    size: int,
    dtype: Any,
    /,
) -> Array:
    coordinate_dtype = np.dtype(dtype)
    real_dtype = np.empty((), dtype=coordinate_dtype).real.dtype
    return jr.rademacher(
        key,
        (count, size),
        dtype=real_dtype,
    ).astype(coordinate_dtype)


def _require_square_root_pairing(space: AbstractVectorSpace, /) -> None:
    if isinstance(space, (ArraySpace, PyTreeSpace)):
        if isinstance(space.pairing, (EuclideanPairing, DiagonalPairing)):
            return
    elif isinstance(space, BlockSpace):
        for block in space.spaces:
            _require_square_root_pairing(block)
        return
    elif isinstance(space, DualSpace):
        _require_square_root_pairing(space.primal)
        return
    raise ValueError(
        "Algebraic trace estimation requires a Euclidean or positive diagonal "
        "pairing with a safe square-root Riesz transform."
    )


def _pairing_power(
    space: AbstractVectorSpace,
    vector: PyTree[Any],
    inverse: bool,
    /,
) -> PyTree[Array]:
    if isinstance(space, (ArraySpace, PyTreeSpace)):
        value = space.validate(vector)
        if isinstance(space.pairing, EuclideanPairing):
            return value
        if isinstance(space.pairing, DiagonalPairing):
            if inverse:
                return jax.tree.map(
                    lambda item, weight: item / jnp.sqrt(weight),
                    value,
                    space.pairing.weights,
                )
            return jax.tree.map(
                lambda item, weight: item * jnp.sqrt(weight),
                value,
                space.pairing.weights,
            )
    elif isinstance(space, BlockSpace):
        values = space.validate(vector)
        return tuple(
            _pairing_power(block, value, inverse)
            for block, value in zip(space.spaces, values, strict=True)
        )
    elif isinstance(space, DualSpace):
        return _pairing_power(space.primal, space.validate(vector), not inverse)
    raise ValueError(
        "The declared Riesz pairing does not provide a safe square-root transform."
    )


def _whiten_coordinates(space: AbstractVectorSpace, coordinates: Array, /) -> Array:
    physical = space.unflatten(coordinates)
    whitened = _pairing_power(space, physical, True)
    return space.flatten(whitened).astype(coordinates.dtype)


def _effective_lanczos_quadrature(
    projected: Array,
    effective_dimension: Array,
    scalar_function: Callable[[Array], Array],
    /,
) -> Array:
    value_spec = jax.eval_shape(
        lambda values: jnp.sum(
            jnp.ones_like(values) * jnp.asarray(scalar_function(values))
        ),
        jax.ShapeDtypeStruct((1,), projected.real.dtype),
    )
    if not isinstance(value_spec, jax.ShapeDtypeStruct) or value_spec.shape != ():
        raise TypeError("scalar_function must return scalar-compatible values.")

    def empty(_):
        return jnp.asarray(jnp.nan, dtype=value_spec.dtype)

    def branch(size: int):
        def evaluate(matrix):
            eigenvalues, eigenvectors = jnp.linalg.eigh(matrix[:size, :size])
            weights = jnp.abs(eigenvectors[0]) ** 2
            values = jnp.asarray(scalar_function(eigenvalues))
            return jnp.asarray(jnp.sum(weights * values), dtype=value_spec.dtype)

        return evaluate

    branches = (empty,) + tuple(branch(size) for size in range(1, projected.shape[0] + 1))
    return jax.lax.switch(effective_dimension, branches, projected)


def _masked_statistics(
    raw_samples: Array,
    valid: Array,
    /,
) -> tuple[Array, Array, Array]:
    trailing_axes = (1,) * (raw_samples.ndim - 1)
    mask = valid.reshape((valid.shape[0],) + trailing_axes)
    valid_count = jnp.sum(valid, dtype=jnp.int32)
    safe_count = jnp.maximum(valid_count, 1)
    zero = jnp.zeros((), dtype=raw_samples.dtype)
    nan = jnp.asarray(jnp.nan, dtype=raw_samples.dtype)
    samples = jnp.where(mask, raw_samples, nan)
    estimate = jnp.sum(jnp.where(mask, raw_samples, zero), axis=0) / safe_count
    estimate = jnp.where(valid_count > 0, estimate, nan)
    centered = jnp.where(mask, raw_samples - estimate, zero)
    squared_error = jnp.abs(centered) ** 2
    variance = jnp.sum(squared_error, axis=0) / jnp.maximum(valid_count - 1, 1)
    standard_error = jnp.sqrt(variance / safe_count)
    standard_error = jnp.where(
        valid_count > 1,
        standard_error,
        jnp.where(
            valid_count == 1,
            jnp.zeros_like(standard_error),
            jnp.asarray(jnp.nan, dtype=standard_error.dtype),
        ),
    )
    return samples, estimate, standard_error


def _golub_kahan_matrix(decomposition, /) -> Array:
    dimension = decomposition.diagonal.size
    matrix = jnp.zeros(
        (dimension + 1, dimension),
        dtype=decomposition.diagonal.dtype,
    )
    indices = jnp.arange(dimension)
    matrix = matrix.at[indices, indices].set(decomposition.diagonal)
    return matrix.at[indices + 1, indices].set(decomposition.superdiagonal)


def _initial_coordinates(operator: AbstractLinearOperator, initial: Any, /) -> Array:
    if initial is not None:
        return operator.source.flatten(operator.source.validate(initial))
    return jnp.ones((operator.source.size,), dtype=_coordinate_dtype(operator.source))


def _initial_target_coordinates(
    operator: AbstractLinearOperator,
    initial: Any,
    /,
) -> Array:
    if initial is not None:
        return operator.target.flatten(operator.target.validate(initial))
    return jnp.ones((operator.target.size,), dtype=_coordinate_dtype(operator.target))


__all__ = [
    "NumericalRangeEstimate",
    "SpectralBoundsEstimate",
    "SpectralEstimate",
    "StochasticEstimate",
    "StochasticProbeDiagnostics",
    "StochasticProbeStatus",
    "estimate_diagonal",
    "estimate_inverse_diagonal",
    "estimate_numerical_range",
    "estimate_operator_norm",
    "estimate_spectral_bounds",
    "estimate_spectral_radius",
    "stochastic_log_determinant",
    "stochastic_trace",
]
