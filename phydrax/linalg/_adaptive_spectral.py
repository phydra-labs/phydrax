#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Callable
from statistics import NormalDist
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._strict import StrictModule
from ._operators import AbstractLinearOperator, estimate_operator_action_cost
from ._spaces import _coordinate_dtype
from ._spectral import (
    _coordinate_action,
    _coordinate_inner,
    _effective_lanczos_quadrature,
    _masked_statistics,
    _orthogonality_is_acceptable,
    _rademacher_probes,
    _require_square_root_pairing,
    _successful_krylov_status,
    _validate_endomorphism,
    _whiten_coordinates,
    StochasticProbeDiagnostics,
    StochasticProbeStatus,
)
from .krylov import KrylovBreakdownStatus, lanczos


class AdaptiveStochasticPolicy(StrictModule):
    """Fixed-capacity batched stopping policy for stochastic estimators."""

    min_probes: int = eqx.field(static=True)
    max_probes: int = eqx.field(static=True)
    batch_size: int = eqx.field(static=True)
    max_dimension: int = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    confidence_level: float = eqx.field(static=True)
    confidence_multiplier: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        min_probes: int = 8,
        max_probes: int = 128,
        batch_size: int = 4,
        max_dimension: int = 32,
        relative_tolerance: float = 1e-2,
        absolute_tolerance: float = 1e-6,
        confidence_level: float = 0.95,
    ):
        minimum = int(min_probes)
        maximum = int(max_probes)
        batch = int(batch_size)
        dimension = int(max_dimension)
        if minimum < 2 or maximum < minimum:
            raise ValueError("Probe bounds must satisfy 2 <= min_probes <= max_probes.")
        if batch < 1 or minimum % batch != 0 or maximum % batch != 0:
            raise ValueError(
                "batch_size must be positive and divide min_probes and max_probes."
            )
        if dimension < 1:
            raise ValueError("max_dimension must be positive.")
        relative = float(relative_tolerance)
        absolute = float(absolute_tolerance)
        confidence = float(confidence_level)
        if (
            not math.isfinite(relative)
            or not math.isfinite(absolute)
            or relative < 0.0
            or absolute < 0.0
        ):
            raise ValueError(
                "Adaptive stochastic tolerances must be finite and non-negative."
            )
        if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
            raise ValueError("confidence_level must lie strictly between zero and one.")
        if relative == 0.0 and absolute == 0.0:
            raise ValueError(
                "At least one adaptive stochastic tolerance must be positive."
            )
        self.min_probes = minimum
        self.max_probes = maximum
        self.batch_size = batch
        self.max_dimension = dimension
        self.relative_tolerance = relative
        self.absolute_tolerance = absolute
        self.confidence_level = confidence
        self.confidence_multiplier = NormalDist().inv_cdf(0.5 + 0.5 * confidence)


class AdaptiveStochasticCostEstimate(StrictModule):
    """Static best/worst-case work and fixed-capacity storage estimate."""

    method: str = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    first_stopping_matvec_budget: int = eqx.field(static=True)
    maximum_matvec_budget: int = eqx.field(static=True)
    retained_storage_bytes: int = eqx.field(static=True)
    batch_workspace_bytes: int = eqx.field(static=True)
    exact: bool = eqx.field(static=True)


class AdaptiveStochasticEstimate(StrictModule):
    """Adaptive estimate with statistical and projection-error evidence."""

    estimate: Array
    standard_error: Array
    confidence_radius: Array
    numerical_error_estimate: Array
    total_error_estimate: Array
    tolerance: Array
    samples: Array
    finite: Array
    converged: Array
    stopped_early: Array
    probe_converged: Array
    probe_statuses: Array
    probe_error_estimates: Array
    num_probes: Array
    iterations: Array
    matvec_count: Array
    adjoint_matvec_count: Array
    diagnostics: StochasticProbeDiagnostics
    policy: AdaptiveStochasticPolicy = eqx.field(static=True)
    cost: AdaptiveStochasticCostEstimate = eqx.field(static=True)
    method: str = eqx.field(static=True)
    quantity: str = eqx.field(static=True)
    confidence_model: str = eqx.field(static=True)


def adaptive_stochastic_trace(
    operator: AbstractLinearOperator,
    scalar_function: Callable[[Array], Array] = lambda value: value,
    /,
    *,
    key: Array,
    policy: AdaptiveStochasticPolicy | None = None,
) -> AdaptiveStochasticEstimate:
    """Adapt probe batches for stochastic Lanczos quadrature of ``trace(f(A))``."""
    return _adaptive_stochastic_trace(
        operator,
        scalar_function,
        key=key,
        policy=policy,
        quantity="trace",
    )


def _adaptive_stochastic_trace(
    operator: AbstractLinearOperator,
    scalar_function: Callable[[Array], Array],
    /,
    *,
    key: Array,
    policy: AdaptiveStochasticPolicy | None,
    quantity: str,
) -> AdaptiveStochasticEstimate:
    _validate_endomorphism(operator)
    if not operator.properties.certifies("self_adjoint"):
        raise ValueError("Adaptive SLQ requires certified self-adjoint structure.")
    if not callable(scalar_function):
        raise TypeError("scalar_function must be callable.")
    _require_square_root_pairing(operator.source)
    selected = AdaptiveStochasticPolicy() if policy is None else policy
    if not isinstance(selected, AdaptiveStochasticPolicy):
        raise TypeError("policy must be an AdaptiveStochasticPolicy or None.")
    dimension = min(selected.max_dimension, operator.source.size)
    coordinate_dtype = _coordinate_dtype(operator.source)
    probes = jax.vmap(lambda probe: _whiten_coordinates(operator.source, probe))(
        _rademacher_probes(
            key,
            selected.max_probes,
            operator.source.size,
            coordinate_dtype,
        )
    )
    one = _trace_probe(operator, scalar_function, dimension, selected)
    specification = jax.eval_shape(
        one,
        jax.ShapeDtypeStruct((operator.source.size,), coordinate_dtype),
    )
    sample_spec = specification[0]
    if not np.issubdtype(sample_spec.dtype, np.inexact):
        raise TypeError("scalar_function must produce a real or complex scalar.")
    real_dtype = np.empty((), dtype=sample_spec.dtype).real.dtype
    maximum = selected.max_probes
    nan_sample = jnp.asarray(jnp.nan, dtype=sample_spec.dtype)
    nan_real = jnp.asarray(jnp.nan, dtype=real_dtype)
    samples = jnp.full((maximum,), nan_sample, dtype=sample_spec.dtype)
    booleans = jnp.zeros((maximum,), dtype=bool)
    statuses = jnp.full(
        (maximum,),
        int(StochasticProbeStatus.NOT_EVALUATED),
        dtype=jnp.int32,
    )
    source_statuses = jnp.full(
        (maximum,),
        -1,
        dtype=jnp.int32,
    )
    integers = jnp.zeros((maximum,), dtype=jnp.int32)
    reals = jnp.full((maximum,), nan_real, dtype=real_dtype)
    state = (
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(False),
        samples,
        booleans,
        booleans,
        statuses,
        source_statuses,
        integers,
        reals,
        reals,
        reals,
        booleans,
        integers,
        integers,
    )

    def condition(values):
        count, done, *_ = values
        return (~done) & (count < maximum)

    def body(values):
        (
            count,
            _,
            samples_,
            usable_,
            probe_converged_,
            statuses_,
            source_statuses_,
            iterations_,
            errors_,
            residuals_,
            relative_residuals_,
            finite_,
            matvecs_,
            adjoint_matvecs_,
        ) = values
        batch = jax.lax.dynamic_slice_in_dim(
            probes,
            count,
            selected.batch_size,
            axis=0,
        )
        outputs = jax.vmap(one)(batch)
        next_count = count + selected.batch_size
        updated = tuple(
            jax.lax.dynamic_update_slice_in_dim(target, value, count, axis=0)
            for target, value in zip(
                (
                    samples_,
                    usable_,
                    probe_converged_,
                    statuses_,
                    source_statuses_,
                    iterations_,
                    errors_,
                    residuals_,
                    relative_residuals_,
                    finite_,
                    matvecs_,
                    adjoint_matvecs_,
                ),
                outputs,
                strict=True,
            )
        )
        active = jnp.arange(maximum) < next_count
        estimate, standard_error, numerical_error, finite = _adaptive_statistics(
            updated[0], updated[1], updated[6], active
        )
        confidence_radius = selected.confidence_multiplier * standard_error
        total_error = confidence_radius + numerical_error
        tolerance = selected.absolute_tolerance + selected.relative_tolerance * jnp.abs(
            estimate
        )
        converged = (
            (next_count >= selected.min_probes)
            & finite
            & jnp.all(total_error <= tolerance)
        )
        return (next_count, converged, *updated)

    (
        count,
        converged,
        raw_samples,
        usable,
        probe_converged,
        probe_statuses,
        source_statuses,
        iterations,
        probe_errors,
        residual_norms,
        relative_residuals,
        probe_finite,
        matvec_counts,
        adjoint_matvec_counts,
    ) = jax.lax.while_loop(condition, body, state)
    active = jnp.arange(maximum) < count
    estimate, standard_error, numerical_error, finite = _adaptive_statistics(
        raw_samples, usable, probe_errors, active
    )
    confidence_radius = selected.confidence_multiplier * standard_error
    total_error = confidence_radius + numerical_error
    tolerance = selected.absolute_tolerance + selected.relative_tolerance * jnp.abs(
        estimate
    )
    active_usable = active & usable
    relative_residuals = jnp.where(
        active_usable, relative_residuals, jnp.asarray(jnp.nan, dtype=real_dtype)
    )
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
    cost = _adaptive_cost(operator, selected, dimension, sample_spec.dtype)
    return AdaptiveStochasticEstimate(
        estimate=estimate,
        standard_error=standard_error,
        confidence_radius=confidence_radius,
        numerical_error_estimate=numerical_error,
        total_error_estimate=total_error,
        tolerance=tolerance,
        samples=jnp.where(active, raw_samples, nan_sample),
        finite=finite,
        converged=converged,
        stopped_early=count < maximum,
        probe_converged=probe_converged,
        probe_statuses=probe_statuses,
        probe_error_estimates=probe_errors,
        num_probes=count,
        iterations=jnp.sum(jnp.where(active, iterations, 0), dtype=jnp.int32),
        matvec_count=jnp.sum(jnp.where(active, matvec_counts, 0), dtype=jnp.int32),
        adjoint_matvec_count=jnp.sum(
            jnp.where(active, adjoint_matvec_counts, 0), dtype=jnp.int32
        ),
        diagnostics=diagnostics,
        policy=selected,
        cost=cost,
        method="adaptive-stochastic-lanczos-quadrature",
        quantity=quantity,
        confidence_model="two-sided normal approximation plus successive-SLQ difference",
    )


def adaptive_stochastic_log_determinant(
    operator: AbstractLinearOperator,
    /,
    *,
    key: Array,
    policy: AdaptiveStochasticPolicy | None = None,
) -> AdaptiveStochasticEstimate:
    """Adaptively estimate ``log(det(A))`` for a certified positive operator."""
    if not isinstance(operator, AbstractLinearOperator):
        raise TypeError("operator must be an AbstractLinearOperator.")
    if not operator.properties.certifies("positive_definite"):
        raise ValueError(
            "Adaptive log-determinant estimation requires positive-definite evidence."
        )
    return _adaptive_stochastic_trace(
        operator,
        jnp.log,
        key=key,
        policy=policy,
        quantity="log-determinant",
    )


def _trace_probe(
    operator: AbstractLinearOperator,
    scalar_function: Callable[[Array], Array],
    dimension: int,
    policy: AdaptiveStochasticPolicy,
    /,
):
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
        previous_dimension = jnp.maximum(decomposition.effective_dimension - 1, 0)
        previous = _effective_lanczos_quadrature(
            projected,
            previous_dimension,
            scalar_function,
        )
        physical = operator.source.unflatten(coordinates)
        norm_squared = jnp.real(operator.source.inner(physical, physical))
        sample = norm_squared * quadrature
        exact = (decomposition.breakdown_status == int(KrylovBreakdownStatus.HAPPY)) | (
            decomposition.effective_dimension == operator.source.size
        )
        difference = jnp.abs(norm_squared * (quadrature - previous))
        infinity = jnp.asarray(jnp.inf, dtype=difference.dtype)
        numerical_error = jnp.where(
            exact,
            jnp.asarray(0.0, dtype=difference.dtype),
            jnp.where(jnp.isfinite(difference), difference, infinity),
        )
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
        probe_tolerance = policy.absolute_tolerance + policy.relative_tolerance * jnp.abs(
            sample
        )
        probe_converged = usable & (numerical_error <= probe_tolerance)
        status = jnp.where(
            ~finite,
            int(StochasticProbeStatus.NONFINITE),
            jnp.where(
                ~usable,
                int(StochasticProbeStatus.KRYLOV_FAILURE),
                jnp.where(
                    exact,
                    int(StochasticProbeStatus.SUCCESS),
                    int(StochasticProbeStatus.TRUNCATED),
                ),
            ),
        ).astype(jnp.int32)
        return (
            sample,
            usable,
            probe_converged,
            status,
            decomposition.breakdown_status,
            decomposition.effective_dimension,
            numerical_error,
            decomposition.residual_norm,
            decomposition.residual_norm
            / jnp.maximum(
                jnp.sqrt(norm_squared),
                jnp.asarray(jnp.finfo(coordinates.real.dtype).tiny),
            ),
            finite,
            decomposition.matvec_count,
            decomposition.adjoint_matvec_count,
        )

    return one


def _adaptive_statistics(
    raw_samples: Array,
    usable: Array,
    probe_errors: Array,
    active: Array,
    /,
) -> tuple[Array, Array, Array, Array]:
    valid = active & usable
    _, estimate, standard_error = _masked_statistics(raw_samples, valid)
    valid_count = jnp.sum(valid, dtype=jnp.int32)
    safe_count = jnp.maximum(valid_count, 1)
    numerical_error = jnp.sum(jnp.where(valid, probe_errors, 0)) / safe_count
    numerical_error = jnp.where(valid_count > 0, numerical_error, jnp.inf)
    finite = (
        jnp.all(jnp.where(active, usable, True))
        & (valid_count >= 2)
        & jnp.all(jnp.isfinite(estimate))
        & jnp.all(jnp.isfinite(standard_error))
    )
    return estimate, standard_error, numerical_error, finite


def _adaptive_cost(
    operator: AbstractLinearOperator,
    policy: AdaptiveStochasticPolicy,
    dimension: int,
    sample_dtype: Any,
    /,
) -> AdaptiveStochasticCostEstimate:
    coordinate_itemsize = _coordinate_dtype(operator.source).itemsize
    sample_itemsize = np.dtype(sample_dtype).itemsize
    real_itemsize = np.empty((), dtype=np.dtype(sample_dtype)).real.dtype.itemsize
    retained_per_probe = (
        sample_itemsize
        + 3 * real_itemsize
        + 2 * jnp.dtype(bool).itemsize
        + 5 * jnp.dtype(jnp.int32).itemsize
    )
    retained = policy.max_probes * retained_per_probe
    krylov_entries = (dimension + 2) * operator.source.size + (dimension + 1) * dimension
    action = estimate_operator_action_cost(operator)
    probe_bank = policy.max_probes * operator.source.size * coordinate_itemsize
    workspace = (
        probe_bank
        + retained
        + policy.batch_size
        * (krylov_entries * coordinate_itemsize + action.apply_workspace_bytes_per_rhs)
    )
    return AdaptiveStochasticCostEstimate(
        method="adaptive-stochastic-lanczos-quadrature",
        dimension=dimension,
        first_stopping_matvec_budget=policy.min_probes * dimension,
        maximum_matvec_budget=policy.max_probes * dimension,
        retained_storage_bytes=retained,
        batch_workspace_bytes=workspace,
        exact=action.exact,
    )


__all__ = [
    "AdaptiveStochasticCostEstimate",
    "AdaptiveStochasticEstimate",
    "AdaptiveStochasticPolicy",
    "adaptive_stochastic_log_determinant",
    "adaptive_stochastic_trace",
]
