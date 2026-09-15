#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._spaces import _coordinate_dtype
from ._spectral import _orthogonality_is_acceptable, _successful_krylov_status
from .krylov import KrylovBreakdownStatus, PreparedKrylovProjection


class LanczosResolventStatus(IntEnum):
    """Per-shift status for a Lanczos resolvent quadratic form."""

    SUCCESS = 0
    TRUNCATED = 1
    SINGULAR = 2
    NONFINITE = 3
    KRYLOV_FAILURE = 4


class LanczosResolventDiagnostics(StrictModule):
    """Numerical and source-projection evidence for a resolvent evaluation."""

    finite: Array
    projection_exact: Array
    effective_dimension: Array
    projection_residual_norm: Array
    projection_orthogonality_error: Array
    projection_breakdown_status: Array
    boundary_coupling: Array
    truncation_indicator: Array
    relative_truncation_indicator: Array
    indicator_available: Array


class LanczosResolventProvenance(StrictModule):
    """Identity and convention evidence for a Lanczos resolvent result."""

    numeric_version: Array
    plan_id: str = eqx.field(static=True)
    projection_id: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    operator_fingerprint: str = eqx.field(static=True)
    initial_fingerprint: str = eqx.field(static=True)
    method: str = eqx.field(static=True)
    convention: str = eqx.field(static=True)
    termination: str = eqx.field(static=True)


class LanczosResolventResult(StrictModule):
    """Lanczos approximation to ``<v, (z I - A)^-1 v>`` over one shift family."""

    value: Array
    shifts: Array
    terminal_resolvent: Array
    status: Array
    diagnostics: LanczosResolventDiagnostics
    provenance: LanczosResolventProvenance

    @property
    def successful(self) -> Array:
        return self.status == int(LanczosResolventStatus.SUCCESS)

    @property
    def all_successful(self) -> Array:
        return jnp.all(self.successful)


def lanczos_resolvent_form(
    projection: PreparedKrylovProjection,
    shifts: ArrayLike,
    /,
    *,
    terminal_resolvent: ArrayLike | None = None,
) -> LanczosResolventResult:
    """Evaluate ``<v, (z I - A)^-1 v>`` from one prepared Lanczos projection.

    A zero terminal resolvent gives the finite projected Jacobi fraction.  An explicit
    terminal resolvent closes the unresolved chain through the retained boundary
    coupling, but does not certify convergence to the original operator.
    """
    _validate_projection(projection)
    shift_values, terminal_values = _validated_inputs(
        projection,
        shifts,
        terminal_resolvent,
    )
    decomposition = projection.decomposition
    projected = decomposition.projected
    capacity = projection.capacity
    indices = jnp.arange(capacity)
    diagonal = jnp.real(projected[indices, indices]).astype(shift_values.dtype)
    next_coupling_squared = jnp.square(jnp.real(projected[indices + 1, indices])).astype(
        shift_values.dtype
    )
    norm_squared = _initial_norm_squared(projection).astype(shift_values.dtype)

    projection_usable = _projection_is_usable(
        projection,
        diagonal,
        next_coupling_squared,
        norm_squared,
    )
    projection_exact = projection_usable & (
        (decomposition.breakdown_status == int(KrylovBreakdownStatus.HAPPY))
        | (decomposition.effective_dimension == projection.operator.source.size)
    )
    raw_value, singular, arithmetic_finite = _evaluate_jacobi_fraction(
        shift_values,
        diagonal,
        next_coupling_squared,
        norm_squared,
        decomposition.effective_dimension,
        terminal_values,
    )
    finite = projection_usable & arithmetic_finite & jnp.isfinite(raw_value)
    value = jnp.where(finite, raw_value, _nan_like(raw_value))
    status = _resolvent_status(
        projection_usable,
        projection_exact,
        finite,
        singular,
        shift_values,
        terminal_values,
    )

    if terminal_resolvent is None:
        previous_value, previous_singular, previous_finite = _evaluate_jacobi_fraction(
            shift_values,
            diagonal,
            next_coupling_squared,
            norm_squared,
            jnp.maximum(decomposition.effective_dimension - 1, 0),
            jnp.zeros_like(terminal_values),
        )
        indicator_available = (
            projection_usable
            & (decomposition.effective_dimension > 1)
            & finite
            & previous_finite
            & ~singular
            & ~previous_singular
            & jnp.isfinite(previous_value)
        )
        indicator = jnp.abs(raw_value - previous_value)
        tiny = jnp.asarray(
            jnp.finfo(raw_value.real.dtype).tiny,
            dtype=raw_value.real.dtype,
        )
        relative_indicator = indicator / jnp.maximum(jnp.abs(raw_value), tiny)
        indicator = jnp.where(
            indicator_available,
            indicator,
            jnp.asarray(jnp.nan, dtype=indicator.dtype),
        )
        relative_indicator = jnp.where(
            indicator_available,
            relative_indicator,
            jnp.asarray(jnp.nan, dtype=relative_indicator.dtype),
        )
        termination = "finite-zero-tail"
    else:
        indicator_available = jnp.zeros(shift_values.shape, dtype=jnp.bool_)
        indicator_dtype = shift_values.real.dtype
        indicator = jnp.full(shift_values.shape, jnp.nan, dtype=indicator_dtype)
        relative_indicator = jnp.full(
            shift_values.shape,
            jnp.nan,
            dtype=indicator_dtype,
        )
        termination = "explicit-terminal-resolvent"

    boundary_index = jnp.maximum(decomposition.effective_dimension - 1, 0)
    boundary_coupling = jnp.real(
        projected[decomposition.effective_dimension, boundary_index]
    )
    diagnostics = LanczosResolventDiagnostics(
        finite=finite,
        projection_exact=projection_exact,
        effective_dimension=decomposition.effective_dimension,
        projection_residual_norm=decomposition.residual_norm,
        projection_orthogonality_error=decomposition.orthogonality_error,
        projection_breakdown_status=decomposition.breakdown_status,
        boundary_coupling=boundary_coupling,
        truncation_indicator=indicator,
        relative_truncation_indicator=relative_indicator,
        indicator_available=indicator_available,
    )
    provenance = LanczosResolventProvenance(
        numeric_version=projection.numeric_version,
        plan_id=projection.plan.plan_id,
        projection_id=projection.projection_id,
        operator_id=projection.operator.operator_id,
        operator_fingerprint=projection.operator_fingerprint,
        initial_fingerprint=projection.initial_fingerprint,
        method="lanczos-jacobi-backward-recurrence",
        convention="initial-adjoint-(shift-operator)-inverse-initial",
        termination=termination,
    )
    return LanczosResolventResult(
        value=value,
        shifts=shift_values,
        terminal_resolvent=terminal_values,
        status=status,
        diagnostics=diagnostics,
        provenance=provenance,
    )


def _evaluate_jacobi_fraction(
    shifts: Array,
    diagonal: Array,
    next_coupling_squared: Array,
    norm_squared: Array,
    effective_dimension: Array,
    terminal_resolvent: Array,
    /,
) -> tuple[Array, Array, Array]:
    initial_finite = jnp.isfinite(shifts) & jnp.isfinite(terminal_resolvent)
    state = (
        terminal_resolvent,
        jnp.zeros(shifts.shape, dtype=jnp.bool_),
        initial_finite,
    )
    capacity = diagonal.size

    def step(offset, current):
        value, singular, healthy = current
        index = capacity - 1 - offset
        active = index < effective_dimension
        denominator = shifts - diagonal[index] - next_coupling_squared[index] * value
        denominator_finite = jnp.isfinite(denominator)
        denominator_zero = denominator == 0
        new_singular = active & healthy & denominator_finite & denominator_zero
        valid = active & healthy & denominator_finite & ~denominator_zero
        safe_denominator = jnp.where(valid, denominator, jnp.ones_like(denominator))
        candidate = jnp.reciprocal(safe_denominator)
        candidate_finite = jnp.isfinite(candidate)
        next_healthy = healthy & (~active | (valid & candidate_finite))
        next_value = jnp.where(
            active,
            jnp.where(valid & candidate_finite, candidate, jnp.zeros_like(candidate)),
            value,
        )
        return next_value, singular | new_singular, next_healthy

    fraction, singular, healthy = jax.lax.fori_loop(0, capacity, step, state)
    result = norm_squared * fraction
    return result, singular, healthy & jnp.isfinite(result)


def _projection_is_usable(
    projection: PreparedKrylovProjection,
    diagonal: Array,
    next_coupling_squared: Array,
    norm_squared: Array,
    /,
) -> Array:
    decomposition = projection.decomposition
    return (
        _successful_krylov_status(decomposition.breakdown_status)
        & (decomposition.effective_dimension > 0)
        & _orthogonality_is_acceptable(
            decomposition.orthogonality_error,
            _coordinate_dtype(projection.operator.source),
        )
        & jnp.isfinite(decomposition.residual_norm)
        & jnp.all(jnp.isfinite(projection.initial_coordinates))
        & jnp.all(jnp.isfinite(diagonal))
        & jnp.all(jnp.isfinite(next_coupling_squared))
        & jnp.isfinite(norm_squared)
        & (jnp.real(norm_squared) > 0)
    )


def _resolvent_status(
    projection_usable: Array,
    projection_exact: Array,
    finite: Array,
    singular: Array,
    shifts: Array,
    terminal_resolvent: Array,
    /,
) -> Array:
    status = jnp.full(
        shifts.shape,
        int(LanczosResolventStatus.TRUNCATED),
        dtype=jnp.int32,
    )
    status = jnp.where(
        projection_exact,
        int(LanczosResolventStatus.SUCCESS),
        status,
    )
    status = jnp.where(
        ~finite | ~jnp.isfinite(shifts) | ~jnp.isfinite(terminal_resolvent),
        int(LanczosResolventStatus.NONFINITE),
        status,
    )
    status = jnp.where(
        singular,
        int(LanczosResolventStatus.SINGULAR),
        status,
    )
    return jnp.where(
        projection_usable,
        status,
        jnp.asarray(int(LanczosResolventStatus.KRYLOV_FAILURE), dtype=jnp.int32),
    )


def _validated_inputs(
    projection: PreparedKrylovProjection,
    shifts: ArrayLike,
    terminal_resolvent: ArrayLike | None,
    /,
) -> tuple[Array, Array]:
    values = jnp.asarray(shifts)
    if values.ndim not in (0, 1) or (values.ndim == 1 and values.size < 1):
        raise ValueError("shifts must be a scalar or one nonempty rank-one array.")
    if not jnp.issubdtype(values.dtype, jnp.number) or jnp.issubdtype(
        values.dtype, jnp.bool_
    ):
        raise TypeError("shifts must contain real or complex numbers.")

    if terminal_resolvent is None:
        terminal = jnp.zeros(values.shape, dtype=values.dtype)
    else:
        terminal = jnp.asarray(terminal_resolvent)
        if terminal.shape not in ((), values.shape):
            raise ValueError(
                "terminal_resolvent must be scalar or have exactly the shifts shape."
            )
        if not jnp.issubdtype(terminal.dtype, jnp.number) or jnp.issubdtype(
            terminal.dtype, jnp.bool_
        ):
            raise TypeError("terminal_resolvent must contain real or complex numbers.")

    dtype = jnp.result_type(
        _coordinate_dtype(projection.operator.source),
        values.dtype,
        terminal.dtype,
    )
    values = values.astype(dtype)
    terminal = jnp.broadcast_to(terminal.astype(dtype), values.shape)
    return values, terminal


def _validate_projection(projection: PreparedKrylovProjection, /) -> None:
    if not isinstance(projection, PreparedKrylovProjection):
        raise TypeError("projection must be a PreparedKrylovProjection.")
    if projection.method != "lanczos":
        raise ValueError("Lanczos resolvent forms require a Lanczos projection.")
    if not projection.operator.properties.certifies("self_adjoint"):
        raise ValueError(
            "Lanczos resolvent forms require certified self-adjoint structure."
        )


def _initial_norm_squared(projection: PreparedKrylovProjection, /) -> Array:
    initial = projection.operator.source.unflatten(projection.initial_coordinates)
    return jnp.real(projection.operator.source.inner(initial, initial))


def _nan_like(value: Array, /) -> Array:
    return jnp.full(value.shape, jnp.nan, dtype=value.dtype)


__all__ = [
    "LanczosResolventDiagnostics",
    "LanczosResolventProvenance",
    "LanczosResolventResult",
    "LanczosResolventStatus",
    "lanczos_resolvent_form",
]
