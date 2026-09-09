#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""AAD, implicit-system and symmetric-bump sensitivity evidence."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...linalg import (
    DenseLinearOperator,
    FactorizationPolicy,
    factorize,
    LinearSystem,
    solve,
)
from ..core._evidence import FinanceEvidenceBinding
from ._types import ValuationResult


GreekMethod = Literal["aad", "implicit", "central-bump"]


class GreekRequest(StrictModule):
    parameter_names: tuple[str, ...] = eqx.field(static=True)
    second_order_names: tuple[str, ...] = eqx.field(static=True)
    bump_sizes: Array | None
    parameter_count: int = eqx.field(static=True)

    def __init__(
        self,
        parameter_names: Sequence[str],
        /,
        *,
        second_order_names: Sequence[str] = (),
        bump_sizes: ArrayLike | None = None,
    ):
        names = tuple(parameter_names)
        if not names or any(not isinstance(value, str) or not value for value in names):
            raise ValueError("parameter_names must contain non-empty strings.")
        if len(set(names)) != len(names):
            raise ValueError("parameter_names must be unique.")
        second = tuple(second_order_names)
        if len(set(second)) != len(second) or any(value not in names for value in second):
            raise ValueError(
                "second_order_names must be a unique subset of parameter_names."
            )
        bumps = None if bump_sizes is None else jnp.asarray(bump_sizes, dtype=float)
        if bumps is not None:
            if bumps.shape != (len(names),):
                raise ValueError("bump_sizes must contain one value per parameter.")
            bumps = eqx.error_if(
                bumps,
                jnp.any(~jnp.isfinite(bumps)) | jnp.any(bumps <= 0.0),
                "bump_sizes must be finite and positive.",
            )
        self.parameter_names, self.second_order_names = names, second
        self.bump_sizes, self.parameter_count = bumps, len(names)


class GreekEvidence(StrictModule):
    finite: Array
    base_value: Array
    residual_norm: Array | None
    condition_estimate: Array | None
    truncation_estimate: Array | None
    binding: FinanceEvidenceBinding | None
    method: GreekMethod = eqx.field(static=True)
    valuation_id: str = eqx.field(static=True)


class GreekResult(StrictModule):
    first_order: Array
    second_order_diagonal: Array
    request: GreekRequest
    evidence: GreekEvidence

    def __init__(
        self,
        first_order: ArrayLike,
        second_order_diagonal: ArrayLike,
        request: GreekRequest,
        evidence: GreekEvidence,
        /,
    ):
        if not isinstance(request, GreekRequest) or not isinstance(
            evidence, GreekEvidence
        ):
            raise TypeError("request/evidence must be Greek records.")
        first = jnp.asarray(first_order)
        second = jnp.asarray(second_order_diagonal, dtype=first.dtype)
        if first.shape != (request.parameter_count,) or second.shape != first.shape:
            raise ValueError(
                "Greek arrays must contain one entry per requested parameter."
            )
        self.first_order, self.second_order_diagonal = first, second
        self.request, self.evidence = request, evidence

    def first(self, name: str, /) -> Array:
        if name not in self.request.parameter_names:
            raise KeyError(name)
        return self.first_order[self.request.parameter_names.index(name)]

    def second(self, name: str, /) -> Array:
        if name not in self.request.second_order_names:
            raise KeyError(f"second-order Greek {name!r} was not requested.")
        return self.second_order_diagonal[self.request.parameter_names.index(name)]


def _scalar_value(value) -> Array:
    resolved = value.value if isinstance(value, ValuationResult) else jnp.asarray(value)
    if resolved.shape != ():
        raise ValueError("Greek evaluation requires a scalar valuation output.")
    return resolved


def evaluate_aad_greeks(
    valuation: StrictModule,
    parameters: ArrayLike,
    request: GreekRequest,
    /,
    *,
    valuation_id: str,
    evidence_binding: FinanceEvidenceBinding | None = None,
) -> GreekResult:
    """Differentiate a content-addressable StrictModule valuation functional."""

    if not isinstance(valuation, StrictModule) or not callable(valuation):
        raise TypeError(
            "valuation must be a callable StrictModule; opaque callables are not accepted."
        )
    if not isinstance(request, GreekRequest):
        raise TypeError("request must be GreekRequest.")
    if not isinstance(valuation_id, str) or not valuation_id:
        raise ValueError("valuation_id must be a non-empty semantic identifier.")
    parameter_array = jnp.asarray(parameters, dtype=float)
    if parameter_array.shape != (request.parameter_count,):
        raise ValueError("parameters must contain one value per requested parameter.")
    function = lambda candidate: _scalar_value(valuation(candidate))
    base, gradient = jax.value_and_grad(function)(parameter_array)
    diagonal = jnp.diag(jax.hessian(function)(parameter_array))
    requested = jnp.asarray(
        tuple(name in request.second_order_names for name in request.parameter_names)
    )
    diagonal = jnp.where(requested, diagonal, jnp.nan)
    finite = (
        jnp.isfinite(base)
        & jnp.all(jnp.isfinite(gradient))
        & jnp.all(jnp.isfinite(jnp.where(requested, diagonal, 0.0)))
    )
    gradient = eqx.error_if(
        gradient, ~finite, "AAD Greek evaluation produced non-finite derivatives."
    )
    evidence = GreekEvidence(
        finite, base, None, None, None, evidence_binding, "aad", valuation_id
    )
    return GreekResult(gradient, diagonal, request, evidence)


def evaluate_implicit_greeks(
    state_jacobian: ArrayLike,
    residual_parameter_jacobian: ArrayLike,
    value_state_gradient: ArrayLike,
    direct_parameter_gradient: ArrayLike,
    request: GreekRequest,
    /,
    *,
    valuation_id: str,
    evidence_binding: FinanceEvidenceBinding | None = None,
) -> GreekResult:
    """Apply the implicit-function theorem to a converged calibration/root state."""

    if not isinstance(request, GreekRequest):
        raise TypeError("request must be GreekRequest.")
    jacobian = jnp.asarray(state_jacobian, dtype=float)
    residual_parameter = jnp.asarray(residual_parameter_jacobian, dtype=float)
    value_state = jnp.asarray(value_state_gradient, dtype=float)
    direct = jnp.asarray(direct_parameter_gradient, dtype=float)
    if jacobian.ndim != 2 or jacobian.shape[0] != jacobian.shape[1]:
        raise ValueError("state_jacobian must be square.")
    state_count = jacobian.shape[0]
    if (
        residual_parameter.shape != (state_count, request.parameter_count)
        or value_state.shape != (state_count,)
        or direct.shape != (request.parameter_count,)
    ):
        raise ValueError("implicit Greek derivative blocks have incompatible shapes.")
    factorization = factorize(DenseLinearOperator(jacobian), FactorizationPolicy("svd"))
    rank = factorization.rank()
    singular_values = factorization.singular_values()
    nonsingular = rank == state_count
    response = solve(LinearSystem(DenseLinearOperator(jacobian)), -residual_parameter)
    first = direct + value_state @ response.value
    residual = jacobian @ response.value + residual_parameter
    residual_norm = jnp.sqrt(jnp.sum(residual**2))
    condition = singular_values[0] / singular_values[-1]
    finite = (
        nonsingular
        & jnp.all(response.successful)
        & jnp.all(jnp.isfinite(first))
        & jnp.isfinite(condition)
    )
    first = eqx.error_if(
        first, ~finite, "implicit Greek system is singular or non-finite."
    )
    second = jnp.full_like(first, jnp.nan)
    evidence = GreekEvidence(
        finite,
        jnp.asarray(jnp.nan),
        residual_norm,
        condition,
        None,
        evidence_binding,
        "implicit",
        valuation_id,
    )
    return GreekResult(first, second, request, evidence)


def evaluate_bump_greeks(
    base_value: ArrayLike,
    upward_values: ArrayLike,
    downward_values: ArrayLike,
    request: GreekRequest,
    /,
    *,
    valuation_id: str,
    evidence_binding: FinanceEvidenceBinding | None = None,
) -> GreekResult:
    """Central first/second differences from explicitly replayable bumped values."""

    if not isinstance(request, GreekRequest) or request.bump_sizes is None:
        raise ValueError("bump Greeks require GreekRequest.bump_sizes.")
    base = jnp.asarray(base_value, dtype=float)
    upward = jnp.asarray(upward_values, dtype=float)
    downward = jnp.asarray(downward_values, dtype=float)
    if (
        base.shape != ()
        or upward.shape != (request.parameter_count,)
        or downward.shape != upward.shape
    ):
        raise ValueError(
            "base/upward/downward valuation shapes are incompatible with the request."
        )
    first = (upward - downward) / (2.0 * request.bump_sizes)
    raw_second = (upward - 2.0 * base + downward) / request.bump_sizes**2
    requested = jnp.asarray(
        tuple(name in request.second_order_names for name in request.parameter_names)
    )
    second = jnp.where(requested, raw_second, jnp.nan)
    asymmetry = jnp.abs((upward - base) - (base - downward))
    finite = (
        jnp.isfinite(base)
        & jnp.all(jnp.isfinite(first))
        & jnp.all(jnp.isfinite(jnp.where(requested, second, 0.0)))
    )
    first = eqx.error_if(
        first, ~finite, "bump Greek inputs or derivatives are non-finite."
    )
    evidence = GreekEvidence(
        finite,
        base,
        None,
        None,
        asymmetry,
        evidence_binding,
        "central-bump",
        valuation_id,
    )
    return GreekResult(first, second, request, evidence)


__all__ = [
    "GreekEvidence",
    "GreekRequest",
    "GreekResult",
    "evaluate_aad_greeks",
    "evaluate_bump_greeks",
    "evaluate_implicit_greeks",
]
