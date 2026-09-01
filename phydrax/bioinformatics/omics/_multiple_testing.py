#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)


MULTIPLE_TESTING_SUCCESS = 0
MULTIPLE_TESTING_EMPTY_FAMILY = 1
MULTIPLE_TESTING_INVALID_P_VALUE = 2

BH = 0
BY = 1


def _multiple_testing_contract(name: str) -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        name,
        MethodKind.EXACT_MODEL,
        ExecutionKind.FLOATING_POINT_DIRECT,
        DifferentiationKind.NONE,
        OutputKind.RANKING,
        conditioning_statement="Monotone adjusted values use the explicitly declared tested family only.",
        truncation_statement="The full fixed-capacity p-value array is sorted; no tail is truncated.",
        capacity_semantics="Family membership is an explicit Boolean array matching p-value capacity.",
        assumptions=(
            "BH controls FDR under independence or positive regression dependence.",
            "BY controls FDR under arbitrary dependence.",
        ),
        nondifferentiable_outputs=(
            "raw_p_values",
            "adjusted_p_values",
            "tested_family",
            "family_size",
        ),
    )


class MultipleTestingResult(StrictModule):
    """Nondifferentiable multiplicity correction over an explicit family."""

    raw_p_values: Array
    adjusted_p_values: Array
    tested_family: Array
    family_size: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract
    method: int = eqx.field(static=True)


def _adjust(
    p_values: ArrayLike,
    tested_family: ArrayLike,
    /,
    *,
    method: int,
) -> MultipleTestingResult:
    p = jnp.asarray(p_values)
    if not jnp.issubdtype(p.dtype, jnp.inexact):
        p = p.astype(float)
    if p.ndim < 1:
        raise ValueError("p_values must have at least one dimension.")
    tested = jnp.asarray(tested_family, dtype=bool)
    if tested.shape != p.shape:
        raise ValueError("tested_family must have the same shape as p_values.")
    shape = p.shape
    flat_p = p.reshape((-1,))
    flat_tested = tested.reshape((-1,))
    family_size = jnp.sum(flat_tested).astype(jnp.int32)
    legal = jnp.isfinite(flat_p) & (flat_p >= 0.0) & (flat_p <= 1.0)
    invalid_tested = flat_tested & ~legal
    eligible = flat_tested & legal
    capacity = int(flat_p.shape[0])
    safe = jnp.where(eligible, flat_p, jnp.inf)
    order = jnp.argsort(safe, stable=True)
    sorted_p = safe[order]
    ranks = jnp.arange(1, capacity + 1, dtype=flat_p.dtype)
    if method == BY:
        indices = jnp.arange(1, capacity + 1, dtype=flat_p.dtype)
        harmonic = jnp.sum(jnp.where(indices <= family_size, 1.0 / indices, 0.0))
    else:
        harmonic = jnp.asarray(1.0, dtype=flat_p.dtype)
    scaled = sorted_p * family_size.astype(flat_p.dtype) * harmonic / ranks
    scaled = jnp.where(ranks <= family_size, scaled, jnp.inf)
    monotone = jnp.minimum.accumulate(scaled[::-1])[::-1]
    sorted_adjusted = jnp.clip(monotone, 0.0, 1.0)
    adjusted = jnp.full_like(flat_p, jnp.nan).at[order].set(sorted_adjusted)
    adjusted = jnp.where(eligible, adjusted, jnp.nan).reshape(shape)
    valid = (family_size > 0) & ~jnp.any(invalid_tested)
    status = jnp.where(
        jnp.any(invalid_tested),
        MULTIPLE_TESTING_INVALID_P_VALUE,
        jnp.where(
            family_size == 0,
            MULTIPLE_TESTING_EMPTY_FAMILY,
            MULTIPLE_TESTING_SUCCESS,
        ),
    ).astype(jnp.int32)
    name = "benjamini-yekutieli" if method == BY else "benjamini-hochberg"
    return MultipleTestingResult(
        raw_p_values=jax.lax.stop_gradient(p),
        adjusted_p_values=jax.lax.stop_gradient(adjusted),
        tested_family=jax.lax.stop_gradient(tested),
        family_size=jax.lax.stop_gradient(family_size),
        valid=jax.lax.stop_gradient(valid),
        status=jax.lax.stop_gradient(status),
        evidence=jax.lax.stop_gradient(
            jnp.stack((family_size.astype(flat_p.dtype), harmonic))
        ),
        method_contract=_multiple_testing_contract(name),
        method=method,
    )


def benjamini_hochberg(
    p_values: ArrayLike,
    tested_family: ArrayLike,
    /,
) -> MultipleTestingResult:
    """Benjamini-Hochberg correction over the declared tested family."""

    return _adjust(p_values, tested_family, method=BH)


def benjamini_yekutieli(
    p_values: ArrayLike,
    tested_family: ArrayLike,
    /,
) -> MultipleTestingResult:
    """Benjamini-Yekutieli correction over the declared tested family."""

    return _adjust(p_values, tested_family, method=BY)


__all__ = [
    "BH",
    "BY",
    "MULTIPLE_TESTING_EMPTY_FAMILY",
    "MULTIPLE_TESTING_INVALID_P_VALUE",
    "MULTIPLE_TESTING_SUCCESS",
    "MultipleTestingResult",
    "benjamini_hochberg",
    "benjamini_yekutieli",
]
