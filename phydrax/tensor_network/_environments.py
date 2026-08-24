#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ._core import LocallyPurifiedDensity, MatrixProductState


def mps_inner(left: MatrixProductState, right: MatrixProductState, /) -> Array:
    if left.physical_dimensions != right.physical_dimensions:
        raise ValueError("MPS physical dimensions must match.")
    if left.precision.policy_id != right.precision.policy_id:
        raise ValueError("MPS precision policies must match.")
    left_tensors = left.precision.contraction(left.tensors)
    right_tensors = right.precision.contraction(right.tensors)
    environment = jnp.ones(
        (1, 1), dtype=jnp.result_type(left_tensors[0], right_tensors[0])
    )
    for left_tensor, right_tensor in zip(left_tensors, right_tensors, strict=True):
        environment = left.precision.accumulation(
            oe.contract(
                "ab,api,bpj->ij",
                environment,
                jnp.conj(left_tensor),
                right_tensor,
            )
        )
    return left.precision.output(environment.reshape(()))


def mps_norm_squared(state: MatrixProductState, /) -> Array:
    return jnp.real(mps_inner(state, state))


def mps_one_site_expectation(
    state: MatrixProductState,
    site: int,
    operator: ArrayLike,
    /,
) -> Array:
    site_ = int(site)
    if not 0 <= site_ < state.site_count:
        raise ValueError("One-site expectation site is outside the MPS.")
    value = state.precision.contraction(jnp.asarray(operator))
    if value.shape != (
        state.physical_dimensions[site_],
        state.physical_dimensions[site_],
    ):
        raise ValueError("One-site operator shape does not match the MPS.")
    tensors = state.precision.contraction(state.tensors)
    environment = jnp.ones((1, 1), dtype=tensors[0].dtype)
    for index, tensor in enumerate(tensors):
        if index == site_:
            updated = oe.contract(
                "ab,api,pq,bqj->ij",
                environment,
                jnp.conj(tensor),
                value,
                tensor,
            )
        else:
            updated = oe.contract(
                "ab,api,bpj->ij", environment, jnp.conj(tensor), tensor
            )
        environment = state.precision.accumulation(updated)
    return state.precision.output(environment.reshape(()))


def lpdo_raw_trace(state: LocallyPurifiedDensity, /) -> Array:
    tensors = state.precision.contraction(state.tensors)
    environment = jnp.ones((1, 1), dtype=tensors[0].dtype)
    for tensor in tensors:
        environment = state.precision.accumulation(
            oe.contract(
                "ab,apkr,bpks->rs",
                environment,
                jnp.conj(tensor),
                tensor,
            )
        )
    return state.precision.decision(jnp.real(environment.reshape(())))


def lpdo_one_site_reduced(state: LocallyPurifiedDensity, site: int, /) -> Array:
    site_ = int(site)
    if not 0 <= site_ < state.site_count:
        raise ValueError("One-site reduced-density site is outside the LPDO.")
    dimension = state.physical_dimensions[site_]
    output_exemplar = state.precision.output(
        jnp.zeros((dimension, dimension), dtype=state.tensors[0].dtype)
    )
    result = jnp.zeros((dimension, dimension), dtype=output_exemplar.dtype)
    for row in range(dimension):
        for column in range(dimension):
            operator = (
                jnp.zeros((dimension, dimension), dtype=result.dtype)
                .at[column, row]
                .set(1.0)
            )
            value = _lpdo_one_site_expectation(state, site_, operator)
            result = result.at[row, column].set(value)
    return result


def _lpdo_one_site_expectation(
    state: LocallyPurifiedDensity, site: int, operator: Array, /
) -> Array:
    tensors = state.precision.contraction(state.tensors)
    operator_ = state.precision.contraction(operator)
    environment = jnp.ones((1, 1), dtype=tensors[0].dtype)
    for index, tensor in enumerate(tensors):
        if index == site:
            updated = oe.contract(
                "ab,apkr,pq,bqks->rs",
                environment,
                jnp.conj(tensor),
                operator_,
                tensor,
            )
        else:
            updated = oe.contract(
                "ab,apkr,bpks->rs",
                environment,
                jnp.conj(tensor),
                tensor,
            )
        environment = state.precision.accumulation(updated)
    return state.precision.output(environment.reshape(()))


__all__ = [
    "lpdo_one_site_reduced",
    "lpdo_raw_trace",
    "mps_inner",
    "mps_norm_squared",
    "mps_one_site_expectation",
]
