#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ._core import LocallyPurifiedDensity, MatrixProductState


def mps_inner(left: MatrixProductState, right: MatrixProductState, /) -> Array:
    if left.physical_dimensions != right.physical_dimensions:
        raise ValueError("MPS physical dimensions must match.")
    environment = jnp.ones(
        (1, 1), dtype=jnp.result_type(left.tensors[0], right.tensors[0])
    )
    for left_tensor, right_tensor in zip(left.tensors, right.tensors, strict=True):
        environment = jnp.einsum(
            "ab,api,bpj->ij",
            environment,
            jnp.conj(left_tensor),
            right_tensor,
        )
    return environment.reshape(())


def mps_norm_squared(state: MatrixProductState, /) -> Array:
    return jnp.real(mps_inner(state, state))


def mps_one_site_expectation(
    state: MatrixProductState,
    site: int,
    operator: ArrayLike,
    /,
) -> Array:
    site_ = int(site)
    value = jnp.asarray(operator)
    if value.shape != (
        state.physical_dimensions[site_],
        state.physical_dimensions[site_],
    ):
        raise ValueError("One-site operator shape does not match the MPS.")
    environment = jnp.ones((1, 1), dtype=state.tensors[0].dtype)
    for index, tensor in enumerate(state.tensors):
        if index == site_:
            environment = jnp.einsum(
                "ab,api,pq,bqj->ij",
                environment,
                jnp.conj(tensor),
                value,
                tensor,
            )
        else:
            environment = jnp.einsum(
                "ab,api,bpj->ij", environment, jnp.conj(tensor), tensor
            )
    return environment.reshape(())


def lpdo_raw_trace(state: LocallyPurifiedDensity, /) -> Array:
    environment = jnp.ones((1, 1), dtype=state.tensors[0].dtype)
    for tensor in state.tensors:
        environment = jnp.einsum(
            "ab,apkr,bpks->rs",
            environment,
            jnp.conj(tensor),
            tensor,
        )
    return jnp.real(environment.reshape(()))


def lpdo_one_site_reduced(state: LocallyPurifiedDensity, site: int, /) -> Array:
    site_ = int(site)
    dimension = state.physical_dimensions[site_]
    result = jnp.zeros((dimension, dimension), dtype=state.tensors[0].dtype)
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
    environment = jnp.ones((1, 1), dtype=state.tensors[0].dtype)
    for index, tensor in enumerate(state.tensors):
        if index == site:
            environment = jnp.einsum(
                "ab,apkr,pq,bqks->rs",
                environment,
                jnp.conj(tensor),
                operator,
                tensor,
            )
        else:
            environment = jnp.einsum(
                "ab,apkr,bpks->rs",
                environment,
                jnp.conj(tensor),
                tensor,
            )
    return environment.reshape(())


__all__ = [
    "lpdo_one_site_reduced",
    "lpdo_raw_trace",
    "mps_inner",
    "mps_norm_squared",
    "mps_one_site_expectation",
]
