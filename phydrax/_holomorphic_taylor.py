#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import jax.numpy as jnp
from jaxtyping import Array

from ._holomorphic_linear import HolomorphicMultiIndexSet, HolomorphicMultiJet


def multi_factorial(value: tuple[int, ...], /) -> int:
    result = 1
    for item in value:
        result *= math.factorial(item)
    return result


def taylor_multiply(
    left: Array,
    right: Array,
    index_set: HolomorphicMultiIndexSet,
    /,
) -> Array:
    """Multiply normalized multivariate Taylor coefficient arrays."""
    if left.shape != right.shape or left.shape[0] != index_set.count:
        raise ValueError(
            "Taylor products require equal index-aligned coefficient arrays."
        )
    positions = {value: position for position, value in enumerate(index_set.indices)}
    coefficients = []
    for target in index_set.indices:
        value = jnp.zeros_like(left[0] * right[0])
        for left_index, left_position in positions.items():
            right_index = tuple(
                target[axis] - left_index[axis]
                for axis in range(index_set.complex_dimension)
            )
            if any(item < 0 for item in right_index) or right_index not in positions:
                continue
            value = value + left[left_position] * right[positions[right_index]]
        coefficients.append(value)
    return jnp.stack(tuple(coefficients))


def taylor_exp(
    coefficients: Array,
    index_set: HolomorphicMultiIndexSet,
    /,
) -> Array:
    """Componentwise exponential of normalized multivariate Taylor coefficients."""
    if not index_set.downward_closed:
        raise ValueError("Taylor exponential requires a downward-closed index set.")
    if coefficients.shape[0] != index_set.count:
        raise ValueError("Taylor exponential coefficients do not match the index set.")
    zero_position = index_set.indices.index((0,) * index_set.complex_dimension)
    delta = coefficients.at[zero_position].set(
        jnp.zeros_like(coefficients[zero_position])
    )
    one = (
        jnp.zeros_like(coefficients)
        .at[zero_position]
        .set(jnp.ones_like(coefficients[zero_position]))
    )
    result = one
    power = one
    for order in range(1, index_set.maximum_total_order + 1):
        power = taylor_multiply(power, delta, index_set)
        result = result + power / math.factorial(order)
    return jnp.exp(coefficients[zero_position])[None, ...] * result


def normalized_coefficients(jet: HolomorphicMultiJet, /) -> Array:
    """Convert a public raw-derivative multijet to normalized Taylor coefficients."""
    values = []
    for multi_index in jet.index_set.indices:
        values.append(jet.derivative(multi_index) / multi_factorial(multi_index))
    return jnp.stack(tuple(values))


def multijet_from_normalized(
    coefficients: Array,
    index_set: HolomorphicMultiIndexSet,
    /,
) -> HolomorphicMultiJet:
    """Convert normalized Taylor coefficients to public raw derivatives."""
    if coefficients.shape[0] != index_set.count:
        raise ValueError("Normalized coefficients do not match the multijet index set.")
    zero = (0,) * index_set.complex_dimension
    value = coefficients[index_set.indices.index(zero)]
    derivatives = tuple(
        coefficients[index_set.indices.index(multi_index)] * multi_factorial(multi_index)
        for multi_index in index_set.nonzero_indices
    )
    return HolomorphicMultiJet(value, derivatives, index_set)


__all__ = [
    "multi_factorial",
    "multijet_from_normalized",
    "normalized_coefficients",
    "taylor_exp",
    "taylor_multiply",
]
