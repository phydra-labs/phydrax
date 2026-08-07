#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from numbers import Real
from typing import Any

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax.domain import Domain, DomainFunction


def coerce_hbar(hbar: ArrayLike, /) -> Array:
    """Return a real scalar Planck constant without forcing JAX concretization."""
    value = jnp.asarray(hbar)
    if value.shape != ():
        raise ValueError(f"hbar must be a scalar, got shape {value.shape}.")
    if jnp.iscomplexobj(value):
        raise TypeError("hbar must be real.")
    if isinstance(hbar, Real) and (isinstance(hbar, bool) or hbar <= 0):
        raise ValueError("hbar must be positive.")
    return value


def validate_matrix_value(value: Any, /, *, role: str) -> Array:
    array = jnp.asarray(value)
    if array.ndim != 2 or int(array.shape[0]) != int(array.shape[1]):
        raise ValueError(
            f"{role} must be a square matrix with shape (n, n), got {array.shape}."
        )
    return array


def validate_vector_value(value: Any, /, *, role: str) -> Array:
    array = jnp.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{role} must be a vector with shape (n,), got {array.shape}.")
    return array


def _structurally_contains(
    container: Domain,
    candidate: Domain,
    /,
) -> bool:
    return all(
        label in container.labels
        and container.coordinate(label).compatible(candidate.coordinate(label))
        for label in candidate.labels
    )


def join_function_arguments(
    *functions: DomainFunction,
) -> tuple[
    Domain,
    tuple[str, ...],
    tuple[DomainFunction, ...],
    tuple[tuple[int, ...], ...],
]:
    if not functions:
        raise ValueError("At least one DomainFunction is required.")
    joined = functions[0].domain
    for function in functions[1:]:
        if _structurally_contains(joined, function.domain):
            continue
        if _structurally_contains(function.domain, joined):
            joined = function.domain
            continue
        joined = joined.join(function.domain)
    promoted = tuple(function.promote(joined) for function in functions)
    deps = tuple(
        label
        for label in joined.labels
        if any(label in function.deps for function in promoted)
    )
    positions = {label: index for index, label in enumerate(deps)}
    argument_positions = tuple(
        tuple(positions[label] for label in function.deps) for function in promoted
    )
    return joined, deps, promoted, argument_positions


__all__ = [
    "coerce_hbar",
    "join_function_arguments",
    "validate_matrix_value",
    "validate_vector_value",
]
