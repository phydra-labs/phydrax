#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import comb
from operator import index
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule


def dense_monomial_count(variable_count: int, max_degree: int, /) -> int:
    """Return the number of monomials in ``variable_count`` variables up to a degree."""

    if isinstance(variable_count, bool) or isinstance(max_degree, bool):
        raise TypeError("variable_count and max_degree must be integers.")
    variables = index(variable_count)
    degree = index(max_degree)
    if variables < 1:
        raise ValueError("variable_count must be positive.")
    if degree < 0:
        raise ValueError("max_degree must be nonnegative.")
    return comb(variables + degree, degree)


def _fixed_degree_exponents(
    variable_count: int, total_degree: int, prefix: tuple[int, ...] = ()
) -> tuple[tuple[int, ...], ...]:
    if variable_count == 1:
        return (prefix + (total_degree,),)
    values: list[tuple[int, ...]] = []
    for first in range(total_degree, -1, -1):
        values.extend(
            _fixed_degree_exponents(
                variable_count - 1,
                total_degree - first,
                prefix + (first,),
            )
        )
    return tuple(values)


def _dense_exponents(variable_count: int, max_degree: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        exponent
        for degree in range(max_degree + 1)
        for exponent in _fixed_degree_exponents(variable_count, degree)
    )


class DenseMonomialBasis(StrictModule):
    """Graded dense monomial basis with deterministic total-degree ordering."""

    exponents: Array
    variable_count: int = eqx.field(static=True)
    max_degree: int = eqx.field(static=True)
    size: int = eqx.field(static=True)
    exponent_tuples: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    basis_id: str = eqx.field(static=True)

    def __init__(self, variable_count: int, max_degree: int, /):
        size = dense_monomial_count(variable_count, max_degree)
        variables = index(variable_count)
        degree = index(max_degree)
        exponents = _dense_exponents(variables, degree)
        if len(exponents) != size:
            raise RuntimeError("Dense monomial enumeration has an inconsistent size.")
        self.exponents = jnp.asarray(exponents, dtype=jnp.int32)
        self.variable_count = variables
        self.max_degree = degree
        self.size = size
        self.exponent_tuples = exponents
        self.basis_id = canonical_fingerprint(
            {
                "kind": "dense-monomial-basis",
                "variable_count": variables,
                "max_degree": degree,
                "ordering": "graded-total-degree-descending-lexicographic",
            }
        )

    def index(self, exponent: Any, /) -> int:
        values = tuple(index(value) for value in exponent)
        if len(values) != self.variable_count:
            raise ValueError(
                f"A monomial exponent must have length {self.variable_count}."
            )
        if any(value < 0 for value in values) or sum(values) > self.max_degree:
            raise ValueError("The exponent is outside this monomial basis.")
        return self.exponent_tuples.index(values)

    def evaluate(self, points: ArrayLike, /) -> Array:
        """Evaluate every basis monomial at points ending in the variable axis."""

        values = jnp.asarray(points)
        if values.ndim < 1 or values.shape[-1] != self.variable_count:
            raise ValueError(
                f"points must end in shape ({self.variable_count},); got {values.shape}."
            )
        if jnp.issubdtype(values.dtype, jnp.complexfloating):
            raise TypeError(
                "Dense real monomial evaluation does not accept complex points."
            )
        powers = values[..., None, :] ** self.exponents
        return jnp.prod(powers, axis=-1)


class DenseMomentBasis(StrictModule):
    """Dense truncated moments and their order-``r`` moment matrix indexing."""

    moments: DenseMonomialBasis
    matrix_monomials: DenseMonomialBasis
    entry_indices: Array
    variable_count: int = eqx.field(static=True)
    order: int = eqx.field(static=True)
    moment_count: int = eqx.field(static=True)
    matrix_size: int = eqx.field(static=True)
    basis_id: str = eqx.field(static=True)

    def __init__(self, variable_count: int, order: int, /):
        if isinstance(order, bool):
            raise TypeError("order must be an integer.")
        relaxation_order = index(order)
        if relaxation_order < 0:
            raise ValueError("order must be nonnegative.")
        moments = DenseMonomialBasis(variable_count, 2 * relaxation_order)
        matrix_monomials = DenseMonomialBasis(variable_count, relaxation_order)
        lookup = {
            exponent: position
            for position, exponent in enumerate(moments.exponent_tuples)
        }
        entries = tuple(
            tuple(
                lookup[
                    tuple(left + right for left, right in zip(row, column, strict=True))
                ]
                for column in matrix_monomials.exponent_tuples
            )
            for row in matrix_monomials.exponent_tuples
        )
        self.moments = moments
        self.matrix_monomials = matrix_monomials
        self.entry_indices = jnp.asarray(entries, dtype=jnp.int32)
        self.variable_count = moments.variable_count
        self.order = relaxation_order
        self.moment_count = moments.size
        self.matrix_size = matrix_monomials.size
        self.basis_id = canonical_fingerprint(
            {
                "kind": "dense-moment-basis",
                "monomials": moments.basis_id,
                "matrix_monomials": matrix_monomials.basis_id,
            }
        )

    def matrix(self, moments: ArrayLike, /) -> Array:
        values = jnp.asarray(moments)
        if values.ndim < 1 or values.shape[-1] != self.moment_count:
            raise ValueError(
                f"moments must end in shape ({self.moment_count},); got {values.shape}."
            )
        return values[..., self.entry_indices]


class DenseLocalizingBasis(StrictModule):
    """Dense localizing matrix indexing for one polynomial support row."""

    moments: DenseMomentBasis
    multipliers: DenseMonomialBasis
    polynomial_exponents: Array
    entry_term_indices: Array
    polynomial_degree: int = eqx.field(static=True)
    localizing_order: int = eqx.field(static=True)
    matrix_size: int = eqx.field(static=True)
    term_count: int = eqx.field(static=True)
    basis_id: str = eqx.field(static=True)

    def __init__(
        self,
        moments: DenseMomentBasis,
        polynomial_exponents: ArrayLike,
        /,
    ):
        if not isinstance(moments, DenseMomentBasis):
            raise TypeError("moments must be a DenseMomentBasis.")
        exponents = np.asarray(polynomial_exponents)
        expected_tail = (moments.variable_count,)
        if exponents.ndim != 2 or exponents.shape[1:] != expected_tail:
            raise ValueError(
                f"polynomial_exponents must have shape (terms, {moments.variable_count})."
            )
        if not np.issubdtype(exponents.dtype, np.integer):
            raise TypeError("polynomial_exponents must be integer-valued.")
        if np.any(exponents < 0):
            raise ValueError("polynomial exponents must be nonnegative.")
        degree = int(np.max(np.sum(exponents, axis=1))) if exponents.shape[0] else 0
        localizing_order = moments.order - (degree + 1) // 2
        if localizing_order < 0:
            raise ValueError(
                "The relaxation order is too small for this localizing polynomial."
            )
        multipliers = DenseMonomialBasis(moments.variable_count, localizing_order)
        lookup = {
            exponent: position
            for position, exponent in enumerate(moments.moments.exponent_tuples)
        }
        exponent_rows = tuple(tuple(row) for row in exponents)
        entries = tuple(
            tuple(
                tuple(
                    lookup[
                        tuple(
                            left + right + polynomial
                            for left, right, polynomial in zip(
                                row, column, term, strict=True
                            )
                        )
                    ]
                    for term in exponent_rows
                )
                for column in multipliers.exponent_tuples
            )
            for row in multipliers.exponent_tuples
        )
        self.moments = moments
        self.multipliers = multipliers
        self.polynomial_exponents = jnp.asarray(exponents, dtype=jnp.int32)
        self.entry_term_indices = jnp.asarray(
            entries,
            dtype=jnp.int32,
        ).reshape((multipliers.size, multipliers.size, exponents.shape[0]))
        self.polynomial_degree = degree
        self.localizing_order = localizing_order
        self.matrix_size = multipliers.size
        self.term_count = exponents.shape[0]
        self.basis_id = canonical_fingerprint(
            {
                "kind": "dense-localizing-basis",
                "moments": moments.basis_id,
                "polynomial_exponents": [list(row) for row in exponent_rows],
                "localizing_order": localizing_order,
            }
        )

    def matrix(self, moments: ArrayLike, coefficients: ArrayLike, /) -> Array:
        values = jnp.asarray(moments)
        weights = jnp.asarray(coefficients)
        if values.ndim < 1 or values.shape[-1] != self.moments.moment_count:
            raise ValueError(
                f"moments must end in shape ({self.moments.moment_count},); got {values.shape}."
            )
        if weights.shape != (self.term_count,):
            raise ValueError(f"coefficients must have shape ({self.term_count},).")
        if self.term_count == 0:
            return jnp.zeros(
                values.shape[:-1] + (self.matrix_size, self.matrix_size),
                dtype=jnp.result_type(values.dtype, weights.dtype),
            )
        gathered = values[..., self.entry_term_indices]
        return jnp.sum(gathered * weights, axis=-1)


__all__ = [
    "DenseLocalizingBasis",
    "DenseMomentBasis",
    "DenseMonomialBasis",
    "dense_monomial_count",
]
