#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Centered Bernoulli lift polynomials for endpoint-jet seam relations.

On an interval ``[a, b]`` with ``L = b - a`` and ``s = (x - a) / L`` the basis is

```text
q_0(x) = 1,    q_n(x) = L^(n-1) B_n(s) / n!    (n >= 1),
```

with Bernoulli polynomials ``B_n``. Because ``B_n' = n B_(n-1)`` and
``B_n(1) - B_n(0) = delta_(n,1)``, the endpoint jump of ``d^k q_(k+1) / dx^k`` is
one and every other endpoint jump of ``d^k q_n / dx^k`` vanishes. Each ``q_n`` with
``n >= 1`` has zero mean on the interval, which fixes the centered lift gauge.
"""

from __future__ import annotations

import math
from fractions import Fraction
from numbers import Integral

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


def _bernoulli_numbers(count: int, /) -> tuple[Fraction, ...]:
    numbers = [Fraction(1)]
    for m in range(1, count):
        numbers.append(
            -sum(Fraction(math.comb(m + 1, k)) * numbers[k] for k in range(m))
            / Fraction(m + 1)
        )
    return tuple(numbers)


def _bernoulli_polynomial(
    degree: int, numbers: tuple[Fraction, ...], /
) -> list[Fraction]:
    # Ascending monomial coefficients of B_n(s) = sum_k C(n, k) B_(n-k) s^k.
    return [
        Fraction(math.comb(degree, k)) * numbers[degree - k] for k in range(degree + 1)
    ]


def _bernoulli_value(
    degree: int, side: int, numbers: tuple[Fraction, ...], /
) -> Fraction:
    # B_n(0) = B_n and B_n(1) = B_n + delta_(n,1).
    return numbers[degree] + (Fraction(1) if side == 1 and degree == 1 else Fraction(0))


class EndpointJetBasis(StrictModule, NonTrainableState):
    """Centered Bernoulli lift basis ``q_0, ..., q_(K+1)`` for jets ``<= K``.

    Orders must be non-boolean integers. Preparation refuses interval scales or
    endpoint jets that overflow or vanish by underflow in float64.
    """

    coefficients: Array
    scales: Array
    max_order: int = eqx.field(static=True)
    basis_size: int = eqx.field(static=True)
    length: float = eqx.field(static=True)
    basis_id: str = eqx.field(static=True)

    def __init__(
        self, max_order: int, length: float, /, *, maximum_order: int = 16
    ) -> None:
        if isinstance(max_order, bool) or not isinstance(max_order, Integral):
            raise TypeError("max_order must be an integer.")
        if isinstance(maximum_order, bool) or not isinstance(maximum_order, Integral):
            raise TypeError("maximum_order must be an integer.")
        order = int(max_order)
        limit = int(maximum_order)
        if limit < 0:
            raise ValueError("maximum_order must be nonnegative.")
        if order < 0:
            raise ValueError("max_order must be nonnegative.")
        if order > limit:
            raise ValueError(
                f"Endpoint jet order {order} exceeds the prepared maximum {limit}."
            )
        length_ = float(length)
        if not math.isfinite(length_) or length_ <= 0.0:
            raise ValueError("Endpoint interval length must be finite and positive.")
        size = order + 2
        numbers = _bernoulli_numbers(size)
        coefficients = np.zeros((size, size), dtype=np.float64)
        coefficients[0, 0] = 1.0
        for degree in range(1, size):
            polynomial = _bernoulli_polynomial(degree, numbers)
            factorial = math.factorial(degree)
            for power, value in enumerate(polynomial):
                coefficients[degree, power] = float(value / factorial)
        with np.errstate(over="ignore", under="ignore"):
            scales = np.concatenate(
                (
                    np.ones((1,), dtype=np.float64),
                    np.power(length_, np.arange(size - 1, dtype=np.int64)),
                )
            )
        if np.any(~np.isfinite(scales)) or np.any(scales == 0.0):
            raise ValueError("Endpoint interval scales are not representable in float64.")
        self.coefficients = jnp.asarray(coefficients)
        self.scales = jnp.asarray(scales)
        self.max_order = order
        self.basis_size = size
        self.length = length_
        self.endpoint_jets(length_)
        self.basis_id = canonical_fingerprint(
            {
                "kind": "centered-bernoulli-endpoint-basis",
                "max_order": order,
                "length": length_,
                "coefficients": array_tree_fingerprint(coefficients),
            }
        )

    def endpoint_jets(self, length: float, /) -> np.ndarray:
        """Return exact host jets ``d^k q_n / dx^k`` at ``s = 0`` and ``s = 1``.

        The result has shape ``(2, max_order + 1, basis_size)``.
        """
        length_ = float(length)
        if length_ != self.length:
            raise ValueError("Endpoint jets must use the prepared interval length.")
        numbers = _bernoulli_numbers(self.basis_size)
        table = np.zeros((2, self.max_order + 1, self.basis_size), dtype=np.float64)
        for side in (0, 1):
            table[side, 0, 0] = 1.0
            for order in range(self.max_order + 1):
                for degree in range(max(order, 1), self.basis_size):
                    reduced = degree - order
                    value = _bernoulli_value(reduced, side, numbers) / math.factorial(
                        reduced
                    )
                    if value == 0:
                        continue
                    with np.errstate(over="ignore", under="ignore", divide="ignore"):
                        entry = float(value) * np.power(length_, degree - 1 - order)
                    if not np.isfinite(entry) or entry == 0.0:
                        raise ValueError(
                            "Endpoint interval jets are not representable in float64."
                        )
                    table[side, order, degree] = entry
        return table

    def values(self, normalized: ArrayLike, /) -> Array:
        """Evaluate every basis function at normalized coordinates ``s``.

        Horner's recurrence keeps every derivative finite at ``s = 0``.
        Integer coordinates adopt the prepared floating dtype; inexact
        coordinates retain their explicitly supplied dtype.
        Unrepresentable coefficient or scale casts are refused lazily during
        evaluation, including traced evaluation; preparation certifies float64.
        """
        s = jnp.asarray(normalized)
        if jnp.issubdtype(s.dtype, jnp.integer):
            s = s.astype(self.coefficients.dtype)
        elif not jnp.issubdtype(s.dtype, jnp.inexact):
            raise TypeError("Normalized coordinates must be numeric and non-boolean.")
        coefficients = self.coefficients.astype(s.dtype)
        coefficients = eqx.error_if(
            coefficients,
            jnp.any(~jnp.isfinite(coefficients))
            | jnp.any((self.coefficients != 0) & (coefficients == 0)),
            "Endpoint coefficients are not representable in the coordinate dtype.",
        )
        scales = self.scales.astype(s.dtype)
        scales = eqx.error_if(
            scales,
            jnp.any(~jnp.isfinite(scales) | (scales == 0)),
            "Endpoint scales are not representable in the coordinate dtype.",
        )
        broadcast_shape = (1,) * s.ndim + (self.basis_size,)
        result = jnp.broadcast_to(coefficients[:, -1], s.shape + (self.basis_size,))
        for power in range(self.basis_size - 2, -1, -1):
            result = result * s[..., None] + coefficients[:, power].reshape(
                broadcast_shape
            )
        return result * scales.reshape(broadcast_shape)


__all__ = ["EndpointJetBasis"]
