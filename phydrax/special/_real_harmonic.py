#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prepared real Cartesian spherical and regular solid harmonics.

The native real basis removes the Condon--Shortley phase from the complex
orthonormal harmonics ``Y_l^m`` of `sph_harm_y_cart`:

- ``S_{l,m} = sqrt(2) (-1)^m Re Y_l^m`` for ``m > 0``;
- ``S_{l,0} = Y_l^0``;
- ``S_{l,m} = sqrt(2) (-1)^m Im Y_l^{|m|}`` for ``m < 0``.

Components are ordered degree-major and, within a degree, by ascending order
``m = -l, ..., l``. Degree one is therefore proportional to ``(y, z, x)``.
Values are evaluated as Cartesian polynomials through the stable three-term
solid-harmonic recurrence, so no polar-angle chart or pole singularity enters
values or derivatives.
"""

from __future__ import annotations

import math
from functools import lru_cache
from numbers import Integral
from typing import assert_never, final, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._model import register_artifact_value
from .._strict import StrictModule
from ..typing import parse
from ._solid_harmonic import _cartesian_vectors
from ._spherical_harmonic import _normalize_directions, _seed, _step


RealHarmonicNormalization: TypeAlias = Literal[
    "orthonormal", "fully_normalized", "schmidt"
]
RealHarmonicArgument: TypeAlias = Literal["direction", "regular_solid"]


def _static_degree(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a static integer degree.")
    degree = int(value)
    if degree < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return degree


@lru_cache(maxsize=None)
def _real_harmonic_basis_terms(
    degree: int, /
) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    """Exact complex-to-real map, one row per real order ``m = -l..l``.

    Each term ``(complex_index, quarter_turns, half_powers)`` contributes
    ``i**quarter_turns * 2**(-half_powers / 2) * Y_l^M`` with
    ``complex_index = M + l``. The phases follow directly from
    ``Y_l^{-M} = (-1)^M conj(Y_l^M)``.
    """

    rows: list[tuple[tuple[int, int, int], ...]] = []
    for order in range(-degree, degree + 1):
        absolute = abs(order)
        if order == 0:
            rows.append(((degree, 0, 0),))
        elif order > 0:
            rows.append(
                ((degree - absolute, 0, 1), (degree + absolute, (2 * absolute) % 4, 1))
            )
        else:
            rows.append(
                (
                    (degree - absolute, 1, 1),
                    (degree + absolute, (3 + 2 * absolute) % 4, 1),
                )
            )
    return tuple(rows)


def real_harmonic_basis(degree: int, /) -> Array:
    """Unitary ``U`` with real harmonics ``S_l = U @ Y_l``.

    ``Y_l`` stacks `sph_harm_y_cart` over ``m = -l..l``; ``S_l`` is the native
    real basis evaluated by `RealCartesianHarmonics`.
    """

    degree_ = _static_degree(degree, "degree")
    size = 2 * degree_ + 1
    table = np.zeros((size, size), dtype=np.complex128)
    for row, terms in enumerate(_real_harmonic_basis_terms(degree_)):
        for column, quarter_turns, half_powers in terms:
            table[row, column] = (1j**quarter_turns) * 2.0 ** (-0.5 * half_powers)
    return jnp.asarray(table, dtype=jnp.complex128)


def _normalization_scale(
    degree: int, normalization: RealHarmonicNormalization, /
) -> float:
    match normalization:
        case "orthonormal":
            return 1.0
        case "fully_normalized":
            return math.sqrt(4.0 * math.pi)
        case "schmidt":
            return math.sqrt(4.0 * math.pi / (2 * degree + 1))
        case _:
            assert_never(normalization)


@final
class RealCartesianHarmonics(StrictModule):
    """Prepared real harmonics through an admitted maximum degree.

    ``argument="direction"`` evaluates the real spherical harmonics of the
    direction ``v / |v|``. A zero-length or non-finite vector has no direction:
    every component of that vector is NaN, while the computation itself runs on
    a fixed admitted direction so derivatives of other entries stay finite.
    Inactive padding must therefore be replaced by an admitted direction before
    evaluation or masked afterward by its owner.

    ``argument="regular_solid"`` evaluates ``|v|**l S_{l,m}(v / |v|)``, a
    homogeneous polynomial that is smooth everywhere, including at zero.

    Normalizations scale each degree uniformly, preserving rotation
    equivariance: ``"orthonormal"`` has unit integral over the sphere,
    ``"fully_normalized"`` integrates to ``4 pi`` (so ``sum_m S_{l,m}^2 = 2l + 1``
    on unit directions), and ``"schmidt"`` integrates to ``4 pi / (2l + 1)`` (so
    ``sum_m S_{l,m}^2 = 1``).
    """

    maximum_degree: int = eqx.field(static=True)
    normalization: RealHarmonicNormalization = eqx.field(static=True)
    argument: RealHarmonicArgument = eqx.field(static=True)

    def __init__(
        self,
        maximum_degree: int,
        /,
        *,
        normalization: RealHarmonicNormalization = "orthonormal",
        argument: RealHarmonicArgument = "direction",
    ) -> None:
        degree = _static_degree(maximum_degree, "maximum_degree")
        normalization_ = parse(normalization, RealHarmonicNormalization, "normalization")
        argument_ = parse(argument, RealHarmonicArgument, "argument")
        self.maximum_degree = degree
        self.normalization = normalization_
        self.argument = argument_

    @property
    def component_count(self) -> int:
        """Number of packed components, ``(maximum_degree + 1) ** 2``."""
        return (self.maximum_degree + 1) ** 2

    def component_offset(self, degree: int, /) -> int:
        """Packed offset of the ``m = -degree`` component of one degree."""
        degree_ = _static_degree(degree, "degree")
        if degree_ > self.maximum_degree:
            raise ValueError("degree exceeds the prepared maximum degree.")
        return degree_ * degree_

    def __call__(self, vectors: ArrayLike, /) -> Array:
        """Evaluate every component; ``(..., 3) -> (..., component_count)``."""
        values = _cartesian_vectors("RealCartesianHarmonics", vectors)
        match self.argument:
            case "direction":
                unit, valid = _normalize_directions(values)
                harmonics = self._evaluate(unit, jnp.ones_like(unit[..., 0]))
                return jnp.where(
                    valid[..., None], harmonics, jnp.full_like(harmonics, jnp.nan)
                )
            case "regular_solid":
                return self._evaluate(values, jnp.sum(values * values, axis=-1))
            case _:
                assert_never(self.argument)

    def _evaluate(self, points: Array, radius_squared: Array, /) -> Array:
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        limit = self.maximum_degree
        components: dict[tuple[int, int], Array] = {}
        real_azimuth = jnp.ones_like(x)
        imaginary_azimuth = jnp.zeros_like(x)
        root_two = math.sqrt(2.0)
        for order in range(limit + 1):
            if order > 0:
                real_azimuth, imaginary_azimuth = (
                    real_azimuth * x - imaginary_azimuth * y,
                    real_azimuth * y + imaginary_azimuth * x,
                )
            # The real basis multiplies by (-1)^m, cancelling the seed's
            # Condon--Shortley phase; the recurrence is the solid-harmonic one.
            previous = jnp.zeros_like(x)
            current = (-1.0) ** order * _seed(order, x)
            for degree in range(order, limit + 1):
                if degree > order:
                    first, second = _step(degree, order)
                    previous, current = (
                        current,
                        first * (z * current - second * radius_squared * previous),
                    )
                scale = _normalization_scale(degree, self.normalization)
                if order == 0:
                    components[(degree, 0)] = scale * current
                else:
                    weighted = (root_two * scale) * current
                    components[(degree, order)] = weighted * real_azimuth
                    components[(degree, -order)] = weighted * imaginary_azimuth
        return jnp.stack(
            [
                components[(degree, order)]
                for degree in range(limit + 1)
                for order in range(-degree, degree + 1)
            ],
            axis=-1,
        )


register_artifact_value("phydrax.special:RealCartesianHarmonics", RealCartesianHarmonics)


__all__ = [
    "RealCartesianHarmonics",
    "RealHarmonicArgument",
    "RealHarmonicNormalization",
    "real_harmonic_basis",
]
