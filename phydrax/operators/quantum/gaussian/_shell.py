#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Immutable general-contraction Cartesian and real-spherical Gaussian shells."""

from __future__ import annotations

from enum import StrEnum
from math import factorial

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState


class GaussianShellRepresentation(StrEnum):
    CARTESIAN = "cartesian"
    REAL_SPHERICAL = "real-spherical"


def odd_double_factorial(value: int, /) -> int:
    integer = int(value)
    if integer <= 0:
        return 1
    return factorial(integer) // (2 ** (integer // 2) * factorial(integer // 2))


def cartesian_angular_exponents(
    angular_momentum: int, /
) -> tuple[tuple[int, int, int], ...]:
    angular = int(angular_momentum)
    if angular < 0:
        raise ValueError("Gaussian angular momentum must be non-negative.")
    return tuple(
        (x, y, angular - x - y)
        for x in range(angular, -1, -1)
        for y in range(angular - x, -1, -1)
    )


def cartesian_primitive_normalization(
    exponent: ArrayLike,
    angular_exponents: tuple[int, int, int],
    /,
) -> Array:
    alpha = jnp.asarray(exponent)
    x, y, z = angular_exponents
    denominator = (
        odd_double_factorial(2 * x - 1)
        * odd_double_factorial(2 * y - 1)
        * odd_double_factorial(2 * z - 1)
    )
    return (2.0 * alpha / jnp.pi) ** 0.75 * jnp.sqrt(
        (4.0 * alpha) ** (x + y + z) / denominator
    )


class GaussianShellPlan(StrictModule, NonTrainableState):
    """One center, angular momentum, and generally contracted primitive shell."""

    center_particle_id: int = eqx.field(static=True)
    angular_momentum: int = eqx.field(static=True)
    exponents: Array
    coefficients: Array
    primitive_mask: Array
    representation: GaussianShellRepresentation = eqx.field(static=True)
    role: str = eqx.field(static=True)
    shell_id: str = eqx.field(static=True)

    def __init__(
        self,
        center_particle_id: int,
        angular_momentum: int,
        exponents: ArrayLike,
        coefficients: ArrayLike,
        /,
        *,
        primitive_mask: ArrayLike | None = None,
        representation: GaussianShellRepresentation = GaussianShellRepresentation.CARTESIAN,
        role: str = "orbital",
    ):
        center = int(center_particle_id)
        angular = int(angular_momentum)
        exponent = np.asarray(exponents, dtype=np.float64).reshape((-1,))
        coefficient = np.asarray(coefficients, dtype=np.float64)
        if coefficient.ndim == 1:
            coefficient = coefficient[None, :]
        if angular < 0 or angular > 12:
            raise ValueError("Gaussian shell angular momentum must lie in [0, 12].")
        if (
            exponent.size == 0
            or coefficient.ndim != 2
            or coefficient.shape[1] != exponent.size
        ):
            raise ValueError(
                "Gaussian shell contractions must align with primitive exponents."
            )
        mask = (
            np.ones(exponent.shape, dtype=np.bool_)
            if primitive_mask is None
            else np.asarray(primitive_mask, dtype=np.bool_)
        )
        if mask.shape != exponent.shape:
            raise ValueError("primitive_mask must align with shell exponents.")
        if (
            np.any(~np.isfinite(exponent[mask]))
            or np.any(exponent[mask] <= 0.0)
            or np.any(~np.isfinite(coefficient[:, mask]))
            or np.any(np.all(coefficient[:, mask] == 0.0, axis=1))
        ):
            raise ValueError(
                "Active Gaussian primitives and contractions must be finite and nonzero."
            )
        if not isinstance(representation, GaussianShellRepresentation):
            raise TypeError("representation must be GaussianShellRepresentation.")
        role_ = str(role).strip()
        if not role_:
            raise ValueError("Gaussian shell role must be non-empty.")
        exponent = np.where(mask, exponent, 1.0)
        coefficient = np.where(mask[None, :], coefficient, 0.0)
        self.center_particle_id = center
        self.angular_momentum = angular
        self.exponents = jnp.asarray(exponent)
        self.coefficients = jnp.asarray(coefficient)
        self.primitive_mask = jnp.asarray(mask)
        self.representation = representation
        self.role = role_
        self.shell_id = canonical_fingerprint(
            {
                "kind": "gaussian-shell",
                "center_particle_id": center,
                "angular_momentum": angular,
                "representation": representation.value,
                "role": role_,
                "arrays": array_tree_fingerprint(
                    {
                        "exponents": exponent,
                        "coefficients": coefficient,
                        "primitive_mask": mask,
                    }
                ),
            }
        )

    @property
    def primitive_count(self) -> int:
        return self.exponents.size

    @property
    def contraction_count(self) -> int:
        return self.coefficients.shape[0]

    @property
    def cartesian_component_count(self) -> int:
        angular = self.angular_momentum
        return (angular + 1) * (angular + 2) // 2

    @property
    def component_count(self) -> int:
        if self.representation is GaussianShellRepresentation.CARTESIAN:
            return self.cartesian_component_count
        return 2 * self.angular_momentum + 1

    @property
    def cartesian_function_count(self) -> int:
        return self.contraction_count * self.cartesian_component_count

    @property
    def function_count(self) -> int:
        return self.contraction_count * self.component_count


__all__ = [
    "GaussianShellPlan",
    "GaussianShellRepresentation",
    "cartesian_angular_exponents",
    "cartesian_primitive_normalization",
    "odd_double_factorial",
]
