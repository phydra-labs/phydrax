#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable, Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ._strict import AbstractAttribute, StrictModule
from ._trainable import NonTrainableState


def double_well_free_energy_density(
    value: ArrayLike,
    scale: ArrayLike = 1.0,
    /,
) -> Array:
    """Evaluate the canonical symmetric binary double-well energy density."""

    values = jnp.asarray(value)
    coefficient = jnp.asarray(scale, dtype=values.dtype)
    if coefficient.shape != ():
        raise ValueError("Double-well scale must be scalar.")
    coefficient = eqx.error_if(
        coefficient,
        ~jnp.isfinite(coefficient) | (coefficient <= 0.0),
        "Double-well scale must be finite and positive.",
    )
    return 0.25 * coefficient * (values**2 - 1.0) ** 2


def double_well_chemical_derivative(
    value: ArrayLike,
    scale: ArrayLike = 1.0,
    /,
) -> Array:
    """Evaluate the canonical double-well chemical derivative."""

    values = jnp.asarray(value)
    coefficient = jnp.asarray(scale, dtype=values.dtype)
    if coefficient.shape != ():
        raise ValueError("Double-well scale must be scalar.")
    coefficient = eqx.error_if(
        coefficient,
        ~jnp.isfinite(coefficient) | (coefficient <= 0.0),
        "Double-well scale must be finite and positive.",
    )
    return coefficient * (values**3 - values)


class BulkPotentialDomain(StrictModule, NonTrainableState):
    """Declared scalar domain for a phase-field bulk potential."""

    lower: float | None = eqx.field(static=True)
    upper: float | None = eqx.field(static=True)
    open_lower: bool = eqx.field(static=True)
    open_upper: bool = eqx.field(static=True)
    domain_id: str = eqx.field(static=True)

    def __init__(
        self,
        lower: float | None = None,
        upper: float | None = None,
        /,
        *,
        open_lower: bool = False,
        open_upper: bool = False,
    ):
        lower_ = None if lower is None else float(lower)
        upper_ = None if upper is None else float(upper)
        if lower_ is not None and not np.isfinite(lower_):
            raise ValueError("Bulk-potential lower bound must be finite or None.")
        if upper_ is not None and not np.isfinite(upper_):
            raise ValueError("Bulk-potential upper bound must be finite or None.")
        if lower_ is not None and upper_ is not None and not lower_ < upper_:
            raise ValueError("Bulk-potential lower bound must be below upper bound.")
        self.lower = lower_
        self.upper = upper_
        self.open_lower = bool(open_lower)
        self.open_upper = bool(open_upper)
        self.domain_id = canonical_fingerprint(
            {
                "kind": "bulk-potential-domain",
                "lower": lower_,
                "upper": upper_,
                "open_lower": self.open_lower,
                "open_upper": self.open_upper,
            }
        )

    def contains(self, value: ArrayLike, /) -> Array:
        values = jnp.asarray(value)
        valid = jnp.isfinite(values)
        if self.lower is not None:
            valid = valid & (
                values > self.lower if self.open_lower else values >= self.lower
            )
        if self.upper is not None:
            valid = valid & (
                values < self.upper if self.open_upper else values <= self.upper
            )
        return valid


class AbstractBulkFreeEnergy(StrictModule, NonTrainableState):
    """Physical density plus an energy-compatible discrete evolution contract."""

    free_energy_id: AbstractAttribute[str]
    domain: AbstractAttribute[BulkPotentialDomain]

    @abc.abstractmethod
    def density(self, value: ArrayLike, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def derivative(self, value: ArrayLike, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def discrete_derivative(
        self,
        current: ArrayLike,
        previous: ArrayLike,
        /,
    ) -> Array:
        """Return ``g`` satisfying ``g*(current-previous)=f(current)-f(previous)``."""

        raise NotImplementedError

    @abc.abstractmethod
    def discrete_incremental_density(
        self,
        current: ArrayLike,
        previous: ArrayLike,
        /,
    ) -> Array:
        """Return a current-state primitive of :meth:`discrete_derivative`."""

        raise NotImplementedError


class PolynomialBulkFreeEnergy(AbstractBulkFreeEnergy):
    """One real scalar polynomial with an exact algebraic discrete gradient."""

    coefficients: Array
    domain: BulkPotentialDomain
    free_energy_id: str = eqx.field(static=True)

    def __init__(self, coefficients: Sequence[float] | ArrayLike, /):
        values = np.asarray(coefficients, dtype=float)
        if (
            values.ndim != 1
            or values.size < 2
            or np.any(~np.isfinite(values))
            or values[-1] <= 0.0
        ):
            raise ValueError(
                "Polynomial bulk energy needs finite rank-1 coefficients and a "
                "positive leading coefficient."
            )
        self.coefficients = jnp.asarray(values)
        self.domain = BulkPotentialDomain()
        self.free_energy_id = canonical_fingerprint(
            {
                "kind": "polynomial-bulk-free-energy",
                "coefficients": array_tree_fingerprint(values),
            }
        )

    def density(self, value: ArrayLike, /) -> Array:
        values = jnp.asarray(value)
        result = jnp.zeros_like(values)
        for coefficient in self.coefficients[::-1]:
            result = result * values + coefficient.astype(values.dtype)
        return result

    def derivative(self, value: ArrayLike, /) -> Array:
        values = jnp.asarray(value)
        result = jnp.zeros_like(values)
        for degree in range(self.coefficients.size - 1, 0, -1):
            result = result * values + (
                degree * self.coefficients[degree].astype(values.dtype)
            )
        return result

    def discrete_derivative(
        self,
        current: ArrayLike,
        previous: ArrayLike,
        /,
    ) -> Array:
        current_ = jnp.asarray(current)
        previous_ = jnp.asarray(previous, dtype=current_.dtype)
        result = jnp.zeros_like(current_)
        for degree in range(1, self.coefficients.size):
            divided = jnp.zeros_like(current_)
            for old_power in range(degree):
                divided = (
                    divided + current_ ** (degree - 1 - old_power) * previous_**old_power
                )
            result = result + self.coefficients[degree].astype(current_.dtype) * divided
        return result

    def discrete_incremental_density(
        self,
        current: ArrayLike,
        previous: ArrayLike,
        /,
    ) -> Array:
        current_ = jnp.asarray(current)
        previous_ = jnp.asarray(previous, dtype=current_.dtype)
        result = jnp.zeros_like(current_)
        for degree in range(1, self.coefficients.size):
            coefficient = self.coefficients[degree].astype(current_.dtype)
            for old_power in range(degree):
                current_power = degree - old_power
                result = result + (
                    coefficient
                    * previous_**old_power
                    * current_**current_power
                    / current_power
                )
        return result


class DoubleWellFreeEnergy(AbstractBulkFreeEnergy):
    scale: Array
    polynomial: PolynomialBulkFreeEnergy
    domain: BulkPotentialDomain
    free_energy_id: str = eqx.field(static=True)

    def __init__(self, scale: ArrayLike = 1.0, /):
        scale_host = np.asarray(scale)
        if scale_host.shape != () or not np.isfinite(scale_host) or scale_host <= 0.0:
            raise ValueError("Double-well scale must be one positive finite scalar.")
        self.scale = jnp.asarray(scale_host)
        self.polynomial = PolynomialBulkFreeEnergy(
            (
                0.25 * float(scale_host),
                0.0,
                -0.5 * float(scale_host),
                0.0,
                0.25 * float(scale_host),
            )
        )
        self.domain = BulkPotentialDomain()
        self.free_energy_id = canonical_fingerprint(
            {"kind": "double-well-free-energy", "scale": float(scale_host)}
        )

    def density(self, value: ArrayLike, /) -> Array:
        return double_well_free_energy_density(value, self.scale)

    def derivative(self, value: ArrayLike, /) -> Array:
        return double_well_chemical_derivative(value, self.scale)

    def discrete_derivative(
        self,
        current: ArrayLike,
        previous: ArrayLike,
        /,
    ) -> Array:
        return self.polynomial.discrete_derivative(current, previous)

    def discrete_incremental_density(
        self,
        current: ArrayLike,
        previous: ArrayLike,
        /,
    ) -> Array:
        return self.polynomial.discrete_incremental_density(current, previous)


class CallableBulkFreeEnergy(AbstractBulkFreeEnergy):
    """Explicitly identified user potential with a complete discrete law."""

    density_function: Callable = eqx.field(static=True)
    derivative_function: Callable = eqx.field(static=True)
    discrete_derivative_function: Callable = eqx.field(static=True)
    incremental_density_function: Callable = eqx.field(static=True)
    domain: BulkPotentialDomain
    free_energy_id: str = eqx.field(static=True)

    def __init__(
        self,
        density: Callable,
        derivative: Callable,
        discrete_derivative: Callable,
        incremental_density: Callable,
        /,
        *,
        potential_id: str,
        domain: BulkPotentialDomain | None = None,
    ):
        functions = (density, derivative, discrete_derivative, incremental_density)
        if any(not callable(function) for function in functions):
            raise TypeError("Callable bulk energy requires four callable laws.")
        identifier = str(potential_id)
        if not identifier:
            raise ValueError("Callable bulk energy requires a stable potential_id.")
        self.density_function = density
        self.derivative_function = derivative
        self.discrete_derivative_function = discrete_derivative
        self.incremental_density_function = incremental_density
        self.domain = BulkPotentialDomain() if domain is None else domain
        self.free_energy_id = canonical_fingerprint(
            {
                "kind": "callable-bulk-free-energy",
                "declared_id": identifier,
                "domain": self.domain.domain_id,
            }
        )

    def density(self, value: ArrayLike, /) -> Array:
        return jnp.asarray(self.density_function(jnp.asarray(value)))

    def derivative(self, value: ArrayLike, /) -> Array:
        return jnp.asarray(self.derivative_function(jnp.asarray(value)))

    def discrete_derivative(
        self,
        current: ArrayLike,
        previous: ArrayLike,
        /,
    ) -> Array:
        return jnp.asarray(
            self.discrete_derivative_function(
                jnp.asarray(current),
                jnp.asarray(previous),
            )
        )

    def discrete_incremental_density(
        self,
        current: ArrayLike,
        previous: ArrayLike,
        /,
    ) -> Array:
        return jnp.asarray(
            self.incremental_density_function(
                jnp.asarray(current),
                jnp.asarray(previous),
            )
        )


class BinaryFreeEnergyEvaluation(StrictModule):
    density: Array
    chemical_derivative: Array
    free_energy_id: str = eqx.field(static=True)


def evaluate_binary_free_energy(
    free_energy: AbstractBulkFreeEnergy,
    value: ArrayLike,
    /,
) -> BinaryFreeEnergyEvaluation:
    if not isinstance(free_energy, AbstractBulkFreeEnergy):
        raise TypeError("free_energy must be AbstractBulkFreeEnergy.")
    values = jnp.asarray(value)
    density = jnp.asarray(free_energy.density(values))
    derivative = jnp.asarray(free_energy.derivative(values))
    if density.shape != values.shape or derivative.shape != values.shape:
        raise ValueError("Bulk free energy must preserve field shape.")
    return BinaryFreeEnergyEvaluation(density, derivative, free_energy.free_energy_id)


__all__ = [
    "AbstractBulkFreeEnergy",
    "BinaryFreeEnergyEvaluation",
    "BulkPotentialDomain",
    "CallableBulkFreeEnergy",
    "DoubleWellFreeEnergy",
    "PolynomialBulkFreeEnergy",
    "double_well_chemical_derivative",
    "double_well_free_energy_density",
    "evaluate_binary_free_energy",
]
