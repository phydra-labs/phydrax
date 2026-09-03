#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from .._quantities import cardiovascular_quantity


_DYNAMIC_VISCOSITY = cardiovascular_quantity("dynamic_viscosity")
_STRAIN_RATE = cardiovascular_quantity("strain_rate")


def _positive_scalar(name: str, value: float, /, *, allow_zero: bool = False) -> float:
    scalar = float(value)
    valid_sign = scalar >= 0.0 if allow_zero else scalar > 0.0
    if not np.isfinite(scalar) or not valid_sign:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be one finite {qualifier} scalar.")
    return scalar


class RheologyEvaluation(StrictModule):
    """Cellwise dynamic viscosity and the validity evidence that produced it."""

    shear_rate_per_ms: Array
    dynamic_viscosity_kpa_ms: Array
    finite: Array
    within_shear_rate_range: Array
    within_viscosity_range: Array
    admissible: Array


class NewtonianRheology(StrictModule, NonTrainableState):
    """Constant dynamic viscosity in the cardiovascular kernel units.

    ``kPa*ms`` has the same numerical scale as ``Pa*s``.  The maximum shear
    rate is an explicit qualification envelope, not a clipping threshold.
    """

    dynamic_viscosity_kpa_ms: Array
    maximum_shear_rate_per_ms: float = eqx.field(static=True)
    quantity_spec_id: str = eqx.field(static=True)
    shear_rate_spec_id: str = eqx.field(static=True)
    rheology_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamic_viscosity_kpa_ms: float,
        /,
        *,
        maximum_shear_rate_per_ms: float = 10.0,
    ):
        viscosity = _positive_scalar("dynamic_viscosity_kpa_ms", dynamic_viscosity_kpa_ms)
        maximum = _positive_scalar("maximum_shear_rate_per_ms", maximum_shear_rate_per_ms)
        self.dynamic_viscosity_kpa_ms = jnp.asarray(viscosity, dtype=jnp.float64)
        self.maximum_shear_rate_per_ms = maximum
        self.quantity_spec_id = _DYNAMIC_VISCOSITY.quantity_id
        self.shear_rate_spec_id = _STRAIN_RATE.quantity_id
        self.rheology_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-newtonian-rheology",
                "dynamic_viscosity_kpa_ms": viscosity,
                "maximum_shear_rate_per_ms": maximum,
                "quantity": self.quantity_spec_id,
                "shear_rate_quantity": self.shear_rate_spec_id,
            }
        )

    @property
    def minimum_dynamic_viscosity_kpa_ms(self) -> Array:
        return self.dynamic_viscosity_kpa_ms

    @property
    def maximum_dynamic_viscosity_kpa_ms(self) -> Array:
        return self.dynamic_viscosity_kpa_ms

    def evaluate(self, shear_rate_per_ms: ArrayLike, /) -> RheologyEvaluation:
        rate = jnp.asarray(shear_rate_per_ms)
        if not jnp.issubdtype(rate.dtype, jnp.inexact):
            rate = rate.astype(jnp.float64)
        viscosity = jnp.broadcast_to(
            self.dynamic_viscosity_kpa_ms.astype(rate.dtype), rate.shape
        )
        finite = jnp.isfinite(rate) & jnp.isfinite(viscosity)
        within_rate = (rate >= 0.0) & (rate <= self.maximum_shear_rate_per_ms)
        within_viscosity = viscosity > 0.0
        admissible = finite & within_rate & within_viscosity
        return RheologyEvaluation(
            rate,
            viscosity,
            finite,
            within_rate,
            within_viscosity,
            admissible,
        )

    def dynamic_viscosity(self, shear_rate_per_ms: ArrayLike, /) -> Array:
        evaluation = self.evaluate(shear_rate_per_ms)
        return eqx.error_if(
            evaluation.dynamic_viscosity_kpa_ms,
            ~jnp.all(evaluation.admissible),
            "Shear rate lies outside the Newtonian rheology validity envelope.",
        )


class CarreauYasudaRheology(StrictModule, NonTrainableState):
    """Shear-thinning Carreau--Yasuda law in ``kPa*ms`` and ``1/ms``.

    The constitutive law is
    ``mu_inf + (mu_zero - mu_inf) * (1 + (lambda*gamma)^a)^((n-1)/a)``.
    No extrapolation is certified beyond ``maximum_shear_rate_per_ms``.
    """

    zero_shear_viscosity_kpa_ms: Array
    infinite_shear_viscosity_kpa_ms: Array
    time_constant_ms: Array
    power_index: Array
    transition_exponent: Array
    maximum_shear_rate_per_ms: float = eqx.field(static=True)
    quantity_spec_id: str = eqx.field(static=True)
    shear_rate_spec_id: str = eqx.field(static=True)
    rheology_id: str = eqx.field(static=True)

    def __init__(
        self,
        zero_shear_viscosity_kpa_ms: float,
        infinite_shear_viscosity_kpa_ms: float,
        time_constant_ms: float,
        power_index: float,
        transition_exponent: float,
        /,
        *,
        maximum_shear_rate_per_ms: float = 10.0,
    ):
        mu_zero = _positive_scalar(
            "zero_shear_viscosity_kpa_ms", zero_shear_viscosity_kpa_ms
        )
        mu_infinite = _positive_scalar(
            "infinite_shear_viscosity_kpa_ms",
            infinite_shear_viscosity_kpa_ms,
        )
        time_constant = _positive_scalar("time_constant_ms", time_constant_ms)
        index = _positive_scalar("power_index", power_index)
        exponent = _positive_scalar("transition_exponent", transition_exponent)
        maximum = _positive_scalar("maximum_shear_rate_per_ms", maximum_shear_rate_per_ms)
        if mu_zero < mu_infinite:
            raise ValueError(
                "zero_shear_viscosity_kpa_ms must be at least the infinite-shear value."
            )
        if index > 1.0:
            raise ValueError(
                "Cardiovascular Carreau--Yasuda power_index must lie in (0, 1]."
            )
        dtype = jnp.float64
        self.zero_shear_viscosity_kpa_ms = jnp.asarray(mu_zero, dtype=dtype)
        self.infinite_shear_viscosity_kpa_ms = jnp.asarray(mu_infinite, dtype=dtype)
        self.time_constant_ms = jnp.asarray(time_constant, dtype=dtype)
        self.power_index = jnp.asarray(index, dtype=dtype)
        self.transition_exponent = jnp.asarray(exponent, dtype=dtype)
        self.maximum_shear_rate_per_ms = maximum
        self.quantity_spec_id = _DYNAMIC_VISCOSITY.quantity_id
        self.shear_rate_spec_id = _STRAIN_RATE.quantity_id
        self.rheology_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-carreau-yasuda-rheology",
                "zero_shear_viscosity_kpa_ms": mu_zero,
                "infinite_shear_viscosity_kpa_ms": mu_infinite,
                "time_constant_ms": time_constant,
                "power_index": index,
                "transition_exponent": exponent,
                "maximum_shear_rate_per_ms": maximum,
                "quantity": self.quantity_spec_id,
                "shear_rate_quantity": self.shear_rate_spec_id,
            }
        )

    @property
    def minimum_dynamic_viscosity_kpa_ms(self) -> Array:
        return self.infinite_shear_viscosity_kpa_ms

    @property
    def maximum_dynamic_viscosity_kpa_ms(self) -> Array:
        return self.zero_shear_viscosity_kpa_ms

    def evaluate(self, shear_rate_per_ms: ArrayLike, /) -> RheologyEvaluation:
        rate = jnp.asarray(shear_rate_per_ms)
        if not jnp.issubdtype(rate.dtype, jnp.inexact):
            rate = rate.astype(jnp.float64)
        safe_rate = jnp.maximum(rate, 0.0)
        mu_zero = self.zero_shear_viscosity_kpa_ms.astype(rate.dtype)
        mu_infinite = self.infinite_shear_viscosity_kpa_ms.astype(rate.dtype)
        time_constant = self.time_constant_ms.astype(rate.dtype)
        index = self.power_index.astype(rate.dtype)
        exponent = self.transition_exponent.astype(rate.dtype)
        transition = 1.0 + (time_constant * safe_rate) ** exponent
        viscosity = mu_infinite + (mu_zero - mu_infinite) * transition ** (
            (index - 1.0) / exponent
        )
        finite = jnp.isfinite(rate) & jnp.isfinite(viscosity)
        within_rate = (rate >= 0.0) & (rate <= self.maximum_shear_rate_per_ms)
        tolerance = 32.0 * jnp.finfo(viscosity.dtype).eps * mu_zero
        within_viscosity = (viscosity >= mu_infinite - tolerance) & (
            viscosity <= mu_zero + tolerance
        )
        admissible = finite & within_rate & within_viscosity
        return RheologyEvaluation(
            rate,
            viscosity,
            finite,
            within_rate,
            within_viscosity,
            admissible,
        )

    def dynamic_viscosity(self, shear_rate_per_ms: ArrayLike, /) -> Array:
        evaluation = self.evaluate(shear_rate_per_ms)
        return eqx.error_if(
            evaluation.dynamic_viscosity_kpa_ms,
            ~jnp.all(evaluation.admissible),
            "Shear rate lies outside the Carreau--Yasuda validity envelope.",
        )


RheologyModel: TypeAlias = NewtonianRheology | CarreauYasudaRheology


__all__ = [
    "CarreauYasudaRheology",
    "NewtonianRheology",
    "RheologyEvaluation",
    "RheologyModel",
]
