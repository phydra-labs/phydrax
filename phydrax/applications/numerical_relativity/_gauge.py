#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._derivatives import FourthOrderDerivatives
from ._state import Z4cState


class Z4cGaugeRates(StrictModule):
    lapse: Array
    shift: Array
    shift_driver: Array


class AbstractZ4cGauge(StrictModule, NonTrainableState):
    """Gauge evolution contract evaluated at the same stage as the Z4c RHS."""

    gauge_id: eqx.AbstractVar[str]

    @abc.abstractmethod
    def rates(
        self,
        state: Z4cState,
        derivatives: FourthOrderDerivatives,
        conformal_connection_rhs: Array,
        /,
    ) -> Z4cGaugeRates:
        raise NotImplementedError


class GeodesicGauge(AbstractZ4cGauge):
    """Frozen unit-lapse, zero-shift geodesic gauge evolution."""

    gauge_id: str = "z4c-gauge:geodesic"

    def rates(
        self,
        state: Z4cState,
        derivatives: FourthOrderDerivatives,
        conformal_connection_rhs: Array,
        /,
    ) -> Z4cGaugeRates:
        del derivatives, conformal_connection_rhs
        return Z4cGaugeRates(
            jnp.zeros_like(state.lapse),
            jnp.zeros_like(state.shift),
            jnp.zeros_like(state.shift_driver),
        )


class HarmonicGauge(AbstractZ4cGauge):
    """Advective harmonic slicing with a frozen spatial gauge."""

    slicing_speed: float = eqx.field(static=True)
    gauge_id: str = eqx.field(static=True)

    def __init__(self, /, *, slicing_speed: float = 1.0):
        speed = float(slicing_speed)
        if not isfinite(speed) or speed <= 0.0:
            raise ValueError("slicing_speed must be finite and positive.")
        self.slicing_speed = speed
        self.gauge_id = canonical_fingerprint(
            {"kind": "z4c-harmonic-gauge", "slicing_speed": speed}
        )

    def rates(
        self,
        state: Z4cState,
        derivatives: FourthOrderDerivatives,
        conformal_connection_rhs: Array,
        /,
    ) -> Z4cGaugeRates:
        del conformal_connection_rhs
        lapse = derivatives.advect(state.lapse, state.shift) - (
            self.slicing_speed * state.lapse**2 * state.trace_extrinsic_curvature
        )
        return Z4cGaugeRates(
            lapse,
            jnp.zeros_like(state.shift),
            jnp.zeros_like(state.shift_driver),
        )


class MovingPunctureGauge(AbstractZ4cGauge):
    """Advective 1+log slicing and hyperbolic Gamma-driver shift."""

    slicing_coefficient: float = eqx.field(static=True)
    shift_coefficient: float = eqx.field(static=True)
    driver_damping: float = eqx.field(static=True)
    advective: bool = eqx.field(static=True)
    gauge_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        slicing_coefficient: float = 2.0,
        shift_coefficient: float = 0.75,
        driver_damping: float = 1.0,
        advective: bool = True,
    ):
        slicing = float(slicing_coefficient)
        shift = float(shift_coefficient)
        damping = float(driver_damping)
        if any(not isfinite(value) or value < 0.0 for value in (slicing, shift, damping)):
            raise ValueError(
                "Moving-puncture coefficients must be finite and non-negative."
            )
        if not isinstance(advective, bool):
            raise TypeError("advective must be Boolean.")
        self.slicing_coefficient = slicing
        self.shift_coefficient = shift
        self.driver_damping = damping
        self.advective = advective
        self.gauge_id = canonical_fingerprint(
            {
                "kind": "z4c-moving-puncture-gauge",
                "slicing_coefficient": slicing,
                "shift_coefficient": shift,
                "driver_damping": damping,
                "advective": advective,
            }
        )

    def rates(
        self,
        state: Z4cState,
        derivatives: FourthOrderDerivatives,
        conformal_connection_rhs: Array,
        /,
    ) -> Z4cGaugeRates:
        if self.advective:
            advected_lapse = derivatives.advect(state.lapse, state.shift)
            advected_shift = derivatives.advect(state.shift, state.shift)
            advected_driver = derivatives.advect(state.shift_driver, state.shift)
            advected_connection = derivatives.advect(
                state.conformal_connection, state.shift
            )
        else:
            advected_lapse = jnp.zeros_like(state.lapse)
            advected_shift = jnp.zeros_like(state.shift)
            advected_driver = jnp.zeros_like(state.shift_driver)
            advected_connection = jnp.zeros_like(state.conformal_connection)
        lapse = advected_lapse - (
            self.slicing_coefficient * state.lapse * state.trace_extrinsic_curvature
        )
        shift = advected_shift + self.shift_coefficient * state.shift_driver
        driver = (
            advected_driver
            + conformal_connection_rhs
            - advected_connection
            - self.driver_damping * state.shift_driver
        )
        return Z4cGaugeRates(lapse, shift, driver)


__all__ = [
    "AbstractZ4cGauge",
    "GeodesicGauge",
    "HarmonicGauge",
    "MovingPunctureGauge",
    "Z4cGaugeRates",
]
