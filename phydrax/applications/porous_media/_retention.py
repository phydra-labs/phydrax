#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Pressure-qualified van Genuchten retention and Mualem liquid mobility."""

from __future__ import annotations

import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
from jaxtyping import Array

from ..._strict import StrictModule
from ...units import convert_value, UnitDefinition
from ._materials import _finite, INVERSE_PASCAL


class VanGenuchtenMualem(StrictModule):
    """Equilibrium nonhysteretic retention in gauge pressure relative to gas.

    alpha is in Pa^-1, n > 1, m = 1-1/n, residual saturation in [0,1).
    p >= 0 is exactly saturated with relative permeability one and zero
    retention storage. As p -> -infinity, saturation reaches the declared
    residual endpoint and permeability reaches zero. No permeability/storage
    floors are applied. The saturated-side derivative at p=0 is selected;
    the unsaturated permeability slope may be singular there for n < 2.
    """

    alpha_Pa_inverse: Array
    n: Array
    residual_saturation: Array
    pore_connectivity: Array

    def __init__(
        self,
        alpha_Pa_inverse,
        n,
        /,
        *,
        residual_saturation=0.0,
        pore_connectivity=0.5,
        alpha_unit: UnitDefinition = INVERSE_PASCAL,
    ):
        self.alpha_Pa_inverse = _finite(
            convert_value(alpha_Pa_inverse, source=alpha_unit, target=INVERSE_PASCAL),
            "van Genuchten alpha",
            positive=True,
        )
        n_ = _finite(n, "van Genuchten n")
        self.n = eqx.error_if(n_, jnp.any(n_ <= 1), "van Genuchten n must exceed one.")
        residual = _finite(residual_saturation, "residual saturation", nonnegative=True)
        self.residual_saturation = eqx.error_if(
            residual, jnp.any(residual >= 1), "Residual saturation must be below one."
        )
        self.pore_connectivity = _finite(
            pore_connectivity, "Mualem pore connectivity", nonnegative=True
        )

    @property
    def m(self):
        return 1 - 1 / self.n

    def _unsaturated_log_power(self, pressure_Pa):
        pressure = jnp.asarray(pressure_Pa)
        # The inactive saturated branch never evaluates log(0) or a fractional
        # power of zero, preserving the explicitly chosen endpoint derivative.
        suction = jnp.where(pressure < 0, -pressure, 1.0)
        return self.n * (jnp.log(self.alpha_Pa_inverse) + jnp.log(suction))

    def effective_saturation(self, pressure_Pa):
        power = self._unsaturated_log_power(pressure_Pa)
        return jnp.where(
            jnp.asarray(pressure_Pa) >= 0, 1.0, jnp.exp(-self.m * jnn.softplus(power))
        )

    def saturation(self, pressure_Pa):
        return self.residual_saturation + (
            1 - self.residual_saturation
        ) * self.effective_saturation(pressure_Pa)

    def relative_permeability(self, pressure_Pa):
        power = self._unsaturated_log_power(pressure_Pa)
        log_se = -self.m * jnn.softplus(power)
        bracket = -jnp.expm1(-self.m * jnn.softplus(-power))
        unsaturated = jnp.exp(self.pore_connectivity * log_se) * bracket**2
        return jnp.where(jnp.asarray(pressure_Pa) >= 0, 1.0, unsaturated)


__all__ = ["VanGenuchtenMualem"]
