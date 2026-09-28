#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""GR-invariant form of local polarized emission coefficients.

Local spectral coefficients are owned by `phydrax.electromagnetics`; this module
only converts them to the affine-invariant transfer convention.
"""

from __future__ import annotations

from math import pi
from typing import assert_never

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from ..._physical import ElectromagneticScaleContract
from ..._strict import StrictModule
from ...electromagnetics import (
    ThermalFreeFreeCoefficients,
    ThermalSynchrotronCoefficients,
)


_SI_SCALE_ID = ElectromagneticScaleContract.si().scale_id


class InvariantEmissionCoefficients(StrictModule):
    emission: Array
    propagation_matrix: Array
    finite: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    model_id: str = eqx.field(static=True)


def invariant_emission_coefficients(
    coefficients: ThermalSynchrotronCoefficients | ThermalFreeFreeCoefficients,
    comoving_frequency_hz: ArrayLike,
    /,
) -> InvariantEmissionCoefficients:
    """Convert local SI coefficients to the affine invariant convention.

    This uses ``J = j_nu / nu^2`` and ``K = nu K_nu`` for an affine
    normalization satisfying ``d ell / d lambda = nu``.  A differently normalized
    affine tangent must rescale the returned pair before transfer. Free–free
    coefficients carry ``j_omega`` per unit angular frequency and are converted
    with ``j_nu = 2 pi j_omega``; they must come from the SI scale and be
    evaluated at ``omega = 2 pi nu``.
    """

    frequency = jnp.asarray(comoving_frequency_hz)
    match coefficients:
        case ThermalSynchrotronCoefficients():
            emission_hz = coefficients.emission
            source_valid = coefficients.evidence.physically_valid
            source_finite = coefficients.evidence.finite
            source_qualified = coefficients.evidence.qualified
            source_derivative = coefficients.evidence.derivative_valid
        case ThermalFreeFreeCoefficients():
            if coefficients.scale_id != _SI_SCALE_ID:
                raise ValueError("Free-free coefficients must use the SI scale.")
            emission_hz = 2.0 * pi * coefficients.emission
            matched = (
                jnp.abs(coefficients.angular_frequency - 2.0 * pi * frequency)
                <= 1.0e-12 * coefficients.angular_frequency
            )
            source_valid = coefficients.evidence.physically_valid & matched
            source_finite = coefficients.evidence.finite
            source_qualified = coefficients.evidence.qualified
            # Frequency-matching and support are discrete gates; the smooth
            # coefficient is differentiable wherever it is qualified.
            source_derivative = coefficients.evidence.qualified
        case _:
            assert_never(coefficients)
    expected = emission_hz.shape[:-1]
    if frequency.shape != expected:
        raise ValueError("Comoving frequency must match coefficient sample shape.")
    frequency_finite = jnp.isfinite(frequency)
    physically_valid = source_valid & (frequency > 0.0)
    safe_frequency = jnp.where(physically_valid, frequency, 1.0)
    emission = emission_hz / safe_frequency[..., None] ** 2
    propagation = coefficients.propagation_matrix * safe_frequency[..., None, None]
    nan = jnp.asarray(jnp.nan, dtype=emission.dtype)
    emission = jnp.where(physically_valid[..., None], emission, nan)
    propagation = jnp.where(physically_valid[..., None, None], propagation, nan)
    finite = (
        source_finite
        & frequency_finite
        & jnp.all(jnp.isfinite(emission), axis=-1)
        & jnp.all(jnp.isfinite(propagation), axis=(-2, -1))
    )
    qualified = source_qualified & finite & physically_valid
    derivative_valid = source_derivative & qualified
    return InvariantEmissionCoefficients(
        emission,
        propagation,
        finite,
        physically_valid,
        qualified,
        derivative_valid,
        coefficients.model_id,
    )


__all__ = ["invariant_emission_coefficients", "InvariantEmissionCoefficients"]
