#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array

from ._fields import IntensityPlane, ScalarPlaneField, TangentialPlaneField


PlaneField = ScalarPlaneField | TangentialPlaneField


def ideal_square_law(field: PlaneField, /) -> IntensityPlane:
    """Measure ideal square-law intensity in the field's amplitude convention.

    Scalar intensity is ``|E|²``. Tangential intensity is the sum of the two
    component intensities in the orthonormal plane-frame basis.
    """
    if not isinstance(field, (ScalarPlaneField, TangentialPlaneField)):
        raise TypeError("field must be a ScalarPlaneField or TangentialPlaneField.")
    intensity = jnp.real(field.values * jnp.conj(field.values))
    if isinstance(field, TangentialPlaneField):
        intensity = jnp.sum(intensity, axis=-1)
    return IntensityPlane(
        field.space,
        intensity,
        field.angular_frequency,
        field.longitudinal_coordinate,
    )


def integrate_intensity(intensity: IntensityPlane, /) -> Array:
    """Integrate an intensity density with the grid's physical area measure."""
    if not isinstance(intensity, IntensityPlane):
        raise TypeError("intensity must be an IntensityPlane.")
    return intensity.space.grid.measure.integrate(intensity.values.reshape((-1,)))


__all__ = ["ideal_square_law", "integrate_intensity"]
