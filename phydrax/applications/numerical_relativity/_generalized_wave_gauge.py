#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Explicit generalized-wave and conformal-gauge source transition contract."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule


class GeneralizedWaveGaugePlan(StrictModule):
    """Transition between initial and timelike-boundary-compatible gauge sources."""

    initial_source: Array
    boundary_source: Array
    transition_time: float = eqx.field(static=True)
    transition_time_width: float = eqx.field(static=True)
    transition_radius: float = eqx.field(static=True)
    transition_radius_width: float = eqx.field(static=True)
    damping: float = eqx.field(static=True)
    scalar_curvature_gauge: float = eqx.field(static=True)
    source_sign: int = eqx.field(static=True)
    gauge_id: str = eqx.field(static=True)

    def __init__(
        self,
        initial_source: ArrayLike,
        boundary_source: ArrayLike,
        /,
        *,
        transition_time: float,
        transition_time_width: float,
        transition_radius: float,
        transition_radius_width: float,
        damping: float,
        scalar_curvature_gauge: float,
        source_sign: int = 1,
    ):
        initial = np.asarray(initial_source, dtype=float)
        boundary = np.asarray(boundary_source, dtype=float)
        values = tuple(
            float(value)
            for value in (
                transition_time,
                transition_time_width,
                transition_radius,
                transition_radius_width,
                damping,
                scalar_curvature_gauge,
            )
        )
        sign = int(source_sign)
        if initial.shape != (4,) or boundary.shape != (4,):
            raise ValueError("Generalized-wave gauge sources must have shape (4,).")
        if not np.all(np.isfinite(initial)) or not np.all(np.isfinite(boundary)):
            raise ValueError("Generalized-wave gauge sources must be finite.")
        if not all(np.isfinite(value) for value in values):
            raise ValueError("Generalized-wave gauge parameters must be finite.")
        if values[1] <= 0.0 or values[3] <= 0.0 or values[4] < 0.0:
            raise ValueError(
                "Gauge transition widths must be positive and damping nonnegative."
            )
        if sign not in (-1, 1):
            raise ValueError("source_sign must be +1 or -1.")
        self.initial_source = jnp.asarray(initial)
        self.boundary_source = jnp.asarray(boundary)
        self.transition_time = values[0]
        self.transition_time_width = values[1]
        self.transition_radius = values[2]
        self.transition_radius_width = values[3]
        self.damping = values[4]
        self.scalar_curvature_gauge = values[5]
        self.source_sign = sign
        self.gauge_id = canonical_fingerprint(
            {
                "kind": "generalized-wave-conformal-gauge-plan",
                "initial_source": array_tree_fingerprint(initial),
                "boundary_source": array_tree_fingerprint(boundary),
                "transition_time": values[0],
                "transition_time_width": values[1],
                "transition_radius": values[2],
                "transition_radius_width": values[3],
                "damping": values[4],
                "scalar_curvature_gauge": values[5],
                "source_sign": sign,
            }
        )

    def source(self, time: ArrayLike, radius: ArrayLike, /) -> Array:
        time_value = jnp.asarray(time)
        radius_value = jnp.asarray(radius, dtype=time_value.dtype)
        temporal = 0.5 * (
            1.0
            + jnp.tanh((time_value - self.transition_time) / self.transition_time_width)
        )
        radial = 0.5 * (
            1.0
            + jnp.tanh(
                (radius_value - self.transition_radius) / self.transition_radius_width
            )
        )
        blend = temporal * radial
        shape = (4,) + (1,) * blend.ndim
        initial = self.initial_source.reshape(shape)
        boundary = self.boundary_source.reshape(shape)
        return (1.0 - blend[None, ...]) * initial + blend[None, ...] * boundary


class GeneralizedWaveGaugeEvidence(StrictModule):
    source: Array
    wave_constraint: Array
    scalar_curvature_residual: Array
    maximum_wave_constraint: Array
    finite: Array
    accepted: Array
    gauge_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


def evaluate_generalized_wave_gauge(
    plan: GeneralizedWaveGaugePlan,
    contracted_christoffel: ArrayLike,
    scalar_curvature: ArrayLike,
    time: ArrayLike,
    radius: ArrayLike,
    /,
    *,
    tolerance: float,
) -> GeneralizedWaveGaugeEvidence:
    if not isinstance(plan, GeneralizedWaveGaugePlan):
        raise TypeError("plan must be GeneralizedWaveGaugePlan.")
    christoffel = jnp.asarray(contracted_christoffel)
    curvature = jnp.asarray(scalar_curvature)
    source = plan.source(time, radius)
    if christoffel.shape != source.shape:
        raise ValueError("contracted_christoffel and gauge source shapes differ.")
    if curvature.shape != christoffel.shape[1:]:
        raise ValueError("scalar_curvature must match the gauge grid shape.")
    residual = christoffel + plan.source_sign * source
    curvature_residual = curvature - plan.scalar_curvature_gauge
    maximum = jnp.max(jnp.abs(residual))
    finite = (
        jnp.all(jnp.isfinite(source))
        & jnp.all(jnp.isfinite(residual))
        & jnp.all(jnp.isfinite(curvature_residual))
    )
    accepted = (
        finite
        & (maximum <= float(tolerance))
        & (jnp.max(jnp.abs(curvature_residual)) <= float(tolerance))
    )
    return GeneralizedWaveGaugeEvidence(
        source=source,
        wave_constraint=residual,
        scalar_curvature_residual=curvature_residual,
        maximum_wave_constraint=maximum,
        finite=finite,
        accepted=accepted,
        gauge_id=plan.gauge_id,
        claim="finite-generalized-wave-and-conformal-scalar-curvature-gauge-evidence",
    )


__all__ = [
    "GeneralizedWaveGaugeEvidence",
    "GeneralizedWaveGaugePlan",
    "evaluate_generalized_wave_gauge",
]
