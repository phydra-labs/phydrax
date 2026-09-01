#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._bodies import CelestialBodyCatalog
from ._data import AstrodynamicsDataProvenance
from ._state import CartesianOrbitState
from ._status import AstrodynamicsStatus


def _clenshaw(coefficients: Array, coordinate: Array, /) -> Array:
    b1 = jnp.zeros(coefficients.shape[:-1], dtype=coefficients.dtype)
    b2 = jnp.zeros_like(b1)
    for index in range(int(coefficients.shape[-1]) - 1, 0, -1):
        b0 = 2.0 * coordinate * b1 - b2 + coefficients[..., index]
        b2, b1 = b1, b0
    return coordinate * b1 - b2 + coefficients[..., 0]


class ChebyshevEphemerisEvaluation(StrictModule):
    state: CartesianOrbitState
    acceleration: Array
    segment_index: Array
    valid: Array
    status: Array
    ephemeris_id: str = eqx.field(static=True)


class ChebyshevEphemeris(StrictModule, NonTrainableState):
    """Piecewise position Chebyshev ephemeris with analytic derivatives."""

    segment_bounds: Array
    position_coefficients: Array
    velocity_coefficients: Array
    acceleration_coefficients: Array
    catalog: CelestialBodyCatalog
    provenance: AstrodynamicsDataProvenance
    ephemeris_id: str = eqx.field(static=True)

    def __init__(
        self,
        segment_bounds: ArrayLike,
        position_coefficients: ArrayLike,
        catalog: CelestialBodyCatalog,
        provenance: AstrodynamicsDataProvenance,
        /,
    ):
        bounds = np.asarray(segment_bounds, dtype=float)
        coefficients = np.asarray(position_coefficients, dtype=float)
        if (
            bounds.ndim != 1
            or bounds.size < 2
            or np.any(np.diff(bounds) <= 0.0)
            or coefficients.ndim != 4
            or coefficients.shape[:3] != (bounds.size - 1, catalog.capacity, 3)
            or coefficients.shape[-1] < 2
            or np.any(~np.isfinite(bounds))
            or np.any(~np.isfinite(coefficients))
        ):
            raise ValueError("Chebyshev ephemeris arrays are invalid.")
        velocity = np.empty_like(coefficients)
        acceleration = np.empty_like(coefficients)
        for segment in range(bounds.size - 1):
            scale = 2.0 / (bounds[segment + 1] - bounds[segment])
            for body in range(catalog.capacity):
                for component in range(3):
                    first = (
                        np.polynomial.chebyshev.chebder(
                            coefficients[segment, body, component]
                        )
                        * scale
                    )
                    second = np.polynomial.chebyshev.chebder(first) * scale
                    velocity[segment, body, component] = 0.0
                    acceleration[segment, body, component] = 0.0
                    velocity[segment, body, component, : first.size] = first
                    acceleration[segment, body, component, : second.size] = second
        self.segment_bounds = jnp.asarray(bounds)
        self.position_coefficients = jnp.asarray(coefficients)
        self.velocity_coefficients = jnp.asarray(velocity)
        self.acceleration_coefficients = jnp.asarray(acceleration)
        self.catalog = catalog
        self.provenance = provenance
        self.ephemeris_id = canonical_fingerprint(
            {
                "kind": "chebyshev-ephemeris",
                "catalog": catalog.catalog_id,
                "provenance": provenance.provenance_id,
                "segments": int(bounds.size - 1),
                "degree": int(coefficients.shape[-1] - 1),
            }
        )

    def evaluate(
        self, relative_seconds: ArrayLike, body_index: ArrayLike, /
    ) -> ChebyshevEphemerisEvaluation:
        time = jnp.asarray(relative_seconds).reshape(())
        body = jnp.asarray(body_index, dtype=jnp.int32).reshape(())
        body_valid = (body >= 0) & (body < self.catalog.capacity)
        safe_body = jnp.clip(body, 0, self.catalog.capacity - 1)
        support = (time >= self.segment_bounds[0]) & (time <= self.segment_bounds[-1])
        segment = jnp.clip(
            jnp.searchsorted(self.segment_bounds, time, side="right") - 1,
            0,
            int(self.segment_bounds.size) - 2,
        )
        start = self.segment_bounds[segment]
        end = self.segment_bounds[segment + 1]
        coordinate = 2.0 * (time - start) / (end - start) - 1.0
        position = _clenshaw(self.position_coefficients[segment, safe_body], coordinate)
        velocity = _clenshaw(self.velocity_coefficients[segment, safe_body], coordinate)
        acceleration = _clenshaw(
            self.acceleration_coefficients[segment, safe_body], coordinate
        )
        valid = (
            support
            & body_valid
            & self.catalog.active_mask[safe_body]
            & jnp.all(jnp.isfinite(position))
            & jnp.all(jnp.isfinite(velocity))
            & jnp.all(jnp.isfinite(acceleration))
        )
        status = jnp.where(
            valid,
            int(AstrodynamicsStatus.SUCCESS),
            jnp.where(
                support,
                int(AstrodynamicsStatus.INVALID_DOMAIN),
                int(AstrodynamicsStatus.INVALID_DOMAIN),
            ),
        ).astype(jnp.int32)
        return ChebyshevEphemerisEvaluation(
            CartesianOrbitState(
                jnp.where(valid, position, 0.0),
                jnp.where(valid, velocity, 0.0),
                self.catalog.context,
            ),
            jnp.where(valid, acceleration, 0.0),
            segment,
            valid,
            status,
            self.ephemeris_id,
        )


__all__ = ["ChebyshevEphemeris", "ChebyshevEphemerisEvaluation"]
