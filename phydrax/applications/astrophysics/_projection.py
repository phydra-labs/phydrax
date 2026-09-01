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
from ..astrodynamics import AstrodynamicsContext
from ._observation_status import AstrophysicsObservationStatus


class ObserverProjectionResult(StrictModule):
    sky_position: Array
    projected_separation: Array
    line_of_sight: Array
    foreground: Array
    valid: Array
    status: Array
    projection_id: str = eqx.field(static=True)


class ObserverProjectionPlan(StrictModule, NonTrainableState):
    """Project Cartesian relative positions into a right-handed observer basis."""

    sky_x: Array
    sky_y: Array
    toward_observer: Array
    context: AstrodynamicsContext
    projection_id: str = eqx.field(static=True)

    def __init__(
        self,
        sky_x: ArrayLike,
        sky_y: ArrayLike,
        toward_observer: ArrayLike,
        context: AstrodynamicsContext,
        /,
        *,
        tolerance: float = 1.0e-10,
    ):
        if not isinstance(context, AstrodynamicsContext):
            raise TypeError("context must be an AstrodynamicsContext.")
        first = np.asarray(sky_x, dtype=float)
        second = np.asarray(sky_y, dtype=float)
        sight = np.asarray(toward_observer, dtype=float)
        if any(value.shape != (3,) for value in (first, second, sight)):
            raise ValueError("Observer basis vectors must have shape (3,).")
        if any(np.any(~np.isfinite(value)) for value in (first, second, sight)):
            raise ValueError("Observer basis vectors must be finite.")
        tolerance_ = float(tolerance)
        gram = np.asarray(
            (
                (np.dot(first, first), np.dot(first, second), np.dot(first, sight)),
                (np.dot(second, first), np.dot(second, second), np.dot(second, sight)),
                (np.dot(sight, first), np.dot(sight, second), np.dot(sight, sight)),
            )
        )
        if not np.allclose(gram, np.eye(3), atol=tolerance_, rtol=0.0):
            raise ValueError("Observer basis must be orthonormal.")
        if not np.allclose(np.cross(first, second), sight, atol=tolerance_, rtol=0.0):
            raise ValueError("Observer basis must be right-handed.")
        self.sky_x = jnp.asarray(first)
        self.sky_y = jnp.asarray(second)
        self.toward_observer = jnp.asarray(sight)
        self.context = context
        self.projection_id = canonical_fingerprint(
            {
                "kind": "observer-projection",
                "context": context.context_id,
                "sky_x": first.tolist(),
                "sky_y": second.tolist(),
                "toward_observer": sight.tolist(),
            }
        )

    def project(
        self,
        relative_position: ArrayLike,
        /,
        *,
        context: AstrodynamicsContext | None = None,
    ) -> ObserverProjectionResult:
        if context is not None:
            self.context.require_compatible(context)
        position = jnp.asarray(relative_position)
        if position.shape[-1:] != (3,):
            raise ValueError("Relative position must have trailing shape (3,).")
        first = jnp.sum(position * self.sky_x, axis=-1)
        second = jnp.sum(position * self.sky_y, axis=-1)
        sight = jnp.sum(position * self.toward_observer, axis=-1)
        sky = jnp.stack((first, second), axis=-1)
        separation = jnp.sqrt(first * first + second * second)
        finite = jnp.all(jnp.isfinite(position), axis=-1)
        status = jnp.where(
            finite,
            int(AstrophysicsObservationStatus.SUCCESS),
            int(AstrophysicsObservationStatus.NONFINITE_INPUT),
        ).astype(jnp.int32)
        return ObserverProjectionResult(
            jnp.where(finite[..., None], sky, 0.0),
            jnp.where(finite, separation, 0.0),
            jnp.where(finite, sight, 0.0),
            finite & (sight > 0.0),
            finite,
            status,
            self.projection_id,
        )


__all__ = ["ObserverProjectionPlan", "ObserverProjectionResult"]
