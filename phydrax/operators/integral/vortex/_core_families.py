#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule


class VortexCoreEvaluation(StrictModule):
    velocity: Array
    velocity_gradient: Array
    vorticity: Array
    finite: Array
    coincident_distinct: Array
    core_id: str = eqx.field(static=True)


def _cross_matrix(vector: Array, /) -> Array:
    x, y, z = vector[..., 0], vector[..., 1], vector[..., 2]
    zero = jnp.zeros_like(x)
    return jnp.stack(
        (
            jnp.stack((zero, -z, y), axis=-1),
            jnp.stack((z, zero, -x), axis=-1),
            jnp.stack((-y, x, zero), axis=-1),
        ),
        axis=-2,
    )


class SingularVortexKernel2D(StrictModule):
    """Free-space point-vortex authority away from distinct coincidences."""

    core_id: str = eqx.field(static=True)

    def __init__(self):
        self.core_id = canonical_fingerprint({"kind": "singular-vortex-core-2d"})

    def evaluate(
        self,
        displacement: ArrayLike,
        strength: ArrayLike,
        /,
        *,
        self_mask: ArrayLike | None = None,
    ) -> VortexCoreEvaluation:
        delta = jnp.asarray(displacement)
        gamma = jnp.asarray(strength, dtype=delta.dtype)
        if delta.shape[-1:] != (2,) or gamma.shape != delta.shape[:-1]:
            raise ValueError("Singular 2-D vortex arrays have incompatible shapes.")
        own = (
            jnp.zeros(gamma.shape, dtype=bool)
            if self_mask is None
            else jnp.asarray(self_mask, dtype=bool)
        )
        if own.shape != gamma.shape:
            raise ValueError("self_mask must match the vortex interaction shape.")
        squared = jnp.sum(delta * delta, axis=-1)
        coincident = (squared == 0.0) & ~own
        safe_squared = jnp.where(own | coincident, 1.0, squared)
        factor = gamma / (2.0 * math.pi * safe_squared)
        perpendicular = jnp.stack((-delta[..., 1], delta[..., 0]), axis=-1)
        velocity = factor[..., None] * perpendicular
        x, y = delta[..., 0], delta[..., 1]
        inverse_squared = 1.0 / safe_squared
        inverse_fourth = inverse_squared**2
        gradient = (
            gamma[..., None, None]
            / (2.0 * math.pi)
            * jnp.stack(
                (
                    jnp.stack(
                        (
                            2.0 * x * y * inverse_fourth,
                            -inverse_squared + 2.0 * y * y * inverse_fourth,
                        ),
                        axis=-1,
                    ),
                    jnp.stack(
                        (
                            inverse_squared - 2.0 * x * x * inverse_fourth,
                            -2.0 * x * y * inverse_fourth,
                        ),
                        axis=-1,
                    ),
                ),
                axis=-2,
            )
        )
        velocity = jnp.where(own[..., None], 0.0, velocity)
        gradient = jnp.where(own[..., None, None], 0.0, gradient)
        vorticity = jnp.zeros_like(gamma)
        finite = (
            jnp.all(jnp.isfinite(velocity))
            & jnp.all(jnp.isfinite(gradient))
            & ~jnp.any(coincident)
        )
        return VortexCoreEvaluation(
            velocity, gradient, vorticity, finite, coincident, self.core_id
        )


class RosenheadVortexKernel2D(StrictModule):
    """Algebraically regularized 2-D vortex core."""

    core_id: str = eqx.field(static=True)

    def __init__(self):
        self.core_id = canonical_fingerprint({"kind": "rosenhead-vortex-core-2d"})

    def evaluate(
        self, displacement: ArrayLike, strength: ArrayLike, core_radius: ArrayLike, /
    ) -> VortexCoreEvaluation:
        delta = jnp.asarray(displacement)
        gamma = jnp.asarray(strength, dtype=delta.dtype)
        core = jnp.asarray(core_radius, dtype=delta.dtype)
        if (
            delta.shape[-1:] != (2,)
            or gamma.shape != delta.shape[:-1]
            or core.shape != gamma.shape
        ):
            raise ValueError("Rosenhead 2-D vortex arrays have incompatible shapes.")
        valid_core = jnp.isfinite(core) & (core > 0.0)
        safe_core = jnp.where(valid_core, core, 1.0)
        squared = jnp.sum(delta * delta, axis=-1)
        denominator = squared + safe_core**2
        factor = gamma / (2.0 * math.pi * denominator)
        perpendicular = jnp.stack((-delta[..., 1], delta[..., 0]), axis=-1)
        velocity = factor[..., None] * perpendicular
        x, y = delta[..., 0], delta[..., 1]
        inverse = 1.0 / denominator
        inverse_squared = inverse**2
        gradient = (
            gamma[..., None, None]
            / (2.0 * math.pi)
            * jnp.stack(
                (
                    jnp.stack(
                        (
                            2.0 * x * y * inverse_squared,
                            -inverse + 2.0 * y * y * inverse_squared,
                        ),
                        axis=-1,
                    ),
                    jnp.stack(
                        (
                            inverse - 2.0 * x * x * inverse_squared,
                            -2.0 * x * y * inverse_squared,
                        ),
                        axis=-1,
                    ),
                ),
                axis=-2,
            )
        )
        vorticity = gamma * safe_core**2 / (math.pi * denominator**2)
        finite = (
            valid_core.all()
            & jnp.all(jnp.isfinite(velocity))
            & jnp.all(jnp.isfinite(gradient))
            & jnp.all(jnp.isfinite(vorticity))
        )
        return VortexCoreEvaluation(
            velocity,
            gradient,
            vorticity,
            finite,
            jnp.zeros_like(gamma, dtype=bool),
            self.core_id,
        )


class SingularVortexKernel3D(StrictModule):
    """Singular three-dimensional vorton authority with explicit self identity."""

    core_id: str = eqx.field(static=True)

    def __init__(self):
        self.core_id = canonical_fingerprint({"kind": "singular-vortex-core-3d"})

    def evaluate(
        self,
        displacement: ArrayLike,
        strength: ArrayLike,
        /,
        *,
        self_mask: ArrayLike | None = None,
    ) -> VortexCoreEvaluation:
        delta = jnp.asarray(displacement)
        gamma = jnp.asarray(strength, dtype=delta.dtype)
        if delta.shape[-1:] != (3,) or gamma.shape != delta.shape:
            raise ValueError(
                "Singular 3-D vortex arrays require matching trailing dimension three."
            )
        own = (
            jnp.zeros(delta.shape[:-1], dtype=bool)
            if self_mask is None
            else jnp.asarray(self_mask, dtype=bool)
        )
        if own.shape != delta.shape[:-1]:
            raise ValueError("self_mask must match the vortex interaction shape.")
        squared = jnp.sum(delta * delta, axis=-1)
        coincident = (squared == 0.0) & ~own
        safe_squared = jnp.where(own | coincident, 1.0, squared)
        inverse_three = safe_squared ** (-1.5)
        inverse_five = safe_squared ** (-2.5)
        cross = jnp.cross(gamma, delta)
        coefficient = 1.0 / (4.0 * math.pi)
        velocity = coefficient * inverse_three[..., None] * cross
        gradient = coefficient * (
            inverse_three[..., None, None] * _cross_matrix(gamma)
            - 3.0
            * inverse_five[..., None, None]
            * cross[..., :, None]
            * delta[..., None, :]
        )
        dot = jnp.sum(gamma * delta, axis=-1)
        vorticity = (
            coefficient
            * inverse_five[..., None]
            * (-safe_squared[..., None] * gamma + 3.0 * dot[..., None] * delta)
        )
        velocity = jnp.where(own[..., None], 0.0, velocity)
        gradient = jnp.where(own[..., None, None], 0.0, gradient)
        vorticity = jnp.where(own[..., None], 0.0, vorticity)
        finite = (
            jnp.all(jnp.isfinite(velocity))
            & jnp.all(jnp.isfinite(gradient))
            & jnp.all(jnp.isfinite(vorticity))
            & ~jnp.any(coincident)
        )
        return VortexCoreEvaluation(
            velocity, gradient, vorticity, finite, coincident, self.core_id
        )


class RosenheadVortexKernel3D(StrictModule):
    """Rosenhead--Moore regularized three-dimensional vorton core."""

    core_id: str = eqx.field(static=True)

    def __init__(self):
        self.core_id = canonical_fingerprint({"kind": "rosenhead-vortex-core-3d"})

    def evaluate(
        self, displacement: ArrayLike, strength: ArrayLike, core_radius: ArrayLike, /
    ) -> VortexCoreEvaluation:
        delta = jnp.asarray(displacement)
        gamma = jnp.asarray(strength, dtype=delta.dtype)
        core = jnp.asarray(core_radius, dtype=delta.dtype)
        if (
            delta.shape[-1:] != (3,)
            or gamma.shape != delta.shape
            or core.shape != delta.shape[:-1]
        ):
            raise ValueError("Rosenhead 3-D vortex arrays have incompatible shapes.")
        valid_core = jnp.isfinite(core) & (core > 0.0)
        safe_core = jnp.where(valid_core, core, 1.0)
        squared = jnp.sum(delta * delta, axis=-1)
        denominator = squared + safe_core**2
        inverse_three = denominator ** (-1.5)
        inverse_five = denominator ** (-2.5)
        cross = jnp.cross(gamma, delta)
        coefficient = 1.0 / (4.0 * math.pi)
        velocity = coefficient * inverse_three[..., None] * cross
        gradient = coefficient * (
            inverse_three[..., None, None] * _cross_matrix(gamma)
            - 3.0
            * inverse_five[..., None, None]
            * cross[..., :, None]
            * delta[..., None, :]
        )
        dot = jnp.sum(gamma * delta, axis=-1)
        vorticity = (
            coefficient
            * inverse_five[..., None]
            * (
                (2.0 * safe_core**2 - squared)[..., None] * gamma
                + 3.0 * dot[..., None] * delta
            )
        )
        finite = (
            valid_core.all()
            & jnp.all(jnp.isfinite(velocity))
            & jnp.all(jnp.isfinite(gradient))
            & jnp.all(jnp.isfinite(vorticity))
        )
        return VortexCoreEvaluation(
            velocity,
            gradient,
            vorticity,
            finite,
            jnp.zeros_like(core, dtype=bool),
            self.core_id,
        )


__all__ = [
    "RosenheadVortexKernel2D",
    "RosenheadVortexKernel3D",
    "SingularVortexKernel2D",
    "SingularVortexKernel3D",
    "VortexCoreEvaluation",
]
