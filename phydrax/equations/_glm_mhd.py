#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from ._hyperbolic_systems import AbstractAdmissibleSystem, IdealMHDSystem
from ._materials import IdealGasMaterial


class GLMIdealMHDSystem(AbstractAdmissibleSystem):
    """Cell-centered ideal MHD with hyperbolic divergence cleaning."""

    ideal: IdealMHDSystem
    cleaning_speed: float = eqx.field(static=True)
    damping_rate: float = eqx.field(static=True)

    def __init__(
        self,
        dimension: int = 3,
        /,
        *,
        material: IdealGasMaterial | None = None,
        cleaning_speed: float = 1.0,
        damping_rate: float = 0.1,
    ):
        speed = float(cleaning_speed)
        damping = float(damping_rate)
        if (
            not np.isfinite(speed)
            or speed <= 0.0
            or not np.isfinite(damping)
            or damping < 0.0
        ):
            raise ValueError("GLM cleaning controls are invalid.")
        ideal = IdealMHDSystem(dimension, material=material)
        self.ideal = ideal
        self.dimension = ideal.dimension
        self.cleaning_speed = speed
        self.damping_rate = damping
        self.component_names = ideal.component_names + ("glm_scalar",)
        self.system_id = canonical_fingerprint(
            {
                "kind": "glm-ideal-mhd-system",
                "ideal": ideal.system_id,
                "cleaning_speed": speed,
                "damping_rate": damping,
            }
        )

    def conserved_to_primitive(self, state: Array, /) -> Array:
        value = jnp.asarray(state)
        return jnp.concatenate(
            (self.ideal.conserved_to_primitive(value[..., :8]), value[..., 8:9]),
            axis=-1,
        )

    def primitive_to_conserved(self, primitive: Array, /) -> Array:
        value = jnp.asarray(primitive)
        return jnp.concatenate(
            (self.ideal.primitive_to_conserved(value[..., :8]), value[..., 8:9]),
            axis=-1,
        )

    def physical_flux(self, state: Array, axis: int, args=None, /) -> Array:
        value = jnp.asarray(state)
        axis_ = int(axis)
        base = self.ideal.physical_flux(value[..., :8], axis_, args)
        psi = value[..., 8]
        base = base.at[..., 5 + axis_].add(psi)
        psi_flux = self.cleaning_speed**2 * value[..., 5 + axis_]
        return jnp.concatenate((base, psi_flux[..., None]), axis=-1)

    def max_wave_speed(
        self,
        left: Array,
        right: Array,
        axis: int,
        args=None,
        /,
    ) -> Array:
        ideal_speed = self.ideal.max_wave_speed(
            jnp.asarray(left)[..., :8],
            jnp.asarray(right)[..., :8],
            axis,
            args,
        )
        return jnp.maximum(ideal_speed, self.cleaning_speed)

    def signal_bounds(
        self,
        left: Array,
        right: Array,
        axis: int,
        args=None,
        /,
    ) -> tuple[Array, Array]:
        speed = self.max_wave_speed(left, right, axis, args)
        return -speed, speed

    def normal_signal_bounds(
        self,
        left: Array,
        right: Array,
        unit_normal: Array,
        args=None,
        /,
    ) -> tuple[Array, Array]:
        lower, upper = self.ideal.normal_signal_bounds(
            jnp.asarray(left)[..., :8],
            jnp.asarray(right)[..., :8],
            unit_normal,
            args,
        )
        return jnp.minimum(lower, -self.cleaning_speed), jnp.maximum(
            upper, self.cleaning_speed
        )

    def damping_source(self, state: Array, /) -> Array:
        value = jnp.asarray(state)
        source = jnp.zeros_like(value)
        return source.at[..., 8].set(-self.damping_rate * value[..., 8])

    def admissible(self, state: Array, /) -> Array:
        value = jnp.asarray(state)
        return self.ideal.admissible(value[..., :8]) & jnp.isfinite(value[..., 8])

    def reflect_state(self, state: Array, axis: int, /) -> Array:
        reflected = self.ideal.reflect_state(jnp.asarray(state)[..., :8], axis)
        return jnp.concatenate((reflected, jnp.asarray(state)[..., 8:9]), axis=-1)


__all__ = ["GLMIdealMHDSystem"]
