#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._lattice import LatticeBoltzmannVelocitySet
from ._moments import central_moments, PreparedMomentBasis, raw_moments
from ._precision import LatticeBoltzmannPrecisionPolicy


class GuoForcingPlan(StrictModule, NonTrainableState):
    """Second-order force discretization with raw and moment-space projections."""

    compatible_collision_families: tuple[str, ...] = eqx.field(static=True)
    forcing_id: str = "lattice-boltzmann-forcing:guo"

    def __init__(self):
        self.compatible_collision_families = (
            "bgk",
            "central-moment",
            "mrt",
            "regularized-second-order",
            "smagorinsky",
            "trt",
        )

    def supports(self, collision_family: str, /) -> bool:
        return str(collision_family) in self.compatible_collision_families


def guo_raw_source(
    velocity: Array,
    force_density: Array,
    velocity_set: LatticeBoltzmannVelocitySet,
    precision: LatticeBoltzmannPrecisionPolicy,
    /,
) -> Array:
    u = precision.compute(velocity)
    force = precision.compute(force_density)
    c = precision.coefficient(velocity_set.velocities)
    weights = precision.coefficient(velocity_set.weights)
    cs2 = precision.coefficient(velocity_set.sound_speed_squared)
    cu = oe.contract("...d,qd->...q", u, c)
    first = (c.reshape((1,) * (u.ndim - 1) + c.shape) - u[..., None, :]) / cs2
    second = cu[..., :, None] * c / cs2**2
    source = weights * oe.contract("...qd,...d->...q", first + second, force)
    return precision.compute(source)


def guo_moment_source(
    velocity: Array,
    force_density: Array,
    velocity_set: LatticeBoltzmannVelocitySet,
    basis: PreparedMomentBasis,
    precision: LatticeBoltzmannPrecisionPolicy,
    /,
) -> Array:
    """Project Guo forcing through the prepared raw-moment transform."""

    basis.require_lattice(velocity_set)
    source = guo_raw_source(velocity, force_density, velocity_set, precision)
    return raw_moments(source, basis, precision)


def guo_central_moment_source(
    velocity: Array,
    force_density: Array,
    velocity_set: LatticeBoltzmannVelocitySet,
    basis: PreparedMomentBasis,
    precision: LatticeBoltzmannPrecisionPolicy,
    /,
) -> Array:
    """Project Guo forcing onto the same local central basis as collision."""

    basis.require_lattice(velocity_set)
    source = guo_raw_source(velocity, force_density, velocity_set, precision)
    return central_moments(source, velocity, velocity_set, basis, precision)


def zero_force_source(populations: Array, /) -> Array:
    return jnp.zeros_like(populations)


__all__ = [
    "GuoForcingPlan",
    "guo_central_moment_source",
    "guo_moment_source",
    "guo_raw_source",
    "zero_force_source",
]
