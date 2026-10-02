#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""On-shell particle contracts and scattering kinematics."""

from __future__ import annotations

import math
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._lorentz import FourMomentum, LorentzFrame, minkowski_dot
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...typing import checked


class Particle(StrictModule, NonTrainableState):
    """Stable particle species data needed by scattering calculations."""

    mass: Array
    charge: Array
    spin_twice: int = eqx.field(static=True)
    name: str = eqx.field(static=True)
    antiparticle: str = eqx.field(static=True)
    statistics: Literal["fermion", "boson"] = eqx.field(static=True)
    particle_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        /,
        *,
        mass: float,
        charge: float,
        spin_twice: int,
        antiparticle: str,
        statistics: Literal["fermion", "boson"],
    ) -> None:
        mass_ = float(mass)
        charge_ = float(charge)
        spin_ = int(spin_twice)
        if not name or not antiparticle:
            raise ValueError("Particle names must be nonempty.")
        if not math.isfinite(mass_) or mass_ < 0.0 or not math.isfinite(charge_):
            raise ValueError(
                "Particle mass and charge must be finite, with nonnegative mass."
            )
        if spin_ < 0 or statistics not in ("fermion", "boson"):
            raise ValueError("Particle spin/statistics declaration is invalid.")
        self.mass = jnp.asarray(mass_)
        self.charge = jnp.asarray(charge_)
        self.spin_twice = spin_
        self.name = str(name)
        self.antiparticle = str(antiparticle)
        self.statistics = statistics
        self.particle_id = canonical_fingerprint(
            {
                "kind": "particle",
                "name": name,
                "mass": mass_,
                "charge": charge_,
                "spin_twice": spin_,
                "antiparticle": antiparticle,
                "statistics": statistics,
            }
        )


class MassShell(StrictModule, NonTrainableState):
    """Future mass shell for one particle species."""

    particle: Particle
    tolerance: float = eqx.field(static=True)
    shell_id: str = eqx.field(static=True)

    @checked
    def __init__(self, particle: Particle, /, *, tolerance: float = 1.0e-9) -> None:
        tolerance_ = float(tolerance)
        if not math.isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("Mass-shell tolerance must be finite and positive.")
        self.particle = particle
        self.tolerance = tolerance_
        self.shell_id = canonical_fingerprint(
            {
                "kind": "future-mass-shell",
                "particle_id": particle.particle_id,
                "tolerance": tolerance_,
            }
        )

    def from_spatial(self, spatial: ArrayLike, /) -> FourMomentum:
        momentum = jnp.asarray(spatial)
        if momentum.shape[-1:] != (3,):
            raise ValueError("Spatial momentum requires a trailing axis of length three.")
        energy = jnp.sqrt(jnp.sum(momentum * momentum, axis=-1) + self.particle.mass**2)
        return FourMomentum(jnp.concatenate((energy[..., None], momentum), axis=-1))

    def residual(self, momentum: FourMomentum | ArrayLike, /) -> Array:
        value = (
            momentum.value
            if isinstance(momentum, FourMomentum)
            else jnp.asarray(momentum)
        )
        return minkowski_dot(value, value) - self.particle.mass**2

    def contains(self, momentum: FourMomentum | ArrayLike, /) -> Array:
        value = (
            momentum.value
            if isinstance(momentum, FourMomentum)
            else jnp.asarray(momentum)
        )
        scale = jnp.maximum(1.0, jnp.abs(value[..., 0]) ** 2 + self.particle.mass**2)
        return (value[..., 0] >= 0.0) & (
            jnp.abs(self.residual(value)) <= self.tolerance * scale
        )


def center_of_momentum_frame(total: FourMomentum | ArrayLike, /) -> LorentzFrame:
    """Return the boost mapping a timelike total momentum to its rest frame."""
    value = (
        total.value
        if isinstance(total, FourMomentum)
        else np.asarray(total, dtype=np.float64)
    )
    value_ = np.asarray(value, dtype=np.float64)
    if value_.shape != (4,) or value_[0] <= 0.0:
        raise ValueError("A center-of-momentum frame requires one future four-momentum.")
    if float(value_[0] ** 2 - value_[1:] @ value_[1:]) <= 0.0:
        raise ValueError("Center-of-momentum frames require a timelike total momentum.")
    return LorentzFrame.boost(-value_[1:] / value_[0])


def mandelstam(
    incoming_one: FourMomentum | ArrayLike,
    incoming_two: FourMomentum | ArrayLike,
    outgoing_one: FourMomentum | ArrayLike,
    outgoing_two: FourMomentum | ArrayLike,
    /,
) -> tuple[Array, Array, Array]:
    """Return ``(s, t, u)`` for a two-to-two process."""
    p1 = (
        incoming_one.value
        if isinstance(incoming_one, FourMomentum)
        else jnp.asarray(incoming_one)
    )
    p2 = (
        incoming_two.value
        if isinstance(incoming_two, FourMomentum)
        else jnp.asarray(incoming_two)
    )
    p3 = (
        outgoing_one.value
        if isinstance(outgoing_one, FourMomentum)
        else jnp.asarray(outgoing_one)
    )
    p4 = (
        outgoing_two.value
        if isinstance(outgoing_two, FourMomentum)
        else jnp.asarray(outgoing_two)
    )
    return (
        minkowski_dot(p1 + p2, p1 + p2),
        minkowski_dot(p1 - p3, p1 - p3),
        minkowski_dot(p1 - p4, p1 - p4),
    )


__all__ = [
    "MassShell",
    "Particle",
    "center_of_momentum_frame",
    "mandelstam",
]
