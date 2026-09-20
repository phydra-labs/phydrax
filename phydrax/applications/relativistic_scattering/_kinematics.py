#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Mostly-minus Lorentz kinematics and on-shell particle contracts."""

from __future__ import annotations

import math
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


MINKOWSKI_METRIC = jnp.diag(jnp.asarray([1.0, -1.0, -1.0, -1.0]))


def minkowski_dot(left: ArrayLike, right: ArrayLike, /) -> Array:
    """Contract four-vectors with metric signature ``(+---)``."""
    left_ = jnp.asarray(left)
    right_ = jnp.asarray(right)
    if left_.shape[-1:] != (4,) or right_.shape[-1:] != (4,):
        raise ValueError("Minkowski products require trailing four-vector axes.")
    return left_[..., 0] * right_[..., 0] - jnp.sum(
        left_[..., 1:] * right_[..., 1:], axis=-1
    )


def lower_four_vector(value: ArrayLike, /) -> Array:
    """Lower one mostly-minus Lorentz index."""
    vector = jnp.asarray(value)
    if vector.shape[-1:] != (4,):
        raise ValueError(
            "Lorentz index lowering requires a trailing axis of length four."
        )
    return vector * jnp.asarray([1.0, -1.0, -1.0, -1.0], dtype=vector.dtype)


class FourMomentum(StrictModule):
    """Contravariant four-momentum with time component first."""

    value: Array

    def __init__(self, value: ArrayLike, /):
        value_ = jnp.asarray(value)
        if value_.shape[-1:] != (4,):
            raise ValueError("FourMomentum requires a trailing axis of length four.")
        self.value = value_

    @property
    def energy(self) -> Array:
        return self.value[..., 0]

    @property
    def spatial(self) -> Array:
        return self.value[..., 1:]

    @property
    def invariant_mass_squared(self) -> Array:
        return minkowski_dot(self.value, self.value)

    @property
    def invariant_mass(self) -> Array:
        return jnp.sqrt(jnp.maximum(self.invariant_mass_squared, 0.0))

    def dot(self, other: "FourMomentum", /) -> Array:
        return minkowski_dot(self.value, other.value)

    def __add__(self, other: "FourMomentum", /) -> "FourMomentum":
        return FourMomentum(self.value + other.value)

    def __sub__(self, other: "FourMomentum", /) -> "FourMomentum":
        return FourMomentum(self.value - other.value)

    def __neg__(self) -> "FourMomentum":
        return FourMomentum(-self.value)


class LorentzFrame(StrictModule, NonTrainableState):
    """Proper orthochronous Lorentz transformation with validation evidence."""

    matrix: Array
    metric_residual: Array
    determinant: Array
    valid: Array
    frame_id: str = eqx.field(static=True)

    def __init__(self, matrix: ArrayLike, /):
        matrix_ = np.asarray(matrix, dtype=np.float64)
        if matrix_.shape != (4, 4) or np.any(~np.isfinite(matrix_)):
            raise ValueError("Lorentz frames require one finite 4x4 matrix.")
        metric = np.diag([1.0, -1.0, -1.0, -1.0])
        residual = matrix_.T @ metric @ matrix_ - metric
        determinant = float(np.linalg.det(matrix_))
        valid = (
            float(np.max(np.abs(residual))) <= 1.0e-10
            and abs(determinant - 1.0) <= 1.0e-10
            and matrix_[0, 0] >= 1.0
        )
        if not valid:
            raise ValueError("matrix is not a proper orthochronous Lorentz transform.")
        self.matrix = jnp.asarray(matrix_)
        self.metric_residual = jnp.asarray(residual)
        self.determinant = jnp.asarray(determinant)
        self.valid = jnp.asarray(valid)
        self.frame_id = canonical_fingerprint(
            {"kind": "lorentz-frame", "matrix": array_tree_fingerprint(matrix_)}
        )

    @classmethod
    def identity(cls) -> "LorentzFrame":
        return cls(np.eye(4))

    @classmethod
    def boost(cls, velocity: ArrayLike, /) -> "LorentzFrame":
        """Return the active boost carrying rest momentum toward ``velocity``."""
        beta = np.asarray(velocity, dtype=np.float64)
        if beta.shape != (3,) or np.any(~np.isfinite(beta)):
            raise ValueError("Boost velocity must be one finite three-vector.")
        speed_squared = float(beta @ beta)
        if speed_squared >= 1.0:
            raise ValueError("Lorentz boost speed must be strictly below light speed.")
        if speed_squared == 0.0:
            return cls.identity()
        gamma = 1.0 / math.sqrt(1.0 - speed_squared)
        spatial = np.eye(3) + (gamma - 1.0) * np.outer(beta, beta) / speed_squared
        matrix = np.empty((4, 4), dtype=np.float64)
        matrix[0, 0] = gamma
        matrix[0, 1:] = gamma * beta
        matrix[1:, 0] = gamma * beta
        matrix[1:, 1:] = spatial
        return cls(matrix)

    def apply(self, momentum: FourMomentum | ArrayLike, /) -> FourMomentum:
        value = (
            momentum.value
            if isinstance(momentum, FourMomentum)
            else jnp.asarray(momentum)
        )
        if value.shape[-1:] != (4,):
            raise ValueError("Lorentz transforms require trailing four-momentum axes.")
        return FourMomentum(jnp.matmul(value, self.matrix.T))

    def inverse(self, /) -> "LorentzFrame":
        metric = np.diag([1.0, -1.0, -1.0, -1.0])
        return LorentzFrame(metric @ np.asarray(self.matrix).T @ metric)

    def compose(self, other: "LorentzFrame", /) -> "LorentzFrame":
        return LorentzFrame(np.asarray(self.matrix) @ np.asarray(other.matrix))


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
    ):
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

    def __init__(self, particle: Particle, /, *, tolerance: float = 1.0e-9):
        tolerance_ = float(tolerance)
        if not isinstance(particle, Particle):
            raise TypeError("MassShell requires a Particle.")
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
    "FourMomentum",
    "LorentzFrame",
    "MINKOWSKI_METRIC",
    "MassShell",
    "Particle",
    "center_of_momentum_frame",
    "lower_four_vector",
    "mandelstam",
    "minkowski_dot",
]
