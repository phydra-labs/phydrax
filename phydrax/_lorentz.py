#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Core special relativity in the mostly-minus ``(+---)`` signature.

Four-vectors carry the time component first. ``boost_matrix(β)`` is the passive
standard boost into a frame moving with dimensionless velocity ``β`` relative to
the source frame, ``x'^μ = Λ^μ_ν(β) x^ν`` with ``t' = γ(t − β·x)``. Every traced
operation accepts batched ``β[..., 3]`` and broadcasts against its operands; a
superluminal ``β`` produces non-finite values, which consumers observe through
their own finiteness evidence.
"""

from __future__ import annotations

import math
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ._strict import StrictModule
from ._trainable import NonTrainableState
from .typing import parse


type SpectralEmissionCompleteness = Literal["complete", "truncated"]

MINKOWSKI_METRIC = jnp.diag(jnp.asarray([1.0, -1.0, -1.0, -1.0], dtype=jnp.float64))


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


def _velocity(beta: ArrayLike, /) -> Array:
    velocity = jnp.asarray(beta)
    if velocity.shape[-1:] != (3,):
        raise ValueError("Boost velocities require a trailing axis of length three.")
    return velocity


def boost_matrix(beta: ArrayLike, /) -> Array:
    """Return the passive boost ``Λ(β)[..., 4, 4]`` into a frame moving with ``β``."""
    velocity = _velocity(beta)
    speed_squared = jnp.sum(velocity * velocity, axis=-1)
    gamma = 1.0 / jnp.sqrt(1.0 - speed_squared)
    # (γ − 1)/β² = γ²/(γ + 1): regular, and differentiable, at β = 0.
    coefficient = gamma * gamma / (1.0 + gamma)
    spatial = (
        jnp.eye(3, dtype=velocity.dtype)
        + coefficient[..., None, None] * velocity[..., :, None] * velocity[..., None, :]
    )
    first_row = jnp.concatenate((gamma[..., None], -gamma[..., None] * velocity), axis=-1)
    remaining = jnp.concatenate(
        (-gamma[..., None, None] * velocity[..., :, None], spatial), axis=-1
    )
    return jnp.concatenate((first_row[..., None, :], remaining), axis=-2)


def boost_event(beta: ArrayLike, event: ArrayLike, /) -> Array:
    """Transform contravariant four-vectors ``(ct, x)[..., 4]`` into the frame ``β``."""
    vector = jnp.asarray(event)
    if vector.shape[-1:] != (4,):
        raise ValueError("Boosted events require a trailing axis of length four.")
    return (boost_matrix(beta) @ vector[..., None])[..., 0]


def boost_proper_velocity(beta: ArrayLike, proper_velocity: ArrayLike, /) -> Array:
    """Transform spatial proper velocities ``u = γv/c`` into the frame ``β``.

    ``u`` is dimensionless (in units of ``c``); the time component
    ``γ = √(1 + |u|²)`` is reconstructed on shell, so the result stays on shell.
    """
    velocity = jnp.asarray(proper_velocity)
    if velocity.shape[-1:] != (3,):
        raise ValueError("Proper velocities require a trailing axis of length three.")
    gamma = jnp.sqrt(1.0 + jnp.sum(velocity * velocity, axis=-1))
    four_velocity = jnp.concatenate((gamma[..., None], velocity), axis=-1)
    return boost_event(beta, four_velocity)[..., 1:]


def _field_tensor(electric: Array, magnetic: Array, speed_of_light: Array) -> Array:
    # F^{i0} = E_i / c and F^{ij} = −ε_ijk B_k in the (+---) signature.
    e = electric / speed_of_light[..., None]
    ex, ey, ez = e[..., 0], e[..., 1], e[..., 2]
    bx, by, bz = magnetic[..., 0], magnetic[..., 1], magnetic[..., 2]
    zero = jnp.zeros_like(ex)
    rows = (
        (zero, -ex, -ey, -ez),
        (ex, zero, -bz, by),
        (ey, bz, zero, -bx),
        (ez, -by, bx, zero),
    )
    return jnp.stack([jnp.stack(row, axis=-1) for row in rows], axis=-2)


def boost_fields(
    beta: ArrayLike,
    electric: ArrayLike,
    magnetic: ArrayLike,
    /,
    *,
    speed_of_light: ArrayLike,
) -> tuple[Array, Array]:
    """Transform ``(E, B)[..., 3]`` into the frame ``β`` through ``F' = Λ F Λᵀ``.

    ``speed_of_light`` is expressed in the units relating ``E`` and ``B``
    (``[E] = [c][B]``); both fields keep their input units.
    """
    e = jnp.asarray(electric)
    b = jnp.asarray(magnetic)
    c = jnp.asarray(speed_of_light)
    if e.shape[-1:] != (3,) or b.shape[-1:] != (3,):
        raise ValueError(
            "Electric and magnetic fields require trailing axes of length three."
        )
    if c.ndim != 0:
        raise ValueError("speed_of_light must be a scalar.")
    transform = boost_matrix(beta)
    tensor = transform @ _field_tensor(e, b, c) @ jnp.swapaxes(transform, -1, -2)
    boosted_electric = c * tensor[..., 1:, 0]
    boosted_magnetic = jnp.stack(
        (-tensor[..., 2, 3], tensor[..., 1, 3], -tensor[..., 1, 2]), axis=-1
    )
    return boosted_electric, boosted_magnetic


def boost_wavevector(
    beta: ArrayLike, angular_frequency: ArrayLike, direction: ArrayLike, /
) -> tuple[Array, Array]:
    """Doppler-shift and aberrate vacuum plane waves into the frame ``β``.

    Returns ``(ω', n')`` with ``ω' = γω(1 − β·n)`` and the unit propagation
    direction ``n'`` of the null wave four-vector ``k^μ = (ω/c)(1, n)``.
    """
    frequency = jnp.asarray(angular_frequency)
    unit = jnp.asarray(direction)
    if unit.shape[-1:] != (3,):
        raise ValueError("Wave directions require a trailing axis of length three.")
    null = jnp.concatenate((jnp.ones_like(unit[..., :1]), unit), axis=-1)
    boosted = boost_event(beta, null)
    doppler = boosted[..., 0]
    return frequency * doppler, boosted[..., 1:] / doppler[..., None]


class LorentzSpectralTransform(StrictModule, NonTrainableState):
    """Vacuum spectral energy density ``d²W/(dω dΩ)`` relabeled into a boosted frame.

    Each source sample ``(ω, n, S)`` maps pointwise to ``(ω', n', S')`` with
    ``S' = D² S`` and ``D = ω'/ω``, the invariance of ``(1/ω²) d²W/(dω dΩ)``.
    Samples move; consumers re-grid in the target frame. ``valid`` marks samples
    with finite values, subluminal ``β``, unit ``n``, positive ``ω`` and ``D``.
    """

    angular_frequencies: Array
    directions: Array
    spectral_energy: Array
    doppler_factor: Array
    valid: Array
    emission: SpectralEmissionCompleteness = eqx.field(static=True)
    refractive_index: float = eqx.field(static=True)


def transform_spectral_energy(
    beta: ArrayLike,
    angular_frequencies: ArrayLike,
    directions: ArrayLike,
    spectral_energy: ArrayLike,
    /,
    *,
    emission: SpectralEmissionCompleteness,
    refractive_index: float = 1.0,
) -> LorentzSpectralTransform:
    """Transform a one-sided vacuum spectral energy density into the frame ``β``.

    The invariant ``(1/ω²) d²W/(dω dΩ)`` is photon-number phase-space density;
    it transforms only for complete emission (the whole emission event is inside
    the spectrum) in vacuum. Truncated time windows and media with
    ``refractive_index != 1`` are refused.
    """
    completeness = parse(emission, SpectralEmissionCompleteness, "emission")
    index = float(refractive_index)
    if not math.isfinite(index) or index <= 0.0:
        raise ValueError("refractive_index must be finite and positive.")
    if index != 1.0:
        raise ValueError(
            "Lorentz spectral-energy transforms are valid only in vacuum "
            "(refractive_index == 1); media are refused."
        )
    match completeness:
        case "complete":
            pass
        case "truncated":
            raise ValueError(
                "Lorentz spectral-energy transforms require complete emission; "
                "a truncated emission window is not frame-covariant."
            )
    velocity = _velocity(beta)
    frequency = jnp.asarray(angular_frequencies)
    unit = jnp.asarray(directions)
    energy = jnp.asarray(spectral_energy)
    boosted_frequency, boosted_direction = boost_wavevector(velocity, frequency, unit)
    doppler = boosted_frequency / frequency
    boosted_energy = doppler * doppler * energy
    unit_error = jnp.abs(jnp.sum(unit * unit, axis=-1) - 1.0)
    tolerance = 64.0 * jnp.finfo(unit.dtype).eps
    valid = (
        jnp.isfinite(boosted_frequency)
        & jnp.all(jnp.isfinite(boosted_direction), axis=-1)
        & jnp.isfinite(boosted_energy)
        & (jnp.sum(velocity * velocity, axis=-1) < 1.0)
        & (unit_error <= tolerance)
        & (frequency > 0.0)
        & (doppler > 0.0)
    )
    return LorentzSpectralTransform(
        angular_frequencies=boosted_frequency,
        directions=boosted_direction,
        spectral_energy=boosted_energy,
        doppler_factor=doppler,
        valid=valid,
        emission=completeness,
        refractive_index=index,
    )


class FourMomentum(StrictModule):
    """Contravariant four-momentum with time component first."""

    value: Array

    def __init__(self, value: ArrayLike, /) -> None:
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

    def __init__(self, matrix: ArrayLike, /) -> None:
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
        # The active boost toward ``velocity`` is the passive boost into ``-velocity``.
        return cls(np.asarray(boost_matrix(jnp.asarray(-beta, dtype=jnp.float64))))

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


__all__ = [
    "FourMomentum",
    "LorentzFrame",
    "LorentzSpectralTransform",
    "MINKOWSKI_METRIC",
    "SpectralEmissionCompleteness",
    "boost_event",
    "boost_fields",
    "boost_matrix",
    "boost_proper_velocity",
    "boost_wavevector",
    "lower_four_vector",
    "minkowski_dot",
    "transform_spectral_energy",
]
