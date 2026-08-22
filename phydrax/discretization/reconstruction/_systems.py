#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._high_resolution import (
    CharacteristicReconstructionPlan,
    CharacteristicSystem,
    HighResolutionReconstructionPlan,
    ReconstructionBoundary,
)


EulerFluxKind: TypeAlias = Literal["rusanov"]


class PositivityLimiterPlan(StrictModule, NonTrainableState):
    """Zhang-Shu line limiter from cell averages to admissible Euler face states."""

    gamma: float = eqx.field(static=True)
    density_floor: float = eqx.field(static=True)
    pressure_floor: float = eqx.field(static=True)
    iterations: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        gamma: float = 1.4,
        density_floor: float = 1e-12,
        pressure_floor: float = 1e-12,
        iterations: int = 32,
    ):
        gamma_ = float(gamma)
        density = float(density_floor)
        pressure = float(pressure_floor)
        iterations_ = int(iterations)
        if (
            not np.isfinite(gamma_)
            or gamma_ <= 1.0
            or density <= 0.0
            or pressure <= 0.0
            or iterations_ <= 0
        ):
            raise ValueError("Positivity limiter controls are invalid.")
        self.gamma = gamma_
        self.density_floor = density
        self.pressure_floor = pressure
        self.iterations = iterations_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "euler-positivity-limiter",
                "gamma": gamma_,
                "density_floor": density,
                "pressure_floor": pressure,
                "iterations": iterations_,
            }
        )

    def pressure(self, state: Array, /) -> Array:
        density = state[..., 0]
        momentum = state[..., 1]
        energy = state[..., 2]
        return (self.gamma - 1.0) * (energy - 0.5 * momentum**2 / density)

    def admissible(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        return (value[..., 0] >= self.density_floor) & (
            self.pressure(value) >= self.pressure_floor
        )

    def limit(self, cell_average: ArrayLike, face_state: ArrayLike, /) -> Array:
        cell = jnp.asarray(cell_average)
        face = jnp.asarray(face_state)
        if cell.shape != face.shape or cell.shape[-1] != 3:
            raise ValueError(
                "Euler positivity limiter requires aligned three-component states."
            )
        cell = eqx.error_if(
            cell,
            jnp.any(~self.admissible(cell)),
            "Euler cell average is not admissible before face limiting.",
        )
        density_delta = face[..., 0] - cell[..., 0]
        density_theta = jnp.where(
            density_delta < 0.0,
            (cell[..., 0] - self.density_floor) / (-density_delta),
            1.0,
        )
        upper = jnp.clip(density_theta, 0.0, 1.0)
        lower = jnp.zeros_like(upper)
        direction = face - cell

        def body(_, bounds):
            lower_, upper_ = bounds
            midpoint = 0.5 * (lower_ + upper_)
            candidate = cell + midpoint[..., None] * direction
            valid = self.admissible(candidate)
            return (
                jnp.where(valid, midpoint, lower_),
                jnp.where(valid, upper_, midpoint),
            )

        lower, upper = jax.lax.fori_loop(
            0,
            self.iterations,
            body,
            (lower, upper),
        )
        return cell + lower[..., None] * direction


class Euler1DSystem(StrictModule, NonTrainableState):
    """Ideal-gas one-dimensional Euler flux, Roe eigensystem, and entropy variables."""

    gamma: float = eqx.field(static=True)
    system_id: str = eqx.field(static=True)

    def __init__(self, gamma: float = 1.4, /):
        gamma_ = float(gamma)
        if not np.isfinite(gamma_) or gamma_ <= 1.0:
            raise ValueError("Euler gamma must be finite and greater than one.")
        self.gamma = gamma_
        self.system_id = canonical_fingerprint(
            {"kind": "euler-1d-system", "gamma": gamma_}
        )

    def pressure(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        density, momentum, energy = value[..., 0], value[..., 1], value[..., 2]
        return (self.gamma - 1.0) * (energy - 0.5 * momentum**2 / density)

    def primitive(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        density = value[..., 0]
        velocity = value[..., 1] / density
        return jnp.stack((density, velocity, self.pressure(value)), axis=-1)

    def conservative(self, primitive: ArrayLike, /) -> Array:
        value = jnp.asarray(primitive)
        density, velocity, pressure = (
            value[..., 0],
            value[..., 1],
            value[..., 2],
        )
        energy = pressure / (self.gamma - 1.0) + 0.5 * density * velocity**2
        return jnp.stack((density, density * velocity, energy), axis=-1)

    def flux(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        density = value[..., 0]
        momentum = value[..., 1]
        energy = value[..., 2]
        velocity = momentum / density
        pressure = self.pressure(value)
        return jnp.stack(
            (momentum, momentum * velocity + pressure, (energy + pressure) * velocity),
            axis=-1,
        )

    def wave_speed(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        left_primitive = self.primitive(left)
        right_primitive = self.primitive(right)
        left_speed = jnp.abs(left_primitive[..., 1]) + jnp.sqrt(
            self.gamma * left_primitive[..., 2] / left_primitive[..., 0]
        )
        right_speed = jnp.abs(right_primitive[..., 1]) + jnp.sqrt(
            self.gamma * right_primitive[..., 2] / right_primitive[..., 0]
        )
        return jnp.maximum(left_speed, right_speed)

    def eigensystem(
        self,
        left: Array,
        right: Array,
        args=None,
    ) -> tuple[Array, Array, Array]:
        del args
        left_primitive = self.primitive(left)
        right_primitive = self.primitive(right)
        left_root = jnp.sqrt(left_primitive[..., 0])
        right_root = jnp.sqrt(right_primitive[..., 0])
        denominator = left_root + right_root
        velocity = (
            left_root * left_primitive[..., 1] + right_root * right_primitive[..., 1]
        ) / denominator
        left_enthalpy = (left[..., 2] + left_primitive[..., 2]) / left_primitive[..., 0]
        right_enthalpy = (right[..., 2] + right_primitive[..., 2]) / right_primitive[
            ..., 0
        ]
        enthalpy = (left_root * left_enthalpy + right_root * right_enthalpy) / denominator
        sound_speed = jnp.sqrt(
            jnp.maximum(
                (self.gamma - 1.0) * (enthalpy - 0.5 * velocity**2),
                jnp.finfo(left.dtype).tiny,
            )
        )
        right_matrix = jnp.stack(
            (
                jnp.stack(
                    (
                        jnp.ones_like(velocity),
                        velocity - sound_speed,
                        enthalpy - velocity * sound_speed,
                    ),
                    axis=-1,
                ),
                jnp.stack(
                    (jnp.ones_like(velocity), velocity, 0.5 * velocity**2),
                    axis=-1,
                ),
                jnp.stack(
                    (
                        jnp.ones_like(velocity),
                        velocity + sound_speed,
                        enthalpy + velocity * sound_speed,
                    ),
                    axis=-1,
                ),
            ),
            axis=-1,
        )
        left_matrix = jnp.linalg.inv(right_matrix)
        eigenvalues = jnp.stack(
            (velocity - sound_speed, velocity, velocity + sound_speed),
            axis=-1,
        )
        return left_matrix, right_matrix, eigenvalues

    def entropy_variables(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        density = value[..., 0]
        velocity = value[..., 1] / density
        pressure = self.pressure(value)
        entropy = jnp.log(pressure) - self.gamma * jnp.log(density)
        beta = density / (2.0 * pressure)
        return jnp.stack(
            (
                (self.gamma - entropy) / (self.gamma - 1.0) - beta * velocity**2,
                2.0 * beta * velocity,
                -2.0 * beta,
            ),
            axis=-1,
        )

    def characteristic_system(self, /) -> CharacteristicSystem:
        return CharacteristicSystem(self.eigensystem, system_id=self.system_id)


class EntropyStableEulerFlux(StrictModule, NonTrainableState):
    """Rusanov entropy-dissipative Euler flux with production diagnostics."""

    system: Euler1DSystem
    flux_kind: EulerFluxKind = eqx.field(static=True)
    flux_id: str = eqx.field(static=True)

    def __init__(self, system: Euler1DSystem, /, *, flux_kind: EulerFluxKind = "rusanov"):
        if not isinstance(system, Euler1DSystem) or flux_kind != "rusanov":
            raise ValueError(
                "Entropy-stable Euler flux requires Euler system/Rusanov kind."
            )
        self.system = system
        self.flux_kind = flux_kind
        self.flux_id = canonical_fingerprint(
            {
                "kind": "entropy-stable-euler-flux",
                "system": system.system_id,
                "flux_kind": flux_kind,
            }
        )

    def face_flux(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        left_ = jnp.asarray(left)
        right_ = jnp.asarray(right)
        speed = self.system.wave_speed(left_, right_)
        return 0.5 * (self.system.flux(left_) + self.system.flux(right_)) - 0.5 * speed[
            ..., None
        ] * (right_ - left_)

    def entropy_dissipation(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        left_ = jnp.asarray(left)
        right_ = jnp.asarray(right)
        speed = self.system.wave_speed(left_, right_)
        entropy_jump = self.system.entropy_variables(
            right_
        ) - self.system.entropy_variables(left_)
        state_jump = right_ - left_
        return -0.5 * speed * jnp.sum(entropy_jump * state_jump, axis=-1)


class Euler1DDynamics(StrictModule):
    """Characteristic high-resolution Euler flux difference with SSPRK3."""

    system: Euler1DSystem
    reconstruction: CharacteristicReconstructionPlan
    limiter: PositivityLimiterPlan
    numerical_flux: EntropyStableEulerFlux
    spacing: Array
    boundary: ReconstructionBoundary = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: Euler1DSystem,
        reconstruction: HighResolutionReconstructionPlan,
        spacing: ArrayLike,
        /,
        *,
        limiter: PositivityLimiterPlan | None = None,
    ):
        if not isinstance(system, Euler1DSystem) or not isinstance(
            reconstruction, HighResolutionReconstructionPlan
        ):
            raise TypeError("Euler dynamics requires system and reconstruction plans.")
        spacing_ = jnp.asarray(spacing)
        if (
            spacing_.shape != ()
            or not bool(np.isfinite(np.asarray(spacing_)))
            or float(spacing_) <= 0.0
        ):
            raise ValueError("Euler spacing must be one finite positive scalar.")
        limiter_ = (
            PositivityLimiterPlan(gamma=system.gamma) if limiter is None else limiter
        )
        if (
            not isinstance(limiter_, PositivityLimiterPlan)
            or limiter_.gamma != system.gamma
        ):
            raise ValueError("Euler limiter must use the system gamma.")
        characteristic = CharacteristicReconstructionPlan(
            reconstruction,
            system.characteristic_system(),
        )
        numerical_flux = EntropyStableEulerFlux(system)
        self.system = system
        self.reconstruction = characteristic
        self.limiter = limiter_
        self.numerical_flux = numerical_flux
        self.spacing = spacing_
        self.boundary = reconstruction.boundary
        self.method_id = canonical_fingerprint(
            {
                "kind": "euler-1d-dynamics",
                "system": system.system_id,
                "reconstruction": characteristic.plan_id,
                "limiter": limiter_.plan_id,
                "flux": numerical_flux.flux_id,
                "spacing": float(spacing_),
            }
        )

    def face_states(self, state: ArrayLike, /) -> tuple[Array, Array, Array]:
        value = jnp.asarray(state)
        left, right, wave_speeds = self.reconstruction.reconstruct(value)
        left_average = value
        right_average = (
            jnp.roll(value, -1, axis=0)
            if self.boundary == "periodic"
            else jnp.concatenate((value[1:], value[-1:]), axis=0)
        )
        return (
            self.limiter.limit(left_average, left),
            self.limiter.limit(right_average, right),
            wave_speeds,
        )

    def face_flux(self, state: ArrayLike, /) -> Array:
        left, right, _ = self.face_states(state)
        return self.numerical_flux.face_flux(left, right)

    def __call__(self, time: Array, state: Array, args=None) -> Array:
        del time, args
        value = jnp.asarray(state)
        flux = self.face_flux(value)
        previous = (
            jnp.roll(flux, 1, axis=0)
            if self.boundary == "periodic"
            else jnp.concatenate((flux[:1], flux[:-1]), axis=0)
        )
        return -(flux - previous) / self.spacing

    def stable_step(self, state: ArrayLike, cfl: float = 0.4, /) -> Array:
        left, right, _ = self.face_states(state)
        speed = jnp.max(self.system.wave_speed(left, right))
        return (
            float(cfl)
            * self.spacing
            / jnp.maximum(
                speed,
                jnp.finfo(jnp.asarray(state).dtype).tiny,
            )
        )

    def _validated(self, state: Array, /) -> Array:
        return eqx.error_if(
            state,
            jnp.any(~self.limiter.admissible(state)) | jnp.any(~jnp.isfinite(state)),
            "Euler SSPRK stage left the admissible state set.",
        )

    def ssprk3_step(
        self,
        time: Array,
        state: Array,
        step_size: ArrayLike,
        args=None,
    ) -> Array:
        dt = jnp.asarray(step_size)
        first = self._validated(state + dt * self(time, state, args))
        second = self._validated(
            0.75 * state + 0.25 * (first + dt * self(jnp.asarray(time) + dt, first, args))
        )
        return self._validated(
            (1.0 / 3.0) * state
            + (2.0 / 3.0)
            * (second + dt * self(jnp.asarray(time) + 0.5 * dt, second, args))
        )


__all__ = [
    "EntropyStableEulerFlux",
    "Euler1DDynamics",
    "Euler1DSystem",
    "EulerFluxKind",
    "PositivityLimiterPlan",
]
