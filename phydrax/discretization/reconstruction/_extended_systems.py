#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._high_resolution import HighResolutionReconstructionPlan


class MultispeciesEuler1DSystem(StrictModule, NonTrainableState):
    """Calorically perfect ideal-gas mixture with conservative species densities."""

    species_gammas: tuple[float, ...] = eqx.field(static=True)
    density_floor: float = eqx.field(static=True)
    pressure_floor: float = eqx.field(static=True)
    system_id: str = eqx.field(static=True)

    def __init__(
        self,
        species_gammas: Sequence[float],
        /,
        *,
        density_floor: float = 1e-12,
        pressure_floor: float = 1e-12,
    ):
        gammas = tuple(float(value) for value in species_gammas)
        if not gammas or any(not np.isfinite(value) or value <= 1.0 for value in gammas):
            raise ValueError("Every species gamma must be finite and greater than one.")
        self.species_gammas = gammas
        self.density_floor = float(density_floor)
        self.pressure_floor = float(pressure_floor)
        self.system_id = canonical_fingerprint(
            {
                "kind": "multispecies-euler-1d",
                "species_gammas": list(gammas),
                "density_floor": float(density_floor),
                "pressure_floor": float(pressure_floor),
            }
        )

    @property
    def species_count(self) -> int:
        return len(self.species_gammas)

    @property
    def component_count(self) -> int:
        return self.species_count + 2

    def density(self, state: Array, /) -> Array:
        return jnp.sum(state[..., : self.species_count], axis=-1)

    def mixture_gamma(self, state: Array, /) -> Array:
        species = state[..., : self.species_count]
        density = jnp.sum(species, axis=-1)
        fractions = species / density[..., None]
        heat_capacity = jnp.sum(
            fractions / (jnp.asarray(self.species_gammas, dtype=state.dtype) - 1.0),
            axis=-1,
        )
        return 1.0 + 1.0 / heat_capacity

    def pressure(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        density = self.density(value)
        momentum = value[..., -2]
        energy = value[..., -1]
        return (self.mixture_gamma(value) - 1.0) * (energy - 0.5 * momentum**2 / density)

    def flux(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        density = self.density(value)
        velocity = value[..., -2] / density
        pressure = self.pressure(value)
        species_flux = value[..., : self.species_count] * velocity[..., None]
        momentum_flux = value[..., -2] * velocity + pressure
        energy_flux = (value[..., -1] + pressure) * velocity
        return jnp.concatenate(
            (species_flux, momentum_flux[..., None], energy_flux[..., None]),
            axis=-1,
        )

    def wave_speed(self, left: Array, right: Array, /) -> Array:
        def speed(state):
            density = self.density(state)
            velocity = state[..., -2] / density
            sound = jnp.sqrt(self.mixture_gamma(state) * self.pressure(state) / density)
            return jnp.abs(velocity) + sound

        return jnp.maximum(speed(left), speed(right))

    def admissible(self, state: Array, /) -> Array:
        return jnp.all(
            state[..., : self.species_count] >= self.density_floor,
            axis=-1,
        ) & (self.pressure(state) >= self.pressure_floor)

    def limit(self, average: Array, face: Array, /, *, iterations: int = 32) -> Array:
        average = eqx.error_if(
            average,
            jnp.any(~self.admissible(average)),
            "Multispecies cell average is not admissible.",
        )
        direction = face - average

        def body(_, bounds):
            lower, upper = bounds
            midpoint = 0.5 * (lower + upper)
            valid = self.admissible(average + midpoint[..., None] * direction)
            return jnp.where(valid, midpoint, lower), jnp.where(valid, upper, midpoint)

        lower, _ = jax.lax.fori_loop(
            0,
            int(iterations),
            body,
            (jnp.zeros(average.shape[:-1]), jnp.ones(average.shape[:-1])),
        )
        return average + lower[..., None] * direction


class MultispeciesEuler1DDynamics(StrictModule):
    system: MultispeciesEuler1DSystem
    reconstruction: HighResolutionReconstructionPlan
    spacing: Array
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: MultispeciesEuler1DSystem,
        reconstruction: HighResolutionReconstructionPlan,
        spacing: ArrayLike,
        /,
    ):
        if not isinstance(system, MultispeciesEuler1DSystem) or not isinstance(
            reconstruction, HighResolutionReconstructionPlan
        ):
            raise TypeError("Multispecies dynamics requires system/reconstruction plans.")
        self.system = system
        self.reconstruction = reconstruction
        self.spacing = jnp.asarray(spacing)
        self.method_id = canonical_fingerprint(
            {
                "kind": "multispecies-euler-dynamics",
                "system": system.system_id,
                "reconstruction": reconstruction.plan_id,
                "spacing": float(self.spacing),
            }
        )

    def face_flux(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        left, right = self.reconstruction.reconstruct(value)
        adjacent = (
            jnp.roll(value, -1, axis=0)
            if self.reconstruction.boundary == "periodic"
            else jnp.concatenate((value[1:], value[-1:]), axis=0)
        )
        left = self.system.limit(value, left)
        right = self.system.limit(adjacent, right)
        speed = self.system.wave_speed(left, right)
        return 0.5 * (self.system.flux(left) + self.system.flux(right)) - 0.5 * speed[
            ..., None
        ] * (right - left)

    def __call__(self, time: Array, state: Array, args=None) -> Array:
        del time, args
        flux = self.face_flux(state)
        previous = (
            jnp.roll(flux, 1, axis=0)
            if self.reconstruction.boundary == "periodic"
            else jnp.concatenate((flux[:1], flux[:-1]), axis=0)
        )
        return -(flux - previous) / self.spacing


class IdealMHD1DSystem(StrictModule, NonTrainableState):
    """Eight-component ideal MHD with exact one-dimensional normal-field constraint."""

    gamma: float = eqx.field(static=True)
    system_id: str = eqx.field(static=True)

    def __init__(self, gamma: float = 1.4, /):
        gamma_ = float(gamma)
        if not np.isfinite(gamma_) or gamma_ <= 1.0:
            raise ValueError("MHD gamma must be finite and greater than one.")
        self.gamma = gamma_
        self.system_id = canonical_fingerprint({"kind": "ideal-mhd-1d", "gamma": gamma_})

    def pressure(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        density = value[..., 0]
        momentum_squared = jnp.sum(value[..., 1:4] ** 2, axis=-1)
        magnetic_squared = jnp.sum(value[..., 5:8] ** 2, axis=-1)
        return (self.gamma - 1.0) * (
            value[..., 4] - 0.5 * momentum_squared / density - 0.5 * magnetic_squared
        )

    def flux(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        density = value[..., 0]
        momentum = value[..., 1:4]
        energy = value[..., 4]
        magnetic = value[..., 5:8]
        velocity = momentum / density[..., None]
        pressure = self.pressure(value)
        magnetic_squared = jnp.sum(magnetic**2, axis=-1)
        total_pressure = pressure + 0.5 * magnetic_squared
        velocity_dot_magnetic = jnp.sum(velocity * magnetic, axis=-1)
        bx = magnetic[..., 0]
        momentum_flux = momentum * velocity[..., :1] - bx[..., None] * magnetic
        momentum_flux = momentum_flux.at[..., 0].add(total_pressure)
        energy_flux = (energy + total_pressure) * velocity[
            ..., 0
        ] - bx * velocity_dot_magnetic
        magnetic_flux = jnp.stack(
            (
                jnp.zeros_like(bx),
                magnetic[..., 1] * velocity[..., 0] - bx * velocity[..., 1],
                magnetic[..., 2] * velocity[..., 0] - bx * velocity[..., 2],
            ),
            axis=-1,
        )
        return jnp.concatenate(
            (
                momentum[..., :1],
                momentum_flux,
                energy_flux[..., None],
                magnetic_flux,
            ),
            axis=-1,
        )

    def wave_speed(self, left: Array, right: Array, /) -> Array:
        def speed(state):
            density = state[..., 0]
            velocity = state[..., 1] / density
            magnetic = state[..., 5:8]
            sound_squared = self.gamma * self.pressure(state) / density
            magnetic_squared = jnp.sum(magnetic**2, axis=-1) / density
            normal_squared = magnetic[..., 0] ** 2 / density
            discriminant = jnp.maximum(
                (sound_squared + magnetic_squared) ** 2
                - 4.0 * sound_squared * normal_squared,
                0.0,
            )
            fast = jnp.sqrt(
                0.5 * (sound_squared + magnetic_squared + jnp.sqrt(discriminant))
            )
            return jnp.abs(velocity) + fast

        return jnp.maximum(speed(left), speed(right))

    def admissible(self, state: Array, /) -> Array:
        return (state[..., 0] > 1e-12) & (self.pressure(state) > 1e-12)


class IdealMHD1DDynamics(StrictModule):
    system: IdealMHD1DSystem
    reconstruction: HighResolutionReconstructionPlan
    spacing: Array
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: IdealMHD1DSystem,
        reconstruction: HighResolutionReconstructionPlan,
        spacing: ArrayLike,
        /,
    ):
        self.system = system
        self.reconstruction = reconstruction
        self.spacing = jnp.asarray(spacing)
        self.method_id = canonical_fingerprint(
            {
                "kind": "ideal-mhd-1d-dynamics",
                "system": system.system_id,
                "reconstruction": reconstruction.plan_id,
                "spacing": float(self.spacing),
            }
        )

    def face_flux(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        left, right = self.reconstruction.reconstruct(value)
        speed = self.system.wave_speed(left, right)
        return 0.5 * (self.system.flux(left) + self.system.flux(right)) - 0.5 * speed[
            ..., None
        ] * (right - left)

    def __call__(self, time: Array, state: Array, args=None) -> Array:
        del time, args
        flux = self.face_flux(state)
        return -(flux - jnp.roll(flux, 1, axis=0)) / self.spacing


class UnsplitFluxDifferenceDynamics(StrictModule):
    """Unsplit multidimensional conservative flux divergence with shared face fluxes."""

    reconstructions: tuple[HighResolutionReconstructionPlan, ...]
    flux: Callable[[Array, int, Any], ArrayLike] = eqx.field(static=True)
    wave_speed: Callable[[Array, Array, int, Any], ArrayLike] = eqx.field(static=True)
    spacing: tuple[float, ...] = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        reconstructions: Sequence[HighResolutionReconstructionPlan],
        flux: Callable[[Array, int, Any], ArrayLike],
        wave_speed: Callable[[Array, Array, int, Any], ArrayLike],
        spacing: Sequence[float],
        /,
    ):
        reconstructions_ = tuple(reconstructions)
        spacing_ = tuple(float(value) for value in spacing)
        if (
            not reconstructions_
            or len(reconstructions_) != len(spacing_)
            or any(value <= 0.0 for value in spacing_)
            or not callable(flux)
            or not callable(wave_speed)
        ):
            raise ValueError(
                "Unsplit flux reconstruction, spacing, or callbacks are invalid."
            )
        self.reconstructions = reconstructions_
        self.flux = flux
        self.wave_speed = wave_speed
        self.spacing = spacing_
        self.method_id = canonical_fingerprint(
            {
                "kind": "unsplit-flux-difference",
                "reconstructions": [value.plan_id for value in reconstructions_],
                "spacing": list(spacing_),
                "flux": repr(flux),
                "wave_speed": repr(wave_speed),
            }
        )

    def _axis_flux(self, state: Array, axis: int, args: Any, /) -> Array:
        moved = jnp.moveaxis(state, axis, 0)
        spatial_tail = moved.shape[1:-1]
        lines = moved.reshape((moved.shape[0], -1, moved.shape[-1]))

        def reconstruct_line(line):
            return self.reconstructions[axis].reconstruct(line)

        left, right = jax.vmap(reconstruct_line, in_axes=1, out_axes=1)(lines)
        left_flux = jnp.asarray(self.flux(left, axis, args))
        right_flux = jnp.asarray(self.flux(right, axis, args))
        speed = jnp.asarray(self.wave_speed(left, right, axis, args))
        numerical = 0.5 * (left_flux + right_flux) - 0.5 * speed[..., None] * (
            right - left
        )
        numerical = numerical.reshape(
            (moved.shape[0],) + spatial_tail + (moved.shape[-1],)
        )
        return jnp.moveaxis(numerical, 0, axis)

    def __call__(self, time: Array, state: Array, args: Any = None) -> Array:
        del time
        value = jnp.asarray(state)
        if value.ndim != len(self.spacing) + 1:
            raise ValueError(
                "Unsplit state requires spatial axes plus one component axis."
            )
        result = jnp.zeros_like(value)
        for axis, spacing in enumerate(self.spacing):
            face_flux = self._axis_flux(value, axis, args)
            result = result - (face_flux - jnp.roll(face_flux, 1, axis=axis)) / spacing
        return result


__all__ = [
    "IdealMHD1DDynamics",
    "IdealMHD1DSystem",
    "MultispeciesEuler1DDynamics",
    "MultispeciesEuler1DSystem",
    "UnsplitFluxDifferenceDynamics",
]
