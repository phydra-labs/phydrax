#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...equations._hyperbolic_systems import (
    CompressibleNavierStokesSystem,
    EulerSystem,
)


CompressibleWaveKind: TypeAlias = Literal[
    "isentropic", "acoustic", "entropy", "vorticity"
]


class CompressibleReferenceWaveEvidence(StrictModule):
    primitive: Array
    conserved: Array
    pressure_relation_residual: Array
    transverse_velocity_residual: Array
    admissible: Array
    finite: Array
    wave_id: str = eqx.field(static=True)


class CompressibleReferenceWavePlan(StrictModule, NonTrainableState):
    """Deterministic smooth Euler waves for route-by-route qualification."""

    mean_velocity: tuple[float, ...] = eqx.field(static=True)
    wave_vector: tuple[float, ...] = eqx.field(static=True)
    polarization: tuple[float, ...] = eqx.field(static=True)
    kind: CompressibleWaveKind = eqx.field(static=True)
    base_density: float = eqx.field(static=True)
    base_pressure: float = eqx.field(static=True)
    amplitude: float = eqx.field(static=True)
    propagation_sign: int = eqx.field(static=True)
    wave_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: CompressibleWaveKind,
        mean_velocity: Sequence[float],
        wave_vector: Sequence[float],
        /,
        *,
        base_density: float = 1.0,
        base_pressure: float = 1.0,
        amplitude: float = 1.0e-3,
        propagation_sign: int = 1,
        polarization: Sequence[float] | None = None,
    ):
        velocity = tuple(float(value) for value in mean_velocity)
        wave = tuple(float(value) for value in wave_vector)
        density = float(base_density)
        pressure = float(base_pressure)
        amplitude_ = float(amplitude)
        sign = int(propagation_sign)
        if polarization is None:
            polarization_ = (0.0,) * len(velocity)
            if len(velocity) >= 2:
                direction = np.asarray(wave, dtype=float)
                direction = direction / np.linalg.norm(direction)
                candidate = np.zeros(len(velocity), dtype=float)
                candidate[int(np.argmin(np.abs(direction)))] = 1.0
                candidate = candidate - np.dot(candidate, direction) * direction
                candidate = candidate / np.linalg.norm(candidate)
                polarization_ = tuple(float(value) for value in candidate)
        else:
            polarization_ = tuple(float(value) for value in polarization)
        if (
            kind not in ("isentropic", "acoustic", "entropy", "vorticity")
            or len(velocity) not in (1, 2, 3)
            or len(wave) != len(velocity)
            or len(polarization_) != len(velocity)
            or any(not np.isfinite(value) for value in (*velocity, *wave, *polarization_))
            or np.linalg.norm(np.asarray(wave)) <= 0.0
            or density <= 0.0
            or pressure <= 0.0
            or not np.isfinite(amplitude_)
            or amplitude_ <= 0.0
            or amplitude_ >= density
            or sign not in (-1, 1)
        ):
            raise ValueError("Compressible reference-wave parameters are invalid.")
        direction = np.asarray(wave, dtype=float)
        direction = direction / np.linalg.norm(direction)
        if kind == "vorticity":
            polarization_array = np.asarray(polarization_, dtype=float)
            if (
                np.linalg.norm(polarization_array) <= 0.0
                or abs(np.dot(direction, polarization_array))
                > 128.0 * np.finfo(float).eps
            ):
                raise ValueError("Vorticity polarization must be nonzero and transverse.")
            polarization_array = polarization_array / np.linalg.norm(polarization_array)
            polarization_ = tuple(float(value) for value in polarization_array)
        self.kind = kind
        self.mean_velocity = velocity
        self.wave_vector = wave
        self.base_density = density
        self.base_pressure = pressure
        self.amplitude = amplitude_
        self.propagation_sign = sign
        self.polarization = polarization_
        self.wave_id = canonical_fingerprint(
            {
                "kind": "compressible-reference-wave",
                "wave_kind": kind,
                "mean_velocity": velocity,
                "wave_vector": wave,
                "base_density": density,
                "base_pressure": pressure,
                "amplitude": amplitude_,
                "propagation_sign": sign,
                "polarization": polarization_,
            }
        )

    @property
    def dimension(self) -> int:
        return len(self.mean_velocity)

    def primitive(
        self,
        system: EulerSystem,
        coordinates: ArrayLike,
        time: ArrayLike,
        /,
    ) -> Array:
        if not isinstance(system, EulerSystem) or system.dimension != self.dimension:
            raise TypeError("Reference waves require a matching EulerSystem.")
        points = jnp.asarray(coordinates)
        if points.shape[-1:] != (self.dimension,):
            raise ValueError("Reference-wave coordinates have the wrong dimension.")
        time_ = jnp.asarray(time, dtype=points.dtype)
        wave = jnp.asarray(self.wave_vector, dtype=points.dtype)
        mean_velocity = jnp.asarray(self.mean_velocity, dtype=points.dtype)
        wave_norm = jnp.sqrt(oe.contract("d,d->", wave, wave, backend="jax"))
        direction = wave / wave_norm
        sound = jnp.sqrt(system.gamma * self.base_pressure / self.base_density)
        convection_frequency = oe.contract("d,d->", mean_velocity, wave, backend="jax")
        frequency = convection_frequency
        if self.kind in ("isentropic", "acoustic"):
            frequency = frequency + self.propagation_sign * sound * wave_norm
        phase = (
            oe.contract("...d,d->...", points, wave, backend="jax") - frequency * time_
        )
        oscillation = jnp.cos(phase)
        density = self.base_density + self.amplitude * oscillation
        velocity = jnp.broadcast_to(mean_velocity, points.shape)
        pressure = jnp.full(points.shape[:-1], self.base_pressure, dtype=points.dtype)
        if self.kind == "isentropic":
            pressure = self.base_pressure * (density / self.base_density) ** system.gamma
            velocity = velocity + (
                self.propagation_sign
                * sound
                * (density - self.base_density)[..., None]
                / self.base_density
                * direction
            )
        elif self.kind == "acoustic":
            pressure = pressure + sound**2 * (density - self.base_density)
            velocity = velocity + (
                self.propagation_sign
                * sound
                * (density - self.base_density)[..., None]
                / self.base_density
                * direction
            )
        elif self.kind == "entropy":
            velocity = velocity
        else:
            density = jnp.full_like(density, self.base_density)
            polarization = jnp.asarray(self.polarization, dtype=points.dtype)
            velocity = velocity + self.amplitude * oscillation[..., None] * polarization
        return jnp.concatenate(
            (density[..., None], velocity, pressure[..., None]), axis=-1
        )

    def evaluate(
        self,
        system: EulerSystem,
        coordinates: ArrayLike,
        time: ArrayLike,
        /,
    ) -> CompressibleReferenceWaveEvidence:
        primitive = self.primitive(system, coordinates, time)
        conserved = system.primitive_to_conserved(primitive)
        density = primitive[..., 0]
        velocity = primitive[..., 1:-1]
        pressure = primitive[..., -1]
        if self.kind == "isentropic":
            expected_pressure = (
                self.base_pressure * (density / self.base_density) ** system.gamma
            )
            pressure_residual = pressure - expected_pressure
        elif self.kind == "acoustic":
            sound_squared = system.gamma * self.base_pressure / self.base_density
            pressure_residual = (pressure - self.base_pressure) - sound_squared * (
                density - self.base_density
            )
        else:
            pressure_residual = pressure - self.base_pressure
        wave = jnp.asarray(self.wave_vector, dtype=primitive.dtype)
        direction = wave / jnp.sqrt(oe.contract("d,d->", wave, wave, backend="jax"))
        fluctuation = velocity - jnp.asarray(self.mean_velocity, dtype=primitive.dtype)
        transverse = (
            fluctuation
            - oe.contract("...d,d->...", fluctuation, direction, backend="jax")[..., None]
            * direction
        )
        longitudinal = fluctuation - transverse
        transverse_residual = longitudinal if self.kind == "vorticity" else transverse
        return CompressibleReferenceWaveEvidence(
            primitive,
            conserved,
            pressure_residual,
            transverse_residual,
            system.admissible(conserved),
            jnp.all(jnp.isfinite(conserved)),
            self.wave_id,
        )


class ManufacturedViscousNSEvidence(StrictModule):
    state: Array
    forcing: Array
    temporal_rate: Array
    inviscid_divergence: Array
    viscous_divergence: Array
    identity_residual: Array
    finite: Array
    plan_id: str = eqx.field(static=True)


class ManufacturedViscousNSPlan(StrictModule, NonTrainableState):
    """Automatic strong-form forcing for a smooth compressible NS state."""

    exact_state: Callable = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    exact_state_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        dimension: int,
        exact_state: Callable[[Array, Array, Any], Array],
        exact_state_id: str,
        /,
    ):
        dimension_ = int(dimension)
        identifier = str(exact_state_id)
        if dimension_ not in (1, 2, 3) or not callable(exact_state) or not identifier:
            raise ValueError("Manufactured viscous-NS plan inputs are invalid.")
        self.dimension = dimension_
        self.exact_state = exact_state
        self.exact_state_id = identifier
        self.plan_id = canonical_fingerprint(
            {
                "kind": "manufactured-viscous-compressible-ns",
                "dimension": dimension_,
                "exact_state": identifier,
            }
        )

    def evaluate(
        self,
        system: CompressibleNavierStokesSystem,
        time: ArrayLike,
        coordinates: ArrayLike,
        args: Any = None,
        /,
    ) -> ManufacturedViscousNSEvidence:
        if (
            not isinstance(system, CompressibleNavierStokesSystem)
            or system.dimension != self.dimension
        ):
            raise TypeError(
                "Manufactured viscous NS requires a matching physical system."
            )
        time_ = jnp.asarray(time)
        points = jnp.asarray(coordinates)
        if time_.shape != () or points.shape[-1:] != (self.dimension,):
            raise ValueError("Manufactured NS time/coordinate shapes are invalid.")
        flat = points.reshape((-1, self.dimension))

        def point_terms(point):
            def state_at_time(local_time):
                return jnp.asarray(self.exact_state(local_time, point, args))

            def state_at_point(local_point):
                return jnp.asarray(self.exact_state(time_, local_point, args))

            state = state_at_point(point)
            temporal = jax.jacfwd(state_at_time)(time_)

            def inviscid_tensor(local_point):
                local_state = state_at_point(local_point)
                return jnp.stack(
                    tuple(
                        system.physical_flux(local_state, axis, args)
                        for axis in range(self.dimension)
                    ),
                    axis=-1,
                )

            inviscid_gradient = jax.jacfwd(inviscid_tensor)(point)
            inviscid_divergence = jnp.trace(inviscid_gradient, axis1=-2, axis2=-1)

            def viscous_tensor(local_point):
                local_state = state_at_point(local_point)
                conserved_gradient = jax.jacfwd(state_at_point)(local_point)
                return system.viscous_flux(local_state, conserved_gradient, args)

            viscous_gradient = jax.jacfwd(viscous_tensor)(point)
            viscous_divergence = jnp.trace(viscous_gradient, axis1=-2, axis2=-1)
            forcing = temporal + inviscid_divergence - viscous_divergence
            identity = temporal + inviscid_divergence - viscous_divergence - forcing
            return (
                state,
                forcing,
                temporal,
                inviscid_divergence,
                viscous_divergence,
                identity,
            )

        values = jax.vmap(point_terms)(flat)
        output_shape = points.shape[:-1] + (self.dimension + 2,)
        reshaped = tuple(value.reshape(output_shape) for value in values)
        return ManufacturedViscousNSEvidence(
            *reshaped,
            jnp.all(jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in reshaped))),
            self.plan_id,
        )


__all__ = [
    "CompressibleReferenceWaveEvidence",
    "CompressibleReferenceWavePlan",
    "ManufacturedViscousNSEvidence",
    "ManufacturedViscousNSPlan",
]
