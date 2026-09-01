#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class WaveComponent(StrictModule, NonTrainableState):
    amplitude: float = eqx.field(static=True)
    angular_frequency: float = eqx.field(static=True)
    direction: float = eqx.field(static=True)
    phase: float = eqx.field(static=True)
    wavenumber: float = eqx.field(static=True)
    component_id: str = eqx.field(static=True)

    def __init__(
        self,
        amplitude: float,
        angular_frequency: float,
        direction: float = 0.0,
        phase: float = 0.0,
        wavenumber: float = 0.0,
        /,
    ):
        values = tuple(
            float(value)
            for value in (
                amplitude,
                angular_frequency,
                direction,
                phase,
                wavenumber,
            )
        )
        if any(not np.isfinite(value) for value in values):
            raise ValueError("Wave component parameters must be finite.")
        if values[0] < 0.0 or values[1] <= 0.0 or values[4] < 0.0:
            raise ValueError("Wave amplitude/frequency/wavenumber are invalid.")
        self.amplitude = values[0]
        self.angular_frequency = values[1]
        self.direction = values[2]
        self.phase = values[3]
        self.wavenumber = values[4]
        self.component_id = canonical_fingerprint(
            {"kind": "wave-component", "values": list(values)}
        )


class WaveSample(StrictModule):
    eta: Array
    eta_rate: Array
    velocity: Array
    pressure_head: Array
    energy_flux: Array
    ramp: Array
    finite: Array
    valid: Array
    provider_id: str = eqx.field(static=True)


class IncidentWavePlan(StrictModule, NonTrainableState):
    """Coherent linear regular/irregular gravity-wave provider."""

    components: tuple[WaveComponent, ...]
    depth: float = eqx.field(static=True)
    gravity: float = eqx.field(static=True)
    current: tuple[float, float] = eqx.field(static=True)
    ramp_time: float = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)

    def __init__(
        self,
        components: Sequence[WaveComponent],
        depth: float,
        /,
        *,
        gravity: float = 9.81,
        current: tuple[float, float] = (0.0, 0.0),
        ramp_time: float = 0.0,
    ):
        supplied = tuple(components)
        if not supplied or any(not isinstance(c, WaveComponent) for c in supplied):
            raise ValueError("IncidentWavePlan requires WaveComponent entries.")
        depth_ = float(depth)
        gravity_ = float(gravity)
        current_ = tuple(float(value) for value in current)
        ramp = float(ramp_time)
        if (
            depth_ <= 0.0
            or gravity_ <= 0.0
            or ramp < 0.0
            or any(not np.isfinite(v) for v in (depth_, gravity_, ramp, *current_))
        ):
            raise ValueError("Incident-wave depth/gravity/current/ramp are invalid.")
        resolved = []
        for component in supplied:
            if component.wavenumber > 0.0:
                k = component.wavenumber
            else:
                k = _dispersion_root(
                    component.angular_frequency,
                    component.direction,
                    depth_,
                    gravity_,
                    current_,
                )
            resolved.append(
                WaveComponent(
                    component.amplitude,
                    component.angular_frequency,
                    component.direction,
                    component.phase,
                    k,
                )
            )
        self.components = tuple(resolved)
        self.depth = depth_
        self.gravity = gravity_
        self.current = current_
        self.ramp_time = ramp
        self.provider_id = canonical_fingerprint(
            {
                "kind": "incident-wave-plan",
                "components": [c.component_id for c in self.components],
                "depth": depth_,
                "gravity": gravity_,
                "current": list(current_),
                "ramp_time": ramp,
                "theory": "linear-airy",
            }
        )

    def sample(self, time: ArrayLike, coordinates: ArrayLike, /) -> WaveSample:
        time_ = jnp.asarray(time).reshape(())
        points = jnp.asarray(coordinates)
        if points.shape[-1] != 3:
            raise ValueError("Wave sample coordinates need trailing xyz components.")
        eta = jnp.zeros(points.shape[:-1], dtype=points.dtype)
        eta_rate = jnp.zeros_like(eta)
        velocity = jnp.broadcast_to(
            jnp.asarray((self.current[0], self.current[1], 0.0), dtype=points.dtype),
            points.shape,
        )
        pressure = jnp.zeros_like(eta)
        energy_flux = jnp.zeros_like(eta)
        for component in self.components:
            direction = jnp.asarray(
                (math.cos(component.direction), math.sin(component.direction))
            )
            intrinsic = component.angular_frequency - component.wavenumber * (
                direction[0] * self.current[0] + direction[1] * self.current[1]
            )
            theta = (
                component.wavenumber
                * (direction[0] * points[..., 0] + direction[1] * points[..., 1])
                - component.angular_frequency * time_
                + component.phase
            )
            cosh = jnp.cosh(component.wavenumber * (points[..., 2] + self.depth))
            sinh = jnp.sinh(component.wavenumber * (points[..., 2] + self.depth))
            denominator = jnp.sinh(component.wavenumber * self.depth)
            surface_denominator = jnp.cosh(component.wavenumber * self.depth)
            eta_component = component.amplitude * jnp.cos(theta)
            horizontal_speed = (
                component.amplitude * intrinsic * cosh / denominator * jnp.cos(theta)
            )
            vertical_speed = (
                component.amplitude * intrinsic * sinh / denominator * jnp.sin(theta)
            )
            eta = eta + eta_component
            eta_rate = (
                eta_rate
                + component.amplitude * component.angular_frequency * jnp.sin(theta)
            )
            velocity = velocity.at[..., 0].add(direction[0] * horizontal_speed)
            velocity = velocity.at[..., 1].add(direction[1] * horizontal_speed)
            velocity = velocity.at[..., 2].add(vertical_speed)
            pressure = pressure + (
                self.gravity
                * component.amplitude
                * cosh
                / surface_denominator
                * jnp.cos(theta)
            )
            group_factor = 0.5 * (
                1.0
                + 2.0
                * component.wavenumber
                * self.depth
                / jnp.sinh(2.0 * component.wavenumber * self.depth)
            )
            energy_flux = energy_flux + (
                0.5
                * self.gravity
                * component.amplitude**2
                * (intrinsic / component.wavenumber)
                * group_factor
            )
        ramp = (
            jnp.asarray(1.0, dtype=points.dtype)
            if self.ramp_time == 0.0
            else jnp.clip(time_ / self.ramp_time, 0.0, 1.0) ** 2
            * (3.0 - 2.0 * jnp.clip(time_ / self.ramp_time, 0.0, 1.0))
        )
        eta = ramp * eta
        eta_rate = ramp * eta_rate
        wave_velocity = velocity - jnp.asarray((self.current[0], self.current[1], 0.0))
        velocity = (
            jnp.asarray((self.current[0], self.current[1], 0.0))
            + ramp[..., None] * wave_velocity
        )
        pressure = ramp * pressure
        energy_flux = ramp**2 * energy_flux
        finite = (
            jnp.all(jnp.isfinite(eta))
            & jnp.all(jnp.isfinite(eta_rate))
            & jnp.all(jnp.isfinite(velocity))
            & jnp.all(jnp.isfinite(pressure))
            & jnp.all(jnp.isfinite(energy_flux))
        )
        return WaveSample(
            eta=eta,
            eta_rate=eta_rate,
            velocity=velocity,
            pressure_head=pressure,
            energy_flux=energy_flux,
            ramp=ramp,
            finite=finite,
            valid=finite,
            provider_id=self.provider_id,
        )


def _dispersion_root(
    angular_frequency: float,
    direction: float,
    depth: float,
    gravity: float,
    current: tuple[float, float],
) -> float:
    projected_current = current[0] * math.cos(direction) + current[1] * math.sin(
        direction
    )

    def residual(k):
        intrinsic = angular_frequency - k * projected_current
        return gravity * k * math.tanh(k * depth) - intrinsic**2

    lower = 1.0e-12
    upper = max(
        angular_frequency**2 / gravity,
        angular_frequency / math.sqrt(gravity * depth),
        1.0 / depth,
    )
    for _ in range(80):
        if residual(upper) > 0.0:
            break
        upper *= 2.0
    else:
        raise ValueError(
            "Incident-wave dispersion relation has no positive bracketed root."
        )
    for _ in range(100):
        midpoint = 0.5 * (lower + upper)
        if residual(midpoint) > 0.0:
            upper = midpoint
        else:
            lower = midpoint
    root = 0.5 * (lower + upper)
    if abs(residual(root)) > 1.0e-10 * max(angular_frequency**2, 1.0):
        raise ValueError("Incident-wave dispersion solve did not converge.")
    intrinsic = angular_frequency - root * projected_current
    if intrinsic <= 0.0:
        raise ValueError(
            "Incident-wave intrinsic-frequency branch is blocked or reversed."
        )
    return root


__all__ = ["IncidentWavePlan", "WaveComponent", "WaveSample"]
