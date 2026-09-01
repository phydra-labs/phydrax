#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class EquationOfStateTable(StrictModule, NonTrainableState):
    pressure: Array
    energy_density: Array
    sound_speed_squared: Array
    eos_id: str = eqx.field(static=True)

    def __init__(self, pressure, energy_density, /, *, eos_id="tabulated-eos"):
        pressure_host = np.asarray(pressure, dtype=float)
        energy_host = np.asarray(energy_density, dtype=float)
        if (
            pressure_host.ndim != 1
            or pressure_host.size < 2
            or energy_host.shape != pressure_host.shape
            or np.any(np.diff(pressure_host) <= 0.0)
            or np.any(np.diff(energy_host) <= 0.0)
            or pressure_host[0] < 0.0
        ):
            raise ValueError(
                "EOS pressure/energy arrays must be positive monotone vectors."
            )
        derivative = np.gradient(pressure_host, energy_host)
        if np.any(derivative <= 0.0) or np.any(derivative > 1.0 + 1.0e-10):
            raise ValueError("EOS must be stable and causal in geometric units.")
        self.pressure = jnp.asarray(pressure_host)
        self.energy_density = jnp.asarray(energy_host)
        self.sound_speed_squared = jnp.asarray(derivative)
        self.eos_id = canonical_fingerprint(
            {
                "kind": "equation-of-state-table",
                "eos_id": str(eos_id),
                "nodes": int(pressure_host.size),
            }
        )

    def energy_from_pressure(self, pressure: ArrayLike, /) -> Array:
        query = jnp.asarray(pressure)
        return jnp.interp(query, self.pressure, self.energy_density)


class TovResult(StrictModule):
    radii: Array
    mass_profile: Array
    pressure_profile: Array
    radius: Array
    mass: Array
    compactness: Array
    valid: Array
    status: Array
    plan_id: str = eqx.field(static=True)


class TovPlan(StrictModule, NonTrainableState):
    eos: EquationOfStateTable
    radial_nodes: Array
    plan_id: str = eqx.field(static=True)

    def __init__(self, eos, radial_nodes, /):
        radii = np.asarray(radial_nodes, dtype=float)
        if (
            radii.ndim != 1
            or radii.size < 2
            or radii[0] <= 0.0
            or np.any(np.diff(radii) <= 0.0)
        ):
            raise ValueError("TOV radial nodes must be positive and increasing.")
        self.eos = eos
        self.radial_nodes = jnp.asarray(radii)
        self.plan_id = canonical_fingerprint(
            {"kind": "tov-plan", "eos": eos.eos_id, "nodes": int(radii.size)}
        )

    def solve(self, central_pressure: ArrayLike, /) -> TovResult:
        pressure0 = jnp.asarray(central_pressure).reshape(())
        radius0 = self.radial_nodes[0]
        energy0 = self.eos.energy_from_pressure(pressure0)
        mass0 = 4.0 / 3.0 * jnp.pi * radius0**3 * energy0

        def derivative(radius, values):
            mass, pressure = values
            energy = self.eos.energy_from_pressure(
                jnp.maximum(pressure, self.eos.pressure[0])
            )
            denominator = radius * (radius - 2.0 * mass)
            return jnp.asarray(
                (
                    4.0 * jnp.pi * radius**2 * energy,
                    -(energy + pressure)
                    * (mass + 4.0 * jnp.pi * radius**3 * pressure)
                    / jnp.maximum(denominator, 1.0e-30),
                )
            )

        def step(carry, interval):
            values, active = carry
            start, end = interval
            dt = end - start
            k1 = derivative(start, values)
            k2 = derivative(start + 0.5 * dt, values + 0.5 * dt * k1)
            k3 = derivative(start + 0.5 * dt, values + 0.5 * dt * k2)
            k4 = derivative(end, values + dt * k3)
            candidate = values + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            still_active = (
                active
                & (candidate[1] > self.eos.pressure[0])
                & (end > 2.0 * candidate[0])
                & jnp.all(jnp.isfinite(candidate))
            )
            accepted = jnp.where(active, candidate, values)
            return (accepted, still_active), (accepted, active)

        intervals = jnp.stack((self.radial_nodes[:-1], self.radial_nodes[1:]), axis=-1)
        (_, _), outputs = jax.lax.scan(
            step, (jnp.asarray((mass0, pressure0)), jnp.asarray(True)), intervals
        )
        profiles = jnp.concatenate(
            (jnp.asarray((mass0, pressure0))[None], outputs[0]), axis=0
        )
        active = jnp.concatenate((jnp.asarray(True)[None], outputs[1]))
        surface_index = jnp.clip(
            jnp.sum(active.astype(jnp.int32)) - 1, 0, int(self.radial_nodes.size) - 1
        )
        radius = self.radial_nodes[surface_index]
        mass = profiles[surface_index, 0]
        compactness = mass / radius
        valid = (pressure0 > 0.0) & jnp.isfinite(mass) & (compactness < 0.5)
        status = jnp.where(valid, 0, 1).astype(jnp.int32)
        return TovResult(
            self.radial_nodes,
            profiles[:, 0],
            profiles[:, 1],
            radius,
            mass,
            compactness,
            valid,
            status,
            self.plan_id,
        )


class TovSequence(StrictModule):
    central_pressure: Array
    mass: Array
    radius: Array
    stable: Array


def solve_tov_sequence(plan: TovPlan, central_pressures: ArrayLike, /) -> TovSequence:
    pressures = jnp.asarray(central_pressures)
    results = jax.vmap(plan.solve)(pressures)
    derivative_tail = (results.mass[1:] - results.mass[:-1]) / (
        pressures[1:] - pressures[:-1]
    )
    derivative = jnp.concatenate((derivative_tail[:1], derivative_tail))
    return TovSequence(pressures, results.mass, results.radius, derivative > 0.0)


__all__ = [
    "EquationOfStateTable",
    "TovPlan",
    "TovResult",
    "TovSequence",
    "solve_tov_sequence",
]
