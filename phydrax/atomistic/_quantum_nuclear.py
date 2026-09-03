#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._ensemble_advanced import GeneralizedLangevinPlan, ThermostatResult
from ._ring_polymer import RingPolymerState


class RingPolymerNormalModePlan(StrictModule, NonTrainableState):
    bead_count: int = eqx.field(static=True)
    frequencies: Array
    transform: Array
    plan_id: str = eqx.field(static=True)

    def __init__(self, bead_count: int, spring_frequency: float, /):
        count = int(bead_count)
        if count <= 0 or float(spring_frequency) <= 0.0:
            raise ValueError(
                "Normal-mode bead count and spring frequency must be positive."
            )
        beads = jnp.arange(count, dtype=float)
        transform = jnp.zeros((count, count), dtype=float)
        transform = transform.at[0].set(jnp.ones((count,)) / jnp.sqrt(count))
        for mode in range(1, (count + 1) // 2):
            angle = 2.0 * jnp.pi * mode * beads / count
            transform = transform.at[mode].set(jnp.sqrt(2.0 / count) * jnp.cos(angle))
            transform = transform.at[count - mode].set(
                jnp.sqrt(2.0 / count) * jnp.sin(angle)
            )
        if count % 2 == 0:
            transform = transform.at[count // 2].set((-1.0) ** beads / jnp.sqrt(count))
        modes = jnp.arange(count, dtype=float)
        self.bead_count = count
        self.frequencies = 2.0 * spring_frequency * jnp.sin(jnp.pi * modes / count)
        self.transform = transform
        self.plan_id = canonical_fingerprint(
            {
                "kind": "ring-polymer-normal-modes",
                "beads": count,
                "spring_frequency": float(spring_frequency),
            }
        )

    def forward(self, value: ArrayLike, /):
        array = jnp.asarray(value)
        if array.shape[0] != self.bead_count:
            raise ValueError("Normal-mode input bead count does not match the plan.")
        return contract("kb,b...->k...", self.transform.astype(array.dtype), array)

    def inverse(self, value: ArrayLike, /):
        array = jnp.asarray(value)
        if array.shape[0] != self.bead_count:
            raise ValueError("Normal-mode input bead count does not match the plan.")
        return contract("kb,k...->b...", self.transform.astype(array.dtype), array)

    def propagate(
        self,
        positions: ArrayLike,
        momenta: ArrayLike,
        masses: ArrayLike,
        step_size: ArrayLike,
        /,
    ):
        q, p = self.forward(positions), self.forward(momenta)
        mass = jnp.asarray(masses)[None, :, None]
        dt = jnp.asarray(step_size)
        omega = self.frequencies[:, None, None]
        cosine, sine = jnp.cos(omega * dt), jnp.sin(omega * dt)
        safe_omega = jnp.where(jnp.abs(omega) > 0.0, omega, 1.0)
        next_q = cosine * q + sine * p / (mass * safe_omega)
        next_p = cosine * p - sine * mass * safe_omega * q
        centroid = omega == 0.0
        next_q = jnp.where(centroid, q + dt * p / mass, next_q)
        next_p = jnp.where(centroid, p, next_p)
        return self.inverse(next_q), self.inverse(next_p)


class StagingCoordinatePlan(StrictModule, NonTrainableState):
    bead_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, bead_count: int, /):
        if int(bead_count) <= 0:
            raise ValueError("Staging bead count must be positive.")
        self.bead_count = int(bead_count)
        self.plan_id = canonical_fingerprint(
            {"kind": "ring-polymer-staging", "beads": self.bead_count}
        )

    def forward(self, positions: ArrayLike, /):
        q = jnp.asarray(positions)
        staging = jnp.zeros_like(q).at[0].set(q[0])
        for bead in range(1, self.bead_count):
            reference = ((self.bead_count - bead) * q[bead - 1] + q[0]) / (
                self.bead_count - bead + 1
            )
            staging = staging.at[bead].set(q[bead] - reference)
        return staging

    def inverse(self, staging: ArrayLike, /):
        u = jnp.asarray(staging)
        q = jnp.zeros_like(u).at[0].set(u[0])
        for bead in range(1, self.bead_count):
            reference = ((self.bead_count - bead) * q[bead - 1] + q[0]) / (
                self.bead_count - bead + 1
            )
            q = q.at[bead].set(u[bead] + reference)
        return q


class QuantumEstimatorResult(StrictModule):
    centroid_virial_kinetic: Array
    primitive_kinetic: Array
    radius_of_gyration: Array
    isotope_log_weight: Array
    successful: Array


def quantum_estimators(
    state: RingPolymerState,
    masses: ArrayLike,
    physical_forces: ArrayLike,
    temperature: float,
    boltzmann_constant: float,
    spring_frequency: float,
    /,
    *,
    isotope_masses: ArrayLike | None = None,
):
    q = state.positions
    mass = jnp.asarray(masses)[None, :, None]
    centroid = jnp.mean(q, axis=0)
    displacement = q - centroid[None, :, :]
    radius = jnp.sqrt(jnp.mean(jnp.sum(displacement**2, axis=-1)))
    spring = jnp.roll(q, -1, axis=0) - q
    spring_energy = 0.5 * spring_frequency**2 * jnp.sum(mass * spring**2)
    dof = 3 * q.shape[1]
    primitive = 0.5 * dof * q.shape[0] * boltzmann_constant * temperature - spring_energy
    forces = jnp.asarray(physical_forces, dtype=q.dtype)
    if forces.shape != q.shape:
        raise ValueError("Centroid-virial forces must match ring-polymer positions.")
    centroid_virial = 0.5 * dof * boltzmann_constant * temperature - 0.5 / q.shape[
        0
    ] * jnp.sum(displacement * forces)
    isotope_weight = jnp.zeros(())
    if isotope_masses is not None:
        ratio = jnp.asarray(isotope_masses) / jnp.asarray(masses)
        isotope_weight = (
            -0.5
            * spring_frequency**2
            / (boltzmann_constant * temperature)
            * jnp.sum((ratio[None, :, None] - 1.0) * mass * spring**2)
        )
    successful = jnp.all(
        jnp.isfinite(jnp.asarray([primitive, centroid_virial, radius, isotope_weight]))
    )
    return QuantumEstimatorResult(
        centroid_virial, primitive, radius, isotope_weight, successful
    )


def ring_polymer_contract(positions: ArrayLike, contracted_beads: int, /):
    q = jnp.asarray(positions)
    target = int(contracted_beads)
    source = q.shape[0]
    if target <= 0 or target > source:
        raise ValueError("Contracted bead count is invalid.")
    source_modes = jnp.fft.fft(q, axis=0)
    target_modes = jnp.zeros((target,) + q.shape[1:], dtype=source_modes.dtype)
    scale = target / source
    target_modes = target_modes.at[0].set(scale * source_modes[0])
    paired = (target - 1) // 2
    for mode in range(1, paired + 1):
        target_modes = target_modes.at[mode].set(scale * source_modes[mode])
        target_modes = target_modes.at[-mode].set(scale * source_modes[-mode])
    if target % 2 == 0:
        nyquist = target // 2
        target_modes = target_modes.at[nyquist].set(
            0.5 * scale * (source_modes[nyquist] + source_modes[-nyquist])
        )
    return jnp.real(jnp.fft.ifft(target_modes, axis=0))


class ThermostattedRPMDPlan(StrictModule, NonTrainableState):
    normal_modes: RingPolymerNormalModePlan
    friction: Array
    plan_id: str = eqx.field(static=True)

    def __init__(self, normal_modes: RingPolymerNormalModePlan, friction: ArrayLike, /):
        values = jnp.asarray(friction, dtype=float).reshape((-1,))
        if values.shape != (normal_modes.bead_count,) or bool(jnp.any(values < 0.0)):
            raise ValueError("TRPMD friction must align with ring-polymer modes.")
        self.normal_modes, self.friction = normal_modes, values
        self.plan_id = canonical_fingerprint(
            {
                "kind": "thermostatted-rpmd",
                "normal_modes": normal_modes.plan_id,
                "friction": list(map(float, values)),
            }
        )

    def apply(self, momenta, masses, temperature, dt, key, units, /):
        mode_momenta = self.normal_modes.forward(momenta)
        mass = jnp.asarray(masses)[None, :, None]
        decay = jnp.exp(-self.friction[:, None, None] * dt)
        noise = self.normal_modes.forward(
            jax.random.normal(key, jnp.asarray(momenta).shape)
        )
        variance = mass * units.boltzmann_constant * temperature / units.kinetic_to_energy
        updated = (
            decay * mode_momenta
            + jnp.sqrt(jnp.maximum(1.0 - decay**2, 0.0) * variance) * noise
        )
        result = self.normal_modes.inverse(updated)
        before = 0.5 * units.kinetic_to_energy * jnp.sum(jnp.asarray(momenta) ** 2 / mass)
        after = 0.5 * units.kinetic_to_energy * jnp.sum(result**2 / mass)
        return ThermostatResult(
            result,
            jnp.zeros((0,), dtype=result.dtype),
            after - before,
            jnp.all(jnp.isfinite(result)),
        )


def open_path_momentum_distribution(open_positions: ArrayLike, momenta: ArrayLike, /):
    q, p = jnp.asarray(open_positions), jnp.asarray(momenta)
    end_to_end = q[-1] - q[0]
    phase = jnp.sum(p * end_to_end[None, ...], axis=(-2, -1))
    return jnp.real(jnp.fft.fft(jnp.exp(1.0j * phase)))


def constant_pressure_ring_polymer(state: RingPolymerState, scale: ArrayLike, /):
    factor = jnp.asarray(scale)
    centroid_position = jnp.mean(state.positions, axis=0)
    centroid_momentum = jnp.mean(state.momenta, axis=0)
    positions = centroid_position[None, ...] * factor + (
        state.positions - centroid_position[None, ...]
    )
    momenta = centroid_momentum[None, ...] / factor + (
        state.momenta - centroid_momentum[None, ...]
    )
    return eqx.tree_at(
        lambda value: (value.positions, value.momenta),
        state,
        (positions, momenta),
    )


def suzuki_chin_correction(force_function, positions: ArrayLike, epsilon: float, /):
    q = jnp.asarray(positions)
    force = force_function(q)
    directional = jax.jvp(force_function, (q,), (force,))[1]
    return (epsilon**2 / 24.0) * jnp.sum(force * force), directional


class ConstantPressureRingPolymerState(StrictModule):
    polymer: RingPolymerState
    cell_vectors: Array
    barostat_momentum: Array
    pressure_work: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class ConstantPressureRingPolymerPlan(StrictModule, NonTrainableState):
    target_pressure: float = eqx.field(static=True)
    barostat_mass: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, target_pressure: float, barostat_mass: float, /):
        values = float(target_pressure), float(barostat_mass)
        if not all(jnp.isfinite(jnp.asarray(values))) or values[1] <= 0.0:
            raise ValueError("Ring-polymer barostat parameters are invalid.")
        self.target_pressure, self.barostat_mass = values
        self.plan_id = canonical_fingerprint(
            {
                "kind": "constant-pressure-ring-polymer",
                "target_pressure": values[0],
                "barostat_mass": values[1],
            }
        )

    def initialize(self, polymer: RingPolymerState, cell_vectors: ArrayLike, /):
        cell = jnp.asarray(cell_vectors, dtype=polymer.positions.dtype)
        if cell.shape != (3, 3):
            raise ValueError("Ring-polymer barostat cell must have shape (3,3).")
        return ConstantPressureRingPolymerState(
            polymer,
            cell,
            jnp.zeros((), dtype=cell.dtype),
            jnp.zeros((), dtype=cell.dtype),
            jnp.all(jnp.isfinite(cell)),
            self.plan_id,
        )

    def step(self, state: ConstantPressureRingPolymerState, internal_pressure, dt, /):
        if state.plan_id != self.plan_id:
            raise ValueError("Ring-polymer barostat state belongs to another plan.")
        volume = jnp.abs(
            jnp.sum(
                state.cell_vectors[0]
                * jnp.cross(state.cell_vectors[1], state.cell_vectors[2])
            )
        )
        force = 3.0 * volume * (jnp.asarray(internal_pressure) - self.target_pressure)
        half_momentum = state.barostat_momentum + 0.5 * dt * force
        scale = jnp.exp(dt * half_momentum / self.barostat_mass)
        polymer = constant_pressure_ring_polymer(state.polymer, scale)
        cell = state.cell_vectors * scale
        next_volume = volume * scale**3
        next_force = (
            3.0 * next_volume * (jnp.asarray(internal_pressure) - self.target_pressure)
        )
        momentum = half_momentum + 0.5 * dt * next_force
        work = state.pressure_work + self.target_pressure * (next_volume - volume)
        successful = (
            state.successful
            & jnp.isfinite(scale)
            & (scale > 0.0)
            & jnp.all(jnp.isfinite(polymer.positions))
            & jnp.all(jnp.isfinite(cell))
        )
        return ConstantPressureRingPolymerState(
            polymer, cell, momentum, work, successful, self.plan_id
        )


class PIGLETPlan(StrictModule, NonTrainableState):
    thermostats: tuple[GeneralizedLangevinPlan, ...]
    normal_modes: RingPolymerNormalModePlan
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        thermostats: tuple[GeneralizedLangevinPlan, ...],
        normal_modes: RingPolymerNormalModePlan,
        /,
    ):
        values = tuple(thermostats)
        if (
            len(values) != normal_modes.bead_count
            or any(not isinstance(value, GeneralizedLangevinPlan) for value in values)
            or len({value.drift_matrix.shape for value in values}) != 1
        ):
            raise ValueError(
                "PIGLET requires one shape-compatible GLE plan per normal mode."
            )
        self.thermostats, self.normal_modes = values, normal_modes
        self.plan_id = canonical_fingerprint(
            {
                "kind": "piglet-plan",
                "thermostats": [value.plan_id for value in values],
                "normal_modes": normal_modes.plan_id,
            }
        )

    def apply(
        self,
        momenta,
        masses,
        mobile_mask,
        key,
        step,
        dt,
        units,
        /,
        *,
        auxiliary=None,
    ):
        mode_momenta = self.normal_modes.forward(momenta)
        auxiliary_values = (
            (None,) * self.normal_modes.bead_count
            if auxiliary is None
            else tuple(
                jnp.asarray(auxiliary)[mode]
                for mode in range(self.normal_modes.bead_count)
            )
        )
        results = tuple(
            thermostat.apply(
                mode_momenta[mode],
                masses,
                mobile_mask,
                jax.random.fold_in(key, mode),
                step,
                dt,
                units,
                auxiliary=auxiliary_values[mode],
            )
            for mode, thermostat in enumerate(self.thermostats)
        )
        physical = self.normal_modes.inverse(
            jnp.stack(tuple(result.momenta for result in results))
        )
        memories = jnp.stack(tuple(result.auxiliary for result in results))
        heat = jnp.sum(jnp.stack(tuple(result.heat for result in results)))
        successful = jnp.all(jnp.stack(tuple(result.successful for result in results)))
        return ThermostatResult(physical, memories, heat, successful)


__all__ = [
    "ConstantPressureRingPolymerPlan",
    "ConstantPressureRingPolymerState",
    "PIGLETPlan",
    "QuantumEstimatorResult",
    "RingPolymerNormalModePlan",
    "StagingCoordinatePlan",
    "ThermostattedRPMDPlan",
    "constant_pressure_ring_polymer",
    "open_path_momentum_distribution",
    "quantum_estimators",
    "ring_polymer_contract",
    "suzuki_chin_correction",
]
