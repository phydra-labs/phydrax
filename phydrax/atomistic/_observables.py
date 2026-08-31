#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..dynamics import StateLayout, TrajectoryData
from ._dynamics import AtomisticDynamicsState, PreparedAtomisticDynamics
from ._rollout import AtomisticTrajectory


class ThermodynamicAccumulator(StrictModule):
    count: Array
    energy_sum: Array
    energy_square_sum: Array
    temperature_sum: Array
    temperature_square_sum: Array
    pressure_sum: Array
    pressure_square_sum: Array

    @classmethod
    def empty(cls, dtype) -> "ThermodynamicAccumulator":
        zero = jnp.zeros((), dtype=dtype)
        return cls(jnp.zeros((), dtype=jnp.int32), zero, zero, zero, zero, zero, zero)

    def update(
        self,
        dynamics: PreparedAtomisticDynamics,
        state: AtomisticDynamicsState,
        /,
    ) -> "ThermodynamicAccumulator":
        diagnostics = dynamics.diagnostics(state)
        pressure = jnp.where(
            jnp.isfinite(diagnostics.pressure), diagnostics.pressure, 0.0
        )
        return ThermodynamicAccumulator(
            self.count + 1,
            self.energy_sum + diagnostics.total_energy,
            self.energy_square_sum + diagnostics.total_energy**2,
            self.temperature_sum + diagnostics.temperature,
            self.temperature_square_sum + diagnostics.temperature**2,
            self.pressure_sum + pressure,
            self.pressure_square_sum + pressure**2,
        )


class ThermodynamicSummary(StrictModule):
    mean_energy: Array
    energy_variance: Array
    mean_temperature: Array
    temperature_variance: Array
    mean_pressure: Array
    pressure_variance: Array
    count: Array


def summarize_thermodynamics(
    accumulator: ThermodynamicAccumulator, /
) -> ThermodynamicSummary:
    if not isinstance(accumulator, ThermodynamicAccumulator):
        raise TypeError("accumulator must be ThermodynamicAccumulator.")
    count = jnp.maximum(accumulator.count, 1).astype(accumulator.energy_sum.dtype)
    energy = accumulator.energy_sum / count
    temperature = accumulator.temperature_sum / count
    pressure = accumulator.pressure_sum / count
    return ThermodynamicSummary(
        energy,
        jnp.maximum(accumulator.energy_square_sum / count - energy**2, 0.0),
        temperature,
        jnp.maximum(accumulator.temperature_square_sum / count - temperature**2, 0.0),
        pressure,
        jnp.maximum(accumulator.pressure_square_sum / count - pressure**2, 0.0),
        accumulator.count,
    )


class RadialDistributionPlan(StrictModule, NonTrainableState):
    bin_count: int = eqx.field(static=True)
    maximum_radius: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, bin_count: int, maximum_radius: float, /):
        bins = int(bin_count)
        radius = float(maximum_radius)
        if bins <= 0 or not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("RDF bin_count and maximum_radius must be positive.")
        self.bin_count = bins
        self.maximum_radius = radius
        self.plan_id = canonical_fingerprint(
            {"kind": "radial-distribution-plan", "bins": bins, "radius": radius}
        )


class RadialDistributionState(StrictModule):
    counts: Array
    samples: Array

    @classmethod
    def empty(cls, plan: RadialDistributionPlan, dtype) -> "RadialDistributionState":
        return cls(
            jnp.zeros((plan.bin_count,), dtype=dtype), jnp.zeros((), dtype=jnp.int32)
        )

    def update(
        self,
        plan: RadialDistributionPlan,
        dynamics: PreparedAtomisticDynamics,
        state: AtomisticDynamicsState,
        /,
    ) -> "RadialDistributionState":
        context = dynamics.potential.context(
            state.kinematics.positions,
            state.neighborhood,
            unwrapped_positions=dynamics._unwrapped(state.kinematics),
            species=state.species,
            cell=dynamics.system.cell,
        )
        width = plan.maximum_radius / plan.bin_count
        indices = jnp.floor(context.pair_distance / width).astype(jnp.int32)
        valid = context.pair_valid & (indices >= 0) & (indices < plan.bin_count)
        safe = jnp.clip(indices, 0, plan.bin_count - 1)
        increment = (
            jnp.zeros_like(self.counts).at[safe].add(valid.astype(self.counts.dtype))
        )
        return RadialDistributionState(self.counts + increment, self.samples + 1)


class RadialDistributionResult(StrictModule):
    radii: Array
    values: Array
    counts: Array
    samples: Array


def radial_distribution(
    plan: RadialDistributionPlan,
    state: RadialDistributionState,
    particle_count: int,
    volume: ArrayLike,
    /,
) -> RadialDistributionResult:
    width = plan.maximum_radius / plan.bin_count
    edges = jnp.arange(plan.bin_count + 1, dtype=state.counts.dtype) * width
    shell = (4.0 * jnp.pi / 3.0) * (edges[1:] ** 3 - edges[:-1] ** 3)
    count = int(particle_count)
    density = count / jnp.asarray(volume, dtype=state.counts.dtype)
    normalization = jnp.maximum(state.samples, 1) * 0.5 * count * density * shell
    return RadialDistributionResult(
        0.5 * (edges[:-1] + edges[1:]),
        state.counts / normalization,
        state.counts,
        state.samples,
    )


class DisplacementCorrelationState(StrictModule):
    reference_positions: Array
    reference_velocities: Array
    msd_sum: Array
    vacf_sum: Array
    count: Array

    @classmethod
    def initialize(
        cls, dynamics: PreparedAtomisticDynamics, state: AtomisticDynamicsState, /
    ) -> "DisplacementCorrelationState":
        dtype = state.kinematics.positions.dtype
        return cls(
            dynamics._unwrapped(state.kinematics),
            dynamics.velocity(state),
            jnp.zeros((), dtype=dtype),
            jnp.zeros((), dtype=dtype),
            jnp.zeros((), dtype=jnp.int32),
        )

    def update(
        self, dynamics: PreparedAtomisticDynamics, state: AtomisticDynamicsState, /
    ) -> "DisplacementCorrelationState":
        active = dynamics.system.active_mask[:, None]
        displacement = dynamics._unwrapped(state.kinematics) - self.reference_positions
        velocity = dynamics.velocity(state)
        active_count = jnp.sum(dynamics.system.active_mask)
        msd = jnp.sum(jnp.where(active, displacement * displacement, 0.0)) / active_count
        vacf = (
            jnp.sum(jnp.where(active, velocity * self.reference_velocities, 0.0))
            / active_count
        )
        return DisplacementCorrelationState(
            self.reference_positions,
            self.reference_velocities,
            self.msd_sum + msd,
            self.vacf_sum + vacf,
            self.count + 1,
        )


class DisplacementCorrelationResult(StrictModule):
    mean_squared_displacement: Array
    velocity_autocorrelation: Array
    count: Array


def displacement_correlation(
    state: DisplacementCorrelationState, /
) -> DisplacementCorrelationResult:
    count = jnp.maximum(state.count, 1).astype(state.msd_sum.dtype)
    return DisplacementCorrelationResult(
        state.msd_sum / count, state.vacf_sum / count, state.count
    )


def atomistic_trajectory_data(
    trajectory: AtomisticTrajectory,
    dynamics: PreparedAtomisticDynamics,
    /,
) -> TrajectoryData:
    if not isinstance(trajectory, AtomisticTrajectory):
        raise TypeError("trajectory must be AtomisticTrajectory.")
    if not isinstance(dynamics, PreparedAtomisticDynamics):
        raise TypeError("dynamics must be PreparedAtomisticDynamics.")
    masses = dynamics.system.plan.masses.astype(trajectory.momenta.dtype)
    velocities = trajectory.momenta / masses[None, :, None]
    states = jnp.stack((trajectory.positions, velocities), axis=1)
    layout = StateLayout(
        (2, dynamics.system.capacity, 3),
        axes=("kinematic", "atom", "cartesian"),
        layout_id=canonical_fingerprint(
            {
                "kind": "atomistic-trajectory-state-layout",
                "system": dynamics.system.prepared_id,
            }
        ),
    )
    valid = trajectory.sample_mask & trajectory.valid
    return TrajectoryData(
        trajectory.times,
        states,
        state_layout=layout,
        sample_valid=valid,
        source_id=trajectory.trajectory_id,
        coordinate_id="time",
    )


__all__ = [
    "DisplacementCorrelationResult",
    "DisplacementCorrelationState",
    "RadialDistributionPlan",
    "RadialDistributionResult",
    "RadialDistributionState",
    "ThermodynamicAccumulator",
    "ThermodynamicSummary",
    "atomistic_trajectory_data",
    "displacement_correlation",
    "radial_distribution",
    "summarize_thermodynamics",
]
