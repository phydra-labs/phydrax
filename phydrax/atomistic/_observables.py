#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..dynamics import StateLayout, TrajectoryData
from ._dynamics import AtomisticDynamicsState, PreparedAtomisticDynamics
from ._rollout import AtomisticTrajectory
from ._thermodynamic import PreparedThermodynamicStateTable


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
        thermodynamic_states: PreparedThermodynamicStateTable,
        /,
    ) -> "ThermodynamicAccumulator":
        diagnostics = dynamics.diagnostics(state, thermodynamic_states)
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


class StaticStructureFactorPlan(StrictModule, NonTrainableState):
    """Fixed reciprocal probes and explicit frame/particle resource bounds."""

    wave_vectors: Array
    maximum_frames: int = eqx.field(static=True)
    maximum_particles: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        wave_vectors: ArrayLike,
        /,
        *,
        maximum_frames: int,
        maximum_particles: int,
    ):
        vectors = np.asarray(wave_vectors, dtype=float)
        frame_capacity = int(maximum_frames)
        particle_capacity = int(maximum_particles)
        if (
            vectors.ndim != 2
            or vectors.shape[0] == 0
            or vectors.shape[1] != 3
            or not np.all(np.isfinite(vectors))
            or frame_capacity <= 0
            or particle_capacity <= 0
        ):
            raise ValueError("Structure-factor probes and resource bounds are invalid.")
        self.wave_vectors = jnp.asarray(vectors)
        self.maximum_frames = frame_capacity
        self.maximum_particles = particle_capacity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "static-structure-factor-plan",
                "wave_vectors": vectors.tolist(),
                "maximum_frames": frame_capacity,
                "maximum_particles": particle_capacity,
            }
        )


class StaticStructureFactorResult(StrictModule):
    wave_vectors: Array
    values: Array
    frame_values: Array
    active_frames: Array
    active_particles: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


def static_structure_factor(
    plan: StaticStructureFactorPlan,
    positions: ArrayLike,
    /,
    *,
    particle_mask: ArrayLike | None = None,
    sample_mask: ArrayLike | None = None,
    scattering_lengths: ArrayLike | None = None,
) -> StaticStructureFactorResult:
    """Return S(k)=<|sum_j b_j exp(i k.r_j)|²>/sum_j b_j²."""

    if not isinstance(plan, StaticStructureFactorPlan):
        raise TypeError("plan must be StaticStructureFactorPlan.")
    values = jnp.asarray(positions)
    if values.ndim == 2:
        values = values[None, ...]
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("positions must have shape (frames, particles, 3).")
    frame_count, particle_count = values.shape[:2]
    if frame_count > plan.maximum_frames or particle_count > plan.maximum_particles:
        raise ValueError("Structure-factor input exceeds its declared resource bounds.")
    particles = (
        jnp.ones((particle_count,), dtype=bool)
        if particle_mask is None
        else jnp.asarray(particle_mask, dtype=bool)
    )
    samples = (
        jnp.ones((frame_count,), dtype=bool)
        if sample_mask is None
        else jnp.asarray(sample_mask, dtype=bool)
    )
    scattering = (
        jnp.ones((particle_count,), dtype=values.dtype)
        if scattering_lengths is None
        else jnp.asarray(scattering_lengths, dtype=values.dtype)
    )
    if (
        particles.shape != (particle_count,)
        or samples.shape != (frame_count,)
        or scattering.shape != (particle_count,)
    ):
        raise ValueError("Structure-factor masks and scattering lengths are misaligned.")
    weights = jnp.where(particles, scattering, 0.0)
    normalization = jnp.sum(weights * weights)
    phase = contract(
        "tnd,kd->tkn",
        values,
        plan.wave_vectors.astype(values.dtype),
    )
    amplitude = jnp.sum(weights[None, None, :] * jnp.exp(1j * phase), axis=-1)
    frame_values = jnp.real(amplitude * jnp.conj(amplitude)) / jnp.maximum(
        normalization, jnp.finfo(values.dtype).tiny
    )
    active_frames = jnp.sum(samples, dtype=jnp.int32)
    averaged = jnp.sum(jnp.where(samples[:, None], frame_values, 0.0), axis=0) / (
        jnp.maximum(active_frames, 1)
    )
    finite = (
        jnp.all(jnp.isfinite(jnp.where(samples[:, None, None], values, 0.0)))
        & jnp.all(jnp.isfinite(weights))
        & jnp.all(jnp.isfinite(frame_values))
    )
    successful = (
        finite
        & (active_frames > 0)
        & (jnp.sum(particles, dtype=jnp.int32) > 0)
        & (normalization > 0.0)
    )
    return StaticStructureFactorResult(
        plan.wave_vectors,
        jnp.where(successful, averaged, jnp.nan),
        jnp.where(samples[:, None], frame_values, 0.0),
        active_frames,
        jnp.sum(particles, dtype=jnp.int32),
        successful,
        plan.plan_id,
    )


class LaggedCorrelationPlan(StrictModule, NonTrainableState):
    """Fixed lag set for explicitly unwrapped trajectories."""

    lag_steps: Array
    spatial_dimension: int = eqx.field(static=True)
    maximum_frames: int = eqx.field(static=True)
    maximum_particles: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        lag_steps: ArrayLike,
        /,
        *,
        maximum_frames: int,
        maximum_particles: int,
        spatial_dimension: int = 3,
    ):
        lags = np.asarray(lag_steps)
        frames = int(maximum_frames)
        particles = int(maximum_particles)
        dimension = int(spatial_dimension)
        if (
            lags.ndim != 1
            or lags.size == 0
            or not np.issubdtype(lags.dtype, np.integer)
            or np.any(lags < 0)
            or np.unique(lags).size != lags.size
            or np.any(np.diff(lags) <= 0)
            or frames <= 0
            or particles <= 0
            or dimension <= 0
            or int(lags[-1]) >= frames
        ):
            raise ValueError("Lag set, dimension, and resource bounds are invalid.")
        self.lag_steps = jnp.asarray(lags, dtype=jnp.int32)
        self.spatial_dimension = dimension
        self.maximum_frames = frames
        self.maximum_particles = particles
        self.plan_id = canonical_fingerprint(
            {
                "kind": "lagged-atomistic-correlation-plan",
                "lag_steps": lags.astype(int).tolist(),
                "spatial_dimension": dimension,
                "maximum_frames": frames,
                "maximum_particles": particles,
            }
        )


class LaggedCorrelationResult(StrictModule):
    lag_steps: Array
    lag_times: Array
    mean_squared_displacement: Array
    velocity_autocorrelation: Array
    origin_counts: Array
    particle_pair_counts: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


def lagged_msd_vacf(
    plan: LaggedCorrelationPlan,
    times: ArrayLike,
    unwrapped_positions: ArrayLike,
    velocities: ArrayLike,
    /,
    *,
    particle_mask: ArrayLike | None = None,
    sample_mask: ArrayLike | None = None,
) -> LaggedCorrelationResult:
    """Compute all-origin MSD and VACF without applying a minimum-image map."""

    if not isinstance(plan, LaggedCorrelationPlan):
        raise TypeError("plan must be LaggedCorrelationPlan.")
    position = jnp.asarray(unwrapped_positions)
    velocity = jnp.asarray(velocities, dtype=position.dtype)
    time = jnp.asarray(times, dtype=position.dtype)
    if position.ndim != 3 or position.shape[-1] != plan.spatial_dimension:
        raise ValueError(
            "unwrapped_positions must have shape (frames, particles, dimension)."
        )
    if velocity.shape != position.shape or time.shape != position.shape[:1]:
        raise ValueError("Times, unwrapped positions, and velocities are misaligned.")
    frame_count, particle_count = position.shape[:2]
    if frame_count > plan.maximum_frames or particle_count > plan.maximum_particles:
        raise ValueError("Lagged-correlation input exceeds declared resource bounds.")
    particles = (
        jnp.ones((particle_count,), dtype=bool)
        if particle_mask is None
        else jnp.asarray(particle_mask, dtype=bool)
    )
    samples = (
        jnp.ones((frame_count,), dtype=bool)
        if sample_mask is None
        else jnp.asarray(sample_mask, dtype=bool)
    )
    if particles.shape != (particle_count,) or samples.shape != (frame_count,):
        raise ValueError("Lagged-correlation masks are misaligned.")
    origin = jnp.arange(frame_count, dtype=jnp.int32)[None, :]
    target = origin + plan.lag_steps[:, None]
    inside = target < frame_count
    safe_target = jnp.clip(target, 0, frame_count - 1)
    pair_frames = inside & samples[None, :] & samples[safe_target]
    pair_particles = pair_frames[..., None] & particles[None, None, :]
    displacement = position[safe_target] - position[origin]
    velocity_product = jnp.sum(velocity[safe_target] * velocity[origin], axis=-1)
    squared = jnp.sum(displacement * displacement, axis=-1)
    particle_pair_counts = jnp.sum(pair_particles, axis=(1, 2), dtype=jnp.int32)
    denominator = jnp.maximum(particle_pair_counts, 1).astype(position.dtype)
    msd = jnp.sum(jnp.where(pair_particles, squared, 0.0), axis=(1, 2)) / denominator
    vacf = (
        jnp.sum(jnp.where(pair_particles, velocity_product, 0.0), axis=(1, 2))
        / denominator
    )
    origin_counts = jnp.sum(pair_frames, axis=1, dtype=jnp.int32)
    lag_times = jnp.sum(
        jnp.where(pair_frames, time[safe_target] - time[origin], 0.0), axis=1
    ) / jnp.maximum(origin_counts, 1)
    finite = (
        jnp.all(jnp.isfinite(jnp.where(samples[:, None, None], position, 0.0)))
        & jnp.all(jnp.isfinite(jnp.where(samples[:, None, None], velocity, 0.0)))
        & jnp.all(jnp.isfinite(jnp.where(samples, time, 0.0)))
    )
    successful = (
        finite
        & jnp.all(particle_pair_counts > 0)
        & jnp.all(jnp.isfinite(msd))
        & jnp.all(jnp.isfinite(vacf))
        & jnp.all(jnp.isfinite(lag_times))
    )
    return LaggedCorrelationResult(
        plan.lag_steps,
        lag_times,
        msd,
        vacf,
        origin_counts,
        particle_pair_counts,
        successful,
        plan.plan_id,
    )


class DiffusionFitPlan(StrictModule, NonTrainableState):
    first_lag_index: int = eqx.field(static=True)
    last_lag_index: int = eqx.field(static=True)
    spatial_dimension: int = eqx.field(static=True)
    minimum_origins: int = eqx.field(static=True)
    minimum_r_squared: float = eqx.field(static=True)
    maximum_relative_standard_error: float = eqx.field(static=True)
    maximum_einstein_green_kubo_relative_error: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        first_lag_index: int,
        last_lag_index: int,
        /,
        *,
        spatial_dimension: int = 3,
        minimum_origins: int = 4,
        minimum_r_squared: float = 0.9,
        maximum_relative_standard_error: float = 0.25,
        maximum_einstein_green_kubo_relative_error: float = 0.5,
    ):
        first = int(first_lag_index)
        last = int(last_lag_index)
        dimension = int(spatial_dimension)
        origins = int(minimum_origins)
        thresholds = (
            float(minimum_r_squared),
            float(maximum_relative_standard_error),
            float(maximum_einstein_green_kubo_relative_error),
        )
        if (
            first < 0
            or last < first + 2
            or dimension <= 0
            or origins <= 0
            or not all(np.isfinite(value) for value in thresholds)
            or not 0.0 <= thresholds[0] <= 1.0
            or thresholds[1] <= 0.0
            or thresholds[2] < 0.0
        ):
            raise ValueError("Diffusion fit window and evidence thresholds are invalid.")
        self.first_lag_index = first
        self.last_lag_index = last
        self.spatial_dimension = dimension
        self.minimum_origins = origins
        self.minimum_r_squared = thresholds[0]
        self.maximum_relative_standard_error = thresholds[1]
        self.maximum_einstein_green_kubo_relative_error = thresholds[2]
        self.plan_id = canonical_fingerprint(
            {
                "kind": "atomistic-diffusion-fit-plan",
                "first_lag_index": first,
                "last_lag_index": last,
                "spatial_dimension": dimension,
                "minimum_origins": origins,
                "minimum_r_squared": thresholds[0],
                "maximum_relative_standard_error": thresholds[1],
                "maximum_einstein_green_kubo_relative_error": thresholds[2],
            }
        )


class DiffusionEvidence(StrictModule):
    diffusion_coefficient: Array
    standard_error: Array
    slope: Array
    intercept: Array
    r_squared: Array
    green_kubo_diffusion: Array
    einstein_green_kubo_relative_error: Array
    fit_point_count: Array
    minimum_origin_count: Array
    successful: Array
    plan_id: str = eqx.field(static=True)
    correlation_plan_id: str = eqx.field(static=True)


def fit_diffusion(
    plan: DiffusionFitPlan, correlation: LaggedCorrelationResult, /
) -> DiffusionEvidence:
    """Fit the diffusive MSD slope and retain an independent VACF integral check."""

    if not isinstance(plan, DiffusionFitPlan):
        raise TypeError("plan must be DiffusionFitPlan.")
    if not isinstance(correlation, LaggedCorrelationResult):
        raise TypeError("correlation must be LaggedCorrelationResult.")
    if plan.last_lag_index >= correlation.lag_steps.size:
        raise ValueError("Diffusion fit window exceeds the lagged correlation result.")
    selected = slice(plan.first_lag_index, plan.last_lag_index + 1)
    time = correlation.lag_times[selected]
    msd = correlation.mean_squared_displacement[selected]
    weight = correlation.origin_counts[selected].astype(msd.dtype)
    total_weight = jnp.sum(weight)
    mean_time = jnp.sum(weight * time) / total_weight
    mean_msd = jnp.sum(weight * msd) / total_weight
    centered_time = time - mean_time
    centered_msd = msd - mean_msd
    denominator = jnp.sum(weight * centered_time * centered_time)
    slope = jnp.sum(weight * centered_time * centered_msd) / denominator
    intercept = mean_msd - slope * mean_time
    residual = msd - (intercept + slope * time)
    residual_sum = jnp.sum(weight * residual * residual)
    total_sum = jnp.sum(weight * centered_msd * centered_msd)
    r_squared = 1.0 - residual_sum / jnp.maximum(total_sum, jnp.finfo(msd.dtype).tiny)
    point_count = jnp.asarray(time.size, dtype=jnp.int32)
    residual_variance = residual_sum / jnp.maximum(
        point_count.astype(msd.dtype) - 2.0, 1.0
    )
    slope_standard_error = jnp.sqrt(
        residual_variance / jnp.maximum(denominator, jnp.finfo(msd.dtype).tiny)
    )
    factor = 2.0 * plan.spatial_dimension
    diffusion = slope / factor
    standard_error = slope_standard_error / factor
    all_times = correlation.lag_times
    all_vacf = correlation.velocity_autocorrelation
    green_kubo = (
        jnp.sum(0.5 * (all_vacf[1:] + all_vacf[:-1]) * (all_times[1:] - all_times[:-1]))
        / plan.spatial_dimension
    )
    comparison_scale = jnp.maximum(
        jnp.maximum(jnp.abs(diffusion), jnp.abs(green_kubo)),
        jnp.finfo(msd.dtype).tiny,
    )
    consistency = jnp.abs(diffusion - green_kubo) / comparison_scale
    relative_error = standard_error / jnp.maximum(
        jnp.abs(diffusion), jnp.finfo(msd.dtype).tiny
    )
    minimum_origins = jnp.min(correlation.origin_counts[selected])
    finite = jnp.all(
        jnp.isfinite(
            jnp.asarray(
                [
                    diffusion,
                    standard_error,
                    slope,
                    intercept,
                    r_squared,
                    green_kubo,
                    consistency,
                ]
            )
        )
    )
    successful = (
        correlation.successful
        & finite
        & (denominator > 0.0)
        & (slope > 0.0)
        & (minimum_origins >= plan.minimum_origins)
        & (r_squared >= plan.minimum_r_squared)
        & (relative_error <= plan.maximum_relative_standard_error)
        & (consistency <= plan.maximum_einstein_green_kubo_relative_error)
    )
    return DiffusionEvidence(
        diffusion,
        standard_error,
        slope,
        intercept,
        r_squared,
        green_kubo,
        consistency,
        point_count,
        minimum_origins,
        successful,
        plan.plan_id,
        correlation.plan_id,
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
    "DiffusionEvidence",
    "DiffusionFitPlan",
    "DisplacementCorrelationState",
    "RadialDistributionPlan",
    "RadialDistributionResult",
    "RadialDistributionState",
    "ThermodynamicAccumulator",
    "LaggedCorrelationPlan",
    "LaggedCorrelationResult",
    "ThermodynamicSummary",
    "StaticStructureFactorPlan",
    "StaticStructureFactorResult",
    "atomistic_trajectory_data",
    "fit_diffusion",
    "lagged_msd_vacf",
    "displacement_correlation",
    "radial_distribution",
    "static_structure_factor",
    "summarize_thermodynamics",
]
