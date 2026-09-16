#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule


class ShearRheologyPlan(StrictModule):
    flow_axis: int = eqx.field(static=True)
    gradient_axis: int = eqx.field(static=True)
    steady_fraction: float = eqx.field(static=True)
    minimum_samples: int = eqx.field(static=True)
    maximum_relative_standard_error: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        flow_axis: int = 0,
        gradient_axis: int = 1,
        steady_fraction: float = 0.25,
        minimum_samples: int = 16,
        maximum_relative_standard_error: float = 0.1,
    ):
        flow = int(flow_axis)
        gradient = int(gradient_axis)
        fraction = float(steady_fraction)
        samples = int(minimum_samples)
        relative = float(maximum_relative_standard_error)
        if flow not in range(3) or gradient not in range(3) or flow == gradient:
            raise ValueError(
                "Shear-rheology axes must be distinct three-dimensional axes."
            )
        if (
            not 0.0 < fraction <= 1.0
            or samples < 4
            or not math.isfinite(relative)
            or relative <= 0.0
        ):
            raise ValueError("Shear-rheology sampling controls are invalid.")
        self.flow_axis = flow
        self.gradient_axis = gradient
        self.steady_fraction = fraction
        self.minimum_samples = samples
        self.maximum_relative_standard_error = relative
        self.plan_id = canonical_fingerprint(
            {
                "kind": "shear-rheology-analysis",
                "flow_axis": flow,
                "gradient_axis": gradient,
                "steady_fraction": fraction,
                "minimum_samples": samples,
                "maximum_relative_standard_error": relative,
            }
        )


class ShearRheologyResult(StrictModule):
    apparent_viscosity: Array
    first_normal_stress_difference: Array
    second_normal_stress_difference: Array
    stress_overshoot_ratio: Array
    steady_shear_stress: Array
    standard_error: Array
    relative_standard_error: Array
    steady_sample_count: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


def shear_rheology(
    plan: ShearRheologyPlan,
    cauchy_stress: ArrayLike,
    shear_rate: ArrayLike,
    /,
) -> ShearRheologyResult:
    if not isinstance(plan, ShearRheologyPlan):
        raise TypeError("plan must be ShearRheologyPlan.")
    stress = jnp.asarray(cauchy_stress)
    rate = jnp.asarray(shear_rate, dtype=stress.dtype)
    if stress.ndim != 3 or stress.shape[1:] != (3, 3):
        raise ValueError("cauchy_stress must have shape (sample, 3, 3).")
    if rate.ndim == 0:
        rate = jnp.full((stress.shape[0],), rate, dtype=stress.dtype)
    if rate.shape != (stress.shape[0],):
        raise ValueError("shear_rate must be scalar or match the stress samples.")
    if stress.shape[0] < plan.minimum_samples:
        raise ValueError("Too few samples for shear-rheology analysis.")
    steady_count = max(2, int(math.ceil(plan.steady_fraction * stress.shape[0])))
    shear = stress[:, plan.flow_axis, plan.gradient_axis]
    steady_shear = shear[-steady_count:]
    steady_rate = rate[-steady_count:]
    rate_mean = jnp.mean(steady_rate)
    shear_mean = jnp.mean(steady_shear)
    viscosity = shear_mean / rate_mean
    flow = plan.flow_axis
    gradient = plan.gradient_axis
    vorticity_axis = next(axis for axis in range(3) if axis not in (flow, gradient))
    first = jnp.mean(
        stress[-steady_count:, flow, flow] - stress[-steady_count:, gradient, gradient]
    )
    second = jnp.mean(
        stress[-steady_count:, gradient, gradient]
        - stress[-steady_count:, vorticity_axis, vorticity_axis]
    )
    standard_error = jnp.std(steady_shear, ddof=1) / jnp.sqrt(steady_count)
    relative_error = standard_error / jnp.maximum(
        jnp.abs(shear_mean), jnp.finfo(stress.dtype).tiny
    )
    signed_peak = jnp.max(jnp.sign(shear_mean) * shear)
    overshoot = signed_peak / jnp.maximum(
        jnp.abs(shear_mean), jnp.finfo(stress.dtype).tiny
    )
    finite = (
        jnp.isfinite(viscosity)
        & jnp.isfinite(first)
        & jnp.isfinite(second)
        & jnp.isfinite(overshoot)
        & jnp.isfinite(relative_error)
    )
    successful = (
        finite
        & (jnp.abs(rate_mean) > jnp.finfo(stress.dtype).tiny)
        & (relative_error <= plan.maximum_relative_standard_error)
    )
    return ShearRheologyResult(
        viscosity,
        first,
        second,
        overshoot,
        shear_mean,
        standard_error,
        relative_error,
        jnp.asarray(steady_count, dtype=jnp.int32),
        finite,
        successful,
        plan.plan_id,
    )


class LAOSAnalysisPlan(StrictModule):
    strain_amplitude: float = eqx.field(static=True)
    angular_frequency: float = eqx.field(static=True)
    harmonic_count: int = eqx.field(static=True)
    discard_cycles: int = eqx.field(static=True)
    minimum_analysis_cycles: int = eqx.field(static=True)
    closure_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        strain_amplitude: float,
        angular_frequency: float,
        /,
        *,
        harmonic_count: int = 9,
        discard_cycles: int = 1,
        minimum_analysis_cycles: int = 2,
        closure_tolerance: float = 1.0e-3,
    ):
        amplitude = float(strain_amplitude)
        frequency = float(angular_frequency)
        harmonics = int(harmonic_count)
        discard = int(discard_cycles)
        cycles = int(minimum_analysis_cycles)
        closure = float(closure_tolerance)
        if (
            not math.isfinite(amplitude)
            or amplitude <= 0.0
            or not math.isfinite(frequency)
            or frequency <= 0.0
            or harmonics <= 0
            or discard < 0
            or cycles <= 0
            or not math.isfinite(closure)
            or closure <= 0.0
        ):
            raise ValueError("LAOS controls are invalid.")
        self.strain_amplitude = amplitude
        self.angular_frequency = frequency
        self.harmonic_count = harmonics
        self.discard_cycles = discard
        self.minimum_analysis_cycles = cycles
        self.closure_tolerance = closure
        self.plan_id = canonical_fingerprint(
            {
                "kind": "laos-analysis",
                "strain_amplitude": amplitude,
                "angular_frequency": frequency,
                "harmonic_count": harmonics,
                "discard_cycles": discard,
                "minimum_analysis_cycles": cycles,
                "closure_tolerance": closure,
            }
        )


class LAOSAnalysisResult(StrictModule):
    harmonic_orders: Array
    storage_moduli: Array
    loss_moduli: Array
    harmonic_magnitudes: Array
    fundamental_phase_lag: Array
    total_harmonic_distortion: Array
    dissipated_energy_per_cycle: Array
    analyzed_cycle_count: Array
    strain_closure_residual: Array
    time_uniformity_residual: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


def _trapezoid(values: Array, times: Array) -> Array:
    spacing = times[1:] - times[:-1]
    weight = spacing.reshape((spacing.size,) + (1,) * (values.ndim - 1))
    return jnp.sum(0.5 * (values[1:] + values[:-1]) * weight, axis=0)


def analyze_laos(
    plan: LAOSAnalysisPlan,
    times: ArrayLike,
    strain: ArrayLike,
    shear_stress: ArrayLike,
    /,
) -> LAOSAnalysisResult:
    if not isinstance(plan, LAOSAnalysisPlan):
        raise TypeError("plan must be LAOSAnalysisPlan.")
    time = jnp.asarray(times)
    strain_value = jnp.asarray(strain, dtype=time.dtype)
    stress = jnp.asarray(shear_stress, dtype=time.dtype)
    if time.ndim != 1 or strain_value.shape != time.shape or stress.shape != time.shape:
        raise ValueError("LAOS time, strain, and stress must be matching rank-1 arrays.")
    if time.size < 8:
        raise ValueError("LAOS analysis requires at least eight samples.")
    host_time = np.asarray(time)
    if np.any(~np.isfinite(host_time)) or np.any(np.diff(host_time) <= 0.0):
        raise ValueError("LAOS times must be finite and strictly increasing.")
    period = 2.0 * math.pi / plan.angular_frequency
    analysis_start = float(host_time[0]) + plan.discard_cycles * period
    start_index = int(np.searchsorted(host_time, analysis_start, side="left"))
    analysis_time = time[start_index:]
    analysis_strain = strain_value[start_index:]
    analysis_stress = stress[start_index:]
    duration = analysis_time[-1] - analysis_time[0]
    cycle_count = duration / period
    if analysis_time.size < 8:
        raise ValueError("Too few post-transient LAOS samples.")
    spacing = jnp.diff(analysis_time)
    time_uniformity = jnp.max(jnp.abs(spacing - jnp.mean(spacing))) / jnp.mean(spacing)
    orders = jnp.arange(1, plan.harmonic_count + 1, dtype=analysis_time.dtype)
    phase = plan.angular_frequency * (analysis_time - analysis_time[0])
    sine = jnp.sin(phase[:, None] * orders[None, :])
    cosine = jnp.cos(phase[:, None] * orders[None, :])
    storage = (
        2.0
        * _trapezoid(analysis_stress[:, None] * sine, analysis_time)
        / duration
        / plan.strain_amplitude
    )
    loss = (
        2.0
        * _trapezoid(analysis_stress[:, None] * cosine, analysis_time)
        / duration
        / plan.strain_amplitude
    )
    magnitude = jnp.sqrt(storage * storage + loss * loss)
    phase_lag = jnp.arctan2(loss[0], storage[0])
    distortion = jnp.sqrt(jnp.sum(magnitude[1:] ** 2)) / jnp.maximum(
        magnitude[0], jnp.finfo(time.dtype).tiny
    )
    energy = jnp.pi * plan.strain_amplitude**2 * loss[0]
    closure = jnp.abs(analysis_strain[-1] - analysis_strain[0]) / plan.strain_amplitude
    finite = (
        jnp.all(jnp.isfinite(storage))
        & jnp.all(jnp.isfinite(loss))
        & jnp.isfinite(energy)
        & jnp.isfinite(cycle_count)
        & jnp.isfinite(time_uniformity)
    )
    successful = (
        finite
        & (cycle_count >= plan.minimum_analysis_cycles)
        & (jnp.abs(cycle_count - jnp.rint(cycle_count)) <= plan.closure_tolerance)
        & (closure <= plan.closure_tolerance)
        & (time_uniformity <= plan.closure_tolerance)
    )
    return LAOSAnalysisResult(
        orders.astype(jnp.int32),
        storage,
        loss,
        magnitude,
        phase_lag,
        distortion,
        energy,
        cycle_count,
        closure,
        time_uniformity,
        finite,
        successful,
        plan.plan_id,
    )


__all__ = [
    "LAOSAnalysisPlan",
    "LAOSAnalysisResult",
    "ShearRheologyPlan",
    "ShearRheologyResult",
    "analyze_laos",
    "shear_rheology",
]
