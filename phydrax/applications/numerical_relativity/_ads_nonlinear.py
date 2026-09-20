#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Nonlinear spherically symmetric Einstein–scalar evolution in compactified AdS."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._interpolation import linear_interpolate
from ..._strict import StrictModule
from ._generalized_wave_gauge import GeneralizedWaveGaugePlan


SphericalAdSBoundaryPolicy: TypeAlias = Literal["reflecting", "driven", "dissipative"]
SphericalAdSRunStatus: TypeAlias = Literal["complete", "horizon", "nonfinite"]


class SphericalConformalAdSPlan(StrictModule):
    """Bounded 3+1 spherical Einstein–scalar initial-boundary value problem."""

    radial_points: Array
    time_step: float = eqx.field(static=True)
    steps: int = eqx.field(static=True)
    ads_length: float = eqx.field(static=True)
    gravitational_coupling: float = eqx.field(static=True)
    scalar_mass_squared: float = eqx.field(static=True)
    boundary_policy: SphericalAdSBoundaryPolicy = eqx.field(static=True)
    drive_amplitude: float = eqx.field(static=True)
    drive_frequency: float = eqx.field(static=True)
    gauge: GeneralizedWaveGaugePlan | None
    gauge_relaxation: float = eqx.field(static=True)
    horizon_threshold: float = eqx.field(static=True)
    maximum_state_elements: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        radial_points: ArrayLike,
        /,
        *,
        time_step: float,
        steps: int,
        ads_length: float = 1.0,
        gravitational_coupling: float = 1.0,
        scalar_mass_squared: float = 0.0,
        boundary_policy: SphericalAdSBoundaryPolicy = "reflecting",
        drive_amplitude: float = 0.0,
        drive_frequency: float = 1.0,
        gauge: GeneralizedWaveGaugePlan | None = None,
        gauge_relaxation: float = 1.0,
        horizon_threshold: float = 0.05,
        maximum_state_elements: int = 10_000_000,
    ):
        points = np.asarray(radial_points, dtype=np.float64)
        step = float(time_step)
        count = int(steps)
        length = float(ads_length)
        coupling = float(gravitational_coupling)
        mass = float(scalar_mass_squared)
        amplitude = float(drive_amplitude)
        frequency = float(drive_frequency)
        relaxation = float(gauge_relaxation)
        threshold = float(horizon_threshold)
        maximum = int(maximum_state_elements)
        if points.ndim != 1 or points.size < 9:
            raise ValueError(
                "radial_points must be a one-dimensional grid of at least nine points."
            )
        spacing = np.diff(points)
        if (
            points[0] != 0.0
            or points[-1] >= 0.5 * np.pi
            or np.any(spacing <= 0.0)
            or not np.allclose(spacing, spacing[0], rtol=1e-12, atol=1e-14)
        ):
            raise ValueError("Spherical AdS requires a uniform grid in [0, pi/2).")
        if boundary_policy not in ("reflecting", "driven", "dissipative"):
            raise ValueError("Unknown spherical AdS boundary policy.")
        if gauge is not None and not isinstance(gauge, GeneralizedWaveGaugePlan):
            raise TypeError("gauge must be GeneralizedWaveGaugePlan or None.")
        scalars = (
            step,
            length,
            coupling,
            mass,
            amplitude,
            frequency,
            relaxation,
            threshold,
        )
        if not all(math.isfinite(value) for value in scalars):
            raise ValueError("Spherical AdS scalar controls must be finite.")
        if (
            step <= 0.0
            or count < 1
            or length <= 0.0
            or coupling <= 0.0
            or frequency <= 0.0
            or relaxation < 0.0
            or not 0.0 < threshold < 1.0
            or step > 0.25 * spacing[0] * length
            or maximum < 7 * points.size
        ):
            raise ValueError(
                "Spherical AdS time, resource, or stability controls are invalid."
            )
        if boundary_policy != "driven" and amplitude != 0.0:
            raise ValueError(
                "Boundary drive amplitude is valid only for driven boundaries."
            )
        content = {
            "kind": "spherical-conformal-ads-plan",
            "radial_points": array_tree_fingerprint(points),
            "time_step": step,
            "steps": count,
            "ads_length": length,
            "gravitational_coupling": coupling,
            "scalar_mass_squared": mass,
            "boundary_policy": boundary_policy,
            "drive_amplitude": amplitude,
            "drive_frequency": frequency,
            "gauge": None if gauge is None else gauge.gauge_id,
            "gauge_relaxation": relaxation,
            "horizon_threshold": threshold,
            "maximum_state_elements": maximum,
        }
        self.radial_points = jnp.asarray(points)
        self.time_step = step
        self.steps = count
        self.ads_length = length
        self.gravitational_coupling = coupling
        self.scalar_mass_squared = mass
        self.boundary_policy = boundary_policy
        self.drive_amplitude = amplitude
        self.drive_frequency = frequency
        self.gauge = gauge
        self.gauge_relaxation = relaxation
        self.horizon_threshold = threshold
        self.maximum_state_elements = maximum
        self.plan_id = canonical_fingerprint(content)

    @property
    def spacing(self) -> float:
        return float(self.radial_points[1] - self.radial_points[0])


class SphericalConformalAdSState(StrictModule):
    scalar: Array
    radial_derivative: Array
    momentum: Array
    metric_a: Array
    metric_delta: Array
    gauge_source: Array
    time: Array
    step: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: SphericalConformalAdSPlan,
        scalar: ArrayLike,
        radial_derivative: ArrayLike,
        momentum: ArrayLike,
        metric_a: ArrayLike,
        metric_delta: ArrayLike,
        gauge_source: ArrayLike,
        time: ArrayLike,
        step: ArrayLike,
        /,
    ):
        count = plan.radial_points.shape[0]
        scalar_ = jnp.asarray(scalar, dtype=plan.radial_points.dtype)
        radial = jnp.asarray(radial_derivative, dtype=scalar_.dtype)
        momentum_ = jnp.asarray(momentum, dtype=scalar_.dtype)
        metric_a_ = jnp.asarray(metric_a, dtype=scalar_.dtype)
        delta = jnp.asarray(metric_delta, dtype=scalar_.dtype)
        gauge = jnp.asarray(gauge_source, dtype=scalar_.dtype)
        if any(
            value.shape != (count,)
            for value in (scalar_, radial, momentum_, metric_a_, delta)
        ):
            raise ValueError("Spherical AdS fields must match radial_points.")
        if gauge.shape != (4, count):
            raise ValueError(
                "Spherical AdS gauge source must have shape (4, radial_points)."
            )
        self.scalar = scalar_
        self.radial_derivative = radial
        self.momentum = momentum_
        self.metric_a = metric_a_
        self.metric_delta = delta
        self.gauge_source = gauge
        self.time = jnp.asarray(time, dtype=scalar_.dtype).reshape(())
        self.step = jnp.asarray(step, dtype=jnp.int32).reshape(())
        self.plan_id = plan.plan_id


class SphericalAdSInitialDataEvidence(StrictModule):
    scalar_gradient_residual: Array
    center_regularity_residual: Array
    boundary_residual: Array
    minimum_metric_a: Array
    finite: Array
    accepted: Array
    evidence_id: str = eqx.field(static=True)


class SphericalAdSInitialData(StrictModule):
    state: SphericalConformalAdSState
    evidence: SphericalAdSInitialDataEvidence


def _first_derivative(values: Array, spacing: float, /) -> Array:
    derivative = jnp.zeros_like(values)
    derivative = derivative.at[1:-1].set((values[2:] - values[:-2]) / (2.0 * spacing))
    derivative = derivative.at[0].set((values[1] - values[0]) / spacing)
    derivative = derivative.at[-1].set((values[-1] - values[-2]) / spacing)
    return derivative


def _constraint_metric(
    plan: SphericalConformalAdSPlan,
    radial_derivative: Array,
    momentum: Array,
    /,
) -> tuple[Array, Array]:
    points = plan.radial_points
    spacing = plan.spacing
    energy = radial_derivative**2 + momentum**2
    metric_a = jnp.ones_like(points)
    delta = jnp.zeros_like(points)
    for index in range(points.shape[0] - 1):
        coordinate = points[index]
        sine = jnp.sin(coordinate)
        cosine = jnp.cos(coordinate)
        denominator = jnp.where(index == 0, 1.0, sine * cosine)
        geometric = ((1.0 + 2.0 * sine**2) / denominator) * (1.0 - metric_a[index])
        source = (
            plan.gravitational_coupling * sine * cosine * metric_a[index] * energy[index]
        )
        derivative_a = jnp.where(index == 0, 0.0, geometric - source)
        derivative_delta = -plan.gravitational_coupling * sine * cosine * energy[index]
        predicted_a = metric_a[index] + spacing * derivative_a
        next_coordinate = points[index + 1]
        next_sine = jnp.sin(next_coordinate)
        next_cosine = jnp.cos(next_coordinate)
        next_denominator = next_sine * next_cosine
        next_geometric = ((1.0 + 2.0 * next_sine**2) / next_denominator) * (
            1.0 - predicted_a
        )
        next_source = (
            plan.gravitational_coupling
            * next_sine
            * next_cosine
            * predicted_a
            * energy[index + 1]
        )
        next_derivative_a = next_geometric - next_source
        next_derivative_delta = (
            -plan.gravitational_coupling * next_sine * next_cosine * energy[index + 1]
        )
        metric_a = metric_a.at[index + 1].set(
            metric_a[index] + 0.5 * spacing * (derivative_a + next_derivative_a)
        )
        delta = delta.at[index + 1].set(
            delta[index] + 0.5 * spacing * (derivative_delta + next_derivative_delta)
        )
    delta = delta - delta[-1]
    return metric_a, delta


def _gauge_target(plan: SphericalConformalAdSPlan, time: Array, /) -> Array:
    count = plan.radial_points.shape[0]
    if plan.gauge is None:
        return jnp.zeros((4, count), dtype=plan.radial_points.dtype)
    return plan.gauge.source(time, plan.radial_points / plan.ads_length)


def prepare_spherical_ads_initial_data(
    plan: SphericalConformalAdSPlan,
    scalar: ArrayLike,
    momentum: ArrayLike,
    /,
    *,
    tolerance: float = 1e-9,
) -> SphericalAdSInitialData:
    """Project scalar data onto radial regularity and Einstein constraints."""

    if not isinstance(plan, SphericalConformalAdSPlan):
        raise TypeError("plan must be SphericalConformalAdSPlan.")
    scalar_ = jnp.asarray(scalar, dtype=plan.radial_points.dtype)
    momentum_ = jnp.asarray(momentum, dtype=scalar_.dtype)
    expected = (plan.radial_points.shape[0],)
    if scalar_.shape != expected or momentum_.shape != expected:
        raise ValueError("Initial scalar and momentum arrays must match radial_points.")
    scalar_ = scalar_.at[0].set(scalar_[1])
    momentum_ = momentum_.at[0].set(momentum_[1])
    if plan.boundary_policy == "reflecting":
        scalar_ = scalar_.at[-1].set(0.0)
        momentum_ = momentum_.at[-1].set(0.0)
    radial = _first_derivative(scalar_, plan.spacing).at[0].set(0.0)
    metric_a, delta = _constraint_metric(plan, radial, momentum_)
    gauge = _gauge_target(plan, jnp.asarray(0.0))
    state = SphericalConformalAdSState(
        plan,
        scalar_,
        radial,
        momentum_,
        metric_a,
        delta,
        gauge,
        0.0,
        0,
    )
    gradient_residual = jnp.linalg.norm(
        radial - _first_derivative(scalar_, plan.spacing).at[0].set(0.0)
    )
    center = jnp.abs(radial[0]) + jnp.abs(momentum_[0] - momentum_[1])
    boundary = (
        jnp.abs(scalar_[-1]) + jnp.abs(momentum_[-1])
        if plan.boundary_policy == "reflecting"
        else jnp.asarray(0.0)
    )
    minimum = jnp.min(metric_a)
    finite = all(
        jnp.all(jnp.isfinite(value))
        for value in (scalar_, radial, momentum_, metric_a, delta, gauge)
    )
    accepted = (
        finite
        & (gradient_residual <= tolerance)
        & (center <= tolerance)
        & (boundary <= tolerance)
        & (minimum > 0.0)
    )
    evidence_id = canonical_fingerprint(
        {
            "kind": "spherical-ads-initial-data-evidence",
            "plan": plan.plan_id,
            "scalar": array_tree_fingerprint(np.asarray(scalar_)),
            "momentum": array_tree_fingerprint(np.asarray(momentum_)),
            "tolerance": float(tolerance),
        }
    )
    evidence = SphericalAdSInitialDataEvidence(
        scalar_gradient_residual=gradient_residual,
        center_regularity_residual=center,
        boundary_residual=boundary,
        minimum_metric_a=minimum,
        finite=jnp.asarray(finite),
        accepted=accepted,
        evidence_id=evidence_id,
    )
    return SphericalAdSInitialData(state=state, evidence=evidence)


def gaussian_spherical_ads_initial_data(
    plan: SphericalConformalAdSPlan,
    /,
    *,
    amplitude: float,
    center: float,
    width: float,
) -> SphericalAdSInitialData:
    points = plan.radial_points
    amplitude_ = float(amplitude)
    center_ = float(center)
    width_ = float(width)
    if width_ <= 0.0 or not all(
        math.isfinite(value) for value in (amplitude_, center_, width_)
    ):
        raise ValueError("Gaussian initial-data controls are invalid.")
    profile = amplitude_ * jnp.exp(-(((points - center_) / width_) ** 2))
    profile = profile * jnp.cos(points) ** 3
    return prepare_spherical_ads_initial_data(plan, profile, jnp.zeros_like(profile))


def _enforce_boundary(
    plan: SphericalConformalAdSPlan,
    scalar: Array,
    radial: Array,
    momentum: Array,
    time: Array,
    /,
) -> tuple[Array, Array, Array]:
    scalar = scalar.at[0].set(scalar[1])
    radial = radial.at[0].set(0.0)
    momentum = momentum.at[0].set(momentum[1])
    if plan.boundary_policy == "reflecting":
        scalar = scalar.at[-1].set(0.0)
        momentum = momentum.at[-1].set(0.0)
    elif plan.boundary_policy == "driven":
        drive = plan.drive_amplitude * jnp.sin(plan.drive_frequency * time)
        momentum = momentum.at[-1].set(drive)
    else:
        momentum = momentum.at[-1].set(-radial[-1])
    return scalar, radial, momentum


def _rhs(
    plan: SphericalConformalAdSPlan,
    scalar: Array,
    radial: Array,
    momentum: Array,
    gauge: Array,
    time: Array,
    /,
) -> tuple[Array, Array, Array, Array]:
    metric_a, delta = _constraint_metric(plan, radial, momentum)
    lapse = metric_a * jnp.exp(-delta)
    scalar_rate = lapse * momentum
    radial_rate = _first_derivative(lapse * momentum, plan.spacing)
    points = plan.radial_points
    weight = jnp.tan(points) ** 2
    flux = weight * lapse * radial
    momentum_rate = jnp.zeros_like(momentum)
    momentum_rate = momentum_rate.at[1:-1].set(
        (flux[2:] - flux[:-2]) / (2.0 * plan.spacing * weight[1:-1])
    )
    momentum_rate = momentum_rate.at[0].set(3.0 * lapse[0] * radial[1] / plan.spacing)
    momentum_rate = momentum_rate - (
        plan.scalar_mass_squared * scalar / (plan.ads_length**2 * jnp.cos(points) ** 2)
    )
    target = _gauge_target(plan, time)
    gauge_rate = -plan.gauge_relaxation * (gauge - target)
    scalar_rate, radial_rate, momentum_rate = _enforce_boundary(
        plan,
        scalar_rate,
        radial_rate,
        momentum_rate,
        time,
    )
    return scalar_rate, radial_rate, momentum_rate, gauge_rate


def _rk4_step(
    plan: SphericalConformalAdSPlan,
    state: SphericalConformalAdSState,
    /,
) -> SphericalConformalAdSState:
    dt = plan.time_step / plan.ads_length
    fields = (state.scalar, state.radial_derivative, state.momentum, state.gauge_source)
    first = _rhs(plan, fields[0], fields[1], fields[2], fields[3], state.time)
    second_fields = tuple(
        value + 0.5 * dt * rate for value, rate in zip(fields, first, strict=True)
    )
    second = _rhs(
        plan,
        second_fields[0],
        second_fields[1],
        second_fields[2],
        second_fields[3],
        state.time + 0.5 * dt,
    )
    third_fields = tuple(
        value + 0.5 * dt * rate for value, rate in zip(fields, second, strict=True)
    )
    third = _rhs(
        plan,
        third_fields[0],
        third_fields[1],
        third_fields[2],
        third_fields[3],
        state.time + 0.5 * dt,
    )
    fourth_fields = tuple(
        value + dt * rate for value, rate in zip(fields, third, strict=True)
    )
    fourth = _rhs(
        plan,
        fourth_fields[0],
        fourth_fields[1],
        fourth_fields[2],
        fourth_fields[3],
        state.time + dt,
    )
    next_fields = tuple(
        value + dt * (a + 2.0 * b + 2.0 * c + d) / 6.0
        for value, a, b, c, d in zip(fields, first, second, third, fourth, strict=True)
    )
    scalar, radial, momentum, gauge = next_fields
    scalar, radial, momentum = _enforce_boundary(
        plan,
        scalar,
        radial,
        momentum,
        state.time + dt,
    )
    metric_a, delta = _constraint_metric(plan, radial, momentum)
    return SphericalConformalAdSState(
        plan,
        scalar,
        radial,
        momentum,
        metric_a,
        delta,
        gauge,
        state.time + dt,
        state.step + 1,
    )


def spherical_ads_mass(
    state: SphericalConformalAdSState, plan: SphericalConformalAdSPlan, /
) -> Array:
    if state.plan_id != plan.plan_id:
        raise ValueError("State and plan identities differ.")
    points = plan.radial_points
    sine = jnp.sin(points)
    cosine = jnp.cos(points)
    denominator = jnp.maximum(cosine**3, jnp.finfo(points.dtype).tiny)
    mass = 0.5 * (1.0 - state.metric_a) * sine / denominator
    return mass[-1]


class SphericalAdSEvolutionEvidence(StrictModule):
    initial_mass: Array
    final_mass: Array
    relative_mass_change: Array
    minimum_metric_a: Array
    maximum_gauge_residual: Array
    boundary_flux: Array
    finite: Array
    status: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class SphericalConformalAdSRun(StrictModule):
    scalar_history: Array
    momentum_history: Array
    metric_a_history: Array
    metric_delta_history: Array
    gauge_history: Array
    times: Array
    final_state: SphericalConformalAdSState
    evidence: SphericalAdSEvolutionEvidence
    plan_id: str = eqx.field(static=True)


def run_spherical_conformal_ads(
    plan: SphericalConformalAdSPlan,
    initial: SphericalAdSInitialData | SphericalConformalAdSState,
    /,
) -> SphericalConformalAdSRun:
    """Evolve nonlinear spherical Einstein–scalar fields with atomic RK stages."""

    if not isinstance(plan, SphericalConformalAdSPlan):
        raise TypeError("plan must be SphericalConformalAdSPlan.")
    state = initial.state if isinstance(initial, SphericalAdSInitialData) else initial
    if not isinstance(state, SphericalConformalAdSState) or state.plan_id != plan.plan_id:
        raise ValueError("Initial state does not match the spherical AdS plan.")
    initial_mass = spherical_ads_mass(state, plan)
    scalar_history = [state.scalar]
    momentum_history = [state.momentum]
    metric_a_history = [state.metric_a]
    delta_history = [state.metric_delta]
    gauge_history = [state.gauge_source]
    times = [state.time]
    boundary_flux = jnp.asarray(0.0, dtype=state.scalar.dtype)
    status: SphericalAdSRunStatus = "complete"
    for _ in range(plan.steps):
        previous = state
        state = _rk4_step(plan, state)
        flux = (
            state.metric_a[-1]
            * jnp.exp(-state.metric_delta[-1])
            * state.momentum[-1]
            * state.radial_derivative[-1]
        )
        boundary_flux = boundary_flux + plan.time_step * flux
        scalar_history.append(state.scalar)
        momentum_history.append(state.momentum)
        metric_a_history.append(state.metric_a)
        delta_history.append(state.metric_delta)
        gauge_history.append(state.gauge_source)
        times.append(state.time)
        if not bool(
            all(
                jnp.all(jnp.isfinite(value))
                for value in (
                    state.scalar,
                    state.momentum,
                    state.metric_a,
                    state.metric_delta,
                )
            )
        ):
            status = "nonfinite"
            state = previous
            break
        if float(jnp.min(state.metric_a)) <= plan.horizon_threshold:
            status = "horizon"
            break
    final_mass = spherical_ads_mass(state, plan)
    relative = jnp.abs(final_mass - initial_mass - boundary_flux) / jnp.maximum(
        1.0,
        jnp.abs(initial_mass),
    )
    target = _gauge_target(plan, state.time)
    gauge_residual = jnp.max(jnp.abs(state.gauge_source - target))
    finite = jnp.isfinite(relative) & jnp.isfinite(gauge_residual)
    evidence_id = canonical_fingerprint(
        {
            "kind": "spherical-ads-evolution-evidence",
            "plan": plan.plan_id,
            "status": status,
            "completed_steps": len(times) - 1,
        }
    )
    evidence = SphericalAdSEvolutionEvidence(
        initial_mass=initial_mass,
        final_mass=final_mass,
        relative_mass_change=relative,
        minimum_metric_a=jnp.min(state.metric_a),
        maximum_gauge_residual=gauge_residual,
        boundary_flux=boundary_flux,
        finite=finite,
        status=status,
        claim="nonlinear-spherical-classical-einstein-ads-only",
        evidence_id=evidence_id,
    )
    return SphericalConformalAdSRun(
        scalar_history=jnp.stack(tuple(scalar_history)),
        momentum_history=jnp.stack(tuple(momentum_history)),
        metric_a_history=jnp.stack(tuple(metric_a_history)),
        metric_delta_history=jnp.stack(tuple(delta_history)),
        gauge_history=jnp.stack(tuple(gauge_history)),
        times=jnp.stack(tuple(times)),
        final_state=state,
        evidence=evidence,
        plan_id=plan.plan_id,
    )


class SphericalAdSRefinementEvidence(StrictModule):
    refined_state: SphericalConformalAdSState
    scalar_integral_residual: Array
    momentum_integral_residual: Array
    constraint_residual: Array
    accepted: Array
    evidence_id: str = eqx.field(static=True)


def refine_spherical_ads_state(
    coarse_plan: SphericalConformalAdSPlan,
    fine_plan: SphericalConformalAdSPlan,
    state: SphericalConformalAdSState,
    /,
) -> SphericalAdSRefinementEvidence:
    """Interpolate one radial state and recompute, rather than interpolate, constraints."""

    if state.plan_id != coarse_plan.plan_id:
        raise ValueError("Coarse state and plan identities differ.")
    coarse = np.asarray(coarse_plan.radial_points)
    fine = np.asarray(fine_plan.radial_points)
    if fine[0] != coarse[0] or fine[-1] != coarse[-1] or fine.size <= coarse.size:
        raise ValueError("Fine radial grid must strictly refine the same interval.")
    scalar = linear_interpolate(coarse, state.scalar, fine, bounds="clip").values
    momentum = linear_interpolate(coarse, state.momentum, fine, bounds="clip").values
    radial = _first_derivative(scalar, fine_plan.spacing).at[0].set(0.0)
    metric_a, delta = _constraint_metric(fine_plan, radial, momentum)
    gauge = jnp.stack(
        tuple(
            linear_interpolate(coarse, component, fine, bounds="clip").values
            for component in state.gauge_source
        )
    )
    refined = SphericalConformalAdSState(
        fine_plan,
        scalar,
        radial,
        momentum,
        metric_a,
        delta,
        gauge,
        state.time,
        state.step,
    )
    coarse_scalar_integral = np.trapezoid(np.asarray(state.scalar), coarse)
    fine_scalar_integral = np.trapezoid(np.asarray(scalar), fine)
    coarse_momentum_integral = np.trapezoid(np.asarray(state.momentum), coarse)
    fine_momentum_integral = np.trapezoid(np.asarray(momentum), fine)
    scalar_residual = abs(fine_scalar_integral - coarse_scalar_integral)
    momentum_residual = abs(fine_momentum_integral - coarse_momentum_integral)
    constraint = jnp.linalg.norm(
        refined.radial_derivative
        - _first_derivative(refined.scalar, fine_plan.spacing).at[0].set(0.0)
    )
    accepted = jnp.isfinite(constraint)
    evidence_id = canonical_fingerprint(
        {
            "kind": "spherical-ads-refinement-evidence",
            "coarse_plan": coarse_plan.plan_id,
            "fine_plan": fine_plan.plan_id,
            "scalar_integral_residual": scalar_residual,
            "momentum_integral_residual": momentum_residual,
        }
    )
    return SphericalAdSRefinementEvidence(
        refined_state=refined,
        scalar_integral_residual=jnp.asarray(scalar_residual),
        momentum_integral_residual=jnp.asarray(momentum_residual),
        constraint_residual=constraint,
        accepted=accepted,
        evidence_id=evidence_id,
    )


class SphericalAdSShard(StrictModule):
    scalar: Array
    radial_derivative: Array
    momentum: Array
    metric_a: Array
    metric_delta: Array
    gauge_source: Array
    halo_start: int = eqx.field(static=True)
    core_start: int = eqx.field(static=True)
    core_stop: int = eqx.field(static=True)
    halo_stop: int = eqx.field(static=True)


class SphericalAdSDistributedState(StrictModule):
    shards: tuple[SphericalAdSShard, ...]
    partition_count: int = eqx.field(static=True)
    halo_width: int = eqx.field(static=True)
    reconstruction_residual: Array
    time: Array
    step: Array
    plan_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


def partition_spherical_ads_state(
    plan: SphericalConformalAdSPlan,
    state: SphericalConformalAdSState,
    /,
    *,
    partition_count: int,
    halo_width: int = 2,
) -> SphericalAdSDistributedState:
    """Partition the compactified radial domain with explicit overlapping halos."""

    if state.plan_id != plan.plan_id:
        raise ValueError("State and plan identities differ.")
    partitions = int(partition_count)
    halo = int(halo_width)
    count = plan.radial_points.shape[0]
    if partitions < 1 or partitions > count or halo < 1:
        raise ValueError("Radial partition count or halo width is invalid.")
    boundaries = np.linspace(0, count, partitions + 1, dtype=np.int64)
    shards: list[SphericalAdSShard] = []
    for index in range(partitions):
        core_start = int(boundaries[index])
        core_stop = int(boundaries[index + 1])
        halo_start = max(0, core_start - halo)
        halo_stop = min(count, core_stop + halo)
        values = slice(halo_start, halo_stop)
        shards.append(
            SphericalAdSShard(
                scalar=state.scalar[values],
                radial_derivative=state.radial_derivative[values],
                momentum=state.momentum[values],
                metric_a=state.metric_a[values],
                metric_delta=state.metric_delta[values],
                gauge_source=state.gauge_source[:, values],
                halo_start=halo_start,
                core_start=core_start,
                core_stop=core_stop,
                halo_stop=halo_stop,
            )
        )
    reconstructed = jnp.concatenate(
        tuple(
            shard.scalar[
                shard.core_start - shard.halo_start : shard.core_stop - shard.halo_start
            ]
            for shard in shards
        )
    )
    residual = jnp.linalg.norm(reconstructed - state.scalar)
    evidence_id = canonical_fingerprint(
        {
            "kind": "spherical-ads-distributed-state",
            "plan": plan.plan_id,
            "partition_count": partitions,
            "halo_width": halo,
            "boundaries": boundaries.tolist(),
        }
    )
    return SphericalAdSDistributedState(
        shards=tuple(shards),
        partition_count=partitions,
        halo_width=halo,
        reconstruction_residual=residual,
        time=state.time,
        step=state.step,
        plan_id=plan.plan_id,
        evidence_id=evidence_id,
    )


def assemble_spherical_ads_state(
    plan: SphericalConformalAdSPlan,
    distributed: SphericalAdSDistributedState,
    /,
) -> SphericalConformalAdSState:
    """Assemble core ownership once and recompute global radial constraints."""

    if distributed.plan_id != plan.plan_id:
        raise ValueError("Distributed state and plan identities differ.")

    def assemble_field(name: str) -> Array:
        return jnp.concatenate(
            tuple(
                getattr(shard, name)[
                    shard.core_start - shard.halo_start : shard.core_stop
                    - shard.halo_start
                ]
                for shard in distributed.shards
            )
        )

    scalar = assemble_field("scalar")
    momentum = assemble_field("momentum")
    gauge = jnp.concatenate(
        tuple(
            shard.gauge_source[
                :,
                shard.core_start - shard.halo_start : shard.core_stop - shard.halo_start,
            ]
            for shard in distributed.shards
        ),
        axis=1,
    )
    radial = _first_derivative(scalar, plan.spacing).at[0].set(0.0)
    metric_a, delta = _constraint_metric(plan, radial, momentum)
    return SphericalConformalAdSState(
        plan,
        scalar,
        radial,
        momentum,
        metric_a,
        delta,
        gauge,
        distributed.time,
        distributed.step,
    )


class FeffermanGrahamExtractionPlan(StrictModule):
    boundary_points: int = eqx.field(static=True)
    delta_minus: float = eqx.field(static=True)
    delta_plus: float = eqx.field(static=True)
    boundary_dimension: int = eqx.field(static=True)
    newton_constant: float = eqx.field(static=True)
    counterterm_scheme: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        boundary_points: int,
        delta_minus: float,
        delta_plus: float,
        boundary_dimension: int,
        /,
        *,
        newton_constant: float,
        counterterm_scheme: str = "minimal-local-ads",
    ):
        count = int(boundary_points)
        lower = float(delta_minus)
        upper = float(delta_plus)
        dimension = int(boundary_dimension)
        constant = float(newton_constant)
        scheme = str(counterterm_scheme).strip()
        if count < 3 or lower >= upper or dimension < 2 or constant <= 0.0 or not scheme:
            raise ValueError("Fefferman–Graham extraction controls are invalid.")
        content = {
            "kind": "fefferman-graham-extraction-plan",
            "boundary_points": count,
            "delta_minus": lower,
            "delta_plus": upper,
            "boundary_dimension": dimension,
            "newton_constant": constant,
            "counterterm_scheme": scheme,
        }
        self.boundary_points = count
        self.delta_minus = lower
        self.delta_plus = upper
        self.boundary_dimension = dimension
        self.newton_constant = constant
        self.counterterm_scheme = scheme
        self.plan_id = canonical_fingerprint(content)


class FeffermanGrahamEvidence(StrictModule):
    scalar_source: Array
    scalar_response: Array
    mass_aspect: Array
    stress_tensor: Array
    scalar_fit_residual: Array
    metric_fit_residual: Array
    trace_residual: Array
    conservation_residual: Array
    accepted: Array
    evidence_id: str = eqx.field(static=True)


def extract_fefferman_graham_data(
    plan: FeffermanGrahamExtractionPlan,
    evolution: SphericalConformalAdSRun,
    radial_plan: SphericalConformalAdSPlan,
    /,
) -> FeffermanGrahamEvidence:
    """Fit scalar/source and mass coefficients and apply local AdS counterterms."""

    if not isinstance(plan, FeffermanGrahamExtractionPlan):
        raise TypeError("plan must be FeffermanGrahamExtractionPlan.")
    if evolution.plan_id != radial_plan.plan_id:
        raise ValueError("Evolution and radial plan identities differ.")
    count = plan.boundary_points
    rho = 0.5 * np.pi - np.asarray(radial_plan.radial_points[-count:])
    scalar_history = np.asarray(evolution.scalar_history[:, -count:])
    metric_history = np.asarray(evolution.metric_a_history[:, -count:])
    scalar_design = np.stack(
        (rho**plan.delta_minus, rho**plan.delta_plus),
        axis=1,
    )
    scalar_coefficients = np.stack(
        [
            np.linalg.lstsq(scalar_design, values, rcond=None)[0]
            for values in scalar_history
        ]
    )
    scalar_fitted = scalar_coefficients @ scalar_design.T
    scalar_residual = np.linalg.norm(scalar_history - scalar_fitted, axis=1) / np.maximum(
        1.0,
        np.linalg.norm(scalar_history, axis=1),
    )
    power = plan.boundary_dimension
    metric_design = (rho**power)[:, None]
    mass = np.stack(
        [
            np.linalg.lstsq(metric_design, 1.0 - values, rcond=None)[0]
            for values in metric_history
        ]
    )[:, 0]
    metric_fitted = mass[:, None] * metric_design.T
    metric_residual = np.linalg.norm(
        (1.0 - metric_history) - metric_fitted, axis=1
    ) / np.maximum(
        1.0,
        np.linalg.norm(1.0 - metric_history, axis=1),
    )
    energy = (plan.boundary_dimension - 1) * mass / (16.0 * np.pi * plan.newton_constant)
    pressure = energy / (plan.boundary_dimension - 1)
    stress = np.zeros(
        (energy.size, plan.boundary_dimension, plan.boundary_dimension),
        dtype=np.float64,
    )
    stress[:, 0, 0] = energy
    for index in range(1, plan.boundary_dimension):
        stress[:, index, index] = pressure
    trace = -stress[:, 0, 0] + np.trace(stress[:, 1:, 1:], axis1=1, axis2=2)
    time = np.asarray(evolution.times)
    conservation = (
        np.zeros_like(energy)
        if energy.size < 2
        else np.gradient(energy, time, edge_order=1)
    )
    accepted = bool(
        np.all(np.isfinite(stress))
        and np.max(scalar_residual) < 1.0
        and np.max(metric_residual) < 1.0
    )
    evidence_id = canonical_fingerprint(
        {
            "kind": "fefferman-graham-evidence",
            "plan": plan.plan_id,
            "evolution": evolution.evidence.evidence_id,
            "radial_plan": radial_plan.plan_id,
        }
    )
    return FeffermanGrahamEvidence(
        scalar_source=jnp.asarray(scalar_coefficients[:, 0]),
        scalar_response=jnp.asarray(scalar_coefficients[:, 1]),
        mass_aspect=jnp.asarray(mass),
        stress_tensor=jnp.asarray(stress),
        scalar_fit_residual=jnp.asarray(scalar_residual),
        metric_fit_residual=jnp.asarray(metric_residual),
        trace_residual=jnp.asarray(trace),
        conservation_residual=jnp.asarray(conservation),
        accepted=jnp.asarray(accepted),
        evidence_id=evidence_id,
    )


class SphericalAdSCampaignEvidence(StrictModule):
    amplitudes: Array
    statuses: tuple[str, ...] = eqx.field(static=True)
    minimum_metric_a: Array
    final_mass: Array
    critical_bracket: Array
    evidence_id: str = eqx.field(static=True)


def run_spherical_ads_collapse_campaign(
    plan: SphericalConformalAdSPlan,
    amplitudes: Sequence[float],
    /,
    *,
    center: float,
    width: float,
) -> SphericalAdSCampaignEvidence:
    values = tuple(float(value) for value in amplitudes)
    if (
        len(values) < 2
        or tuple(sorted(values)) != values
        or len(set(values)) != len(values)
    ):
        raise ValueError(
            "Collapse amplitudes must be unique, increasing, and nontrivial."
        )
    runs = tuple(
        run_spherical_conformal_ads(
            plan,
            gaussian_spherical_ads_initial_data(
                plan,
                amplitude=value,
                center=center,
                width=width,
            ),
        )
        for value in values
    )
    statuses = tuple(run.evidence.status for run in runs)
    collapsed = np.asarray([status == "horizon" for status in statuses])
    if np.any(collapsed) and np.any(~collapsed):
        lower = max(
            value for value, status in zip(values, collapsed, strict=True) if not status
        )
        upper = min(
            value for value, status in zip(values, collapsed, strict=True) if status
        )
        bracket = (lower, upper)
    else:
        bracket = (math.nan, math.nan)
    evidence_id = canonical_fingerprint(
        {
            "kind": "spherical-ads-collapse-campaign",
            "plan": plan.plan_id,
            "amplitudes": values,
            "statuses": statuses,
            "center": float(center),
            "width": float(width),
        }
    )
    return SphericalAdSCampaignEvidence(
        amplitudes=jnp.asarray(values),
        statuses=statuses,
        minimum_metric_a=jnp.asarray([run.evidence.minimum_metric_a for run in runs]),
        final_mass=jnp.asarray([run.evidence.final_mass for run in runs]),
        critical_bracket=jnp.asarray(bracket),
        evidence_id=evidence_id,
    )


__all__ = [
    "FeffermanGrahamEvidence",
    "FeffermanGrahamExtractionPlan",
    "SphericalAdSBoundaryPolicy",
    "SphericalAdSCampaignEvidence",
    "SphericalAdSEvolutionEvidence",
    "SphericalAdSInitialData",
    "SphericalAdSInitialDataEvidence",
    "SphericalAdSDistributedState",
    "SphericalAdSShard",
    "SphericalAdSRefinementEvidence",
    "SphericalAdSRunStatus",
    "SphericalConformalAdSPlan",
    "SphericalConformalAdSRun",
    "SphericalConformalAdSState",
    "assemble_spherical_ads_state",
    "extract_fefferman_graham_data",
    "gaussian_spherical_ads_initial_data",
    "partition_spherical_ads_state",
    "prepare_spherical_ads_initial_data",
    "refine_spherical_ads_state",
    "run_spherical_ads_collapse_campaign",
    "run_spherical_conformal_ads",
    "spherical_ads_mass",
]
