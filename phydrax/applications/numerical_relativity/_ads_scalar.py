#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-background conformal-cylinder scalar evolution and stress evidence."""

from __future__ import annotations

from math import pi

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax import ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule


class ConformalAdSScalarPlan(StrictModule):
    """One-dimensional Einstein-cylinder wave reference with reflecting boundaries."""

    radial_points: Array
    laplacian: Array
    ads_length: float = eqx.field(static=True)
    time_step: float = eqx.field(static=True)
    potential: float = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        point_count: int,
        /,
        *,
        ads_length: float = 1.0,
        time_step: float,
        potential: float = 0.0,
        maximum_steps: int = 1_000_000,
    ):
        count = int(point_count)
        length = float(ads_length)
        step = float(time_step)
        potential_value = float(potential)
        maximum = int(maximum_steps)
        if count < 5 or length <= 0.0 or step <= 0.0 or maximum < 1:
            raise ValueError(
                "Conformal scalar grid/time/resource parameters are invalid."
            )
        if not all(np.isfinite(value) for value in (length, step, potential_value)):
            raise ValueError("Conformal scalar parameters must be finite.")
        if potential_value <= -4.0:
            raise ValueError(
                "potential must keep the lowest reflecting normal-mode frequency real."
            )
        domain_length = 0.5 * pi
        spacing = domain_length / (count - 1)
        if step > 0.5 * spacing * length:
            raise ValueError("time_step violates the conservative scalar CFL bound.")
        points = np.linspace(0.0, domain_length, count)
        physical_spacing = spacing * length
        laplacian = np.zeros((count, count), dtype=np.float64)
        for index in range(1, count - 1):
            laplacian[index, index - 1] = 1.0 / physical_spacing**2
            laplacian[index, index] = -2.0 / physical_spacing**2
            laplacian[index, index + 1] = 1.0 / physical_spacing**2
        self.radial_points = jnp.asarray(points)
        self.laplacian = jnp.asarray(laplacian)
        self.ads_length = length
        self.time_step = step
        self.potential = potential_value
        self.maximum_steps = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fixed-background-conformal-ads-scalar-plan",
                "radial_points": array_tree_fingerprint(points),
                "laplacian": array_tree_fingerprint(laplacian),
                "ads_length": length,
                "time_step": step,
                "potential": potential_value,
                "maximum_steps": maximum,
                "boundary": "homogeneous-dirichlet-reflecting",
                "integrator": "classical-rk4",
            }
        )

    @property
    def point_count(self) -> int:
        return self.radial_points.shape[0]

    @property
    def spacing(self) -> float:
        return float(self.radial_points[1] - self.radial_points[0])


class ConformalAdSScalarState(StrictModule):
    field: Array
    momentum: Array
    time: Array
    step_index: Array
    valid: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: ConformalAdSScalarPlan,
        field: ArrayLike,
        momentum: ArrayLike,
        /,
        *,
        time: ArrayLike = 0.0,
        step_index: ArrayLike = 0,
        valid: ArrayLike = True,
    ):
        if not isinstance(plan, ConformalAdSScalarPlan):
            raise TypeError("plan must be ConformalAdSScalarPlan.")
        field_value = jnp.asarray(field, dtype=plan.radial_points.dtype)
        momentum_value = jnp.asarray(momentum, dtype=field_value.dtype)
        if (
            field_value.shape != (plan.point_count,)
            or momentum_value.shape != field_value.shape
        ):
            raise ValueError("Conformal scalar state vectors have the wrong shape.")
        self.field = field_value
        self.momentum = momentum_value
        self.time = jnp.asarray(time, dtype=field_value.dtype).reshape(())
        self.step_index = jnp.asarray(step_index, dtype=jnp.int32).reshape(())
        self.valid = jnp.asarray(valid, dtype=jnp.bool_).reshape(())
        self.plan_id = plan.plan_id


class ConformalAdSScalarHistory(StrictModule):
    field: Array
    momentum: Array
    time: Array
    step_index: Array
    valid: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: ConformalAdSScalarPlan,
        field: ArrayLike,
        momentum: ArrayLike,
        time: ArrayLike,
        step_index: ArrayLike,
        valid: ArrayLike,
        /,
    ):
        field_value = jnp.asarray(field, dtype=plan.radial_points.dtype)
        momentum_value = jnp.asarray(momentum, dtype=field_value.dtype)
        time_value = jnp.asarray(time, dtype=field_value.dtype)
        steps = jnp.asarray(step_index, dtype=jnp.int32)
        validity = jnp.asarray(valid, dtype=jnp.bool_)
        if (
            field_value.ndim != 2
            or field_value.shape[1] != plan.point_count
            or momentum_value.shape != field_value.shape
            or time_value.shape != field_value.shape[:1]
            or steps.shape != time_value.shape
            or validity.shape != time_value.shape
        ):
            raise ValueError("Conformal scalar history arrays are incompatible.")
        self.field = field_value
        self.momentum = momentum_value
        self.time = time_value
        self.step_index = steps
        self.valid = validity
        self.plan_id = plan.plan_id


class ConformalAdSScalarEvidence(StrictModule):
    initial_energy: Array
    final_energy: Array
    relative_energy_drift: Array
    maximum_boundary_residual: Array
    finite: Array
    accepted: Array
    step_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


class ConformalAdSScalarRun(StrictModule):
    states: ConformalAdSScalarHistory
    final_state: ConformalAdSScalarState
    evidence: ConformalAdSScalarEvidence


def _enforce_boundary(field: Array, momentum: Array, /) -> tuple[Array, Array]:
    field_value = field.at[0].set(0.0).at[-1].set(0.0)
    momentum_value = momentum.at[0].set(0.0).at[-1].set(0.0)
    return field_value, momentum_value


def _scalar_rhs(
    plan: ConformalAdSScalarPlan, field: Array, momentum: Array, /
) -> tuple[Array, Array]:
    field_rate = momentum
    momentum_rate = plan.laplacian @ field - plan.potential / plan.ads_length**2 * field
    return _enforce_boundary(field_rate, momentum_rate)


def _rk4_step(
    plan: ConformalAdSScalarPlan,
    field: Array,
    momentum: Array,
    /,
) -> tuple[Array, Array]:
    dt = plan.time_step
    k1_field, k1_momentum = _scalar_rhs(plan, field, momentum)
    k2_field, k2_momentum = _scalar_rhs(
        plan,
        field + 0.5 * dt * k1_field,
        momentum + 0.5 * dt * k1_momentum,
    )
    k3_field, k3_momentum = _scalar_rhs(
        plan,
        field + 0.5 * dt * k2_field,
        momentum + 0.5 * dt * k2_momentum,
    )
    k4_field, k4_momentum = _scalar_rhs(
        plan,
        field + dt * k3_field,
        momentum + dt * k3_momentum,
    )
    next_field = field + dt / 6.0 * (
        k1_field + 2.0 * k2_field + 2.0 * k3_field + k4_field
    )
    next_momentum = momentum + dt / 6.0 * (
        k1_momentum + 2.0 * k2_momentum + 2.0 * k3_momentum + k4_momentum
    )
    return _enforce_boundary(next_field, next_momentum)


def conformal_scalar_energy(
    plan: ConformalAdSScalarPlan,
    state: ConformalAdSScalarState,
    /,
) -> Array:
    if state.plan_id != plan.plan_id:
        raise ValueError("Scalar state and plan identities differ.")
    physical_spacing = plan.spacing * plan.ads_length
    gradient = (state.field[1:] - state.field[:-1]) / physical_spacing
    kinetic = 0.5 * physical_spacing * jnp.sum(state.momentum**2)
    gradient_energy = 0.5 * physical_spacing * jnp.sum(gradient**2)
    potential_energy = (
        0.5
        * physical_spacing
        * plan.potential
        / plan.ads_length**2
        * jnp.sum(state.field**2)
    )
    return kinetic + gradient_energy + potential_energy


def conformal_scalar_normal_mode(
    plan: ConformalAdSScalarPlan,
    mode_number: int,
    /,
    *,
    amplitude: float = 1.0,
) -> tuple[ConformalAdSScalarState, float]:
    mode = int(mode_number)
    if mode < 1 or not np.isfinite(amplitude):
        raise ValueError("mode_number and amplitude are invalid.")
    domain_length = 0.5 * pi
    wave_number = mode * pi / domain_length
    frequency = float(np.sqrt(wave_number**2 + plan.potential) / plan.ads_length)
    field = float(amplitude) * jnp.sin(wave_number * plan.radial_points)
    momentum = jnp.zeros_like(field)
    field, momentum = _enforce_boundary(field, momentum)
    return ConformalAdSScalarState(plan, field, momentum), frequency


def run_conformal_ads_scalar(
    plan: ConformalAdSScalarPlan,
    initial: ConformalAdSScalarState,
    /,
    *,
    steps: int,
    energy_tolerance: float = 1e-3,
    boundary_tolerance: float = 1e-12,
) -> ConformalAdSScalarRun:
    if not isinstance(plan, ConformalAdSScalarPlan):
        raise TypeError("plan must be ConformalAdSScalarPlan.")
    if (
        not isinstance(initial, ConformalAdSScalarState)
        or initial.plan_id != plan.plan_id
    ):
        raise TypeError("initial must be a matching ConformalAdSScalarState.")
    count = int(steps)
    if count < 1 or count > plan.maximum_steps:
        raise ValueError("steps are outside the scalar runtime resource bound.")
    fields = []
    momenta = []
    times = []
    valid = []
    current = initial
    initial_energy = conformal_scalar_energy(plan, current)
    maximum_boundary = jnp.maximum(
        jnp.max(jnp.abs(current.field[jnp.asarray((0, -1))])),
        jnp.max(jnp.abs(current.momentum[jnp.asarray((0, -1))])),
    )
    for _ in range(count):
        field, momentum = _rk4_step(plan, current.field, current.momentum)
        finite = jnp.all(jnp.isfinite(field)) & jnp.all(jnp.isfinite(momentum))
        current = ConformalAdSScalarState(
            plan,
            field,
            momentum,
            time=current.time + plan.time_step,
            step_index=current.step_index + 1,
            valid=current.valid & finite,
        )
        fields.append(current.field)
        momenta.append(current.momentum)
        times.append(current.time)
        valid.append(current.valid)
        maximum_boundary = jnp.maximum(
            maximum_boundary,
            jnp.maximum(
                jnp.max(jnp.abs(field[jnp.asarray((0, -1))])),
                jnp.max(jnp.abs(momentum[jnp.asarray((0, -1))])),
            ),
        )
    final_energy = conformal_scalar_energy(plan, current)
    energy_drift = jnp.abs(final_energy - initial_energy) / jnp.maximum(
        1.0, jnp.abs(initial_energy)
    )
    finite = current.valid & jnp.isfinite(energy_drift)
    evidence = ConformalAdSScalarEvidence(
        initial_energy=initial_energy,
        final_energy=final_energy,
        relative_energy_drift=energy_drift,
        maximum_boundary_residual=maximum_boundary,
        finite=finite,
        accepted=finite
        & (energy_drift <= float(energy_tolerance))
        & (maximum_boundary <= float(boundary_tolerance)),
        step_count=count,
        plan_id=plan.plan_id,
        claim="fixed-background-conformal-cylinder-scalar-reference-only",
    )
    history = ConformalAdSScalarHistory(
        plan,
        jnp.stack(fields),
        jnp.stack(momenta),
        jnp.stack(times),
        jnp.arange(1, count + 1, dtype=jnp.int32),
        jnp.stack(valid),
    )
    return ConformalAdSScalarRun(states=history, final_state=current, evidence=evidence)


class ConformalScalarStressEvidence(StrictModule):
    stress_energy: Array
    trace: Array
    maximum_trace_residual: Array
    symmetry_residual: Array
    finite: Array
    tracefree: Array
    claim: str = eqx.field(static=True)


def conformal_scalar_stress_energy(
    metric: ArrayLike,
    inverse_metric: ArrayLike,
    scalar: ArrayLike,
    gradient: ArrayLike,
    hessian: ArrayLike,
    einstein_tensor: ArrayLike,
    /,
    *,
    trace_tolerance: float = 1e-8,
) -> ConformalScalarStressEvidence:
    """Evaluate the four-dimensional conformally coupled scalar stress tensor."""
    g = jnp.asarray(metric)
    inverse = jnp.asarray(inverse_metric, dtype=g.dtype)
    field = jnp.asarray(scalar, dtype=g.dtype)
    derivative = jnp.asarray(gradient, dtype=g.dtype)
    second = jnp.asarray(hessian, dtype=g.dtype)
    einstein = jnp.asarray(einstein_tensor, dtype=g.dtype)
    shape = field.shape
    if (
        g.shape != (4, 4) + shape
        or inverse.shape != g.shape
        or derivative.shape != (4,) + shape
        or second.shape != g.shape
        or einstein.shape != g.shape
    ):
        raise ValueError("Conformal scalar stress inputs have incompatible shapes.")
    gradient_up = ein.contract("ab...,b...->a...", inverse, derivative)
    gradient_squared = ein.contract("a...,a...->...", derivative, gradient_up)
    box_scalar = ein.contract("ab...,ab...->...", inverse, second)
    minimal = (
        ein.contract("a...,b...->ab...", derivative, derivative)
        - 0.5 * g * gradient_squared
    )
    improvement = (
        g * (2.0 * field * box_scalar + 2.0 * gradient_squared)[None, None, ...]
        - (
            2.0 * ein.contract("a...,b...->ab...", derivative, derivative)
            + 2.0 * field[None, None, ...] * second
        )
        + einstein * field[None, None, ...] ** 2
    ) / 6.0
    stress = minimal + improvement
    trace = ein.contract("ab...,ab...->...", inverse, stress)
    scale = jnp.maximum(1.0, jnp.max(jnp.abs(stress)))
    trace_residual = jnp.max(jnp.abs(trace)) / scale
    symmetry = jnp.max(jnp.abs(stress - jnp.swapaxes(stress, 0, 1))) / scale
    finite = jnp.all(jnp.isfinite(stress)) & jnp.all(jnp.isfinite(trace))
    return ConformalScalarStressEvidence(
        stress_energy=stress,
        trace=trace,
        maximum_trace_residual=trace_residual,
        symmetry_residual=symmetry,
        finite=finite,
        tracefree=finite & (trace_residual <= float(trace_tolerance)),
        claim="finite-conformally-coupled-scalar-stress-requires-on-shell-trace-audit",
    )


__all__ = [
    "ConformalAdSScalarEvidence",
    "ConformalAdSScalarHistory",
    "ConformalAdSScalarPlan",
    "ConformalAdSScalarRun",
    "ConformalAdSScalarState",
    "ConformalScalarStressEvidence",
    "conformal_scalar_energy",
    "conformal_scalar_normal_mode",
    "conformal_scalar_stress_energy",
    "run_conformal_ads_scalar",
]
