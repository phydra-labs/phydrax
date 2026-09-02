#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-attempt adaptive stochastic TDVP controller for prepared vector fields."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from .._sampling import derive_key, SampleAddress
from .._strict import StrictModule


_STAGE_ADDRESS = SampleAddress(
    "quantum", "adaptive-tdvp", target="stage", role="common-random-number"
)


class AdaptiveTDVPPlan(StrictModule):
    time_start: float = eqx.field(static=True)
    time_stop: float = eqx.field(static=True)
    initial_step_size: float = eqx.field(static=True)
    minimum_step_size: float = eqx.field(static=True)
    maximum_step_size: float = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    maximum_attempts: int = eqx.field(static=True)
    maximum_accepted_steps: int = eqx.field(static=True)
    midpoint_iterations: int = eqx.field(static=True)
    scheme: Literal["adaptive-heun", "symmetric-midpoint"] = eqx.field(static=True)

    def __init__(
        self,
        time_span: tuple[float, float],
        /,
        *,
        initial_step_size: float,
        step_size_bounds: tuple[float, float],
        absolute_tolerance: float,
        relative_tolerance: float = 0.0,
        maximum_attempts: int,
        maximum_accepted_steps: int,
        scheme: Literal["adaptive-heun", "symmetric-midpoint"] = "adaptive-heun",
        midpoint_iterations: int = 4,
    ):
        start, stop = map(float, time_span)
        initial = float(initial_step_size)
        lower, upper = map(float, step_size_bounds)
        absolute, relative = float(absolute_tolerance), float(relative_tolerance)
        attempts, accepted, iterations = (
            int(maximum_attempts),
            int(maximum_accepted_steps),
            int(midpoint_iterations),
        )
        if not start < stop or not 0.0 < lower <= initial <= upper:
            raise ValueError("time span and step-size bounds are invalid.")
        if (
            absolute <= 0.0
            or relative < 0.0
            or attempts <= 0
            or accepted <= 0
            or iterations <= 0
        ):
            raise ValueError("TDVP tolerances/capacities must be positive.")
        if scheme not in ("adaptive-heun", "symmetric-midpoint") or not all(
            np.isfinite(v)
            for v in (start, stop, initial, lower, upper, absolute, relative)
        ):
            raise ValueError("Adaptive TDVP plan values are invalid.")
        self.time_start = start
        self.time_stop = stop
        self.initial_step_size = initial
        self.minimum_step_size = lower
        self.maximum_step_size = upper
        self.absolute_tolerance = absolute
        self.relative_tolerance = relative
        self.maximum_attempts = attempts
        self.maximum_accepted_steps = accepted
        self.midpoint_iterations = iterations
        self.scheme = scheme


class AdaptiveTDVPResult(StrictModule):
    final_parameters: Array
    accepted_times: Array
    parameter_history: Array
    accepted_mask: Array
    attempt_times: Array
    attempt_step_sizes: Array
    temporal_defects: Array
    sampling_uncertainties: Array
    accepted_attempts: Array
    noise_dominated: Array
    midpoint_residuals: Array
    overflow: Array
    valid: Array
    root_key: Array
    scheme: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


def solve_adaptive_tdvp(
    vector_field: Callable[[Array, Array, Key[Array, ""]], tuple[Array, Array]],
    initial_parameters: ArrayLike,
    plan: AdaptiveTDVPPlan,
    /,
    *,
    key: Key[Array, ""],
) -> AdaptiveTDVPResult:
    """Control temporal defect separately from caller-reported sampling uncertainty.

    ``vector_field(parameters, time, key)`` returns ``(velocity,
    velocity_sampling_standard_error)``. The controller propagates that
    uncertainty over the attempted step before comparing it with the
    parameter-space tolerance. The same semantic stage key is used by the
    embedded pair, so rejected attempts never commit parameter or chain-owned
    state outside this pure callback.
    """
    if not callable(vector_field) or not isinstance(plan, AdaptiveTDVPPlan):
        raise TypeError("vector_field/plan types are invalid.")
    parameters = jnp.asarray(initial_parameters)
    time = jnp.asarray(
        plan.time_start, dtype=jnp.result_type(parameters.real.dtype, float)
    )
    step_size = jnp.asarray(plan.initial_step_size, dtype=time.dtype)
    accepted_count = jnp.asarray(0, dtype=jnp.int32)
    accepted_times = (
        jnp.full((plan.maximum_accepted_steps + 1,), jnp.nan, dtype=time.dtype)
        .at[0]
        .set(time)
    )
    history = (
        jnp.full(
            (plan.maximum_accepted_steps + 1,) + parameters.shape,
            jnp.nan,
            dtype=parameters.dtype,
        )
        .at[0]
        .set(parameters)
    )
    accepted_mask = (
        jnp.zeros((plan.maximum_accepted_steps + 1,), dtype=bool).at[0].set(True)
    )
    attempt_times = []
    sizes = []
    defects = []
    uncertainties = []
    decisions = []
    noise_flags = []
    midpoint_residuals = []
    for attempt in range(plan.maximum_attempts):
        active = (time < plan.time_stop) & (accepted_count < plan.maximum_accepted_steps)
        dt = jnp.minimum(step_size, plan.time_stop - time)
        stage_key = derive_key(key, _STAGE_ADDRESS, attempt, 0)
        velocity0, uncertainty0 = vector_field(parameters, time, stage_key)
        uncertainty0 = jnp.asarray(uncertainty0)
        velocity_uncertainty = uncertainty0
        velocity_uncertainty_valid = jnp.all(
            jnp.isfinite(uncertainty0) & (uncertainty0 >= 0.0)
        )
        euler = parameters + dt * velocity0
        if plan.scheme == "adaptive-heun":
            velocity1, uncertainty1 = vector_field(euler, time + dt, stage_key)
            uncertainty1 = jnp.asarray(uncertainty1)
            candidate = parameters + 0.5 * dt * (velocity0 + velocity1)
            temporal_defect = jnp.sqrt(jnp.sum(jnp.abs(candidate - euler) ** 2))
            velocity_uncertainty = jnp.maximum(velocity_uncertainty, uncertainty1)
            velocity_uncertainty_valid = velocity_uncertainty_valid & jnp.all(
                jnp.isfinite(uncertainty1) & (uncertainty1 >= 0.0)
            )
            midpoint_residual = jnp.asarray(jnp.nan)
        else:
            candidate = euler
            for _ in range(plan.midpoint_iterations):
                midpoint = 0.5 * (parameters + candidate)
                midpoint_velocity, midpoint_uncertainty = vector_field(
                    midpoint, time + 0.5 * dt, stage_key
                )
                midpoint_uncertainty = jnp.asarray(midpoint_uncertainty)
                velocity_uncertainty = jnp.maximum(
                    velocity_uncertainty, midpoint_uncertainty
                )
                velocity_uncertainty_valid = velocity_uncertainty_valid & jnp.all(
                    jnp.isfinite(midpoint_uncertainty) & (midpoint_uncertainty >= 0.0)
                )
                candidate = parameters + dt * midpoint_velocity
            midpoint = 0.5 * (parameters + candidate)
            midpoint_velocity, midpoint_uncertainty = vector_field(
                midpoint, time + 0.5 * dt, stage_key
            )
            midpoint_uncertainty = jnp.asarray(midpoint_uncertainty)
            velocity_uncertainty = jnp.maximum(velocity_uncertainty, midpoint_uncertainty)
            velocity_uncertainty_valid = velocity_uncertainty_valid & jnp.all(
                jnp.isfinite(midpoint_uncertainty) & (midpoint_uncertainty >= 0.0)
            )
            midpoint_residual = jnp.sqrt(
                jnp.sum(jnp.abs(candidate - parameters - dt * midpoint_velocity) ** 2)
            )
            temporal_defect = midpoint_residual
        scale = plan.absolute_tolerance + plan.relative_tolerance * jnp.sqrt(
            jnp.sum(jnp.abs(candidate) ** 2)
        )
        sampling_uncertainty = dt * jnp.sqrt(jnp.sum(jnp.abs(velocity_uncertainty) ** 2))
        sampling_valid = velocity_uncertainty_valid & jnp.isfinite(sampling_uncertainty)
        noise_dominated = ~sampling_valid | (sampling_uncertainty >= scale)
        temporal_acceptable = (
            jnp.isfinite(temporal_defect)
            & jnp.all(jnp.isfinite(candidate))
            & (temporal_defect <= scale)
        )
        accept = active & ~noise_dominated & temporal_acceptable
        next_count = accepted_count + accept.astype(jnp.int32)
        parameters = jnp.where(accept, candidate, parameters)
        time = jnp.where(accept, time + dt, time)
        accepted_times = accepted_times.at[next_count].set(
            jnp.where(accept, time, accepted_times[next_count])
        )
        history = history.at[next_count].set(
            jnp.where(accept, parameters, history[next_count])
        )
        accepted_mask = accepted_mask.at[next_count].set(
            jnp.where(accept, True, accepted_mask[next_count])
        )
        safe_defect = jnp.where(
            jnp.isfinite(temporal_defect),
            jnp.maximum(temporal_defect, jnp.finfo(time.dtype).tiny),
            jnp.asarray(jnp.inf, dtype=time.dtype),
        )
        factor = jnp.sqrt(scale / safe_defect)
        temporal_step = dt * jnp.where(
            temporal_acceptable,
            0.9 * factor,
            0.5 * jnp.minimum(factor, 1.0),
        )
        raw_step = jnp.where(noise_dominated & temporal_acceptable, dt, temporal_step)
        step_size = jax.lax.stop_gradient(
            jnp.minimum(
                plan.maximum_step_size, jnp.maximum(plan.minimum_step_size, raw_step)
            )
        )
        accepted_count = next_count
        attempt_times.append(time)
        sizes.append(dt)
        defects.append(temporal_defect)
        uncertainties.append(sampling_uncertainty)
        decisions.append(accept)
        noise_flags.append(noise_dominated)
        midpoint_residuals.append(midpoint_residual)
    overflow = time < plan.time_stop
    return AdaptiveTDVPResult(
        final_parameters=parameters,
        accepted_times=accepted_times,
        parameter_history=history,
        accepted_mask=accepted_mask,
        attempt_times=jnp.stack(attempt_times),
        attempt_step_sizes=jnp.stack(sizes),
        temporal_defects=jnp.stack(defects),
        sampling_uncertainties=jnp.stack(uncertainties),
        accepted_attempts=jnp.stack(decisions),
        noise_dominated=jnp.stack(noise_flags),
        midpoint_residuals=jnp.stack(midpoint_residuals),
        overflow=overflow,
        valid=~overflow & jnp.all(jnp.isfinite(parameters)),
        root_key=jnp.asarray(key),
        scheme=plan.scheme,
        claim=(
            "adaptive-stochastic-temporal-control-separate-from-sampling-error"
            if plan.scheme == "adaptive-heun"
            else "symmetric-midpoint-no-exact-generic-conservation-claim"
        ),
    )


__all__ = ["AdaptiveTDVPPlan", "AdaptiveTDVPResult", "solve_adaptive_tdvp"]
