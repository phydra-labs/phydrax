# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Native replay, finite-horizon HVAC control, identification, and predictive UQ."""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...control import (
    LinearControlQPSolution,
    LinearQuadraticControlProblem,
    solve_linear_quadratic_control,
)
from ...dynamics import TimeGrid
from ...ein import contract
from ...linalg import DenseLinearOperator, DenseQR, LinearSolvePolicy
from ...linalg.svd import svd, SVDProblem, SVDSolvePolicy
from ...optim import (
    Bounds,
    ConvexSolvePolicy,
    ConvexTermination,
    least_squares,
    LeastSquaresResult,
    LevenbergMarquardt,
    MinimizationResult,
    minimize,
    OptimizationTermination,
    ProjectedLBFGS,
)
from ...uq import FunctionalConformal
from ._model import BuildingCompilation, BuildingSource, compile_building


class BuildingReplay(StrictModule):
    temperature: Array
    heat: Array
    time: Array
    successful: Array
    step_residual: Array


def _boundary_horizon(
    compilation: BuildingCompilation, temperatures: ArrayLike, count: int
) -> Array:
    values = jnp.asarray(temperatures)
    if values.shape == (count,) and len(compilation.boundary_ids) == 1:
        values = values[:, None]
    if values.shape != (count, len(compilation.boundary_ids)):
        raise ValueError(
            "Environmental horizon must have time/boundary axes in compiled boundary_ids order."
        )
    return values


def replay_building(
    compilation: BuildingCompilation,
    initial_temperature: ArrayLike,
    time: ArrayLike,
    boundary_temperature: ArrayLike,
    heat: ArrayLike,
) -> BuildingReplay:
    """Replay interval-held ordered boundary K and nodal W on increasing seconds."""
    times, gains = jnp.asarray(time), jnp.asarray(heat)
    count, n = times.size - 1, len(compilation.node_ids)
    environments = _boundary_horizon(compilation, boundary_temperature, count)
    if times.ndim != 1 or count < 1 or gains.shape != (count, n):
        raise ValueError(
            "Replay requires N+1 times, N boundary vectors, and N by node heat values."
        )
    times = eqx.error_if(
        times,
        jnp.any(~jnp.isfinite(times)) | jnp.any(jnp.diff(times) <= 0),
        "Replay time must be finite and strictly increasing.",
    )
    state = compilation.consistent_temperature(
        initial_temperature, environments[0], gains[0]
    )

    def advance(value, forcing):
        tout, q, dt = forcing
        result = compilation.step(value, tout, q, dt)
        return result.temperature, (
            result.temperature,
            result.successful,
            result.residual_estimate,
        )

    _, (states, success, residual) = jax.lax.scan(
        advance, state, (environments, gains, jnp.diff(times))
    )
    return BuildingReplay(
        jnp.concatenate((state[None], states)), gains, times, jnp.all(success), residual
    )


class HVACControlResult(StrictModule):
    electrical_power: Array
    delivered_heat: Array
    replay: BuildingReplay
    optimization: MinimizationResult | LinearControlQPSolution
    objective: Array
    successful: Array
    state_reference_temperature: Array
    mode: str = eqx.field(static=True)


def _linear_hvac_problem(
    compilation,
    reference,
    time,
    environments,
    base,
    target,
    distribution,
    factors,
    lower,
    upper,
    power_scale,
    temperature_weight,
    electricity_weight,
    price,
):
    """Lower exact frozen physical transitions and costs; the native control compiler owns the QP."""
    count, n = time.size - 1, len(compilation.node_ids)
    devices = distribution.shape[1]
    zero_state, zero_boundary = jnp.zeros(n), jnp.zeros(len(compilation.boundary_ids))
    dt = jnp.diff(time)

    def checked_step(state, boundary, heat, duration):
        result = compilation.step(state, boundary, heat, duration)
        return eqx.error_if(
            result.temperature,
            ~result.successful,
            "Affine HVAC transition preparation failed.",
        )

    def matrices(duration, boundary, gains, factor):
        a = jax.vmap(
            lambda column: checked_step(column, zero_boundary, zero_state, duration)
        )(jnp.eye(n)).T
        heat_columns = distribution * factor[None, :] * power_scale
        b = jax.vmap(
            lambda column: checked_step(zero_state, zero_boundary, column, duration),
            in_axes=1,
            out_axes=1,
        )(heat_columns)
        bias = checked_step(reference, boundary, gains, duration) - reference
        return a, b, bias

    a, b, bias = jax.vmap(matrices)(dt, environments, base, factors)
    zone_count = len(compilation.source.zones)
    observation = jnp.eye(n)[:zone_count]
    metric = contract("zi,zj->ij", observation, observation)
    shifted_targets = target - reference[:zone_count]
    endpoint_q = 2 * temperature_weight * dt[:, None, None] * metric[None]
    endpoint_linear = (
        -2
        * temperature_weight
        * dt[:, None]
        * contract("zi,tz->ti", observation, shifted_targets)
    )
    endpoint_constants = temperature_weight * dt * jnp.sum(shifted_targets**2, axis=1)
    # Native stages cost x[t]; move endpoint costs to x[1:H] and the true terminal node.
    q = jnp.concatenate((jnp.zeros((1, n, n)), endpoint_q[:-1]))
    linear = jnp.concatenate((jnp.zeros((1, n)), endpoint_linear[:-1]))
    constants = jnp.concatenate((jnp.zeros(1), endpoint_constants[:-1]))
    return LinearQuadraticControlProblem(
        a,
        b,
        jnp.zeros(n),
        q,
        jnp.zeros((count, devices, devices)),
        endpoint_q[-1],
        dynamics_bias=bias,
        state_linear=linear,
        stage_constants=constants,
        terminal_linear=endpoint_linear[-1],
        terminal_constant=endpoint_constants[-1],
        control_linear=electricity_weight * dt[:, None] * price * power_scale,
        control_lower_bounds=jnp.broadcast_to(
            jnp.asarray(lower) / power_scale, (count, devices)
        ),
        control_upper_bounds=jnp.broadcast_to(
            jnp.asarray(upper) / power_scale, (count, devices)
        ),
        time_grid=TimeGrid(time, time_id=compilation.source.source_id + ":hvac-time"),
        problem_id=compilation.source.source_id + ":hvac",
        dynamics_id=compilation.source.source_id + ":exact-rc",
    )


def optimize_hvac(
    compilation: BuildingCompilation,
    initial_temperature: ArrayLike,
    time: ArrayLike,
    boundary_temperature: ArrayLike,
    base_heat: ArrayLike,
    target_temperature: ArrayLike,
    *,
    heat_distribution: ArrayLike,
    conversion_law,
    supply_temperature: ArrayLike,
    power_lower: ArrayLike = 0.0,
    power_upper: ArrayLike,
    initial_power: ArrayLike,
    power_scale: float = 1000.0,
    temperature_weight: float = 1.0,
    electricity_weight: float = 0.0,
    electricity_price: ArrayLike = 1.0,
    source_boundary_id: str | None = None,
    termination: OptimizationTermination | None = None,
    control_policy: ConvexSolvePolicy | None = None,
) -> HVACControlResult:
    """Optimize constant-COP/resistance devices through native finite-horizon control.

    Controls are electrical W; the thermofluids HeatConversionLaw supplies heat W.
    Distribution maps device delivered heat into nodes, with nonnegative columns
    summing to one. Temperature cost is integrated squared K error; price is per J.
    Bounds are hard electrical limits; comfort is a weighted objective, not a
    falsely certified hard constraint. Receding-horizon users apply the first row
    and call again after measurement.
    """
    from ..thermofluids import (
        ConstantCOPHeatPumpLaw,
        HeatConversionLaw,
        ResistiveHeatingLaw,
    )

    if not isinstance(conversion_law, HeatConversionLaw):
        raise TypeError(
            "conversion_law must be the native thermofluids HeatConversionLaw."
        )
    if power_scale <= 0 or temperature_weight < 0 or electricity_weight < 0:
        raise ValueError("Control scales/weights must be positive/nonnegative.")
    time_, base = jnp.asarray(time), jnp.asarray(base_heat)
    initial, distribution = jnp.asarray(initial_power), jnp.asarray(heat_distribution)
    count, n = time_.size - 1, len(compilation.node_ids)
    environments = _boundary_horizon(compilation, boundary_temperature, count)
    if source_boundary_id is None:
        ambient = tuple(
            boundary.boundary_id
            for boundary in compilation.source.boundaries
            if boundary.kind == "ambient"
        )
        if len(ambient) != 1:
            raise ValueError(
                "HVAC conversion needs explicit source_boundary_id unless exactly one ambient boundary exists."
            )
        source_boundary_id = ambient[0]
    if source_boundary_id not in compilation.boundary_ids:
        raise ValueError(
            "HVAC conversion references an unknown source-temperature boundary."
        )
    source_temperature = environments[
        :, compilation.boundary_ids.index(source_boundary_id)
    ]
    if (
        initial.ndim != 2
        or initial.shape[0] != count
        or distribution.shape != (n, initial.shape[1])
    ):
        raise ValueError(
            "Power needs time/device axes and distribution needs node/device axes."
        )
    distribution = eqx.error_if(
        distribution,
        jnp.any(distribution < 0)
        | jnp.any(~jnp.isfinite(distribution))
        | jnp.any(jnp.abs(jnp.sum(distribution, axis=0) - 1) > 1e-8),
        "Heat distribution must conserve delivered heat.",
    )
    target = jnp.broadcast_to(
        jnp.asarray(target_temperature), (count, len(compilation.source.zones))
    )
    dt = jnp.diff(time_)
    price = jnp.broadcast_to(jnp.asarray(electricity_price), initial.shape)

    def evaluate(power):
        conversion = conversion_law.evaluate(
            power, source_temperature[:, None], jnp.asarray(supply_temperature)
        )
        delivered = conversion.delivered_heat
        gains = base + contract("nd,td->tn", distribution, delivered)
        rollout = replay_building(
            compilation, initial_temperature, time_, environments, gains
        )
        error = rollout.temperature[1:, : len(compilation.source.zones)] - target
        objective = temperature_weight * jnp.sum(
            dt[:, None] * error**2
        ) + electricity_weight * jnp.sum(dt[:, None] * price * power)
        objective = eqx.error_if(
            objective,
            ~jnp.all(conversion.successful) | ~rollout.successful,
            "HVAC conversion or thermal rollout failed.",
        )
        return objective, rollout, delivered

    def objective(normalized, args):
        del args
        return evaluate(normalized * power_scale)[0]

    reference = jnp.asarray(initial_temperature)
    if isinstance(conversion_law, (ConstantCOPHeatPumpLaw, ResistiveHeatingLaw)):
        unit_conversion = conversion_law.evaluate(
            jnp.ones_like(initial),
            source_temperature[:, None],
            jnp.asarray(supply_temperature),
        )
        factors = eqx.error_if(
            unit_conversion.delivered_heat,
            ~jnp.all(unit_conversion.successful),
            "HVAC affine device law is outside its supported temperature domain.",
        )
        problem = _linear_hvac_problem(
            compilation,
            reference,
            time_,
            environments,
            base,
            target,
            distribution,
            factors,
            power_lower,
            power_upper,
            power_scale,
            temperature_weight,
            electricity_weight,
            price,
        )
        if control_policy is not None and termination is not None:
            raise ValueError(
                "Specify native control_policy or shared termination, not both."
            )
        policy = control_policy
        if policy is None and termination is not None:
            policy = ConvexSolvePolicy(
                termination=ConvexTermination(
                    absolute=termination.absolute_optimality,
                    relative=termination.relative_optimality,
                    maximum_steps=termination.maximum_steps,
                )
            )
        result = solve_linear_quadratic_control(problem, policy=policy)
        power = result.controls * power_scale
        mode = "linear-quadratic"
    else:
        if control_policy is not None:
            raise ValueError(
                "control_policy applies only to affine device laws; nonlinear shooting uses termination."
            )
        result = minimize(
            objective,
            initial / power_scale,
            method=ProjectedLBFGS(),
            bounds=Bounds(
                jnp.asarray(power_lower) / power_scale,
                jnp.asarray(power_upper) / power_scale,
            ),
            termination=termination,
        )
        power = result.parameters * power_scale
        mode = "nonlinear-shooting"
    cost, rollout, delivered = evaluate(power)
    success = result.successful & rollout.successful
    if isinstance(result, LinearControlQPSolution):
        mismatch = jnp.max(
            jnp.abs(result.states[1:] + reference - rollout.temperature[1:])
        )
        success = success & (mismatch <= 1e-5)
    return HVACControlResult(
        power, delivered, rollout, result, cost, success, reference, mode
    )


class BuildingExperiment(StrictModule):
    initial_temperature: Array
    time: Array
    boundary_temperature: Array
    heat: Array
    observed_temperature: Array
    experiment_id: str = eqx.field(static=True)

    def __init__(
        self,
        initial_temperature: ArrayLike,
        time: ArrayLike,
        boundary_temperature: ArrayLike,
        heat: ArrayLike,
        observed_temperature: ArrayLike,
        *,
        experiment_id: str,
    ):
        (
            self.initial_temperature,
            self.time,
            self.boundary_temperature,
            self.heat,
            self.observed_temperature,
        ) = (
            jnp.asarray(initial_temperature),
            jnp.asarray(time),
            jnp.asarray(boundary_temperature),
            jnp.asarray(heat),
            jnp.asarray(observed_temperature),
        )
        if not experiment_id or self.observed_temperature.shape[0] != self.time.size - 1:
            raise ValueError(
                "Experiment requires identity and interval-end observations."
            )
        self.observed_temperature = eqx.error_if(
            self.observed_temperature,
            ~jnp.all(jnp.isfinite(self.observed_temperature)),
            "Calibration observations must be finite; select valid observations explicitly.",
        )
        self.experiment_id = experiment_id


class BuildingCalibration(StrictModule):
    parameters: Array
    optimization: LeastSquaresResult
    singular_values: Array
    identifiable: Array
    heldout_prediction: Array
    heldout_rmse: Array
    successful: Array


def calibrate_building(
    make_source: Callable[[Array], BuildingSource],
    initial_parameters: ArrayLike,
    training: BuildingExperiment,
    heldout: BuildingExperiment,
    *,
    observation_nodes: tuple[int, ...],
    observation_scale: ArrayLike = 1.0,
    identifiability_relative_tolerance: float = 1e-6,
    termination: OptimizationTermination | None = None,
) -> BuildingCalibration:
    """Fit explicit physical parameters and report local sensitivity rank and held-out error.

    ``make_source`` must implement admissible parameterization (e.g. exponentials
    for positive capacities/resistances). Distinct experiment identities prevent
    accidental train/test self-comparison; callers own truly held-out data.
    A fitted but rank-deficient model is not reported as successful identification.
    """
    if training.experiment_id == heldout.experiment_id:
        raise ValueError("Training and held-out experiments require distinct identities.")
    parameters = jnp.asarray(initial_parameters)
    if parameters.ndim != 1 or not parameters.size or not observation_nodes:
        raise ValueError(
            "Calibration needs a nonempty parameter vector and observation nodes."
        )
    scale = jnp.asarray(observation_scale)
    scale = eqx.error_if(
        scale,
        jnp.any(~jnp.isfinite(scale) | (scale <= 0)),
        "Observation scale must be positive.",
    )
    if identifiability_relative_tolerance <= 0:
        raise ValueError("Identifiability tolerance must be positive.")

    def predict(p, experiment):
        model = compile_building(make_source(p))
        if any(i < 0 or i >= len(model.node_ids) for i in observation_nodes):
            raise ValueError("Unknown observation node.")
        trajectory = replay_building(
            model,
            experiment.initial_temperature,
            experiment.time,
            experiment.boundary_temperature,
            experiment.heat,
        )
        values = trajectory.temperature[1:, jnp.asarray(observation_nodes)]
        if values.shape != experiment.observed_temperature.shape:
            raise ValueError("Observed temperatures must match selected node/time axes.")
        return eqx.error_if(
            values, ~trajectory.successful, "Calibration thermal trajectory failed."
        )

    def residual(p, args):
        del args
        return ((predict(p, training) - training.observed_temperature) / scale).reshape(
            -1
        )

    # Identification already materializes the finite parameter Jacobian for rank
    # evidence; QR avoids nesting a matrix-free Krylov solve inside thermal AD.
    result = least_squares(
        residual,
        parameters,
        method=LevenbergMarquardt(linear_policy=LinearSolvePolicy(DenseQR())),
        termination=termination,
    )
    jacobian = jax.jacfwd(lambda p: residual(p, None))(result.parameters)
    spectrum = svd(
        SVDProblem(DenseLinearOperator(jacobian)),
        policy=SVDSolvePolicy(count=min(jacobian.shape)),
    )
    values = spectrum.singular_values
    identifiable = (
        (jacobian.shape[0] >= parameters.size)
        & spectrum.successful
        & (values[-1] > identifiability_relative_tolerance * values[0])
    )
    prediction = predict(result.parameters, heldout)
    rmse = jnp.sqrt(jnp.mean((prediction - heldout.observed_temperature) ** 2))
    return BuildingCalibration(
        result.parameters,
        result,
        values,
        identifiable,
        prediction,
        rmse,
        result.successful & identifiable & jnp.isfinite(rmse),
    )


def calibrate_building_prediction_band(
    predicted_episodes: ArrayLike, observed_episodes: ArrayLike, *, alpha: float = 0.1
) -> FunctionalConformal:
    """Native simultaneous bands across exchangeable held-out *episodes*.

    The leading axis is independent buildings/days/experiments, not adjacent
    autocorrelated samples. Exchangeability is a caller assumption, not inferred.
    """
    prediction, observed = jnp.asarray(predicted_episodes), jnp.asarray(observed_episodes)
    if prediction.ndim != 3 or observed.shape != prediction.shape:
        raise ValueError("Prediction-band calibration needs episode/time/node arrays.")
    return FunctionalConformal.calibrate(
        prediction, observed, alpha=alpha, case_dim=0, score="max"
    )
