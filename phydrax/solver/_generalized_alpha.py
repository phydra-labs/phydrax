#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import DiscretizationBundle
from ..dynamics import SecondOrderDifferentialProblem, TimeGrid
from ..linalg import ArraySpace
from ..nonlinear import (
    implicit_root_result,
    NewtonKrylov,
    NonlinearStatus,
    NonlinearSystemProblem,
    NonlinearTermination,
    prepare_nonlinear,
    refresh_nonlinear,
)
from ._temporal_method import TemporalMethodCapabilities


_DEFAULT_ARGS = object()


class GeneralizedAlphaMethod(StrictModule, NonTrainableState):
    """Second-order generalized-alpha method with controlled high-frequency damping."""

    capabilities: TemporalMethodCapabilities
    alpha_m: float = eqx.field(static=True)
    alpha_f: float = eqx.field(static=True)
    beta: float = eqx.field(static=True)
    gamma: float = eqx.field(static=True)
    spectral_radius: float | None = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        spectral_radius: float | None = 1.0,
        /,
        *,
        alpha_m: float | None = None,
        alpha_f: float | None = None,
        beta: float | None = None,
        gamma: float | None = None,
    ):
        explicit = (alpha_m, alpha_f, beta, gamma)
        if any(value is not None for value in explicit):
            if spectral_radius is not None or any(value is None for value in explicit):
                raise ValueError(
                    "Explicit generalized-alpha parameters require spectral_radius=None "
                    "and all four coefficients."
                )
            assert alpha_m is not None
            assert alpha_f is not None
            assert beta is not None
            assert gamma is not None
            am = float(alpha_m)
            af = float(alpha_f)
            beta_ = float(beta)
            gamma_ = float(gamma)
            radius = None
        else:
            if spectral_radius is None:
                raise ValueError(
                    "spectral_radius is required without explicit parameters."
                )
            radius = float(spectral_radius)
            if not isfinite(radius) or not 0.0 <= radius <= 1.0:
                raise ValueError("spectral_radius must lie in [0, 1].")
            am = float((2.0 * radius - 1.0) / (radius + 1.0))
            af = float(radius / (radius + 1.0))
            gamma_ = float(0.5 + af - am)
            beta_ = float(0.25 * (1.0 + af - am) ** 2)
        if any(not isfinite(value) for value in (am, af, beta_, gamma_)):
            raise ValueError("Generalized-alpha coefficients must be finite.")
        if am >= 1.0 or af >= 1.0 or beta_ <= 0.0 or gamma_ <= 0.0:
            raise ValueError("Generalized-alpha coefficients violate stage solvability.")
        radius_id = "explicit" if radius is None else radius.hex()
        self.alpha_m = am
        self.alpha_f = af
        self.beta = beta_
        self.gamma = gamma_
        self.spectral_radius = radius
        self.method_id = f"temporal:generalized-alpha:{radius_id}:{am.hex()}:{af.hex()}"
        self.capabilities = TemporalMethodCapabilities(
            equation_forms=("second-order",),
            method_class="generalized-alpha",
            order=2,
            adaptive=False,
            history_depth=1,
            stage_abscissae=(1.0 - af,),
            causal_stage_extent=1.0,
            a_stable=True,
            verified=True,
            method_id=self.method_id,
        )


class _InitialAccelerationResidual(eqx.Module):
    problem: SecondOrderDifferentialProblem
    time: Array
    args: Any

    def __call__(self, acceleration: Array, unused: Any, /) -> Array:
        del unused
        system = self.problem.system
        return system.scaled_residual(
            self.time,
            self.problem.initial_configuration,
            self.problem.initial_velocity,
            acceleration,
            self.args,
        )


class _GeneralizedAlphaArguments(StrictModule):
    time: Array
    step_size: Array
    configuration: Array
    velocity: Array
    acceleration: Array
    args: Any


class _GeneralizedAlphaResidual(eqx.Module):
    problem: SecondOrderDifferentialProblem
    method: GeneralizedAlphaMethod

    def kinematics(
        self, next_acceleration: Array, arguments: _GeneralizedAlphaArguments, /
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        h = arguments.step_size
        method = self.method
        next_configuration = (
            arguments.configuration
            + h * arguments.velocity
            + h**2
            * (
                (0.5 - method.beta) * arguments.acceleration
                + method.beta * next_acceleration
            )
        )
        next_velocity = arguments.velocity + h * (
            (1.0 - method.gamma) * arguments.acceleration
            + method.gamma * next_acceleration
        )
        weighted_configuration = (
            1.0 - method.alpha_f
        ) * next_configuration + method.alpha_f * arguments.configuration
        weighted_velocity = (
            1.0 - method.alpha_f
        ) * next_velocity + method.alpha_f * arguments.velocity
        weighted_acceleration = (
            1.0 - method.alpha_m
        ) * next_acceleration + method.alpha_m * arguments.acceleration
        weighted_time = (1.0 - method.alpha_f) * arguments.time + method.alpha_f * (
            arguments.time - h
        )
        return (
            next_configuration,
            next_velocity,
            weighted_time,
            weighted_configuration,
            weighted_velocity,
            weighted_acceleration,
        )

    def __call__(
        self,
        next_acceleration: Array,
        arguments: _GeneralizedAlphaArguments,
        /,
    ) -> Array:
        _, _, time, configuration, velocity, acceleration = self.kinematics(
            next_acceleration, arguments
        )
        return self.problem.system.scaled_residual(
            time,
            configuration,
            velocity,
            acceleration,
            arguments.args,
        )


class GeneralizedAlphaSolution(StrictModule):
    """Configuration, velocity, acceleration, and stage evidence."""

    times: Array
    configurations: Array
    velocities: Array
    accelerations: Array
    valid: Array
    stage_residual_norm: Array
    nonlinear_iterations: Array
    discretization_bundle: DiscretizationBundle | None
    method_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    time_id: str = eqx.field(static=True)
    discretization_bundle_id: str | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        times: Array,
        configurations: Array,
        velocities: Array,
        accelerations: Array,
        valid: Array,
        stage_residual_norm: Array,
        nonlinear_iterations: Array,
        discretization_bundle: DiscretizationBundle | None,
        method_id: str,
        problem_id: str,
        time_id: str,
    ):
        count = int(jnp.asarray(times).size)
        prefix = (count,)
        if (
            configurations.shape[0] != count
            or velocities.shape != configurations.shape
            or accelerations.shape != configurations.shape
            or valid.shape != prefix
            or stage_residual_norm.shape != prefix
        ):
            raise ValueError("Generalized-alpha solution arrays do not align.")
        self.times = jnp.asarray(times)
        self.configurations = jnp.asarray(configurations)
        self.velocities = jnp.asarray(velocities)
        self.accelerations = jnp.asarray(accelerations)
        self.valid = jnp.asarray(valid, dtype=bool)
        self.stage_residual_norm = jnp.asarray(stage_residual_norm)
        self.nonlinear_iterations = jnp.asarray(nonlinear_iterations, dtype=jnp.int32)
        self.discretization_bundle = discretization_bundle
        self.method_id = str(method_id)
        self.problem_id = str(problem_id)
        self.time_id = str(time_id)
        self.discretization_bundle_id = (
            None if discretization_bundle is None else discretization_bundle.bundle_id
        )

    @property
    def successful(self) -> Array:
        return jnp.all(self.valid)


def solve_generalized_alpha(
    problem: SecondOrderDifferentialProblem,
    time_grid: TimeGrid,
    /,
    *,
    method: GeneralizedAlphaMethod | None = None,
    nonlinear_method: NewtonKrylov | None = None,
    termination: NonlinearTermination | None = None,
    args: Any = _DEFAULT_ARGS,
) -> GeneralizedAlphaSolution:
    """Solve one fixed-grid second-order residual problem."""
    if not isinstance(problem, SecondOrderDifferentialProblem):
        raise TypeError("problem must be SecondOrderDifferentialProblem.")
    if not isinstance(time_grid, TimeGrid):
        raise TypeError("time_grid must be TimeGrid.")
    selected = GeneralizedAlphaMethod() if method is None else method
    root_method = NewtonKrylov() if nonlinear_method is None else nonlinear_method
    root_termination = (
        NonlinearTermination(
            absolute_residual=1e-9, relative_residual=0.0, maximum_steps=16
        )
        if termination is None
        else termination
    )
    if not isinstance(selected, GeneralizedAlphaMethod):
        raise TypeError("method must be GeneralizedAlphaMethod or None.")
    if not isinstance(root_method, NewtonKrylov):
        raise TypeError("nonlinear_method must be NewtonKrylov or None.")
    if not isinstance(root_termination, NonlinearTermination):
        raise TypeError("termination must be NonlinearTermination or None.")
    runtime_args = problem.args if args is _DEFAULT_ARGS else args
    state = problem.initial_configuration
    space = ArraySpace(state.shape, dtype=state.dtype)
    initial_problem = NonlinearSystemProblem(
        _InitialAccelerationResidual(problem, time_grid.times[0], runtime_args),
        state_space=space,
        residual_space=space,
        problem_id=f"{problem.system.system_id}:initial-acceleration",
    )
    initial_prepared = prepare_nonlinear(
        initial_problem,
        problem.initial_acceleration,
        method=root_method,
        termination=root_termination,
        args=None,
    )
    initial_result = implicit_root_result(initial_prepared)
    initial_acceleration = jnp.asarray(initial_result.state)
    initial_valid = initial_result.status == int(NonlinearStatus.SUCCESS)
    first_arguments = _GeneralizedAlphaArguments(
        time_grid.times[1],
        time_grid.durations[0],
        state,
        problem.initial_velocity,
        initial_acceleration,
        runtime_args,
    )
    stage_residual = _GeneralizedAlphaResidual(problem, selected)
    stage_problem = NonlinearSystemProblem(
        stage_residual,
        state_space=space,
        residual_space=space,
        problem_id=f"{problem.system.system_id}:generalized-alpha-stage",
    )
    stage_prepared = prepare_nonlinear(
        stage_problem,
        initial_acceleration,
        method=root_method,
        termination=root_termination,
        args=first_arguments,
    )

    def advance(carry, values):
        configuration, velocity, acceleration, prior_valid = carry
        target_time, step_size = values
        arguments = _GeneralizedAlphaArguments(
            target_time,
            step_size,
            configuration,
            velocity,
            acceleration,
            runtime_args,
        )

        def solve_step(_):
            refreshed = refresh_nonlinear(
                stage_prepared,
                stage_problem,
                acceleration,
                args=arguments,
            )
            result = implicit_root_result(refreshed)
            next_acceleration = jnp.asarray(result.state)
            next_configuration, next_velocity, *_ = stage_residual.kinematics(
                next_acceleration, arguments
            )
            residual = stage_residual(next_acceleration, arguments)
            residual_norm = jnp.sqrt(jnp.mean(jnp.abs(residual) ** 2))
            finite = (
                jnp.all(jnp.isfinite(next_configuration))
                & jnp.all(jnp.isfinite(next_velocity))
                & jnp.all(jnp.isfinite(next_acceleration))
                & jnp.isfinite(residual_norm)
            )
            valid = (result.status == int(NonlinearStatus.SUCCESS)) & finite
            return (
                next_configuration,
                next_velocity,
                next_acceleration,
                valid,
                residual_norm,
                result.diagnostics.iterations,
            )

        def skip_step(_):
            nan = jnp.full_like(configuration, jnp.nan)
            return (
                nan,
                nan,
                nan,
                jnp.asarray(False),
                jnp.asarray(jnp.inf, dtype=configuration.real.dtype),
                jnp.asarray(0, dtype=jnp.int32),
            )

        output = lax.cond(prior_valid, solve_step, skip_step, operand=None)
        q, v, a, valid, *_ = output
        return (q, v, a, valid), output

    _, outputs = lax.scan(
        advance,
        (
            problem.initial_configuration,
            problem.initial_velocity,
            initial_acceleration,
            initial_valid,
        ),
        (time_grid.times[1:], time_grid.durations),
    )
    step_q, step_v, step_a, step_valid, residuals, iterations = outputs
    initial_residual = problem.system.scaled_residual(
        time_grid.times[0],
        problem.initial_configuration,
        problem.initial_velocity,
        initial_acceleration,
        runtime_args,
    )
    return GeneralizedAlphaSolution(
        times=time_grid.times,
        configurations=jnp.concatenate(
            (problem.initial_configuration[None, ...], step_q), axis=0
        ),
        velocities=jnp.concatenate((problem.initial_velocity[None, ...], step_v), axis=0),
        accelerations=jnp.concatenate((initial_acceleration[None, ...], step_a), axis=0),
        valid=jnp.concatenate((initial_valid[None], step_valid)),
        stage_residual_norm=jnp.concatenate(
            (
                jnp.sqrt(jnp.mean(jnp.abs(initial_residual) ** 2))[None],
                residuals,
            )
        ),
        nonlinear_iterations=jnp.concatenate(
            (initial_result.diagnostics.iterations[None], iterations)
        ),
        discretization_bundle=problem.discretization_bundle,
        method_id=selected.method_id,
        problem_id=problem.problem_id,
        time_id=time_grid.time_id,
    )


__all__ = [
    "GeneralizedAlphaMethod",
    "GeneralizedAlphaSolution",
    "solve_generalized_alpha",
]
