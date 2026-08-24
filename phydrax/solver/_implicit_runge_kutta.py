#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import sqrt
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, ArrayLike

from .._nonlinear_precision import NonlinearPrecisionPolicy
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..dynamics import TimeGrid
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
from ._differential import DifferentialProblem, DifferentialSolution
from ._temporal_method import (
    configuration_id,
    TemporalMethodCapabilities,
    TemporalSolveEvidence,
)
from ._temporal_precision import TemporalPrecisionPolicy


_DEFAULT_ARGS = object()


def _gauss_tableau(stages: int):
    if stages == 1:
        return ((0.5,),), (1.0,), (0.5,)
    if stages == 2:
        root = sqrt(3.0) / 6.0
        return (
            (
                (0.25, 0.25 - root),
                (0.25 + root, 0.25),
            ),
            (0.5, 0.5),
            (0.5 - root, 0.5 + root),
        )
    root = sqrt(15.0)
    return (
        (
            (5.0 / 36.0, 2.0 / 9.0 - root / 15.0, 5.0 / 36.0 - root / 30.0),
            (5.0 / 36.0 + root / 24.0, 2.0 / 9.0, 5.0 / 36.0 - root / 24.0),
            (5.0 / 36.0 + root / 30.0, 2.0 / 9.0 + root / 15.0, 5.0 / 36.0),
        ),
        (5.0 / 18.0, 4.0 / 9.0, 5.0 / 18.0),
        (
            0.5 - root / 10.0,
            0.5,
            0.5 + root / 10.0,
        ),
    )


def _integrated_lagrange_coefficients(
    nodes: tuple[float, ...],
) -> tuple[tuple[float, ...], ...]:
    from numpy.polynomial import Polynomial

    output = []
    for index, node in enumerate(nodes):
        polynomial = Polynomial((1.0,))
        denominator = 1.0
        for other, other_node in enumerate(nodes):
            if other != index:
                polynomial = polynomial * Polynomial((-other_node, 1.0))
                denominator *= node - other_node
        integrated = (polynomial / denominator).integ()
        coefficients = tuple(float(value) for value in integrated.coef)
        output.append(coefficients + (0.0,) * (len(nodes) + 1 - len(coefficients)))
    return tuple(output)


class GaussLegendreIRK(StrictModule, NonTrainableState):
    """A-stable symplectic Gauss--Legendre collocation with one to three stages."""

    capabilities: TemporalMethodCapabilities
    tableau: tuple[tuple[float, ...], ...] = eqx.field(static=True)
    weights: tuple[float, ...] = eqx.field(static=True)
    nodes: tuple[float, ...] = eqx.field(static=True)
    dense_coefficients: tuple[tuple[float, ...], ...] = eqx.field(static=True)
    stages: int = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(self, stages: int = 2, /):
        count = int(stages)
        if count not in (1, 2, 3):
            raise ValueError("GaussLegendreIRK stages must be one, two, or three.")
        tableau, weights, nodes = _gauss_tableau(count)
        identifier = f"temporal:irk:gauss-legendre:{count}-stage"
        self.tableau = tableau
        self.weights = weights
        self.nodes = nodes
        self.dense_coefficients = _integrated_lagrange_coefficients(nodes)
        self.stages = count
        self.method_id = identifier
        self.capabilities = TemporalMethodCapabilities(
            equation_forms=("explicit-ode",),
            method_class="irk",
            order=2 * count,
            stage_order=count,
            dense_order=2 * count,
            adaptive=False,
            history_depth=1,
            stage_abscissae=nodes,
            causal_stage_extent=1.0,
            a_stable=True,
            symplectic=True,
            reversible=True,
            verified=True,
            method_id=identifier,
        )


class _IRKArguments(StrictModule):
    time: Array
    step_size: Array
    state: Array
    args: Any


class _IRKResidual(eqx.Module):
    problem: DifferentialProblem
    method: GaussLegendreIRK
    precision: TemporalPrecisionPolicy

    def __call__(self, stage_rates: Array, arguments: _IRKArguments, /) -> Array:
        tableau = jnp.asarray(self.method.tableau, dtype=stage_rates.real.dtype)
        nodes = jnp.asarray(self.method.nodes, dtype=arguments.time.dtype)
        accumulated_state = self.precision.accumulation(arguments.state)
        accumulated_rates = self.precision.accumulation(stage_rates)
        accumulated_tableau = self.precision.accumulation(tableau)
        stage_states = (
            accumulated_state
            + self.precision.accumulation(arguments.step_size)
            * jnp.tensordot(accumulated_tableau, accumulated_rates, axes=1)
        ).astype(arguments.state.dtype)
        return jax.vmap(
            lambda node, state, rate: self.precision.residual(
                rate
                - jnp.asarray(
                    self.problem.drift(
                        arguments.time + node * arguments.step_size,
                        state,
                        arguments.args,
                    )
                )
            )
        )(nodes, stage_states, stage_rates)


class GaussLegendreInterpolation(eqx.Module):
    """Collocation-polynomial dense output over accepted fixed steps."""

    times: Array
    states: Array
    stage_rates: Array
    coefficients: Array
    precision: TemporalPrecisionPolicy

    @eqx.filter_jit
    def evaluate(self, query_times: ArrayLike, /, *, left: bool = True) -> Array:
        if not isinstance(left, bool):
            raise TypeError("left must be a bool.")
        query = jnp.asarray(query_times, dtype=self.times.dtype)
        query = eqx.error_if(
            query,
            jnp.any(~jnp.isfinite(query))
            | jnp.any(query < self.times[0])
            | jnp.any(query > self.times[-1]),
            "IRK dense query times must lie inside the solution interval.",
        )
        flat = query.reshape((-1,))
        side = "left" if left else "right"
        indices = jnp.searchsorted(self.times, flat, side=side) - 1
        indices = jnp.clip(indices, 0, self.times.size - 2)
        starts = self.times[indices]
        widths = self.times[indices + 1] - starts
        theta = (flat - starts) / widths
        powers = (
            theta[:, None]
            ** jnp.arange(self.coefficients.shape[1], dtype=theta.dtype)[None, :]
        )
        integrated_weights = powers @ self.coefficients.T
        selected_stages = self.stage_rates[indices]
        updates = jax.vmap(lambda weight, stage: jnp.tensordot(weight, stage, axes=1))(
            integrated_weights, selected_stages
        )
        values = (
            self.states[indices]
            + widths.reshape(widths.shape + (1,) * (self.states.ndim - 1)) * updates
        )
        return self.precision.output(values.reshape(query.shape + self.states.shape[1:]))


def solve_implicit_runge_kutta(
    problem: DifferentialProblem,
    time_grid: TimeGrid,
    /,
    *,
    method: GaussLegendreIRK | None = None,
    nonlinear_method: NewtonKrylov | None = None,
    termination: NonlinearTermination | None = None,
    args: Any = _DEFAULT_ARGS,
    dense: bool = False,
    precision: TemporalPrecisionPolicy | None = None,
) -> DifferentialSolution:
    """Integrate one deterministic ODE by fixed-grid Gauss collocation."""
    if not isinstance(problem, DifferentialProblem) or problem.stochastic:
        raise TypeError("IRK requires a deterministic DifferentialProblem.")
    if not isinstance(time_grid, TimeGrid):
        raise TypeError("time_grid must be TimeGrid.")
    geometry = problem.state_geometry
    if geometry is not None and not geometry.trivial:
        raise ValueError("GaussLegendreIRK currently requires Euclidean geometry.")
    times = lax.stop_gradient(time_grid.times)
    times = eqx.error_if(
        times,
        ~jnp.isclose(times[0], problem.t0) | ~jnp.isclose(times[-1], problem.t1),
        "TimeGrid endpoints must match the differential problem.",
    )
    selected = GaussLegendreIRK() if method is None else method
    root_method = NewtonKrylov() if nonlinear_method is None else nonlinear_method
    root_termination = (
        NonlinearTermination(
            absolute_residual=1e-9, relative_residual=0.0, maximum_steps=16
        )
        if termination is None
        else termination
    )
    if not isinstance(selected, GaussLegendreIRK):
        raise TypeError("method must be GaussLegendreIRK or None.")
    if not isinstance(root_method, NewtonKrylov):
        raise TypeError("nonlinear_method must be NewtonKrylov or None.")
    if not isinstance(root_termination, NonlinearTermination):
        raise TypeError("termination must be NonlinearTermination or None.")
    if not isinstance(dense, bool):
        raise TypeError("dense must be a bool.")
    precision_ = TemporalPrecisionPolicy() if precision is None else precision
    if not isinstance(precision_, TemporalPrecisionPolicy):
        raise TypeError("precision must be a TemporalPrecisionPolicy or None.")
    precision_.validate_implicit_state(problem.initial_state)
    runtime_args = problem.args if args is _DEFAULT_ARGS else args
    stage_shape = (selected.stages,) + problem.initial_state.shape
    space = ArraySpace(stage_shape, dtype=problem.initial_state.dtype)
    residual = _IRKResidual(problem, selected, precision_)
    stage_problem = NonlinearSystemProblem(
        residual,
        state_space=space,
        residual_space=space,
        problem_id=f"{problem.problem_id}:gauss-stage-root",
    )
    initial_rate = precision_.residual(
        problem.drift(times[0], problem.initial_state, runtime_args)
    )
    initial_guess = jnp.broadcast_to(initial_rate, stage_shape)
    first_arguments = _IRKArguments(
        times[0],
        times[1] - times[0],
        problem.initial_state,
        runtime_args,
    )
    nonlinear_precision = NonlinearPrecisionPolicy(
        state_dtype=problem.initial_state.dtype,
        residual_dtype=problem.initial_state.dtype,
        accumulation_dtype=precision_.accumulation_dtype,
        decision_dtype=precision_.decision_dtype,
    )
    prepared = prepare_nonlinear(
        stage_problem,
        initial_guess,
        method=root_method,
        termination=root_termination,
        args=first_arguments,
        precision=nonlinear_precision,
    )
    weights = jnp.asarray(selected.weights, dtype=problem.initial_state.real.dtype)

    def advance(carry, values):
        state, previous_stages, prior_valid = carry
        time, step_size = values
        arguments = _IRKArguments(time, step_size, state, runtime_args)

        def solve_step(_):
            refreshed = refresh_nonlinear(
                prepared,
                stage_problem,
                previous_stages,
                args=arguments,
            )
            result = implicit_root_result(refreshed)
            stages = jnp.asarray(result.state)
            accumulated = precision_.accumulation(state) + precision_.accumulation(
                step_size
            ) * jnp.tensordot(
                precision_.accumulation(weights),
                precision_.accumulation(stages),
                axes=1,
            )
            next_state = accumulated.astype(state.dtype)
            finite = jnp.all(jnp.isfinite(stages)) & jnp.all(jnp.isfinite(next_state))
            valid = (result.status == int(NonlinearStatus.SUCCESS)) & finite
            return next_state, stages, valid, result.diagnostics.iterations

        def skip_step(_):
            return (
                jnp.full_like(state, jnp.nan),
                jnp.full_like(previous_stages, jnp.nan),
                jnp.asarray(False),
                jnp.asarray(0, dtype=jnp.int32),
            )

        output = lax.cond(prior_valid, solve_step, skip_step, operand=None)
        next_state, stages, valid, _ = output
        return (next_state, stages, valid), output

    _, outputs = lax.scan(
        advance,
        (problem.initial_state, initial_guess, jnp.asarray(True)),
        (times[:-1], jnp.diff(times)),
    )
    step_states, stage_rates, step_valid, iterations = outputs
    states = jnp.concatenate((problem.initial_state[None, ...], step_states), axis=0)
    valid = jnp.concatenate((jnp.asarray([True]), step_valid))
    successful = jnp.all(valid)
    interpolation = (
        GaussLegendreInterpolation(
            times,
            states,
            stage_rates,
            jnp.asarray(selected.dense_coefficients, dtype=states.real.dtype),
            precision_,
        )
        if dense
        else None
    )
    evidence = TemporalSolveEvidence(
        selected.capabilities,
        equation_form="explicit-ode",
        backend_id="backend:phydrax:gauss-irk",
        configuration_id=configuration_id(
            (
                selected,
                root_method,
                root_termination,
                precision_.policy_id,
                time_grid.time_id,
                dense,
            ),
            prefix="temporal-configuration",
        ),
        controller_id=f"controller:fixed-grid:{time_grid.time_id}",
        adjoint_id="adjoint:jax-discrete-implicit-root",
        event_id=None,
        adaptive=False,
        dense=dense,
        maximum_steps=time_grid.num_steps,
        precision_evidence=precision_.evidence_for(problem.initial_state, times),
    )
    output_states = jax.vmap(precision_.output)(states)
    return DifferentialSolution(
        times=times,
        states=output_states,
        valid=valid,
        interpolation=interpolation,
        backend_result=jnp.where(successful, 0, 1),
        stats={
            "num_steps": jnp.asarray(time_grid.num_steps, dtype=jnp.int32),
            "nonlinear_iterations": jnp.sum(iterations),
            "stages": selected.stages,
        },
        solver_name="GaussLegendreIRK",
        interpretation=problem.interpretation,
        state_geometry_id=problem.state_geometry_id,
        solver_id=selected.method_id,
        resolved_method=f"gauss-legendre:{selected.stages}-stage",
        discretization_bundle=problem.discretization_bundle,
        backend_successful=successful,
        temporal_evidence=evidence,
        problem_id=problem.problem_id,
    )


__all__ = [
    "GaussLegendreIRK",
    "GaussLegendreInterpolation",
    "solve_implicit_runge_kutta",
]
