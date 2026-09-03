#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import prod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Key

import phydrax.ein as ein
from phydrax.domain import DomainFunction

from .._strict import StrictModule
from ..stochastic._bsde import (
    _predictor_value,
    BSDEEvaluation,
    BSDEPathBatch,
    BSDEProblem,
    BSDEQuadrature,
    evaluate_bsde,
)
from ..stochastic._wiener import WienerRealization


def _shape(value: Sequence[int], /, *, owner: str) -> tuple[int, ...]:
    shape = tuple(int(size) for size in value)
    if any(size <= 0 for size in shape):
        raise ValueError(f"{owner} dimensions must be positive.")
    return shape


class CoupledFBSDEProblem(StrictModule):
    """Explicit-grid fully coupled forward-backward stochastic system."""

    times: Array
    initial_state: Array
    forward_drift: Callable[[Array, Array, Array, Array, Any], Array]
    forward_diffusion: Callable[[Array, Array, Array, Array, Any], Array]
    backward_generator: Callable[[Array, Array, Array, Array, Any], Array]
    terminal: Callable[[Array, Any], Array]
    args: Any
    state_shape: tuple[int, ...] = eqx.field(static=True)
    noise_shape: tuple[int, ...] = eqx.field(static=True)
    output_shape: tuple[int, ...] = eqx.field(static=True)
    num_paths: int = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    wiener_tolerance: float = eqx.field(static=True)
    time_label: str = eqx.field(static=True)
    state_label: str = eqx.field(static=True)

    def __init__(
        self,
        times: ArrayLike,
        initial_state: ArrayLike,
        forward_drift: Callable[[Array, Array, Array, Array, Any], Array],
        forward_diffusion: Callable[[Array, Array, Array, Array, Any], Array],
        backward_generator: Callable[[Array, Array, Array, Array, Any], Array],
        terminal: Callable[[Array, Any], Array],
        /,
        *,
        state_shape: Sequence[int],
        noise_shape: Sequence[int],
        output_shape: Sequence[int],
        num_paths: int,
        problem_id: str,
        process_id: str,
        args: Any = None,
        wiener_tolerance: float = 1e-3,
        time_label: str = "t",
        state_label: str = "x",
    ):
        for owner, value in (
            ("forward_drift", forward_drift),
            ("forward_diffusion", forward_diffusion),
            ("backward_generator", backward_generator),
            ("terminal", terminal),
        ):
            if not callable(value):
                raise TypeError(f"{owner} must be callable.")
        time_values = jnp.asarray(times, dtype=float)
        if time_values.ndim != 1 or time_values.shape[0] < 2:
            raise ValueError("times must contain at least two one-dimensional nodes.")
        if bool(jnp.any(~jnp.isfinite(time_values))) or bool(
            jnp.any(jnp.diff(time_values) <= 0.0)
        ):
            raise ValueError("times must be finite and strictly increasing.")
        state_event = _shape(state_shape, owner="state_shape")
        initial = jnp.asarray(initial_state)
        if initial.shape != state_event:
            raise ValueError("initial_state must have exactly state_shape.")
        count = int(num_paths)
        if count < 1:
            raise ValueError("num_paths must be positive.")
        tolerance = float(wiener_tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("wiener_tolerance must be finite and positive.")
        if tolerance >= float(jnp.min(jnp.diff(time_values))):
            raise ValueError("wiener_tolerance must be smaller than every time step.")
        for owner, value in (
            ("problem_id", problem_id),
            ("process_id", process_id),
            ("time_label", time_label),
            ("state_label", state_label),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{owner} must be a non-empty string.")
        if time_label == state_label:
            raise ValueError("time_label and state_label must be distinct.")
        self.times = time_values
        self.initial_state = initial
        self.forward_drift = forward_drift
        self.forward_diffusion = forward_diffusion
        self.backward_generator = backward_generator
        self.terminal = terminal
        self.args = args
        self.state_shape = state_event
        self.noise_shape = _shape(noise_shape, owner="noise_shape")
        self.output_shape = _shape(output_shape, owner="output_shape")
        self.num_paths = count
        self.problem_id = problem_id
        self.process_id = process_id
        self.wiener_tolerance = tolerance
        self.time_label = time_label
        self.state_label = state_label


class CoupledFBSDEResult(StrictModule):
    """Explicit coupled forward paths and their backward residual evaluation."""

    paths: BSDEPathBatch
    evaluation: BSDEEvaluation
    realization: WienerRealization
    valid: Array
    problem_id: str = eqx.field(static=True)
    scheme: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.valid & self.evaluation.valid_paths


def solve_coupled_fbsde_explicit(
    key: Key[Array, ""],
    problem: CoupledFBSDEProblem,
    value_predictor: Callable | DomainFunction,
    control_predictor: Callable | DomainFunction,
    /,
    *,
    realization: WienerRealization | None = None,
    quadrature: BSDEQuadrature = "left",
    raise_on_failure: bool = False,
) -> CoupledFBSDEResult:
    """Euler--Maruyama forward coupling followed by explicit BSDE residual evaluation."""
    if not isinstance(problem, CoupledFBSDEProblem):
        raise TypeError("problem must be a CoupledFBSDEProblem.")
    if not callable(value_predictor) and not isinstance(value_predictor, DomainFunction):
        raise TypeError("value_predictor must be callable or a DomainFunction.")
    if not callable(control_predictor) and not isinstance(
        control_predictor, DomainFunction
    ):
        raise TypeError("control_predictor must be callable or a DomainFunction.")
    support = (float(problem.times[0]), float(problem.times[-1]))
    if realization is None:
        driver = WienerRealization(
            key,
            problem.noise_shape,
            support=support,
            sample_shape=(problem.num_paths,),
            tolerance=problem.wiener_tolerance,
            noise_id=problem.process_id,
            label=f"{problem.process_id}:coupled-fbsde",
        )
    else:
        if not isinstance(realization, WienerRealization):
            raise TypeError("realization must be a WienerRealization or None.")
        if realization.sample_shape != (problem.num_paths,):
            raise ValueError("Wiener realization sample_shape does not match num_paths.")
        if realization.noise_shape != problem.noise_shape:
            raise ValueError("Wiener realization noise_shape does not match the problem.")
        if realization.noise_id != problem.process_id:
            raise ValueError("Wiener realization noise_id does not match the process.")
        if realization.support != support:
            raise ValueError("Wiener realization support does not match the time grid.")
        driver = realization
    increments = driver.increments(problem.times[:-1], problem.times[1:])
    states = jnp.broadcast_to(
        problem.initial_state,
        (problem.num_paths, 1) + problem.state_shape,
    )
    valid = jnp.ones((problem.num_paths,), dtype=bool)
    predictor_problem = BSDEProblem(
        lambda sample_key: None,
        lambda time, state, args: jnp.zeros(problem.state_shape),
        lambda time, state, args: jnp.zeros(problem.state_shape + problem.noise_shape),
        problem.backward_generator,
        problem.terminal,
        state_shape=problem.state_shape,
        noise_shape=problem.noise_shape,
        output_shape=problem.output_shape,
        problem_id=problem.problem_id,
        process_id=problem.process_id,
        args=problem.args,
        time_label=problem.time_label,
        state_label=problem.state_label,
    )
    valid_nodes = [valid]
    for step in range(problem.times.shape[0] - 1):
        time = problem.times[step]
        current = states[:, -1]
        value_keys = jax.vmap(
            lambda path: jax.random.fold_in(key, step * problem.num_paths + path)
        )(jnp.arange(problem.num_paths, dtype=jnp.uint32))
        values = jax.vmap(
            lambda state, point_key: _predictor_value(
                value_predictor,
                time,
                state,
                predictor_problem,
                key=point_key,
            )
        )(current, value_keys)
        controls = jax.vmap(
            lambda state, point_key: _predictor_value(
                control_predictor,
                time,
                state,
                predictor_problem,
                key=point_key,
            )
        )(current, value_keys)
        expected_values = (problem.num_paths,) + problem.output_shape
        expected_controls = (
            (problem.num_paths,) + problem.output_shape + problem.noise_shape
        )
        if values.shape != expected_values or controls.shape != expected_controls:
            raise ValueError("Coupled FBSDE predictors returned incompatible shapes.")
        drift = jax.vmap(
            lambda state, value, control: jnp.asarray(
                problem.forward_drift(time, state, value, control, problem.args)
            )
        )(current, values, controls)
        diffusion = jax.vmap(
            lambda state, value, control: jnp.asarray(
                problem.forward_diffusion(time, state, value, control, problem.args)
            )
        )(current, values, controls)
        if drift.shape != (problem.num_paths,) + problem.state_shape:
            raise ValueError("forward_drift returned an incompatible state shape.")
        if (
            diffusion.shape
            != (problem.num_paths,) + problem.state_shape + problem.noise_shape
        ):
            raise ValueError("forward_diffusion returned an incompatible shape.")
        state_size = prod(problem.state_shape)
        noise_size = prod(problem.noise_shape)
        diffusion_flat = diffusion.reshape((problem.num_paths, state_size, noise_size))
        increment_flat = increments[:, step].reshape((problem.num_paths, noise_size))
        stochastic_increment = ein.contract(
            "psn,pn->ps", diffusion_flat, increment_flat
        ).reshape((problem.num_paths,) + problem.state_shape)
        next_state = (
            current + drift * (problem.times[step + 1] - time) + stochastic_increment
        )
        step_valid = (
            jnp.all(
                jnp.isfinite(next_state),
                axis=tuple(range(1, next_state.ndim)),
            )
            & jnp.all(jnp.isfinite(values), axis=tuple(range(1, values.ndim)))
            & jnp.all(jnp.isfinite(controls), axis=tuple(range(1, controls.ndim)))
        )
        valid = valid & step_valid
        valid_broadcast = valid.reshape(valid.shape + (1,) * len(problem.state_shape))
        next_state = jnp.where(valid_broadcast, next_state, current)
        states = jnp.concatenate((states, next_state[:, None]), axis=1)
        valid_nodes.append(valid)
    path_valid = jnp.stack(valid_nodes, axis=1)
    paths = BSDEPathBatch(
        problem.times,
        states,
        increments,
        sample_shape=(problem.num_paths,),
        state_shape=problem.state_shape,
        noise_shape=problem.noise_shape,
        path_id=f"{problem.problem_id}:explicit",
        process_id=problem.process_id,
        valid=path_valid,
        realization=driver,
        metadata={"scheme": "explicit-euler-maruyama"},
    )

    def coupled_drift(time, state, args):
        value = _predictor_value(value_predictor, time, state, predictor_problem, key=key)
        control = _predictor_value(
            control_predictor, time, state, predictor_problem, key=key
        )
        return problem.forward_drift(time, state, value, control, args)

    def coupled_diffusion(time, state, args):
        value = _predictor_value(value_predictor, time, state, predictor_problem, key=key)
        control = _predictor_value(
            control_predictor, time, state, predictor_problem, key=key
        )
        return problem.forward_diffusion(time, state, value, control, args)

    bsde_problem = BSDEProblem(
        lambda sample_key: paths,
        coupled_drift,
        coupled_diffusion,
        problem.backward_generator,
        problem.terminal,
        state_shape=problem.state_shape,
        noise_shape=problem.noise_shape,
        output_shape=problem.output_shape,
        problem_id=problem.problem_id,
        process_id=problem.process_id,
        args=problem.args,
        time_label=problem.time_label,
        state_label=problem.state_label,
    )
    evaluation = evaluate_bsde(
        bsde_problem,
        paths,
        value_predictor,
        control_predictor=control_predictor,
        control_mode="explicit",
        quadrature=quadrature,
        key=key,
    )
    result = CoupledFBSDEResult(
        paths=paths,
        evaluation=evaluation,
        realization=driver,
        valid=valid,
        problem_id=problem.problem_id,
        scheme="explicit-euler-maruyama",
    )
    if raise_on_failure and not bool(jnp.all(result.successful)):
        raise RuntimeError("Coupled FBSDE simulation failed for at least one path.")
    return result


__all__ = [
    "CoupledFBSDEProblem",
    "CoupledFBSDEResult",
    "solve_coupled_fbsde_explicit",
]
