#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from math import prod
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jaxtyping import Array, ArrayLike, Key

from phydrax.domain import Domain, DomainFunction

from .._strict import StrictModule
from ..objectives._deep_splitting import (
    deep_splitting_labels,
    DeepSplittingLabelBatch,
    DeepSplittingRegressionDiagnostics,
    DeepSplittingRegressionObjective,
)
from ..stochastic._bsde import _predictor_value, BSDEPathBatch, BSDEProblem
from ._functional_solver import FunctionalSolver


DeepSplittingInterpolation: TypeAlias = Literal["linear", "nearest"]
DeepSplittingSamplingMode: TypeAlias = Literal["resample", "fixed"]


class _TerminalPredictor(StrictModule):
    problem: BSDEProblem

    def __call__(self, time: Array, state: Array, /) -> Array:
        del time
        value = jnp.asarray(self.problem.terminal(state, self.problem.args))
        if value.shape != self.problem.output_shape:
            raise ValueError("BSDE terminal condition returned an incompatible output shape.")
        return value


class DeepSplittingSolution(StrictModule):
    """Discrete slice family with JAX-compatible temporal interpolation."""

    problem: BSDEProblem
    times: Array
    slices: tuple[DomainFunction, ...]
    interpolation: DeepSplittingInterpolation = eqx.field(static=True)

    def __init__(
        self,
        problem: BSDEProblem,
        times: ArrayLike,
        slices: tuple[DomainFunction, ...],
        /,
        *,
        interpolation: DeepSplittingInterpolation = "linear",
    ):
        if not isinstance(problem, BSDEProblem):
            raise TypeError("problem must be a BSDEProblem.")
        time_values = jnp.asarray(times, dtype=float)
        if time_values.ndim != 1 or time_values.shape[0] < 2:
            raise ValueError("Deep splitting times must contain at least two nodes.")
        if bool(jnp.any(~jnp.isfinite(time_values))) or bool(
            jnp.any(jnp.diff(time_values) <= 0.0)
        ):
            raise ValueError("Deep splitting times must be finite and increasing.")
        slice_values = tuple(slices)
        if len(slice_values) != int(time_values.shape[0]) - 1:
            raise ValueError("Deep splitting requires one learned slice per interval.")
        if any(not isinstance(value, DomainFunction) for value in slice_values):
            raise TypeError("Deep splitting slices must be DomainFunction objects.")
        if interpolation not in ("linear", "nearest"):
            raise ValueError("interpolation must be 'linear' or 'nearest'.")
        self.problem = problem
        self.times = time_values
        self.slices = slice_values
        self.interpolation = interpolation

    @property
    def num_steps(self) -> int:
        return len(self.slices)

    def at_node(
        self,
        index: int,
        state: ArrayLike,
        /,
        *,
        key: Key[Array, ""] = jr.key(0),
    ) -> Array:
        """Evaluate one learned node, or the exact terminal condition at the last node."""
        node = int(index)
        if node < 0 or node > self.num_steps:
            raise ValueError("Deep splitting node index is out of range.")
        state_value = jnp.asarray(state)
        if state_value.shape != self.problem.state_shape:
            raise ValueError("state must have exactly problem.state_shape.")
        if node == self.num_steps:
            return _TerminalPredictor(self.problem)(self.times[-1], state_value)
        value = _predictor_value(
            self.slices[node],
            self.times[node],
            state_value,
            self.problem,
            key=key,
        )
        if value.shape != self.problem.output_shape:
            raise ValueError("Deep splitting slice returned an incompatible output shape.")
        return value

    def _node_value(self, index: Array, state: Array, key: Array, /) -> Array:
        branches: list[Callable[[tuple[Array, Array]], Array]] = []

        def learned_branch(predictor: DomainFunction, node_time: Array):
            def branch(operand):
                state_value, branch_key = operand
                value = _predictor_value(
                    predictor,
                    node_time,
                    state_value,
                    self.problem,
                    key=branch_key,
                )
                if value.shape != self.problem.output_shape:
                    raise ValueError(
                        "Deep splitting slice returned an incompatible output shape."
                    )
                return value

            return branch

        for node, predictor in enumerate(self.slices):
            branches.append(learned_branch(predictor, self.times[node]))

        def terminal_branch(operand):
            state_value, branch_key = operand
            del branch_key
            return _TerminalPredictor(self.problem)(self.times[-1], state_value)

        branches.append(terminal_branch)
        return jax.lax.switch(index, tuple(branches), (state, key))

    def __call__(
        self,
        time: ArrayLike,
        state: ArrayLike,
        /,
        *,
        key: Key[Array, ""] = jr.key(0),
    ) -> Array:
        """Evaluate the nearest or linearly interpolated learned time-slice field."""
        time_value = jnp.asarray(time, dtype=self.times.dtype)
        state_value = jnp.asarray(state)
        if time_value.shape != ():
            raise ValueError("time must be scalar.")
        if state_value.shape != self.problem.state_shape:
            raise ValueError("state must have exactly problem.state_shape.")
        time_value = eqx.error_if(
            time_value,
            (time_value < self.times[0]) | (time_value > self.times[-1]),
            "Deep splitting query time lies outside the trained grid.",
        )
        upper = jnp.clip(
            jnp.searchsorted(self.times, time_value, side="right"),
            1,
            self.num_steps,
        )
        lower = upper - 1
        weight = (time_value - self.times[lower]) / (
            self.times[upper] - self.times[lower]
        )
        lower_key, upper_key = jr.split(key)
        if self.interpolation == "nearest":
            nearest = jnp.where(weight <= 0.5, lower, upper)
            return self._node_value(nearest, state_value, lower_key)
        lower_value = self._node_value(lower, state_value, lower_key)
        upper_value = self._node_value(upper, state_value, upper_key)
        return (1.0 - weight) * lower_value + weight * upper_value

    def control(
        self,
        time: ArrayLike,
        state: ArrayLike,
        /,
        *,
        key: Key[Array, ""] = jr.key(0),
    ) -> Array:
        """Differentiate the interpolated value and contract it with the diffusion."""
        time_value = jnp.asarray(time)
        state_value = jnp.asarray(state)
        gradient = jax.jacrev(
            lambda argument: self(time_value, argument, key=key)
        )(state_value)
        diffusion = jnp.asarray(
            self.problem.diffusion(time_value, state_value, self.problem.args)
        )
        expected = self.problem.state_shape + self.problem.noise_shape
        if diffusion.shape != expected:
            raise ValueError(f"diffusion must have shape {expected}.")
        return (
            gradient.reshape((prod(self.problem.output_shape), prod(self.problem.state_shape)))
            @ diffusion.reshape(
                (prod(self.problem.state_shape), prod(self.problem.noise_shape))
            )
        ).reshape(self.problem.output_shape + self.problem.noise_shape)

    def as_domain_function(self, domain: Domain, /) -> DomainFunction:
        """Expose the interpolated slice family through the standard field API."""
        required = (self.problem.time_label, self.problem.state_label)
        if any(label not in domain.labels for label in required):
            raise ValueError("domain must contain the BSDE time and state labels.")
        return DomainFunction(domain=domain, deps=required, func=self)


class DeepSplittingDiagnostics(StrictModule):
    """Per-slice held-out one-step diagnostics in ascending time order."""

    slice_indices: Array
    times: Array
    one_step_rmse: Array
    target_rms: Array
    source_rms: Array
    valid_fraction: Array
    finite: Array

    @property
    def passed(self) -> bool:
        return bool(jnp.all(self.finite)) and bool(self.slice_indices.shape[0] > 0)


class DeepSplittingResult(StrictModule):
    """Backward-trained slice family and the solver holding the initial slice."""

    solver: FunctionalSolver
    solution: DeepSplittingSolution
    diagnostics: DeepSplittingDiagnostics
    completed_slices: int = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    value_name: str = eqx.field(static=True)


def _require_time_grid(paths: BSDEPathBatch, times: Array, /) -> None:
    if paths.times.shape != times.shape or not bool(jnp.allclose(paths.times, times)):
        raise ValueError("Deep splitting samples must use the declared time grid.")


def _label_provider(
    problem: BSDEProblem,
    next_predictor: Callable | DomainFunction,
    slice_index: int,
    times: Array,
    /,
):
    def provider(key):
        paths = problem.sample(key)
        _require_time_grid(paths, times)
        return deep_splitting_labels(
            problem,
            paths,
            next_predictor,
            slice_index,
            key=jr.fold_in(key, 1),
        )

    return provider


def solve_deep_splitting(
    solver: FunctionalSolver,
    problem: BSDEProblem,
    /,
    *,
    value_name: str,
    inner_num_iter: int,
    time_grid: ArrayLike | None = None,
    optim: Any = None,
    sampling_mode: DeepSplittingSamplingMode = "resample",
    fixed_paths: BSDEPathBatch | None = None,
    validation_paths: BSDEPathBatch | None = None,
    value_weight: ArrayLike = 1.0,
    warm_start: bool = True,
    interpolation: DeepSplittingInterpolation = "linear",
    seed: int = 0,
    jit: bool = True,
    keep_best: bool = True,
    log_every: int = 0,
) -> DeepSplittingResult:
    """Train local conditional-expectation regressions backward over a time grid."""
    if not isinstance(solver, FunctionalSolver) or not isinstance(problem, BSDEProblem):
        raise TypeError("solver and problem must be FunctionalSolver and BSDEProblem.")
    inner_steps = int(inner_num_iter)
    if inner_steps < 1:
        raise ValueError("inner_num_iter must be positive.")
    if not isinstance(value_name, str) or not value_name:
        raise ValueError("value_name must be a non-empty string.")
    if value_name not in solver.ansatz_functions():
        raise KeyError(f"Missing deep splitting value function {value_name!r}.")
    if sampling_mode not in ("resample", "fixed"):
        raise ValueError("sampling_mode must be 'resample' or 'fixed'.")
    if sampling_mode == "resample" and fixed_paths is not None:
        raise ValueError("fixed_paths is valid only for fixed sampling.")
    if interpolation not in ("linear", "nearest"):
        raise ValueError("interpolation must be 'linear' or 'nearest'.")
    if optim is None:
        optim = optax.adam(1e-3)
    root_key = jr.key(int(seed))
    held_out_paths = (
        problem.sample(jr.fold_in(root_key, 200))
        if validation_paths is None
        else validation_paths
    )
    if held_out_paths.process_id != problem.process_id:
        raise ValueError("Validation paths do not match the BSDE process.")
    times = (
        jnp.asarray(held_out_paths.times)
        if time_grid is None
        else jnp.asarray(time_grid, dtype=float)
    )
    if times.ndim != 1 or times.shape[0] < 2:
        raise ValueError("time_grid must be one-dimensional with at least two nodes.")
    if bool(jnp.any(~jnp.isfinite(times))) or bool(jnp.any(jnp.diff(times) <= 0.0)):
        raise ValueError("time_grid must be finite and strictly increasing.")
    _require_time_grid(held_out_paths, times)
    training_paths = None
    if sampling_mode == "fixed":
        training_paths = (
            problem.sample(jr.fold_in(root_key, 100))
            if fixed_paths is None
            else fixed_paths
        )
        _require_time_grid(training_paths, times)

    num_steps = int(times.shape[0]) - 1
    slice_models: list[DomainFunction | None] = [None] * num_steps
    slice_diagnostics: list[DeepSplittingRegressionDiagnostics | None] = [
        None
    ] * num_steps
    working = solver
    next_predictor: Callable | DomainFunction = _TerminalPredictor(problem)

    for index in range(num_steps - 1, -1, -1):
        if sampling_mode == "fixed":
            if training_paths is None:
                raise RuntimeError("Fixed deep splitting paths are unavailable.")
            labels: DeepSplittingLabelBatch | Callable = deep_splitting_labels(
                problem,
                training_paths,
                next_predictor,
                index,
                key=jr.fold_in(root_key, 1000 + index),
            )
        else:
            labels = _label_provider(problem, next_predictor, index, times)
        objective = DeepSplittingRegressionObjective(
            problem,
            value_name=value_name,
            slice_index=index,
            labels=labels,
            value_weight=value_weight,
            label=f"deep-splitting-{index}",
        )
        initial_solver = working if warm_start else solver
        temporary = eqx.tree_at(
            lambda value: value.objectives,
            initial_solver,
            initial_solver.objectives + (objective,),
        )
        trained = temporary.solve(
            num_iter=inner_steps,
            optim=optim,
            seed=int(seed) + num_steps - index,
            jit=jit,
            keep_best=keep_best,
            log_every=log_every,
        )
        trained = eqx.tree_at(
            lambda value: value.objectives,
            trained,
            trained.objectives[:-1],
        )
        learned_slice = trained.ansatz_functions()[value_name]
        validation_labels = deep_splitting_labels(
            problem,
            held_out_paths,
            next_predictor,
            index,
            key=jr.fold_in(root_key, 2000 + index),
        )
        validation_objective = DeepSplittingRegressionObjective(
            problem,
            value_name=value_name,
            slice_index=index,
            labels=validation_labels,
        )
        slice_models[index] = learned_slice
        slice_diagnostics[index] = validation_objective.diagnostics(
            trained.ansatz_functions(),
            batch=validation_labels,
            key=jr.fold_in(root_key, 3000 + index),
        )
        next_predictor = learned_slice
        working = trained

    if any(value is None for value in slice_models) or any(
        value is None for value in slice_diagnostics
    ):
        raise RuntimeError("Deep splitting did not produce every requested time slice.")
    learned_slices = tuple(value for value in slice_models if value is not None)
    diagnostics_values = tuple(
        value for value in slice_diagnostics if value is not None
    )
    diagnostics = DeepSplittingDiagnostics(
        slice_indices=jnp.arange(num_steps),
        times=times[:-1],
        one_step_rmse=jnp.stack(
            tuple(value.one_step_rmse for value in diagnostics_values)
        ),
        target_rms=jnp.stack(tuple(value.target_rms for value in diagnostics_values)),
        source_rms=jnp.stack(tuple(value.source_rms for value in diagnostics_values)),
        valid_fraction=jnp.stack(
            tuple(value.valid_fraction for value in diagnostics_values)
        ),
        finite=jnp.stack(tuple(value.finite for value in diagnostics_values)),
    )
    solution = DeepSplittingSolution(
        problem,
        times,
        learned_slices,
        interpolation=interpolation,
    )
    return DeepSplittingResult(
        solver=working,
        solution=solution,
        diagnostics=diagnostics,
        completed_slices=num_steps,
        problem_id=problem.problem_id,
        process_id=problem.process_id,
        value_name=value_name,
    )


__all__ = [
    "DeepSplittingDiagnostics",
    "DeepSplittingInterpolation",
    "DeepSplittingResult",
    "DeepSplittingSamplingMode",
    "DeepSplittingSolution",
    "solve_deep_splitting",
]
