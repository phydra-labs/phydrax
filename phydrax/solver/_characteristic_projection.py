#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from math import isfinite
from typing import Any, TypeAlias

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jaxtyping import Array, ArrayLike

from phydrax.conditions import Residual
from phydrax.domain import (
    BatchEvaluator,
    DomainComponent,
    DomainFunction,
    PointBatch,
    PointSampling,
)

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..dynamics import TimeGrid
from ..integration import fixed, from_samples, mean_over
from ..terms import ResidualPenalty
from ._differential import DifferentialProblem, DifferentialSolution
from ._diffrax_backend import solve_diffrax
from ._functional_solver import FunctionalSolver
from ._temporal_precision import TemporalPrecisionPolicy


CharacteristicVelocity: TypeAlias = Callable[[Array, Array, Any], ArrayLike]
CharacteristicWrap: TypeAlias = Callable[[Array], ArrayLike]


class _BackwardCharacteristicDrift(StrictModule):
    velocity: CharacteristicVelocity
    terminal_time: Array
    args: Any

    def __call__(self, pseudo_time: Array, state: Array, args: Any) -> Array:
        del args
        physical_time = self.terminal_time - pseudo_time
        value = jnp.asarray(self.velocity(physical_time, state, self.args))
        if value.shape != state.shape:
            raise ValueError("Characteristic velocity must preserve point shape.")
        return -value


class CharacteristicTraceResult(StrictModule):
    """Backward characteristic feet and complete Diffrax evidence."""

    foot_points: Array
    solution: DifferentialSolution
    valid: Array

    @property
    def successful(self) -> Array:
        return (
            jnp.asarray(self.solution.backend_successful, dtype=bool)
            & jnp.all(self.solution.valid)
            & jnp.all(self.valid)
        )


def trace_characteristics(
    velocity: CharacteristicVelocity,
    terminal_points: ArrayLike,
    t0: ArrayLike,
    t1: ArrayLike,
    /,
    *,
    args: Any = None,
    wrap: CharacteristicWrap | None = None,
    solver: Any | None = None,
    stepsize_controller: Any | None = None,
    adjoint: Any | None = None,
    dt0: ArrayLike | None = None,
    rtol: float = 1e-7,
    atol: float = 1e-9,
    max_steps: int | None = 4096,
    throw: bool = False,
    precision: TemporalPrecisionPolicy | None = None,
    problem_id: str | None = None,
) -> CharacteristicTraceResult:
    """Trace terminal points backward using a forward pseudo-time Diffrax solve."""
    if not callable(velocity):
        raise TypeError("velocity must be callable.")
    if wrap is not None and not callable(wrap):
        raise TypeError("wrap must be callable or None.")
    start = jnp.asarray(t0, dtype=float)
    end = jnp.asarray(t1, dtype=float)
    if start.shape != () or end.shape != ():
        raise ValueError("Characteristic time bounds must be scalar.")
    if not bool(jnp.isfinite(start) & jnp.isfinite(end) & (end > start)):
        raise ValueError("Characteristic time bounds must be finite and increasing.")
    points = jnp.asarray(terminal_points)
    if points.ndim < 1 or not bool(jnp.all(jnp.isfinite(points))):
        raise ValueError("terminal_points must be a non-empty finite array.")
    duration = end - start
    identifier = (
        canonical_fingerprint(
            {
                "kind": "backward-characteristic",
                "velocity": f"{type(velocity).__module__}.{type(velocity).__qualname__}",
                "point_shape": tuple(points.shape),
            }
        )
        if problem_id is None
        else str(problem_id)
    )
    drift = _BackwardCharacteristicDrift(velocity, end, args)
    differential = DifferentialProblem(
        drift,
        points,
        t0=0.0,
        t1=duration,
        args=None,
        problem_id=identifier,
    )
    solution = solve_diffrax(
        differential,
        save_times=jnp.asarray([duration]),
        solver=solver,
        stepsize_controller=stepsize_controller,
        adjoint=adjoint,
        dt0=dt0,
        rtol=rtol,
        atol=atol,
        dense=False,
        max_steps=max_steps,
        throw=throw,
        precision=precision,
    )
    feet = jnp.asarray(solution.states[-1])
    if wrap is not None:
        feet = jnp.asarray(wrap(feet))
        if feet.shape != points.shape:
            raise ValueError("Characteristic wrap must preserve point shape.")
    valid = jnp.all(jnp.isfinite(feet), axis=-1) if feet.ndim > 1 else jnp.isfinite(feet)
    return CharacteristicTraceResult(
        foot_points=feet,
        solution=solution,
        valid=valid,
    )


class CharacteristicProjectionProblem(StrictModule):
    """Fixed-capacity characteristic pullback followed by neural field projection."""

    component: DomainComponent
    sampling: PointSampling
    velocity: CharacteristicVelocity
    wrap: CharacteristicWrap | None
    args: Any
    field: str = eqx.field(static=True)
    coordinate_label: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        field: str,
        component: DomainComponent,
        sampling: PointSampling,
        velocity: CharacteristicVelocity,
        /,
        *,
        coordinate_label: str = "x",
        wrap: CharacteristicWrap | None = None,
        args: Any = None,
        problem_id: str | None = None,
    ):
        name = str(field)
        coordinate = str(coordinate_label)
        if not name or not coordinate:
            raise ValueError("field and coordinate_label must be non-empty.")
        if not isinstance(component, DomainComponent):
            raise TypeError("component must be a DomainComponent.")
        if coordinate not in component.domain.labels:
            raise ValueError("coordinate_label must belong to the component domain.")
        if not isinstance(sampling, PointSampling) or not isinstance(sampling.count, int):
            raise TypeError("sampling must be a fixed-count PointSampling.")
        if not callable(velocity):
            raise TypeError("velocity must be callable.")
        if wrap is not None and not callable(wrap):
            raise TypeError("wrap must be callable or None.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "characteristic-projection",
                    "field": name,
                    "coordinate": coordinate,
                    "sampling_count": sampling.count,
                    "velocity": f"{type(velocity).__module__}.{type(velocity).__qualname__}",
                }
            )
            if problem_id is None
            else str(problem_id)
        )
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        self.field = name
        self.component = component
        self.sampling = sampling
        self.velocity = velocity
        self.coordinate_label = coordinate
        self.wrap = wrap
        self.args = args
        self.problem_id = identifier


class _StoredBatchTarget(StrictModule, BatchEvaluator):
    values: cx.Field

    def __call_batch__(self, batch: Any, /, *, key: Any = None, **kwargs: Any):
        del batch, key, kwargs
        return self.values

    def __call__(self, *args: Any, key: Any = None, **kwargs: Any):
        del args, key, kwargs
        return self.values.data


class CharacteristicProjectionResult(StrictModule):
    """Learned time-slice fields with characteristic and projection evidence."""

    solver: FunctionalSolver
    times: Array
    fields: tuple[DomainFunction, ...]
    traces: tuple[CharacteristicTraceResult, ...]
    projection_losses: Array
    completed_steps: int = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return jnp.asarray(
            self.completed_steps == int(self.times.shape[0]) - 1
            and all(bool(trace.successful) for trace in self.traces),
            dtype=bool,
        ) & jnp.all(jnp.isfinite(self.projection_losses))


def _replace_coordinate_batch(
    batch: PointBatch,
    label: str,
    coordinates: Array,
    /,
) -> PointBatch:
    old = batch.points[label]
    if not isinstance(old, cx.Field):
        raise TypeError("Characteristic coordinate payload must be a coordax.Field.")
    replacement = cx.Field(jnp.asarray(coordinates), dims=old.dims)
    points = dict(batch.points)
    points[label] = replacement
    return PointBatch(points, batch.structure, metadata=batch.metadata)


def solve_characteristic_projection(
    solver: FunctionalSolver,
    problem: CharacteristicProjectionProblem,
    time_grid: TimeGrid,
    /,
    *,
    inner_num_iter: int,
    optim: Any = None,
    seed: int = 0,
    jit: bool = True,
    keep_best: bool = True,
    log_every: int = 0,
    maximum_projection_loss: float = float("inf"),
    characteristic_solver: Any | None = None,
    characteristic_stepsize_controller: Any | None = None,
    characteristic_adjoint: Any | None = None,
    characteristic_dt0: ArrayLike | None = None,
    characteristic_rtol: float = 1e-7,
    characteristic_atol: float = 1e-9,
    characteristic_max_steps: int | None = 4096,
    characteristic_precision: TemporalPrecisionPolicy | None = None,
) -> CharacteristicProjectionResult:
    """Advance one learned field by Diffrax characteristics and fixed projections."""
    if not isinstance(solver, FunctionalSolver):
        raise TypeError("solver must be a FunctionalSolver.")
    if not isinstance(problem, CharacteristicProjectionProblem):
        raise TypeError("problem must be a CharacteristicProjectionProblem.")
    if not isinstance(time_grid, TimeGrid):
        raise TypeError("time_grid must be a TimeGrid.")
    if problem.field not in solver.ansatz_functions():
        raise KeyError(f"Missing characteristic field {problem.field!r}.")
    if not solver[problem.field].domain.same_support(problem.component.domain):
        raise ValueError("Characteristic field and component domains must agree.")
    inner_steps = int(inner_num_iter)
    if inner_steps < 1:
        raise ValueError("inner_num_iter must be positive.")
    loss_limit = float(maximum_projection_loss)
    if not isfinite(loss_limit) and loss_limit != float("inf"):
        raise ValueError("maximum_projection_loss must be finite or positive infinity.")
    if loss_limit < 0.0:
        raise ValueError("maximum_projection_loss must be non-negative.")
    optimizer = optax.adam(1e-3) if optim is None else optim
    root_key = jr.key(int(seed))
    working = solver
    base_term_count = len(solver.terms)
    fields: list[DomainFunction] = [working[problem.field]]
    traces: list[CharacteristicTraceResult] = []
    losses: list[Array] = []

    for index, (left, right) in enumerate(
        zip(tuple(time_grid.times[:-1]), tuple(time_grid.times[1:]), strict=True)
    ):
        batch = problem.component.sample(
            problem.sampling,
            key=jr.fold_in(root_key, 1000 + index),
        )
        if not isinstance(batch, PointBatch):
            raise TypeError("Characteristic projection requires a PointBatch.")
        coordinate_field = batch.points[problem.coordinate_label]
        if not isinstance(coordinate_field, cx.Field):
            raise TypeError("Characteristic coordinates must be a coordax.Field.")
        terminal_coordinates = jnp.asarray(coordinate_field.data)
        trace = trace_characteristics(
            problem.velocity,
            terminal_coordinates,
            left,
            right,
            args=problem.args,
            wrap=problem.wrap,
            solver=characteristic_solver,
            stepsize_controller=characteristic_stepsize_controller,
            adjoint=characteristic_adjoint,
            dt0=characteristic_dt0,
            rtol=characteristic_rtol,
            atol=characteristic_atol,
            max_steps=characteristic_max_steps,
            throw=False,
            precision=characteristic_precision,
            problem_id=f"{problem.problem_id}:step-{index}:trace",
        )
        traces.append(trace)
        if not bool(trace.successful):
            break
        foot_batch = _replace_coordinate_batch(
            batch,
            problem.coordinate_label,
            trace.foot_points,
        )
        previous = working[problem.field]
        target_values = previous(
            foot_batch,
            key=jr.fold_in(root_key, 2000 + index),
        )
        if not isinstance(target_values, cx.Field):
            raise TypeError("Characteristic target field must return coordax.Field.")
        target_values = cx.Field(
            jax.lax.stop_gradient(jnp.asarray(target_values.data)),
            dims=target_values.dims,
        )
        target = DomainFunction(
            domain=problem.component.domain,
            deps=problem.component.domain.labels,
            func=_StoredBatchTarget(target_values),
        )
        condition = Residual(
            problem.field,
            problem.component,
            lambda field, target=target: field - target,
            label=f"characteristic-projection-{index}",
        )
        realization = from_samples(mean_over(condition.on), batch)
        term = ResidualPenalty(
            condition,
            fixed(realization),
            label=f"characteristic-projection-{index}",
        )
        temporary = working._append_training_terms(
            term,
            key=jr.fold_in(root_key, 3000 + index),
        )
        trained = temporary.solve(
            num_iter=inner_steps,
            optim=optimizer,
            seed=int(seed) + index,
            jit=jit,
            keep_best=keep_best,
            log_every=log_every,
        )
        loss = term.loss(
            trained.ansatz_functions(),
            key=jr.fold_in(root_key, 4000 + index),
        )
        losses.append(loss)
        if not bool(jnp.isfinite(loss)) or float(loss) > loss_limit:
            break
        working = trained._retain_training_prefix(base_term_count)
        fields.append(working[problem.field])

    return CharacteristicProjectionResult(
        solver=working,
        times=time_grid.times,
        fields=tuple(fields),
        traces=tuple(traces),
        projection_losses=(jnp.stack(tuple(losses)) if losses else jnp.empty((0,))),
        completed_steps=len(fields) - 1,
        problem_id=problem.problem_id,
    )


__all__ = [
    "CharacteristicProjectionProblem",
    "CharacteristicProjectionResult",
    "CharacteristicTraceResult",
    "CharacteristicVelocity",
    "CharacteristicWrap",
    "solve_characteristic_projection",
    "trace_characteristics",
]
