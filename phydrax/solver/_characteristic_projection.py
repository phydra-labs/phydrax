#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from math import isfinite, prod
from typing import Any, Literal, TypeAlias

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
from ..integration import fixed, from_samples, IntegrationRealization, mean_over
from ..stochastic import (
    PreparedStochasticPathEnsemble,
    solve_stochastic_path_ensemble,
    StochasticPathEnsembleResult,
)
from ..terms import ResidualPenalty
from ._differential import DifferentialProblem, DifferentialSolution
from ._diffrax_backend import solve_diffrax
from ._functional_solver import FunctionalSolver
from ._hybrid_schedule import (
    execute_hybrid_schedule,
    PreparedHybridSchedule,
)
from ._temporal_precision import TemporalPrecisionPolicy


CharacteristicVelocity: TypeAlias = Callable[[Array, Array, Any], ArrayLike]
CharacteristicWrap: TypeAlias = Callable[[Array], ArrayLike]
CharacteristicBoundaryAction: TypeAlias = Literal[
    "stop",
    "reflect",
    "absorb",
    "reset",
    "periodic",
]


class CharacteristicBoundaryPolicy(StrictModule):
    """DCD-prepared fixed-capacity boundary action for independent paths."""

    schedule: PreparedHybridSchedule
    brackets: Array
    action: CharacteristicBoundaryAction = eqx.field(static=True)
    reset_map: Callable[[Array], ArrayLike] | None = eqx.field(static=True)
    priority: int = eqx.field(static=True)

    def __init__(
        self,
        schedule: PreparedHybridSchedule,
        action: CharacteristicBoundaryAction,
        brackets: ArrayLike,
        /,
        *,
        reset_map: Callable[[Array], ArrayLike] | None = None,
        priority: int = 0,
    ):
        if not isinstance(schedule, PreparedHybridSchedule):
            raise TypeError("schedule must be PreparedHybridSchedule.")
        if action not in ("stop", "reflect", "absorb", "reset", "periodic"):
            raise ValueError("Unknown characteristic boundary action.")
        if action in ("reflect", "reset", "periodic") and reset_map is None:
            raise ValueError(f"{action} boundary action requires reset_map.")
        intervals = jnp.asarray(brackets, dtype=float)
        if (
            intervals.ndim != 2
            or intervals.shape[-1] != 2
            or int(intervals.shape[0]) == 0
        ):
            raise ValueError("Boundary brackets must have fixed shape (N, 2).")
        if (
            bool(jnp.any(~jnp.isfinite(intervals)))
            or bool(jnp.any(intervals[:, 1] <= intervals[:, 0]))
            or bool(jnp.any((intervals < 0.0) | (intervals > 1.0)))
        ):
            raise ValueError(
                "Boundary brackets must be finite increasing fractions in [0, 1]."
            )
        self.schedule = schedule
        self.brackets = intervals
        self.action = action
        self.reset_map = reset_map
        self.priority = int(priority)

    def apply_terminal(self, feet: Array, /) -> Array:
        if self.action == "absorb":
            return jnp.zeros_like(feet)
        if self.reset_map is None:
            return feet
        result = jnp.asarray(self.reset_map(feet))
        if result.shape != feet.shape:
            raise ValueError("Characteristic reset_map must preserve point shape.")
        return result


class DiffusiveCharacteristicPlan(StrictModule):
    """Finite SST path ensemble and CID integration realization for diffusion."""

    ensemble: PreparedStochasticPathEnsemble
    integration: IntegrationRealization
    weights: Array
    interpretation: Literal["ito", "stratonovich"] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        ensemble: PreparedStochasticPathEnsemble,
        integration: IntegrationRealization,
        weights: ArrayLike,
        /,
        *,
        interpretation: Literal["ito", "stratonovich"] = "ito",
    ):
        if not isinstance(ensemble, PreparedStochasticPathEnsemble):
            raise TypeError("ensemble must be PreparedStochasticPathEnsemble.")
        if not isinstance(integration, IntegrationRealization):
            raise TypeError("integration must be IntegrationRealization.")
        if interpretation not in ("ito", "stratonovich"):
            raise ValueError("Unknown stochastic interpretation.")
        values = jnp.asarray(weights, dtype=float)
        if values.shape != (ensemble.plan.path_count,):
            raise ValueError("Diffusive cubature weights must match path capacity.")
        if bool(jnp.any(~jnp.isfinite(values))) or not bool(jnp.sum(values) > 0.0):
            raise ValueError(
                "Diffusive cubature weights must be finite with positive sum."
            )
        self.ensemble = ensemble
        self.integration = integration
        self.weights = values / jnp.sum(values)
        self.interpretation = interpretation
        self.plan_id = (
            f"diffusive-characteristic:{ensemble.prepared_id}:"
            f"{type(integration.plan).__module__}.{type(integration.plan).__qualname__}:"
            f"{interpretation}"
        )


class DiffusiveCharacteristicResult(StrictModule):
    feet: Array
    weights: Array
    path_mask: Array
    ensemble_result: StochasticPathEnsembleResult
    integration: IntegrationRealization
    plan_id: str = eqx.field(static=True)


def trace_diffusive_characteristics(
    plan: DiffusiveCharacteristicPlan,
    /,
) -> DiffusiveCharacteristicResult:
    """Execute one fixed finite diffusion ensemble without synthesizing paths."""
    if not isinstance(plan, DiffusiveCharacteristicPlan):
        raise TypeError("plan must be DiffusiveCharacteristicPlan.")
    result = solve_stochastic_path_ensemble(plan.ensemble)
    final = result.states[:, -1]
    feet = jnp.swapaxes(final, 0, -2) if final.ndim >= 3 else final
    return DiffusiveCharacteristicResult(
        feet,
        plan.weights,
        result.path_valid,
        result,
        plan.integration,
        plan.plan_id,
    )


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
    event_times: Array | None = None
    event_states: Array | None = None
    event_actions: Array | None = None
    event_mask: Array | None = None
    schedule_id: str = eqx.field(static=True, default="")

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
    boundary_policy: CharacteristicBoundaryPolicy | None = None,
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
    if boundary_policy is not None and not isinstance(
        boundary_policy,
        CharacteristicBoundaryPolicy,
    ):
        raise TypeError("boundary_policy must be CharacteristicBoundaryPolicy or None.")
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
        dense=boundary_policy is not None,
        max_steps=max_steps,
        throw=throw,
        precision=precision,
    )
    feet = jnp.asarray(solution.states[-1])
    point_shape = points.shape[:-1] if points.ndim > 1 else points.shape
    event_capacity = (
        0
        if boundary_policy is None
        else boundary_policy.schedule.replay_policy.maximum_events
    )
    if boundary_policy is None:
        event_times = jnp.zeros(point_shape + (0,), dtype=feet.real.dtype)
        event_states = jnp.zeros(
            point_shape + (0,) + points.shape[-1:],
            dtype=feet.dtype,
        )
        event_actions = jnp.zeros(point_shape + (0,), dtype=jnp.int32)
        event_mask = jnp.zeros(point_shape + (0,), dtype=bool)
        capacity_valid = jnp.ones(point_shape, dtype=bool)
    else:
        if boundary_policy.schedule.state_shape != points.shape[-1:]:
            raise ValueError(
                "Prepared boundary schedule state shape must match one point."
            )
        if solution.interpolation is None:
            raise RuntimeError(
                "Eventful characteristic solve requires dense interpolation."
            )
        flat_count = prod(point_shape) if point_shape else 1
        scaled_brackets = boundary_policy.brackets * duration
        flat_feet = feet.reshape((flat_count,) + points.shape[-1:])
        schedule_results = []
        for point_index in range(flat_count):
            schedule_results.append(
                execute_hybrid_schedule(
                    boundary_policy.schedule,
                    lambda pseudo_time, _args, index=point_index: (
                        solution.interpolation.evaluate(pseudo_time).reshape(
                            (flat_count,) + points.shape[-1:]
                        )[index]
                    ),
                    scaled_brackets,
                )
            )
        event_times = jnp.stack(
            tuple(end - result.event_times for result in schedule_results)
        ).reshape(point_shape + (event_capacity,))
        event_states = jnp.stack(
            tuple(result.event_states_after for result in schedule_results)
        ).reshape(point_shape + (event_capacity,) + points.shape[-1:])
        event_mask = jnp.stack(
            tuple(result.valid for result in schedule_results)
        ).reshape(point_shape + (event_capacity,))
        action_code = {
            "stop": 1,
            "reflect": 2,
            "absorb": 3,
            "reset": 4,
            "periodic": 5,
        }[boundary_policy.action]
        event_actions = jnp.where(
            event_mask,
            action_code,
            0,
        ).astype(jnp.int32)
        capacity_valid = ~jnp.stack(
            tuple(result.capacity_exceeded for result in schedule_results)
        ).reshape(point_shape)
        counts = jnp.sum(event_mask.reshape((flat_count, event_capacity)), axis=-1)
        last = jnp.maximum(counts - 1, 0)
        last_states = event_states.reshape(
            (flat_count, event_capacity) + points.shape[-1:]
        )[jnp.arange(flat_count), last]
        acted = boundary_policy.apply_terminal(last_states)
        flat_feet = jnp.where(
            (counts > 0).reshape((flat_count,) + (1,) * len(points.shape[-1:])),
            acted,
            flat_feet,
        )
        feet = flat_feet.reshape(points.shape)
    if wrap is not None:
        feet = jnp.asarray(wrap(feet))
        if feet.shape != points.shape:
            raise ValueError("Characteristic wrap must preserve point shape.")
    valid = (
        jnp.all(jnp.isfinite(feet), axis=-1) if feet.ndim > 1 else jnp.isfinite(feet)
    ) & capacity_valid
    return CharacteristicTraceResult(
        foot_points=feet,
        solution=solution,
        valid=valid,
        event_times=event_times,
        event_states=event_states,
        event_actions=event_actions,
        event_mask=event_mask,
        schedule_id=(
            "" if boundary_policy is None else boundary_policy.schedule.preparation_id
        ),
    )


class CharacteristicProjectionProblem(StrictModule):
    """Fixed-capacity characteristic pullback followed by neural field projection."""

    component: DomainComponent
    sampling: PointSampling
    velocity: CharacteristicVelocity
    boundary_policy: CharacteristicBoundaryPolicy | None
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
        boundary_policy: CharacteristicBoundaryPolicy | None = None,
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
        if boundary_policy is not None and not isinstance(
            boundary_policy,
            CharacteristicBoundaryPolicy,
        ):
            raise TypeError(
                "boundary_policy must be CharacteristicBoundaryPolicy or None."
            )
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
        self.boundary_policy = boundary_policy
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
            boundary_policy=problem.boundary_policy,
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
    "CharacteristicBoundaryAction",
    "CharacteristicBoundaryPolicy",
    "CharacteristicProjectionProblem",
    "CharacteristicProjectionResult",
    "CharacteristicTraceResult",
    "CharacteristicVelocity",
    "CharacteristicWrap",
    "DiffusiveCharacteristicPlan",
    "DiffusiveCharacteristicResult",
    "solve_characteristic_projection",
    "trace_characteristics",
    "trace_diffusive_characteristics",
]
