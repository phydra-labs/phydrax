#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import ceil, isfinite
from typing import Any, Literal, TypeAlias

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._delay import (
    ConstantDelay,
    DelayDifferentialProblem,
    DerivativeDelay,
    DistributedDelay,
    FunctionalDelay,
    NeutralDelayProblem,
    StateDependentDelay,
)
from ._geometric import AbstractGeometricSolver, RKMK, SRKMK


DelayExecutionMode: TypeAlias = Literal["whole", "segmented"]
DelayHistoryMode: TypeAlias = Literal["full", "rolling"]


class DelayHistoryRequirements(StrictModule):
    """Static channels required from every accepted delay-history entry."""

    values: bool = eqx.field(static=True)
    derivatives: bool = eqx.field(static=True)
    increments: bool = eqx.field(static=True)
    jump_limits: bool = eqx.field(static=True)


class DelayExecutionPlan(StrictModule):
    """Compiled, immutable capabilities for one delay solve.

    This object contains no mutable numerical state. Backends construct it once before
    tracing and use it as the single source of truth for solver, history, geometry,
    discontinuity, and driver requirements.
    """

    minimum_delay: Array
    maximum_delay: Array | None
    stage_time_extent: Array
    constant_lags: tuple[Array, ...]
    state_dependent_delays: tuple[StateDependentDelay, ...]
    history: DelayHistoryRequirements
    execution: DelayExecutionMode = eqx.field(static=True)
    history_mode: DelayHistoryMode = eqx.field(static=True)
    equation_kind: str = eqx.field(static=True)
    solver_name: str = eqx.field(static=True)
    delay_names: tuple[str, ...] = eqx.field(static=True)
    delay_types: tuple[str, ...] = eqx.field(static=True)
    stochastic: bool = eqx.field(static=True)
    interpretation: str = eqx.field(static=True)
    geometric: bool = eqx.field(static=True)
    state_geometry_id: str | None = eqx.field(static=True)
    has_distributed_delays: bool = eqx.field(static=True)
    has_functional_delays: bool = eqx.field(static=True)
    has_infinite_memory: bool = eqx.field(static=True)
    supports_rolling: bool = eqx.field(static=True)
    supports_segmented: bool = eqx.field(static=True)


_DelayProblemContract: TypeAlias = DelayDifferentialProblem | NeutralDelayProblem

def stage_time_extent(solver: dfx.AbstractSolver, /) -> Array:
    """Return a certified positive bound for every solver stage abscissa."""

    if isinstance(solver, dfx.AbstractWrappedSolver):
        return stage_time_extent(solver.solver)
    if isinstance(solver, AbstractGeometricSolver):
        nodes = solver.stage_abscissae
        extent = solver.causal_stage_extent
        if not nodes or any(not isfinite(node) or node < 0.0 for node in nodes):
            raise ValueError(
                "Geometric solver stage_abscissae must be finite and nonnegative."
            )
        if not isfinite(extent) or extent <= 0.0 or any(node > extent for node in nodes):
            raise ValueError(
                "Geometric solver causal_stage_extent must be finite, positive, "
                "and bound every stage abscissa."
            )
        return jnp.asarray(extent)
    if isinstance(solver, (dfx.Euler, dfx.EulerHeun, dfx.ImplicitEuler)):
        return jnp.asarray(1.0)
    if not isinstance(solver, dfx.AbstractRungeKutta):
        raise ValueError(
            "Delay execution requires a solver with declared causal stage times."
        )
    tableau = solver.tableau
    if isinstance(tableau, dfx.MultiButcherTableau):
        stages = jnp.concatenate(
            tuple(jnp.asarray(component.c) for component in tableau.tableaus)
        )
    else:
        stages = jnp.asarray(tableau.c)
    return jnp.maximum(1.0, jnp.max(stages))


def resolve_delay_solver(
    problem: _DelayProblemContract,
    solver: Any | None,
    /,
) -> dfx.AbstractSolver:
    """Resolve the default solver without duplicating backend-specific policy."""

    if solver is None:
        geometry = problem.state_geometry
        if problem.stochastic:
            if geometry is not None and not geometry.trivial:
                if problem.interpretation == "ito":
                    raise ValueError(
                        "Nontrivial Itô geometry requires an explicit second-order "
                        "geometric interpretation."
                    )
                return SRKMK(geometry)
            return dfx.Euler() if problem.interpretation == "ito" else dfx.EulerHeun()
        if geometry is not None and not geometry.trivial:
            return RKMK(geometry)
        return dfx.Tsit5()
    if not isinstance(solver, dfx.AbstractSolver):
        raise TypeError("solver must be a Diffrax AbstractSolver or None.")
    return solver


def _validate_geometry(
    problem: _DelayProblemContract,
    solver: dfx.AbstractSolver,
    /,
) -> None:
    geometry = problem.state_geometry
    geometric = isinstance(solver, AbstractGeometricSolver)
    if geometry is None:
        if geometric:
            raise ValueError(
                "A geometric solver requires the delay problem to declare state_geometry."
            )
        return
    if not geometry.trivial and not geometric:
        raise ValueError(
            "A nontrivial state_geometry requires an AbstractGeometricSolver; "
            f"got {type(solver).__name__}."
        )
    if geometric and solver.geometry.geometry_id != geometry.geometry_id:
        raise ValueError(
            "Geometric solver and delay problem must carry the same state_geometry_id."
        )
    if problem.stochastic and not geometry.trivial:
        if problem.interpretation == "ito":
            raise ValueError(
                "Nontrivial Itô geometry requires an explicit second-order geometric "
                "interpretation."
            )
        if not isinstance(solver, SRKMK):
            raise ValueError(
                "Intrinsic Stratonovich delay execution requires phydrax.solver.SRKMK."
            )


def compile_delay_execution_plan(
    problem: _DelayProblemContract,
    solver: dfx.AbstractSolver,
    /,
    *,
    execution: DelayExecutionMode,
    history_mode: DelayHistoryMode,
) -> DelayExecutionPlan:
    """Validate and compile all static capabilities for one delay execution."""

    if execution not in ("whole", "segmented"):
        raise ValueError("execution must be 'whole' or 'segmented'.")
    if history_mode not in ("full", "rolling"):
        raise ValueError("history_mode must be 'full' or 'rolling'.")
    _validate_geometry(problem, solver)

    constant_lags: list[Array] = []
    state_dependent: list[StateDependentDelay] = []
    has_distributed = False
    has_functional = False
    requires_derivatives = False
    for term in problem.delay_terms:
        if isinstance(term, ConstantDelay):
            constant_lags.append(term.delay)
        elif isinstance(term, StateDependentDelay):
            state_dependent.append(term)
        elif isinstance(term, DistributedDelay):
            constant_lags.extend(tuple(term.nodes))
            has_distributed = True
        elif isinstance(term, FunctionalDelay):
            constant_lags.extend(tuple(term.discontinuity_lags))
            has_functional = True
        elif isinstance(term, DerivativeDelay):
            requires_derivatives = True
            if isinstance(term.delay, ConstantDelay):
                constant_lags.append(term.delay.delay)
            else:
                state_dependent.append(term.delay)
        else:
            raise TypeError(f"Unsupported delay term {type(term).__name__}.")

    maximum_delay = problem.maximum_delay
    has_infinite_memory = any(
        isinstance(term, FunctionalDelay) and term.infinite_memory
        for term in problem.delay_terms
    )
    bounded = maximum_delay is not None and not has_infinite_memory
    if (history_mode == "rolling" or execution == "segmented") and not bounded:
        raise ValueError(
            "Rolling and segmented delay execution require every term to declare "
            "a finite maximum delay."
        )

    base_equation_kind = (
        "neutral"
        if problem.neutral
        else "functional-retarded"
        if has_functional
        else "retarded"
    )
    equation_kind = (
        f"stochastic-{base_equation_kind}"
        if problem.stochastic
        else base_equation_kind
    )

    return DelayExecutionPlan(
        minimum_delay=jnp.asarray(problem.minimum_delay),
        maximum_delay=(None if maximum_delay is None else jnp.asarray(maximum_delay)),
        stage_time_extent=stage_time_extent(solver),
        constant_lags=tuple(constant_lags),
        state_dependent_delays=tuple(state_dependent),
        history=DelayHistoryRequirements(
            values=True,
            derivatives=requires_derivatives,
            increments=bool(problem.stochastic),
            jump_limits=False,
        ),
        execution=execution,
        history_mode=history_mode,
        equation_kind=equation_kind,
        solver_name=type(solver).__name__,
        delay_names=problem.delay_names,
        delay_types=tuple(type(term).__name__ for term in problem.delay_terms),
        stochastic=bool(problem.stochastic),
        interpretation=problem.interpretation,
        geometric=isinstance(solver, AbstractGeometricSolver),
        state_geometry_id=problem.state_geometry_id,
        has_distributed_delays=has_distributed,
        has_functional_delays=has_functional,
        supports_rolling=bounded,
        has_infinite_memory=has_infinite_memory,
        supports_segmented=bounded,
    )


def fixed_delay_history_capacity(
    maximum_lag: ArrayLike,
    nominal_step: ArrayLike,
    /,
    *,
    margin: int = 2,
    breakpoints: ArrayLike | None = None,
    initial_time: ArrayLike = 0.0,
) -> int:
    """Exact fixed-step lag-window count including declared step breakpoints."""

    if not isinstance(margin, int) or isinstance(margin, bool) or margin < 1:
        raise ValueError("history margin must be a positive integer.")
    lag = float(jax.device_get(jnp.asarray(maximum_lag, dtype=float)))
    step = float(jax.device_get(jnp.asarray(nominal_step, dtype=float)))
    if not np.isfinite(lag) or lag <= 0.0:
        raise ValueError("maximum_lag must be finite and positive.")
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("nominal_step must be finite and positive.")
    origin = float(jax.device_get(jnp.asarray(initial_time, dtype=float)))
    if not np.isfinite(origin):
        raise ValueError("initial_time must be finite.")
    extra = 0
    if breakpoints is not None:
        values = np.asarray(jax.device_get(jnp.asarray(breakpoints, dtype=float)))
        if values.ndim != 1:
            raise ValueError("history-capacity breakpoints must be rank-1.")
        finite = np.unique(values[np.isfinite(values) & (values > origin)])
        additional = []
        previous = origin
        epsilon = 100.0 * np.finfo(float).eps
        for breakpoint in finite:
            step_count = (breakpoint - previous) / step
            tolerance = epsilon * max(1.0, abs(step_count))
            if abs(step_count - round(step_count)) > tolerance:
                additional.append(breakpoint)
            previous = breakpoint
        if additional:
            additional_array = np.asarray(additional)
            left = np.searchsorted(
                additional_array,
                additional_array - lag,
                side="left",
            )
            extra = int(np.max(np.arange(1, len(additional) + 1) - left))
    return ceil(lag / step) + extra + margin


__all__: list[str] = []
