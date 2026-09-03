#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import prod
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ....nonlinear import (
    NonlinearResult,
    NonlinearSystemProblem,
    NonlinearTermination,
    RobustRoot,
    root,
)


CycleMap = Callable[[Array, Array, Any], Array]


class PeriodicShootingPlan(StrictModule):
    """Fixed-period single-shooting policy for a forced circulation cycle."""

    cycle_length: Array
    state_shape: tuple[int, ...] = eqx.field(static=True)
    termination: NonlinearTermination
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        cycle_length: ArrayLike,
        state_shape: Sequence[int],
        /,
        *,
        absolute_residual: float = 1.0e-8,
        relative_residual: float = 1.0e-8,
        maximum_steps: int = 64,
    ) -> None:
        cycle = jnp.asarray(cycle_length)
        shape = tuple(int(value) for value in state_shape)
        if (
            cycle.shape != ()
            or not bool(jnp.isfinite(cycle) & (cycle > 0.0))
            or not shape
            or any(value <= 0 for value in shape)
        ):
            raise ValueError(
                "Periodic shooting requires a positive cycle and state shape."
            )
        termination = NonlinearTermination(
            absolute_residual=absolute_residual,
            relative_residual=relative_residual,
            maximum_steps=maximum_steps,
        )
        self.cycle_length = cycle
        self.state_shape = shape
        self.termination = termination
        self.plan_id = canonical_fingerprint(
            {
                "kind": "circulation-periodic-shooting-plan",
                "cycle_length_ms": float(cycle).hex(),
                "state_shape": list(shape),
                "absolute_residual": float(absolute_residual).hex(),
                "relative_residual": float(relative_residual).hex(),
                "maximum_steps": int(maximum_steps),
            }
        )

    @property
    def state_size(self) -> int:
        return prod(self.state_shape)


class PreparedPeriodicShooting(StrictModule):
    plan: PeriodicShootingPlan
    cycle_map: CycleMap
    map_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: PeriodicShootingPlan,
        cycle_map: CycleMap,
        map_id: str,
        /,
    ) -> None:
        if not isinstance(plan, PeriodicShootingPlan):
            raise TypeError("plan must be a PeriodicShootingPlan.")
        if not callable(cycle_map):
            raise TypeError("cycle_map must be callable.")
        identifier = str(map_id).strip()
        if not identifier:
            raise ValueError("map_id must be non-empty.")
        self.plan = plan
        self.cycle_map = cycle_map
        self.map_id = identifier
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-circulation-periodic-shooting",
                "plan": plan.plan_id,
                "map": identifier,
            }
        )


class PeriodicClosureEvidence(StrictModule):
    closure_residual: Array
    maximum_absolute_closure: Array
    relative_closure: Array
    finite: Array
    successful: Array
    nonlinear_status: Array
    evidence_id: str = eqx.field(static=True)


class PeriodicShootingCandidate(StrictModule):
    initial_state: Array
    terminal_state: Array
    evidence: PeriodicClosureEvidence
    nonlinear_result: NonlinearResult
    prepared: PreparedPeriodicShooting


class CommittedPeriodicState(StrictModule):
    state: Array
    cycle_length: Array
    prepared_id: str = eqx.field(static=True)


def prepare_periodic_shooting(
    plan: PeriodicShootingPlan,
    cycle_map: CycleMap,
    map_id: str,
    /,
) -> PreparedPeriodicShooting:
    """Bind a differentiable one-cycle map to a fixed shooting plan."""

    return PreparedPeriodicShooting(plan, cycle_map, map_id)


def solve_periodic_shooting(
    prepared: PreparedPeriodicShooting,
    initial_state: ArrayLike,
    /,
    *,
    args: Any = None,
) -> PeriodicShootingCandidate:
    """Solve Φ_T(x) - x = 0 through the native nonlinear root substrate."""

    if not isinstance(prepared, PreparedPeriodicShooting):
        raise TypeError("prepared must be PreparedPeriodicShooting.")
    initial = jnp.asarray(initial_state)
    if initial.shape != prepared.plan.state_shape:
        raise ValueError(f"initial_state must have shape {prepared.plan.state_shape}.")
    if not jnp.issubdtype(initial.dtype, jnp.inexact):
        initial = initial.astype(float)
    if not bool(jnp.all(jnp.isfinite(initial))):
        raise ValueError("initial_state must be finite.")

    def closure(state: Array, user_args: Any) -> Array:
        terminal = jnp.asarray(
            prepared.cycle_map(state, prepared.plan.cycle_length, user_args)
        )
        return terminal - state

    problem = NonlinearSystemProblem(
        closure,
        problem_id=f"circulation-periodic-closure:{prepared.prepared_id}",
    )
    nonlinear_result = root(
        problem,
        initial,
        method=RobustRoot(),
        termination=prepared.plan.termination,
        args=args,
    )
    periodic_state = nonlinear_result.state
    terminal = jnp.asarray(
        prepared.cycle_map(periodic_state, prepared.plan.cycle_length, args)
    )
    residual = terminal - periodic_state
    maximum = jnp.max(jnp.abs(residual))
    scale = jnp.maximum(jnp.max(jnp.abs(periodic_state)), 1.0)
    relative = maximum / scale
    finite = (
        jnp.all(jnp.isfinite(periodic_state))
        & jnp.all(jnp.isfinite(terminal))
        & jnp.all(jnp.isfinite(residual))
    )
    threshold = prepared.plan.termination.residual_threshold(
        nonlinear_result.diagnostics.initial_residual_norm
    )
    successful = nonlinear_result.successful & finite & (maximum <= threshold)
    evidence = PeriodicClosureEvidence(
        residual,
        maximum,
        relative,
        finite,
        successful,
        jnp.asarray(nonlinear_result.status),
        canonical_fingerprint(
            {
                "kind": "circulation-periodic-closure-evidence",
                "prepared": prepared.prepared_id,
            }
        ),
    )
    return PeriodicShootingCandidate(
        periodic_state,
        terminal,
        evidence,
        nonlinear_result,
        prepared,
    )


def commit_periodic_state(
    candidate: PeriodicShootingCandidate,
    /,
) -> CommittedPeriodicState:
    """Fail closed unless shooting produced finite certified closure."""

    if not isinstance(candidate, PeriodicShootingCandidate):
        raise TypeError("candidate must be a PeriodicShootingCandidate.")
    if not bool(candidate.evidence.successful):
        raise ValueError("Cannot commit an unsuccessful periodic shooting candidate.")
    return CommittedPeriodicState(
        candidate.initial_state,
        candidate.prepared.plan.cycle_length,
        candidate.prepared.prepared_id,
    )


def pressure_volume_work(
    pressure: ArrayLike,
    volume: ArrayLike,
    /,
    *,
    axis: int = -1,
) -> Array:
    """Compute chamber work delivered to blood, -∮p dV, in kPa·mm³."""

    pressure_ = jnp.asarray(pressure)
    volume_ = jnp.asarray(volume)
    if pressure_.shape != volume_.shape or pressure_.ndim == 0:
        raise ValueError("pressure and volume must share a non-scalar shape.")
    resolved_axis = int(axis)
    if resolved_axis < 0:
        resolved_axis += pressure_.ndim
    if resolved_axis < 0 or resolved_axis >= pressure_.ndim:
        raise ValueError("axis is out of range.")
    if pressure_.shape[resolved_axis] < 2:
        raise ValueError("Pressure-volume work requires at least two cycle samples.")
    if not bool(jnp.all(jnp.isfinite(pressure_)) & jnp.all(jnp.isfinite(volume_))):
        raise ValueError("pressure and volume must be finite.")
    left = tuple(
        slice(None, -1) if index == resolved_axis else slice(None)
        for index in range(pressure_.ndim)
    )
    right = tuple(
        slice(1, None) if index == resolved_axis else slice(None)
        for index in range(pressure_.ndim)
    )
    segment_pressure = 0.5 * (pressure_[left] + pressure_[right])
    volume_change = volume_[right] - volume_[left]
    return -jnp.sum(segment_pressure * volume_change, axis=resolved_axis)


__all__ = [
    "CommittedPeriodicState",
    "CycleMap",
    "PeriodicClosureEvidence",
    "PeriodicShootingCandidate",
    "PeriodicShootingPlan",
    "PreparedPeriodicShooting",
    "commit_periodic_state",
    "prepare_periodic_shooting",
    "pressure_volume_work",
    "solve_periodic_shooting",
]
