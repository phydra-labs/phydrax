#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..dynamics import DifferentialAlgebraicSystem, HeldInputPolicy, TimeGrid
from ..solver import (
    DAEInitializationSpec,
    DAESolvePolicy,
    DifferentialAlgebraicProblem,
    DifferentialAlgebraicSolution,
    solve_dae,
)
from ._direct_collocation import DirectCollocationResult
from ._trajectory_optimization import (
    TrajectoryOptimizationContext,
    TrajectoryOptimizationView,
)


DirectCollocationReplayFailureMode = Literal["record", "error"]


def _tolerance(value: float, owner: str, /) -> float:
    resolved = float(value)
    if not isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{owner} must be finite and non-negative.")
    return resolved


class DirectCollocationReplayPolicy(StrictModule):
    """Native DAE replay grid, solve policy, thresholds, and failure semantics."""

    dae_policy: DAESolvePolicy
    time_grid: TimeGrid | None
    node_state_tolerance: float = eqx.field(static=True)
    terminal_state_tolerance: float = eqx.field(static=True)
    algebraic_constraint_tolerance: float = eqx.field(static=True)
    failure_mode: DirectCollocationReplayFailureMode = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        dae_policy: DAESolvePolicy | None = None,
        time_grid: TimeGrid | None = None,
        node_state_tolerance: float = 1.0e-5,
        terminal_state_tolerance: float = 1.0e-5,
        algebraic_constraint_tolerance: float = 1.0e-6,
        failure_mode: DirectCollocationReplayFailureMode = "record",
        policy_id: str = "control:direct-collocation:dae-replay",
    ):
        dae = DAESolvePolicy() if dae_policy is None else dae_policy
        if not isinstance(dae, DAESolvePolicy):
            raise TypeError("dae_policy must be DAESolvePolicy or None.")
        if time_grid is not None and not isinstance(time_grid, TimeGrid):
            raise TypeError("time_grid must be TimeGrid or None.")
        if failure_mode not in ("record", "error"):
            raise ValueError("failure_mode must be 'record' or 'error'.")
        if not isinstance(policy_id, str) or not policy_id:
            raise ValueError("policy_id must be a non-empty string.")
        self.dae_policy = dae
        self.time_grid = time_grid
        self.node_state_tolerance = _tolerance(
            node_state_tolerance, "node_state_tolerance"
        )
        self.terminal_state_tolerance = _tolerance(
            terminal_state_tolerance, "terminal_state_tolerance"
        )
        self.algebraic_constraint_tolerance = _tolerance(
            algebraic_constraint_tolerance,
            "algebraic_constraint_tolerance",
        )
        self.failure_mode = failure_mode
        self.policy_id = policy_id


class DirectCollocationReplayEvidence(StrictModule):
    """Independent causal DAE solution and discrepancy against collocation."""

    input_policy: HeldInputPolicy
    solution: DifferentialAlgebraicSolution
    collocated_states: Array
    state_discrepancy: Array
    maximum_node_discrepancy: Array
    terminal_discrepancy: Array
    maximum_algebraic_residual: Array
    passed: Array
    source_result_id: str = eqx.field(static=True)
    source_problem_id: str = eqx.field(static=True)
    source_plan_id: str = eqx.field(static=True)
    replay_id: str = eqx.field(static=True)


def replay_direct_collocation(
    result: DirectCollocationResult,
    policy: DirectCollocationReplayPolicy,
    /,
) -> DirectCollocationReplayEvidence:
    """Replay one successful unbatched controlled-DAE collocation result."""
    if not isinstance(result, DirectCollocationResult):
        raise TypeError("result must be a DirectCollocationResult.")
    if not isinstance(policy, DirectCollocationReplayPolicy):
        raise TypeError("policy must be DirectCollocationReplayPolicy.")
    if not bool(np.asarray(result.successful)):
        raise ValueError("DAE replay requires a successful direct-collocation result.")
    problem = result.compilation.problem
    if problem.case_shape:
        raise ValueError("Controlled DAE replay currently requires an unbatched result.")
    if not isinstance(problem.dynamics, DifferentialAlgebraicSystem):
        raise TypeError("Direct-collocation DAE replay requires implicit DAE dynamics.")
    system = problem.dynamics
    if system.input_layout is None:
        raise ValueError("Direct-collocation DAE replay requires input-aware dynamics.")
    physical_grid = result.trajectory.time_grid
    input_policy = HeldInputPolicy(
        physical_grid.times,
        result.decision.controls,
        input_layout=system.input_layout,
        node_side="left",
        policy_id=canonical_fingerprint(
            {
                "kind": "direct-collocation-held-input",
                "result": result.result_id,
                "control": result.trajectory.control_id,
                "input_layout": system.input_layout.layout_id,
            }
        ),
    )
    callback_args = (
        problem.args
        if problem.parameter_space is None
        and not result.compilation.plan.variable_duration
        else TrajectoryOptimizationContext(
            problem.args,
            result.decision.parameters,
            result.duration,
        )
    )
    replay_grid = physical_grid if policy.time_grid is None else policy.time_grid
    if replay_grid.t0 < physical_grid.t0 or replay_grid.t1 > physical_grid.t1:
        raise ValueError("Replay grid must lie inside the collocation horizon.")
    initial_rate = result.state_rates[0]
    dae_problem = DifferentialAlgebraicProblem(
        system,
        result.decision.states[0],
        initial_state_rate=initial_rate,
        args=callback_args,
        input_policy=input_policy,
        initialization=DAEInitializationSpec.index_one(),
        problem_id=canonical_fingerprint(
            {
                "kind": "direct-collocation-replay-problem",
                "source": problem.problem_id,
                "result": result.result_id,
                "input_policy": input_policy.policy_id,
            }
        ),
    )
    solution = solve_dae(
        dae_problem,
        replay_grid,
        policy=policy.dae_policy,
    )
    view = TrajectoryOptimizationView(
        physical_grid.times,
        result.decision.states,
        result.decision.controls,
        case_shape=(),
        state_shape=problem.state_shape,
        control_shape=problem.control_shape,
    )
    collocated = view.evaluate_state(replay_grid.times)
    discrepancy = solution.states - collocated
    maximum = jnp.max(jnp.abs(discrepancy), initial=0.0)
    terminal = jnp.max(jnp.abs(discrepancy[-1]), initial=0.0)
    algebraic = jnp.max(solution.constraint_norm, initial=0.0)
    passed = (
        solution.successful
        & (maximum <= policy.node_state_tolerance)
        & (terminal <= policy.terminal_state_tolerance)
        & (algebraic <= policy.algebraic_constraint_tolerance)
    )
    if policy.failure_mode == "error" and not bool(np.asarray(passed)):
        raise ValueError("Controlled DAE replay failed its declared thresholds.")
    return DirectCollocationReplayEvidence(
        input_policy,
        solution,
        collocated,
        discrepancy,
        maximum,
        terminal,
        algebraic,
        passed,
        source_result_id=result.result_id,
        source_problem_id=problem.problem_id,
        source_plan_id=result.compilation.plan.plan_id,
        replay_id=canonical_fingerprint(
            {
                "kind": "direct-collocation-dae-replay",
                "result": result.result_id,
                "policy": policy.policy_id,
                "dae_problem": dae_problem.problem_id,
            }
        ),
    )


__all__ = [
    "DirectCollocationReplayEvidence",
    "DirectCollocationReplayFailureMode",
    "DirectCollocationReplayPolicy",
    "replay_direct_collocation",
]
