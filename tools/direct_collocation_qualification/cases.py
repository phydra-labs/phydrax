#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

import phydrax as phx

from .contracts import DirectCollocationQualificationCase


@dataclass(frozen=True)
class DirectCollocationQualificationSetup:
    case: DirectCollocationQualificationCase
    problem: phx.control.TrajectoryOptimizationProblem
    plan: phx.control.DirectCollocationPlan
    initial_states: Any
    initial_controls: Any
    bounds: phx.control.DirectCollocationBounds | None = None
    parameter_guess: Any = None
    duration_guess: Any = None
    reference_objective: float | None = None


def _mesh(case_id: str, intervals: int = 4):
    return phx.discretization.TemporalMesh.uniform(
        0.0,
        1.0,
        intervals,
        role="collocation",
        mesh_id=f"qualification:{case_id}:mesh",
    )


def _plan(case_id: str, *, variable_duration=False):
    return phx.control.DirectCollocationPlan(
        _mesh(case_id),
        method=phx.solver.ThetaMethod(0.5, endpoint=False),
        variable_duration=variable_duration,
        derivatives=phx.control.DirectCollocationDerivativePolicy(
            verify=True,
            num_verification_probes=2,
        ),
        audit=phx.control.DirectCollocationAuditPolicy(off_grid_points=2),
        plan_id=f"qualification:{case_id}:plan",
    )


def _fixed_integrator():
    case_id = "fixed-integrator"
    system = phx.dynamics.ContinuousSystem(
        lambda time, state, control, args: control,
        state_layout=phx.dynamics.StateLayout((1,)),
        input_layout=phx.dynamics.InputLayout((1,), roles="control"),
        system_id=f"qualification:{case_id}:system",
    )
    terminal = phx.control.BoundedTrajectoryConstraint(
        lambda trajectory, args: trajectory.final_state[0],
        lower=1.0,
        upper=1.0,
        constraint_id=f"{case_id}:terminal",
    )
    problem = phx.control.TrajectoryOptimizationProblem(
        system,
        initial_state=jnp.zeros(1),
        running_cost=lambda time, state, control, args: 0.5 * control[0] ** 2,
        trajectory_constraints=(terminal,),
        problem_id=f"qualification:{case_id}",
    )
    mesh = _mesh(case_id)
    return DirectCollocationQualificationSetup(
        DirectCollocationQualificationCase(
            case_id, "analytic", "analytic", True, False
        ),
        problem,
        _plan(case_id),
        mesh.nodes[:, None],
        jnp.ones((mesh.num_steps, 1)),
        reference_objective=0.5,
    )


def _variable_integrator():
    case_id = "variable-duration-integrator"
    system = phx.dynamics.ContinuousSystem(
        lambda time, state, control, args: control,
        state_layout=phx.dynamics.StateLayout((1,)),
        input_layout=phx.dynamics.InputLayout((1,), roles="control"),
        system_id=f"qualification:{case_id}:system",
    )
    terminal = phx.control.BoundedTrajectoryConstraint(
        lambda trajectory, context: trajectory.final_state[0],
        lower=1.0,
        upper=1.0,
        constraint_id=f"{case_id}:terminal",
    )
    problem = phx.control.TrajectoryOptimizationProblem(
        system,
        initial_state=jnp.zeros(1),
        running_cost=lambda time, state, control, context: control[0] ** 2,
        trajectory_cost=lambda trajectory, context: context.duration,
        trajectory_constraints=(terminal,),
        problem_id=f"qualification:{case_id}",
    )
    mesh = _mesh(case_id)
    return DirectCollocationQualificationSetup(
        DirectCollocationQualificationCase(
            case_id, "variable-duration", "analytic", True, False
        ),
        problem,
        _plan(case_id, variable_duration=True),
        mesh.nodes[:, None],
        jnp.ones((mesh.num_steps, 1)),
        bounds=phx.control.DirectCollocationBounds(duration=(0.25, 4.0)),
        duration_guess=1.0,
        reference_objective=2.0,
    )


def _controlled_dae():
    case_id = "controlled-semi-explicit-dae"
    input_layout = phx.dynamics.InputLayout((1,), roles="control")
    system = phx.dynamics.DifferentialAlgebraicSystem(
        lambda time, state, state_rate, control, args: jnp.asarray(
            (state_rate[0] - control[0], state[1] - state[0])
        ),
        state_shape=(2,),
        structure=phx.dynamics.DAEStructure(("differential", "algebraic")),
        input_layout=input_layout,
        system_id=f"qualification:{case_id}:system",
    )
    terminal = phx.control.BoundedTrajectoryConstraint(
        lambda trajectory, args: trajectory.final_state[0],
        lower=1.0,
        upper=1.0,
        constraint_id=f"{case_id}:terminal",
    )
    problem = phx.control.TrajectoryOptimizationProblem(
        system,
        initial_state=jnp.zeros(2),
        running_cost=lambda time, state, control, args: 0.5 * control[0] ** 2,
        trajectory_constraints=(terminal,),
        problem_id=f"qualification:{case_id}",
    )
    mesh = _mesh(case_id)
    states = jnp.stack((mesh.nodes, mesh.nodes), axis=-1)
    return DirectCollocationQualificationSetup(
        DirectCollocationQualificationCase(
            case_id, "controlled-dae", "analytic-and-replay", True, True
        ),
        problem,
        _plan(case_id),
        states,
        jnp.ones((mesh.num_steps, 1)),
        reference_objective=0.5,
    )


def _active_path_constraint():
    case_id = "active-path-inequality"
    system = phx.dynamics.ContinuousSystem(
        lambda time, state, control, args: control,
        state_layout=phx.dynamics.StateLayout((1,)),
        input_layout=phx.dynamics.InputLayout((1,), roles="control"),
        system_id=f"qualification:{case_id}:system",
    )
    path = phx.control.BoundedPathConstraint(
        lambda time, state, control, args: (
            control[0] - jnp.where(time < 0.25, 0.2, 1.0)
        ),
        upper=0.0,
        constraint_id=f"{case_id}:control-limit",
    )
    terminal = phx.control.BoundedTrajectoryConstraint(
        lambda trajectory, args: trajectory.final_state[0],
        lower=0.5,
        upper=0.5,
        constraint_id=f"{case_id}:terminal",
    )
    problem = phx.control.TrajectoryOptimizationProblem(
        system,
        initial_state=jnp.zeros(1),
        running_cost=lambda time, state, control, args: 0.5 * control[0] ** 2,
        path_constraints=(path,),
        trajectory_constraints=(terminal,),
        problem_id=f"qualification:{case_id}",
    )
    mesh = _mesh(case_id)
    return DirectCollocationQualificationSetup(
        DirectCollocationQualificationCase(
            case_id, "active-inequality", "analytic-kkt", True, False
        ),
        problem,
        _plan(case_id),
        jnp.asarray(((0.0,), (0.05,), (0.2,), (0.35,), (0.5,))),
        jnp.asarray(((0.2,), (0.6,), (0.6,), (0.6,))),
        reference_objective=0.14,
    )


def _shared_parameter():
    case_id = "shared-parameter-cases"
    space = phx.linalg.ArraySpace((1,), dtype=jnp.float64)
    system = phx.dynamics.ContinuousSystem(
        lambda time, state, control, context: context.parameters[0] * control,
        state_layout=phx.dynamics.StateLayout((1,)),
        input_layout=phx.dynamics.InputLayout((1,), roles="control"),
        system_id=f"qualification:{case_id}:system",
    )
    path = phx.control.BoundedPathConstraint(
        lambda time, state, control, context: control[0],
        lower=1.0,
        upper=1.0,
        constraint_id=f"{case_id}:unit-control",
    )
    terminal = phx.control.BoundedTrajectoryConstraint(
        lambda trajectory, context: trajectory.final_state[0, 0],
        lower=2.0,
        upper=2.0,
        constraint_id=f"{case_id}:terminal-first-case",
    )
    problem = phx.control.TrajectoryOptimizationProblem(
        system,
        initial_state=jnp.zeros((2, 1)),
        trajectory_cost=lambda trajectory, context: (
            (context.parameters[0] - 2.0) ** 2
            + (trajectory.final_state[1, 0] - 2.0) ** 2
        ),
        path_constraints=(path,),
        trajectory_constraints=(terminal,),
        parameter_space=space,
        problem_id=f"qualification:{case_id}",
    )
    mesh = _mesh(case_id)
    states = jnp.broadcast_to(
        2.0 * mesh.nodes[None, :, None],
        (2, mesh.num_nodes, 1),
    )
    controls = jnp.ones((2, mesh.num_steps, 1))
    return DirectCollocationQualificationSetup(
        DirectCollocationQualificationCase(
            case_id, "shared-parameter", "analytic", True, False
        ),
        problem,
        _plan(case_id),
        states,
        controls,
        parameter_guess=jnp.asarray((2.0,)),
        reference_objective=0.0,
    )


def _stiff_dae():
    case_id = "stiff-controlled-dae"
    system = phx.dynamics.DifferentialAlgebraicSystem(
        lambda time, state, state_rate, control, args: jnp.asarray(
            (state_rate[0] + 1000.0 * (state[0] - control[0]), state[1] - state[0])
        ),
        state_shape=(2,),
        structure=phx.dynamics.DAEStructure(("differential", "algebraic")),
        input_layout=phx.dynamics.InputLayout((1,), roles="control"),
        system_id=f"qualification:{case_id}:system",
    )
    path = phx.control.BoundedPathConstraint(
        lambda time, state, control, args: control[0],
        lower=1.0,
        upper=1.0,
        constraint_id=f"{case_id}:unit-control",
    )
    problem = phx.control.TrajectoryOptimizationProblem(
        system,
        initial_state=jnp.ones(2),
        running_cost=lambda time, state, control, args: 0.5 * control[0] ** 2,
        path_constraints=(path,),
        problem_id=f"qualification:{case_id}",
    )
    mesh = _mesh(case_id)
    return DirectCollocationQualificationSetup(
        DirectCollocationQualificationCase(
            case_id, "stiff-dae", "analytic-and-replay", True, True
        ),
        problem,
        _plan(case_id),
        jnp.ones((mesh.num_nodes, 2)),
        jnp.ones((mesh.num_steps, 1)),
        reference_objective=0.5,
    )


def _unstable_system():
    case_id = "open-loop-unstable"
    system = phx.dynamics.ContinuousSystem(
        lambda time, state, control, args: state + control,
        state_layout=phx.dynamics.StateLayout((1,)),
        input_layout=phx.dynamics.InputLayout((1,), roles="control"),
        system_id=f"qualification:{case_id}:system",
    )
    terminal = phx.control.BoundedTrajectoryConstraint(
        lambda trajectory, args: trajectory.final_state[0],
        lower=1.0,
        upper=1.0,
        constraint_id=f"{case_id}:terminal",
    )
    problem = phx.control.TrajectoryOptimizationProblem(
        system,
        initial_state=jnp.zeros(1),
        running_cost=lambda time, state, control, args: 0.5 * control[0] ** 2,
        trajectory_constraints=(terminal,),
        problem_id=f"qualification:{case_id}",
    )
    mesh = _mesh(case_id)
    return DirectCollocationQualificationSetup(
        DirectCollocationQualificationCase(
            case_id, "unstable", "refined-peer", True, False
        ),
        problem,
        _plan(case_id),
        mesh.nodes[:, None],
        (1.0 - mesh.nodes[:-1])[:, None],
    )


def _nonholonomic_constraint():
    case_id = "nonholonomic-constraint"
    system = phx.dynamics.ContinuousSystem(
        lambda time, state, control, args: control,
        state_layout=phx.dynamics.StateLayout((3,)),
        input_layout=phx.dynamics.InputLayout((3,), roles="control"),
        system_id=f"qualification:{case_id}:system",
    )
    path = phx.control.BoundedPathConstraint(
        lambda time, state, control, args: (
            control[0] * jnp.sin(state[2]) - control[1] * jnp.cos(state[2])
        ),
        lower=0.0,
        upper=0.0,
        constraint_id=f"{case_id}:lateral-velocity",
    )
    terminal = phx.control.BoundedTrajectoryConstraint(
        lambda trajectory, args: trajectory.final_state,
        lower=jnp.asarray((1.0, 0.0, 0.0)),
        upper=jnp.asarray((1.0, 0.0, 0.0)),
        constraint_id=f"{case_id}:terminal",
    )
    problem = phx.control.TrajectoryOptimizationProblem(
        system,
        initial_state=jnp.zeros(3),
        running_cost=lambda time, state, control, args: 0.5 * jnp.vdot(control, control),
        path_constraints=(path,),
        trajectory_constraints=(terminal,),
        problem_id=f"qualification:{case_id}",
    )
    mesh = _mesh(case_id)
    states = jnp.stack((mesh.nodes, jnp.zeros_like(mesh.nodes), jnp.zeros_like(mesh.nodes)), axis=-1)
    controls = jnp.broadcast_to(jnp.asarray((1.0, 0.0, 0.0)), (mesh.num_steps, 3))
    return DirectCollocationQualificationSetup(
        DirectCollocationQualificationCase(
            case_id, "nonholonomic", "analytic", True, False
        ),
        problem,
        _plan(case_id),
        states,
        controls,
        reference_objective=0.5,
    )


def qualification_setups() -> tuple[DirectCollocationQualificationSetup, ...]:
    return (
        _fixed_integrator(),
        _variable_integrator(),
        _controlled_dae(),
        _active_path_constraint(),
        _shared_parameter(),
        _stiff_dae(),
        _unstable_system(),
        _nonholonomic_constraint(),
    )


__all__ = ["DirectCollocationQualificationSetup", "qualification_setups"]
