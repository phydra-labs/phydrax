#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from .._evolution import EVOLUTION_SUCCESS, EvolutionTrajectory
from .._layout import InputLayout, StateLayout
from .._trajectory import CaseAxisRole, TrajectoryData


if TYPE_CHECKING:
    from ...control import ControlTrajectory
    from ...solver import (
        ControlledDifferentialSolution,
        DifferentialAlgebraicSolution,
        DifferentialSolution,
        MemoryEquationSolution,
        RoughDifferentialSolution,
    )
    from ...stochastic import StochasticTrajectory


def trajectory_data_from_evolution(
    trajectory: EvolutionTrajectory,
    /,
    *,
    source_id: str | None = None,
) -> TrajectoryData:
    """Convert one canonical evolution trajectory without changing its masks."""
    if not isinstance(trajectory, EvolutionTrajectory):
        raise TypeError("trajectory must be an EvolutionTrajectory.")
    transition_valid = (
        trajectory.valid[:-1]
        & trajectory.valid[1:]
        & (trajectory.status == EVOLUTION_SUCCESS)
    )
    return TrajectoryData(
        trajectory.grid.coordinates,
        trajectory.states,
        state_layout=trajectory.state_layout,
        sample_valid=trajectory.valid,
        transition_valid=transition_valid,
        coordinate_id=trajectory.grid.grid_id,
        source_id=(
            f"evolution:{trajectory.evolution_id}" if source_id is None else source_id
        ),
    )


def trajectory_data_from_control(
    trajectory: ControlTrajectory,
    /,
    *,
    state_layout: StateLayout | None = None,
    input_layout: InputLayout | None = None,
    case_axes: tuple[str, ...] | None = None,
    case_axis_roles: tuple[CaseAxisRole, ...] | None = None,
    source_id: str | None = None,
) -> TrajectoryData:
    """Convert a controlled rollout with source-aligned transition inputs."""
    from ...control import ControlTrajectory

    if not isinstance(trajectory, ControlTrajectory):
        raise TypeError("trajectory must be a ControlTrajectory.")
    resolved_state = (
        StateLayout(trajectory.state_shape) if state_layout is None else state_layout
    )
    resolved_input = (
        InputLayout(trajectory.control_shape, roles="control")
        if input_layout is None
        else input_layout
    )
    if resolved_state.shape != trajectory.state_shape:
        raise ValueError("state_layout shape does not match the control trajectory.")
    if resolved_input.shape != trajectory.control_shape:
        raise ValueError("input_layout shape does not match the control trajectory.")
    coordinates = jnp.broadcast_to(
        trajectory.time_grid.times,
        trajectory.case_shape + (trajectory.time_grid.num_times,),
    )
    transitions = trajectory.valid[..., :-1] & trajectory.valid[..., 1:]
    return TrajectoryData(
        coordinates,
        trajectory.states,
        state_layout=resolved_state,
        sample_valid=trajectory.valid,
        transition_valid=transitions,
        inputs=trajectory.controls,
        input_layout=resolved_input,
        input_valid=transitions,
        case_axes=case_axes,
        case_axis_roles=case_axis_roles,
        coordinate_id=trajectory.time_grid.time_id,
        source_id=(
            f"control:{trajectory.problem_id}:{trajectory.dynamics_id}"
            if source_id is None
            else source_id
        ),
    )


def trajectory_data_from_differential_solution(
    solution: (
        DifferentialAlgebraicSolution
        | DifferentialSolution
        | MemoryEquationSolution
        | RoughDifferentialSolution
        | ControlledDifferentialSolution
    ),
    /,
    *,
    state_layout: StateLayout | None = None,
    sample_axes: tuple[str, ...] | None = None,
    sample_axis_roles: tuple[CaseAxisRole, ...] | None = None,
    coordinate_id: str | None = None,
    source_id: str | None = None,
) -> TrajectoryData:
    """Convert differential solver output, retaining native DAE state rates."""
    from ...solver import (
        ControlledDifferentialSolution,
        DifferentialAlgebraicSolution,
        DifferentialSolution,
        MemoryEquationSolution,
        RoughDifferentialSolution,
    )

    supported = (
        DifferentialAlgebraicSolution,
        DifferentialSolution,
        MemoryEquationSolution,
        RoughDifferentialSolution,
        ControlledDifferentialSolution,
    )
    if not isinstance(solution, supported):
        raise TypeError(
            "solution must be DAE, differential, delay/memory, rough, or "
            "controlled differential solver output."
        )
    derivatives = None
    derivative_valid = None
    if isinstance(solution, DifferentialAlgebraicSolution):
        states = solution.states
        times = solution.times
        valid = solution.valid
        sample_shape = solution.sample_shape
        state_shape = solution.state_shape
        geometry_id = None
        derivatives = solution.state_rates
        state_rank = len(solution.state_shape)
        state_axes = tuple(
            range(solution.rate_valid.ndim - state_rank, solution.rate_valid.ndim)
        )
        derivative_valid = jnp.all(solution.rate_valid, axis=state_axes)
        default_coordinate_id = solution.time_id
        default_source_id = (
            f"dae:{solution.problem_id}:{solution.integration_method}:{solution.plan_id}"
        )
    elif isinstance(solution, ControlledDifferentialSolution):
        base = solution.differential_solution
        states = base.states
        times = base.times
        valid = base.valid
        sample_shape = base.sample_shape
        state_shape = tuple(base.states.shape[len(sample_shape) + 1 :])
        geometry_id = base.state_geometry_id
        default_source_id = (
            f"controlled-differential:{solution.problem_id}:{solution.path_id}:"
            f"{base.solver_id}:{base.resolved_method}"
        )
        default_coordinate_id = "time"
    elif isinstance(solution, DifferentialSolution):
        states = solution.states
        times = solution.times
        valid = solution.valid
        sample_shape = solution.sample_shape
        state_shape = tuple(solution.states.shape[len(sample_shape) + 1 :])
        geometry_id = solution.state_geometry_id
        default_source_id = (
            f"differential:{solution.solver_id}:{solution.resolved_method}"
        )
        default_coordinate_id = "time"
    elif isinstance(solution, MemoryEquationSolution):
        states = solution.states
        times = solution.times
        valid = solution.valid
        sample_shape = solution.sample_shape
        state_shape = solution.state_shape
        geometry_id = None
        default_source_id = f"memory:{solution.solver_id}:{solution.resolved_method}"
        default_coordinate_id = "time"
    else:
        states = solution.states
        times = solution.times
        valid = solution.valid
        sample_shape = solution.sample_shape
        state_shape = solution.state_shape
        geometry_id = solution.state_geometry_id
        default_source_id = f"rough:{solution.solver.solver_name}"
        default_coordinate_id = "time"
    default_layout = StateLayout(state_shape)
    if state_layout is None:
        if geometry_id is not None and geometry_id != default_layout.geometry.geometry_id:
            raise ValueError(
                "A non-Euclidean solver result requires its original state_layout."
            )
        resolved_state = default_layout
    else:
        resolved_state = state_layout
    if resolved_state.shape != state_shape:
        raise ValueError("state_layout shape does not match the solver solution.")
    if geometry_id is not None and resolved_state.geometry.geometry_id != geometry_id:
        raise ValueError("state_layout geometry does not match the solver solution.")
    axes = (
        tuple(f"sample_{index}" for index in range(len(sample_shape)))
        if sample_axes is None
        else sample_axes
    )
    roles = (
        ("realization",) * len(sample_shape)
        if sample_axis_roles is None
        else sample_axis_roles
    )
    coordinate_values = jnp.asarray(times)
    if coordinate_values.ndim == 1 and sample_shape:
        coordinate_values = jnp.broadcast_to(
            coordinate_values,
            sample_shape + (int(coordinate_values.size),),
        )
    transitions = valid[..., :-1] & valid[..., 1:]
    return TrajectoryData(
        coordinate_values,
        states,
        state_layout=resolved_state,
        sample_valid=valid,
        transition_valid=transitions,
        derivatives=derivatives,
        derivative_valid=derivative_valid,
        case_axes=axes,
        case_axis_roles=roles,
        coordinate_id=(default_coordinate_id if coordinate_id is None else coordinate_id),
        source_id=default_source_id if source_id is None else source_id,
    )


def trajectory_data_from_stochastic(
    trajectory: StochasticTrajectory,
    /,
    *,
    state_layout: StateLayout | None = None,
    source_id: str | None = None,
) -> TrajectoryData:
    """Convert stochastic trajectories while retaining realization axes."""
    from ...stochastic import StochasticTrajectory

    if not isinstance(trajectory, StochasticTrajectory):
        raise TypeError("trajectory must be a StochasticTrajectory.")
    resolved_state = (
        StateLayout(trajectory.state_shape, axes=trajectory.state_axes)
        if state_layout is None
        else state_layout
    )
    if resolved_state.shape != trajectory.state_shape:
        raise ValueError("state_layout shape does not match the stochastic trajectory.")
    leading_shape = trajectory.case_shape + trajectory.realization_shape
    axes = trajectory.case_axes + trajectory.realization_axes
    roles: tuple[CaseAxisRole, ...] = ("case",) * len(trajectory.case_shape) + (
        "realization",
    ) * len(trajectory.realization_shape)
    transitions = trajectory.valid[..., :-1] & trajectory.valid[..., 1:]
    return TrajectoryData(
        trajectory.times.reshape(leading_shape + (trajectory.times.shape[-1],)),
        trajectory.states,
        state_layout=resolved_state,
        sample_valid=trajectory.valid,
        transition_valid=transitions,
        case_axes=axes,
        case_axis_roles=roles,
        coordinate_id=trajectory.time_axis,
        source_id=("stochastic-trajectory" if source_id is None else source_id),
    )


__all__ = [
    "trajectory_data_from_control",
    "trajectory_data_from_differential_solution",
    "trajectory_data_from_evolution",
    "trajectory_data_from_stochastic",
]
