#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._frozendict import frozendict
from .._strict import AbstractAttribute, StrictModule
from ..metrix import AbstractStateGeometry, EuclideanStateGeometry
from ..stochastic import (
    AbstractRoughControl,
    FractionalGaussianRealization,
    StochasticTrajectory,
)
from ..stochastic._trajectory import _TrajectoryRecord
from ._solution_validation import validate_solution_arrays


RoughVectorFields: TypeAlias = Callable[[Array, Array, Any], ArrayLike]
RoughDrift: TypeAlias = Callable[[Array, Array, Any], ArrayLike]


class _ZeroRoughDrift(eqx.Module):
    def __call__(self, time, state, args):
        del time, args
        return jnp.zeros_like(state)


class RoughDifferentialProblem(StrictModule):
    """Geometric rough differential equation ``dY = V₀ dt + Vᵢ dXⁱ``."""

    vector_fields: RoughVectorFields
    drift: RoughDrift
    initial_state: Array
    args: Any
    geometry: AbstractStateGeometry
    state_shape: tuple[int, ...] = eqx.field(static=True)
    driver_dimension: int = eqx.field(static=True)
    has_drift: bool = eqx.field(static=True)
    time_dependent: bool = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        vector_fields: RoughVectorFields,
        initial_state: ArrayLike,
        /,
        *,
        driver_dimension: int,
        drift: RoughDrift | None = None,
        args: Any = None,
        geometry: AbstractStateGeometry | None = None,
        time_dependent: bool = False,
        problem_id: str = "rough-differential-problem",
    ):
        if not callable(vector_fields):
            raise TypeError("vector_fields must be callable.")
        dimension = int(driver_dimension)
        if dimension <= 0:
            raise ValueError("driver_dimension must be positive.")
        if drift is None:
            resolved_drift: RoughDrift = _ZeroRoughDrift()
        else:
            if not callable(drift):
                raise TypeError("drift must be callable or None.")
            resolved_drift = drift
        state = jnp.asarray(initial_state)
        state_shape = tuple(int(size) for size in state.shape)
        if not state_shape or any(size <= 0 for size in state_shape):
            raise ValueError("initial_state must have a non-empty positive shape.")
        resolved_geometry = EuclideanStateGeometry() if geometry is None else geometry
        if not isinstance(resolved_geometry, AbstractStateGeometry):
            raise TypeError("geometry must be an AbstractStateGeometry or None.")
        membership = jnp.asarray(resolved_geometry.contains(state), dtype=bool)
        if membership.shape != () or not bool(membership):
            raise ValueError("initial_state must belong to the state geometry.")
        fields = jnp.asarray(vector_fields(jnp.asarray(0.0), state, args))
        expected_fields = state_shape + (dimension,)
        if fields.shape != expected_fields:
            raise ValueError(
                f"vector_fields must return shape {expected_fields}; got {fields.shape}."
            )
        drift_value = jnp.asarray(resolved_drift(jnp.asarray(0.0), state, args))
        if drift_value.shape != state_shape:
            raise ValueError("drift must preserve initial_state shape.")
        identifier = str(problem_id)
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        self.vector_fields = vector_fields
        self.drift = resolved_drift
        self.initial_state = state
        self.args = args
        self.geometry = resolved_geometry
        self.state_shape = state_shape
        self.driver_dimension = dimension
        self.has_drift = drift is not None
        self.time_dependent = bool(time_dependent)
        self.problem_id = identifier


class AbstractRoughSolver(StrictModule):
    """Algorithm consuming one finite-depth geometric rough control."""

    solver_name: AbstractAttribute[str]
    solver_id: AbstractAttribute[str]
    required_depth: AbstractAttribute[int]

    @abstractmethod
    def integrate(
        self,
        problem: RoughDifferentialProblem,
        control: AbstractRoughControl,
        /,
    ) -> tuple[Array, Array, Mapping[str, Array]]:
        """Return all node states, interval statuses, and interval statistics."""
        raise NotImplementedError


def _fractional_hurst(control: AbstractRoughControl, /) -> float | None:
    realization = control.realization
    if isinstance(realization, FractionalGaussianRealization):
        return float(realization.process.hurst)
    return None


def _validate_classical_control(
    problem: RoughDifferentialProblem,
    control: AbstractRoughControl,
    /,
    *,
    required_depth: int,
    minimum_hurst: float,
    solver_name: str,
) -> None:
    if control.dimension != problem.driver_dimension:
        raise ValueError("Problem and rough control driver dimensions must match.")
    if control.depth < required_depth:
        raise ValueError(
            f"{solver_name} requires control depth at least {required_depth}; "
            f"got {control.depth}."
        )
    hurst = _fractional_hurst(control)
    if hurst is not None and hurst <= minimum_hurst:
        threshold = "1/2" if minimum_hurst == 0.5 else "1/3"
        raise ValueError(
            f"{solver_name} requires fractional Gaussian Hurst > {threshold}."
        )


def _davie_correction(
    problem: RoughDifferentialProblem,
    time: Array,
    state: Array,
    fields: Array,
    second_level: Array,
    /,
) -> Array:
    directions = jnp.moveaxis(fields, -1, 0)

    def differentiate(direction):
        return jax.jvp(
            lambda value: jnp.asarray(problem.vector_fields(time, value, problem.args)),
            (state,),
            (direction,),
        )[1]

    derivatives = jax.vmap(differentiate)(directions)
    flattened = derivatives.reshape(
        (problem.driver_dimension, int(state.size), problem.driver_dimension)
    )
    return ein.contract("isj,ij->s", flattened, second_level).reshape(problem.state_shape)


def _classical_integrate(
    problem: RoughDifferentialProblem,
    control: AbstractRoughControl,
    /,
    *,
    davie: bool,
) -> tuple[Array, Array, Mapping[str, Array]]:
    first_level = control.levels[0]
    second_level = control.levels[1] if davie else None
    steps = jnp.diff(control.times)

    def one_path(first, second):
        def advance(state, item):
            time, step, first_increment, second_increment = item
            drift = jnp.asarray(problem.drift(time, state, problem.args))
            fields = jnp.asarray(problem.vector_fields(time, state, problem.args))
            first_update = jnp.tensordot(fields, first_increment, axes=((-1,), (0,)))
            second_update = (
                _davie_correction(problem, time, state, fields, second_increment)
                if davie
                else jnp.zeros_like(state)
            )
            ambient_update = step * drift + first_update + second_update
            tangent = problem.geometry.project_tangent(state, ambient_update)
            local_update = problem.geometry.to_local(state, tangent)
            next_state = problem.geometry.retract(state, local_update)
            return next_state, next_state

        if second is None:
            second = jnp.zeros((control.num_steps, 0), dtype=first.dtype)
        _, stepped = jax.lax.scan(
            advance,
            problem.initial_state,
            (control.times[:-1], steps, first, second),
        )
        return jnp.concatenate((problem.initial_state[None, ...], stepped), axis=0)

    if control.sample_shape:
        path_count = int(np.prod(control.sample_shape))
        first = first_level.reshape((path_count, control.num_steps, control.dimension))
        if second_level is None:
            second = jnp.zeros((path_count, control.num_steps, 0), dtype=first.dtype)
        else:
            second = second_level.reshape(
                (path_count, control.num_steps, control.dimension, control.dimension)
            )
        states = jax.vmap(one_path)(first, second).reshape(
            control.sample_shape + (control.num_steps + 1,) + problem.state_shape
        )
    else:
        states = one_path(first_level, second_level)
    interval_shape = control.sample_shape + (control.num_steps,)
    statuses = jnp.zeros(interval_shape, dtype=jnp.int32)
    statistics = {
        "num_steps": jnp.ones(interval_shape, dtype=jnp.int32),
        "num_accepted_steps": jnp.ones(interval_shape, dtype=jnp.int32),
        "num_rejected_steps": jnp.zeros(interval_shape, dtype=jnp.int32),
    }
    return states, statuses, statistics


class RoughEuler(AbstractRoughSolver):
    """First-level Young/Euler rough solver."""

    solver_name: str = eqx.field(static=True)
    solver_id: str = eqx.field(static=True)
    required_depth: int = eqx.field(static=True)

    def __init__(self):
        self.solver_name = "RoughEuler"
        self.solver_id = "rough-solver:rough-euler:v1"
        self.required_depth = 1

    def integrate(
        self,
        problem: RoughDifferentialProblem,
        control: AbstractRoughControl,
        /,
    ) -> tuple[Array, Array, Mapping[str, Array]]:
        _validate_classical_control(
            problem,
            control,
            required_depth=1,
            minimum_hurst=0.5,
            solver_name=self.solver_name,
        )
        return _classical_integrate(problem, control, davie=False)


class Davie(AbstractRoughSolver):
    """Depth-2 Davie expansion with JVP-computed word differentials."""

    solver_name: str = eqx.field(static=True)
    solver_id: str = eqx.field(static=True)
    required_depth: int = eqx.field(static=True)

    def __init__(self):
        self.solver_name = "Davie"
        self.solver_id = "rough-solver:davie:v1"
        self.required_depth = 2

    def integrate(
        self,
        problem: RoughDifferentialProblem,
        control: AbstractRoughControl,
        /,
    ) -> tuple[Array, Array, Mapping[str, Array]]:
        _validate_classical_control(
            problem,
            control,
            required_depth=2,
            minimum_hurst=1.0 / 3.0,
            solver_name=self.solver_name,
        )
        return _classical_integrate(problem, control, davie=True)


class RoughDifferentialSolution(StrictModule):
    """RDE states, interval statuses, and statistics for one rough control."""

    times: Array
    states: Array
    valid: Array
    statuses: Array
    control: AbstractRoughControl
    solver: AbstractRoughSolver
    metadata: frozendict[str, Any]
    statistics: frozendict[str, Array]
    sample_shape: tuple[int, ...] = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    state_geometry_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        times: ArrayLike,
        states: ArrayLike,
        valid: ArrayLike,
        statuses: ArrayLike,
        control: AbstractRoughControl,
        solver: AbstractRoughSolver,
        state_shape: Sequence[int],
        state_geometry_id: str,
        statistics: Mapping[str, ArrayLike],
        metadata: Mapping[str, Any] | None = None,
    ):
        if not isinstance(control, AbstractRoughControl):
            raise TypeError("control must be an AbstractRoughControl.")
        if not isinstance(solver, AbstractRoughSolver):
            raise TypeError("solver must be an AbstractRoughSolver.")
        arrays = validate_solution_arrays(
            times,
            states,
            valid,
            sample_shape=control.sample_shape,
            state_shape=state_shape,
            time_layout="shared",
            owner="RoughDifferentialSolution",
        )
        time_values = arrays.times
        state_values = arrays.states
        valid_values = arrays.valid
        shape = arrays.state_shape
        status_values = jnp.asarray(statuses, dtype=jnp.int32)
        expected_statuses = control.sample_shape + (control.num_steps,)
        if status_values.shape != expected_statuses:
            raise ValueError("RDE interval statuses do not align with the control.")
        resolved_statistics = {
            str(name): jnp.asarray(value) for name, value in statistics.items()
        }
        if any(
            value.shape != expected_statuses for value in resolved_statistics.values()
        ):
            raise ValueError("RDE statistics must align with control intervals.")
        geometry_id = str(state_geometry_id)
        if not geometry_id:
            raise ValueError("state_geometry_id must be non-empty.")
        self.times = time_values
        self.states = state_values
        self.valid = valid_values
        self.statuses = status_values
        self.control = control
        self.solver = solver
        self.metadata = frozendict({} if metadata is None else metadata)
        self.statistics = frozendict(resolved_statistics)
        self.sample_shape = control.sample_shape
        self.state_shape = shape
        self.state_geometry_id = geometry_id

    @property
    def solver_name(self) -> str:
        return self.solver.solver_name

    @property
    def solver_id(self) -> str:
        return self.solver.solver_id

    @property
    def successful(self) -> Array:
        return jnp.all(self.valid, axis=-1) & jnp.all(self.statuses == 0, axis=-1)

    def to_stochastic_trajectory(
        self,
        /,
        *,
        realization_axes: Sequence[str] | None = None,
        state_axes: Sequence[str] | None = None,
    ) -> StochasticTrajectory:
        axes = (
            tuple(f"path_{index}" for index in range(len(self.sample_shape)))
            if realization_axes is None
            else tuple(realization_axes)
        )
        resolved_state_axes = (
            tuple(f"state_{index}" for index in range(len(self.state_shape)))
            if state_axes is None
            else tuple(state_axes)
        )
        record = _TrajectoryRecord(
            self.times,
            self.states,
            state_shape=self.state_shape,
            realization_shape=self.sample_shape,
            valid=self.valid,
            realizations=(self.control.realization,),
            approximation_id=self.control.control_id,
            solver_name=self.solver_name,
            solver_id=self.solver_id,
            state_geometry_id=self.state_geometry_id,
            uncertainty_source="process",
            metadata={
                **dict(self.metadata),
                "rough_control_id": self.control.control_id,
            },
        )
        return record.to_stochastic_trajectory(
            realization_axes=axes,
            state_axes=resolved_state_axes,
        )


def _save_indices(
    control: AbstractRoughControl, save_times: ArrayLike, /
) -> tuple[Array, Array]:
    saved = jnp.asarray(save_times, dtype=float)
    if saved.ndim != 1 or int(saved.size) <= 0:
        raise ValueError("save_times must be a non-empty rank-1 array.")
    saved = eqx.error_if(
        saved,
        jnp.any(~jnp.isfinite(saved)) | jnp.any(jnp.diff(saved) <= 0.0),
        "save_times must be finite and strictly increasing.",
    )
    indices = jnp.searchsorted(control.times, saved).astype(jnp.int32)
    safe_indices = jnp.minimum(indices, control.num_steps)
    tolerance = 100.0 * jnp.finfo(saved.dtype).eps
    indices = eqx.error_if(
        indices,
        jnp.any(indices > control.num_steps)
        | jnp.any(jnp.abs(control.times[safe_indices] - saved) > tolerance),
        "save_times must be nodes of the rough control partition.",
    )
    return indices, saved


def solve_rough_differential(
    problem: RoughDifferentialProblem,
    control: AbstractRoughControl,
    /,
    *,
    save_times: ArrayLike | None = None,
    solver: AbstractRoughSolver = Davie(),
) -> RoughDifferentialSolution:
    """Integrate an RDE using an explicit rough-solver object."""
    if not isinstance(problem, RoughDifferentialProblem):
        raise TypeError("problem must be a RoughDifferentialProblem.")
    if not isinstance(control, AbstractRoughControl):
        raise TypeError("control must be an AbstractRoughControl.")
    if not isinstance(solver, AbstractRoughSolver):
        raise TypeError("solver must be an AbstractRoughSolver.")
    selected_times = control.times if save_times is None else save_times
    save_indices, saved = _save_indices(control, selected_times)
    all_states, statuses, statistics = solver.integrate(problem, control)
    states = jnp.take(all_states, save_indices, axis=len(control.sample_shape))
    state_axes = tuple(range(len(control.sample_shape) + 1, states.ndim))
    valid = jnp.all(jnp.isfinite(states), axis=state_axes)
    return RoughDifferentialSolution(
        times=saved,
        states=states,
        valid=valid,
        statuses=statuses,
        control=control,
        solver=solver,
        state_shape=problem.state_shape,
        state_geometry_id=problem.geometry.geometry_id,
        statistics=statistics,
        metadata={
            "problem_id": problem.problem_id,
            "num_intervals": control.num_steps,
            "driver_dimension": control.dimension,
            "control_depth": control.depth,
            "state_geometry_id": problem.geometry.geometry_id,
        },
    )


__all__ = [
    "AbstractRoughSolver",
    "Davie",
    "RoughDifferentialProblem",
    "RoughDifferentialSolution",
    "RoughDrift",
    "RoughEuler",
    "RoughVectorFields",
    "solve_rough_differential",
]
