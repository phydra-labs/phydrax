#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._frozendict import frozendict
from .._strict import StrictModule
from ..stochastic import (
    FractionalGaussianRealization,
    GeometricRoughPath,
    StochasticTrajectory,
)


RoughVectorFields: TypeAlias = Callable[[Array, Array, Any], ArrayLike]
RoughDrift: TypeAlias = Callable[[Array, Array, Any], ArrayLike]
RoughDifferentialScheme: TypeAlias = Literal["euler", "davie"]


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
    state_shape: tuple[int, ...] = eqx.field(static=True)
    driver_dimension: int = eqx.field(static=True)
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
        self.state_shape = state_shape
        self.driver_dimension = dimension
        self.problem_id = identifier


class RoughDifferentialSolution(StrictModule):
    """RDE states aligned with one explicit lifted driver partition."""

    times: Array
    states: Array
    valid: Array
    rough_path: GeometricRoughPath
    metadata: frozendict[str, Any]
    sample_shape: tuple[int, ...] = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    scheme: RoughDifferentialScheme = eqx.field(static=True)
    solver_name: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        times: ArrayLike,
        states: ArrayLike,
        valid: ArrayLike,
        rough_path: GeometricRoughPath,
        state_shape: Sequence[int],
        scheme: RoughDifferentialScheme,
        metadata: Mapping[str, Any] | None = None,
    ):
        if not isinstance(rough_path, GeometricRoughPath):
            raise TypeError("rough_path must be a GeometricRoughPath.")
        time_values = jnp.asarray(times, dtype=float)
        state_values = jnp.asarray(states)
        valid_values = jnp.asarray(valid, dtype=bool)
        shape = tuple(int(size) for size in state_shape)
        expected_states = rough_path.sample_shape + (int(time_values.size),) + shape
        expected_valid = rough_path.sample_shape + (int(time_values.size),)
        if time_values.ndim != 1 or int(time_values.size) <= 0:
            raise ValueError("times must be a non-empty rank-1 array.")
        if state_values.shape != expected_states or valid_values.shape != expected_valid:
            raise ValueError("RDE states and validity do not align with declared axes.")
        if scheme not in ("euler", "davie"):
            raise ValueError("Unknown rough differential scheme.")
        self.times = time_values
        self.states = state_values
        self.valid = valid_values
        self.rough_path = rough_path
        self.metadata = frozendict({} if metadata is None else metadata)
        self.sample_shape = rough_path.sample_shape
        self.state_shape = shape
        self.scheme = scheme
        self.solver_name = "RoughEuler" if scheme == "euler" else "RoughDavie"

    @property
    def successful(self) -> Array:
        return jnp.all(self.valid, axis=-1)

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
        return StochasticTrajectory(
            self.times,
            self.states,
            valid=self.valid,
            realization_axes=axes,
            realization_shape=self.sample_shape,
            state_axes=resolved_state_axes,
            realizations=(self.rough_path.realization,),
            approximation_id=self.rough_path.rough_path_id,
            metadata={
                **dict(self.metadata),
                "solver_name": self.solver_name,
                "rough_path_id": self.rough_path.rough_path_id,
                "uncertainty_source": "process",
            },
        )


def _save_indices(
    rough_path: GeometricRoughPath, save_times: ArrayLike, /
) -> tuple[Array, Array]:
    saved = np.asarray(save_times, dtype=float)
    nodes = np.asarray(jax.device_get(rough_path.times))
    if saved.ndim != 1 or saved.size <= 0:
        raise ValueError("save_times must be a non-empty rank-1 array.")
    if np.any(~np.isfinite(saved)) or np.any(np.diff(saved) <= 0.0):
        raise ValueError("save_times must be finite and strictly increasing.")
    indices = np.searchsorted(nodes, saved)
    if np.any(indices >= nodes.size) or not np.allclose(
        nodes[np.minimum(indices, nodes.size - 1)],
        saved,
        rtol=0.0,
        atol=100.0 * np.finfo(float).eps,
    ):
        raise ValueError("save_times must be nodes of the rough path partition.")
    return jnp.asarray(indices, dtype=jnp.int32), jnp.asarray(saved, dtype=float)


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
    return jnp.einsum("isj,ij->s", flattened, second_level).reshape(problem.state_shape)


def solve_rough_differential(
    problem: RoughDifferentialProblem,
    rough_path: GeometricRoughPath,
    /,
    *,
    save_times: ArrayLike | None = None,
    scheme: RoughDifferentialScheme = "davie",
) -> RoughDifferentialSolution:
    """Integrate a geometric step-2 RDE with Euler or Davie's expansion."""
    if not isinstance(problem, RoughDifferentialProblem):
        raise TypeError("problem must be a RoughDifferentialProblem.")
    if not isinstance(rough_path, GeometricRoughPath):
        raise TypeError("rough_path must be a GeometricRoughPath.")
    if problem.driver_dimension != rough_path.dimension:
        raise ValueError("Problem and rough path driver dimensions must match.")
    if scheme not in ("euler", "davie"):
        raise ValueError("scheme must be 'euler' or 'davie'.")
    realization = rough_path.realization
    if isinstance(realization, FractionalGaussianRealization):
        hurst = realization.process.hurst
        if scheme == "davie" and hurst <= 1.0 / 3.0:
            raise ValueError(
                "A step-2 Davie scheme requires fractional Gaussian Hurst > 1/3."
            )
        if scheme == "euler" and hurst <= 0.5:
            raise ValueError(
                "Rough Euler is a Young scheme and requires fractional Gaussian "
                "Hurst > 1/2."
            )
    selected_times = rough_path.times if save_times is None else save_times
    save_indices, saved = _save_indices(rough_path, selected_times)
    steps = jnp.diff(rough_path.times)

    def one_path(first_level, second_level):
        def advance(state, item):
            time, step, first, second = item
            drift = jnp.asarray(problem.drift(time, state, problem.args))
            if drift.shape != problem.state_shape:
                raise ValueError("drift must preserve the declared state shape.")
            fields = jnp.asarray(problem.vector_fields(time, state, problem.args))
            expected = problem.state_shape + (problem.driver_dimension,)
            if fields.shape != expected:
                raise ValueError(
                    f"vector_fields must return shape {expected}; got {fields.shape}."
                )
            first_update = jnp.tensordot(fields, first, axes=((-1,), (0,)))
            second_update = (
                _davie_correction(problem, time, state, fields, second)
                if scheme == "davie"
                else jnp.zeros_like(state)
            )
            next_state = state + step * drift + first_update + second_update
            return next_state, next_state

        _, stepped = jax.lax.scan(
            advance,
            problem.initial_state,
            (
                rough_path.times[:-1],
                steps,
                first_level,
                second_level,
            ),
        )
        all_states = jnp.concatenate(
            (problem.initial_state[None, ...], stepped),
            axis=0,
        )
        return all_states[save_indices]

    if rough_path.sample_shape:
        path_count = int(np.prod(rough_path.sample_shape))
        first = rough_path.first_level.reshape(
            (path_count, rough_path.num_steps, rough_path.dimension)
        )
        second = rough_path.second_level.reshape(
            (
                path_count,
                rough_path.num_steps,
                rough_path.dimension,
                rough_path.dimension,
            )
        )
        states = jax.vmap(one_path)(first, second).reshape(
            rough_path.sample_shape + (int(saved.size),) + problem.state_shape
        )
    else:
        states = one_path(rough_path.first_level, rough_path.second_level)
    state_axes = tuple(range(len(rough_path.sample_shape) + 1, states.ndim))
    valid = jnp.all(jnp.isfinite(states), axis=state_axes)
    return RoughDifferentialSolution(
        times=saved,
        states=states,
        valid=valid,
        rough_path=rough_path,
        state_shape=problem.state_shape,
        scheme=scheme,
        metadata={
            "problem_id": problem.problem_id,
            "num_steps": rough_path.num_steps,
            "driver_dimension": rough_path.dimension,
        },
    )


__all__ = [
    "RoughDifferentialProblem",
    "RoughDifferentialScheme",
    "RoughDifferentialSolution",
    "RoughDrift",
    "RoughVectorFields",
    "solve_rough_differential",
]
