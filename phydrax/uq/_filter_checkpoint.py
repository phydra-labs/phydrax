#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from math import prod
from os import PathLike
from pathlib import Path
from typing import Literal, TypeAlias

import jax.numpy as jnp
import jax.random as jr

from ..stochastic._state_space import StateSpaceProblem
from ._checkpoint import (
    array_tree_fingerprint,
    read_checkpoint_archive,
    write_checkpoint_archive,
)
from ._ensemble_filter import EnsembleFilterState
from ._kalman import KalmanFilterState
from ._particle import (
    ParticleFilterState,
    read_particle_filter_checkpoint,
    ResamplingMethod,
    ResamplingPolicy,
    write_particle_filter_checkpoint,
)


FilterCheckpointAlgorithm: TypeAlias = Literal["kalman", "particle", "ensemble"]
FilterState: TypeAlias = KalmanFilterState | ParticleFilterState | EnsembleFilterState


def _base_compatibility(problem: StateSpaceProblem, /) -> dict[str, object]:
    return {
        "problem_id": problem.problem_id,
        "model_id": problem.model.model_id,
        "sequence_id": problem.observations.sequence_id,
        "state_shape": list(problem.model.state_shape),
        "observation_shape": list(problem.model.observation_shape),
        "case_shape": list(problem.observations.case_shape),
        "case_ids": list(problem.observations.case_ids),
        "problem_arrays": array_tree_fingerprint(problem),
    }


def _validate_problem(problem: StateSpaceProblem, /) -> None:
    if not isinstance(problem, StateSpaceProblem):
        raise TypeError("problem must be a StateSpaceProblem.")


def _validate_step_index(step_index: int, problem: StateSpaceProblem, /) -> int:
    index = int(step_index)
    if not 0 <= index <= problem.observations.num_steps:
        raise ValueError("Filter checkpoint step_index is outside the schedule.")
    return index


def _validate_arrays(
    arrays: Mapping[str, jnp.ndarray],
    expected_shapes: Mapping[str, tuple[int, ...]],
    /,
    *,
    owner: str,
) -> None:
    if set(arrays) != set(expected_shapes):
        raise ValueError(f"{owner} checkpoint array inventory is invalid.")
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(
                f"{owner} checkpoint array {name!r} has shape "
                f"{arrays[name].shape}; expected {shape}."
            )


def write_kalman_filter_checkpoint(
    path: str | PathLike[str],
    problem: StateSpaceProblem,
    state: KalmanFilterState,
    /,
) -> Path:
    """Atomically save a pickle-free streaming Kalman-filter state."""
    _validate_problem(problem)
    if not isinstance(state, KalmanFilterState):
        raise TypeError("state must be a KalmanFilterState.")
    if state.problem_id != problem.problem_id:
        raise ValueError("Kalman-filter state and problem IDs do not match.")
    compatibility = _base_compatibility(problem)
    compatibility["covariance_regularization"] = state.covariance_regularization
    return write_checkpoint_archive(
        path,
        kind="kalman-filter-state-v1",
        compatibility=compatibility,
        state={"step_index": state.step_index},
        arrays={
            "mean": state.mean,
            "covariance": state.covariance,
            "time": state.time,
            "log_likelihood": state.log_likelihood,
            "valid": state.valid,
            "status": state.status,
        },
    )


def read_kalman_filter_checkpoint(
    path: str | PathLike[str],
    problem: StateSpaceProblem,
    /,
    *,
    covariance_regularization: float = 0.0,
) -> KalmanFilterState:
    """Load a Kalman state only when model, schedule, and settings match."""
    _validate_problem(problem)
    regularization = float(covariance_regularization)
    if not jnp.isfinite(regularization) or regularization < 0.0:
        raise ValueError("covariance_regularization must be finite and nonnegative.")
    compatibility = _base_compatibility(problem)
    compatibility["covariance_regularization"] = regularization
    state_data, arrays = read_checkpoint_archive(
        path,
        kind="kalman-filter-state-v1",
        compatibility=compatibility,
    )
    if set(state_data) != {"step_index"}:
        raise ValueError("Kalman-filter checkpoint state manifest is invalid.")
    step_index = _validate_step_index(state_data["step_index"], problem)
    case_shape = problem.observations.case_shape
    state_size = prod(problem.model.state_shape)
    _validate_arrays(
        arrays,
        {
            "mean": case_shape + problem.model.state_shape,
            "covariance": case_shape + (state_size, state_size),
            "time": case_shape,
            "log_likelihood": case_shape,
            "valid": case_shape,
            "status": case_shape,
        },
        owner="Kalman-filter",
    )
    return KalmanFilterState(
        mean=arrays["mean"],
        covariance=arrays["covariance"],
        time=arrays["time"],
        log_likelihood=arrays["log_likelihood"],
        valid=arrays["valid"].astype(bool),
        status=arrays["status"].astype(jnp.int32),
        step_index=step_index,
        problem_id=problem.problem_id,
        covariance_regularization=regularization,
    )


def write_ensemble_filter_checkpoint(
    path: str | PathLike[str],
    problem: StateSpaceProblem,
    state: EnsembleFilterState,
    /,
) -> Path:
    """Atomically save a pickle-free streaming ensemble-filter state."""
    _validate_problem(problem)
    if not isinstance(state, EnsembleFilterState):
        raise TypeError("state must be an EnsembleFilterState.")
    if state.problem_id != problem.problem_id:
        raise ValueError("Ensemble-filter state and problem IDs do not match.")
    compatibility = _base_compatibility(problem)
    compatibility.update(
        {
            "ensemble_size": state.ensemble_size,
            "inflation": state.inflation,
            "covariance_regularization": state.covariance_regularization,
        }
    )
    return write_checkpoint_archive(
        path,
        kind="ensemble-filter-state-v1",
        compatibility=compatibility,
        state={"step_index": state.step_index},
        arrays={
            "ensemble": state.ensemble,
            "time": state.time,
            "log_likelihood": state.log_likelihood,
            "valid": state.valid,
            "status": state.status,
            "root_key_data": jr.key_data(state.root_key),
        },
    )


def read_ensemble_filter_checkpoint(
    path: str | PathLike[str],
    problem: StateSpaceProblem,
    /,
    *,
    ensemble_size: int,
    inflation: float = 1.0,
    covariance_regularization: float = 0.0,
) -> EnsembleFilterState:
    """Load an ensemble state only when model, schedule, and settings match."""
    _validate_problem(problem)
    count = int(ensemble_size)
    inflation_value = float(inflation)
    regularization = float(covariance_regularization)
    if count < 2:
        raise ValueError("ensemble_size must be at least two.")
    if not jnp.isfinite(inflation_value) or inflation_value <= 0.0:
        raise ValueError("inflation must be finite and positive.")
    if not jnp.isfinite(regularization) or regularization < 0.0:
        raise ValueError("covariance_regularization must be finite and nonnegative.")
    compatibility = _base_compatibility(problem)
    compatibility.update(
        {
            "ensemble_size": count,
            "inflation": inflation_value,
            "covariance_regularization": regularization,
        }
    )
    state_data, arrays = read_checkpoint_archive(
        path,
        kind="ensemble-filter-state-v1",
        compatibility=compatibility,
    )
    if set(state_data) != {"step_index"}:
        raise ValueError("Ensemble-filter checkpoint state manifest is invalid.")
    step_index = _validate_step_index(state_data["step_index"], problem)
    case_shape = problem.observations.case_shape
    _validate_arrays(
        arrays,
        {
            "ensemble": case_shape + (count,) + problem.model.state_shape,
            "time": case_shape,
            "log_likelihood": case_shape,
            "valid": case_shape,
            "status": case_shape,
            "root_key_data": jr.key_data(jr.key(0)).shape,
        },
        owner="Ensemble-filter",
    )
    return EnsembleFilterState(
        ensemble=arrays["ensemble"],
        time=arrays["time"],
        log_likelihood=arrays["log_likelihood"],
        valid=arrays["valid"].astype(bool),
        status=arrays["status"].astype(jnp.int32),
        root_key=jr.wrap_key_data(arrays["root_key_data"].astype(jnp.uint32)),
        step_index=step_index,
        ensemble_size=count,
        problem_id=problem.problem_id,
        inflation=inflation_value,
        covariance_regularization=regularization,
    )


def write_filter_checkpoint(
    path: str | PathLike[str],
    problem: StateSpaceProblem,
    state: FilterState,
    /,
) -> Path:
    """Write any native streaming filter state using its canonical format."""
    if isinstance(state, KalmanFilterState):
        return write_kalman_filter_checkpoint(path, problem, state)
    if isinstance(state, ParticleFilterState):
        return write_particle_filter_checkpoint(path, problem, state)
    if isinstance(state, EnsembleFilterState):
        return write_ensemble_filter_checkpoint(path, problem, state)
    raise TypeError("state must be a Kalman, particle, or ensemble filter state.")


def read_filter_checkpoint(
    path: str | PathLike[str],
    problem: StateSpaceProblem,
    algorithm: FilterCheckpointAlgorithm,
    /,
    *,
    covariance_regularization: float = 0.0,
    ensemble_size: int | None = None,
    inflation: float = 1.0,
    num_particles: int | None = None,
    resampling_method: ResamplingMethod = "systematic",
    resampling_policy: ResamplingPolicy = "ess",
    resampling_threshold: float = 0.5,
) -> FilterState:
    """Read one native streaming filter state under an explicit algorithm contract."""
    if algorithm == "kalman":
        return read_kalman_filter_checkpoint(
            path,
            problem,
            covariance_regularization=covariance_regularization,
        )
    if algorithm == "ensemble":
        if ensemble_size is None:
            raise ValueError("ensemble_size is required for an ensemble checkpoint.")
        return read_ensemble_filter_checkpoint(
            path,
            problem,
            ensemble_size=ensemble_size,
            inflation=inflation,
            covariance_regularization=covariance_regularization,
        )
    if algorithm == "particle":
        if num_particles is None:
            raise ValueError("num_particles is required for a particle checkpoint.")
        return read_particle_filter_checkpoint(
            path,
            problem,
            num_particles=num_particles,
            resampling_method=resampling_method,
            resampling_policy=resampling_policy,
            resampling_threshold=resampling_threshold,
        )
    raise ValueError("algorithm must be 'kalman', 'particle', or 'ensemble'.")


__all__ = [
    "FilterCheckpointAlgorithm",
    "FilterState",
    "read_ensemble_filter_checkpoint",
    "read_filter_checkpoint",
    "read_kalman_filter_checkpoint",
    "write_ensemble_filter_checkpoint",
    "write_filter_checkpoint",
    "write_kalman_filter_checkpoint",
]
