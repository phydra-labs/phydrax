#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike


_TimeLayout = Literal["shared", "per_path"]


@dataclass(frozen=True, slots=True)
class ValidatedSolutionArrays:
    """Normalized arrays and axes for one saved numerical solution."""

    times: Array
    states: Array
    valid: Array
    sample_shape: tuple[int, ...]
    state_shape: tuple[int, ...]


def validate_solution_arrays(
    times: ArrayLike,
    states: ArrayLike,
    valid: ArrayLike,
    /,
    *,
    sample_shape: Sequence[int],
    state_shape: Sequence[int] | None,
    time_layout: _TimeLayout,
    owner: str,
) -> ValidatedSolutionArrays:
    """Validate the common sample/time/state layout of a saved result."""
    samples = tuple(int(size) for size in sample_shape)
    if any(size <= 0 for size in samples):
        raise ValueError(f"{owner} sample dimensions must be positive.")
    time_values = jnp.asarray(times, dtype=float)
    state_values = jnp.asarray(states)
    valid_values = jnp.asarray(valid, dtype=bool)
    if time_layout == "shared":
        if time_values.ndim != 1 or int(time_values.size) <= 0:
            raise ValueError(f"{owner} times must be a non-empty rank-1 array.")
        trajectory_shape = samples + (int(time_values.size),)
    elif time_layout == "per_path":
        if time_values.ndim != len(samples) + 1:
            raise ValueError(
                f"{owner} times must have shape sample_shape + (num_times,)."
            )
        if tuple(time_values.shape[: len(samples)]) != samples:
            raise ValueError(f"{owner} times do not match sample_shape.")
        trajectory_shape = samples + (int(time_values.shape[-1]),)
    else:
        raise ValueError(f"Unknown saved-solution time layout {time_layout!r}.")
    if tuple(state_values.shape[: len(trajectory_shape)]) != trajectory_shape:
        raise ValueError(
            f"{owner} states must begin with sample_shape + (num_times,)."
        )
    inferred_state = tuple(state_values.shape[len(trajectory_shape) :])
    declared_state = (
        inferred_state
        if state_shape is None
        else tuple(int(size) for size in state_shape)
    )
    if inferred_state != declared_state:
        raise ValueError(
            f"{owner} states must end with state shape {declared_state}; "
            f"got {inferred_state}."
        )
    if valid_values.shape != trajectory_shape:
        raise ValueError(
            f"{owner} valid must have shape {trajectory_shape}; "
            f"got {valid_values.shape}."
        )
    return ValidatedSolutionArrays(
        time_values,
        state_values,
        valid_values,
        samples,
        declared_state,
    )


__all__ = ["ValidatedSolutionArrays", "validate_solution_arrays"]
