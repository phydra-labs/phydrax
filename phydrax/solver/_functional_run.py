#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from .._training import (
    DelayedTargetPolicy,
    EvaluationParametersFn,
    ExponentialMovingAverageTargetPolicy,
)
from ._functional_objective import _FunctionalObjective
from ._functional_precision import FunctionalPrecisionPolicy
from ._functional_training import FunctionalTrainingPlan


@dataclass(frozen=True, slots=True)
class FunctionalSolveConfig:
    """Backend-neutral controls for one functional optimization run."""

    num_iter: int
    evaluation_parameters: EvaluationParametersFn | None = None
    parameter_paths: tuple[str, ...] | None = None
    parameter_shapes: tuple[tuple[int, ...], ...] = ()
    parameter_dtypes: tuple[str, ...] = ()
    parameter_alias_groups: tuple[tuple[str, ...], ...] = ()
    seed: int = 0
    jit: bool = True
    keep_best: bool = True
    log_every: int = 1
    log_terms: bool = True
    log_path: str | Path | None = None
    tensorboard_log_dir: str | Path | None = None
    tensorboard_every: int | None = None
    tensorboard_flush_every: int = 10
    profile_adaptive: bool = False
    train_term_sample_size: int | None = None
    precision: FunctionalPrecisionPolicy | None = None
    training: FunctionalTrainingPlan | None = None
    resume: bool = False
    accepted_update_hook: Any = None
    target_policy: DelayedTargetPolicy | ExponentialMovingAverageTargetPolicy | None = (
        None
    )

    def __post_init__(self):
        iterations = int(self.num_iter)
        log_every = int(self.log_every)
        flush_every = int(self.tensorboard_flush_every)
        if iterations <= 0:
            raise ValueError("FunctionalSolveConfig.num_iter must be positive.")
        if log_every < 0:
            raise ValueError("log_every must be >= 0.")
        if flush_every <= 0:
            raise ValueError("tensorboard_flush_every must be positive.")
        if self.precision is not None and not isinstance(
            self.precision, FunctionalPrecisionPolicy
        ):
            raise TypeError("precision must be a FunctionalPrecisionPolicy or None.")
        if self.training is not None and not isinstance(
            self.training, FunctionalTrainingPlan
        ):
            raise TypeError("training must be a FunctionalTrainingPlan or None.")
        if self.resume and self.training is None:
            raise ValueError("resume=True requires a FunctionalTrainingPlan.")
        if self.accepted_update_hook is not None and not callable(
            self.accepted_update_hook
        ):
            raise TypeError("accepted_update_hook must be callable or None.")
        if self.target_policy is not None and not isinstance(
            self.target_policy,
            (DelayedTargetPolicy, ExponentialMovingAverageTargetPolicy),
        ):
            raise TypeError("target_policy has an unsupported type.")
        if self.parameter_paths is None:
            if (
                self.parameter_shapes
                or self.parameter_dtypes
                or self.parameter_alias_groups
            ):
                raise ValueError(
                    "Parameter shapes, dtypes, and aliases require parameter_paths."
                )
        elif len(self.parameter_paths) != len(self.parameter_shapes) or len(
            self.parameter_paths
        ) != len(self.parameter_dtypes):
            raise ValueError(
                "Parameter paths, shapes, and dtypes must have equal lengths."
            )
        object.__setattr__(self, "num_iter", iterations)
        object.__setattr__(self, "log_every", log_every)
        object.__setattr__(self, "tensorboard_flush_every", flush_every)


def validate_term_sample_size(
    value: int | None,
    /,
    *,
    num_terms: int,
) -> int | None:
    """Normalize optional unbiased objective-term subsampling."""
    if value is None:
        return None
    count = int(num_terms)
    if count <= 0:
        raise ValueError("train_term_sample_size requires at least one training term.")
    sample_size = int(value)
    if sample_size <= 0:
        raise ValueError("train_term_sample_size must be positive.")
    if sample_size >= count:
        return None
    return sample_size


def select_train_terms(
    terms: tuple[Any, ...],
    /,
    *,
    sample_size: int | None,
    key: Any,
) -> tuple[tuple[Any, ...], tuple[int, ...], Any]:
    """Select one unbiased subset while retaining original term indices."""
    count = len(terms)
    if sample_size is None:
        return terms, tuple(range(count)), jnp.asarray(1.0, dtype=float)
    sampled = jr.choice(
        key,
        count,
        shape=(int(sample_size),),
        replace=False,
    )
    active_indices = tuple(int(index) for index in np.asarray(sampled, dtype=np.int32))
    active = tuple(terms[index] for index in active_indices)
    scale = jnp.asarray(count / int(sample_size), dtype=float)
    return active, active_indices, scale


def expand_train_terms(
    active_terms: Any,
    /,
    *,
    active_term_indices: tuple[int, ...],
    num_terms: int,
) -> Any:
    """Restore selected term values to the stable full-objective layout."""
    active_array = jnp.asarray(active_terms, dtype=float).reshape((-1,))
    if int(active_array.shape[0]) == 0:
        return jnp.zeros((int(num_terms),), dtype=float)
    expanded = jnp.full((int(num_terms),), jnp.nan, dtype=float)
    for local_index, term_index in enumerate(active_term_indices):
        expanded = expanded.at[int(term_index)].set(active_array[int(local_index)])
    return expanded


def replace_solver_state(
    solver: Any,
    /,
    *,
    functions: Any,
    objective: _FunctionalObjective,
) -> Any:
    """Persist the two mutable products of a functional training run."""
    return eqx.tree_at(
        lambda item: (item.functions, item.objective),
        solver,
        (functions, objective),
    )


__all__ = [
    "FunctionalSolveConfig",
    "expand_train_terms",
    "replace_solver_state",
    "select_train_terms",
    "validate_term_sample_size",
]
