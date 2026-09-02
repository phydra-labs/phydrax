#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jaxtyping import Array, ArrayLike, PyTree

import phydrax.ein as ein

from .._sampling import derive_key, SampleAddress
from .._strict import StrictModule
from ..nn.parameters import ParameterSubspace
from ..solver._functional_solver import FunctionalSolver


class SWAGCollectionPlan(StrictModule):
    """Static committed-iterate collection and fixed snapshot capacity."""

    start_step: int = eqx.field(static=True)
    collect_every: int = eqx.field(static=True)
    snapshot_capacity: int = eqx.field(static=True)
    num_draws: int = eqx.field(static=True)
    diagonal_regularization: float = eqx.field(static=True)
    accumulation_precision: Any = eqx.field(static=True)

    def __init__(
        self,
        *,
        start_step: int,
        collect_every: int,
        snapshot_capacity: int,
        num_draws: int,
        diagonal_regularization: float = 0.0,
        accumulation_precision: Any = jnp.float64,
    ):
        start = int(start_step)
        cadence = int(collect_every)
        capacity = int(snapshot_capacity)
        draws = int(num_draws)
        regularization = float(diagonal_regularization)
        dtype = jnp.dtype(accumulation_precision)
        if start < 1 or cadence <= 0 or capacity < 2 or draws <= 0:
            raise ValueError(
                "start_step/cadence/draws must be positive and snapshot_capacity >= 2."
            )
        if not jnp.issubdtype(dtype, jnp.floating):
            raise TypeError("SWAG accumulation precision must be real floating.")
        if not jnp.isfinite(regularization) or regularization < 0.0:
            raise ValueError(
                "SWAG diagonal regularization must be finite and nonnegative."
            )
        self.start_step = start
        self.collect_every = cadence
        self.snapshot_capacity = capacity
        self.num_draws = draws
        self.diagonal_regularization = regularization
        self.accumulation_precision = dtype


class SWAGState(StrictModule):
    """Welford diagonal moments and a fixed-capacity raw-snapshot ring."""

    mean: Array
    m2: Array
    snapshots: Array
    snapshot_mask: Array
    count: Array
    cursor: Array
    last_solver_step: Array
    parameter_paths: tuple[str, ...] = eqx.field(static=True)

    @classmethod
    def initialize(
        cls,
        position: ArrayLike,
        /,
        *,
        snapshot_capacity: int,
        parameter_paths: tuple[str, ...],
        accumulation_precision: Any,
    ) -> SWAGState:
        value = jnp.asarray(position, dtype=accumulation_precision)
        if value.ndim != 1 or value.size == 0:
            raise ValueError("SWAG position must be a nonempty flat vector.")
        capacity = int(snapshot_capacity)
        if capacity < 2:
            raise ValueError("SWAG snapshot_capacity must be at least two.")
        return cls(
            mean=jnp.zeros_like(value),
            m2=jnp.zeros_like(value),
            snapshots=jnp.zeros((capacity, value.size), dtype=value.dtype),
            snapshot_mask=jnp.zeros((capacity,), dtype=bool),
            count=jnp.asarray(0, dtype=jnp.int32),
            cursor=jnp.asarray(0, dtype=jnp.int32),
            last_solver_step=jnp.asarray(0, dtype=jnp.int32),
            parameter_paths=tuple(parameter_paths),
        )

    @property
    def active_snapshot_count(self) -> Array:
        return jnp.sum(self.snapshot_mask, dtype=jnp.int32)

    @property
    def effective_rank(self) -> Array:
        return jnp.maximum(self.active_snapshot_count - 1, 0)


class SWAGResult(StrictModule):
    """Final solver, SWAG approximation, and deterministic sampling adapter."""

    solver: FunctionalSolver
    state: SWAGState
    parameter_subspace: ParameterSubspace
    collection: SWAGCollectionPlan
    approximation: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        solver: FunctionalSolver,
        state: SWAGState,
        parameter_subspace: ParameterSubspace,
        collection: SWAGCollectionPlan,
    ):
        if int(state.count) < 2 or int(state.active_snapshot_count) < 2:
            raise ValueError("SWAG requires at least two committed collected iterates.")
        if tuple(parameter_subspace.leaf_paths) != state.parameter_paths:
            raise ValueError("SWAG parameter layout changed during fitting.")
        self.solver = solver
        self.state = state
        self.parameter_subspace = parameter_subspace
        self.collection = collection
        self.approximation = "swag_low_rank_diagonal"

    def sample(
        self,
        key: Array,
        /,
        *,
        num_draws: int | None = None,
    ) -> PyTree[Array]:
        draws = self.collection.num_draws if num_draws is None else int(num_draws)
        if draws <= 0:
            raise ValueError("num_draws must be positive.")
        address = SampleAddress(
            "uq.swag",
            "parameter-draw",
            target=self.state.parameter_paths,
            role="posterior",
        )
        vectors = jnp.stack(
            tuple(
                sample_swag_vector(
                    self.state,
                    derive_key(key, address, draw),
                    diagonal_regularization=self.collection.diagonal_regularization,
                )
                for draw in range(draws)
            )
        )
        models = tuple(
            self.parameter_subspace.reconstruct_vector(vectors[index])
            for index in range(draws)
        )
        return jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *models)

    def predict(
        self,
        predict: Callable[..., Any],
        key: Array,
        /,
        *args: Any,
        num_draws: int | None = None,
        **kwargs: Any,
    ) -> Any:
        if not callable(predict):
            raise TypeError("predict must be callable.")
        models = self.sample(key, num_draws=num_draws)
        draws = self.collection.num_draws if num_draws is None else int(num_draws)
        outputs = tuple(
            predict(
                jax.tree_util.tree_map(lambda leaf: leaf[index], models), *args, **kwargs
            )
            for index in range(draws)
        )
        return jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *outputs)


def update_swag_state(
    state: SWAGState,
    position: ArrayLike,
    /,
    *,
    solver_step: int | Array,
) -> SWAGState:
    """Pure fixed-shape Welford and snapshot-ring update."""
    value = jnp.asarray(position, dtype=state.mean.dtype)
    if value.shape != state.mean.shape:
        raise ValueError("SWAG position shape changed.")
    step = jnp.asarray(solver_step, dtype=jnp.int32)
    next_count = state.count + 1
    delta = value - state.mean
    mean = state.mean + delta / next_count.astype(value.dtype)
    m2 = state.m2 + delta * (value - mean)
    snapshots = state.snapshots.at[state.cursor].set(value)
    snapshot_mask = state.snapshot_mask.at[state.cursor].set(True)
    cursor = (state.cursor + 1) % state.snapshots.shape[0]
    return eqx.tree_at(
        lambda item: (
            item.mean,
            item.m2,
            item.snapshots,
            item.snapshot_mask,
            item.count,
            item.cursor,
            item.last_solver_step,
        ),
        state,
        (mean, m2, snapshots, snapshot_mask, next_count, cursor, step),
    )


def sample_swag_vector(
    state: SWAGState,
    key: Array,
    /,
    *,
    diagonal_regularization: float = 0.0,
) -> Array:
    """Draw without materializing a dense parameter covariance."""
    count = state.count.astype(state.mean.dtype)
    active_count = state.active_snapshot_count.astype(state.mean.dtype)
    state = eqx.error_if(
        state,
        (state.count < 2) | (state.active_snapshot_count < 2),
        "SWAG sampling requires at least two collected iterates.",
    )
    diagonal_variance = state.m2 / (count - 1.0) + jnp.asarray(
        diagonal_regularization, dtype=state.mean.dtype
    )
    state = eqx.error_if(
        state,
        jnp.any(~jnp.isfinite(diagonal_variance)) | jnp.any(diagonal_variance < 0.0),
        "SWAG diagonal covariance is invalid.",
    )
    centered = jnp.where(
        state.snapshot_mask[:, None],
        state.snapshots - state.mean[None, :],
        0.0,
    )
    diagonal_key, low_rank_key = jr.split(key)
    diagonal_noise = jr.normal(diagonal_key, state.mean.shape, dtype=state.mean.dtype)
    low_rank_noise = jr.normal(
        low_rank_key, (state.snapshots.shape[0],), dtype=state.mean.dtype
    )
    low_rank = ein.contract("kp,k->p", centered, low_rank_noise) / jnp.sqrt(
        active_count - 1.0
    )
    return state.mean + (
        jnp.sqrt(diagonal_variance) * diagonal_noise + low_rank
    ) / jnp.sqrt(jnp.asarray(2.0, dtype=state.mean.dtype))


def fit_swag(
    solver: FunctionalSolver,
    /,
    *,
    optim: optax.GradientTransformation | optax.GradientTransformationExtraArgs,
    num_iter: int,
    collection: SWAGCollectionPlan,
    parameter_subspace: ParameterSubspace | None = None,
    evaluation_parameters: Callable[..., Any] | None = None,
    seed: int = 0,
    jit: bool = True,
    log_every: int = 0,
) -> SWAGResult:
    """Fit SWAG through the FunctionalSolver's private committed-update hook."""
    if not isinstance(solver, FunctionalSolver):
        raise TypeError("solver must be a FunctionalSolver.")
    if not isinstance(collection, SWAGCollectionPlan):
        raise TypeError("collection must be a SWAGCollectionPlan.")
    if not isinstance(
        optim,
        (optax.GradientTransformation, optax.GradientTransformationExtraArgs),
    ):
        raise TypeError("SWAG currently supports standard or extra-argument Optax only.")
    iterations = int(num_iter)
    if iterations < collection.start_step:
        raise ValueError("num_iter must reach the SWAG collection start step.")
    subspace = (
        ParameterSubspace(solver.functions, eqx.is_inexact_array)
        if parameter_subspace is None
        else parameter_subspace
    )
    if not isinstance(subspace, ParameterSubspace):
        raise TypeError("parameter_subspace must be ParameterSubspace or None.")
    subspace.validate_root(solver.functions)
    state = SWAGState.initialize(
        subspace.pack(),
        snapshot_capacity=collection.snapshot_capacity,
        parameter_paths=subspace.leaf_paths,
        accumulation_precision=collection.accumulation_precision,
    )

    def collect(step: int, parameters: Any) -> None:
        nonlocal state
        if step < collection.start_step:
            return
        if (step - collection.start_step) % collection.collect_every != 0:
            return
        current_subspace = subspace.rebase(parameters)
        state = update_swag_state(
            state,
            current_subspace.pack(),
            solver_step=step,
        )

    fitted = solver.solve(
        num_iter=iterations,
        optim=optim,
        evaluation_parameters=evaluation_parameters,
        parameter_subspace=parameter_subspace,
        seed=seed,
        jit=jit,
        keep_best=False,
        log_every=log_every,
        _accepted_update_hook=collect,
    )
    final_subspace = subspace.rebase(fitted.functions)
    return SWAGResult(
        solver=fitted,
        state=state,
        parameter_subspace=final_subspace,
        collection=collection,
    )


__all__ = [
    "SWAGCollectionPlan",
    "SWAGResult",
    "SWAGState",
    "fit_swag",
    "sample_swag_vector",
    "update_swag_state",
]
