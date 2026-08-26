#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from time import perf_counter
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._sampling import (
    design_signature,
    materialize_design,
    SobolDesign,
    UnitDesign,
)
from .._strict import StrictModule
from ..kernels import AbstractPositiveDefiniteKernel
from ..optim._bounded_search import _BoundedVectorDomain
from ._gp_backend import (
    exact_gp_cholesky,
    exact_gp_predict_diagonal_from_covariances,
)
from ._gp_likelihood import GaussianProcessLikelihoodState


_PROPOSAL_INITIAL = 0
_PROPOSAL_ACQUISITION = 1
_PROPOSAL_FALLBACK = 2


class GaussianProcessMAPSearch(StrictModule):
    """Sequential expected-improvement search for bounded posterior modes.

    ``surrogate.noise_scale`` is expressed in raw negative-log-density units.
    Observations are standardized before GP fitting, so the covariance uses
    ``noise_scale / objective_scale``. ``surrogate.jitter`` is dimensionless and
    acts directly in standardized covariance units.
    """

    max_evaluations: int = eqx.field(static=True)
    initial_evaluations: int = eqx.field(static=True)
    candidate_count: int = eqx.field(static=True)
    surrogate: GaussianProcessLikelihoodState
    design: UnitDesign
    improvement_margin: float = eqx.field(static=True)
    minimum_separation: float = eqx.field(static=True)

    def __init__(
        self,
        max_evaluations: int,
        /,
        *,
        surrogate: GaussianProcessLikelihoodState,
        initial_evaluations: int = 8,
        candidate_count: int = 512,
        design: UnitDesign | None = None,
        improvement_margin: float = 0.01,
        minimum_separation: float = 1e-6,
    ):
        maximum = int(max_evaluations)
        initial = int(initial_evaluations)
        candidates = int(candidate_count)
        if maximum < 2:
            raise ValueError("max_evaluations must be at least two.")
        if initial < 2:
            raise ValueError("initial_evaluations must be at least two.")
        if initial > maximum:
            raise ValueError("initial_evaluations cannot exceed max_evaluations.")
        if candidates < 2:
            raise ValueError("candidate_count must be at least two.")
        if not isinstance(surrogate, GaussianProcessLikelihoodState):
            raise TypeError("surrogate must be a GaussianProcessLikelihoodState.")
        if not isinstance(surrogate.kernel, AbstractPositiveDefiniteKernel):
            raise TypeError("surrogate.kernel must be an AbstractPositiveDefiniteKernel.")
        if surrogate.kernel.input_ndim != 1:
            raise ValueError("surrogate.kernel must accept vector inputs.")
        if surrogate.noise_scale.ndim != 0:
            raise ValueError("surrogate.noise_scale must be scalar for MAP search.")
        resolved_design = SobolDesign(scrambled=True) if design is None else design
        if not isinstance(resolved_design, UnitDesign):
            raise TypeError("design must be a UnitDesign.")
        margin = float(improvement_margin)
        separation = float(minimum_separation)
        if not isfinite(margin) or margin < 0.0:
            raise ValueError("improvement_margin must be finite and nonnegative.")
        if not isfinite(separation) or separation <= 0.0:
            raise ValueError("minimum_separation must be finite and positive.")
        self.max_evaluations = maximum
        self.initial_evaluations = initial
        self.candidate_count = candidates
        self.surrogate = surrogate
        self.design = resolved_design
        self.improvement_margin = margin
        self.minimum_separation = separation

    @property
    def method_id(self) -> str:
        return "gaussian-process-map-search-v1"


class _GaussianProcessMAPEvidence(StrictModule):
    best_vector: Array
    raw_objective: Array
    evaluated_vectors: Array
    raw_objectives: Array
    valid_evaluations: Array
    proposal_kinds: Array
    best_objective_history: Array
    best_history_valid: Array
    lower_bounds: Array
    upper_bounds: Array
    key: Array
    search: GaussianProcessMAPSearch
    valid: bool = eqx.field(static=True)
    termination_reason: str = eqx.field(static=True)
    objective_evaluations: int = eqx.field(static=True)
    invalid_evaluations: int = eqx.field(static=True)
    fallback_count: int = eqx.field(static=True)
    surrogate_failure_count: int = eqx.field(static=True)
    design_signature: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    proposal_seconds: float = eqx.field(static=True)
    objective_seconds: float = eqx.field(static=True)


def _objective_normalization(values: Array, /) -> tuple[Array, Array]:
    mean = jnp.mean(values)
    scale = jnp.maximum(
        jnp.std(values),
        jnp.sqrt(jnp.finfo(values.dtype).eps),
    )
    return mean, scale


@eqx.filter_jit
def _posterior(
    points: Array,
    raw_values: Array,
    queries: Array,
    surrogate: GaussianProcessLikelihoodState,
    /,
) -> tuple[Array, Array, Array]:
    objective_mean, objective_scale = _objective_normalization(raw_values)
    residual = (raw_values - objective_mean) / objective_scale
    raw_noise = jnp.asarray(surrogate.noise_scale, dtype=raw_values.dtype)
    standardized_noise = raw_noise / objective_scale
    standardized_jitter = jnp.asarray(surrogate.jitter, dtype=raw_values.dtype)
    cholesky = exact_gp_cholesky(
        points,
        kernel=surrogate.kernel,
        noise_scale=standardized_noise,
        jitter=standardized_jitter,
    )
    cross = surrogate.kernel.matrix(queries, points)
    standardized_mean, standardized_variance = exact_gp_predict_diagonal_from_covariances(
        cholesky,
        cross,
        surrogate.kernel.diagonal(queries),
        residual,
    )
    standard_deviation = objective_scale * jnp.sqrt(
        jnp.maximum(standardized_variance, 0.0)
    )
    mean = objective_mean + objective_scale * standardized_mean
    usable = (
        jnp.all(jnp.isfinite(cholesky))
        & jnp.all(jnp.isfinite(mean))
        & jnp.all(jnp.isfinite(standard_deviation))
    )
    return mean, standard_deviation, usable


def _expected_improvement(mean, standard_deviation, incumbent, margin, /):
    improvement = incumbent - mean - margin
    uncertain = standard_deviation > 0.0
    safe_scale = jnp.where(uncertain, standard_deviation, 1.0)
    standardized = improvement / safe_scale
    regular = improvement * jsp.stats.norm.cdf(
        standardized
    ) + safe_scale * jsp.stats.norm.pdf(standardized)
    return jnp.where(uncertain, regular, jnp.maximum(improvement, 0.0))


def _space_filling_index(candidates, occupied, /):
    squared = jnp.sum(
        (candidates[:, None, :] - occupied[None, :, :]) ** 2,
        axis=-1,
    )
    return jnp.argmax(jnp.min(squared, axis=1))


def _initial_points(
    box: _BoundedVectorDomain,
    search: GaussianProcessMAPSearch,
    key,
    /,
):
    initial = box.to_unit(box.initial)[None, :]
    if search.initial_evaluations == 1:
        return initial
    sampled = materialize_design(
        search.design,
        count=search.initial_evaluations - 1,
        dimension=box.dimension,
        key=key,
    ).astype(initial.dtype)
    return jnp.concatenate([initial, sampled], axis=0)


def _evaluate_batch(objective, unit_points, box, /):
    physical = box.from_unit(unit_points)
    values = jax.vmap(objective)(physical)
    if values.shape != (unit_points.shape[0],):
        raise ValueError("MAP objective must return one scalar per position.")
    return values


def _bounded_gaussian_process_map_search(
    objective,
    initial_vector: ArrayLike,
    lower_bounds: ArrayLike,
    upper_bounds: ArrayLike,
    search: GaussianProcessMAPSearch,
    /,
    *,
    key: Array,
) -> _GaussianProcessMAPEvidence:
    box = _BoundedVectorDomain(initial_vector, lower_bounds, upper_bounds)
    root_key = key
    key, design_key = jax.random.split(key)
    proposal_started = perf_counter()
    proposal_count = search.max_evaluations - search.initial_evaluations
    initial_design_count = search.initial_evaluations - 1
    design_points = materialize_design(
        search.design,
        count=initial_design_count + proposal_count * search.candidate_count,
        dimension=box.dimension,
        key=design_key,
    ).astype(box.initial.dtype)
    initial = box.to_unit(box.initial)[None, :]
    unit_points = jnp.concatenate(
        (initial, design_points[:initial_design_count]),
        axis=0,
    )
    candidate_pools = design_points[initial_design_count:].reshape(
        proposal_count,
        search.candidate_count,
        box.dimension,
    )
    proposal_seconds = perf_counter() - proposal_started
    objective_started = perf_counter()
    raw_objectives = jax.block_until_ready(_evaluate_batch(objective, unit_points, box))
    objective_seconds = perf_counter() - objective_started
    proposal_kinds = jnp.full(
        (search.initial_evaluations,),
        _PROPOSAL_INITIAL,
        dtype=jnp.int32,
    )
    fallback_count = 0
    surrogate_failure_count = 0
    for evaluation in range(search.initial_evaluations, search.max_evaluations):
        proposal_started = perf_counter()
        candidates = candidate_pools[evaluation - search.initial_evaluations]
        valid = jnp.isfinite(raw_objectives)
        valid_count = int(jnp.sum(valid))
        proposed_kind: Literal[1, 2]
        if valid_count >= 2:
            valid_points = unit_points[np.asarray(valid)]
            valid_values = raw_objectives[np.asarray(valid)]
            mean, standard_deviation, usable = _posterior(
                valid_points,
                valid_values,
                candidates,
                search.surrogate,
            )
            utility = _expected_improvement(
                mean,
                standard_deviation,
                jnp.min(valid_values),
                search.improvement_margin,
            )
            squared = jnp.sum(
                (candidates[:, None, :] - unit_points[None, :, :]) ** 2,
                axis=-1,
            )
            separated = jnp.min(squared, axis=1) > search.minimum_separation**2
            usable_utility = jnp.where(
                separated & jnp.isfinite(utility) & usable,
                utility,
                -jnp.inf,
            )
            if bool(jnp.any(jnp.isfinite(usable_utility))):
                candidate_index = int(jnp.argmax(usable_utility))
                proposed_kind = _PROPOSAL_ACQUISITION
            else:
                candidate_index = int(_space_filling_index(candidates, unit_points))
                proposed_kind = _PROPOSAL_FALLBACK
                fallback_count += 1
                surrogate_failure_count += int(not bool(usable))
        else:
            candidate_index = int(_space_filling_index(candidates, unit_points))
            proposed_kind = _PROPOSAL_FALLBACK
            fallback_count += 1
        proposed = candidates[candidate_index]
        proposal_seconds += perf_counter() - proposal_started
        objective_started = perf_counter()
        raw_value = jax.block_until_ready(jnp.asarray(objective(box.from_unit(proposed))))
        objective_seconds += perf_counter() - objective_started
        if raw_value.ndim != 0:
            raise ValueError("MAP objective must return a scalar.")
        unit_points = jnp.concatenate([unit_points, proposed[None, :]], axis=0)
        raw_objectives = jnp.concatenate([raw_objectives, raw_value[None]], axis=0)
        proposal_kinds = jnp.concatenate(
            [proposal_kinds, jnp.asarray([proposed_kind], dtype=jnp.int32)],
            axis=0,
        )
    valid = jnp.isfinite(raw_objectives)
    masked = jnp.where(valid, raw_objectives, jnp.inf)
    best_index = int(jnp.argmin(masked))
    has_valid = bool(jnp.any(valid))
    running = jnp.minimum.accumulate(masked)
    running_valid = jnp.isfinite(running)
    raw_objective = raw_objectives[best_index] if has_valid else jnp.asarray(jnp.nan)
    return _GaussianProcessMAPEvidence(
        best_vector=box.from_unit(unit_points[best_index]),
        raw_objective=raw_objective,
        evaluated_vectors=box.from_unit(unit_points),
        raw_objectives=raw_objectives,
        valid_evaluations=valid,
        proposal_kinds=proposal_kinds,
        best_objective_history=running,
        best_history_valid=running_valid,
        lower_bounds=box.lower,
        upper_bounds=box.upper,
        key=root_key,
        search=search,
        valid=has_valid,
        termination_reason=(
            "evaluation_budget_exhausted" if has_valid else "no_finite_candidates"
        ),
        objective_evaluations=search.max_evaluations,
        invalid_evaluations=int(jnp.sum(~valid)),
        fallback_count=fallback_count,
        surrogate_failure_count=surrogate_failure_count,
        design_signature=design_signature(search.design),
        method_id=search.method_id,
        proposal_seconds=proposal_seconds,
        objective_seconds=objective_seconds,
    )


__all__ = ["GaussianProcessMAPSearch"]
