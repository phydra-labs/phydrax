#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..fidelity import FidelityPath
from ._gp_fidelity import FidelityGaussianProcess
from ._gp_multioutput import (
    MultiOutputDesign,
    MultiOutputGaussianProcessLikelihoodState,
)


class TargetVarianceAcquisitionPolicy(StrictModule, NonTrainableState):
    """Target-fidelity integrated-variance reduction per common cost unit."""

    path: FidelityPath
    target_points: Array
    target_weights: Array
    level_costs: Array
    batch_size: int = eqx.field(static=True)
    cost_unit: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        path: FidelityPath,
        target_points: ArrayLike,
        level_costs: ArrayLike,
        /,
        *,
        target_weights: ArrayLike | None = None,
        batch_size: int = 1,
        cost_unit: str = "relative-cost",
        policy_id: str | None = None,
    ):
        if not isinstance(path, FidelityPath):
            raise TypeError("path must be a FidelityPath.")
        points = _as_points(target_points)
        weights = (
            jnp.ones((points.shape[0],), dtype=points.dtype)
            if target_weights is None
            else jnp.asarray(target_weights, dtype=points.dtype)
        )
        if weights.shape != (points.shape[0],) or bool(
            jnp.any(~jnp.isfinite(weights) | (weights < 0.0))
        ):
            raise ValueError(
                "target_weights must be finite non-negative target-point weights."
            )
        mass = jnp.sum(weights)
        if not bool(jnp.isfinite(mass)) or float(np.asarray(mass)) <= 0.0:
            raise ValueError("target_weights must have positive total mass.")
        costs = jnp.asarray(level_costs, dtype=float)
        if costs.shape != (path.num_levels,) or bool(
            jnp.any(~jnp.isfinite(costs) | (costs <= 0.0))
        ):
            raise ValueError(
                "level_costs must contain one finite positive cost per level."
            )
        count = int(batch_size)
        if count <= 0:
            raise ValueError("batch_size must be positive.")
        unit = str(cost_unit)
        if not unit:
            raise ValueError("cost_unit must be non-empty.")
        self.path = path
        self.target_points = points
        self.target_weights = weights / mass
        self.level_costs = costs
        self.batch_size = count
        self.cost_unit = unit
        self.policy_id = (
            canonical_fingerprint(
                {
                    "kind": "target-variance-fidelity-acquisition",
                    "path": path.path_id,
                    "target": array_tree_fingerprint(points),
                    "weights": array_tree_fingerprint(self.target_weights),
                    "costs": array_tree_fingerprint(costs),
                    "batch_size": count,
                    "cost_unit": unit,
                }
            )
            if policy_id is None
            else _identifier(policy_id, "policy_id")
        )


class FidelityAcquisitionResult(StrictModule, NonTrainableState):
    """Greedy input-level selections and their target-variance evidence."""

    selected_indices: Array
    selected_level_ids: tuple[str, ...] = eqx.field(static=True)
    scores: Array
    marginal_variance_reductions: Array
    initial_target_variance: Array
    final_target_variance: Array
    policy_id: str = eqx.field(static=True)
    model_dataset_id: str = eqx.field(static=True)


def select_fidelity_acquisition(
    model: FidelityGaussianProcess,
    state: MultiOutputGaussianProcessLikelihoodState,
    candidate_points: ArrayLike,
    candidate_level_ids: Sequence[str],
    policy: TargetVarianceAcquisitionPolicy,
    /,
    *,
    available: ArrayLike | None = None,
) -> FidelityAcquisitionResult:
    """Greedily maximize target posterior-variance reduction per evaluation cost."""

    if not isinstance(model, FidelityGaussianProcess):
        raise TypeError("model must be a FidelityGaussianProcess.")
    if not isinstance(state, MultiOutputGaussianProcessLikelihoodState):
        raise TypeError("state must be a MultiOutputGaussianProcessLikelihoodState.")
    if not isinstance(policy, TargetVarianceAcquisitionPolicy):
        raise TypeError("policy must be a TargetVarianceAcquisitionPolicy.")
    if model.path.path_id != policy.path.path_id:
        raise ValueError("Acquisition policy and GP must use the same fidelity path.")
    model.validate_state(state)
    candidates = _as_points(candidate_points)
    level_ids = tuple(str(value) for value in candidate_level_ids)
    if len(level_ids) != int(candidates.shape[0]):
        raise ValueError("candidate_level_ids must contain one level per candidate.")
    unknown = tuple(sorted(set(level_ids) - set(model.path.level_ids)))
    if unknown:
        raise ValueError(f"Candidate fidelity levels are not on the GP path: {unknown}.")
    level_indices = jnp.asarray(
        tuple(model.path.level_ids.index(level_id) for level_id in level_ids),
        dtype=jnp.int32,
    )
    admitted = (
        jnp.ones((candidates.shape[0],), dtype=bool)
        if available is None
        else jnp.asarray(available, dtype=bool)
    )
    if admitted.shape != (candidates.shape[0],):
        raise ValueError("available must contain one flag per candidate.")
    selection_count = min(policy.batch_size, int(jnp.sum(admitted)))
    if selection_count <= 0:
        raise ValueError("No fidelity acquisition candidates are available.")

    target_count = int(policy.target_points.shape[0])
    target_level = model.path.num_levels - 1
    combined_points = jnp.concatenate((policy.target_points, candidates), axis=0)
    combined_outputs = jnp.concatenate(
        (
            jnp.full((target_count,), target_level, dtype=jnp.int32),
            level_indices,
        )
    )
    query = MultiOutputDesign(
        combined_points,
        combined_outputs,
        output_names=model.path.level_ids,
    )
    condition = model.discrepancy.condition(
        jnp.zeros_like(model.observations),
        query,
        state=state,
    )
    covariance = condition.covariance
    initial_variance = jnp.sum(
        policy.target_weights * jnp.diag(covariance)[:target_count]
    )
    candidate_design = MultiOutputDesign(
        candidates,
        level_indices,
        output_names=model.path.level_ids,
    )
    observation_noise = state.observation_noise(candidate_design)
    costs = policy.level_costs[level_indices]
    active = admitted
    selected: list[int] = []
    selected_scores: list[Array] = []
    reductions: list[Array] = []

    for _ in range(selection_count):
        candidate_variance = jnp.diag(covariance)[target_count:]
        cross = covariance[:target_count, target_count:]
        denominators = candidate_variance + observation_noise * observation_noise
        marginal = jnp.sum(
            policy.target_weights[:, None] * cross * cross,
            axis=0,
        ) / jnp.maximum(denominators, jnp.finfo(covariance.dtype).tiny)
        scores = jnp.where(active & (marginal > 0.0), marginal / costs, -jnp.inf)
        if not bool(jnp.any(jnp.isfinite(scores))):
            raise ValueError(
                "Available candidates provide no positive target-fidelity information."
            )
        chosen = int(np.asarray(jnp.argmax(scores)))
        selected.append(chosen)
        selected_scores.append(scores[chosen])
        reductions.append(marginal[chosen])
        joint_index = target_count + chosen
        denominator = denominators[chosen]
        column = covariance[:, joint_index]
        covariance = covariance - column[:, None] * column[None, :] / denominator
        covariance = 0.5 * (covariance + covariance.T)
        active = active.at[chosen].set(False)

    final_variance = jnp.sum(policy.target_weights * jnp.diag(covariance)[:target_count])
    return FidelityAcquisitionResult(
        selected_indices=jnp.asarray(selected, dtype=jnp.int32),
        selected_level_ids=tuple(level_ids[index] for index in selected),
        scores=jnp.stack(selected_scores),
        marginal_variance_reductions=jnp.stack(reductions),
        initial_target_variance=initial_variance,
        final_target_variance=jnp.maximum(final_variance, 0.0),
        policy_id=policy.policy_id,
        model_dataset_id=model.dataset_id,
    )


def _as_points(value: ArrayLike, /) -> Array:
    points = jnp.asarray(value, dtype=float)
    if points.ndim == 1:
        points = points[:, None]
    if points.ndim != 2 or points.shape[0] == 0:
        raise ValueError("Points must have shape (point, coordinate).")
    if not bool(jnp.all(jnp.isfinite(points))):
        raise ValueError("Points must be finite.")
    return points


def _identifier(value: str, name: str, /) -> str:
    identifier = str(value)
    if not identifier:
        raise ValueError(f"{name} must be non-empty.")
    return identifier


__all__ = [
    "FidelityAcquisitionResult",
    "TargetVarianceAcquisitionPolicy",
    "select_fidelity_acquisition",
]
