#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._dem import DEMDiagnostics
from ._dem_sensitivity import (
    dem_local_validity_certificate,
    DEMLocalValidityCertificate,
    DEMSensitivityPolicy,
)


class DEMInverseProblem(StrictModule, NonTrainableState):
    forward_case: Callable[[PyTree[Any], PyTree[Any]], tuple[Array, DEMDiagnostics]]
    observations: Array
    observation_mask: Array
    sensitivity_policy: DEMSensitivityPolicy
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        forward_case: Callable[[PyTree[Any], PyTree[Any]], tuple[Array, DEMDiagnostics]],
        observations: Array,
        observation_mask: Array,
        sensitivity_policy: DEMSensitivityPolicy,
        /,
        *,
        problem_id: str,
    ):
        if not callable(forward_case):
            raise TypeError("forward_case must be callable.")
        observation = jnp.asarray(observations)
        mask = jnp.asarray(observation_mask, dtype=bool)
        if observation.shape != mask.shape or observation.ndim < 1:
            raise ValueError("Observations and masks must have matching batched shape.")
        identifier = str(problem_id)
        if not identifier:
            raise ValueError("problem_id must be nonempty.")
        self.forward_case = forward_case
        self.observations = observation
        self.observation_mask = mask
        self.sensitivity_policy = sensitivity_policy
        self.problem_id = canonical_fingerprint(
            {
                "kind": "dem-inverse-problem",
                "user_id": identifier,
                "observation_shape": list(observation.shape),
                "sensitivity": sensitivity_policy.policy_id,
            }
        )


class DEMInverseQualification(StrictModule):
    singular_values: Array
    condition_number: Array
    rank: Array
    locally_valid_fraction: Array
    identifiable: Array


class DEMInverseResult(StrictModule):
    loss: Array
    predictions: Array
    gradient: Any
    certificates: DEMLocalValidityCertificate
    qualification: DEMInverseQualification
    usable: Array
    problem_id: str = eqx.field(static=True)


class DEMEnsembleResult(StrictModule):
    predictions: Array
    locally_valid: Array
    successful_fraction: Array
    mean: Array
    variance: Array


def _batched_forward(
    problem: DEMInverseProblem,
    parameters: PyTree[Any],
    cases: PyTree[Any],
    /,
):
    return jax.vmap(lambda case: problem.forward_case(parameters, case))(cases)


def evaluate_dem_inverse(
    problem: DEMInverseProblem,
    parameters: PyTree[Any],
    cases: PyTree[Any],
    /,
    *,
    rank_tolerance: float = 1.0e-8,
) -> DEMInverseResult:
    predictions, diagnostics = _batched_forward(problem, parameters, cases)
    if predictions.shape != problem.observations.shape:
        raise ValueError("Forward predictions do not match observation shape.")
    mask = problem.observation_mask
    residual = jnp.where(mask, predictions - problem.observations, 0.0)
    count = jnp.maximum(jnp.sum(mask), 1)
    loss = 0.5 * jnp.sum(residual**2) / count

    def loss_function(value):
        prediction, _ = _batched_forward(problem, value, cases)
        error = jnp.where(mask, prediction - problem.observations, 0.0)
        return 0.5 * jnp.sum(error**2) / count

    gradient = jax.grad(loss_function)(parameters)
    certificates = jax.vmap(
        lambda diagnostic: dem_local_validity_certificate(
            diagnostic, problem.sensitivity_policy
        )
    )(diagnostics)
    locally_valid = certificates.locally_valid
    valid_fraction = jnp.mean(locally_valid.astype(predictions.dtype))
    jacobian_tree = jax.jacrev(lambda value: _batched_forward(problem, value, cases)[0])(
        parameters
    )
    leaves = jax.tree.leaves(jacobian_tree)
    flattened = jnp.concatenate(
        [leaf.reshape((predictions.size, -1)) for leaf in leaves], axis=1
    )
    masked_rows = problem.observation_mask.reshape(-1)
    flattened = flattened * masked_rows[:, None]
    singular = jnp.linalg.svd(flattened, compute_uv=False)
    largest = jnp.max(singular, initial=0.0)
    threshold = rank_tolerance * jnp.maximum(largest, 1.0)
    rank = jnp.sum(singular > threshold, dtype=jnp.int32)
    smallest = jnp.min(jnp.where(singular > threshold, singular, jnp.inf))
    condition = largest / jnp.where(jnp.isfinite(smallest), smallest, 1.0)
    parameter_count = flattened.shape[1]
    identifiable = rank == parameter_count
    usable = jnp.all(locally_valid) & identifiable & jnp.isfinite(loss)
    invalid_gradient = jax.tree.map(
        lambda leaf: jnp.where(usable, leaf, jnp.full_like(leaf, jnp.nan)),
        gradient,
    )
    qualification = DEMInverseQualification(
        singular,
        condition,
        rank,
        valid_fraction,
        identifiable,
    )
    return DEMInverseResult(
        loss,
        predictions,
        invalid_gradient,
        certificates,
        qualification,
        usable,
        problem.problem_id,
    )


def evaluate_dem_parameter_ensemble(
    problem: DEMInverseProblem,
    parameter_samples: PyTree[Any],
    cases: PyTree[Any],
    /,
) -> DEMEnsembleResult:
    predictions, diagnostics = jax.vmap(
        lambda parameters: _batched_forward(problem, parameters, cases)
    )(parameter_samples)
    certificates = jax.vmap(
        jax.vmap(
            lambda diagnostic: dem_local_validity_certificate(
                diagnostic, problem.sensitivity_policy
            )
        )
    )(diagnostics)
    valid = certificates.locally_valid
    sample_valid = jnp.all(valid, axis=-1)
    expand = sample_valid.reshape(
        sample_valid.shape + (1,) * (predictions.ndim - sample_valid.ndim)
    )
    weights = expand.astype(predictions.dtype)
    count = jnp.maximum(jnp.sum(weights, axis=0), 1.0)
    mean = jnp.sum(jnp.where(expand, predictions, 0.0), axis=0) / count
    variance = jnp.sum(jnp.where(expand, (predictions - mean) ** 2, 0.0), axis=0) / count
    return DEMEnsembleResult(
        predictions,
        valid,
        jnp.mean(sample_valid.astype(predictions.dtype)),
        mean,
        variance,
    )


__all__ = [
    "DEMEnsembleResult",
    "DEMInverseProblem",
    "DEMInverseQualification",
    "DEMInverseResult",
    "evaluate_dem_inverse",
    "evaluate_dem_parameter_ensemble",
]
