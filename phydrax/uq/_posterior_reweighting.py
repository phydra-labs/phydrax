#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PyTree

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..integration import WeightedSampleTarget
from ._particle import effective_sample_size, normalize_log_weights, resample_indices


def _raw(value: Array | cx.Field | None, /) -> Array | None:
    if value is None:
        return None
    return jnp.asarray(value.data if isinstance(value, cx.Field) else value)


def _flatten_target(
    target: WeightedSampleTarget, /
) -> tuple[PyTree[Array], Array, Array, tuple[int, ...]]:
    if not isinstance(target, WeightedSampleTarget):
        raise TypeError("target must be WeightedSampleTarget.")
    weights = _raw(target.log_weights)
    if weights is None:
        raise RuntimeError("WeightedSampleTarget lost its log weights.")
    shape = tuple(int(size) for size in weights.shape)
    if not shape:
        raise ValueError("Posterior weights require at least one sample axis.")
    if isinstance(target.log_weights, cx.Field):
        if tuple(target.log_weights.named_dims) != tuple(target.sample_axes):
            raise ValueError(
                "Named posterior weights must consist exactly of sample axes."
            )
    else:
        expected_axes = tuple(range(weights.ndim))
        if tuple(target.sample_axes) != expected_axes:
            raise ValueError(
                "Raw posterior weights must declare every leading weight axis."
            )
    mask = _raw(target.mask)
    active = (
        jnp.ones(shape, dtype=bool) if mask is None else jnp.asarray(mask, dtype=bool)
    )
    if active.shape != shape or not bool(jnp.any(active)):
        raise ValueError("Posterior mask must select at least one weighted sample.")
    count = int(np.prod(shape))

    def flatten(value):
        array = jnp.asarray(value)
        if array.ndim < len(shape) or tuple(array.shape[: len(shape)]) != shape:
            raise ValueError(
                "Posterior sample leaves must begin with the complete weight shape."
            )
        return array.reshape((count, *array.shape[len(shape) :]))

    samples = jax.tree_util.tree_map(flatten, target.samples)
    return samples, weights.reshape((count,)), active.reshape((count,)), shape


def _restore_log_weights(
    target: WeightedSampleTarget, values: Array, shape: tuple[int, ...], /
):
    restored = values.reshape(shape)
    if isinstance(target.log_weights, cx.Field):
        return cx.Field(restored, dims=target.log_weights.dims)
    return restored


class PosteriorReweightingPolicy(StrictModule):
    minimum_effective_sample_size: float = eqx.field(static=True)
    minimum_effective_sample_fraction: float = eqx.field(static=True)

    def __init__(
        self,
        minimum_effective_sample_size: float = 20.0,
        minimum_effective_sample_fraction: float = 0.01,
    ):
        count = float(minimum_effective_sample_size)
        fraction = float(minimum_effective_sample_fraction)
        if not np.isfinite(count) or count <= 0.0 or not 0.0 < fraction <= 1.0:
            raise ValueError("Posterior reweighting ESS policy is invalid.")
        self.minimum_effective_sample_size = count
        self.minimum_effective_sample_fraction = fraction


class PosteriorReweightingPlan(StrictModule):
    old_log_density: Callable[[PyTree[Any]], Array]
    new_log_density: Callable[[PyTree[Any]], Array]
    policy: PosteriorReweightingPolicy
    old_target_id: str = eqx.field(static=True)
    new_target_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        old_log_density: Callable[[PyTree[Any]], Array],
        new_log_density: Callable[[PyTree[Any]], Array],
        /,
        *,
        old_target_id: str,
        new_target_id: str,
        policy: PosteriorReweightingPolicy | None = None,
    ):
        if not callable(old_log_density) or not callable(new_log_density):
            raise TypeError("Posterior target densities must be callable.")
        old_id, new_id = str(old_target_id).strip(), str(new_target_id).strip()
        if not old_id or not new_id or old_id == new_id:
            raise ValueError("Old and new target IDs must be distinct and non-empty.")
        policy_ = PosteriorReweightingPolicy() if policy is None else policy
        if not isinstance(policy_, PosteriorReweightingPolicy):
            raise TypeError("policy must be PosteriorReweightingPolicy.")
        self.old_log_density = old_log_density
        self.new_log_density = new_log_density
        self.policy = policy_
        self.old_target_id = old_id
        self.new_target_id = new_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "posterior-reweighting-plan",
                "old_target": old_id,
                "new_target": new_id,
                "minimum_ess": policy_.minimum_effective_sample_size,
                "minimum_fraction": policy_.minimum_effective_sample_fraction,
            }
        )


class PosteriorReweightingResult(StrictModule):
    target: WeightedSampleTarget
    old_log_density: Array
    new_log_density: Array
    log_normalizer_ratio: Array
    importance_effective_sample_size: Array
    effective_sample_fraction: Array
    maximum_normalized_weight: Array
    support_loss_count: Array
    valid: Array
    status: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def resample(self, key: Array, /, *, num_samples: int) -> PyTree[Array]:
        count = int(num_samples)
        if count <= 0:
            raise ValueError("num_samples must be positive.")
        samples, weights, _, _ = _flatten_target(self.target)
        if count == int(weights.size):
            indices = resample_indices(key, weights, method="systematic")
        else:
            normalized, _, valid = normalize_log_weights(weights)
            normalized = eqx.error_if(
                normalized, ~valid, "Cannot resample invalid weights."
            )
            indices = jax.random.choice(
                key,
                int(weights.size),
                shape=(count,),
                p=jnp.exp(normalized),
                replace=True,
            )
        return jax.tree_util.tree_map(lambda value: value[indices], samples)


def reweight_posterior(
    target: WeightedSampleTarget,
    plan: PosteriorReweightingPlan,
    /,
) -> PosteriorReweightingResult:
    if not isinstance(plan, PosteriorReweightingPlan):
        raise TypeError("plan must be PosteriorReweightingPlan.")
    samples, raw_weights, active, shape = _flatten_target(target)
    source_weights, _, source_valid = normalize_log_weights(
        jnp.where(active, raw_weights, -jnp.inf)
    )
    old = jax.vmap(plan.old_log_density)(samples)
    new = jax.vmap(plan.new_log_density)(samples)
    if old.shape != raw_weights.shape or new.shape != raw_weights.shape:
        raise ValueError(
            "Posterior target densities must return one scalar per weighted sample."
        )
    source_mass = active & jnp.isfinite(source_weights)
    old_usable = source_mass & jnp.isfinite(old)
    new_usable = source_mass & jnp.isfinite(new)
    old_support_valid = jnp.all(~source_mass | jnp.isfinite(old))
    new_density_valid = jnp.all(~source_mass | (~jnp.isnan(new) & ~jnp.isposinf(new)))
    delta = jnp.where(old_usable & new_usable, new - old, -jnp.inf)
    candidates = source_weights + delta
    normalized, log_ratio, weight_valid = normalize_log_weights(candidates)
    ess = effective_sample_size(normalized)
    source_count = jnp.sum(source_mass)
    fraction = ess / jnp.maximum(source_count, 1)
    maximum = jnp.max(jnp.exp(normalized))
    support_loss = jnp.sum(old_usable & jnp.isneginf(new))
    valid = (
        source_valid
        & old_support_valid
        & new_density_valid
        & weight_valid
        & (ess >= plan.policy.minimum_effective_sample_size)
        & (fraction >= plan.policy.minimum_effective_sample_fraction)
    )
    status = (
        "success"
        if bool(valid)
        else "source-invalid"
        if not bool(source_valid)
        else "source-support-invalid"
        if not bool(old_support_valid)
        else "target-density-invalid"
        if not bool(new_density_valid)
        else "target-support-empty"
        if not bool(weight_valid)
        else "insufficient-overlap"
    )
    restored_weights = _restore_log_weights(target, normalized, shape)
    reweighted = WeightedSampleTarget(
        target.samples,
        restored_weights,
        normalized=True,
        independent=target.independent,
        ancestry=target.ancestry,
        support_valid=valid,
        stratum_ids=target.stratum_ids,
        pair_ids=target.pair_ids,
        replicate_ids=target.replicate_ids,
        mask=target.mask,
        sample_axes=target.sample_axes,
        provenance=f"posterior-reweighting:{plan.plan_id}",
    )
    return PosteriorReweightingResult(
        reweighted,
        old.reshape(shape),
        new.reshape(shape),
        log_ratio,
        ess,
        fraction,
        maximum,
        support_loss,
        valid,
        status,
        plan.plan_id,
    )


__all__ = [
    "PosteriorReweightingPlan",
    "PosteriorReweightingPolicy",
    "PosteriorReweightingResult",
    "reweight_posterior",
]
