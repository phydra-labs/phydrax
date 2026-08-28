#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from blackjax.ns.base import NSInfo, StateWithLogLikelihood
from blackjax.ns.utils import compute_num_live, logX
from jaxtyping import Array

from .._strict import StrictModule
from ._particle import effective_sample_size


class NestedQuadratureResult(StrictModule):
    """Order-statistic evidence and posterior weights for a completed run."""

    particles: StateWithLogLikelihood
    sort_indices: Array
    log_prior_volume: Array
    log_prior_volume_replicates: Array
    log_weight_replicates: Array
    posterior_log_weights: Array
    live_counts: Array
    log_evidence: Array
    log_evidence_replicates: Array
    log_evidence_shrinkage_std: Array
    information: Array
    posterior_effective_sample_size: Array
    valid: Array

    def __init__(
        self,
        *,
        particles: StateWithLogLikelihood,
        sort_indices: Array,
        log_prior_volume: Array,
        log_prior_volume_replicates: Array,
        log_weight_replicates: Array,
        posterior_log_weights: Array,
        live_counts: Array,
        log_evidence: Array,
        log_evidence_replicates: Array,
        log_evidence_shrinkage_std: Array,
        information: Array,
        posterior_effective_sample_size: Array,
        valid: Array,
    ):
        self.particles = particles
        self.sort_indices = jnp.asarray(sort_indices, dtype=jnp.int32)
        self.log_prior_volume = jnp.asarray(log_prior_volume)
        self.log_prior_volume_replicates = jnp.asarray(log_prior_volume_replicates)
        self.log_weight_replicates = jnp.asarray(log_weight_replicates)
        self.posterior_log_weights = jnp.asarray(posterior_log_weights)
        self.live_counts = jnp.asarray(live_counts, dtype=jnp.int32)
        self.log_evidence = jnp.asarray(log_evidence)
        self.log_evidence_replicates = jnp.asarray(log_evidence_replicates)
        self.log_evidence_shrinkage_std = jnp.asarray(log_evidence_shrinkage_std)
        self.information = jnp.asarray(information)
        self.posterior_effective_sample_size = jnp.asarray(
            posterior_effective_sample_size
        )
        self.valid = jnp.asarray(valid, dtype=bool)


def compute_nested_quadrature(
    particles: StateWithLogLikelihood,
    key: Array,
    /,
    *,
    num_replicates: int,
) -> NestedQuadratureResult:
    """Recompute evidence and posterior mass from complete birth/death records."""
    replicates = int(num_replicates)
    if replicates < 2:
        raise ValueError("num_replicates must be at least two.")
    likelihood = jnp.asarray(particles.loglikelihood)
    if likelihood.ndim != 1 or int(likelihood.size) < 2:
        raise ValueError("Nested quadrature requires at least two particles.")
    if bool(jnp.any(jnp.isnan(likelihood))) or bool(jnp.any(jnp.isposinf(likelihood))):
        raise ValueError("Nested likelihoods cannot contain NaN or positive infinity.")

    order = jnp.argsort(likelihood, stable=True)
    sorted_particles = jax.tree.map(lambda value: value[order], particles)
    info = NSInfo(sorted_particles, None)
    log_volume_replicates, log_volume_elements = logX(
        key,
        info,
        shape=replicates,
    )
    log_weight_replicates = log_volume_elements + sorted_particles.loglikelihood[:, None]
    log_evidence_replicates = jsp.special.logsumexp(
        log_weight_replicates,
        axis=0,
    )
    normalized_replicates = log_weight_replicates - log_evidence_replicates[None, :]
    posterior_log_weights = jsp.special.logsumexp(
        normalized_replicates, axis=1
    ) - jnp.log(jnp.asarray(replicates, dtype=normalized_replicates.dtype))
    posterior_log_weights = posterior_log_weights - jsp.special.logsumexp(
        posterior_log_weights
    )
    log_evidence = jnp.mean(log_evidence_replicates)
    posterior_weights = jnp.exp(posterior_log_weights)
    information = jnp.sum(
        jnp.where(
            posterior_weights > 0.0,
            posterior_weights * (sorted_particles.loglikelihood - log_evidence),
            0.0,
        )
    )
    live_counts = compute_num_live(info)
    posterior_ess = effective_sample_size(posterior_log_weights)
    valid = (
        jnp.isfinite(log_evidence)
        & jnp.isfinite(information)
        & jnp.isfinite(posterior_ess)
        & jnp.all(jnp.isfinite(log_evidence_replicates))
        & jnp.all(~jnp.isnan(posterior_log_weights))
        & jnp.isclose(jsp.special.logsumexp(posterior_log_weights), 0.0)
        & jnp.all(live_counts > 0)
    )
    return NestedQuadratureResult(
        particles=sorted_particles,
        sort_indices=order,
        log_prior_volume=jnp.mean(log_volume_replicates, axis=1),
        log_prior_volume_replicates=log_volume_replicates,
        log_weight_replicates=log_weight_replicates,
        posterior_log_weights=posterior_log_weights,
        live_counts=live_counts,
        log_evidence=log_evidence,
        log_evidence_replicates=log_evidence_replicates,
        log_evidence_shrinkage_std=jnp.std(log_evidence_replicates),
        information=information,
        posterior_effective_sample_size=posterior_ess,
        valid=valid,
    )


__all__ = ["NestedQuadratureResult", "compute_nested_quadrature"]
