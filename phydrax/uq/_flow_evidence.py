#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, PyTree

from .._sampling import derive_key, SampleAddress
from .._strict import StrictModule
from ._flow_mcmc import FlowNUTSResult
from ._posterior import PosteriorProblem
from ._smc import sample_tempered_smc, TemperedSMCResult


class FlowNUTSEvidenceResult(StrictModule):
    """Overlap-gated generalized bridge evidence from frozen production draws."""

    log_evidence: Array
    bridge_residual: Array
    posterior_overlap_ess: Array
    proposal_overlap_ess: Array
    jackknife_standard_error: Array
    nonfinite_count: Array
    iterations: Array
    valid: Array
    status: str = eqx.field(static=True)
    block_length: int = eqx.field(static=True)
    num_posterior_samples: int = eqx.field(static=True)
    num_proposal_samples: int = eqx.field(static=True)
    approximation: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        log_evidence: Array,
        bridge_residual: Array,
        posterior_overlap_ess: Array,
        proposal_overlap_ess: Array,
        jackknife_standard_error: Array,
        nonfinite_count: Array,
        iterations: int,
        valid: Array,
        status: str,
        block_length: int,
        num_posterior_samples: int,
        num_proposal_samples: int,
    ):
        self.log_evidence = jnp.asarray(log_evidence).reshape(())
        self.bridge_residual = jnp.asarray(bridge_residual).reshape(())
        self.posterior_overlap_ess = jnp.asarray(posterior_overlap_ess).reshape(())
        self.proposal_overlap_ess = jnp.asarray(proposal_overlap_ess).reshape(())
        self.jackknife_standard_error = jnp.asarray(jackknife_standard_error).reshape(())
        self.nonfinite_count = jnp.asarray(nonfinite_count, dtype=jnp.int32).reshape(())
        self.iterations = jnp.asarray(iterations, dtype=jnp.int32).reshape(())
        self.valid = jnp.asarray(valid, dtype=bool).reshape(())
        self.status = str(status)
        self.block_length = int(block_length)
        self.num_posterior_samples = int(num_posterior_samples)
        self.num_proposal_samples = int(num_proposal_samples)
        self.approximation = "overlap_gated_generalized_bridge"


class FlowNUTSModeInitialization(StrictModule):
    """Tempered-SMC-owned diverse initialization with explicit selection evidence."""

    initial_positions: PyTree[Array]
    selected_indices: Array
    selected_weights: Array
    nearest_selected_distances: Array
    ancestry: Array
    represented_clusters: Array
    duplicate_count: Array
    selection: str = eqx.field(static=True)
    missed_mode_certification: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        initial_positions: PyTree[Array],
        selected_indices: Array,
        selected_weights: Array,
        nearest_selected_distances: Array,
    ):
        indices = jnp.asarray(selected_indices, dtype=jnp.int32)
        self.initial_positions = initial_positions
        self.selected_indices = indices
        self.selected_weights = jnp.asarray(selected_weights)
        self.nearest_selected_distances = jnp.asarray(nearest_selected_distances)
        self.ancestry = indices
        self.represented_clusters = jnp.arange(indices.size, dtype=jnp.int32)
        self.duplicate_count = indices.size - jnp.unique(indices).size
        self.selection = "weighted-farthest"
        self.missed_mode_certification = False


def estimate_flow_nuts_evidence(
    result: FlowNUTSResult,
    /,
    *,
    key: Array,
    num_proposal_samples: int,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    minimum_overlap_ess: float = 20.0,
    block_length: int = 16,
) -> FlowNUTSEvidenceResult:
    """Estimate evidence separately from Flow-NUTS sampling and adaptation."""
    if not isinstance(result, FlowNUTSResult):
        raise TypeError("result must be a FlowNUTSResult.")
    proposals = int(num_proposal_samples)
    iterations = int(max_iterations)
    block = int(block_length)
    tolerance_ = float(tolerance)
    minimum_ess = float(minimum_overlap_ess)
    if proposals < 2 or iterations <= 0 or block <= 0:
        raise ValueError("Proposal/iteration/block capacities must be positive.")
    if not math.isfinite(tolerance_) or tolerance_ <= 0.0:
        raise ValueError("tolerance must be finite and positive.")
    if not math.isfinite(minimum_ess) or minimum_ess <= 0.0:
        raise ValueError("minimum_overlap_ess must be finite and positive.")
    posterior_positions = _flatten_sample_tree(result.unconstrained_samples)
    posterior_log_f = jnp.asarray(result.log_density).reshape((-1,))
    posterior_log_q = jnp.asarray(result.flow.log_prob(posterior_positions))
    address = SampleAddress(
        "uq.flow-nuts", "bridge-proposal", target="heldout-flow", role="evidence"
    )
    proposal_positions, proposal_log_q = result.flow.sample_and_log_prob(
        derive_key(key, address, proposals),
        sample_shape=(proposals,),
    )
    _, unravel = ravel_pytree(result.problem.initial_position)
    proposal_log_f = jax.vmap(lambda value: result.problem.log_density(unravel(value)))(
        proposal_positions
    )
    finite_posterior = jnp.isfinite(posterior_log_f) & jnp.isfinite(posterior_log_q)
    finite_proposal = jnp.isfinite(proposal_log_f) & jnp.isfinite(proposal_log_q)
    nonfinite_count = jnp.sum(~finite_posterior) + jnp.sum(~finite_proposal)
    if not bool(jnp.all(finite_posterior)) or not bool(jnp.all(finite_proposal)):
        return FlowNUTSEvidenceResult(
            log_evidence=jnp.nan,
            bridge_residual=jnp.inf,
            posterior_overlap_ess=0.0,
            proposal_overlap_ess=0.0,
            jackknife_standard_error=jnp.nan,
            nonfinite_count=nonfinite_count,
            iterations=0,
            valid=False,
            status="nonfinite",
            block_length=block,
            num_posterior_samples=posterior_positions.shape[0],
            num_proposal_samples=proposals,
        )
    initial_log_z = _logmeanexp(proposal_log_f - proposal_log_q)
    log_z, residual, used = _bridge_fixed_point(
        posterior_log_f,
        posterior_log_q,
        proposal_log_f,
        proposal_log_q,
        initial_log_z=initial_log_z,
        max_iterations=iterations,
        tolerance=tolerance_,
    )
    posterior_weights, proposal_weights = _bridge_weights(
        posterior_log_f,
        posterior_log_q,
        proposal_log_f,
        proposal_log_q,
        log_z,
    )
    posterior_ess = _ess(posterior_weights)
    proposal_ess = _ess(proposal_weights)
    standard_error = _blocked_jackknife_standard_error(
        posterior_log_f,
        posterior_log_q,
        proposal_log_f,
        proposal_log_q,
        log_z,
        block_length=block,
        max_iterations=iterations,
        tolerance=tolerance_,
    )
    converged = residual <= tolerance_
    overlap = (posterior_ess >= minimum_ess) & (proposal_ess >= minimum_ess)
    valid = converged & overlap & jnp.isfinite(standard_error)
    status = (
        "success"
        if bool(valid)
        else "low-overlap"
        if not bool(overlap)
        else "nonconverged"
    )
    return FlowNUTSEvidenceResult(
        log_evidence=log_z,
        bridge_residual=residual,
        posterior_overlap_ess=posterior_ess,
        proposal_overlap_ess=proposal_ess,
        jackknife_standard_error=standard_error,
        nonfinite_count=nonfinite_count,
        iterations=used,
        valid=valid,
        status=status,
        block_length=block,
        num_posterior_samples=posterior_positions.shape[0],
        num_proposal_samples=proposals,
    )


def initialize_flow_nuts_from_tempered_smc(
    problem: PosteriorProblem,
    /,
    *,
    key: Array,
    num_chains: int,
    smc_result: TemperedSMCResult | None = None,
    smc_kwargs: dict[str, Any] | None = None,
) -> FlowNUTSModeInitialization:
    """Use tempered SMC for optional mode discovery without certification claims."""
    if not isinstance(problem, PosteriorProblem):
        raise TypeError("problem must be a PosteriorProblem.")
    chains = int(num_chains)
    if chains <= 0:
        raise ValueError("num_chains must be positive.")
    if smc_result is not None and smc_kwargs is not None:
        raise ValueError("Supply smc_result or smc_kwargs, not both.")
    address = SampleAddress(
        "uq.flow-nuts", "smc-mode-initialization", target="tempered-smc", role="mode"
    )
    if smc_result is None:
        kwargs = {} if smc_kwargs is None else dict(smc_kwargs)
        smc_result = sample_tempered_smc(
            problem,
            key=derive_key(key, address, 0),
            **kwargs,
        )
    if not isinstance(smc_result, TemperedSMCResult):
        raise TypeError("smc_result must be a TemperedSMCResult.")
    if smc_result.problem is not problem:
        raise ValueError("Tempered SMC result belongs to a different posterior problem.")
    flat = _flatten_particle_tree(smc_result.unconstrained_samples)
    weights = jnp.asarray(smc_result.final_weights)
    if flat.shape[0] < chains:
        raise ValueError("Tempered SMC must contain at least num_chains particles.")
    if (
        weights.shape != (flat.shape[0],)
        or bool(jnp.any(weights < 0.0))
        or not bool(jnp.isclose(jnp.sum(weights), 1.0))
    ):
        raise ValueError("Tempered SMC final weights are invalid.")
    first = jr.choice(derive_key(key, address, 1), flat.shape[0], p=weights)
    selected = [int(first)]
    distances = [jnp.asarray(jnp.inf, dtype=flat.dtype)]
    for _ in range(1, chains):
        chosen = flat[jnp.asarray(selected)]
        squared = jnp.sum((flat[:, None, :] - chosen[None, :, :]) ** 2, axis=-1)
        nearest = jnp.sqrt(jnp.min(squared, axis=1))
        nearest = nearest.at[jnp.asarray(selected)].set(-jnp.inf)
        index = int(jnp.argmax(weights * jnp.maximum(nearest, 0.0)))
        selected.append(index)
        distances.append(nearest[index])
    indices = jnp.asarray(selected, dtype=jnp.int32)
    initial = jax.tree_util.tree_map(
        lambda leaf: leaf[indices], smc_result.unconstrained_samples
    )
    return FlowNUTSModeInitialization(
        initial_positions=initial,
        selected_indices=indices,
        selected_weights=weights[indices],
        nearest_selected_distances=jnp.stack(distances),
    )


def _bridge_fixed_point(
    posterior_log_f: Array,
    posterior_log_q: Array,
    proposal_log_f: Array,
    proposal_log_q: Array,
    /,
    *,
    initial_log_z: Array,
    max_iterations: int,
    tolerance: float,
) -> tuple[Array, Array, int]:
    log_z = initial_log_z
    residual = jnp.asarray(jnp.inf, dtype=log_z.dtype)
    used = 0
    for iteration in range(max_iterations):
        posterior_weight, proposal_weight = _bridge_weights(
            posterior_log_f,
            posterior_log_q,
            proposal_log_f,
            proposal_log_q,
            log_z,
        )
        next_log_z = _logmeanexp(jnp.log(proposal_weight)) - _logmeanexp(
            jnp.log(posterior_weight)
        )
        residual = jnp.abs(next_log_z - log_z)
        log_z = next_log_z
        used = iteration + 1
        if bool(residual <= tolerance):
            break
    return log_z, residual, used


def _bridge_weights(
    posterior_log_f: Array,
    posterior_log_q: Array,
    proposal_log_f: Array,
    proposal_log_q: Array,
    log_z: Array,
    /,
) -> tuple[Array, Array]:
    posterior_count = posterior_log_f.size
    proposal_count = proposal_log_f.size
    total = posterior_count + proposal_count
    log_s_p = jnp.log(jnp.asarray(posterior_count / total))
    log_s_q = jnp.log(jnp.asarray(proposal_count / total))
    posterior_denominator = jnp.logaddexp(
        log_s_p + posterior_log_f,
        log_s_q + log_z + posterior_log_q,
    )
    proposal_denominator = jnp.logaddexp(
        log_s_p + proposal_log_f,
        log_s_q + log_z + proposal_log_q,
    )
    posterior_weight = jnp.exp(posterior_log_q - posterior_denominator)
    proposal_weight = jnp.exp(proposal_log_f - proposal_denominator)
    return posterior_weight, proposal_weight


def _blocked_jackknife_standard_error(
    posterior_log_f: Array,
    posterior_log_q: Array,
    proposal_log_f: Array,
    proposal_log_q: Array,
    log_z: Array,
    /,
    *,
    block_length: int,
    max_iterations: int,
    tolerance: float,
) -> Array:
    posterior_blocks = posterior_log_f.size // block_length
    proposal_blocks = proposal_log_f.size // block_length
    block_count = min(posterior_blocks, proposal_blocks)
    if block_count < 2:
        return jnp.asarray(jnp.nan, dtype=log_z.dtype)
    estimates = []
    for block in range(block_count):
        p_mask = (
            jnp.ones((posterior_log_f.size,), dtype=bool)
            .at[block * block_length : (block + 1) * block_length]
            .set(False)
        )
        q_mask = (
            jnp.ones((proposal_log_f.size,), dtype=bool)
            .at[block * block_length : (block + 1) * block_length]
            .set(False)
        )
        estimate, _, _ = _bridge_fixed_point(
            posterior_log_f[p_mask],
            posterior_log_q[p_mask],
            proposal_log_f[q_mask],
            proposal_log_q[q_mask],
            initial_log_z=log_z,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        estimates.append(estimate)
    values = jnp.stack(estimates)
    return jnp.sqrt((block_count - 1) * jnp.var(values, ddof=0))


def _ess(weights: Array, /) -> Array:
    return jnp.sum(weights) ** 2 / jnp.sum(weights**2)


def _logmeanexp(values: Array, /) -> Array:
    maximum = jnp.max(values)
    return maximum + jnp.log(jnp.mean(jnp.exp(values - maximum)))


def _flatten_sample_tree(samples: PyTree[Array], /) -> Array:
    leaves = jax.tree_util.tree_leaves(samples)
    if not leaves or any(leaf.ndim < 2 for leaf in leaves):
        raise ValueError("Flow-NUTS samples must retain chain and draw axes.")
    leading = leaves[0].shape[:2]
    if any(leaf.shape[:2] != leading for leaf in leaves):
        raise ValueError("Flow-NUTS sample leaves must share chain/draw axes.")
    return jnp.concatenate(
        tuple(leaf.reshape((leading[0] * leading[1], -1)) for leaf in leaves),
        axis=1,
    )


def _flatten_particle_tree(samples: PyTree[Array], /) -> Array:
    leaves = jax.tree_util.tree_leaves(samples)
    if not leaves or any(leaf.ndim < 1 for leaf in leaves):
        raise ValueError("Tempered SMC samples must retain a particle axis.")
    count = leaves[0].shape[0]
    if any(leaf.shape[0] != count for leaf in leaves):
        raise ValueError("Tempered SMC leaves must share a particle axis.")
    return jnp.concatenate(tuple(leaf.reshape((count, -1)) for leaf in leaves), axis=1)


__all__ = [
    "FlowNUTSEvidenceResult",
    "FlowNUTSModeInitialization",
    "estimate_flow_nuts_evidence",
    "initialize_flow_nuts_from_tempered_smc",
]
