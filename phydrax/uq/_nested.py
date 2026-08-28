#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, TypeAlias

import blackjax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from blackjax.ns.base import NSInfo, StateWithLogLikelihood
from blackjax.ns.utils import finalise
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, PyTree

from .._frozendict import frozendict
from .._probability import AbstractProbabilityLaw
from .._sampling._addressing import derive_key, SampleAddress
from .._strict import StrictModule
from ..integration import WeightedSampleTarget
from ._checkpoint import (
    checkpoint_compatibility,
    pack_array_tree,
    read_checkpoint_archive,
    unpack_array_tree,
    write_checkpoint_archive,
)
from ._nested_diagnostics import build_nested_diagnostics, NestedSamplingDiagnostics
from ._nested_quadrature import compute_nested_quadrature
from ._particle import resample_indices
from ._posterior import PosteriorProblem
from ._posterior_predictive import (
    predict_from_position_samples,
    sample_observations_from_position_samples,
)
from ._predictive import PredictiveField


NestedSamplingMethod: TypeAlias = Literal["hit-and-run", "slice-within-gibbs"]
NestedSamplingStatus: TypeAlias = Literal[0, 1, 2, 3, 4, 5, 6]

NESTED_SAMPLING_SUCCESS: NestedSamplingStatus = 0
NESTED_SAMPLING_MAX_DEAD_POINTS: NestedSamplingStatus = 1
NESTED_SAMPLING_MAX_LIKELIHOOD_EVALUATIONS: NestedSamplingStatus = 2
NESTED_SAMPLING_NO_FINITE_LIVE_POINT: NestedSamplingStatus = 3
NESTED_SAMPLING_LIKELIHOOD_PLATEAU: NestedSamplingStatus = 4
NESTED_SAMPLING_INVALID_LIKELIHOOD: NestedSamplingStatus = 5
NESTED_SAMPLING_INNER_KERNEL_FAILURE: NestedSamplingStatus = 6

_CHECKPOINT_KIND = "nested_sampling"
_PRIOR_ADDRESS = SampleAddress("phydrax.uq", "nested-prior", role="live-points")
_INIT_ADDRESS = SampleAddress("phydrax.uq", "nested-init", role="kernel")
_STEP_ADDRESS = SampleAddress("phydrax.uq", "nested-step", role="kernel")
_VOLUME_ADDRESS = SampleAddress("phydrax.uq", "nested-volume", role="quadrature")
_RESAMPLE_ADDRESS = SampleAddress("phydrax.uq", "nested-resample", role="posterior")


def nested_sampling_status_name(value: int, /) -> str:
    """Return the stable name of one nested-sampling status code."""
    names = (
        "success",
        "maximum_dead_points",
        "maximum_likelihood_evaluations",
        "no_finite_live_point",
        "likelihood_plateau",
        "invalid_likelihood",
        "inner_kernel_failure",
    )
    code = int(value)
    if code < 0 or code >= len(names):
        raise ValueError(f"Unknown nested-sampling status code {code}.")
    return names[code]


class NestedSamplingResult(StrictModule):
    """Weighted nested quadrature, evidence, diagnostics, and live state."""

    problem: PosteriorProblem
    samples: PyTree[Array]
    unconstrained_samples: PyTree[Array]
    log_prior: Array
    log_likelihood: Array
    birth_log_likelihood: Array
    posterior_log_weights: Array
    log_prior_volume: Array
    live_counts: Array
    sample_ids: Array
    batch_indices: Array
    log_evidence: Array
    log_evidence_replicates: Array
    log_evidence_shrinkage_std: Array
    information: Array
    posterior_effective_sample_size: Array
    remaining_log_evidence: Array
    remaining_evidence_fraction: Array
    final_state: Any
    diagnostics: NestedSamplingDiagnostics
    root_key: Array
    status: Array
    valid: Array
    num_live: int = eqx.field(static=True)
    num_dead: int = eqx.field(static=True)
    num_likelihood_evaluations: int = eqx.field(static=True)
    num_inner_steps: int = eqx.field(static=True)
    num_delete: int = eqx.field(static=True)
    method: str = eqx.field(static=True)
    duration_seconds: float = eqx.field(static=True)
    sample_memory_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        problem: PosteriorProblem,
        samples: PyTree[Array],
        unconstrained_samples: PyTree[Array],
        log_prior: Array,
        log_likelihood: Array,
        birth_log_likelihood: Array,
        posterior_log_weights: Array,
        log_prior_volume: Array,
        live_counts: Array,
        sample_ids: Array,
        batch_indices: Array,
        log_evidence: Array,
        log_evidence_replicates: Array,
        log_evidence_shrinkage_std: Array,
        information: Array,
        posterior_effective_sample_size: Array,
        remaining_log_evidence: Array,
        remaining_evidence_fraction: Array,
        final_state: Any,
        diagnostics: NestedSamplingDiagnostics,
        root_key: Array,
        status: Array,
        valid: Array,
        num_live: int,
        num_dead: int,
        num_likelihood_evaluations: int,
        num_inner_steps: int,
        num_delete: int,
        method: str,
        duration_seconds: float,
    ):
        self.problem = problem
        self.samples = samples
        self.unconstrained_samples = unconstrained_samples
        self.log_prior = jnp.asarray(log_prior)
        self.log_likelihood = jnp.asarray(log_likelihood)
        self.birth_log_likelihood = jnp.asarray(birth_log_likelihood)
        self.posterior_log_weights = jnp.asarray(posterior_log_weights)
        self.log_prior_volume = jnp.asarray(log_prior_volume)
        self.live_counts = jnp.asarray(live_counts, dtype=jnp.int32)
        self.sample_ids = jnp.asarray(sample_ids, dtype=jnp.int32)
        self.batch_indices = jnp.asarray(batch_indices, dtype=jnp.int32)
        self.log_evidence = jnp.asarray(log_evidence)
        self.log_evidence_replicates = jnp.asarray(log_evidence_replicates)
        self.log_evidence_shrinkage_std = jnp.asarray(log_evidence_shrinkage_std)
        self.information = jnp.asarray(information)
        self.posterior_effective_sample_size = jnp.asarray(
            posterior_effective_sample_size
        )
        self.remaining_log_evidence = jnp.asarray(remaining_log_evidence)
        self.remaining_evidence_fraction = jnp.asarray(remaining_evidence_fraction)
        self.final_state = final_state
        self.diagnostics = diagnostics
        self.root_key = jnp.asarray(root_key)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.valid = jnp.asarray(valid, dtype=bool)
        self.num_live = int(num_live)
        self.num_dead = int(num_dead)
        self.num_likelihood_evaluations = int(num_likelihood_evaluations)
        self.num_inner_steps = int(num_inner_steps)
        self.num_delete = int(num_delete)
        self.method = str(method)
        self.duration_seconds = float(duration_seconds)
        self.sample_memory_bytes = _tree_nbytes(samples) + _tree_nbytes(
            unconstrained_samples
        )

    @property
    def converged(self) -> bool:
        return bool(self.valid) and int(self.status) == NESTED_SAMPLING_SUCCESS

    @property
    def num_samples(self) -> int:
        return int(self.log_likelihood.shape[0])

    def resample_posterior(
        self,
        key: Array,
        /,
        *,
        num_samples: int,
        constrained: bool = True,
    ) -> PyTree[Array]:
        """Draw equally weighted posterior samples from nested quadrature."""
        count = int(num_samples)
        if count <= 0:
            raise ValueError("num_samples must be positive.")
        indices = resample_indices(
            derive_key(key, _RESAMPLE_ADDRESS, count),
            self.posterior_log_weights,
            method="systematic",
        )
        if count != self.num_samples:
            uniform_key = derive_key(key, _RESAMPLE_ADDRESS, count, self.num_samples)
            probabilities = jnp.exp(self.posterior_log_weights)
            indices = jr.choice(
                uniform_key,
                self.num_samples,
                shape=(count,),
                p=probabilities,
                replace=True,
            )
        source = self.samples if constrained else self.unconstrained_samples
        return jax.tree.map(lambda value: value[indices], source)

    def posterior_measure(self) -> WeightedSampleTarget:
        """Expose the dependent weighted posterior as an empirical measure."""
        return WeightedSampleTarget(
            self.samples,
            self.posterior_log_weights,
            normalized=True,
            independent=False,
            ancestry=self.sample_ids,
            stratum_ids=self.batch_indices,
            sample_axes=0,
            provenance=f"nested-sampling:{self.method}",
        )

    def predict(
        self,
        *args: Any,
        batch_size: int | None = None,
        valid_policy: Literal["record", "raise"] = "record",
        **kwargs: Any,
    ) -> PredictiveField | frozendict[str, PredictiveField]:
        """Evaluate latent predictions at every weighted quadrature sample."""
        return predict_from_position_samples(
            self.problem,
            self.unconstrained_samples,
            *args,
            sample_dims=("__phydra_uq_nested",),
            sample_sources=("epistemic",),
            batch_size=batch_size,
            valid_policy=valid_policy,
            **kwargs,
        )

    def predict_observations(
        self,
        key: Array,
        /,
        *args: Any,
        num_observation_samples: int = 1,
        batch_size: int | None = None,
        valid_policy: Literal["record", "raise"] = "record",
        observation_dim: str = "__phydra_uq_observation",
        **kwargs: Any,
    ) -> PredictiveField | frozendict[str, PredictiveField]:
        """Draw conditional observations at every nested quadrature sample."""
        return sample_observations_from_position_samples(
            self.problem,
            key,
            self.unconstrained_samples,
            *args,
            sample_dims=("__phydra_uq_nested",),
            sample_sources=("epistemic",),
            num_observation_samples=num_observation_samples,
            batch_size=batch_size,
            valid_policy=valid_policy,
            observation_dim=observation_dim,
            **kwargs,
        )

    def diagnostic_report(self) -> dict[str, Any]:
        """Return machine-readable termination and constrained-sampling evidence."""
        return {
            "status": nested_sampling_status_name(int(self.status)),
            "valid": bool(self.valid),
            "converged": self.converged,
            "num_live": self.num_live,
            "num_dead": self.num_dead,
            "num_likelihood_evaluations": self.num_likelihood_evaluations,
            "log_evidence": float(self.log_evidence),
            "log_evidence_shrinkage_std": float(self.log_evidence_shrinkage_std),
            "posterior_effective_sample_size": float(
                self.posterior_effective_sample_size
            ),
            "diagnostics": self.diagnostics.as_dict(),
        }


def sample_nested(
    problem: PosteriorProblem,
    /,
    *,
    key: Array,
    num_live: int = 500,
    method: NestedSamplingMethod = "hit-and-run",
    num_inner_steps: int | None = None,
    num_delete: int = 1,
    max_expansions: int = 10,
    max_shrinkage: int = 100,
    remaining_evidence_tolerance: float = 0.01,
    max_dead_points: int = 100_000,
    max_likelihood_evaluations: int | None = None,
    num_volume_replicates: int = 256,
    prior_position_sampler: Callable[[Array, int], PyTree[Array]] | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_id: str | None = None,
    checkpoint_every: int = 100,
    resume_from: str | Path | None = None,
) -> NestedSamplingResult:
    """Run static nested slice sampling over a deterministic posterior problem."""
    if not isinstance(problem, PosteriorProblem):
        raise TypeError("problem must be a PosteriorProblem.")
    live_count = int(num_live)
    delete_count = int(num_delete)
    expansions_cap = int(max_expansions)
    shrinkage_cap = int(max_shrinkage)
    dead_limit = int(max_dead_points)
    volume_replicates = int(num_volume_replicates)
    checkpoint_interval = int(checkpoint_every)
    if live_count < 2:
        raise ValueError("num_live must be at least two.")
    if delete_count < 1 or delete_count >= live_count:
        raise ValueError("num_delete must lie in [1, num_live).")
    if expansions_cap < 1 or shrinkage_cap < 1:
        raise ValueError("max_expansions and max_shrinkage must be positive.")
    if dead_limit < 1:
        raise ValueError("max_dead_points must be positive.")
    if volume_replicates < 2:
        raise ValueError("num_volume_replicates must be at least two.")
    if checkpoint_interval < 1:
        raise ValueError("checkpoint_every must be positive.")
    tolerance = float(remaining_evidence_tolerance)
    if not jnp.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("remaining_evidence_tolerance must be finite and positive.")
    likelihood_limit = (
        None if max_likelihood_evaluations is None else int(max_likelihood_evaluations)
    )
    if likelihood_limit is not None and likelihood_limit < live_count:
        raise ValueError("max_likelihood_evaluations cannot be smaller than num_live.")
    if method not in ("hit-and-run", "slice-within-gibbs"):
        raise ValueError("method must be 'hit-and-run' or 'slice-within-gibbs'.")

    dimension = _validate_parameter_space(problem, prior_position_sampler)
    if method == "hit-and-run" and live_count <= dimension:
        raise ValueError(
            "hit-and-run requires num_live greater than the flattened dimension."
        )
    inner_steps = (
        max(5, 2 * dimension) if num_inner_steps is None else int(num_inner_steps)
    )
    if inner_steps < 1:
        raise ValueError("num_inner_steps must be positive.")

    root_key = jnp.asarray(key)
    destination = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else (Path(resume_from) if resume_from is not None else None)
    )
    if destination is not None and (checkpoint_id is None or not str(checkpoint_id)):
        raise ValueError("checkpoint_id is required for nested checkpointing.")

    settings = {
        "num_live": live_count,
        "method": method,
        "num_inner_steps": inner_steps,
        "num_delete": delete_count,
        "max_expansions": expansions_cap,
        "max_shrinkage": shrinkage_cap,
        "num_volume_replicates": volume_replicates,
        "custom_prior_position_sampler": prior_position_sampler is not None,
        "root_key": [int(value) for value in jr.key_data(root_key).reshape(-1)],
    }
    compatibility = (
        checkpoint_compatibility(
            problem,
            checkpoint_id=str(checkpoint_id),
            settings=settings,
            gradient_probe=False,
        )
        if destination is not None
        else None
    )

    if resume_from is None:
        prior_key = derive_key(root_key, _PRIOR_ADDRESS, live_count)
        initial_positions = (
            problem.parameter_space.sample_prior(prior_key, num_samples=live_count)
            if prior_position_sampler is None
            else prior_position_sampler(prior_key, live_count)
        )
    else:
        initial_positions = jax.tree.map(
            lambda value: jnp.broadcast_to(
                value,
                (live_count,) + jnp.asarray(value).shape,
            ),
            problem.initial_position,
        )
    _validate_initial_positions(problem, initial_positions, live_count)

    logprior_fn = problem.parameter_space.unconstrained_log_prior

    def loglikelihood_fn(position):
        return problem.log_likelihood(problem.parameter_space.constrain(position))

    algorithm = (
        blackjax.nss(
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            num_inner_steps=inner_steps,
            num_delete=delete_count,
            max_steps=expansions_cap,
            max_shrinkage=shrinkage_cap,
            proposal=_covariance_factor_proposal,
            inner_kernel_params=_live_covariance_factor,
        )
        if method == "hit-and-run"
        else blackjax.nsswig(
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            num_inner_steps=inner_steps,
            num_delete=delete_count,
            max_steps=expansions_cap,
            max_shrinkage=shrinkage_cap,
        )
    )
    initialize = jax.jit(algorithm.init)
    step = jax.jit(algorithm.step)
    started = time.perf_counter()
    template_live = initialize(
        initial_positions,
        rng_key=derive_key(root_key, _INIT_ADDRESS),
    )
    jax.block_until_ready(template_live.particles.loglikelihood)
    _validate_initial_particle_state(template_live.particles)

    if resume_from is None:
        live = template_live
        initial_log_likelihood = live.particles.loglikelihood
        dead_infos: list[NSInfo] = []
        dead_ids_parts: list[Array] = []
        insertion_parts: list[Array] = []
        accepted_parts: list[Array] = []
        expansion_parts: list[Array] = []
        shrinkage_parts: list[Array] = []
        completed_steps = 0
        likelihood_evaluations = live_count
        previous_duration = 0.0
        status: NestedSamplingStatus = NESTED_SAMPLING_MAX_DEAD_POINTS
        finished = False
    else:
        if compatibility is None:
            raise RuntimeError("Nested checkpoint compatibility was not initialized.")
        (
            live,
            initial_log_likelihood,
            dead_infos,
            dead_ids_parts,
            insertion_parts,
            accepted_parts,
            expansion_parts,
            shrinkage_parts,
            completed_steps,
            likelihood_evaluations,
            previous_duration,
            status,
            finished,
        ) = _read_nested_checkpoint(
            Path(resume_from),
            compatibility=compatibility,
            template_live=template_live,
            inner_steps=inner_steps,
        )

    if not bool(jnp.any(jnp.isfinite(live.particles.loglikelihood))):
        status = NESTED_SAMPLING_NO_FINITE_LIVE_POINT
        finished = True

    while not finished:
        remaining_fraction, _remaining_log_evidence = _remaining_evidence(live)
        if completed_steps > 0 and float(remaining_fraction) <= tolerance:
            status = NESTED_SAMPLING_SUCCESS
            break
        dead_count = sum(int(part.shape[0]) for part in dead_ids_parts)
        if dead_count + delete_count > dead_limit:
            status = NESTED_SAMPLING_MAX_DEAD_POINTS
            break
        if likelihood_limit is not None and likelihood_evaluations >= likelihood_limit:
            status = NESTED_SAMPLING_MAX_LIKELIHOOD_EVALUATIONS
            break
        current_likelihood = live.particles.loglikelihood
        if bool(jnp.ptp(current_likelihood) == 0.0):
            status = NESTED_SAMPLING_LIKELIHOOD_PLATEAU
            break

        _, dead_indices = jax.lax.top_k(-current_likelihood, delete_count)
        step_key = derive_key(root_key, _STEP_ADDRESS, completed_steps)
        live, info = step(step_key, live)
        jax.block_until_ready(live.particles.loglikelihood)
        _validate_step_state(info.particles, live.particles)

        replacement_likelihood = live.particles.loglikelihood[dead_indices]
        insertion_ranks = jnp.sum(
            live.particles.loglikelihood[None, :] < replacement_likelihood[:, None],
            axis=1,
        )
        update = info.update_info
        dead_infos.append(info)
        dead_ids_parts.append(dead_indices.astype(jnp.int32))
        insertion_parts.append(insertion_ranks.astype(jnp.int32))
        accepted_parts.append(jnp.asarray(update.is_accepted, dtype=bool))
        expansion_parts.append(jnp.asarray(update.num_expansions))
        shrinkage_parts.append(jnp.asarray(update.num_shrink))
        base_calls = 2 if method == "hit-and-run" else 2 * dimension
        likelihood_evaluations += int(
            jnp.sum(update.num_expansions + update.num_shrink + base_calls)
        )
        completed_steps += 1

        if (
            destination is not None
            and compatibility is not None
            and completed_steps % checkpoint_interval == 0
        ):
            _write_nested_checkpoint(
                destination,
                compatibility=compatibility,
                live=live,
                initial_log_likelihood=initial_log_likelihood,
                dead_infos=dead_infos,
                dead_ids_parts=dead_ids_parts,
                insertion_parts=insertion_parts,
                accepted_parts=accepted_parts,
                expansion_parts=expansion_parts,
                shrinkage_parts=shrinkage_parts,
                completed_steps=completed_steps,
                likelihood_evaluations=likelihood_evaluations,
                duration_seconds=previous_duration + time.perf_counter() - started,
                status=status,
                finished=False,
                inner_steps=inner_steps,
            )

    remaining_fraction, remaining_log_evidence = _remaining_evidence(live)
    completed = finalise(live, dead_infos, update_info=False)
    quadrature = compute_nested_quadrature(
        completed.particles,
        derive_key(root_key, _VOLUME_ADDRESS, volume_replicates),
        num_replicates=volume_replicates,
    )
    dead_particles = _concatenate_dead_particles(dead_infos, live.particles)
    dead_likelihood = dead_particles.loglikelihood
    dead_birth = dead_particles.loglikelihood_birth
    dead_ids = _concatenate_or_empty(dead_ids_parts, dtype=jnp.int32)
    all_ids = jnp.concatenate((dead_ids, jnp.arange(live_count, dtype=jnp.int32)))
    sorted_ids = all_ids[quadrature.sort_indices]
    batch_indices = jnp.zeros_like(sorted_ids)
    accepted = _concatenate_updates(
        accepted_parts,
        dead_ids.shape[0],
        inner_steps,
        dtype=bool,
    )
    expansions = _concatenate_updates(
        expansion_parts,
        dead_ids.shape[0],
        inner_steps,
        dtype=jnp.int32,
    )
    shrinkages = _concatenate_updates(
        shrinkage_parts,
        dead_ids.shape[0],
        inner_steps,
        dtype=jnp.int32,
    )
    insertions = _concatenate_or_empty(insertion_parts, dtype=jnp.int32)
    diagnostics = build_nested_diagnostics(
        dead_log_likelihood=dead_likelihood,
        dead_birth_log_likelihood=dead_birth,
        insertion_ranks=insertions,
        inner_accepted=accepted,
        num_expansions=expansions,
        num_shrink=shrinkages,
        max_expansions=expansions_cap,
        max_shrinkage=shrinkage_cap,
        initial_log_likelihood=initial_log_likelihood,
        sample_ids=sorted_ids,
        posterior_log_weights=quadrature.posterior_log_weights,
        num_live=live_count,
        quadrature_valid=quadrature.valid,
        final_live_positions=live.particles.position,
    )
    unconstrained_samples = quadrature.particles.position
    samples = problem.parameter_space.constrain(unconstrained_samples)
    valid = (
        quadrature.valid
        & diagnostics.likelihood_monotonic
        & diagnostics.constraints_satisfied
    )
    duration = previous_duration + time.perf_counter() - started

    if destination is not None and compatibility is not None:
        _write_nested_checkpoint(
            destination,
            compatibility=compatibility,
            live=live,
            initial_log_likelihood=initial_log_likelihood,
            dead_infos=dead_infos,
            dead_ids_parts=dead_ids_parts,
            insertion_parts=insertion_parts,
            accepted_parts=accepted_parts,
            expansion_parts=expansion_parts,
            shrinkage_parts=shrinkage_parts,
            completed_steps=completed_steps,
            likelihood_evaluations=likelihood_evaluations,
            duration_seconds=duration,
            status=status,
            finished=True,
            inner_steps=inner_steps,
        )

    return NestedSamplingResult(
        problem=problem,
        samples=samples,
        unconstrained_samples=unconstrained_samples,
        log_prior=quadrature.particles.logdensity,
        log_likelihood=quadrature.particles.loglikelihood,
        birth_log_likelihood=quadrature.particles.loglikelihood_birth,
        posterior_log_weights=quadrature.posterior_log_weights,
        log_prior_volume=quadrature.log_prior_volume,
        live_counts=quadrature.live_counts,
        sample_ids=sorted_ids,
        batch_indices=batch_indices,
        log_evidence=quadrature.log_evidence,
        log_evidence_replicates=quadrature.log_evidence_replicates,
        log_evidence_shrinkage_std=quadrature.log_evidence_shrinkage_std,
        information=quadrature.information,
        posterior_effective_sample_size=quadrature.posterior_effective_sample_size,
        remaining_log_evidence=remaining_log_evidence,
        remaining_evidence_fraction=remaining_fraction,
        final_state=live,
        diagnostics=diagnostics,
        root_key=root_key,
        status=jnp.asarray(status, dtype=jnp.int32),
        valid=valid,
        num_live=live_count,
        num_dead=int(dead_ids.shape[0]),
        num_likelihood_evaluations=likelihood_evaluations,
        num_inner_steps=inner_steps,
        num_delete=delete_count,
        method=method,
        duration_seconds=duration,
    )


def _live_covariance_factor(_key, state, _info, _parameters):
    """Factor one regularized live covariance per outer nested step."""
    leaves = jax.tree_util.tree_leaves(state.particles.position)
    count = int(leaves[0].shape[0])
    matrix = jnp.concatenate(
        tuple(jnp.asarray(leaf).reshape((count, -1)) for leaf in leaves),
        axis=1,
    )
    centered = matrix - jnp.mean(matrix, axis=0, keepdims=True)
    covariance = centered.T @ centered / jnp.asarray(count, dtype=matrix.dtype)
    dimension = int(covariance.shape[0])
    scale = jnp.trace(covariance) / jnp.asarray(dimension, dtype=matrix.dtype)
    epsilon = jnp.sqrt(jnp.finfo(matrix.dtype).eps)
    ridge = epsilon * jnp.maximum(scale, jnp.finfo(matrix.dtype).tiny / epsilon)
    regularized = covariance + ridge * jnp.eye(dimension, dtype=matrix.dtype)
    factor = jnp.linalg.cholesky(regularized)
    factor = eqx.error_if(
        factor,
        jnp.any(~jnp.isfinite(factor)),
        "Nested live covariance factorization failed.",
    )
    return {"covariance_factor": factor}


def _covariance_factor_proposal(
    init_state_fn,
    loglikelihood_threshold,
    covariance_factor,
):
    """Generate hit-and-run directions without repeated covariance inversion."""

    def proposal_generator(key, position, _logdensity_fn):
        flat, unravel = ravel_pytree(position)
        normal = jr.normal(key, flat.shape, dtype=flat.dtype)
        norm = jnp.linalg.norm(normal)
        direction = covariance_factor @ normal
        direction = (
            2.0
            * direction
            / jnp.maximum(
                norm,
                jnp.finfo(flat.dtype).tiny,
            )
        )
        direction_tree = unravel(direction)

        def slice_fn(distance):
            candidate = jax.tree.map(
                lambda value, delta: value + distance * delta,
                position,
                direction_tree,
            )
            state = init_state_fn(
                candidate,
                loglikelihood_birth=loglikelihood_threshold,
            )
            return state, state.loglikelihood > loglikelihood_threshold

        return slice_fn

    return proposal_generator


def _validate_parameter_space(
    problem: PosteriorProblem,
    prior_position_sampler: Callable[[Array, int], PyTree[Array]] | None,
) -> int:
    initial_leaves = jax.tree_util.tree_leaves(problem.initial_position)
    if any(jnp.iscomplexobj(leaf) for leaf in initial_leaves):
        raise TypeError("Nested slice sampling currently requires real parameters.")
    dimension = sum(int(jnp.asarray(leaf).size) for leaf in initial_leaves)
    if dimension < 1:
        raise ValueError("Nested sampling requires at least one scalar parameter.")
    priors = problem.parameter_space.priors
    if priors is None:
        if prior_position_sampler is None:
            raise ValueError(
                "A custom log prior requires prior_position_sampler for nested sampling."
            )
    else:
        prior_leaves = jax.tree_util.tree_leaves(
            priors,
            is_leaf=lambda value: isinstance(value, AbstractProbabilityLaw),
        )
        if any(prior.density_measure_kind != "lebesgue" for prior in prior_leaves):
            raise TypeError(
                "Nested slice sampling currently supports Lebesgue-density priors only."
            )
        if prior_position_sampler is not None:
            raise ValueError(
                "prior_position_sampler is only valid with a custom ParameterSpace prior."
            )
    return dimension


def _validate_initial_positions(
    problem: PosteriorProblem,
    positions: PyTree[Any],
    count: int,
) -> None:
    if jax.tree_util.tree_structure(positions) != jax.tree_util.tree_structure(
        problem.initial_position
    ):
        raise ValueError("Prior positions have incompatible PyTree structure.")
    for value, reference in zip(
        jax.tree_util.tree_leaves(positions),
        jax.tree_util.tree_leaves(problem.initial_position),
        strict=True,
    ):
        array = jnp.asarray(value)
        expected = (count,) + tuple(jnp.asarray(reference).shape)
        if array.shape != expected:
            raise ValueError(
                f"Prior position leaf has shape {array.shape}; expected {expected}."
            )
        if jnp.iscomplexobj(array) or not jnp.issubdtype(array.dtype, jnp.inexact):
            raise TypeError("Prior positions must be real inexact arrays.")
        if bool(jnp.any(~jnp.isfinite(array))):
            raise ValueError("Prior positions must be finite.")


def _validate_initial_particle_state(particles: StateWithLogLikelihood) -> None:
    log_prior = jnp.asarray(particles.logdensity)
    log_likelihood = jnp.asarray(particles.loglikelihood)
    if bool(jnp.any(~jnp.isfinite(log_prior))):
        raise ValueError("Initial prior log densities must be finite.")
    if bool(jnp.any(jnp.isnan(log_likelihood))) or bool(
        jnp.any(jnp.isposinf(log_likelihood))
    ):
        raise ValueError("Initial likelihoods cannot contain NaN or positive infinity.")


def _validate_step_state(
    dead: StateWithLogLikelihood,
    live: StateWithLogLikelihood,
) -> None:
    values = jnp.concatenate((dead.loglikelihood, live.loglikelihood))
    if bool(jnp.any(jnp.isnan(values))) or bool(jnp.any(jnp.isposinf(values))):
        raise FloatingPointError(
            "Nested likelihood evaluation produced NaN or positive infinity."
        )


def _remaining_evidence(live) -> tuple[Array, Array]:
    log_z = live.integrator.logZ
    remaining = jnp.max(live.particles.loglikelihood) + live.integrator.logX
    fraction = jnp.logaddexp(log_z, remaining) - log_z
    return fraction, remaining


def _concatenate_dead_particles(
    dead_infos: list[NSInfo],
    template: StateWithLogLikelihood,
) -> StateWithLogLikelihood:
    if dead_infos:
        return jax.tree.map(
            lambda *values: jnp.concatenate(values, axis=0),
            *[info.particles for info in dead_infos],
        )
    return jax.tree.map(lambda value: value[:0], template)


def _concatenate_or_empty(parts: list[Array], *, dtype) -> Array:
    if parts:
        return jnp.concatenate(tuple(parts), axis=0).astype(dtype)
    return jnp.empty((0,), dtype=dtype)


def _concatenate_updates(
    parts: list[Array],
    dead_count: int,
    inner_steps: int,
    *,
    dtype,
) -> Array:
    if parts:
        return jnp.concatenate(tuple(parts), axis=0).astype(dtype)
    return jnp.empty((dead_count, inner_steps), dtype=dtype)


def _tree_nbytes(tree: PyTree[Any], /) -> int:
    return sum(int(jnp.asarray(leaf).nbytes) for leaf in jax.tree_util.tree_leaves(tree))


def _write_nested_checkpoint(
    path: Path,
    *,
    compatibility: dict[str, Any],
    live,
    initial_log_likelihood: Array,
    dead_infos: list[NSInfo],
    dead_ids_parts: list[Array],
    insertion_parts: list[Array],
    accepted_parts: list[Array],
    expansion_parts: list[Array],
    shrinkage_parts: list[Array],
    completed_steps: int,
    likelihood_evaluations: int,
    duration_seconds: float,
    status: NestedSamplingStatus,
    finished: bool,
    inner_steps: int,
) -> None:
    arrays: dict[str, Any] = {}
    live_spec = pack_array_tree("live", live, arrays)
    dead_particles = _concatenate_dead_particles(dead_infos, live.particles)
    dead_spec = pack_array_tree("dead", dead_particles, arrays)
    arrays["initial_log_likelihood"] = initial_log_likelihood
    arrays["dead_ids"] = _concatenate_or_empty(dead_ids_parts, dtype=jnp.int32)
    arrays["insertion_ranks"] = _concatenate_or_empty(insertion_parts, dtype=jnp.int32)
    dead_count = int(arrays["dead_ids"].shape[0])
    arrays["inner_accepted"] = _concatenate_updates(
        accepted_parts, dead_count, inner_steps, dtype=bool
    )
    arrays["num_expansions"] = _concatenate_updates(
        expansion_parts, dead_count, inner_steps, dtype=jnp.int32
    )
    arrays["num_shrink"] = _concatenate_updates(
        shrinkage_parts, dead_count, inner_steps, dtype=jnp.int32
    )
    write_checkpoint_archive(
        path,
        kind=_CHECKPOINT_KIND,
        compatibility=compatibility,
        state={
            "live": live_spec,
            "dead": dead_spec,
            "completed_steps": int(completed_steps),
            "likelihood_evaluations": int(likelihood_evaluations),
            "duration_seconds": float(duration_seconds),
            "status": int(status),
            "finished": bool(finished),
        },
        arrays=arrays,
    )


def _read_nested_checkpoint(
    path: Path,
    *,
    compatibility: dict[str, Any],
    template_live,
    inner_steps: int,
):
    state, arrays = read_checkpoint_archive(
        path,
        kind=_CHECKPOINT_KIND,
        compatibility=compatibility,
    )
    live = unpack_array_tree(state["live"], arrays, template_live)
    dead_ids = jnp.asarray(arrays["dead_ids"], dtype=jnp.int32)
    dead_template = jax.tree.map(
        lambda value: jnp.empty(
            (int(dead_ids.shape[0]),) + value.shape[1:],
            dtype=value.dtype,
        ),
        template_live.particles,
    )
    dead_particles = unpack_array_tree(state["dead"], arrays, dead_template)
    dead_infos = [] if int(dead_ids.size) == 0 else [NSInfo(dead_particles, None)]
    accepted = jnp.asarray(arrays["inner_accepted"], dtype=bool)
    expansions = jnp.asarray(arrays["num_expansions"], dtype=jnp.int32)
    shrinkages = jnp.asarray(arrays["num_shrink"], dtype=jnp.int32)
    expected_update_shape = (int(dead_ids.shape[0]), int(inner_steps))
    if (
        accepted.shape != expected_update_shape
        or expansions.shape != expected_update_shape
        or shrinkages.shape != expected_update_shape
    ):
        raise ValueError("Nested checkpoint inner statistics have invalid shape.")
    return (
        live,
        jnp.asarray(arrays["initial_log_likelihood"]),
        dead_infos,
        [] if int(dead_ids.size) == 0 else [dead_ids],
        [jnp.asarray(arrays["insertion_ranks"], dtype=jnp.int32)],
        [] if int(dead_ids.size) == 0 else [accepted],
        [] if int(dead_ids.size) == 0 else [expansions],
        [] if int(dead_ids.size) == 0 else [shrinkages],
        int(state["completed_steps"]),
        int(state["likelihood_evaluations"]),
        float(state["duration_seconds"]),
        int(state["status"]),
        bool(state["finished"]),
    )


__all__ = [
    "NESTED_SAMPLING_INNER_KERNEL_FAILURE",
    "NESTED_SAMPLING_INVALID_LIKELIHOOD",
    "NESTED_SAMPLING_LIKELIHOOD_PLATEAU",
    "NESTED_SAMPLING_MAX_DEAD_POINTS",
    "NESTED_SAMPLING_MAX_LIKELIHOOD_EVALUATIONS",
    "NESTED_SAMPLING_NO_FINITE_LIVE_POINT",
    "NESTED_SAMPLING_SUCCESS",
    "NestedSamplingMethod",
    "NestedSamplingResult",
    "NestedSamplingStatus",
    "nested_sampling_status_name",
    "sample_nested",
]
