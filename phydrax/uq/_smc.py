#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import blackjax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from blackjax.smc import adaptive_tempered, resampling
from jaxtyping import Array, PyTree

from .._fingerprint import array_tree_fingerprint
from .._frozendict import frozendict
from .._strict import StrictModule
from ._checkpoint import (
    checkpoint_compatibility,
    CheckpointCorruptionError,
    pack_array_tree,
    read_checkpoint_archive,
    unpack_array_tree,
    write_checkpoint_archive,
)
from ._posterior import PosteriorProblem
from ._posterior_predictive import (
    predict_from_position_samples,
    sample_observations_from_position_samples,
)
from ._predictive import PredictiveField


ResamplingMethod = Literal["systematic", "stratified"]


class TemperedSMCResult(StrictModule):
    """Adaptive likelihood-tempering particles and degeneracy diagnostics."""

    problem: PosteriorProblem
    state: Any
    samples: PyTree[Array]
    unconstrained_samples: PyTree[Array]
    final_weights: Array
    temperatures: Array
    effective_sample_sizes: Array
    acceptance_rates: Array
    divergence_rates: Array
    log_evidence: Array
    root_key: Array
    duration_seconds: float = eqx.field(static=True)
    sample_memory_bytes: int = eqx.field(static=True)
    num_unique_initial_particles: int = eqx.field(static=True)
    resampling_method: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        problem: PosteriorProblem,
        state: Any,
        samples: PyTree[Array],
        unconstrained_samples: PyTree[Array],
        final_weights: Array,
        temperatures: Array,
        effective_sample_sizes: Array,
        acceptance_rates: Array,
        divergence_rates: Array,
        log_evidence: Array,
        root_key: Array,
        duration_seconds: float,
        num_unique_initial_particles: int,
        resampling_method: ResamplingMethod,
    ):
        self.problem = problem
        self.state = state
        self.samples = samples
        self.unconstrained_samples = unconstrained_samples
        self.final_weights = jnp.asarray(final_weights)
        self.temperatures = jnp.asarray(temperatures)
        self.effective_sample_sizes = jnp.asarray(effective_sample_sizes)
        self.acceptance_rates = jnp.asarray(acceptance_rates)
        self.divergence_rates = jnp.asarray(divergence_rates)
        self.log_evidence = jnp.asarray(log_evidence)
        self.root_key = jnp.asarray(root_key)
        self.duration_seconds = float(duration_seconds)
        self.sample_memory_bytes = _tree_nbytes(samples) + _tree_nbytes(
            unconstrained_samples
        )
        self.num_unique_initial_particles = int(num_unique_initial_particles)
        self.resampling_method = str(resampling_method)

    @property
    def num_particles(self) -> int:
        return int(self.final_weights.shape[0])

    @property
    def num_tempering_steps(self) -> int:
        return int(self.temperatures.shape[0] - 1)

    def predict(
        self,
        *args: Any,
        batch_size: int | None = None,
        valid_policy: Literal["record", "raise"] = "record",
        **kwargs: Any,
    ) -> PredictiveField | frozendict[str, PredictiveField]:
        """Evaluate latent predictions while retaining the particle dimension."""
        return predict_from_position_samples(
            self.problem,
            self.unconstrained_samples,
            *args,
            sample_dims=("__phydra_uq_particle",),
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
        """Draw conditional measurements for every posterior particle."""
        return sample_observations_from_position_samples(
            self.problem,
            key,
            self.unconstrained_samples,
            *args,
            sample_dims=("__phydra_uq_particle",),
            sample_sources=("epistemic",),
            num_observation_samples=num_observation_samples,
            batch_size=batch_size,
            valid_policy=valid_policy,
            observation_dim=observation_dim,
            **kwargs,
        )


def sample_tempered_smc(
    problem: PosteriorProblem,
    /,
    *,
    key: Array,
    num_particles: int = 1_000,
    prior_position_sampler: Callable[[Array, int], PyTree[Array]] | None = None,
    target_ess: float = 0.8,
    num_mcmc_steps: int = 10,
    step_size: float = 0.1,
    num_integration_steps: int = 10,
    inverse_mass_matrix: Array | None = None,
    max_tempering_steps: int = 100,
    resampling_method: ResamplingMethod = "systematic",
    batch_size: int = 0,
    checkpoint_path: str | Path | None = None,
    checkpoint_id: str | None = None,
    resume_from: str | Path | None = None,
) -> TemperedSMCResult:
    """Run adaptive tempered SMC with resumable HMC rejuvenation stages."""
    if not isinstance(problem, PosteriorProblem):
        raise TypeError("problem must be a PosteriorProblem.")
    particles_count = int(num_particles)
    mcmc_steps = int(num_mcmc_steps)
    integration_steps = int(num_integration_steps)
    tempering_steps = int(max_tempering_steps)
    sequential_batch = int(batch_size)
    if particles_count < 2:
        raise ValueError("num_particles must be at least two.")
    if mcmc_steps <= 0 or integration_steps <= 0 or tempering_steps <= 0:
        raise ValueError(
            "num_mcmc_steps, num_integration_steps, and max_tempering_steps "
            "must be positive."
        )
    if not 0.0 < float(target_ess) < 1.0:
        raise ValueError("target_ess must lie strictly between zero and one.")
    if not jnp.isfinite(step_size) or float(step_size) <= 0.0:
        raise ValueError("step_size must be finite and positive.")
    if sequential_batch < 0:
        raise ValueError("batch_size must be non-negative.")
    if resampling_method not in ("systematic", "stratified"):
        raise ValueError("resampling_method must be 'systematic' or 'stratified'.")
    if prior_position_sampler is not None and not callable(prior_position_sampler):
        raise TypeError("prior_position_sampler must be callable or None.")

    destination = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else (Path(resume_from) if resume_from is not None else None)
    )
    if destination is not None and (checkpoint_id is None or not str(checkpoint_id)):
        raise ValueError("checkpoint_id is required for tempered SMC checkpointing.")

    dimension = sum(
        int(jnp.asarray(leaf).size)
        for leaf in jax.tree_util.tree_leaves(problem.initial_position)
    )
    if inverse_mass_matrix is None:
        mass_matrix = jnp.ones(dimension)
    else:
        mass_matrix = jnp.asarray(inverse_mass_matrix, dtype=float)
        if mass_matrix.shape not in ((dimension,), (dimension, dimension)):
            raise ValueError(
                "inverse_mass_matrix must have shape (dimension,) or "
                "(dimension, dimension)."
            )
        if not bool(jnp.all(jnp.isfinite(mass_matrix))):
            raise ValueError("inverse_mass_matrix must be finite.")

    root_key = jnp.asarray(key)
    prior_key, transition_key, final_resampling_key = jr.split(root_key, 3)
    settings = {
        "num_particles": particles_count,
        "target_ess": float(target_ess),
        "num_mcmc_steps": mcmc_steps,
        "step_size": float(step_size),
        "num_integration_steps": integration_steps,
        "inverse_mass_matrix": array_tree_fingerprint(mass_matrix),
        "resampling_method": resampling_method,
        "batch_size": sequential_batch,
        "custom_prior_position_sampler": prior_position_sampler is not None,
        "root_key": [int(value) for value in jr.key_data(root_key).reshape(-1)],
    }
    compatibility = (
        checkpoint_compatibility(
            problem,
            checkpoint_id=str(checkpoint_id),
            settings=settings,
        )
        if destination is not None
        else None
    )
    resampler = (
        resampling.systematic
        if resampling_method == "systematic"
        else resampling.stratified
    )
    hmc_step = blackjax.hmc.build_kernel()
    uncompiled_kernel = adaptive_tempered.build_kernel(
        problem.parameter_space.unconstrained_log_prior,
        lambda position: problem.log_likelihood(
            problem.parameter_space.constrain(position)
        ),
        hmc_step,
        blackjax.hmc.init,
        resampler,
        float(target_ess),
        batch_size=sequential_batch,
    )
    kernel = jax.jit(
        lambda step_key, current_state, parameters: uncompiled_kernel(
            step_key,
            current_state,
            mcmc_steps,
            parameters,
        )
    )
    mcmc_parameters = {
        "step_size": jnp.asarray([step_size], dtype=float),
        "inverse_mass_matrix": mass_matrix[None, ...],
        "num_integration_steps": jnp.asarray([integration_steps]),
    }
    started = time.perf_counter()

    if resume_from is None:
        if prior_position_sampler is None:
            initial_particles = problem.parameter_space.sample_prior(
                prior_key,
                num_samples=particles_count,
            )
        else:
            initial_particles = prior_position_sampler(prior_key, particles_count)
        _validate_particles(problem, initial_particles, particles_count)
        state = adaptive_tempered.init(initial_particles)
        completed = 0
        temperatures = jnp.zeros((1,))
        effective_sample_sizes = jnp.asarray([float(particles_count)])
        acceptance_rates = jnp.empty((0,))
        divergence_rates = jnp.empty((0,))
        log_evidence_terms = jnp.empty((0,))
        lineage = jnp.arange(particles_count)
        previous_duration = 0.0
        if destination is not None and compatibility is not None:
            _write_smc_checkpoint(
                destination,
                compatibility=compatibility,
                completed=completed,
                state=state,
                lineage=lineage,
                temperatures=temperatures,
                effective_sample_sizes=effective_sample_sizes,
                acceptance_rates=acceptance_rates,
                divergence_rates=divergence_rates,
                log_evidence_terms=log_evidence_terms,
                duration_seconds=time.perf_counter() - started,
            )
    else:
        if compatibility is None:
            raise RuntimeError("Tempered SMC resume compatibility was not initialized.")
        (
            completed,
            state,
            lineage,
            temperatures,
            effective_sample_sizes,
            acceptance_rates,
            divergence_rates,
            log_evidence_terms,
            previous_duration,
        ) = _read_smc_checkpoint(
            Path(resume_from),
            compatibility=compatibility,
            problem=problem,
            particles_count=particles_count,
        )
        if completed > tempering_steps:
            raise ValueError(
                "max_tempering_steps cannot be smaller than completed stages."
            )

    for step in range(completed, tempering_steps):
        if float(state.tempering_param) >= 1.0 - 1e-7:
            break
        step_key = jr.fold_in(transition_key, step)
        state, info = kernel(step_key, state, mcmc_parameters)
        jax.block_until_ready(state.tempering_param)
        temperatures = jnp.concatenate(
            (temperatures, jnp.asarray(state.tempering_param).reshape(1))
        )
        effective_sample_sizes = jnp.concatenate(
            (
                effective_sample_sizes,
                (1.0 / jnp.sum(state.weights**2)).reshape(1),
            )
        )
        acceptance_rates = jnp.concatenate(
            (
                acceptance_rates,
                jnp.mean(info.update_info.acceptance_rate).reshape(1),
            )
        )
        divergence_rates = jnp.concatenate(
            (
                divergence_rates,
                jnp.mean(info.update_info.is_divergent).reshape(1),
            )
        )
        log_evidence_terms = jnp.concatenate(
            (
                log_evidence_terms,
                jnp.asarray(info.log_likelihood_increment).reshape(1),
            )
        )
        lineage = lineage[info.ancestors]
        completed = step + 1
        if destination is not None and compatibility is not None:
            _write_smc_checkpoint(
                destination,
                compatibility=compatibility,
                completed=completed,
                state=state,
                lineage=lineage,
                temperatures=temperatures,
                effective_sample_sizes=effective_sample_sizes,
                acceptance_rates=acceptance_rates,
                divergence_rates=divergence_rates,
                log_evidence_terms=log_evidence_terms,
                duration_seconds=(previous_duration + time.perf_counter() - started),
            )

    if float(state.tempering_param) < 1.0 - 1e-7:
        raise RuntimeError(
            f"Tempered SMC did not reach unit temperature within {tempering_steps} steps."
        )
    final_indices = resampler(
        final_resampling_key,
        state.weights,
        particles_count,
    )
    unconstrained_samples = jax.tree_util.tree_map(
        lambda value: value[final_indices],
        state.particles,
    )
    samples = problem.parameter_space.constrain(unconstrained_samples)
    lineage = lineage[final_indices]
    jax.block_until_ready(jax.tree_util.tree_leaves(samples)[0])
    duration = previous_duration + time.perf_counter() - started
    return TemperedSMCResult(
        problem=problem,
        state=state,
        samples=samples,
        unconstrained_samples=unconstrained_samples,
        final_weights=state.weights,
        temperatures=temperatures,
        effective_sample_sizes=effective_sample_sizes,
        acceptance_rates=acceptance_rates,
        divergence_rates=divergence_rates,
        log_evidence=jnp.sum(log_evidence_terms),
        root_key=root_key,
        duration_seconds=duration,
        num_unique_initial_particles=int(jnp.unique(lineage).size),
        resampling_method=resampling_method,
    )


def _write_smc_checkpoint(
    destination,
    *,
    compatibility,
    completed,
    state,
    lineage,
    temperatures,
    effective_sample_sizes,
    acceptance_rates,
    divergence_rates,
    log_evidence_terms,
    duration_seconds,
):
    arrays = {
        "lineage": lineage,
        "temperatures": temperatures,
        "effective_sample_sizes": effective_sample_sizes,
        "acceptance_rates": acceptance_rates,
        "divergence_rates": divergence_rates,
        "log_evidence_terms": log_evidence_terms,
    }
    checkpoint_state = {
        "completed_stages": int(completed),
        "duration_seconds": float(duration_seconds),
        "tempered_state_tree": pack_array_tree("tempered_state", state, arrays),
    }
    write_checkpoint_archive(
        destination,
        kind="tempered_smc",
        compatibility=compatibility,
        state=checkpoint_state,
        arrays=arrays,
    )


def _read_smc_checkpoint(
    source,
    *,
    compatibility,
    problem,
    particles_count,
):
    checkpoint_state, arrays = read_checkpoint_archive(
        source,
        kind="tempered_smc",
        compatibility=compatibility,
    )
    completed = int(checkpoint_state.get("completed_stages", -1))
    if completed < 0:
        raise CheckpointCorruptionError("Checkpoint completed stage count is invalid.")
    particle_template = jax.tree_util.tree_map(
        lambda value: jnp.empty(
            (particles_count, *value.shape),
            dtype=value.dtype,
        ),
        problem.initial_position,
    )
    state_template = adaptive_tempered.init(particle_template)
    state = unpack_array_tree(
        checkpoint_state["tempered_state_tree"],
        arrays,
        state_template,
    )
    lineage = _smc_checkpoint_array(arrays, "lineage", shape=(particles_count,))
    temperatures = _smc_checkpoint_array(arrays, "temperatures", shape=(completed + 1,))
    effective_sample_sizes = _smc_checkpoint_array(
        arrays,
        "effective_sample_sizes",
        shape=(completed + 1,),
    )
    acceptance_rates = _smc_checkpoint_array(
        arrays, "acceptance_rates", shape=(completed,)
    )
    divergence_rates = _smc_checkpoint_array(
        arrays, "divergence_rates", shape=(completed,)
    )
    log_evidence_terms = _smc_checkpoint_array(
        arrays, "log_evidence_terms", shape=(completed,)
    )
    return (
        completed,
        state,
        lineage,
        temperatures,
        effective_sample_sizes,
        acceptance_rates,
        divergence_rates,
        log_evidence_terms,
        float(checkpoint_state["duration_seconds"]),
    )


def _smc_checkpoint_array(arrays, name, *, shape):
    if name not in arrays:
        raise CheckpointCorruptionError(f"Checkpoint array {name!r} is missing.")
    value = jnp.asarray(arrays[name])
    if value.shape != shape:
        raise CheckpointCorruptionError(
            f"Checkpoint array {name!r} has an invalid shape."
        )
    return value


def _validate_particles(
    problem: PosteriorProblem,
    particles: PyTree[Array],
    expected_count: int,
) -> None:
    leaves = jax.tree_util.tree_leaves(particles)
    if not leaves or any(
        jnp.asarray(leaf).ndim == 0 or jnp.asarray(leaf).shape[0] != expected_count
        for leaf in leaves
    ):
        raise ValueError(
            "Prior particles must have a leading axis equal to num_particles."
        )
    one_particle = jax.tree_util.tree_map(lambda value: value[0], particles)
    problem.parameter_space.constrain(one_particle)
    finite = all(bool(jnp.all(jnp.isfinite(jnp.asarray(leaf)))) for leaf in leaves)
    if not finite:
        raise FloatingPointError("Prior particles must be finite.")


def _tree_nbytes(tree: PyTree[Any], /) -> int:
    return sum(
        int(jnp.asarray(leaf).nbytes)
        for leaf in jax.tree_util.tree_leaves(tree)
        if eqx.is_array(leaf)
    )


__all__ = ["TemperedSMCResult", "sample_tempered_smc"]
