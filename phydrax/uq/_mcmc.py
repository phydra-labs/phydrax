#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Literal

import blackjax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PyTree

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
from ._diagnostics import (
    mcmc_diagnostics,
    MCMCConvergenceReport,
    MCMCConvergenceThresholds,
    MCMCDiagnostics,
)
from ._posterior import PosteriorProblem
from ._posterior_predictive import (
    predict_from_position_samples,
    sample_observations_from_position_samples,
)
from ._predictive import PredictiveField


ChainMethod = Literal["sequential", "vectorized"]


class MCMCChainWarmup(StrictModule):
    """Final warmup state and tuned parameters for one chain."""

    state: Any
    step_size: Array
    inverse_mass_matrix: Array
    num_integration_steps: int | None = eqx.field(static=True)
    duration_seconds: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        state: Any,
        step_size: Array,
        inverse_mass_matrix: Array,
        num_integration_steps: int | None,
        duration_seconds: float,
    ):
        self.state = state
        self.step_size = jnp.asarray(step_size)
        self.inverse_mass_matrix = jnp.asarray(inverse_mass_matrix)
        self.num_integration_steps = num_integration_steps
        self.duration_seconds = float(duration_seconds)


class MCMCResult(StrictModule):
    """Chain-preserving posterior draws, diagnostics, states, and predictions."""

    problem: PosteriorProblem
    samples: PyTree[Array]
    unconstrained_samples: PyTree[Array]
    log_density: Array
    acceptance_rate: Array
    divergent: Array
    energy: Array
    num_integration_steps: Array
    num_trajectory_expansions: Array
    final_states: tuple[Any, ...]
    warmup: tuple[MCMCChainWarmup, ...]
    diagnostics: MCMCDiagnostics
    root_key: Array
    chain_keys: Array
    algorithm: str = eqx.field(static=True)
    duration_seconds: float = eqx.field(static=True)
    sample_memory_bytes: int = eqx.field(static=True)
    max_num_doublings: int | None = eqx.field(static=True)
    chain_method: ChainMethod = eqx.field(static=True)
    adaptation_duration_seconds: float = eqx.field(static=True)
    sampling_duration_seconds: float = eqx.field(static=True)
    samples_per_second: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        problem: PosteriorProblem,
        samples: PyTree[Array],
        unconstrained_samples: PyTree[Array],
        log_density: Array,
        acceptance_rate: Array,
        divergent: Array,
        energy: Array,
        num_integration_steps: Array,
        num_trajectory_expansions: Array,
        final_states: tuple[Any, ...],
        warmup: tuple[MCMCChainWarmup, ...],
        diagnostics: MCMCDiagnostics,
        root_key: Array,
        chain_keys: Array,
        algorithm: str,
        duration_seconds: float,
        max_num_doublings: int | None,
        chain_method: ChainMethod,
        adaptation_duration_seconds: float,
        sampling_duration_seconds: float,
    ):
        self.problem = problem
        self.samples = samples
        self.unconstrained_samples = unconstrained_samples
        self.log_density = jnp.asarray(log_density)
        self.acceptance_rate = jnp.asarray(acceptance_rate)
        self.divergent = jnp.asarray(divergent, dtype=bool)
        self.energy = jnp.asarray(energy)
        self.num_integration_steps = jnp.asarray(num_integration_steps)
        self.num_trajectory_expansions = jnp.asarray(num_trajectory_expansions)
        self.final_states = final_states
        self.warmup = warmup
        self.diagnostics = diagnostics
        self.root_key = root_key
        self.chain_keys = chain_keys
        self.algorithm = str(algorithm)
        self.duration_seconds = float(duration_seconds)
        self.sample_memory_bytes = _tree_nbytes(samples) + _tree_nbytes(
            unconstrained_samples
        )
        self.max_num_doublings = max_num_doublings
        self.chain_method = chain_method
        self.adaptation_duration_seconds = float(adaptation_duration_seconds)
        self.sampling_duration_seconds = float(sampling_duration_seconds)
        self.samples_per_second = (
            self.num_chains * self.num_draws / self.sampling_duration_seconds
        )

    @property
    def num_chains(self) -> int:
        return int(self.log_density.shape[0])

    @property
    def num_draws(self) -> int:
        return int(self.log_density.shape[1])

    def predict(
        self,
        *args: Any,
        batch_size: int | None = None,
        valid_policy: Literal["record", "raise"] = "record",
        chain_dim: str = "__phydra_uq_chain",
        draw_dim: str = "__phydra_uq_draw",
        **kwargs: Any,
    ) -> PredictiveField | frozendict[str, PredictiveField]:
        """Evaluate posterior draws without merging chain and draw dimensions."""
        return predict_from_position_samples(
            self.problem,
            self.unconstrained_samples,
            *args,
            sample_dims=(chain_dim, draw_dim),
            sample_sources=("epistemic", "epistemic"),
            batch_size=batch_size,
            valid_policy=valid_policy,
            **kwargs,
        )

    def predict_observations(
        self,
        key: Array,
        /,
        *args: Any,
        num_observation_samples: int,
        batch_size: int | None = None,
        valid_policy: Literal["record", "raise"] = "record",
        chain_dim: str = "__phydra_uq_chain",
        draw_dim: str = "__phydra_uq_draw",
        observation_dim: str = "__phydra_uq_observation",
        **kwargs: Any,
    ) -> PredictiveField | frozendict[str, PredictiveField]:
        """Draw measurement realizations without merging posterior chains."""
        return sample_observations_from_position_samples(
            self.problem,
            key,
            self.unconstrained_samples,
            *args,
            num_observation_samples=num_observation_samples,
            sample_dims=(chain_dim, draw_dim),
            sample_sources=("epistemic", "epistemic"),
            observation_dim=observation_dim,
            batch_size=batch_size,
            valid_policy=valid_policy,
            **kwargs,
        )

    def convergence_report(
        self,
        *,
        max_rhat: float = 1.01,
        min_bulk_ess: float = 400.0,
        min_tail_ess: float = 400.0,
        allow_divergences: bool = False,
        allow_trajectory_saturation: bool = False,
    ) -> MCMCConvergenceReport:
        """Evaluate explicit release gates without altering raw diagnostics."""
        thresholds = MCMCConvergenceThresholds(
            max_rhat=max_rhat,
            min_bulk_ess=min_bulk_ess,
            min_tail_ess=min_tail_ess,
            allow_divergences=allow_divergences,
            allow_trajectory_saturation=allow_trajectory_saturation,
        )
        return MCMCConvergenceReport(
            diagnostics=self.diagnostics,
            thresholds=thresholds,
            divergent=self.divergent,
            num_integration_steps=self.num_integration_steps,
            num_trajectory_expansions=self.num_trajectory_expansions,
            max_num_doublings=self.max_num_doublings,
            num_chains=self.num_chains,
            num_draws=self.num_draws,
            sample_memory_bytes=self.sample_memory_bytes,
            duration_seconds=self.duration_seconds,
            adaptation_duration_seconds=self.adaptation_duration_seconds,
            sampling_duration_seconds=self.sampling_duration_seconds,
            samples_per_second=self.samples_per_second,
        )


def sample_nuts(
    problem: PosteriorProblem,
    /,
    *,
    key: Array,
    num_chains: int = 4,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    initial_position: PyTree[Any] | None = None,
    target_acceptance_rate: float = 0.8,
    initial_step_size: float = 1.0,
    is_mass_matrix_diagonal: bool = True,
    max_num_doublings: int = 10,
    chain_method: ChainMethod = "sequential",
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int | None = None,
    checkpoint_id: str | None = None,
    resume_from: str | Path | None = None,
) -> MCMCResult:
    """Run independently adapted BlackJAX No-U-Turn sampler chains."""
    if int(max_num_doublings) <= 0:
        raise ValueError("max_num_doublings must be positive.")
    return _sample_mcmc(
        problem,
        key=key,
        algorithm="nuts",
        num_chains=num_chains,
        num_warmup=num_warmup,
        num_samples=num_samples,
        initial_position=initial_position,
        target_acceptance_rate=target_acceptance_rate,
        initial_step_size=initial_step_size,
        is_mass_matrix_diagonal=is_mass_matrix_diagonal,
        extra_parameters={"max_num_doublings": int(max_num_doublings)},
        chain_method=chain_method,
        checkpoint_path=checkpoint_path,
        checkpoint_every=checkpoint_every,
        checkpoint_id=checkpoint_id,
        resume_from=resume_from,
    )


def sample_hmc(
    problem: PosteriorProblem,
    /,
    *,
    key: Array,
    num_integration_steps: int,
    num_chains: int = 4,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    initial_position: PyTree[Any] | None = None,
    target_acceptance_rate: float = 0.8,
    initial_step_size: float = 1.0,
    is_mass_matrix_diagonal: bool = True,
    chain_method: ChainMethod = "sequential",
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int | None = None,
    checkpoint_id: str | None = None,
    resume_from: str | Path | None = None,
) -> MCMCResult:
    """Run independently adapted fixed-trajectory BlackJAX HMC chains."""
    if int(num_integration_steps) <= 0:
        raise ValueError("num_integration_steps must be positive.")
    return _sample_mcmc(
        problem,
        key=key,
        algorithm="hmc",
        num_chains=num_chains,
        num_warmup=num_warmup,
        num_samples=num_samples,
        initial_position=initial_position,
        target_acceptance_rate=target_acceptance_rate,
        initial_step_size=initial_step_size,
        is_mass_matrix_diagonal=is_mass_matrix_diagonal,
        extra_parameters={"num_integration_steps": int(num_integration_steps)},
        chain_method=chain_method,
        checkpoint_path=checkpoint_path,
        checkpoint_every=checkpoint_every,
        checkpoint_id=checkpoint_id,
        resume_from=resume_from,
    )


def _sample_mcmc(
    problem: PosteriorProblem,
    /,
    *,
    key: Array,
    algorithm: Literal["nuts", "hmc"],
    num_chains: int,
    num_warmup: int,
    num_samples: int,
    initial_position: PyTree[Any] | None,
    target_acceptance_rate: float,
    initial_step_size: float,
    is_mass_matrix_diagonal: bool,
    extra_parameters: dict[str, Any],
    chain_method: ChainMethod,
    checkpoint_path: str | Path | None,
    checkpoint_every: int | None,
    checkpoint_id: str | None,
    resume_from: str | Path | None,
) -> MCMCResult:
    if not isinstance(problem, PosteriorProblem):
        raise TypeError("problem must be a PosteriorProblem.")
    chains = int(num_chains)
    warmup_steps = int(num_warmup)
    draws = int(num_samples)
    if chains < 2:
        raise ValueError("num_chains must be at least two for convergence diagnostics.")
    if warmup_steps <= 0:
        raise ValueError("num_warmup must be positive.")
    if draws < 4:
        raise ValueError("num_samples must be at least four.")
    target = float(target_acceptance_rate)
    if not 0.0 < target < 1.0:
        raise ValueError("target_acceptance_rate must lie strictly between zero and one.")
    initial_step = float(initial_step_size)
    if initial_step <= 0.0:
        raise ValueError("initial_step_size must be positive.")
    method: ChainMethod = chain_method
    if method not in ("sequential", "vectorized"):
        raise ValueError("chain_method must be 'sequential' or 'vectorized'.")
    if resume_from is not None and initial_position is not None:
        raise ValueError("initial_position cannot be supplied when resuming MCMC.")

    destination = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else (Path(resume_from) if resume_from is not None else None)
    )
    if checkpoint_every is not None and destination is None:
        raise ValueError("checkpoint_every requires checkpoint_path or resume_from.")
    interval = min(100, draws) if checkpoint_every is None else int(checkpoint_every)
    if interval <= 0:
        raise ValueError("checkpoint_every must be positive.")
    if destination is not None and (checkpoint_id is None or not str(checkpoint_id)):
        raise ValueError("checkpoint_id is required for MCMC checkpointing.")

    position = problem.initial_position if initial_position is None else initial_position
    problem.parameter_space.constrain(position)
    value, gradient = jax.value_and_grad(problem.log_density)(position)
    if not bool(jnp.isfinite(value)) or any(
        bool(jnp.any(~jnp.isfinite(jnp.asarray(leaf))))
        for leaf in jax.tree_util.tree_leaves(gradient)
    ):
        raise FloatingPointError("Initial MCMC log density and gradient must be finite.")

    root_key = jnp.asarray(key)
    chain_keys = jr.split(root_key, chains)
    split_keys = jax.vmap(lambda chain_key: jr.split(chain_key, 2))(chain_keys)
    warmup_keys = split_keys[:, 0]
    sample_keys = split_keys[:, 1]
    settings = {
        "algorithm": algorithm,
        "num_chains": chains,
        "num_warmup": warmup_steps,
        "target_acceptance_rate": target,
        "initial_step_size": initial_step,
        "is_mass_matrix_diagonal": bool(is_mass_matrix_diagonal),
        "chain_method": method,
        "extra_parameters": extra_parameters,
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
    algorithm_factory = blackjax.nuts if algorithm == "nuts" else blackjax.hmc
    logdensity_fn = lambda current: problem.log_density(current)
    started = time.perf_counter()

    if resume_from is None:
        (
            current_states,
            warmup_states,
            step_sizes,
            inverse_mass_matrices,
            warmup_durations,
            adaptation_duration,
        ) = _adapt_mcmc(
            algorithm_factory,
            logdensity_fn,
            position,
            warmup_keys=warmup_keys,
            warmup_steps=warmup_steps,
            target_acceptance_rate=target,
            initial_step_size=initial_step,
            is_mass_matrix_diagonal=bool(is_mass_matrix_diagonal),
            extra_parameters=extra_parameters,
            chain_method=method,
        )
        completed = 0
        unconstrained_samples = _empty_sample_tree(position, chains)
        log_density = jnp.empty((chains, 0), dtype=float)
        acceptance_rate = jnp.empty((chains, 0), dtype=float)
        divergent = jnp.empty((chains, 0), dtype=bool)
        energy = jnp.empty((chains, 0), dtype=float)
        num_integration_steps_array = jnp.empty((chains, 0), dtype=jnp.int32)
        num_trajectory_expansions_array = jnp.empty((chains, 0), dtype=jnp.int32)
        sampling_duration = 0.0
        previous_duration = 0.0
        if destination is not None and compatibility is not None:
            _write_mcmc_checkpoint(
                destination,
                compatibility=compatibility,
                completed=completed,
                current_states=current_states,
                warmup_states=warmup_states,
                step_sizes=step_sizes,
                inverse_mass_matrices=inverse_mass_matrices,
                warmup_durations=warmup_durations,
                unconstrained_samples=unconstrained_samples,
                log_density=log_density,
                acceptance_rate=acceptance_rate,
                divergent=divergent,
                energy=energy,
                num_integration_steps=num_integration_steps_array,
                num_trajectory_expansions=num_trajectory_expansions_array,
                adaptation_duration=adaptation_duration,
                sampling_duration=sampling_duration,
                duration_seconds=time.perf_counter() - started,
            )
    else:
        if compatibility is None:
            raise RuntimeError("MCMC resume compatibility was not initialized.")
        (
            completed,
            current_states,
            warmup_states,
            step_sizes,
            inverse_mass_matrices,
            warmup_durations,
            unconstrained_samples,
            log_density,
            acceptance_rate,
            divergent,
            energy,
            num_integration_steps_array,
            num_trajectory_expansions_array,
            adaptation_duration,
            sampling_duration,
            previous_duration,
        ) = _read_mcmc_checkpoint(
            Path(resume_from),
            compatibility=compatibility,
            algorithm_factory=algorithm_factory,
            logdensity_fn=logdensity_fn,
            position=position,
            chains=chains,
        )
        if completed > draws:
            raise ValueError(
                "num_samples cannot be smaller than completed checkpoint draws."
            )

    while completed < draws:
        chunk = min(interval, draws - completed)
        sampling_started = time.perf_counter()
        current_states, chunk_samples, chunk_metrics = _advance_mcmc(
            algorithm_factory,
            logdensity_fn,
            current_states,
            step_sizes,
            inverse_mass_matrices,
            sample_keys,
            start=completed,
            count=chunk,
            algorithm=algorithm,
            extra_parameters=extra_parameters,
            chain_method=method,
        )
        jax.block_until_ready(chunk_metrics["log_density"])
        sampling_duration += time.perf_counter() - sampling_started
        unconstrained_samples = jax.tree_util.tree_map(
            lambda previous, new: jnp.concatenate((previous, new), axis=1),
            unconstrained_samples,
            chunk_samples,
        )
        log_density = jnp.concatenate((log_density, chunk_metrics["log_density"]), axis=1)
        acceptance_rate = jnp.concatenate(
            (acceptance_rate, chunk_metrics["acceptance_rate"]), axis=1
        )
        divergent = jnp.concatenate((divergent, chunk_metrics["divergent"]), axis=1)
        energy = jnp.concatenate((energy, chunk_metrics["energy"]), axis=1)
        num_integration_steps_array = jnp.concatenate(
            (
                num_integration_steps_array,
                chunk_metrics["num_integration_steps"],
            ),
            axis=1,
        )
        num_trajectory_expansions_array = jnp.concatenate(
            (
                num_trajectory_expansions_array,
                chunk_metrics["num_trajectory_expansions"],
            ),
            axis=1,
        )
        completed += chunk
        if destination is not None and compatibility is not None:
            _write_mcmc_checkpoint(
                destination,
                compatibility=compatibility,
                completed=completed,
                current_states=current_states,
                warmup_states=warmup_states,
                step_sizes=step_sizes,
                inverse_mass_matrices=inverse_mass_matrices,
                warmup_durations=warmup_durations,
                unconstrained_samples=unconstrained_samples,
                log_density=log_density,
                acceptance_rate=acceptance_rate,
                divergent=divergent,
                energy=energy,
                num_integration_steps=num_integration_steps_array,
                num_trajectory_expansions=num_trajectory_expansions_array,
                adaptation_duration=adaptation_duration,
                sampling_duration=sampling_duration,
                duration_seconds=previous_duration + time.perf_counter() - started,
            )

    samples = problem.parameter_space.constrain(unconstrained_samples)
    diagnostics = mcmc_diagnostics(
        samples,
        acceptance_rate=acceptance_rate,
        divergent=divergent,
    )
    jax.block_until_ready(diagnostics.max_rhat)
    final_states = _unstack_tree(current_states, chains)
    warmup_state_tuple = _unstack_tree(warmup_states, chains)
    warmups = tuple(
        MCMCChainWarmup(
            state=warmup_state_tuple[index],
            step_size=step_sizes[index],
            inverse_mass_matrix=inverse_mass_matrices[index],
            num_integration_steps=(
                int(extra_parameters["num_integration_steps"])
                if algorithm == "hmc"
                else None
            ),
            duration_seconds=float(warmup_durations[index]),
        )
        for index in range(chains)
    )
    duration = previous_duration + time.perf_counter() - started
    return MCMCResult(
        problem=problem,
        samples=samples,
        unconstrained_samples=unconstrained_samples,
        log_density=log_density,
        acceptance_rate=acceptance_rate,
        divergent=divergent,
        energy=energy,
        num_integration_steps=num_integration_steps_array,
        num_trajectory_expansions=num_trajectory_expansions_array,
        final_states=final_states,
        warmup=warmups,
        diagnostics=diagnostics,
        root_key=root_key,
        chain_keys=chain_keys,
        algorithm=algorithm,
        duration_seconds=duration,
        max_num_doublings=(
            int(extra_parameters["max_num_doublings"]) if algorithm == "nuts" else None
        ),
        adaptation_duration_seconds=adaptation_duration,
        sampling_duration_seconds=sampling_duration,
        chain_method=method,
    )


def _adapt_mcmc(
    algorithm_factory,
    logdensity_fn,
    position,
    *,
    warmup_keys,
    warmup_steps,
    target_acceptance_rate,
    initial_step_size,
    is_mass_matrix_diagonal,
    extra_parameters,
    chain_method,
):
    adaptation = blackjax.window_adaptation(
        algorithm_factory,
        logdensity_fn,
        is_mass_matrix_diagonal=is_mass_matrix_diagonal,
        initial_step_size=initial_step_size,
        target_acceptance_rate=target_acceptance_rate,
        **extra_parameters,
    )
    adaptation_run: Any = adaptation.run
    if chain_method == "vectorized":
        started = time.perf_counter()

        def adapt_chain(warmup_key):
            result, _ = adaptation_run(
                warmup_key,
                position,
                num_steps=warmup_steps,
            )
            return result

        results = jax.jit(jax.vmap(adapt_chain))(warmup_keys)
        jax.block_until_ready(results.state.position)
        duration = time.perf_counter() - started
        durations = jnp.full((warmup_keys.shape[0],), duration)
        return (
            results.state,
            results.state,
            results.parameters["step_size"],
            results.parameters["inverse_mass_matrix"],
            durations,
            duration,
        )

    states = []
    step_sizes = []
    mass_matrices = []
    durations = []
    for warmup_key in warmup_keys:
        started = time.perf_counter()
        result, _ = adaptation_run(
            warmup_key,
            position,
            num_steps=warmup_steps,
        )
        jax.block_until_ready(result.state.position)
        durations.append(time.perf_counter() - started)
        states.append(result.state)
        step_sizes.append(result.parameters["step_size"])
        mass_matrices.append(result.parameters["inverse_mass_matrix"])
    batched_states = _stack_trees(states)
    duration_array = jnp.asarray(durations)
    return (
        batched_states,
        batched_states,
        jnp.stack(step_sizes),
        jnp.stack(mass_matrices),
        duration_array,
        float(sum(durations)),
    )


def _advance_mcmc(
    algorithm_factory,
    logdensity_fn,
    current_states,
    step_sizes,
    inverse_mass_matrices,
    sample_keys,
    *,
    start,
    count,
    algorithm,
    extra_parameters,
    chain_method,
):
    kernel = algorithm_factory.build_kernel()
    if algorithm == "nuts":
        max_num_doublings = int(extra_parameters["max_num_doublings"])

        def transition(draw_key, state, step_size, inverse_mass_matrix):
            return kernel(
                draw_key,
                state,
                logdensity_fn,
                step_size,
                inverse_mass_matrix,
                max_num_doublings,
            )

    else:
        num_integration_steps = int(extra_parameters["num_integration_steps"])

        def transition(draw_key, state, step_size, inverse_mass_matrix):
            return kernel(
                draw_key,
                state,
                logdensity_fn,
                step_size,
                inverse_mass_matrix,
                num_integration_steps,
            )

    indices = jnp.arange(start, start + count, dtype=jnp.uint32)
    draw_keys = jax.vmap(
        lambda sample_key: jax.vmap(lambda index: jr.fold_in(sample_key, index))(indices)
    )(sample_keys)

    def run_chain(initial_state, keys, step_size, inverse_mass_matrix):
        def one_step(state, draw_key):
            next_state, info = transition(
                draw_key,
                state,
                step_size,
                inverse_mass_matrix,
            )
            return next_state, (next_state, info)

        return jax.lax.scan(one_step, initial_state, keys)

    if chain_method == "vectorized":
        final_states, (states, infos) = jax.jit(jax.vmap(run_chain))(
            current_states,
            draw_keys,
            step_sizes,
            inverse_mass_matrices,
        )
    else:
        chain_states = _unstack_tree(current_states, int(sample_keys.shape[0]))
        final_values = []
        state_values = []
        info_values = []
        compiled = jax.jit(run_chain)
        for index, chain_state in enumerate(chain_states):
            final_state, (states, infos) = compiled(
                chain_state,
                draw_keys[index],
                step_sizes[index],
                inverse_mass_matrices[index],
            )
            final_values.append(final_state)
            state_values.append(states)
            info_values.append(infos)
        final_states = _stack_trees(final_values)
        states = _stack_trees(state_values)
        infos = _stack_trees(info_values)

    trajectory_expansions = (
        infos.num_trajectory_expansions
        if hasattr(infos, "num_trajectory_expansions")
        else jnp.zeros_like(infos.num_integration_steps)
    )
    return (
        final_states,
        states.position,
        {
            "log_density": states.logdensity,
            "acceptance_rate": infos.acceptance_rate,
            "divergent": infos.is_divergent,
            "energy": infos.energy,
            "num_integration_steps": infos.num_integration_steps,
            "num_trajectory_expansions": trajectory_expansions,
        },
    )


def _write_mcmc_checkpoint(
    destination,
    *,
    compatibility,
    completed,
    current_states,
    warmup_states,
    step_sizes,
    inverse_mass_matrices,
    warmup_durations,
    unconstrained_samples,
    log_density,
    acceptance_rate,
    divergent,
    energy,
    num_integration_steps,
    num_trajectory_expansions,
    adaptation_duration,
    sampling_duration,
    duration_seconds,
):
    arrays = {
        "step_sizes": step_sizes,
        "inverse_mass_matrices": inverse_mass_matrices,
        "warmup_durations": warmup_durations,
        "log_density": log_density,
        "acceptance_rate": acceptance_rate,
        "divergent": divergent,
        "energy": energy,
        "num_integration_steps": num_integration_steps,
        "num_trajectory_expansions": num_trajectory_expansions,
    }
    state = {
        "completed_draws": int(completed),
        "adaptation_duration_seconds": float(adaptation_duration),
        "sampling_duration_seconds": float(sampling_duration),
        "duration_seconds": float(duration_seconds),
        "current_state_tree": pack_array_tree("current_state", current_states, arrays),
        "warmup_state_tree": pack_array_tree("warmup_state", warmup_states, arrays),
        "sample_tree": pack_array_tree(
            "unconstrained_samples", unconstrained_samples, arrays
        ),
    }
    write_checkpoint_archive(
        destination,
        kind="mcmc",
        compatibility=compatibility,
        state=state,
        arrays=arrays,
    )


def _read_mcmc_checkpoint(
    source,
    *,
    compatibility,
    algorithm_factory,
    logdensity_fn,
    position,
    chains,
):
    state, arrays = read_checkpoint_archive(
        source,
        kind="mcmc",
        compatibility=compatibility,
    )
    completed = int(state.get("completed_draws", -1))
    if completed < 0:
        raise CheckpointCorruptionError("Checkpoint completed draw count is invalid.")
    one_state = algorithm_factory.init(position, logdensity_fn)
    state_template = jax.tree_util.tree_map(
        lambda value: jnp.broadcast_to(value, (chains, *value.shape)),
        one_state,
    )
    sample_template = jax.tree_util.tree_map(
        lambda value: jnp.empty((chains, completed, *value.shape), dtype=value.dtype),
        position,
    )
    current_states = unpack_array_tree(
        state["current_state_tree"], arrays, state_template
    )
    warmup_states = unpack_array_tree(state["warmup_state_tree"], arrays, state_template)
    unconstrained_samples = unpack_array_tree(
        state["sample_tree"], arrays, sample_template
    )
    expected_draw_shape = (chains, completed)
    step_sizes = _checkpoint_array(arrays, "step_sizes", leading=chains)
    inverse_mass_matrices = _checkpoint_array(
        arrays, "inverse_mass_matrices", leading=chains
    )
    warmup_durations = _checkpoint_array(arrays, "warmup_durations", shape=(chains,))
    log_density = _checkpoint_array(arrays, "log_density", shape=expected_draw_shape)
    acceptance_rate = _checkpoint_array(
        arrays, "acceptance_rate", shape=expected_draw_shape
    )
    divergent = _checkpoint_array(arrays, "divergent", shape=expected_draw_shape)
    energy = _checkpoint_array(arrays, "energy", shape=expected_draw_shape)
    integration_steps = _checkpoint_array(
        arrays, "num_integration_steps", shape=expected_draw_shape
    )
    trajectory_expansions = _checkpoint_array(
        arrays, "num_trajectory_expansions", shape=expected_draw_shape
    )
    return (
        completed,
        current_states,
        warmup_states,
        step_sizes,
        inverse_mass_matrices,
        warmup_durations,
        unconstrained_samples,
        log_density,
        acceptance_rate,
        divergent,
        energy,
        integration_steps,
        trajectory_expansions,
        float(state["adaptation_duration_seconds"]),
        float(state["sampling_duration_seconds"]),
        float(state["duration_seconds"]),
    )


def _checkpoint_array(arrays, name, *, shape=None, leading=None):
    if name not in arrays:
        raise CheckpointCorruptionError(f"Checkpoint array {name!r} is missing.")
    value = jnp.asarray(arrays[name])
    if shape is not None and value.shape != shape:
        raise CheckpointCorruptionError(
            f"Checkpoint array {name!r} has an invalid shape."
        )
    if leading is not None and (value.ndim == 0 or value.shape[0] != leading):
        raise CheckpointCorruptionError(
            f"Checkpoint array {name!r} has an invalid leading axis."
        )
    return value


def _empty_sample_tree(position, chains):
    return jax.tree_util.tree_map(
        lambda value: jnp.empty((chains, 0, *value.shape), dtype=value.dtype),
        position,
    )


def _stack_trees(values):
    return jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *values)


def _unstack_tree(tree, count):
    return tuple(
        jax.tree_util.tree_map(lambda value: value[index], tree) for index in range(count)
    )


def _tree_nbytes(tree: PyTree[Any], /) -> int:
    return sum(int(jnp.asarray(leaf).nbytes) for leaf in jax.tree_util.tree_leaves(tree))


__all__ = [
    "MCMCChainWarmup",
    "MCMCDiagnostics",
    "MCMCResult",
    "sample_hmc",
    "sample_nuts",
]
