#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, cast, Literal, NamedTuple

import blackjax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from blackjax.mcmc import (
    hmc as blackjax_hmc,
    integrators,
    trajectory,
)
from blackjax.mcmc.proposal import safe_energy_diff, static_binomial_sampling
from jaxtyping import Array, PyTree

from .._frozendict import frozendict
from .._strict import StrictModule
from ._causal_hmc import (
    _causal_block,
    CausalHMCConfig,
    CausalHMCDiagnostics,
    CausalNUTSConfig,
)
from ._chain import (
    _prepare_chain_positions,
    _split_chain_keys,
    _stack_trees,
    _tree_nbytes,
    _unstack_tree,
    _validate_chain_method,
    _validate_nuts_chain_method,
    ChainMethod,
    NUTSChainMethod,
)
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
from ._interleaved_nuts import _advance_one_quantum, _initialize_transition
from ._mcmc_kinetic import (
    MCMCMassAdaptationPlan,
    prepare_mcmc_kinetic,
    PreparedMCMCKinetic,
)
from ._posterior import PosteriorProblem
from ._posterior_predictive import (
    predict_from_position_samples,
    sample_observations_from_position_samples,
)
from ._predictive import PredictiveField


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
    causal_diagnostics: CausalHMCDiagnostics | None
    causal_config: CausalHMCConfig | CausalNUTSConfig | None
    algorithm: str = eqx.field(static=True)
    duration_seconds: float = eqx.field(static=True)
    sample_memory_bytes: int = eqx.field(static=True)
    max_num_doublings: int | None = eqx.field(static=True)
    chain_method: NUTSChainMethod = eqx.field(static=True)
    trajectory_method: Literal["sequential", "causal"] = eqx.field(static=True)
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
        chain_method: NUTSChainMethod,
        adaptation_duration_seconds: float,
        sampling_duration_seconds: float,
        trajectory_method: Literal["sequential", "causal"] = "sequential",
        causal_diagnostics: CausalHMCDiagnostics | None = None,
        causal_config: CausalHMCConfig | CausalNUTSConfig | None = None,
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
        if trajectory_method not in ("sequential", "causal"):
            raise ValueError("Unknown HMC trajectory method.")
        if causal_diagnostics is not None and not isinstance(
            causal_diagnostics, CausalHMCDiagnostics
        ):
            raise TypeError("causal_diagnostics must be CausalHMCDiagnostics or None.")
        if causal_config is not None and not isinstance(
            causal_config, (CausalHMCConfig, CausalNUTSConfig)
        ):
            raise TypeError(
                "causal_config must be CausalHMCConfig, CausalNUTSConfig, or None."
            )
        if trajectory_method == "causal" and (
            causal_diagnostics is None or causal_config is None
        ):
            raise ValueError("Causal trajectory results require config and diagnostics.")
        if trajectory_method == "sequential" and (
            causal_diagnostics is not None or causal_config is not None
        ):
            raise ValueError("Sequential trajectory results cannot carry causal state.")
        self.trajectory_method = trajectory_method
        self.causal_diagnostics = causal_diagnostics
        self.causal_config = causal_config
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
    initial_positions: PyTree[Any] | None = None,
    target_acceptance_rate: float = 0.8,
    initial_step_size: float = 1.0,
    kinetic: MCMCMassAdaptationPlan | None = None,
    max_num_doublings: int = 10,
    chain_method: NUTSChainMethod = "sequential",
    trajectory: Literal["sequential", "causal"] = "sequential",
    causal_config: CausalNUTSConfig | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int | None = None,
    checkpoint_id: str | None = None,
    resume_from: str | Path | None = None,
) -> MCMCResult:
    """Run independently adapted BlackJAX No-U-Turn sampler chains.

    ``chain_method=\"interleaved\"`` keeps vectorized warmup and lets production
    chains cross draw boundaries independently within each sampling chunk. It
    targets many-chain accelerator workloads with unequal NUTS trajectory lengths;
    sequential and vectorized execution remain available for cheaper targets.
    """
    if int(max_num_doublings) <= 0:
        raise ValueError("max_num_doublings must be positive.")
    if trajectory not in ("sequential", "causal"):
        raise ValueError("trajectory must be 'sequential' or 'causal'.")
    causal = (
        CausalNUTSConfig(max_num_doublings=int(max_num_doublings))
        if trajectory == "causal" and causal_config is None
        else causal_config
    )
    if trajectory == "causal" and not isinstance(causal, CausalNUTSConfig):
        raise TypeError("causal_config must be CausalNUTSConfig for causal NUTS.")
    if trajectory == "causal" and causal.max_num_doublings != int(max_num_doublings):
        raise ValueError(
            "causal_config.max_num_doublings must agree with max_num_doublings."
        )
    if trajectory == "sequential" and causal_config is not None:
        raise ValueError("causal_config requires trajectory='causal'.")
    return _sample_mcmc(
        problem,
        key=key,
        algorithm="nuts",
        num_chains=num_chains,
        num_warmup=num_warmup,
        num_samples=num_samples,
        initial_position=initial_position,
        initial_positions=initial_positions,
        target_acceptance_rate=target_acceptance_rate,
        initial_step_size=initial_step_size,
        kinetic_plan=(MCMCMassAdaptationPlan.diagonal() if kinetic is None else kinetic),
        extra_parameters={"max_num_doublings": int(max_num_doublings)},
        chain_method=chain_method,
        trajectory_method=trajectory,
        causal_config=causal,
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
    initial_positions: PyTree[Any] | None = None,
    target_acceptance_rate: float = 0.8,
    initial_step_size: float = 1.0,
    kinetic: MCMCMassAdaptationPlan | None = None,
    chain_method: ChainMethod = "sequential",
    trajectory_method: Literal["sequential", "causal"] = "sequential",
    causal_config: CausalHMCConfig | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int | None = None,
    checkpoint_id: str | None = None,
    resume_from: str | Path | None = None,
) -> MCMCResult:
    """Run independently adapted fixed-trajectory BlackJAX HMC chains."""
    if int(num_integration_steps) <= 0:
        raise ValueError("num_integration_steps must be positive.")
    if trajectory_method not in ("sequential", "causal"):
        raise ValueError("trajectory_method must be 'sequential' or 'causal'.")
    kinetic_plan = MCMCMassAdaptationPlan.diagonal() if kinetic is None else kinetic
    if not isinstance(kinetic_plan, MCMCMassAdaptationPlan):
        raise TypeError("kinetic must be MCMCMassAdaptationPlan or None.")
    if trajectory_method == "causal":
        causal = CausalHMCConfig() if causal_config is None else causal_config
        if not isinstance(causal, CausalHMCConfig):
            raise TypeError("causal_config must be CausalHMCConfig or None.")
        if kinetic_plan.kind != "diagonal" and causal.linearization != "dense-exact":
            raise ValueError("Structured causal HMC requires dense-exact linearization.")
    else:
        if causal_config is not None:
            raise ValueError("causal_config requires trajectory_method='causal'.")
        causal = None
    return _sample_mcmc(
        problem,
        key=key,
        algorithm="hmc",
        num_chains=num_chains,
        num_warmup=num_warmup,
        num_samples=num_samples,
        initial_position=initial_position,
        initial_positions=initial_positions,
        target_acceptance_rate=target_acceptance_rate,
        initial_step_size=initial_step_size,
        kinetic_plan=kinetic_plan,
        extra_parameters={"num_integration_steps": int(num_integration_steps)},
        chain_method=chain_method,
        trajectory_method=trajectory_method,
        causal_config=causal,
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
    initial_positions: PyTree[Any] | None,
    target_acceptance_rate: float,
    initial_step_size: float,
    kinetic_plan: MCMCMassAdaptationPlan,
    extra_parameters: dict[str, Any],
    chain_method: NUTSChainMethod,
    trajectory_method: Literal["sequential", "causal"],
    causal_config: CausalHMCConfig | CausalNUTSConfig | None,
    checkpoint_path: str | Path | None,
    checkpoint_every: int | None,
    checkpoint_id: str | None,
    resume_from: str | Path | None,
) -> MCMCResult:
    if not isinstance(problem, PosteriorProblem):
        raise TypeError("problem must be a PosteriorProblem.")
    if not isinstance(kinetic_plan, MCMCMassAdaptationPlan):
        raise TypeError("kinetic must be MCMCMassAdaptationPlan or None.")
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
    method = (
        _validate_nuts_chain_method(chain_method)
        if algorithm == "nuts"
        else _validate_chain_method(cast(ChainMethod, chain_method))
    )
    if trajectory_method == "causal":
        expected = CausalNUTSConfig if algorithm == "nuts" else CausalHMCConfig
        if not isinstance(causal_config, expected):
            raise ValueError(f"Causal {algorithm.upper()} requires {expected.__name__}.")
    elif causal_config is not None:
        raise ValueError("causal_config requires causal trajectory execution.")
    if resume_from is not None and (
        initial_position is not None or initial_positions is not None
    ):
        raise ValueError(
            "initial_position and initial_positions cannot be supplied when "
            "resuming MCMC."
        )

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

    position, chain_positions = _prepare_chain_positions(
        problem.initial_position,
        num_chains=chains,
        initial_position=initial_position,
        initial_positions=initial_positions,
    )
    if any(
        bool(jnp.any(~jnp.isfinite(jnp.asarray(leaf))))
        for leaf in jax.tree_util.tree_leaves(chain_positions)
    ):
        raise FloatingPointError("Initial MCMC positions must be finite.")
    values, gradient = jax.vmap(jax.value_and_grad(problem.log_density))(chain_positions)
    if not bool(jnp.all(jnp.isfinite(values))) or any(
        bool(jnp.any(~jnp.isfinite(jnp.asarray(leaf))))
        for leaf in jax.tree_util.tree_leaves(gradient)
    ):
        raise FloatingPointError("Initial MCMC log density and gradient must be finite.")

    root_key, chain_keys = _split_chain_keys(key, chains)
    split_keys = jax.vmap(lambda chain_key: jr.split(chain_key, 2))(chain_keys)
    warmup_keys = split_keys[:, 0]
    sample_keys = split_keys[:, 1]
    settings = {
        "algorithm": algorithm,
        "num_chains": chains,
        "num_warmup": warmup_steps,
        "target_acceptance_rate": target,
        "initial_step_size": initial_step,
        "kinetic_kind": kinetic_plan.kind,
        "kinetic": {
            "kind": kinetic_plan.kind,
            "parameter_blocks": kinetic_plan.parameter_blocks,
            "max_block_size": kinetic_plan.max_block_size,
            "rank": kinetic_plan.rank,
            "memory_cap_bytes": kinetic_plan.memory_cap_bytes,
        },
        "chain_method": method,
        "extra_parameters": extra_parameters,
        "trajectory_method": trajectory_method,
        "causal_config": (None if causal_config is None else causal_config.as_dict()),
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
    advance_mcmc = None
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
            chain_positions,
            warmup_keys=warmup_keys,
            warmup_steps=warmup_steps,
            target_acceptance_rate=target,
            initial_step_size=initial_step,
            is_mass_matrix_diagonal=kinetic_plan.kind == "diagonal",
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
        causal_converged = jnp.empty((chains, 0), dtype=bool)
        causal_fallback_used = jnp.empty((chains, 0), dtype=bool)
        causal_outer_iterations = jnp.empty((chains, 0), dtype=jnp.int32)
        causal_maximum_residual = jnp.empty((chains, 0), dtype=float)
        causal_accepted_steps = jnp.empty((chains, 0), dtype=jnp.int32)
        causal_rejected_steps = jnp.empty((chains, 0), dtype=jnp.int32)
        causal_transition_evaluations = jnp.empty((chains, 0), dtype=jnp.int32)
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
                causal_converged=causal_converged,
                causal_fallback_used=causal_fallback_used,
                causal_outer_iterations=causal_outer_iterations,
                causal_maximum_residual=causal_maximum_residual,
                causal_accepted_steps=causal_accepted_steps,
                causal_rejected_steps=causal_rejected_steps,
                causal_transition_evaluations=causal_transition_evaluations,
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
            causal_converged,
            causal_fallback_used,
            causal_outer_iterations,
            causal_maximum_residual,
            causal_accepted_steps,
            causal_rejected_steps,
            causal_transition_evaluations,
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

    prepared_kinetics = tuple(
        _prepare_warmup_kinetic(
            position,
            kinetic_plan,
            inverse_mass_matrices[index],
        )
        for index in range(chains)
    )
    advance_mcmc = _build_prepared_mcmc_advancer(
        logdensity_fn,
        prepared_kinetics,
        algorithm=algorithm,
        extra_parameters=extra_parameters,
        trajectory_method=trajectory_method,
        causal_config=causal_config,
        chain_method=method,
    )
    while completed < draws:
        chunk = min(interval, draws - completed)
        sampling_started = time.perf_counter()
        current_states, chunk_samples, chunk_metrics = advance_mcmc(
            current_states,
            step_sizes,
            inverse_mass_matrices,
            sample_keys,
            start=completed,
            count=chunk,
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
        causal_converged = jnp.concatenate(
            (causal_converged, chunk_metrics["causal_converged"]), axis=1
        )
        causal_fallback_used = jnp.concatenate(
            (causal_fallback_used, chunk_metrics["causal_fallback_used"]), axis=1
        )
        causal_outer_iterations = jnp.concatenate(
            (causal_outer_iterations, chunk_metrics["causal_outer_iterations"]),
            axis=1,
        )
        causal_maximum_residual = jnp.concatenate(
            (causal_maximum_residual, chunk_metrics["causal_maximum_residual"]),
            axis=1,
        )
        causal_accepted_steps = jnp.concatenate(
            (causal_accepted_steps, chunk_metrics["causal_accepted_steps"]), axis=1
        )
        causal_rejected_steps = jnp.concatenate(
            (causal_rejected_steps, chunk_metrics["causal_rejected_steps"]), axis=1
        )
        causal_transition_evaluations = jnp.concatenate(
            (
                causal_transition_evaluations,
                chunk_metrics["causal_transition_evaluations"],
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
                causal_converged=causal_converged,
                causal_fallback_used=causal_fallback_used,
                causal_outer_iterations=causal_outer_iterations,
                causal_maximum_residual=causal_maximum_residual,
                causal_accepted_steps=causal_accepted_steps,
                causal_rejected_steps=causal_rejected_steps,
                causal_transition_evaluations=causal_transition_evaluations,
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
    causal_diagnostics = (
        CausalHMCDiagnostics(
            converged=causal_converged,
            fallback_used=causal_fallback_used,
            outer_iterations=causal_outer_iterations,
            maximum_residual=causal_maximum_residual,
            accepted_nonlinear_steps=causal_accepted_steps,
            rejected_nonlinear_steps=causal_rejected_steps,
            transition_evaluations=causal_transition_evaluations,
        )
        if trajectory_method == "causal"
        else None
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
        trajectory_method=trajectory_method,
        causal_diagnostics=causal_diagnostics,
        causal_config=causal_config,
    )


def _adapt_mcmc(
    algorithm_factory,
    logdensity_fn,
    positions,
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
    adaptation_run = cast(Any, adaptation.run)
    if chain_method in ("vectorized", "interleaved"):
        started = time.perf_counter()

        def adapt_chain(warmup_key, chain_position):
            result, _ = adaptation_run(
                warmup_key,
                chain_position,
                num_steps=warmup_steps,
            )
            return result

        results = jax.jit(jax.vmap(adapt_chain))(warmup_keys, positions)
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

    chain_positions = _unstack_tree(positions, int(warmup_keys.shape[0]))
    states = []
    step_sizes = []
    mass_matrices = []
    durations = []
    for warmup_key, chain_position in zip(warmup_keys, chain_positions, strict=True):
        started = time.perf_counter()
        result, _ = adaptation_run(
            warmup_key,
            chain_position,
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


class _PreparedMetric(NamedTuple):
    sample_momentum: Any
    kinetic_energy: Any
    check_turning: Any


class _CausalRecord(NamedTuple):
    converged: Array
    fallback_used: Array
    outer_iterations: Array
    maximum_residual: Array
    accepted_steps: Array
    rejected_steps: Array
    transition_evaluations: Array


class _CausalIntegratorState(NamedTuple):
    position: Any
    momentum: Any
    logdensity: Array
    logdensity_grad: Any
    causal: _CausalRecord


class _PreparedHMCInfo(NamedTuple):
    momentum: Any
    acceptance_rate: Array
    is_accepted: Array
    is_divergent: Array
    energy: Array
    proposal: Any
    num_integration_steps: Array
    causal_converged: Array
    causal_fallback_used: Array
    causal_outer_iterations: Array
    causal_maximum_residual: Array
    causal_accepted_steps: Array
    causal_rejected_steps: Array
    causal_transition_evaluations: Array


class _PreparedNUTSInfo(NamedTuple):
    acceptance_rate: Array
    is_divergent: Array
    energy: Array
    num_integration_steps: Array
    num_trajectory_expansions: Array
    causal_converged: Array
    causal_fallback_used: Array
    causal_outer_iterations: Array
    causal_maximum_residual: Array
    causal_accepted_steps: Array
    causal_rejected_steps: Array
    causal_transition_evaluations: Array


class _PreparedNUTSBuffers(NamedTuple):
    position: Any
    logdensity: Array
    acceptance_rate: Array
    is_divergent: Array
    energy: Array
    num_integration_steps: Array
    num_trajectory_expansions: Array
    causal_converged: Array
    causal_fallback_used: Array
    causal_outer_iterations: Array
    causal_maximum_residual: Array
    causal_accepted_steps: Array
    causal_rejected_steps: Array
    causal_transition_evaluations: Array


class _PreparedNUTSSchedulerCarry(NamedTuple):
    continuations: Any
    completed: Array
    buffers: _PreparedNUTSBuffers


def _prepare_warmup_kinetic(
    reference: PyTree[Any],
    plan: MCMCMassAdaptationPlan,
    inverse_mass_matrix: Array,
    /,
) -> PreparedMCMCKinetic:
    matrix = jnp.asarray(inverse_mass_matrix)
    if plan.kind == "diagonal":
        diagonal = jnp.diag(matrix) if matrix.ndim == 2 else matrix
        return prepare_mcmc_kinetic(reference, plan, diagonal=diagonal)
    if matrix.ndim != 2:
        raise ValueError(
            "Structured MCMC warmup must produce a dense covariance estimate."
        )
    if plan.kind == "blocks":
        template = prepare_mcmc_kinetic(
            reference,
            MCMCMassAdaptationPlan.diagonal(memory_cap_bytes=plan.memory_cap_bytes),
            diagonal=jnp.diag(matrix),
        )
        offsets: dict[str, Array] = {}
        start = 0
        for path, shape in zip(
            template.subspace.leaf_paths,
            template.subspace.leaf_shapes,
            strict=True,
        ):
            size = 1
            for extent in shape:
                size *= int(extent)
            offsets[path] = jnp.arange(start, start + size, dtype=jnp.int32)
            start += size
        blocks = []
        for paths in plan.parameter_blocks:
            indices = jnp.concatenate(tuple(offsets[path] for path in paths))
            blocks.append(matrix[indices[:, None], indices[None, :]])
        return prepare_mcmc_kinetic(
            reference,
            plan,
            diagonal=jnp.diag(matrix),
            block_inverse_masses=tuple(blocks),
        )
    eigenvalues, eigenvectors = jnp.linalg.eigh(matrix)
    baseline = jnp.maximum(
        jnp.min(eigenvalues),
        jnp.finfo(eigenvalues.dtype).eps,
    )
    residual = jnp.maximum(eigenvalues[-plan.rank :] - baseline, 0.0)
    factor = eigenvectors[:, -plan.rank :] * jnp.sqrt(residual)[None, :]
    return prepare_mcmc_kinetic(
        reference,
        plan,
        diagonal=jnp.full((matrix.shape[0],), baseline),
        low_rank_factor=factor,
    )


def _prepared_metric(kinetic: PreparedMCMCKinetic, /) -> _PreparedMetric:
    def sample_momentum(key, position):
        del position
        return kinetic.sample_momentum(key)

    def kinetic_energy(momentum, position=None):
        del position
        return kinetic.kinetic_energy_vector(kinetic.pack(momentum))

    def check_turning(
        momentum_left,
        momentum_right,
        momentum_sum,
        position_left=None,
        position_right=None,
    ):
        del position_left, position_right
        left_momentum_vector = jnp.asarray(momentum_left)
        right_momentum_vector = jnp.asarray(momentum_right)
        total = jnp.asarray(momentum_sum)
        left = kinetic.inverse_mass_action_vector(left_momentum_vector)
        right = kinetic.inverse_mass_action_vector(right_momentum_vector)
        rho = total - 0.5 * (left_momentum_vector + right_momentum_vector)
        return (jnp.vdot(left, rho) <= 0.0) | (jnp.vdot(right, rho) <= 0.0)

    return _PreparedMetric(sample_momentum, kinetic_energy, check_turning)


def _empty_causal_record(value: Array, /, *, converged: bool) -> _CausalRecord:
    zero = jnp.zeros_like(value, dtype=jnp.int32)
    return _CausalRecord(
        converged=jnp.asarray(converged),
        fallback_used=jnp.asarray(False),
        outer_iterations=zero,
        maximum_residual=jnp.zeros_like(value),
        accepted_steps=zero,
        rejected_steps=zero,
        transition_evaluations=zero,
    )


def _merge_causal_records(left: _CausalRecord, right: _CausalRecord) -> _CausalRecord:
    return _CausalRecord(
        converged=left.converged & right.converged,
        fallback_used=left.fallback_used | right.fallback_used,
        outer_iterations=jnp.maximum(left.outer_iterations, right.outer_iterations),
        maximum_residual=jnp.maximum(left.maximum_residual, right.maximum_residual),
        accepted_steps=left.accepted_steps + right.accepted_steps,
        rejected_steps=left.rejected_steps + right.rejected_steps,
        transition_evaluations=(
            left.transition_evaluations + right.transition_evaluations
        ),
    )


def _causal_continuation(continuation):
    state = continuation.global_proposal.state
    initial = _CausalIntegratorState(
        position=state.position,
        momentum=state.momentum,
        logdensity=state.logdensity,
        logdensity_grad=state.logdensity_grad,
        causal=_empty_causal_record(state.logdensity, converged=True),
    )
    global_proposal = continuation.global_proposal._replace(state=initial)
    global_trajectory = continuation.global_trajectory._replace(
        leftmost_state=initial,
        rightmost_state=initial,
    )
    local_proposal = continuation.local_proposal._replace(state=initial)
    local_trajectory = continuation.local_trajectory._replace(
        leftmost_state=initial,
        rightmost_state=initial,
    )
    return continuation._replace(
        global_proposal=global_proposal,
        global_trajectory=global_trajectory,
        local_proposal=local_proposal,
        local_trajectory=local_trajectory,
    )


def _causal_trajectory_record(continuation) -> _CausalRecord:
    trajectory_ = continuation.global_trajectory
    return _merge_causal_records(
        trajectory_.leftmost_state.causal,
        trajectory_.rightmost_state.causal,
    )


def _causal_prepared_integrator(
    logdensity_fn,
    metric: _PreparedMetric,
    kinetic: PreparedMCMCKinetic,
    config: CausalHMCConfig,
    probe_key: Array,
):
    def step(state: _CausalIntegratorState, step_size):
        block_probe_key = jr.fold_in(
            probe_key,
            state.causal.transition_evaluations,
        )
        phase, values = _causal_block(
            logdensity_fn,
            metric,
            kinetic.diagonal,
            step_size,
            (state.position, state.momentum),
            1,
            block_probe_key,
            config,
        )
        position, momentum = phase
        logdensity, gradient = jax.value_and_grad(logdensity_fn)(position)
        current = state.causal
        block = _CausalRecord(*values)
        return _CausalIntegratorState(
            position=position,
            momentum=momentum,
            logdensity=logdensity,
            logdensity_grad=gradient,
            causal=_merge_causal_records(current, block),
        )

    return step


def _prepared_hmc_transition(
    key,
    state,
    logdensity_fn,
    step_size,
    kinetic: PreparedMCMCKinetic,
    num_integration_steps: int,
    *,
    causal_config: CausalHMCConfig | None,
):
    metric = _prepared_metric(kinetic)
    momentum_key, integration_key, acceptance_key = jr.split(key, 3)
    momentum = metric.sample_momentum(momentum_key, state.position)
    initial = integrators.IntegratorState(
        state.position, momentum, state.logdensity, state.logdensity_grad
    )
    if causal_config is None:
        integrator = integrators.velocity_verlet(logdensity_fn, metric.kinetic_energy)
        final = jax.lax.fori_loop(
            0,
            int(num_integration_steps),
            lambda _, current: integrator(current, step_size),
            initial,
        )
        causal_record = _empty_causal_record(state.logdensity, converged=False)
    else:
        phase = (state.position, momentum)
        causal_record = _empty_causal_record(state.logdensity, converged=True)
        block_start = 0
        block_index = 0
        while block_start < int(num_integration_steps):
            block_length = min(
                causal_config.trajectory_block_size,
                int(num_integration_steps) - block_start,
            )
            phase, values = _causal_block(
                logdensity_fn,
                metric,
                kinetic.diagonal,
                step_size,
                phase,
                block_length,
                jr.fold_in(integration_key, block_index),
                causal_config,
            )
            causal_record = _merge_causal_records(
                causal_record,
                _CausalRecord(*values),
            )
            block_start += block_length
            block_index += 1
        position, final_momentum = phase
        logdensity, gradient = jax.value_and_grad(logdensity_fn)(position)
        final = integrators.IntegratorState(
            position,
            final_momentum,
            logdensity,
            gradient,
        )
    final = blackjax_hmc.flip_momentum(final)
    initial_energy = trajectory.hmc_energy(metric.kinetic_energy)(initial)
    proposed_energy = trajectory.hmc_energy(metric.kinetic_energy)(final)
    delta = safe_energy_diff(initial_energy, proposed_energy)
    divergent = -delta > 1000.0
    selected, acceptance_info = static_binomial_sampling(
        acceptance_key, delta, initial, final
    )
    accepted, acceptance_rate, _ = acceptance_info
    next_state = blackjax_hmc.HMCState(
        selected.position, selected.logdensity, selected.logdensity_grad
    )
    return next_state, _PreparedHMCInfo(
        momentum=momentum,
        acceptance_rate=acceptance_rate,
        is_accepted=accepted,
        is_divergent=divergent,
        energy=proposed_energy,
        proposal=final,
        num_integration_steps=jnp.asarray(num_integration_steps, dtype=jnp.int32),
        causal_converged=causal_record.converged,
        causal_fallback_used=causal_record.fallback_used,
        causal_outer_iterations=causal_record.outer_iterations,
        causal_maximum_residual=causal_record.maximum_residual,
        causal_accepted_steps=causal_record.accepted_steps,
        causal_rejected_steps=causal_record.rejected_steps,
        causal_transition_evaluations=causal_record.transition_evaluations,
    )


def _prepared_nuts_transition(
    key,
    state,
    logdensity_fn,
    step_size,
    kinetic: PreparedMCMCKinetic,
    max_num_doublings: int,
    *,
    causal_config: CausalNUTSConfig | None,
):
    metric = _prepared_metric(kinetic)
    momentum_key, integrator_key = jr.split(key)
    momentum = metric.sample_momentum(momentum_key, state.position)
    continuation = _initialize_transition(
        state,
        momentum,
        metric.kinetic_energy(momentum),
        integrator_key,
        max_num_doublings=max_num_doublings,
    )
    if causal_config is None:
        integrator_override = None
    else:
        continuation = _causal_continuation(continuation)
        integrator_override = _causal_prepared_integrator(
            logdensity_fn,
            metric,
            kinetic,
            causal_config.recurrence,
            integrator_key,
        )

    def advance(current):
        return _advance_one_quantum(
            current,
            step_size,
            kinetic.diagonal,
            logdensity_fn=logdensity_fn,
            max_num_doublings=max_num_doublings,
            divergence_threshold=1000.0,
            metric_override=metric,
            integrator_override=integrator_override,
        )

    continuation, emitted, emission = advance(continuation)
    continuation, emitted, emission = jax.lax.while_loop(
        lambda carry: ~carry[1],
        lambda carry: advance(carry[0]),
        (continuation, emitted, emission),
    )
    del emitted
    causal_record = (
        _empty_causal_record(emission.logdensity, converged=False)
        if causal_config is None
        else _causal_trajectory_record(continuation)
    )
    return emission.state, _PreparedNUTSInfo(
        acceptance_rate=emission.acceptance_rate,
        is_divergent=emission.is_divergent,
        energy=emission.energy,
        num_integration_steps=emission.num_integration_steps,
        num_trajectory_expansions=emission.num_trajectory_expansions,
        causal_converged=causal_record.converged,
        causal_fallback_used=causal_record.fallback_used,
        causal_outer_iterations=causal_record.outer_iterations,
        causal_maximum_residual=causal_record.maximum_residual,
        causal_accepted_steps=causal_record.accepted_steps,
        causal_rejected_steps=causal_record.rejected_steps,
        causal_transition_evaluations=causal_record.transition_evaluations,
    )


def _choose_prepared_value(condition, when_true, when_false):
    return jax.lax.cond(
        condition,
        lambda _: when_true,
        lambda _: when_false,
        operand=None,
    )


def _index_prepared_tree(values, index):
    return jax.tree_util.tree_map(lambda value: value[index], values)


def _write_prepared_nuts_buffer(
    buffers: _PreparedNUTSBuffers,
    emission,
    causal_record: _CausalRecord,
    index,
    should_write,
):
    safe_index = jnp.minimum(index, buffers.logdensity.shape[0] - 1)

    def write(_):
        return _PreparedNUTSBuffers(
            position=jax.tree_util.tree_map(
                lambda buffer, value: buffer.at[safe_index].set(value),
                buffers.position,
                emission.state.position,
            ),
            logdensity=buffers.logdensity.at[safe_index].set(emission.logdensity),
            acceptance_rate=buffers.acceptance_rate.at[safe_index].set(
                emission.acceptance_rate
            ),
            is_divergent=buffers.is_divergent.at[safe_index].set(emission.is_divergent),
            energy=buffers.energy.at[safe_index].set(emission.energy),
            num_integration_steps=buffers.num_integration_steps.at[safe_index].set(
                emission.num_integration_steps
            ),
            num_trajectory_expansions=buffers.num_trajectory_expansions.at[
                safe_index
            ].set(emission.num_trajectory_expansions),
            causal_converged=buffers.causal_converged.at[safe_index].set(
                causal_record.converged
            ),
            causal_fallback_used=buffers.causal_fallback_used.at[safe_index].set(
                causal_record.fallback_used
            ),
            causal_outer_iterations=buffers.causal_outer_iterations.at[safe_index].set(
                causal_record.outer_iterations
            ),
            causal_maximum_residual=buffers.causal_maximum_residual.at[safe_index].set(
                causal_record.maximum_residual
            ),
            causal_accepted_steps=buffers.causal_accepted_steps.at[safe_index].set(
                causal_record.accepted_steps
            ),
            causal_rejected_steps=buffers.causal_rejected_steps.at[safe_index].set(
                causal_record.rejected_steps
            ),
            causal_transition_evaluations=(
                buffers.causal_transition_evaluations.at[safe_index].set(
                    causal_record.transition_evaluations
                )
            ),
        )

    return jax.lax.cond(
        should_write,
        write,
        lambda _: buffers,
        operand=None,
    )


def _build_prepared_interleaved_nuts_advancer(
    logdensity_fn,
    kinetic_arrays,
    *,
    max_num_doublings: int,
    causal_config: CausalNUTSConfig | None,
):
    def run(current_states, step_sizes, draw_keys):
        chains, count = draw_keys.shape

        def prepare_chain(state, keys, kinetic):
            metric = _prepared_metric(kinetic)
            split_keys = jax.vmap(lambda key: jr.split(key, 2))(keys)
            momenta = jax.vmap(metric.sample_momentum, in_axes=(0, None))(
                split_keys[:, 0],
                state.position,
            )
            kinetic_energies = jax.vmap(metric.kinetic_energy)(momenta)
            return momenta, kinetic_energies, split_keys[:, 1]

        momenta, kinetic_energies, integrator_keys = jax.vmap(prepare_chain)(
            current_states,
            draw_keys,
            kinetic_arrays,
        )
        first_momenta = jax.tree_util.tree_map(lambda value: value[:, 0], momenta)

        def initialize(state, momentum, kinetic_energy, integrator_key):
            continuation = _initialize_transition(
                state,
                momentum,
                kinetic_energy,
                integrator_key,
                max_num_doublings=max_num_doublings,
            )
            return (
                continuation
                if causal_config is None
                else _causal_continuation(continuation)
            )

        continuations = jax.vmap(initialize)(
            current_states,
            first_momenta,
            kinetic_energies[:, 0],
            integrator_keys[:, 0],
        )
        scalar_shape = (chains, count)
        scalar_dtype = current_states.logdensity.dtype
        buffers = _PreparedNUTSBuffers(
            position=jax.tree_util.tree_map(
                lambda value: jnp.zeros(
                    (chains, count, *value.shape[1:]),
                    dtype=value.dtype,
                ),
                current_states.position,
            ),
            logdensity=jnp.zeros(scalar_shape, dtype=scalar_dtype),
            acceptance_rate=jnp.zeros(scalar_shape, dtype=scalar_dtype),
            is_divergent=jnp.zeros(scalar_shape, dtype=bool),
            energy=jnp.zeros(scalar_shape, dtype=scalar_dtype),
            num_integration_steps=jnp.zeros(scalar_shape, dtype=jnp.int32),
            num_trajectory_expansions=jnp.zeros(scalar_shape, dtype=jnp.int32),
            causal_converged=jnp.zeros(scalar_shape, dtype=bool),
            causal_fallback_used=jnp.zeros(scalar_shape, dtype=bool),
            causal_outer_iterations=jnp.zeros(scalar_shape, dtype=jnp.int32),
            causal_maximum_residual=jnp.zeros(scalar_shape, dtype=scalar_dtype),
            causal_accepted_steps=jnp.zeros(scalar_shape, dtype=jnp.int32),
            causal_rejected_steps=jnp.zeros(scalar_shape, dtype=jnp.int32),
            causal_transition_evaluations=jnp.zeros(
                scalar_shape,
                dtype=jnp.int32,
            ),
        )
        initial = _PreparedNUTSSchedulerCarry(
            continuations=continuations,
            completed=jnp.zeros((chains,), dtype=jnp.int32),
            buffers=buffers,
        )

        def has_unfinished_chains(carry):
            return jnp.any(carry.completed < count)

        def advance_chains(carry):
            def advance_one(continuation, step_size, kinetic):
                metric = _prepared_metric(kinetic)
                integrator_override = (
                    None
                    if causal_config is None
                    else _causal_prepared_integrator(
                        logdensity_fn,
                        metric,
                        kinetic,
                        causal_config.recurrence,
                        continuation.integrator_key,
                    )
                )
                return _advance_one_quantum(
                    continuation,
                    step_size,
                    kinetic.diagonal,
                    logdensity_fn=logdensity_fn,
                    max_num_doublings=max_num_doublings,
                    divergence_threshold=1000.0,
                    metric_override=metric,
                    integrator_override=integrator_override,
                )

            raw_continuations, raw_emitted, emissions = jax.vmap(advance_one)(
                carry.continuations,
                step_sizes,
                kinetic_arrays,
            )
            active = carry.completed < count
            emitted = raw_emitted & active
            continuations_after_work = jax.vmap(_choose_prepared_value)(
                active,
                raw_continuations,
                carry.continuations,
            )
            if causal_config is None:
                causal_records = jax.vmap(
                    lambda value: _empty_causal_record(value, converged=False)
                )(emissions.logdensity)
            else:
                causal_records = jax.vmap(_causal_trajectory_record)(raw_continuations)
            buffers_after_write = jax.vmap(_write_prepared_nuts_buffer)(
                carry.buffers,
                emissions,
                causal_records,
                carry.completed,
                emitted,
            )
            completed = carry.completed + emitted.astype(jnp.int32)
            next_indices = jnp.minimum(completed, count - 1)
            next_momenta = jax.vmap(_index_prepared_tree)(momenta, next_indices)
            next_kinetic_energies = jax.vmap(lambda values, index: values[index])(
                kinetic_energies, next_indices
            )
            next_integrator_keys = jax.vmap(lambda values, index: values[index])(
                integrator_keys,
                next_indices,
            )
            next_continuations = jax.vmap(initialize)(
                emissions.state,
                next_momenta,
                next_kinetic_energies,
                next_integrator_keys,
            )
            needs_next_transition = emitted & (completed < count)
            continuations = jax.vmap(_choose_prepared_value)(
                needs_next_transition,
                next_continuations,
                continuations_after_work,
            )
            return _PreparedNUTSSchedulerCarry(
                continuations=continuations,
                completed=completed,
                buffers=buffers_after_write,
            )

        result = jax.lax.while_loop(has_unfinished_chains, advance_chains, initial)
        metrics = {
            "log_density": result.buffers.logdensity,
            "acceptance_rate": result.buffers.acceptance_rate,
            "divergent": result.buffers.is_divergent,
            "energy": result.buffers.energy,
            "num_integration_steps": result.buffers.num_integration_steps,
            "num_trajectory_expansions": result.buffers.num_trajectory_expansions,
            "causal_converged": result.buffers.causal_converged,
            "causal_fallback_used": result.buffers.causal_fallback_used,
            "causal_outer_iterations": result.buffers.causal_outer_iterations,
            "causal_maximum_residual": result.buffers.causal_maximum_residual,
            "causal_accepted_steps": result.buffers.causal_accepted_steps,
            "causal_rejected_steps": result.buffers.causal_rejected_steps,
            "causal_transition_evaluations": (
                result.buffers.causal_transition_evaluations
            ),
        }
        return (
            result.continuations.current_state,
            result.buffers.position,
            metrics,
        )

    return jax.jit(run)


def _prepared_metrics(states, infos, *, algorithm: Literal["nuts", "hmc"]):
    expansions = (
        infos.num_trajectory_expansions
        if algorithm == "nuts"
        else jnp.zeros_like(infos.num_integration_steps)
    )
    return {
        "log_density": states.logdensity,
        "acceptance_rate": infos.acceptance_rate,
        "divergent": infos.is_divergent,
        "energy": infos.energy,
        "num_integration_steps": infos.num_integration_steps,
        "num_trajectory_expansions": expansions,
        "causal_converged": infos.causal_converged,
        "causal_fallback_used": infos.causal_fallback_used,
        "causal_outer_iterations": infos.causal_outer_iterations,
        "causal_maximum_residual": infos.causal_maximum_residual,
        "causal_accepted_steps": infos.causal_accepted_steps,
        "causal_rejected_steps": infos.causal_rejected_steps,
        "causal_transition_evaluations": infos.causal_transition_evaluations,
    }


def _build_prepared_mcmc_advancer(
    logdensity_fn,
    kinetics: tuple[PreparedMCMCKinetic, ...],
    /,
    *,
    algorithm: Literal["nuts", "hmc"],
    extra_parameters: dict[str, Any],
    chain_method: NUTSChainMethod,
    trajectory_method: Literal["sequential", "causal"],
    causal_config: CausalHMCConfig | CausalNUTSConfig | None,
):
    if not kinetics:
        raise ValueError("At least one prepared MCMC kinetic is required.")
    template = kinetics[0]
    kinetic_arrays = _stack_trees(
        tuple(
            PreparedMCMCKinetic(
                diagonal=kinetic.diagonal,
                low_rank_factor=kinetic.low_rank_factor,
                block_factors=kinetic.block_factors,
                block_indices=kinetic.block_indices,
                subspace=kinetic.subspace,
                kind=template.kind,
                parameter_count=template.parameter_count,
                block_paths=template.block_paths,
                rank=template.rank,
                memory_bytes=template.memory_bytes,
                condition_estimate=template.condition_estimate,
            )
            for kinetic in kinetics
        )
    )

    causal_nuts = causal_config if isinstance(causal_config, CausalNUTSConfig) else None
    causal_hmc = causal_config if isinstance(causal_config, CausalHMCConfig) else None
    if algorithm == "nuts":
        depth = int(extra_parameters["max_num_doublings"])

        def transition(key, state, step, kinetic):
            return _prepared_nuts_transition(
                key,
                state,
                logdensity_fn,
                step,
                kinetic,
                depth,
                causal_config=causal_nuts,
            )

    else:
        steps = int(extra_parameters["num_integration_steps"])

        def transition(key, state, step, kinetic):
            return _prepared_hmc_transition(
                key,
                state,
                logdensity_fn,
                step,
                kinetic,
                steps,
                causal_config=causal_hmc,
            )

    def run_chain(initial_state, keys, step_size, dynamic_kinetic):
        def one_step(current, draw_key):
            next_state, info = transition(
                draw_key,
                current,
                step_size,
                dynamic_kinetic,
            )
            return next_state, (next_state, info)

        return jax.lax.scan(one_step, initial_state, keys)

    if chain_method == "interleaved":
        if algorithm != "nuts":
            raise ValueError("Interleaved execution is available only for NUTS.")
        compiled_interleaved = _build_prepared_interleaved_nuts_advancer(
            logdensity_fn,
            kinetic_arrays,
            max_num_doublings=depth,
            causal_config=causal_nuts,
        )

        def advance_interleaved(
            current_states,
            step_sizes,
            inverse_mass_matrices,
            sample_keys,
            *,
            start,
            count,
        ):
            del inverse_mass_matrices
            draw_keys = _mcmc_draw_keys(sample_keys, start=start, count=count)
            return compiled_interleaved(current_states, step_sizes, draw_keys)

        return advance_interleaved

    if chain_method == "vectorized":
        compiled_vectorized = jax.jit(jax.vmap(run_chain))

        def advance_vectorized(
            current_states,
            step_sizes,
            inverse_mass_matrices,
            sample_keys,
            *,
            start,
            count,
        ):
            del inverse_mass_matrices
            draw_keys = _mcmc_draw_keys(sample_keys, start=start, count=count)
            final_states, (states, infos) = compiled_vectorized(
                current_states,
                draw_keys,
                step_sizes,
                kinetic_arrays,
            )
            return (
                final_states,
                states.position,
                _prepared_metrics(states, infos, algorithm=algorithm),
            )

        return advance_vectorized

    compiled_sequential = jax.jit(run_chain)

    def advance_sequential(
        current_states,
        step_sizes,
        inverse_mass_matrices,
        sample_keys,
        *,
        start,
        count,
    ):
        del inverse_mass_matrices
        draw_keys = _mcmc_draw_keys(sample_keys, start=start, count=count)
        chain_states = _unstack_tree(current_states, len(kinetics))
        dynamic_by_chain = _unstack_tree(kinetic_arrays, len(kinetics))
        final_values = []
        sample_values = []
        info_values = []
        for index, (initial_state, dynamic_kinetic) in enumerate(
            zip(chain_states, dynamic_by_chain, strict=True)
        ):
            final, (states, infos) = compiled_sequential(
                initial_state,
                draw_keys[index],
                step_sizes[index],
                dynamic_kinetic,
            )
            final_values.append(final)
            sample_values.append(states)
            info_values.append(infos)
        final_states = _stack_trees(final_values)
        states = _stack_trees(sample_values)
        infos = _stack_trees(info_values)
        return (
            final_states,
            states.position,
            _prepared_metrics(states, infos, algorithm=algorithm),
        )

    return advance_sequential


def _mcmc_draw_keys(sample_keys, *, start, count):
    indices = jnp.arange(start, start + count, dtype=jnp.uint32)
    return jax.vmap(
        lambda sample_key: jax.vmap(lambda index: jr.fold_in(sample_key, index))(indices)
    )(sample_keys)


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
    causal_converged,
    causal_fallback_used,
    causal_outer_iterations,
    causal_maximum_residual,
    causal_accepted_steps,
    causal_rejected_steps,
    causal_transition_evaluations,
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
        "causal_converged": causal_converged,
        "causal_fallback_used": causal_fallback_used,
        "causal_outer_iterations": causal_outer_iterations,
        "causal_maximum_residual": causal_maximum_residual,
        "causal_accepted_steps": causal_accepted_steps,
        "causal_rejected_steps": causal_rejected_steps,
        "causal_transition_evaluations": causal_transition_evaluations,
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
    causal_converged = _checkpoint_array(
        arrays, "causal_converged", shape=expected_draw_shape
    )
    causal_fallback_used = _checkpoint_array(
        arrays, "causal_fallback_used", shape=expected_draw_shape
    )
    causal_outer_iterations = _checkpoint_array(
        arrays, "causal_outer_iterations", shape=expected_draw_shape
    )
    causal_maximum_residual = _checkpoint_array(
        arrays, "causal_maximum_residual", shape=expected_draw_shape
    )
    causal_accepted_steps = _checkpoint_array(
        arrays, "causal_accepted_steps", shape=expected_draw_shape
    )
    causal_rejected_steps = _checkpoint_array(
        arrays, "causal_rejected_steps", shape=expected_draw_shape
    )
    causal_transition_evaluations = _checkpoint_array(
        arrays, "causal_transition_evaluations", shape=expected_draw_shape
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
        causal_converged,
        causal_fallback_used,
        causal_outer_iterations,
        causal_maximum_residual,
        causal_accepted_steps,
        causal_rejected_steps,
        causal_transition_evaluations,
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


__all__ = [
    "MCMCChainWarmup",
    "MCMCDiagnostics",
    "MCMCResult",
    "sample_hmc",
    "sample_nuts",
]
