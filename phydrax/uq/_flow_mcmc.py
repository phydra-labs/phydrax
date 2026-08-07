#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import importlib.metadata
import time
from pathlib import Path
from typing import Any

import blackjax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from blackjax.mcmc.hmc import HMCState
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, PyTree

from .._strict import StrictModule
from ._chain import (
    _split_chain_keys,
    _stack_trees,
    _tree_nbytes,
    _unstack_tree,
    _validate_chain_method,
    ChainMethod,
)
from ._checkpoint import (
    array_tree_fingerprint,
    checkpoint_compatibility,
    CheckpointCompatibilityError,
    CheckpointCorruptionError,
    pack_array_tree,
    read_checkpoint_archive,
    unpack_array_tree,
    write_checkpoint_archive,
)
from ._diagnostics import mcmc_diagnostics, MCMCConvergenceReport
from ._flow_proposal import (
    _build_default_flow,
    _fit_flow,
    _initialize_replay,
    _proposal_effective_sample_size,
    _replay_data,
    _ReplayBuffer,
    _run_flow_block,
    _update_replay,
    _validate_flow,
)
from ._mcmc import _adapt_mcmc, MCMCChainWarmup, MCMCResult
from ._posterior import PosteriorProblem


_INITIALIZATION_TAG = 0
_WARMUP_TAG = 1
_ADAPTATION_LOCAL_TAG = 2
_REPLAY_TAG = 3
_FLOW_INITIALIZATION_TAG = 4
_FLOW_TRAINING_TAG = 5
_ADAPTATION_GLOBAL_TAG = 6
_STABILIZATION_TAG = 7
_PRODUCTION_TAG = 8


class FlowNUTSConfig(StrictModule):
    """Static adaptation, proposal, and production controls for flow-assisted NUTS."""

    num_adaptation_rounds: int = eqx.field(static=True)
    num_local_adaptation_steps: int = eqx.field(static=True)
    num_global_adaptation_steps: int = eqx.field(static=True)
    num_stabilization_steps: int = eqx.field(static=True)
    num_local_steps: int = eqx.field(static=True)
    num_global_steps: int = eqx.field(static=True)
    history_capacity_per_chain: int = eqx.field(static=True)
    history_thinning: int = eqx.field(static=True)
    flow_layers: int = eqx.field(static=True)
    num_knots: int = eqx.field(static=True)
    nn_width: int = eqx.field(static=True)
    nn_depth: int = eqx.field(static=True)
    learning_rate: float = eqx.field(static=True)
    max_epochs: int = eqx.field(static=True)
    max_patience: int = eqx.field(static=True)
    batch_size: int = eqx.field(static=True)
    validation_fraction: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        num_adaptation_rounds: int = 4,
        num_local_adaptation_steps: int = 100,
        num_global_adaptation_steps: int = 20,
        num_stabilization_steps: int = 100,
        num_local_steps: int = 10,
        num_global_steps: int = 1,
        history_capacity_per_chain: int = 1000,
        history_thinning: int = 1,
        flow_layers: int = 6,
        num_knots: int = 8,
        nn_width: int = 64,
        nn_depth: int = 2,
        learning_rate: float = 5e-4,
        max_epochs: int = 100,
        max_patience: int = 10,
        batch_size: int = 256,
        validation_fraction: float = 0.1,
    ):
        positive = {
            "num_adaptation_rounds": num_adaptation_rounds,
            "num_local_adaptation_steps": num_local_adaptation_steps,
            "num_global_adaptation_steps": num_global_adaptation_steps,
            "num_local_steps": num_local_steps,
            "num_global_steps": num_global_steps,
            "history_capacity_per_chain": history_capacity_per_chain,
            "history_thinning": history_thinning,
            "flow_layers": flow_layers,
            "num_knots": num_knots,
            "nn_width": nn_width,
            "nn_depth": nn_depth,
            "max_epochs": max_epochs,
            "max_patience": max_patience,
            "batch_size": batch_size,
        }
        normalized: dict[str, int] = {}
        for name, value in positive.items():
            count = int(value)
            if count <= 0:
                raise ValueError(f"{name} must be positive.")
            normalized[name] = count
        if normalized["history_thinning"] > normalized["num_local_adaptation_steps"]:
            raise ValueError("history_thinning cannot exceed num_local_adaptation_steps.")
        stabilization = int(num_stabilization_steps)
        if stabilization < 0:
            raise ValueError("num_stabilization_steps cannot be negative.")
        rate = float(learning_rate)
        if not jnp.isfinite(rate) or rate <= 0.0:
            raise ValueError("learning_rate must be positive and finite.")
        validation = float(validation_fraction)
        if not 0.0 < validation < 0.5:
            raise ValueError(
                "validation_fraction must lie strictly between zero and one half."
            )

        self.num_adaptation_rounds = normalized["num_adaptation_rounds"]
        self.num_local_adaptation_steps = normalized["num_local_adaptation_steps"]
        self.num_global_adaptation_steps = normalized["num_global_adaptation_steps"]
        self.num_stabilization_steps = stabilization
        self.num_local_steps = normalized["num_local_steps"]
        self.num_global_steps = normalized["num_global_steps"]
        self.history_capacity_per_chain = normalized["history_capacity_per_chain"]
        self.history_thinning = normalized["history_thinning"]
        self.flow_layers = normalized["flow_layers"]
        self.num_knots = normalized["num_knots"]
        self.nn_width = normalized["nn_width"]
        self.nn_depth = normalized["nn_depth"]
        self.learning_rate = rate
        self.max_epochs = normalized["max_epochs"]
        self.max_patience = normalized["max_patience"]
        self.batch_size = normalized["batch_size"]
        self.validation_fraction = validation

    def as_dict(self) -> dict[str, int | float]:
        """Return the canonical JSON-compatible configuration identity."""
        return {
            "num_adaptation_rounds": self.num_adaptation_rounds,
            "num_local_adaptation_steps": self.num_local_adaptation_steps,
            "num_global_adaptation_steps": self.num_global_adaptation_steps,
            "num_stabilization_steps": self.num_stabilization_steps,
            "num_local_steps": self.num_local_steps,
            "num_global_steps": self.num_global_steps,
            "history_capacity_per_chain": self.history_capacity_per_chain,
            "history_thinning": self.history_thinning,
            "flow_layers": self.flow_layers,
            "num_knots": self.num_knots,
            "nn_width": self.nn_width,
            "nn_depth": self.nn_depth,
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
            "max_patience": self.max_patience,
            "batch_size": self.batch_size,
            "validation_fraction": self.validation_fraction,
        }


class FlowNUTSResult(StrictModule):
    """Exact flow-assisted NUTS draws and global-proposal diagnostics."""

    mcmc: MCMCResult
    flow: Any
    config: FlowNUTSConfig
    training_losses: tuple[Array, ...]
    validation_losses: tuple[Array, ...]
    flow_training_duration_seconds: tuple[float, ...] = eqx.field(static=True)
    adaptation_global_acceptance_rate: Array
    adaptation_proposal_ess: Array
    adaptation_history_size: Array
    global_acceptance_rate: Array
    global_accepted_count: Array
    global_mean_log_acceptance_ratio: Array
    global_nonfinite_count: Array
    num_unique_initial_positions: int = eqx.field(static=True)
    nuts_adaptation_duration_seconds: float = eqx.field(static=True)
    flow_adaptation_duration_seconds: float = eqx.field(static=True)
    stabilization_duration_seconds: float = eqx.field(static=True)
    sampling_duration_seconds: float = eqx.field(static=True)
    duration_seconds: float = eqx.field(static=True)
    flow_parameter_memory_bytes: int = eqx.field(static=True)
    history_memory_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        mcmc: MCMCResult,
        flow: Any,
        config: FlowNUTSConfig,
        training_losses: tuple[Array, ...],
        validation_losses: tuple[Array, ...],
        flow_training_duration_seconds: tuple[float, ...],
        adaptation_global_acceptance_rate: Array,
        adaptation_proposal_ess: Array,
        adaptation_history_size: Array,
        global_acceptance_rate: Array,
        global_accepted_count: Array,
        global_mean_log_acceptance_ratio: Array,
        global_nonfinite_count: Array,
        num_unique_initial_positions: int,
        nuts_adaptation_duration_seconds: float,
        flow_adaptation_duration_seconds: float,
        stabilization_duration_seconds: float,
        sampling_duration_seconds: float,
        duration_seconds: float,
        history_memory_bytes: int,
    ):
        if not isinstance(mcmc, MCMCResult):
            raise TypeError("mcmc must be an MCMCResult.")
        if not isinstance(config, FlowNUTSConfig):
            raise TypeError("config must be a FlowNUTSConfig.")
        rounds = config.num_adaptation_rounds
        if (
            len(training_losses) != rounds
            or len(validation_losses) != rounds
            or len(flow_training_duration_seconds) != rounds
        ):
            raise ValueError(
                "Flow loss and training-duration histories must match adaptation rounds."
            )
        expected_adaptation_shape = (rounds,)
        if jnp.asarray(adaptation_global_acceptance_rate).shape != (
            rounds,
            mcmc.num_chains,
        ):
            raise ValueError(
                "adaptation_global_acceptance_rate has an incompatible shape."
            )
        if jnp.asarray(adaptation_proposal_ess).shape != expected_adaptation_shape:
            raise ValueError("adaptation_proposal_ess has an incompatible shape.")
        if jnp.asarray(adaptation_history_size).shape != expected_adaptation_shape:
            raise ValueError("adaptation_history_size has an incompatible shape.")
        expected_draw_shape = (mcmc.num_chains, mcmc.num_draws)
        for name, value in (
            ("global_acceptance_rate", global_acceptance_rate),
            ("global_accepted_count", global_accepted_count),
            (
                "global_mean_log_acceptance_ratio",
                global_mean_log_acceptance_ratio,
            ),
            ("global_nonfinite_count", global_nonfinite_count),
        ):
            if jnp.asarray(value).shape != expected_draw_shape:
                raise ValueError(f"{name} has an incompatible shape.")
        timings = (
            *flow_training_duration_seconds,
            nuts_adaptation_duration_seconds,
            flow_adaptation_duration_seconds,
            stabilization_duration_seconds,
            sampling_duration_seconds,
            duration_seconds,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in timings):
            raise ValueError("Flow-NUTS durations must be finite and nonnegative.")

        flow_parameters, _ = eqx.partition(flow, eqx.is_array)
        self.mcmc = mcmc
        self.flow = flow
        self.config = config
        self.training_losses = tuple(jnp.asarray(value) for value in training_losses)
        self.validation_losses = tuple(jnp.asarray(value) for value in validation_losses)
        self.flow_training_duration_seconds = tuple(
            float(value) for value in flow_training_duration_seconds
        )
        self.adaptation_global_acceptance_rate = jnp.asarray(
            adaptation_global_acceptance_rate
        )
        self.adaptation_proposal_ess = jnp.asarray(adaptation_proposal_ess)
        self.adaptation_history_size = jnp.asarray(adaptation_history_size)
        self.global_acceptance_rate = jnp.asarray(global_acceptance_rate)
        self.global_accepted_count = jnp.asarray(global_accepted_count)
        self.global_mean_log_acceptance_ratio = jnp.asarray(
            global_mean_log_acceptance_ratio
        )
        self.global_nonfinite_count = jnp.asarray(global_nonfinite_count)
        self.num_unique_initial_positions = int(num_unique_initial_positions)
        self.nuts_adaptation_duration_seconds = float(nuts_adaptation_duration_seconds)
        self.flow_adaptation_duration_seconds = float(flow_adaptation_duration_seconds)
        self.stabilization_duration_seconds = float(stabilization_duration_seconds)
        self.sampling_duration_seconds = float(sampling_duration_seconds)
        self.duration_seconds = float(duration_seconds)
        self.flow_parameter_memory_bytes = _tree_nbytes(flow_parameters)
        self.history_memory_bytes = int(history_memory_bytes)

    @property
    def problem(self) -> PosteriorProblem:
        return self.mcmc.problem

    @property
    def samples(self) -> PyTree[Array]:
        return self.mcmc.samples

    @property
    def unconstrained_samples(self) -> PyTree[Array]:
        return self.mcmc.unconstrained_samples

    @property
    def log_density(self) -> Array:
        return self.mcmc.log_density

    @property
    def acceptance_rate(self) -> Array:
        return self.mcmc.acceptance_rate

    @property
    def divergent(self) -> Array:
        return self.mcmc.divergent

    @property
    def energy(self) -> Array:
        return self.mcmc.energy

    @property
    def num_integration_steps(self) -> Array:
        return self.mcmc.num_integration_steps

    @property
    def num_trajectory_expansions(self) -> Array:
        return self.mcmc.num_trajectory_expansions

    @property
    def final_states(self) -> tuple[Any, ...]:
        return self.mcmc.final_states

    @property
    def warmup(self) -> tuple[MCMCChainWarmup, ...]:
        return self.mcmc.warmup

    @property
    def diagnostics(self):
        return self.mcmc.diagnostics

    @property
    def root_key(self) -> Array:
        return self.mcmc.root_key

    @property
    def chain_keys(self) -> Array:
        return self.mcmc.chain_keys

    @property
    def algorithm(self) -> str:
        return self.mcmc.algorithm

    @property
    def chain_method(self) -> ChainMethod:
        return self.mcmc.chain_method

    @property
    def num_chains(self) -> int:
        return self.mcmc.num_chains

    @property
    def num_draws(self) -> int:
        return self.mcmc.num_draws

    @property
    def sample_memory_bytes(self) -> int:
        return self.mcmc.sample_memory_bytes

    @property
    def samples_per_second(self) -> float:
        return self.mcmc.samples_per_second

    @property
    def adaptation_duration_seconds(self) -> float:
        return self.mcmc.adaptation_duration_seconds

    def predict(self, *args: Any, **kwargs: Any):
        """Evaluate latent predictions while preserving chain and draw axes."""
        return self.mcmc.predict(*args, **kwargs)

    def predict_observations(self, key: Array, /, *args: Any, **kwargs: Any):
        """Draw observation predictions while preserving chain and draw axes."""
        return self.mcmc.predict_observations(key, *args, **kwargs)

    def convergence_report(self, **kwargs: Any) -> MCMCConvergenceReport:
        """Apply the ordinary MCMC convergence gates to retained composite draws."""
        return self.mcmc.convergence_report(**kwargs)


def sample_flow_nuts(
    problem: PosteriorProblem,
    /,
    *,
    key: Array,
    num_chains: int = 8,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    initial_positions: PyTree[Any] | None = None,
    target_acceptance_rate: float = 0.8,
    initial_step_size: float = 1.0,
    is_mass_matrix_diagonal: bool = True,
    max_num_doublings: int = 10,
    config: FlowNUTSConfig | None = None,
    chain_method: ChainMethod = "sequential",
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int | None = None,
    checkpoint_id: str | None = None,
    resume_from: str | Path | None = None,
) -> FlowNUTSResult:
    """Run frozen-production NUTS with an exact learned global flow proposal."""
    if not isinstance(problem, PosteriorProblem):
        raise TypeError("problem must be a PosteriorProblem.")
    flow_config = FlowNUTSConfig() if config is None else config
    if not isinstance(flow_config, FlowNUTSConfig):
        raise TypeError("config must be a FlowNUTSConfig or None.")
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
    if not jnp.isfinite(initial_step) or initial_step <= 0.0:
        raise ValueError("initial_step_size must be positive and finite.")
    doublings = int(max_num_doublings)
    if doublings <= 0:
        raise ValueError("max_num_doublings must be positive.")
    method = _validate_chain_method(chain_method)
    if resume_from is not None and initial_positions is not None:
        raise ValueError("initial_positions cannot be supplied when resuming flow NUTS.")

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
        raise ValueError("checkpoint_id is required for flow-NUTS checkpointing.")

    root_key, chain_keys = _split_chain_keys(key, chains)
    flat_reference, unravel = ravel_pytree(problem.initial_position)
    if flat_reference.size == 0:
        raise ValueError("Flow NUTS requires at least one scalar parameter.")
    if not jnp.issubdtype(flat_reference.dtype, jnp.floating):
        raise TypeError("Flow NUTS requires real floating unconstrained coordinates.")
    dimension = int(flat_reference.size)
    minimum_training_samples = (
        min(
            flow_config.history_capacity_per_chain,
            (flow_config.num_local_adaptation_steps + flow_config.history_thinning - 1)
            // flow_config.history_thinning,
        )
        * chains
    )
    validation_count = round(flow_config.validation_fraction * minimum_training_samples)
    if validation_count <= 0 or validation_count >= minimum_training_samples:
        raise ValueError(
            "The first adaptation round must provide non-empty flow train and "
            "validation splits; increase chains, local steps, or replay capacity."
        )

    settings = {
        "algorithm": "flow_nuts",
        "num_chains": chains,
        "num_warmup": warmup_steps,
        "target_acceptance_rate": target,
        "initial_step_size": initial_step,
        "is_mass_matrix_diagonal": bool(is_mass_matrix_diagonal),
        "max_num_doublings": doublings,
        "chain_method": method,
        "dimension": dimension,
        "dtype": flat_reference.dtype.str,
        "config": flow_config.as_dict(),
        "flowjax_version": importlib.metadata.version("flowjax"),
        "optax_version": importlib.metadata.version("optax"),
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
    flat_logdensity = lambda position: problem.log_density(unravel(position))
    started = time.perf_counter()

    if resume_from is None:
        if initial_positions is None:
            if problem.parameter_space.priors is None:
                raise ValueError(
                    "Automatic flow-NUTS initialization requires factorized priors; "
                    "provide initial_positions for a custom joint prior."
                )
            initial_tree = problem.parameter_space.sample_prior(
                _fold_path(root_key, _INITIALIZATION_TAG),
                num_samples=chains,
            )
        else:
            initial_tree = initial_positions
        flat_positions = _flatten_chain_positions(
            initial_tree,
            problem.initial_position,
            chains=chains,
            dimension=dimension,
        )
        values, gradients = jax.vmap(jax.value_and_grad(flat_logdensity))(flat_positions)
        jax.block_until_ready(values)
        if not bool(jnp.all(jnp.isfinite(values))) or not bool(
            jnp.all(jnp.isfinite(gradients))
        ):
            raise FloatingPointError(
                "Every initial flow-NUTS position needs finite target density and gradient."
            )
        num_unique_initial_positions = int(
            np.unique(np.asarray(flat_positions), axis=0).shape[0]
        )
        warmup_keys = jax.vmap(lambda chain_key: _fold_path(chain_key, _WARMUP_TAG))(
            chain_keys
        )
        (
            current_states,
            warmup_states,
            step_sizes,
            inverse_mass_matrices,
            warmup_durations,
            nuts_adaptation_duration,
        ) = _adapt_mcmc(
            blackjax.nuts,
            flat_logdensity,
            flat_positions,
            warmup_keys=warmup_keys,
            warmup_steps=warmup_steps,
            target_acceptance_rate=target,
            initial_step_size=initial_step,
            is_mass_matrix_diagonal=bool(is_mass_matrix_diagonal),
            extra_parameters={"max_num_doublings": doublings},
            chain_method=method,
        )
        replay = _initialize_replay(
            num_chains=chains,
            capacity_per_chain=flow_config.history_capacity_per_chain,
            dimension=dimension,
            dtype=flat_reference.dtype,
        )
        flow = None
        training_losses: list[Array] = []
        validation_losses: list[Array] = []
        flow_training_durations: list[float] = []
        adaptation_acceptance: list[Array] = []
        adaptation_ess: list[Array] = []
        adaptation_history_size: list[Array] = []
        completed_rounds = 0
        completed_stabilization = 0
        completed_draws = 0
        flat_samples = jnp.empty((chains, 0, dimension), dtype=flat_reference.dtype)
        log_density = jnp.empty((chains, 0), dtype=flat_reference.dtype)
        acceptance_rate = jnp.empty((chains, 0), dtype=flat_reference.dtype)
        divergent = jnp.empty((chains, 0), dtype=bool)
        energy = jnp.empty((chains, 0), dtype=flat_reference.dtype)
        num_integration_steps_array = jnp.empty((chains, 0), dtype=jnp.int32)
        num_trajectory_expansions_array = jnp.empty((chains, 0), dtype=jnp.int32)
        global_acceptance_rate = jnp.empty((chains, 0), dtype=flat_reference.dtype)
        global_accepted_count = jnp.empty((chains, 0), dtype=jnp.int32)
        global_mean_log_acceptance_ratio = jnp.empty(
            (chains, 0), dtype=flat_reference.dtype
        )
        global_nonfinite_count = jnp.empty((chains, 0), dtype=jnp.int32)
        flow_adaptation_duration = 0.0
        stabilization_duration = 0.0
        sampling_duration = 0.0
        previous_duration = 0.0
        frozen_flow_fingerprint = None
    else:
        if compatibility is None:
            raise RuntimeError("Flow-NUTS resume compatibility was not initialized.")
        (
            current_states,
            warmup_states,
            step_sizes,
            inverse_mass_matrices,
            warmup_durations,
            replay,
            flow,
            training_losses,
            validation_losses,
            flow_training_durations,
            adaptation_acceptance,
            adaptation_ess,
            adaptation_history_size,
            completed_rounds,
            completed_stabilization,
            completed_draws,
            flat_samples,
            log_density,
            acceptance_rate,
            divergent,
            energy,
            num_integration_steps_array,
            num_trajectory_expansions_array,
            global_acceptance_rate,
            global_accepted_count,
            global_mean_log_acceptance_ratio,
            global_nonfinite_count,
            num_unique_initial_positions,
            nuts_adaptation_duration,
            flow_adaptation_duration,
            stabilization_duration,
            sampling_duration,
            previous_duration,
            frozen_flow_fingerprint,
        ) = _read_flow_nuts_checkpoint(
            Path(resume_from),
            compatibility=compatibility,
            problem=problem,
            flat_reference=flat_reference,
            flat_logdensity=flat_logdensity,
            root_key=root_key,
            chains=chains,
            config=flow_config,
        )
        if completed_draws > draws:
            raise ValueError(
                "num_samples cannot be smaller than completed checkpoint draws."
            )

    while completed_rounds < flow_config.num_adaptation_rounds:
        round_started = time.perf_counter()
        local_keys = _indexed_chain_keys(
            chain_keys,
            phase=_ADAPTATION_LOCAL_TAG,
            group=completed_rounds,
            start=0,
            count=flow_config.num_local_adaptation_steps,
        )
        current_states, local_positions, _ = _advance_nuts_collect(
            current_states,
            step_sizes,
            inverse_mass_matrices,
            local_keys,
            flat_logdensity,
            max_num_doublings=doublings,
            chain_method=method,
        )
        replay_samples = local_positions[:, :: flow_config.history_thinning, :]
        replay_keys = _indexed_chain_keys(
            chain_keys,
            phase=_REPLAY_TAG,
            group=completed_rounds,
            start=0,
            count=int(replay_samples.shape[1]),
        )
        replay = jax.jit(_update_replay)(replay, replay_samples, replay_keys)
        replay_data = _replay_data(replay)
        if flow is None:
            flow = _build_default_flow(
                _fold_path(root_key, _FLOW_INITIALIZATION_TAG),
                replay_data,
                flow_layers=flow_config.flow_layers,
                num_knots=flow_config.num_knots,
                nn_width=flow_config.nn_width,
                nn_depth=flow_config.nn_depth,
            )
        training_started = time.perf_counter()
        flow, train_loss, validation_loss = _fit_flow(
            _fold_path(root_key, _FLOW_TRAINING_TAG, completed_rounds),
            flow,
            replay_data,
            learning_rate=flow_config.learning_rate,
            max_epochs=flow_config.max_epochs,
            max_patience=flow_config.max_patience,
            batch_size=flow_config.batch_size,
            validation_fraction=flow_config.validation_fraction,
        )
        flow_training_durations.append(time.perf_counter() - training_started)
        _validate_flow(
            flow,
            replay_data,
            _fold_path(root_key, _FLOW_TRAINING_TAG, completed_rounds, 1),
        )
        global_keys = _indexed_chain_keys(
            chain_keys,
            phase=_ADAPTATION_GLOBAL_TAG,
            group=completed_rounds,
            start=0,
            count=1,
        )[:, 0]
        flow_states, flow_info = _advance_flow_chains(
            flow,
            current_states,
            global_keys,
            flat_logdensity,
            num_steps=flow_config.num_global_adaptation_steps,
            chain_method=method,
        )
        current_states = _initialize_nuts_states(
            flow_states.position,
            flat_logdensity,
            chain_method=method,
        )
        training_losses.append(train_loss)
        validation_losses.append(validation_loss)
        adaptation_acceptance.append(jnp.mean(flow_info.accepted, axis=1))
        adaptation_ess.append(
            _proposal_effective_sample_size(
                flow_info.proposed_log_target,
                flow_info.proposed_log_density,
            )
        )
        adaptation_history_size.append(jnp.sum(replay.size))
        completed_rounds += 1
        jax.block_until_ready(current_states.position)
        flow_adaptation_duration += time.perf_counter() - round_started
        if destination is not None and compatibility is not None:
            _write_flow_nuts_checkpoint(
                destination,
                compatibility=compatibility,
                phase="adaptation",
                completed_rounds=completed_rounds,
                completed_stabilization=completed_stabilization,
                completed_draws=completed_draws,
                current_states=current_states,
                warmup_states=warmup_states,
                step_sizes=step_sizes,
                inverse_mass_matrices=inverse_mass_matrices,
                warmup_durations=warmup_durations,
                replay=replay,
                flow=flow,
                training_losses=tuple(training_losses),
                validation_losses=tuple(validation_losses),
                flow_training_durations=tuple(flow_training_durations),
                adaptation_acceptance=_stack_rows(adaptation_acceptance, chains),
                adaptation_ess=jnp.asarray(adaptation_ess),
                adaptation_history_size=jnp.asarray(adaptation_history_size),
                flat_samples=flat_samples,
                log_density=log_density,
                acceptance_rate=acceptance_rate,
                divergent=divergent,
                energy=energy,
                num_integration_steps=num_integration_steps_array,
                num_trajectory_expansions=num_trajectory_expansions_array,
                global_acceptance_rate=global_acceptance_rate,
                global_accepted_count=global_accepted_count,
                global_mean_log_acceptance_ratio=(global_mean_log_acceptance_ratio),
                global_nonfinite_count=global_nonfinite_count,
                num_unique_initial_positions=num_unique_initial_positions,
                nuts_adaptation_duration=nuts_adaptation_duration,
                flow_adaptation_duration=flow_adaptation_duration,
                stabilization_duration=stabilization_duration,
                sampling_duration=sampling_duration,
                duration_seconds=previous_duration + time.perf_counter() - started,
                frozen_flow_fingerprint=None,
            )

    if flow is None:
        raise RuntimeError("Flow adaptation completed without a proposal distribution.")
    if frozen_flow_fingerprint is None:
        frozen_flow_fingerprint = _flow_fingerprint(flow)

    while completed_stabilization < flow_config.num_stabilization_steps:
        chunk = min(
            interval,
            flow_config.num_stabilization_steps - completed_stabilization,
        )
        phase_started = time.perf_counter()
        draw_keys = _indexed_chain_keys(
            chain_keys,
            phase=_STABILIZATION_TAG,
            group=0,
            start=completed_stabilization,
            count=chunk,
        )
        current_states, _, _ = _advance_nuts_collect(
            current_states,
            step_sizes,
            inverse_mass_matrices,
            draw_keys,
            flat_logdensity,
            max_num_doublings=doublings,
            chain_method=method,
        )
        jax.block_until_ready(current_states.position)
        stabilization_duration += time.perf_counter() - phase_started
        completed_stabilization += chunk
        if destination is not None and compatibility is not None:
            _write_flow_nuts_checkpoint(
                destination,
                compatibility=compatibility,
                phase="stabilization",
                completed_rounds=completed_rounds,
                completed_stabilization=completed_stabilization,
                completed_draws=completed_draws,
                current_states=current_states,
                warmup_states=warmup_states,
                step_sizes=step_sizes,
                inverse_mass_matrices=inverse_mass_matrices,
                warmup_durations=warmup_durations,
                replay=replay,
                flow=flow,
                training_losses=tuple(training_losses),
                validation_losses=tuple(validation_losses),
                flow_training_durations=tuple(flow_training_durations),
                adaptation_acceptance=_stack_rows(adaptation_acceptance, chains),
                adaptation_ess=jnp.asarray(adaptation_ess),
                adaptation_history_size=jnp.asarray(adaptation_history_size),
                flat_samples=flat_samples,
                log_density=log_density,
                acceptance_rate=acceptance_rate,
                divergent=divergent,
                energy=energy,
                num_integration_steps=num_integration_steps_array,
                num_trajectory_expansions=num_trajectory_expansions_array,
                global_acceptance_rate=global_acceptance_rate,
                global_accepted_count=global_accepted_count,
                global_mean_log_acceptance_ratio=(global_mean_log_acceptance_ratio),
                global_nonfinite_count=global_nonfinite_count,
                num_unique_initial_positions=num_unique_initial_positions,
                nuts_adaptation_duration=nuts_adaptation_duration,
                flow_adaptation_duration=flow_adaptation_duration,
                stabilization_duration=stabilization_duration,
                sampling_duration=sampling_duration,
                duration_seconds=previous_duration + time.perf_counter() - started,
                frozen_flow_fingerprint=frozen_flow_fingerprint,
            )

    while completed_draws < draws:
        chunk = min(interval, draws - completed_draws)
        phase_started = time.perf_counter()
        draw_keys = _indexed_chain_keys(
            chain_keys,
            phase=_PRODUCTION_TAG,
            group=0,
            start=completed_draws,
            count=chunk,
        )
        current_states, chunk_positions, metrics = _advance_composite_chains(
            flow,
            current_states,
            step_sizes,
            inverse_mass_matrices,
            draw_keys,
            flat_logdensity,
            num_global_steps=flow_config.num_global_steps,
            num_local_steps=flow_config.num_local_steps,
            max_num_doublings=doublings,
            chain_method=method,
        )
        jax.block_until_ready(metrics["log_density"])
        sampling_duration += time.perf_counter() - phase_started
        flat_samples = jnp.concatenate((flat_samples, chunk_positions), axis=1)
        log_density = jnp.concatenate((log_density, metrics["log_density"]), axis=1)
        acceptance_rate = jnp.concatenate(
            (acceptance_rate, metrics["acceptance_rate"]), axis=1
        )
        divergent = jnp.concatenate((divergent, metrics["divergent"]), axis=1)
        energy = jnp.concatenate((energy, metrics["energy"]), axis=1)
        num_integration_steps_array = jnp.concatenate(
            (
                num_integration_steps_array,
                metrics["num_integration_steps"],
            ),
            axis=1,
        )
        num_trajectory_expansions_array = jnp.concatenate(
            (
                num_trajectory_expansions_array,
                metrics["num_trajectory_expansions"],
            ),
            axis=1,
        )
        global_acceptance_rate = jnp.concatenate(
            (global_acceptance_rate, metrics["global_acceptance_rate"]), axis=1
        )
        global_accepted_count = jnp.concatenate(
            (global_accepted_count, metrics["global_accepted_count"]), axis=1
        )
        global_mean_log_acceptance_ratio = jnp.concatenate(
            (
                global_mean_log_acceptance_ratio,
                metrics["global_mean_log_acceptance_ratio"],
            ),
            axis=1,
        )
        global_nonfinite_count = jnp.concatenate(
            (global_nonfinite_count, metrics["global_nonfinite_count"]), axis=1
        )
        completed_draws += chunk
        if destination is not None and compatibility is not None:
            _write_flow_nuts_checkpoint(
                destination,
                compatibility=compatibility,
                phase="production",
                completed_rounds=completed_rounds,
                completed_stabilization=completed_stabilization,
                completed_draws=completed_draws,
                current_states=current_states,
                warmup_states=warmup_states,
                step_sizes=step_sizes,
                inverse_mass_matrices=inverse_mass_matrices,
                warmup_durations=warmup_durations,
                replay=replay,
                flow=flow,
                training_losses=tuple(training_losses),
                validation_losses=tuple(validation_losses),
                flow_training_durations=tuple(flow_training_durations),
                adaptation_acceptance=_stack_rows(adaptation_acceptance, chains),
                adaptation_ess=jnp.asarray(adaptation_ess),
                adaptation_history_size=jnp.asarray(adaptation_history_size),
                flat_samples=flat_samples,
                log_density=log_density,
                acceptance_rate=acceptance_rate,
                divergent=divergent,
                energy=energy,
                num_integration_steps=num_integration_steps_array,
                num_trajectory_expansions=num_trajectory_expansions_array,
                global_acceptance_rate=global_acceptance_rate,
                global_accepted_count=global_accepted_count,
                global_mean_log_acceptance_ratio=(global_mean_log_acceptance_ratio),
                global_nonfinite_count=global_nonfinite_count,
                num_unique_initial_positions=num_unique_initial_positions,
                nuts_adaptation_duration=nuts_adaptation_duration,
                flow_adaptation_duration=flow_adaptation_duration,
                stabilization_duration=stabilization_duration,
                sampling_duration=sampling_duration,
                duration_seconds=previous_duration + time.perf_counter() - started,
                frozen_flow_fingerprint=frozen_flow_fingerprint,
            )

    if _flow_fingerprint(flow) != frozen_flow_fingerprint:
        raise RuntimeError("Frozen flow parameters changed during retained sampling.")
    unconstrained_samples = jax.vmap(jax.vmap(unravel))(flat_samples)
    samples = problem.parameter_space.constrain(unconstrained_samples)
    diagnostics = mcmc_diagnostics(
        samples,
        acceptance_rate=acceptance_rate,
        divergent=divergent,
    )
    jax.block_until_ready(diagnostics.max_rhat)
    final_state_tuple = tuple(
        _unravel_hmc_state(state, unravel)
        for state in _unstack_tree(current_states, chains)
    )
    warmup_state_tuple = tuple(
        _unravel_hmc_state(state, unravel)
        for state in _unstack_tree(warmup_states, chains)
    )
    warmups = tuple(
        MCMCChainWarmup(
            state=warmup_state_tuple[index],
            step_size=step_sizes[index],
            inverse_mass_matrix=inverse_mass_matrices[index],
            num_integration_steps=None,
            duration_seconds=float(warmup_durations[index]),
        )
        for index in range(chains)
    )
    total_duration = previous_duration + time.perf_counter() - started
    mcmc = MCMCResult(
        problem=problem,
        samples=samples,
        unconstrained_samples=unconstrained_samples,
        log_density=log_density,
        acceptance_rate=acceptance_rate,
        divergent=divergent,
        energy=energy,
        num_integration_steps=num_integration_steps_array,
        num_trajectory_expansions=num_trajectory_expansions_array,
        final_states=final_state_tuple,
        warmup=warmups,
        diagnostics=diagnostics,
        root_key=root_key,
        chain_keys=chain_keys,
        algorithm="flow_nuts",
        duration_seconds=total_duration,
        max_num_doublings=doublings,
        chain_method=method,
        adaptation_duration_seconds=(
            nuts_adaptation_duration + flow_adaptation_duration + stabilization_duration
        ),
        sampling_duration_seconds=sampling_duration,
    )
    return FlowNUTSResult(
        mcmc=mcmc,
        flow=flow,
        config=flow_config,
        training_losses=tuple(training_losses),
        validation_losses=tuple(validation_losses),
        flow_training_duration_seconds=tuple(flow_training_durations),
        adaptation_global_acceptance_rate=_stack_rows(adaptation_acceptance, chains),
        adaptation_proposal_ess=jnp.asarray(adaptation_ess),
        adaptation_history_size=jnp.asarray(adaptation_history_size),
        global_acceptance_rate=global_acceptance_rate,
        global_accepted_count=global_accepted_count,
        global_mean_log_acceptance_ratio=global_mean_log_acceptance_ratio,
        global_nonfinite_count=global_nonfinite_count,
        num_unique_initial_positions=num_unique_initial_positions,
        nuts_adaptation_duration_seconds=nuts_adaptation_duration,
        flow_adaptation_duration_seconds=flow_adaptation_duration,
        stabilization_duration_seconds=stabilization_duration,
        sampling_duration_seconds=sampling_duration,
        duration_seconds=total_duration,
        history_memory_bytes=_tree_nbytes(replay),
    )


def _fold_path(key: Array, *indices: int) -> Array:
    result = key
    for index in indices:
        result = jr.fold_in(result, int(index))
    return result


def _indexed_chain_keys(
    chain_keys: Array,
    /,
    *,
    phase: int,
    group: int,
    start: int,
    count: int,
) -> Array:
    indices = jnp.arange(start, start + count, dtype=jnp.uint32)

    def chain_schedule(chain_key):
        base = jr.fold_in(jr.fold_in(chain_key, phase), group)
        return jax.vmap(lambda index: jr.fold_in(base, index))(indices)

    return jax.vmap(chain_schedule)(chain_keys)


def _flatten_chain_positions(
    positions: PyTree[Any],
    reference: PyTree[Any],
    /,
    *,
    chains: int,
    dimension: int,
) -> Array:
    if jax.tree_util.tree_structure(positions) != jax.tree_util.tree_structure(reference):
        raise ValueError("initial_positions has an incompatible PyTree structure.")
    position_leaves = jax.tree_util.tree_leaves(positions)
    reference_leaves = jax.tree_util.tree_leaves(reference)
    for value, expected in zip(position_leaves, reference_leaves, strict=True):
        array = jnp.asarray(value)
        expected_shape = (chains, *jnp.asarray(expected).shape)
        if array.shape != expected_shape:
            raise ValueError(
                "Every initial_positions leaf needs shape "
                f"{expected_shape}; received {array.shape}."
            )
        if not jnp.issubdtype(array.dtype, jnp.floating):
            raise TypeError("Every initial_positions leaf must be a real floating array.")
        if not bool(jnp.all(jnp.isfinite(array))):
            raise ValueError("Every initial_positions leaf must be finite.")
    flattened = []
    for index in range(chains):
        chain_position = jax.tree_util.tree_map(lambda value: value[index], positions)
        flat, _ = ravel_pytree(chain_position)
        if flat.shape != (dimension,):
            raise ValueError("An initial position has an incompatible flattened shape.")
        flattened.append(flat)
    return jnp.stack(flattened)


def _advance_nuts_collect(
    current_states,
    step_sizes,
    inverse_mass_matrices,
    keys,
    logdensity_fn,
    *,
    max_num_doublings: int,
    chain_method: ChainMethod,
):
    kernel = blackjax.nuts.build_kernel()

    def run_chain(state, chain_keys, step_size, inverse_mass_matrix):
        def transition(current, transition_key):
            next_state, info = kernel(
                transition_key,
                current,
                logdensity_fn,
                step_size,
                inverse_mass_matrix,
                max_num_doublings,
            )
            return next_state, (next_state.position, info)

        return jax.lax.scan(transition, state, chain_keys)

    if chain_method == "vectorized":
        return_values = jax.jit(jax.vmap(run_chain))(
            current_states,
            keys,
            step_sizes,
            inverse_mass_matrices,
        )
        final_states, (positions, infos) = return_values
        return final_states, positions, infos

    states = _unstack_tree(current_states, int(keys.shape[0]))
    final_values = []
    position_values = []
    info_values = []
    compiled = jax.jit(run_chain)
    for index, state in enumerate(states):
        final_state, (positions, infos) = compiled(
            state,
            keys[index],
            step_sizes[index],
            inverse_mass_matrices[index],
        )
        final_values.append(final_state)
        position_values.append(positions)
        info_values.append(infos)
    return (
        _stack_trees(final_values),
        jnp.stack(position_values),
        _stack_trees(info_values),
    )


def _advance_flow_chains(
    flow,
    current_states,
    keys,
    logdensity_fn,
    *,
    num_steps: int,
    chain_method: ChainMethod,
):
    def run_chain(flow_value, state, transition_key):
        return _run_flow_block(
            transition_key,
            state.position,
            state.logdensity,
            flow_value,
            logdensity_fn,
            num_steps=num_steps,
        )

    if chain_method == "vectorized":

        def run_vectorized(flow_value, states, transition_keys):
            return jax.vmap(
                lambda state, transition_key: run_chain(flow_value, state, transition_key)
            )(states, transition_keys)

        return eqx.filter_jit(run_vectorized)(flow, current_states, keys)

    states = _unstack_tree(current_states, int(keys.shape[0]))
    final_values = []
    info_values = []
    compiled = eqx.filter_jit(run_chain)
    for index, state in enumerate(states):
        final_state, info = compiled(flow, state, keys[index])
        final_values.append(final_state)
        info_values.append(info)
    return _stack_trees(final_values), _stack_trees(info_values)


def _initialize_nuts_states(
    positions,
    logdensity_fn,
    *,
    chain_method: ChainMethod,
):
    if chain_method == "vectorized":
        return jax.jit(jax.vmap(lambda value: blackjax.nuts.init(value, logdensity_fn)))(
            positions
        )
    compiled = jax.jit(lambda value: blackjax.nuts.init(value, logdensity_fn))
    return _stack_trees([compiled(position) for position in positions])


def _advance_composite_chains(
    flow,
    current_states,
    step_sizes,
    inverse_mass_matrices,
    draw_keys,
    logdensity_fn,
    *,
    num_global_steps: int,
    num_local_steps: int,
    max_num_doublings: int,
    chain_method: ChainMethod,
):
    kernel = blackjax.nuts.build_kernel()

    def run_chain(
        flow_value,
        state,
        chain_draw_keys,
        step_size,
        inverse_mass_matrix,
    ):
        def composite(current_state, draw_key):
            global_key = jr.fold_in(draw_key, 0)
            local_base_key = jr.fold_in(draw_key, 1)
            flow_state, flow_info = _run_flow_block(
                global_key,
                current_state.position,
                current_state.logdensity,
                flow_value,
                logdensity_fn,
                num_steps=num_global_steps,
            )
            local_state = blackjax.nuts.init(flow_state.position, logdensity_fn)
            local_keys = jax.vmap(lambda index: jr.fold_in(local_base_key, index))(
                jnp.arange(num_local_steps, dtype=jnp.uint32)
            )

            def local_transition(carry, transition_key):
                next_state, info = kernel(
                    transition_key,
                    carry,
                    logdensity_fn,
                    step_size,
                    inverse_mass_matrix,
                    max_num_doublings,
                )
                return next_state, info

            final_state, local_info = jax.lax.scan(
                local_transition,
                local_state,
                local_keys,
            )
            finite_log_ratio = jnp.isfinite(flow_info.log_acceptance_ratio)
            finite_count = jnp.sum(finite_log_ratio)
            mean_log_ratio = jnp.where(
                finite_count > 0,
                jnp.sum(
                    jnp.where(
                        finite_log_ratio,
                        flow_info.log_acceptance_ratio,
                        jnp.zeros_like(flow_info.log_acceptance_ratio),
                    )
                )
                / finite_count,
                -jnp.inf,
            )
            metrics = {
                "log_density": final_state.logdensity,
                "acceptance_rate": jnp.mean(local_info.acceptance_rate),
                "divergent": jnp.any(local_info.is_divergent),
                "energy": local_info.energy[-1],
                "num_integration_steps": jnp.sum(local_info.num_integration_steps),
                "num_trajectory_expansions": jnp.max(
                    local_info.num_trajectory_expansions
                ),
                "global_acceptance_rate": jnp.mean(
                    flow_info.accepted.astype(final_state.position.dtype)
                ),
                "global_accepted_count": jnp.sum(flow_info.accepted, dtype=jnp.int32),
                "global_mean_log_acceptance_ratio": mean_log_ratio,
                "global_nonfinite_count": jnp.sum(flow_info.nonfinite, dtype=jnp.int32),
            }
            return final_state, (final_state.position, metrics)

        return jax.lax.scan(composite, state, chain_draw_keys)

    if chain_method == "vectorized":

        def run_vectorized(
            flow_value,
            states,
            keys,
            chain_step_sizes,
            chain_mass_matrices,
        ):
            return jax.vmap(
                lambda state, chain_keys, step_size, mass_matrix: run_chain(
                    flow_value,
                    state,
                    chain_keys,
                    step_size,
                    mass_matrix,
                )
            )(states, keys, chain_step_sizes, chain_mass_matrices)

        final_states, (positions, metrics) = eqx.filter_jit(run_vectorized)(
            flow,
            current_states,
            draw_keys,
            step_sizes,
            inverse_mass_matrices,
        )
        return final_states, positions, metrics

    states = _unstack_tree(current_states, int(draw_keys.shape[0]))
    final_values = []
    position_values = []
    metric_values = []
    compiled = eqx.filter_jit(run_chain)
    for index, state in enumerate(states):
        final_state, (positions, metrics) = compiled(
            flow,
            state,
            draw_keys[index],
            step_sizes[index],
            inverse_mass_matrices[index],
        )
        final_values.append(final_state)
        position_values.append(positions)
        metric_values.append(metrics)
    return (
        _stack_trees(final_values),
        jnp.stack(position_values),
        _stack_trees(metric_values),
    )


def _flow_fingerprint(flow) -> dict[str, Any]:
    parameters, _ = eqx.partition(flow, eqx.is_array)
    return array_tree_fingerprint(parameters)


def _unravel_hmc_state(state, unravel) -> HMCState:
    return HMCState(
        position=unravel(state.position),
        logdensity=state.logdensity,
        logdensity_grad=unravel(state.logdensity_grad),
    )


def _stack_rows(values: list[Array], width: int) -> Array:
    if not values:
        return jnp.empty((0, width), dtype=float)
    return jnp.stack(values)


def _write_flow_nuts_checkpoint(
    destination,
    *,
    compatibility,
    phase,
    completed_rounds,
    completed_stabilization,
    completed_draws,
    current_states,
    warmup_states,
    step_sizes,
    inverse_mass_matrices,
    warmup_durations,
    replay,
    flow,
    training_losses,
    validation_losses,
    flow_training_durations,
    adaptation_acceptance,
    adaptation_ess,
    adaptation_history_size,
    flat_samples,
    log_density,
    acceptance_rate,
    divergent,
    energy,
    num_integration_steps,
    num_trajectory_expansions,
    global_acceptance_rate,
    global_accepted_count,
    global_mean_log_acceptance_ratio,
    global_nonfinite_count,
    num_unique_initial_positions,
    nuts_adaptation_duration,
    flow_adaptation_duration,
    stabilization_duration,
    sampling_duration,
    duration_seconds,
    frozen_flow_fingerprint,
):
    arrays = {
        "step_sizes": step_sizes,
        "inverse_mass_matrices": inverse_mass_matrices,
        "warmup_durations": warmup_durations,
        "replay_values": replay.values,
        "replay_size": replay.size,
        "replay_seen": replay.seen,
        "adaptation_acceptance": adaptation_acceptance,
        "adaptation_ess": adaptation_ess,
        "adaptation_history_size": adaptation_history_size,
        "flat_samples": flat_samples,
        "log_density": log_density,
        "acceptance_rate": acceptance_rate,
        "divergent": divergent,
        "energy": energy,
        "num_integration_steps": num_integration_steps,
        "num_trajectory_expansions": num_trajectory_expansions,
        "global_acceptance_rate": global_acceptance_rate,
        "global_accepted_count": global_accepted_count,
        "global_mean_log_acceptance_ratio": global_mean_log_acceptance_ratio,
        "global_nonfinite_count": global_nonfinite_count,
    }
    flow_parameters, _ = eqx.partition(flow, eqx.is_array)
    training_names = []
    validation_names = []
    for index, loss in enumerate(training_losses):
        name = f"training_loss/{index:06d}"
        arrays[name] = loss
        training_names.append(name)
    for index, loss in enumerate(validation_losses):
        name = f"validation_loss/{index:06d}"
        arrays[name] = loss
        validation_names.append(name)
    state = {
        "phase": str(phase),
        "completed_rounds": int(completed_rounds),
        "completed_stabilization": int(completed_stabilization),
        "completed_draws": int(completed_draws),
        "num_unique_initial_positions": int(num_unique_initial_positions),
        "nuts_adaptation_duration_seconds": float(nuts_adaptation_duration),
        "flow_adaptation_duration_seconds": float(flow_adaptation_duration),
        "stabilization_duration_seconds": float(stabilization_duration),
        "sampling_duration_seconds": float(sampling_duration),
        "duration_seconds": float(duration_seconds),
        "frozen_flow_fingerprint": frozen_flow_fingerprint,
        "current_state_tree": pack_array_tree("current_state", current_states, arrays),
        "warmup_state_tree": pack_array_tree("warmup_state", warmup_states, arrays),
        "flow_parameter_tree": pack_array_tree(
            "flow_parameters", flow_parameters, arrays
        ),
        "training_loss_arrays": training_names,
        "validation_loss_arrays": validation_names,
        "flow_training_duration_seconds": [
            float(value) for value in flow_training_durations
        ],
    }
    write_checkpoint_archive(
        destination,
        kind="flow_nuts",
        compatibility=compatibility,
        state=state,
        arrays=arrays,
    )


def _read_flow_nuts_checkpoint(
    source,
    *,
    compatibility,
    problem,
    flat_reference,
    flat_logdensity,
    root_key,
    chains,
    config,
):
    state, arrays = read_checkpoint_archive(
        source,
        kind="flow_nuts",
        compatibility=compatibility,
    )
    phase = state.get("phase")
    if phase not in ("adaptation", "stabilization", "production"):
        raise CheckpointCorruptionError("Flow-NUTS checkpoint phase is invalid.")
    completed_rounds = int(state.get("completed_rounds", -1))
    completed_stabilization = int(state.get("completed_stabilization", -1))
    completed_draws = int(state.get("completed_draws", -1))
    if not 0 <= completed_rounds <= config.num_adaptation_rounds:
        raise CheckpointCorruptionError(
            "Flow-NUTS checkpoint adaptation-round count is invalid."
        )
    if not 0 <= completed_stabilization <= config.num_stabilization_steps:
        raise CheckpointCorruptionError(
            "Flow-NUTS checkpoint stabilization count is invalid."
        )
    if completed_draws < 0:
        raise CheckpointCorruptionError(
            "Flow-NUTS checkpoint production-draw count is invalid."
        )
    if phase != "adaptation" and completed_rounds != config.num_adaptation_rounds:
        raise CheckpointCorruptionError(
            "Flow-NUTS checkpoint entered a frozen phase before adaptation completed."
        )

    one_state = blackjax.nuts.init(flat_reference, flat_logdensity)
    state_template = jax.tree_util.tree_map(
        lambda value: jnp.broadcast_to(value, (chains, *value.shape)),
        one_state,
    )
    current_states = unpack_array_tree(
        state["current_state_tree"], arrays, state_template
    )
    warmup_states = unpack_array_tree(state["warmup_state_tree"], arrays, state_template)
    dummy_data = jnp.stack(
        (jnp.zeros_like(flat_reference), jnp.ones_like(flat_reference))
    )
    flow_template = _build_default_flow(
        _fold_path(root_key, _FLOW_INITIALIZATION_TAG),
        dummy_data,
        flow_layers=config.flow_layers,
        num_knots=config.num_knots,
        nn_width=config.nn_width,
        nn_depth=config.nn_depth,
    )
    parameter_template, static = eqx.partition(flow_template, eqx.is_array)
    flow_parameters = unpack_array_tree(
        state["flow_parameter_tree"], arrays, parameter_template
    )
    flow = eqx.combine(flow_parameters, static)

    training_names = state.get("training_loss_arrays")
    validation_names = state.get("validation_loss_arrays")
    if not isinstance(training_names, list) or not isinstance(validation_names, list):
        raise CheckpointCorruptionError("Flow loss-array specifications are invalid.")
    if (
        len(training_names) != completed_rounds
        or len(validation_names) != completed_rounds
    ):
        raise CheckpointCorruptionError(
            "Flow loss histories do not match completed adaptation rounds."
        )
    training_losses = [_loss_array(arrays, name) for name in training_names]
    validation_losses = [_loss_array(arrays, name) for name in validation_names]
    flow_training_durations = state.get("flow_training_duration_seconds")
    if (
        not isinstance(flow_training_durations, list)
        or len(flow_training_durations) != completed_rounds
        or any(
            not isinstance(value, int | float) or not np.isfinite(value) or value < 0.0
            for value in flow_training_durations
        )
    ):
        raise CheckpointCorruptionError(
            "Flow training durations do not match completed adaptation rounds."
        )
    flow_training_durations = [float(value) for value in flow_training_durations]

    step_sizes = _required_array(arrays, "step_sizes", leading=chains)
    inverse_mass_matrices = _required_array(
        arrays, "inverse_mass_matrices", leading=chains
    )
    warmup_durations = _required_array(arrays, "warmup_durations", shape=(chains,))
    replay = _ReplayBuffer(
        values=_required_array(
            arrays,
            "replay_values",
            shape=(chains, config.history_capacity_per_chain, flat_reference.size),
        ),
        size=_required_array(arrays, "replay_size", shape=(chains,)),
        seen=_required_array(arrays, "replay_seen", shape=(chains,)),
    )
    adaptation_acceptance = _required_array(
        arrays,
        "adaptation_acceptance",
        shape=(completed_rounds, chains),
    )
    adaptation_ess = _required_array(arrays, "adaptation_ess", shape=(completed_rounds,))
    adaptation_history_size = _required_array(
        arrays,
        "adaptation_history_size",
        shape=(completed_rounds,),
    )
    draw_shape = (chains, completed_draws)
    flat_samples = _required_array(
        arrays,
        "flat_samples",
        shape=(chains, completed_draws, flat_reference.size),
    )
    log_density = _required_array(arrays, "log_density", shape=draw_shape)
    acceptance_rate = _required_array(arrays, "acceptance_rate", shape=draw_shape)
    divergent = _required_array(arrays, "divergent", shape=draw_shape)
    energy = _required_array(arrays, "energy", shape=draw_shape)
    num_integration_steps = _required_array(
        arrays, "num_integration_steps", shape=draw_shape
    )
    num_trajectory_expansions = _required_array(
        arrays, "num_trajectory_expansions", shape=draw_shape
    )
    global_acceptance_rate = _required_array(
        arrays, "global_acceptance_rate", shape=draw_shape
    )
    global_accepted_count = _required_array(
        arrays, "global_accepted_count", shape=draw_shape
    )
    global_mean_log_acceptance_ratio = _required_array(
        arrays,
        "global_mean_log_acceptance_ratio",
        shape=draw_shape,
    )
    global_nonfinite_count = _required_array(
        arrays, "global_nonfinite_count", shape=draw_shape
    )
    frozen_flow_fingerprint = state.get("frozen_flow_fingerprint")
    if phase != "adaptation" and not isinstance(frozen_flow_fingerprint, dict):
        raise CheckpointCorruptionError(
            "Frozen flow-NUTS checkpoint is missing its flow fingerprint."
        )
    if isinstance(frozen_flow_fingerprint, dict) and (
        _flow_fingerprint(flow) != frozen_flow_fingerprint
    ):
        raise CheckpointCompatibilityError(
            "Checkpoint flow parameters do not match the frozen fingerprint."
        )
    _validate_flow(
        flow,
        _replay_data(replay),
        _fold_path(root_key, _FLOW_TRAINING_TAG, completed_rounds, 1),
    )
    return (
        current_states,
        warmup_states,
        step_sizes,
        inverse_mass_matrices,
        warmup_durations,
        replay,
        flow,
        training_losses,
        validation_losses,
        flow_training_durations,
        [value for value in adaptation_acceptance],
        [value for value in adaptation_ess],
        [value for value in adaptation_history_size],
        completed_rounds,
        completed_stabilization,
        completed_draws,
        flat_samples,
        log_density,
        acceptance_rate,
        divergent,
        energy,
        num_integration_steps,
        num_trajectory_expansions,
        global_acceptance_rate,
        global_accepted_count,
        global_mean_log_acceptance_ratio,
        global_nonfinite_count,
        int(state["num_unique_initial_positions"]),
        float(state["nuts_adaptation_duration_seconds"]),
        float(state["flow_adaptation_duration_seconds"]),
        float(state["stabilization_duration_seconds"]),
        float(state["sampling_duration_seconds"]),
        float(state["duration_seconds"]),
        frozen_flow_fingerprint,
    )


def _required_array(arrays, name, *, shape=None, leading=None):
    if name not in arrays:
        raise CheckpointCorruptionError(f"Checkpoint array {name!r} is missing.")
    value = jnp.asarray(arrays[name])
    if shape is not None and value.shape != shape:
        raise CheckpointCompatibilityError(
            f"Checkpoint array {name!r} has an incompatible shape."
        )
    if leading is not None and (value.ndim == 0 or value.shape[0] != leading):
        raise CheckpointCompatibilityError(
            f"Checkpoint array {name!r} has an incompatible leading axis."
        )
    return value


def _loss_array(arrays, name) -> Array:
    if not isinstance(name, str) or name not in arrays:
        raise CheckpointCorruptionError("A checkpoint flow loss array is missing.")
    value = jnp.asarray(arrays[name])
    if value.ndim != 1 or not bool(jnp.all(jnp.isfinite(value))):
        raise CheckpointCorruptionError("A checkpoint flow loss history is invalid.")
    return value


__all__ = ["FlowNUTSConfig", "FlowNUTSResult", "sample_flow_nuts"]
