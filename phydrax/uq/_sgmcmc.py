#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, cast, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from blackjax.sgmcmc import diffusions
from blackjax.sgmcmc.sgnht import init as init_sgnht, SGNHTState
from jaxtyping import Array, PyTree

from .._fingerprint import array_tree_fingerprint, array_tree_signature
from .._frozendict import frozendict
from .._strict import StrictModule
from ._chain import (
    _prepare_chain_positions,
    _split_chain_keys,
    _stack_trees,
    _tree_nbytes,
    _unstack_tree,
    _validate_chain_method,
    ChainMethod,
)
from ._checkpoint import (
    CheckpointCompatibilityError,
    CheckpointCorruptionError,
    pack_array_tree,
    read_checkpoint_archive,
    unpack_array_tree,
    write_checkpoint_archive,
)
from ._minibatch_posterior import (
    LikelihoodBatch,
    MinibatchPosteriorProblem,
    MinibatchSource,
)
from ._posterior_predictive import (
    predict_from_position_samples,
    sample_observations_from_position_samples,
)
from ._predictive import PredictiveField
from ._sgmcmc_diagnostics import (
    sgmcmc_diagnostics,
    SGMCMCDiagnostics,
    SGMCMCMixingReport,
    SGMCMCMixingThresholds,
)
from ._stochastic_gradient import (
    AbstractStochasticGradientEstimator,
    AutodiffStochasticGradientEstimator,
    StochasticGradientEstimate,
)


SGMCMCAlgorithm = Literal["sgld", "sgnht"]
_APPROXIMATION = "unadjusted_fixed_step"
_INITIALIZATION_TAG = 0
_TRANSITION_TAG = 1
_CHECKPOINT_KIND = "sgmcmc"


class SGMCMCControlVariate(StrictModule):
    """Full-gradient reference used by a minibatch difference estimator."""

    center: PyTree[Array]
    full_gradient: PyTree[Array]
    problem_fingerprint: str = eqx.field(static=True)
    source_fingerprint: str = eqx.field(static=True)
    fingerprint: str = eqx.field(static=True)
    construction_duration_seconds: float = eqx.field(static=True)
    construction_gradient_evaluations: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        center: PyTree[Array],
        full_gradient: PyTree[Array],
        problem_fingerprint: str,
        source_fingerprint: str,
        construction_duration_seconds: float,
        construction_gradient_evaluations: int,
    ):
        payload = {
            "center": array_tree_fingerprint(center),
            "full_gradient": array_tree_fingerprint(full_gradient),
            "problem_fingerprint": str(problem_fingerprint),
            "source_fingerprint": str(source_fingerprint),
        }
        canonical = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        self.center = center
        self.full_gradient = full_gradient
        self.problem_fingerprint = str(problem_fingerprint)
        self.source_fingerprint = str(source_fingerprint)
        self.fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        self.construction_duration_seconds = float(construction_duration_seconds)
        self.construction_gradient_evaluations = int(construction_gradient_evaluations)


class SGMCMCResult(StrictModule):
    """Chain-preserving fixed-step SG-MCMC draws and honest mixing evidence."""

    problem: MinibatchPosteriorProblem
    samples: PyTree[Array]
    unconstrained_samples: PyTree[Array]
    final_states: Any
    burnin_states: Any
    diagnostics: SGMCMCDiagnostics
    gradient_norm: Array
    log_density: Array | None
    thermostat: Array | None
    momentum_norm: Array | None
    root_key: Array
    chain_keys: Array
    control_variate: SGMCMCControlVariate | None
    algorithm: SGMCMCAlgorithm = eqx.field(static=True)
    approximation: str = eqx.field(static=True)
    step_size: float = eqx.field(static=True)
    diffusion: float | None = eqx.field(static=True)
    initial_thermostat: float | None = eqx.field(static=True)
    num_burnin: int = eqx.field(static=True)
    num_samples: int = eqx.field(static=True)
    steps_per_sample: int = eqx.field(static=True)
    num_updates: int = eqx.field(static=True)
    num_gradient_evaluations: int = eqx.field(static=True)
    source_num_factors: int = eqx.field(static=True)
    batch_capacity: int = eqx.field(static=True)
    source_fingerprint: str = eqx.field(static=True)
    _source_configuration_json: str = eqx.field(static=True)
    chain_method: ChainMethod = eqx.field(static=True)
    gradient_estimator_id: str = eqx.field(static=True)
    compilation_duration_seconds: float = eqx.field(static=True)
    burnin_duration_seconds: float = eqx.field(static=True)
    sampling_duration_seconds: float = eqx.field(static=True)
    duration_seconds: float = eqx.field(static=True)
    samples_per_second: float = eqx.field(static=True)
    updates_per_second: float = eqx.field(static=True)
    gradient_evaluations_per_second: float = eqx.field(static=True)
    sample_memory_bytes: int = eqx.field(static=True)
    mean_update_gradient_norm: float = eqx.field(static=True)
    max_update_gradient_norm: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        problem: MinibatchPosteriorProblem,
        samples: PyTree[Array],
        unconstrained_samples: PyTree[Array],
        final_states: Any,
        burnin_states: Any,
        diagnostics: SGMCMCDiagnostics,
        gradient_norm: Array,
        log_density: Array | None,
        thermostat: Array | None,
        momentum_norm: Array | None,
        root_key: Array,
        chain_keys: Array,
        control_variate: SGMCMCControlVariate | None,
        algorithm: SGMCMCAlgorithm,
        step_size: float,
        diffusion: float | None,
        initial_thermostat: float | None,
        num_burnin: int,
        num_samples: int,
        steps_per_sample: int,
        num_updates: int,
        num_gradient_evaluations: int,
        source_num_factors: int,
        batch_capacity: int,
        source_fingerprint: str,
        source_configuration_json: str,
        chain_method: ChainMethod,
        gradient_estimator_id: str,
        compilation_duration_seconds: float,
        burnin_duration_seconds: float,
        sampling_duration_seconds: float,
        mean_update_gradient_norm: float,
        max_update_gradient_norm: float,
    ):
        total_duration = (
            float(compilation_duration_seconds)
            + float(burnin_duration_seconds)
            + float(sampling_duration_seconds)
            + (
                0.0
                if control_variate is None
                else control_variate.construction_duration_seconds
            )
        )
        retained = int(num_samples) * int(chain_keys.shape[0])
        update_count = int(num_updates) * int(chain_keys.shape[0])
        gradient_count = int(num_gradient_evaluations)
        self.problem = problem
        self.samples = samples
        self.unconstrained_samples = unconstrained_samples
        self.final_states = final_states
        self.burnin_states = burnin_states
        self.diagnostics = diagnostics
        self.gradient_norm = jnp.asarray(gradient_norm)
        self.log_density = None if log_density is None else jnp.asarray(log_density)
        self.thermostat = None if thermostat is None else jnp.asarray(thermostat)
        self.momentum_norm = None if momentum_norm is None else jnp.asarray(momentum_norm)
        self.root_key = jnp.asarray(root_key)
        self.chain_keys = jnp.asarray(chain_keys)
        self.control_variate = control_variate
        self.algorithm = algorithm
        self.approximation = _APPROXIMATION
        self.step_size = float(step_size)
        self.diffusion = None if diffusion is None else float(diffusion)
        self.initial_thermostat = (
            None if initial_thermostat is None else float(initial_thermostat)
        )
        self.num_burnin = int(num_burnin)
        self.num_samples = int(num_samples)
        self.steps_per_sample = int(steps_per_sample)
        self.num_updates = int(num_updates)
        self.num_gradient_evaluations = gradient_count
        self.source_num_factors = int(source_num_factors)
        self.batch_capacity = int(batch_capacity)
        self.source_fingerprint = str(source_fingerprint)
        self._source_configuration_json = str(source_configuration_json)
        self.gradient_estimator_id = str(gradient_estimator_id)
        self.chain_method = chain_method
        self.compilation_duration_seconds = float(compilation_duration_seconds)
        self.burnin_duration_seconds = float(burnin_duration_seconds)
        self.sampling_duration_seconds = float(sampling_duration_seconds)
        self.duration_seconds = total_duration
        self.samples_per_second = retained / max(total_duration, 1e-12)
        self.updates_per_second = update_count / max(total_duration, 1e-12)
        self.gradient_evaluations_per_second = gradient_count / max(total_duration, 1e-12)
        self.sample_memory_bytes = (
            _tree_nbytes(samples)
            + _tree_nbytes(unconstrained_samples)
            + int(self.gradient_norm.nbytes)
            + (0 if self.thermostat is None else int(self.thermostat.nbytes))
            + (0 if self.momentum_norm is None else int(self.momentum_norm.nbytes))
            + (0 if self.log_density is None else int(self.log_density.nbytes))
        )
        self.mean_update_gradient_norm = float(mean_update_gradient_norm)
        self.max_update_gradient_norm = float(max_update_gradient_norm)

    @property
    def num_chains(self) -> int:
        return int(self.chain_keys.shape[0])

    @property
    def num_draws(self) -> int:
        return self.num_samples

    @property
    def batch_fraction(self) -> float:
        return self.batch_capacity / self.source_num_factors

    @property
    def source_configuration(self) -> dict[str, Any]:
        return json.loads(self._source_configuration_json)

    def predict(
        self,
        *args: Any,
        batch_size: int | None = None,
        valid_policy: Literal["record", "raise"] = "record",
        chain_dim: str = "__phydra_uq_chain",
        draw_dim: str = "__phydra_uq_draw",
        **kwargs: Any,
    ) -> PredictiveField | frozendict[str, PredictiveField]:
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

    def mixing_report(
        self,
        *,
        max_rhat: float = 1.01,
        min_bulk_ess: float = 400.0,
        min_tail_ess: float = 400.0,
        allow_nonfinite_updates: bool = False,
    ) -> SGMCMCMixingReport:
        thresholds = SGMCMCMixingThresholds(
            max_rhat=max_rhat,
            min_bulk_ess=min_bulk_ess,
            min_tail_ess=min_tail_ess,
            allow_nonfinite_updates=allow_nonfinite_updates,
        )
        return SGMCMCMixingReport(
            diagnostics=self.diagnostics,
            thresholds=thresholds,
            algorithm=self.algorithm,
            approximation=self.approximation,
            num_chains=self.num_chains,
            num_draws=self.num_draws,
            step_size=self.step_size,
            batch_fraction=self.batch_fraction,
            sample_memory_bytes=self.sample_memory_bytes,
            duration_seconds=self.duration_seconds,
            samples_per_second=self.samples_per_second,
        )


def build_sgmcmc_control_variate(
    problem: MinibatchPosteriorProblem,
    source: MinibatchSource,
    center: PyTree[Any],
    /,
) -> SGMCMCControlVariate:
    """Build an exact full-gradient reference from one complete source epoch."""
    source_configuration_json, batches = _validate_problem_source(problem, source)
    del source_configuration_json
    center_position, _ = _prepare_chain_positions(
        problem.initial_position,
        num_chains=1,
        initial_position=center,
    )
    problem.parameter_space.constrain(center_position)
    started = time.perf_counter()
    prior_gradient = jax.grad(problem.parameter_space.unconstrained_log_prior)(
        center_position
    )
    likelihood_gradient = jax.tree_util.tree_map(jnp.zeros_like, center_position)

    def batch_log_likelihood(position, batch):
        physical = problem.parameter_space.constrain(position)
        return jnp.sum(problem.log_likelihood_factors(physical, batch))

    gradient_fn = eqx.filter_jit(
        lambda current, current_batch: jax.grad(batch_log_likelihood)(
            current, current_batch
        )
    )
    for batch in batches:
        batch_gradient = gradient_fn(center_position, batch)
        likelihood_gradient = jax.tree_util.tree_map(
            lambda total, value: total + value,
            likelihood_gradient,
            batch_gradient,
        )
    full_gradient = jax.tree_util.tree_map(
        lambda prior, likelihood: prior + likelihood,
        prior_gradient,
        likelihood_gradient,
    )
    jax.block_until_ready(full_gradient)
    duration = time.perf_counter() - started
    problem_fingerprint = _problem_fingerprint(problem, batches[0])
    return SGMCMCControlVariate(
        center=center_position,
        full_gradient=full_gradient,
        problem_fingerprint=problem_fingerprint,
        source_fingerprint=source.fingerprint,
        construction_duration_seconds=duration,
        construction_gradient_evaluations=len(batches) + 2,
    )


def sample_sgld(
    problem: MinibatchPosteriorProblem,
    source: MinibatchSource,
    /,
    *,
    key: Array,
    step_size: float,
    num_chains: int = 4,
    num_burnin: int = 1000,
    num_samples: int = 1000,
    steps_per_sample: int = 1,
    initial_position: PyTree[Any] | None = None,
    initial_positions: PyTree[Any] | None = None,
    chain_method: ChainMethod = "vectorized",
    control_variate: SGMCMCControlVariate | None = None,
    gradient_estimator: AbstractStochasticGradientEstimator | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int | None = None,
    checkpoint_id: str | None = None,
    resume_from: str | Path | None = None,
) -> SGMCMCResult:
    """Run fixed-step stochastic-gradient Langevin dynamics."""
    return _sample_sgmcmc(
        problem,
        source,
        key=key,
        algorithm="sgld",
        step_size=step_size,
        diffusion=None,
        initial_thermostat=None,
        num_chains=num_chains,
        num_burnin=num_burnin,
        num_samples=num_samples,
        steps_per_sample=steps_per_sample,
        initial_position=initial_position,
        initial_positions=initial_positions,
        chain_method=chain_method,
        control_variate=control_variate,
        gradient_estimator=gradient_estimator,
        checkpoint_path=checkpoint_path,
        checkpoint_every=checkpoint_every,
        checkpoint_id=checkpoint_id,
        resume_from=resume_from,
    )


def sample_sgnht(
    problem: MinibatchPosteriorProblem,
    source: MinibatchSource,
    /,
    *,
    key: Array,
    step_size: float,
    diffusion: float = 0.01,
    initial_thermostat: float | None = None,
    num_chains: int = 4,
    num_burnin: int = 1000,
    num_samples: int = 1000,
    steps_per_sample: int = 1,
    initial_position: PyTree[Any] | None = None,
    initial_positions: PyTree[Any] | None = None,
    chain_method: ChainMethod = "vectorized",
    control_variate: SGMCMCControlVariate | None = None,
    gradient_estimator: AbstractStochasticGradientEstimator | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int | None = None,
    checkpoint_id: str | None = None,
    resume_from: str | Path | None = None,
) -> SGMCMCResult:
    """Run fixed-step stochastic-gradient Nosé-Hoover thermostat dynamics."""
    diffusion_value = float(diffusion)
    thermostat = (
        diffusion_value if initial_thermostat is None else float(initial_thermostat)
    )
    return _sample_sgmcmc(
        problem,
        source,
        key=key,
        algorithm="sgnht",
        step_size=step_size,
        diffusion=diffusion_value,
        initial_thermostat=thermostat,
        num_chains=num_chains,
        num_burnin=num_burnin,
        num_samples=num_samples,
        steps_per_sample=steps_per_sample,
        initial_position=initial_position,
        initial_positions=initial_positions,
        chain_method=chain_method,
        control_variate=control_variate,
        gradient_estimator=gradient_estimator,
        checkpoint_path=checkpoint_path,
        checkpoint_every=checkpoint_every,
        checkpoint_id=checkpoint_id,
        resume_from=resume_from,
    )


def _sample_sgmcmc(
    problem: MinibatchPosteriorProblem,
    source: MinibatchSource,
    /,
    *,
    key: Array,
    algorithm: SGMCMCAlgorithm,
    step_size: float,
    diffusion: float | None,
    initial_thermostat: float | None,
    num_chains: int,
    num_burnin: int,
    num_samples: int,
    steps_per_sample: int,
    initial_position: PyTree[Any] | None,
    initial_positions: PyTree[Any] | None,
    chain_method: ChainMethod,
    control_variate: SGMCMCControlVariate | None,
    gradient_estimator: AbstractStochasticGradientEstimator | None,
    checkpoint_path: str | Path | None,
    checkpoint_every: int | None,
    checkpoint_id: str | None,
    resume_from: str | Path | None,
) -> SGMCMCResult:
    source_configuration_json, initial_batches = _validate_problem_source(problem, source)
    chains = int(num_chains)
    burnin = int(num_burnin)
    draws = int(num_samples)
    thinning = int(steps_per_sample)
    if chains < 2:
        raise ValueError("num_chains must be at least two for mixing diagnostics.")
    if burnin <= 0:
        raise ValueError("num_burnin must be positive.")
    if draws < 4:
        raise ValueError("num_samples must be at least four.")
    if thinning <= 0:
        raise ValueError("steps_per_sample must be positive.")
    step = float(step_size)
    if not jnp.isfinite(step) or step <= 0.0:
        raise ValueError("step_size must be positive and finite.")
    if algorithm == "sgnht":
        if diffusion is None or not jnp.isfinite(diffusion) or diffusion <= 0.0:
            raise ValueError("diffusion must be positive and finite.")
        if initial_thermostat is None or not jnp.isfinite(initial_thermostat):
            raise ValueError("initial_thermostat must be finite.")
    method = _validate_chain_method(chain_method)
    if control_variate is not None and not isinstance(
        control_variate, SGMCMCControlVariate
    ):
        raise TypeError("control_variate must be an SGMCMCControlVariate or None.")
    estimator = (
        AutodiffStochasticGradientEstimator()
        if gradient_estimator is None
        else gradient_estimator
    )
    if not isinstance(estimator, AbstractStochasticGradientEstimator):
        raise TypeError(
            "gradient_estimator must implement AbstractStochasticGradientEstimator."
        )
    if control_variate is not None and not estimator.supports_control_variate:
        raise ValueError(
            "The selected stochastic-gradient estimator does not support "
            "SGMCMCControlVariate."
        )
    if resume_from is not None and (
        initial_position is not None or initial_positions is not None
    ):
        raise ValueError(
            "initial_position and initial_positions cannot be supplied when resuming."
        )

    destination = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else (Path(resume_from) if resume_from is not None else None)
    )
    if checkpoint_every is not None and destination is None:
        raise ValueError("checkpoint_every requires checkpoint_path or resume_from.")
    total_updates = burnin + draws * thinning
    interval = (
        min(100, total_updates) if checkpoint_every is None else int(checkpoint_every)
    )
    if interval <= 0:
        raise ValueError("checkpoint_every must be positive.")
    if destination is not None and (checkpoint_id is None or not str(checkpoint_id)):
        raise ValueError("checkpoint_id is required for SG-MCMC checkpointing.")

    position, chain_positions = _prepare_chain_positions(
        problem.initial_position,
        num_chains=chains,
        initial_position=initial_position,
        initial_positions=initial_positions,
    )
    problem.parameter_space.constrain(position)
    root_key, chain_keys = _split_chain_keys(key, chains)
    problem_fingerprint = _problem_fingerprint(problem, initial_batches[0])
    if control_variate is not None:
        if control_variate.source_fingerprint != source.fingerprint:
            raise ValueError("control_variate was built for a different source.")
        if control_variate.problem_fingerprint != problem_fingerprint:
            raise ValueError("control_variate was built for a different problem.")
        _prepare_chain_positions(
            problem.initial_position,
            num_chains=1,
            initial_position=control_variate.center,
        )

    settings = {
        "algorithm": algorithm,
        "num_chains": chains,
        "num_burnin": burnin,
        "step_size": step,
        "steps_per_sample": thinning,
        "chain_method": method,
        "diffusion": diffusion,
        "initial_thermostat": initial_thermostat,
        "control_variate_fingerprint": (
            None if control_variate is None else control_variate.fingerprint
        ),
        "gradient_estimator": estimator.configuration(),
        "root_key": [int(value) for value in jr.key_data(root_key).reshape(-1)],
    }
    compatibility = (
        {
            "checkpoint_id": str(checkpoint_id),
            "source_type": f"{type(source).__module__}.{type(source).__qualname__}",
            "source_fingerprint": source.fingerprint,
            "source_configuration": json.loads(source_configuration_json),
            "problem_type": f"{type(problem).__module__}.{type(problem).__qualname__}",
            "parameter_tree": array_tree_signature(problem.initial_position),
            "problem_fingerprint": problem_fingerprint,
            "settings": settings,
        }
        if destination is not None
        else None
    )
    gradient_fn = _gradient_estimator(problem, control_variate, estimator)
    state_template = _initialize_states(
        algorithm,
        chain_positions,
        chain_keys,
        initial_thermostat=initial_thermostat,
        chain_method=method,
    )

    if resume_from is None:
        initial_estimator_keys = jax.vmap(
            lambda chain_key: jr.fold_in(chain_key, _INITIALIZATION_TAG + 1)
        )(chain_keys)
        estimates = jax.vmap(
            lambda current, estimator_key: gradient_fn(
                current,
                initial_batches[0],
                estimator_key,
            )
        )(chain_positions, initial_estimator_keys)
        values = estimates.log_density
        gradients = estimates.gradient
        invalid_value_chains = tuple(
            int(index)
            for index in jnp.argwhere(~jnp.isfinite(values) | ~estimates.valid).reshape(
                -1
            )
        )
        invalid_gradient_locations = _invalid_chain_locations(gradients, chains)
        if invalid_value_chains or invalid_gradient_locations:
            raise FloatingPointError(
                "Initial SG-MCMC evaluation is nonfinite; "
                f"log-density chains={invalid_value_chains}, "
                f"gradient locations={invalid_gradient_locations}."
            )
        current_states = state_template
        burnin_states = None
        stored_samples = _empty_sample_tree(position, chains)
        retained_gradient_norm = jnp.empty((chains, 0), dtype=float)
        retained_thermostat = (
            jnp.empty((chains, 0), dtype=float) if algorithm == "sgnht" else None
        )
        retained_momentum_norm = (
            jnp.empty((chains, 0), dtype=float) if algorithm == "sgnht" else None
        )
        completed_updates = 0
        completed_draws = 0
        compilation_duration = 0.0
        burnin_duration = 0.0
        sampling_duration = 0.0
        gradient_evaluations = 1 + chains * (2 if control_variate is not None else 1)
        if control_variate is not None:
            gradient_evaluations += control_variate.construction_gradient_evaluations
        gradient_norm_sum = 0.0
        gradient_norm_count = 0
        gradient_norm_max = 0.0
        min_active_factors = problem.num_factors
        max_active_factors = 0
        nonfinite_update_count = 0
    else:
        if compatibility is None:
            raise RuntimeError("SG-MCMC resume compatibility was not constructed.")
        checkpoint_state, arrays = read_checkpoint_archive(
            resume_from,
            kind=_CHECKPOINT_KIND,
            compatibility=compatibility,
        )
        restored = _read_sgmcmc_checkpoint(
            checkpoint_state,
            arrays,
            algorithm=algorithm,
            state_template=state_template,
            position_template=position,
            num_chains=chains,
            num_burnin=burnin,
            num_samples=draws,
            steps_per_sample=thinning,
        )
        current_states = restored["current_states"]
        burnin_states = restored["burnin_states"]
        stored_samples = restored["samples"]
        retained_gradient_norm = restored["gradient_norm"]
        retained_thermostat = restored["thermostat"]
        retained_momentum_norm = restored["momentum_norm"]
        completed_updates = restored["completed_updates"]
        completed_draws = restored["completed_draws"]
        compilation_duration = restored["compilation_duration_seconds"]
        burnin_duration = restored["burnin_duration_seconds"]
        sampling_duration = restored["sampling_duration_seconds"]
        gradient_evaluations = restored["gradient_evaluations"]
        gradient_evaluations += 1
        gradient_norm_sum = restored["gradient_norm_sum"]
        gradient_norm_count = restored["gradient_norm_count"]
        gradient_norm_max = restored["gradient_norm_max"]
        min_active_factors = restored["min_active_factors"]
        max_active_factors = restored["max_active_factors"]
        nonfinite_update_count = restored["nonfinite_update_count"]

    transition = _compile_transition(
        algorithm,
        problem,
        control_variate,
        estimator,
        step_size=step,
        diffusion=diffusion,
        chain_method=method,
        states=current_states,
        chain_keys=chain_keys,
        batch=initial_batches[0],
    )
    compilation_duration += transition[1]
    advance = transition[0]
    new_samples: list[PyTree[Array]] = []
    new_gradient_norms: list[Array] = []
    new_thermostats: list[Array] = []
    new_momentum_norms: list[Array] = []
    cached_epoch = -1
    epoch_batches: tuple[LikelihoodBatch, ...] = ()

    for update in range(completed_updates, total_updates):
        epoch = update // int(source.batches_per_epoch)
        if epoch != cached_epoch:
            epoch_batches = _materialize_source_epoch(source, epoch)
            cached_epoch = epoch
        batch = epoch_batches[update % int(source.batches_per_epoch)]
        transition_keys = _transition_keys(chain_keys, update)
        update_started = time.perf_counter()
        new_states, gradient_norm, gradient_valid = advance(
            current_states, transition_keys, batch
        )
        jax.block_until_ready(new_states)
        update_duration = time.perf_counter() - update_started
        invalid_state_locations = _invalid_chain_locations(new_states, chains)
        invalid_gradient_locations = tuple(
            f"chain[{int(index)}].gradient_norm"
            for index in jnp.argwhere(
                ~jnp.isfinite(gradient_norm) | ~gradient_valid
            ).reshape(-1)
        )
        if invalid_state_locations or invalid_gradient_locations:
            raise FloatingPointError(
                "Nonfinite SG-MCMC transition at logical update "
                f"{update}; state locations={invalid_state_locations}, "
                f"gradient locations={invalid_gradient_locations}."
            )
        current_states = new_states
        completed_updates = update + 1
        if completed_updates <= burnin:
            burnin_duration += update_duration
        else:
            sampling_duration += update_duration
        active_factors = int(batch.factor_count)
        min_active_factors = min(min_active_factors, active_factors)
        max_active_factors = max(max_active_factors, active_factors)
        gradient_norm_sum += float(jnp.sum(gradient_norm))
        gradient_norm_count += chains
        gradient_norm_max = max(gradient_norm_max, float(jnp.max(gradient_norm)))
        gradient_evaluations += chains * (2 if control_variate is not None else 1)

        if completed_updates == burnin:
            burnin_states = current_states
        if completed_updates > burnin and (completed_updates - burnin) % thinning == 0:
            new_samples.append(_state_position(algorithm, current_states))
            new_gradient_norms.append(gradient_norm)
            if algorithm == "sgnht":
                new_thermostats.append(current_states.xi)
                new_momentum_norms.append(_batched_tree_norm(current_states.momentum))
            completed_draws += 1

        if destination is not None and (
            completed_updates % interval == 0 or completed_updates == total_updates
        ):
            stored_samples = _combine_sample_trees(stored_samples, new_samples)
            retained_gradient_norm = _combine_statistic(
                retained_gradient_norm, new_gradient_norms
            )
            if algorithm == "sgnht":
                retained_thermostat = _combine_statistic(
                    retained_thermostat, new_thermostats
                )
                retained_momentum_norm = _combine_statistic(
                    retained_momentum_norm, new_momentum_norms
                )
            new_samples.clear()
            new_gradient_norms.clear()
            new_thermostats.clear()
            new_momentum_norms.clear()
            if compatibility is None:
                raise RuntimeError("SG-MCMC checkpoint compatibility is unavailable.")
            _write_sgmcmc_checkpoint(
                destination,
                compatibility=compatibility,
                algorithm=algorithm,
                completed_updates=completed_updates,
                completed_draws=completed_draws,
                current_states=current_states,
                burnin_states=burnin_states,
                samples=stored_samples,
                gradient_norm=retained_gradient_norm,
                thermostat=retained_thermostat,
                momentum_norm=retained_momentum_norm,
                compilation_duration_seconds=compilation_duration,
                burnin_duration_seconds=burnin_duration,
                sampling_duration_seconds=sampling_duration,
                gradient_evaluations=gradient_evaluations,
                gradient_norm_sum=gradient_norm_sum,
                gradient_norm_count=gradient_norm_count,
                gradient_norm_max=gradient_norm_max,
                min_active_factors=min_active_factors,
                max_active_factors=max_active_factors,
                nonfinite_update_count=nonfinite_update_count,
            )

    stored_samples = _combine_sample_trees(stored_samples, new_samples)
    retained_gradient_norm = _combine_statistic(
        retained_gradient_norm, new_gradient_norms
    )
    if algorithm == "sgnht":
        retained_thermostat = _combine_statistic(retained_thermostat, new_thermostats)
        retained_momentum_norm = _combine_statistic(
            retained_momentum_norm, new_momentum_norms
        )
    if completed_draws != draws:
        raise RuntimeError("SG-MCMC retained an unexpected number of draws.")
    if burnin_states is None:
        raise RuntimeError("SG-MCMC did not preserve its burn-in terminal state.")
    unconstrained_samples = stored_samples
    samples = problem.parameter_space.constrain(unconstrained_samples)
    log_density = (
        None
        if problem.full_log_likelihood_fn is None
        else _evaluate_full_log_density(problem, unconstrained_samples)
    )
    diagnostics = sgmcmc_diagnostics(
        samples,
        gradient_norm=retained_gradient_norm,
        min_active_factors=min_active_factors,
        max_active_factors=max_active_factors,
        nonfinite_update_count=nonfinite_update_count,
    )
    jax.block_until_ready(diagnostics.max_rhat)
    mean_update_gradient_norm = gradient_norm_sum / max(gradient_norm_count, 1)
    return SGMCMCResult(
        problem=problem,
        samples=samples,
        unconstrained_samples=unconstrained_samples,
        final_states=current_states,
        burnin_states=burnin_states,
        diagnostics=diagnostics,
        gradient_norm=retained_gradient_norm,
        log_density=log_density,
        thermostat=retained_thermostat,
        momentum_norm=retained_momentum_norm,
        root_key=root_key,
        chain_keys=chain_keys,
        control_variate=control_variate,
        algorithm=algorithm,
        step_size=step,
        diffusion=diffusion,
        initial_thermostat=initial_thermostat,
        num_burnin=burnin,
        num_samples=draws,
        steps_per_sample=thinning,
        num_updates=total_updates,
        num_gradient_evaluations=gradient_evaluations,
        source_num_factors=problem.num_factors,
        batch_capacity=int(source.batch_capacity),
        source_fingerprint=source.fingerprint,
        source_configuration_json=source_configuration_json,
        chain_method=method,
        gradient_estimator_id=estimator.estimator_id,
        compilation_duration_seconds=compilation_duration,
        burnin_duration_seconds=burnin_duration,
        sampling_duration_seconds=sampling_duration,
        mean_update_gradient_norm=mean_update_gradient_norm,
        max_update_gradient_norm=gradient_norm_max,
    )


def _gradient_estimator(
    problem: MinibatchPosteriorProblem,
    control_variate: SGMCMCControlVariate | None,
    estimator: AbstractStochasticGradientEstimator | None = None,
):
    if estimator is None:
        ordinary_gradient = jax.grad(problem.log_density_estimate)
        if control_variate is None:
            return ordinary_gradient

        def legacy_control_variate_gradient(position, batch):
            gradient = ordinary_gradient(position, batch)
            center_gradient = ordinary_gradient(control_variate.center, batch)
            return jax.tree_util.tree_map(
                lambda full, value, reference: full + value - reference,
                control_variate.full_gradient,
                gradient,
                center_gradient,
            )

        return legacy_control_variate_gradient

    def ordinary_estimate(position, batch, key):
        return estimator.estimate(problem, position, batch, key)

    if control_variate is None:
        return ordinary_estimate

    def control_variate_estimate(position, batch, key):
        current = ordinary_estimate(position, batch, key)
        center = ordinary_estimate(control_variate.center, batch, key)
        gradient = jax.tree_util.tree_map(
            lambda full, value, reference: full + value - reference,
            control_variate.full_gradient,
            current.gradient,
            center.gradient,
        )
        gradient_norm = _tree_norm(gradient)
        finite = (
            current.valid
            & center.valid
            & jnp.isfinite(gradient_norm)
            & jnp.all(
                jnp.stack(
                    [jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(gradient)]
                )
            )
        )
        return StochasticGradientEstimate(
            gradient=gradient,
            log_density=current.log_density,
            gradient_norm=gradient_norm,
            valid=finite,
            status=jnp.where(finite, 0, 1).astype(jnp.int32),
            likelihood_estimate=current.likelihood_estimate,
            estimator_id=current.estimator_id,
        )

    return control_variate_estimate


def _compile_transition(
    algorithm: SGMCMCAlgorithm,
    problem: MinibatchPosteriorProblem,
    control_variate: SGMCMCControlVariate | None,
    estimator: AbstractStochasticGradientEstimator,
    *,
    step_size: float,
    diffusion: float | None,
    chain_method: ChainMethod,
    states: Any,
    chain_keys: Array,
    batch: LikelihoodBatch,
):
    gradient_fn = _gradient_estimator(problem, control_variate, estimator)
    if algorithm == "sgld":
        integrator = diffusions.overdamped_langevin()

        def one_step(step_key, state, minibatch):
            estimator_key = jr.fold_in(step_key, 0x65726164)
            estimate = gradient_fn(state, minibatch, estimator_key)
            return (
                integrator(
                    step_key,
                    state,
                    estimate.gradient,
                    step_size,
                    1.0,
                ),
                estimate.gradient_norm,
                estimate.valid,
            )

    else:
        if diffusion is None:
            raise ValueError("diffusion is required for SGNHT transitions.")
        integrator = diffusions.sgnht(float(diffusion), 0.0)

        def one_step(step_key, state, minibatch):
            estimator_key = jr.fold_in(step_key, 0x65726164)
            estimate = gradient_fn(state.position, minibatch, estimator_key)
            position, momentum, xi = integrator(
                step_key,
                state.position,
                state.momentum,
                state.xi,
                estimate.gradient,
                step_size,
                1.0,
            )
            return (
                SGNHTState(position, momentum, xi),
                estimate.gradient_norm,
                estimate.valid,
            )

    compile_started = time.perf_counter()
    if chain_method == "vectorized":
        vectorized_transition = cast(
            Any,
            eqx.filter_jit(
                lambda current_states, keys, minibatch: jax.vmap(
                    lambda state, step_key: one_step(step_key, state, minibatch)
                )(current_states, keys)
            ),
        )
        compiled = vectorized_transition.lower(states, chain_keys, batch).compile()

        def advance(current_states, keys, minibatch):
            return compiled(current_states, keys, minibatch)

    else:
        state_values = _unstack_tree(states, int(chain_keys.shape[0]))
        sequential_transition = cast(Any, eqx.filter_jit(one_step))
        compiled = sequential_transition.lower(
            chain_keys[0], state_values[0], batch
        ).compile()

        def advance(current_states, keys, minibatch):
            current_values = _unstack_tree(current_states, int(keys.shape[0]))
            next_states = []
            gradient_norms = []
            gradient_validity = []
            for state, step_key in zip(current_values, keys, strict=True):
                next_state, gradient_norm, gradient_valid = compiled(
                    step_key, state, minibatch
                )
                next_states.append(next_state)
                gradient_norms.append(gradient_norm)
                gradient_validity.append(gradient_valid)
            return (
                _stack_trees(next_states),
                jnp.stack(gradient_norms),
                jnp.stack(gradient_validity),
            )

    compilation_duration = time.perf_counter() - compile_started
    return advance, compilation_duration


def _initialize_states(
    algorithm: SGMCMCAlgorithm,
    positions: PyTree[Array],
    chain_keys: Array,
    *,
    initial_thermostat: float | None,
    chain_method: ChainMethod,
):
    if algorithm == "sgld":
        return positions
    if initial_thermostat is None:
        raise ValueError("initial_thermostat is required for SGNHT states.")
    initialization_keys = jax.vmap(
        lambda chain_key: jr.fold_in(chain_key, _INITIALIZATION_TAG)
    )(chain_keys)
    initialize = lambda position, init_key: init_sgnht(
        position, init_key, float(initial_thermostat)
    )
    if chain_method == "vectorized":
        return jax.jit(jax.vmap(initialize))(positions, initialization_keys)
    position_values = _unstack_tree(positions, int(chain_keys.shape[0]))
    return _stack_trees(
        [
            initialize(position, init_key)
            for position, init_key in zip(
                position_values, initialization_keys, strict=True
            )
        ]
    )


def _transition_keys(chain_keys: Array, update: int, /) -> Array:
    return jax.vmap(
        lambda chain_key: jr.fold_in(jr.fold_in(chain_key, _TRANSITION_TAG), int(update))
    )(chain_keys)


def _state_position(algorithm: SGMCMCAlgorithm, states: Any, /):
    return states if algorithm == "sgld" else states.position


def _tree_norm(tree: PyTree[Any], /) -> Array:
    return jnp.sqrt(
        sum(
            (
                jnp.sum(jnp.asarray(leaf, dtype=float) ** 2)
                for leaf in jax.tree_util.tree_leaves(tree)
            ),
            jnp.zeros((), dtype=float),
        )
    )


def _batched_tree_norm(tree: PyTree[Any], /) -> Array:
    leaves = jax.tree_util.tree_leaves(tree)
    squared = sum(
        (
            jnp.sum(
                jnp.asarray(leaf, dtype=float).reshape((leaf.shape[0], -1)) ** 2,
                axis=1,
            )
            for leaf in leaves
        ),
        jnp.zeros((leaves[0].shape[0],), dtype=float),
    )
    return jnp.sqrt(squared)


def _invalid_chain_locations(
    tree: PyTree[Any],
    num_chains: int,
    /,
) -> tuple[str, ...]:
    locations: list[str] = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        array = jnp.asarray(leaf)
        path_name = jax.tree_util.keystr(path)
        if array.ndim == 0 or int(array.shape[0]) != num_chains:
            locations.append(
                f"{path_name or '<root>'}: expected leading chain axis "
                f"of length {num_chains}"
            )
            continue
        for index in jnp.argwhere(~jnp.isfinite(array)):
            chain = int(index[0])
            trailing = tuple(int(value) for value in index[1:])
            suffix = (
                ""
                if not trailing
                else "[" + ",".join(str(value) for value in trailing) + "]"
            )
            locations.append(f"chain[{chain}]{path_name}{suffix}")
    return tuple(locations)


def _invalid_chain_indices(tree: PyTree[Any], num_chains: int, /) -> tuple[int, ...]:
    valid = jnp.ones((num_chains,), dtype=bool)
    for leaf in jax.tree_util.tree_leaves(tree):
        array = jnp.asarray(leaf)
        if array.ndim == 0 or int(array.shape[0]) != num_chains:
            return tuple(range(num_chains))
        valid = valid & jnp.all(jnp.isfinite(array).reshape((num_chains, -1)), axis=1)
    return tuple(int(index) for index in jnp.argwhere(~valid).reshape(-1))


def _empty_sample_tree(position: PyTree[Any], chains: int, /):
    return jax.tree_util.tree_map(
        lambda value: jnp.empty((chains, 0, *value.shape), dtype=value.dtype),
        position,
    )


def _combine_sample_trees(stored, additions):
    if not additions:
        return stored
    added = jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves, axis=1), *additions)
    return jax.tree_util.tree_map(
        lambda previous, current: jnp.concatenate((previous, current), axis=1),
        stored,
        added,
    )


def _combine_statistic(stored, additions):
    if stored is None:
        return None
    if not additions:
        return stored
    return jnp.concatenate((stored, jnp.stack(additions, axis=1)), axis=1)


def _evaluate_full_log_density(problem, samples):
    leaves = jax.tree_util.tree_leaves(samples)
    chains, draws = int(leaves[0].shape[0]), int(leaves[0].shape[1])
    flattened = jax.tree_util.tree_map(
        lambda value: value.reshape((chains * draws, *value.shape[2:])), samples
    )
    values = jax.vmap(problem.full_log_density)(flattened)
    if not bool(jnp.all(jnp.isfinite(values))):
        raise FloatingPointError(
            "Full log density is nonfinite at retained SG-MCMC draws."
        )
    return values.reshape((chains, draws))


def _validate_problem_source(problem, source):
    if not isinstance(problem, MinibatchPosteriorProblem):
        raise TypeError("problem must be a MinibatchPosteriorProblem.")
    if not isinstance(source, MinibatchSource):
        raise TypeError("source must implement MinibatchSource.")
    if int(source.num_factors) != problem.num_factors:
        raise ValueError("source num_factors does not match the posterior problem.")
    if int(source.batch_capacity) <= 0 or int(source.batches_per_epoch) <= 0:
        raise ValueError("source batch capacity and epoch length must be positive.")
    if not source.fingerprint:
        raise ValueError("source fingerprint must be non-empty.")
    configuration_json = json.dumps(
        source.configuration(),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    batches = _materialize_source_epoch(source, 0)
    return configuration_json, batches


def _materialize_source_epoch(
    source: MinibatchSource, epoch: int, /
) -> tuple[LikelihoodBatch, ...]:
    batches = tuple(source.epoch(epoch))
    if len(batches) != int(source.batches_per_epoch):
        raise ValueError("source epoch length does not match batches_per_epoch.")
    if not batches:
        raise ValueError("source epochs must contain at least one batch.")
    for batch in batches:
        if not isinstance(batch, LikelihoodBatch):
            raise TypeError("source epochs must emit LikelihoodBatch objects.")
        if batch.capacity != int(source.batch_capacity):
            raise ValueError("source emitted a batch with incompatible capacity.")
    if sum(int(batch.factor_count) for batch in batches) != int(source.num_factors):
        raise ValueError("source epoch active-factor count does not match num_factors.")
    return batches


def _problem_fingerprint(
    problem: MinibatchPosteriorProblem,
    batch: LikelihoodBatch,
    /,
) -> str:
    probe, gradient = jax.value_and_grad(problem.log_density_estimate)(
        problem.initial_position,
        batch,
    )
    jax.block_until_ready((probe, gradient))
    payload = {
        "problem_arrays": array_tree_fingerprint(problem)["sha256"],
        "initial_position": array_tree_fingerprint(problem.initial_position),
        "initial_probe": array_tree_fingerprint({"value": probe, "gradient": gradient}),
        "num_factors": problem.num_factors,
    }
    canonical = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _write_sgmcmc_checkpoint(
    destination,
    *,
    compatibility,
    algorithm,
    completed_updates,
    completed_draws,
    current_states,
    burnin_states,
    samples,
    gradient_norm,
    thermostat,
    momentum_norm,
    compilation_duration_seconds,
    burnin_duration_seconds,
    sampling_duration_seconds,
    gradient_evaluations,
    gradient_norm_sum,
    gradient_norm_count,
    gradient_norm_max,
    min_active_factors,
    max_active_factors,
    nonfinite_update_count,
):
    arrays: dict[str, Any] = {
        "gradient_norm": gradient_norm,
    }
    if thermostat is not None:
        arrays["thermostat"] = thermostat
    if momentum_norm is not None:
        arrays["momentum_norm"] = momentum_norm
    state = {
        "algorithm": algorithm,
        "completed_updates": int(completed_updates),
        "completed_draws": int(completed_draws),
        "current_state_tree": pack_array_tree("current_state", current_states, arrays),
        "burnin_state_tree": (
            None
            if burnin_states is None
            else pack_array_tree("burnin_state", burnin_states, arrays)
        ),
        "sample_tree": pack_array_tree("samples", samples, arrays),
        "gradient_norm_array": "gradient_norm",
        "thermostat_array": None if thermostat is None else "thermostat",
        "momentum_norm_array": None if momentum_norm is None else "momentum_norm",
        "compilation_duration_seconds": float(compilation_duration_seconds),
        "burnin_duration_seconds": float(burnin_duration_seconds),
        "sampling_duration_seconds": float(sampling_duration_seconds),
        "gradient_evaluations": int(gradient_evaluations),
        "gradient_norm_sum": float(gradient_norm_sum),
        "gradient_norm_count": int(gradient_norm_count),
        "gradient_norm_max": float(gradient_norm_max),
        "min_active_factors": int(min_active_factors),
        "max_active_factors": int(max_active_factors),
        "nonfinite_update_count": int(nonfinite_update_count),
    }
    write_checkpoint_archive(
        destination,
        kind=_CHECKPOINT_KIND,
        compatibility=compatibility,
        state=state,
        arrays=arrays,
    )


def _read_sgmcmc_checkpoint(
    state,
    arrays,
    *,
    algorithm,
    state_template,
    position_template,
    num_chains,
    num_burnin,
    num_samples,
    steps_per_sample,
):
    if state.get("algorithm") != algorithm:
        raise CheckpointCompatibilityError(
            "Checkpoint algorithm does not match the requested SG-MCMC method."
        )
    completed_updates = _checkpoint_int(state, "completed_updates", minimum=0)
    completed_draws = _checkpoint_int(state, "completed_draws", minimum=0)
    maximum_updates = num_burnin + num_samples * steps_per_sample
    if completed_updates > maximum_updates or completed_draws > num_samples:
        raise CheckpointCompatibilityError(
            "Checkpoint contains more progress than the requested sample count."
        )
    expected_draws = max(0, completed_updates - num_burnin) // steps_per_sample
    if completed_draws != expected_draws:
        raise CheckpointCorruptionError(
            "Checkpoint update and retained-draw progress are inconsistent."
        )
    current_spec = state.get("current_state_tree")
    sample_spec = state.get("sample_tree")
    if not isinstance(current_spec, dict) or not isinstance(sample_spec, dict):
        raise CheckpointCorruptionError("Checkpoint state tree metadata is invalid.")
    current_states = unpack_array_tree(current_spec, arrays, state_template)
    sample_template = jax.tree_util.tree_map(
        lambda value: jnp.empty(
            (num_chains, completed_draws, *value.shape), dtype=value.dtype
        ),
        position_template,
    )
    samples = unpack_array_tree(sample_spec, arrays, sample_template)
    burnin_spec = state.get("burnin_state_tree")
    if completed_updates < num_burnin:
        if burnin_spec is not None:
            raise CheckpointCorruptionError(
                "Checkpoint contains burn-in states before burn-in completion."
            )
        burnin_states = None
    else:
        if not isinstance(burnin_spec, dict):
            raise CheckpointCorruptionError(
                "Checkpoint is missing completed burn-in states."
            )
        burnin_states = unpack_array_tree(burnin_spec, arrays, state_template)
    gradient_norm = _checkpoint_array(
        arrays,
        state.get("gradient_norm_array"),
        shape=(num_chains, completed_draws),
    )
    if algorithm == "sgnht":
        thermostat = _checkpoint_array(
            arrays,
            state.get("thermostat_array"),
            shape=(num_chains, completed_draws),
        )
        momentum_norm = _checkpoint_array(
            arrays,
            state.get("momentum_norm_array"),
            shape=(num_chains, completed_draws),
        )
    else:
        if (
            state.get("thermostat_array") is not None
            or state.get("momentum_norm_array") is not None
        ):
            raise CheckpointCorruptionError(
                "SGLD checkpoint contains thermostat statistics."
            )
        thermostat = None
        momentum_norm = None
    invalid = _invalid_chain_indices(current_states, num_chains)
    if invalid:
        raise CheckpointCorruptionError(
            f"Checkpoint current states are nonfinite for chains {invalid}."
        )
    return {
        "current_states": current_states,
        "burnin_states": burnin_states,
        "samples": samples,
        "gradient_norm": gradient_norm,
        "thermostat": thermostat,
        "momentum_norm": momentum_norm,
        "completed_updates": completed_updates,
        "completed_draws": completed_draws,
        "compilation_duration_seconds": _checkpoint_float(
            state, "compilation_duration_seconds", minimum=0.0
        ),
        "burnin_duration_seconds": _checkpoint_float(
            state, "burnin_duration_seconds", minimum=0.0
        ),
        "sampling_duration_seconds": _checkpoint_float(
            state, "sampling_duration_seconds", minimum=0.0
        ),
        "gradient_evaluations": _checkpoint_int(state, "gradient_evaluations", minimum=0),
        "gradient_norm_sum": _checkpoint_float(state, "gradient_norm_sum", minimum=0.0),
        "gradient_norm_count": _checkpoint_int(state, "gradient_norm_count", minimum=0),
        "gradient_norm_max": _checkpoint_float(state, "gradient_norm_max", minimum=0.0),
        "min_active_factors": _checkpoint_int(state, "min_active_factors", minimum=1),
        "max_active_factors": _checkpoint_int(state, "max_active_factors", minimum=1),
        "nonfinite_update_count": _checkpoint_int(
            state, "nonfinite_update_count", minimum=0
        ),
    }


def _checkpoint_array(arrays, name, *, shape):
    if not isinstance(name, str) or name not in arrays:
        raise CheckpointCorruptionError("Checkpoint statistic array is missing.")
    value = jnp.asarray(arrays[name])
    if value.shape != shape:
        raise CheckpointCorruptionError(
            f"Checkpoint array {name!r} has an invalid shape."
        )
    return value


def _checkpoint_int(state, name, *, minimum):
    value = state.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise CheckpointCorruptionError(f"Checkpoint field {name!r} is invalid.")
    return int(value)


def _checkpoint_float(state, name, *, minimum):
    value = state.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise CheckpointCorruptionError(f"Checkpoint field {name!r} is invalid.")
    result = float(value)
    if not jnp.isfinite(result) or result < minimum:
        raise CheckpointCorruptionError(f"Checkpoint field {name!r} is invalid.")
    return result


__all__ = [
    "build_sgmcmc_control_variate",
    "sample_sgld",
    "sample_sgnht",
    "SGMCMCControlVariate",
    "SGMCMCDiagnostics",
    "SGMCMCMixingReport",
    "SGMCMCMixingThresholds",
    "SGMCMCResult",
]
