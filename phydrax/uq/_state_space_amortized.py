#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite, prod, sqrt
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import opt_einsum as oe
from jax.typing import DTypeLike
from jaxtyping import Array

from .._strict import StrictModule
from ..stochastic import StateSpaceProblem
from ._posterior import ParameterSpace, PosteriorProblem
from ._state_space_path_density import state_space_path_log_density
from ._state_space_variational import GaussianMarkovVariationalFamily
from ._variational import (
    AbstractVariationalFamily,
    fit_variational,
    VariationalConfig,
    VariationalDiagnostics,
)


def _sequence_features(problem: StateSpaceProblem, /) -> Array:
    observations = problem.observations
    case_count = prod(observations.case_shape) if observations.case_shape else 1
    steps = observations.num_steps
    observation_size = prod(observations.observation_shape)
    values = observations.values.reshape((case_count, steps, observation_size))
    masks = observations.observation_mask.reshape((case_count, steps, observation_size))
    valid = observations.step_valid.reshape((case_count, steps))
    times = observations.times.reshape((case_count, steps))
    initial = problem.initial_time.reshape((case_count,))
    valid_count = jnp.sum(valid, axis=-1, dtype=jnp.int32)
    final_time = jnp.take_along_axis(
        times,
        (valid_count - 1)[:, None],
        axis=1,
    )[:, 0]
    duration = jnp.maximum(final_time - initial, jnp.finfo(times.dtype).eps)
    normalized_time = (times - initial[:, None]) / duration[:, None]
    return jnp.concatenate(
        (
            jnp.where(masks, values, 0.0),
            masks.astype(values.dtype),
            normalized_time[..., None],
        ),
        axis=-1,
    )


class AmortizedGaussianMarkovEncoder(StrictModule):
    """Shared observation encoder producing one Gaussian Markov path family."""

    hidden_weight: Array
    hidden_bias: Array
    temporal_weight: Array
    temporal_bias: Array
    initial_weight: Array
    initial_bias: Array
    input_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    state_size: int = eqx.field(static=True)

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        state_size: int,
        /,
        *,
        key: Array,
        dtype: DTypeLike = jnp.float32,
    ):
        inputs = int(input_size)
        hidden = int(hidden_size)
        state = int(state_size)
        if inputs < 1 or hidden < 1 or state < 1:
            raise ValueError("Amortized encoder dimensions must be positive.")
        hidden_key, temporal_key, initial_key = jr.split(key, 3)
        hidden_scale = 1.0 / sqrt(float(inputs))
        head_scale = 1.0 / sqrt(float(hidden))
        self.hidden_weight = hidden_scale * jr.normal(
            hidden_key, (hidden, inputs), dtype=dtype
        )
        self.hidden_bias = jnp.zeros((hidden,), dtype=dtype)
        self.temporal_weight = (
            0.05 * head_scale * jr.normal(temporal_key, (3 * state, hidden), dtype=dtype)
        )
        self.temporal_bias = jnp.zeros((3 * state,), dtype=dtype)
        self.initial_weight = (
            0.05 * head_scale * jr.normal(initial_key, (2 * state, hidden), dtype=dtype)
        )
        self.initial_bias = jnp.zeros((2 * state,), dtype=dtype)
        self.input_size = inputs
        self.hidden_size = hidden
        self.state_size = state

    def family(
        self,
        features: Array,
        step_valid: Array,
        prior_location: Array,
        /,
        *,
        case_shape: tuple[int, ...],
        state_shape: tuple[int, ...],
        scale_floor: float,
        context_valid: Array | None = None,
    ) -> GaussianMarkovVariationalFamily:
        feature_values = jax.lax.stop_gradient(jnp.asarray(features))
        valid = jax.lax.stop_gradient(jnp.asarray(step_valid, dtype=bool))
        context = (
            valid
            if context_valid is None
            else jax.lax.stop_gradient(jnp.asarray(context_valid, dtype=bool))
        )
        if context.shape != valid.shape:
            raise ValueError("context_valid must match step_valid.")
        prior = jax.lax.stop_gradient(jnp.asarray(prior_location))
        case_count, steps, feature_size = feature_values.shape
        if feature_size != self.input_size:
            raise ValueError("Amortized feature size does not match the encoder.")
        local_hidden = jnp.tanh(
            oe.contract("hi,cti->cth", self.hidden_weight, feature_values)
            + self.hidden_bias
        )
        context_float = context.astype(local_hidden.dtype)
        masked_hidden = local_hidden * context_float[..., None]
        prefix_count = jnp.cumsum(context_float, axis=1)
        prefix = jnp.cumsum(masked_hidden, axis=1) / jnp.maximum(
            prefix_count[..., None], 1.0
        )
        suffix_count = jnp.cumsum(context_float[:, ::-1], axis=1)[:, ::-1]
        suffix = jnp.cumsum(masked_hidden[:, ::-1], axis=1)[:, ::-1] / jnp.maximum(
            suffix_count[..., None], 1.0
        )
        hidden = jnp.tanh(local_hidden + 0.5 * (prefix + suffix))
        temporal = (
            oe.contract("oh,cth->cto", self.temporal_weight, hidden) + self.temporal_bias
        )
        transition_raw, offsets, raw_scale = jnp.split(temporal, 3, axis=-1)
        transition_diagonal = 0.95 * jnp.tanh(transition_raw)
        transitions = jax.vmap(jax.vmap(jnp.diag))(transition_diagonal)
        pooled = jnp.sum(hidden * context_float[..., None], axis=1) / jnp.maximum(
            jnp.sum(context_float, axis=1, keepdims=True),
            1.0,
        )
        initial = oe.contract("oh,ch->co", self.initial_weight, pooled) + self.initial_bias
        initial_offset, initial_raw_scale = jnp.split(initial, 2, axis=-1)
        flat_prior = prior.reshape((case_count, self.state_size))
        initial_location = flat_prior + initial_offset
        offsets = flat_prior[:, None, :] + offsets
        return GaussianMarkovVariationalFamily(
            initial_location.reshape(case_shape + state_shape),
            initial_raw_scale.reshape(case_shape + state_shape),
            transitions.reshape(case_shape + (steps, self.state_size, self.state_size)),
            offsets.reshape(case_shape + (steps,) + state_shape),
            raw_scale.reshape(case_shape + (steps,) + state_shape),
            valid.reshape(case_shape + (steps,)),
            case_shape=case_shape,
            state_shape=state_shape,
            scale_floor=scale_floor,
        )


class AmortizedGaussianMarkovFamily(AbstractVariationalFamily):
    """Observation-conditioned Gaussian Markov family with shared encoder arrays."""

    encoder: AmortizedGaussianMarkovEncoder
    features: Array
    step_valid: Array
    context_mask: Array
    prior_location: Array
    case_shape: tuple[int, ...] = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    scale_floor: float = eqx.field(static=True)

    def __init__(
        self,
        encoder: AmortizedGaussianMarkovEncoder,
        problem: StateSpaceProblem,
        /,
        *,
        scale_floor: float = 1e-6,
        context_mask: Array | None = None,
    ):
        if not isinstance(encoder, AmortizedGaussianMarkovEncoder):
            raise TypeError("encoder must be AmortizedGaussianMarkovEncoder.")
        if not isinstance(problem, StateSpaceProblem):
            raise TypeError("problem must be StateSpaceProblem.")
        floor = float(scale_floor)
        if not isfinite(floor) or floor <= 0.0:
            raise ValueError("scale_floor must be positive and finite.")
        features = _sequence_features(problem)
        if features.shape[-1] != encoder.input_size:
            raise ValueError("Problem observations do not match the encoder input size.")
        state_size = prod(problem.model.state_shape) if problem.model.state_shape else 1
        if state_size != encoder.state_size:
            raise ValueError("Problem state size does not match the encoder.")
        self.encoder = encoder
        self.features = features
        self.step_valid = problem.observations.step_valid
        context = (
            problem.observations.step_valid
            if context_mask is None
            else jnp.asarray(context_mask, dtype=bool)
        )
        if context.shape != problem.observations.step_valid.shape:
            raise ValueError("context_mask must match the observation step mask.")
        self.context_mask = context
        self.prior_location = problem.model.prior.location
        self.case_shape = problem.observations.case_shape
        self.state_shape = problem.model.state_shape
        self.scale_floor = floor

    @classmethod
    def from_problem(
        cls,
        problem: StateSpaceProblem,
        /,
        *,
        hidden_size: int = 64,
        scale_floor: float = 1e-6,
        key: Array,
    ) -> "AmortizedGaussianMarkovFamily":
        features = _sequence_features(problem)
        state_size = prod(problem.model.state_shape) if problem.model.state_shape else 1
        encoder = AmortizedGaussianMarkovEncoder(
            int(features.shape[-1]),
            hidden_size,
            state_size,
            key=key,
            dtype=features.dtype,
        )
        return cls(encoder, problem, scale_floor=scale_floor)

    @property
    def family_id(self) -> str:
        return "amortized-gaussian-markov-path"

    @property
    def conditional_family(self) -> GaussianMarkovVariationalFamily:
        case_count = prod(self.case_shape) if self.case_shape else 1
        return self.encoder.family(
            self.features.reshape(
                (case_count, self.step_valid.shape[-1], self.encoder.input_size)
            ),
            self.step_valid.reshape((case_count, self.step_valid.shape[-1])),
            self.prior_location,
            context_valid=self.context_mask.reshape(
                (case_count, self.step_valid.shape[-1])
            ),
            case_shape=self.case_shape,
            state_shape=self.state_shape,
            scale_floor=self.scale_floor,
        )

    def condition(
        self,
        problem: StateSpaceProblem,
        /,
    ) -> "AmortizedGaussianMarkovFamily":
        return AmortizedGaussianMarkovFamily(
            self.encoder,
            problem,
            scale_floor=self.scale_floor,
        )

    def sample_and_log_prob(
        self,
        key: Array,
        /,
        *,
        sample_shape: tuple[int, ...] = (),
    ):
        return self.conditional_family.sample_and_log_prob(
            key,
            sample_shape=sample_shape,
        )

    def log_prob(self, value: Array, /) -> Array:
        return self.conditional_family.log_prob(value)


class AmortizedStateSpaceVariationalConfig(StrictModule):
    """Shared encoder shape and full-path reverse-KL controls."""

    optimization: VariationalConfig
    hidden_size: int = eqx.field(static=True)
    scale_floor: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        optimization: VariationalConfig | None = None,
        hidden_size: int = 64,
        scale_floor: float = 1e-6,
    ):
        optimization_ = VariationalConfig() if optimization is None else optimization
        if not isinstance(optimization_, VariationalConfig):
            raise TypeError("optimization must be VariationalConfig or None.")
        hidden = int(hidden_size)
        floor = float(scale_floor)
        if hidden < 1:
            raise ValueError("hidden_size must be positive.")
        if not isfinite(floor) or floor <= 0.0:
            raise ValueError("scale_floor must be positive and finite.")
        self.optimization = optimization_
        self.hidden_size = hidden
        self.scale_floor = floor


class AmortizedStateSpaceVariationalResult(StrictModule):
    """Fitted reusable encoder and full latent-path posterior draws."""

    problem: StateSpaceProblem
    family: AmortizedGaussianMarkovFamily
    states: Array
    log_model: Array
    log_variational: Array
    diagnostics: VariationalDiagnostics
    root_key: Array
    config: AmortizedStateSpaceVariationalConfig
    duration_seconds: float = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)

    @property
    def num_draws(self) -> int:
        return int(self.log_model.shape[0])


def fit_amortized_state_space_variational(
    problem: StateSpaceProblem,
    /,
    *,
    key: Array,
    family: AmortizedGaussianMarkovFamily | None = None,
    config: AmortizedStateSpaceVariationalConfig | None = None,
    num_samples: int = 1000,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int | None = None,
    checkpoint_id: str | None = None,
    resume_from: str | Path | None = None,
) -> AmortizedStateSpaceVariationalResult:
    """Fit one observation-conditioned full-path posterior encoder."""

    if not isinstance(problem, StateSpaceProblem):
        raise TypeError("problem must be StateSpaceProblem.")
    config_ = AmortizedStateSpaceVariationalConfig() if config is None else config
    if not isinstance(config_, AmortizedStateSpaceVariationalConfig):
        raise TypeError("config must be AmortizedStateSpaceVariationalConfig or None.")
    family_ = (
        AmortizedGaussianMarkovFamily.from_problem(
            problem,
            hidden_size=config_.hidden_size,
            scale_floor=config_.scale_floor,
            key=jr.fold_in(key, 0xA60B),
        )
        if family is None
        else family
    )
    if not isinstance(family_, AmortizedGaussianMarkovFamily):
        raise TypeError("family must be AmortizedGaussianMarkovFamily or None.")
    initial_path, _ = family_.sample_and_log_prob(jr.fold_in(key, 0x1A17))
    path_problem = PosteriorProblem(
        ParameterSpace(
            initial_path,
            log_prior=lambda _: jnp.zeros((), dtype=initial_path.dtype),
        ),
        lambda path: state_space_path_log_density(problem, path).log_density,
    )
    fitted = fit_variational(
        path_problem,
        key=key,
        family=family_,
        config=config_.optimization,
        num_samples=num_samples,
        checkpoint_path=checkpoint_path,
        checkpoint_every=checkpoint_every,
        checkpoint_id=checkpoint_id,
        resume_from=resume_from,
    )
    log_model = jax.vmap(
        lambda path: state_space_path_log_density(problem, path).log_density
    )(fitted.unconstrained_samples)
    return AmortizedStateSpaceVariationalResult(
        problem=problem,
        family=fitted.family,
        states=fitted.unconstrained_samples,
        log_model=log_model,
        log_variational=fitted.log_variational,
        diagnostics=fitted.diagnostics,
        root_key=fitted.root_key,
        config=config_,
        duration_seconds=fitted.duration_seconds,
        approximation_id="reverse-kl/amortized-gaussian-markov-path",
    )


__all__ = [
    "AmortizedGaussianMarkovEncoder",
    "AmortizedGaussianMarkovFamily",
    "AmortizedStateSpaceVariationalConfig",
    "AmortizedStateSpaceVariationalResult",
    "fit_amortized_state_space_variational",
]
