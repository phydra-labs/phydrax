#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite, prod
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array

import phydrax.ein as ein

from .._strict import StrictModule
from ..linalg._causal_linear import associative_affine_solve
from ..stochastic import StateSpaceProblem
from ._posterior import ParameterSpace, PosteriorProblem
from ._state_space_path_density import state_space_path_log_density
from ._variational import (
    AbstractVariationalFamily,
    fit_variational,
    VariationalConfig,
    VariationalDiagnostics,
)


def _inverse_softplus(value: float, /) -> Array:
    return jnp.log(jnp.expm1(jnp.asarray(value)))


class GaussianMarkovVariationalFamily(AbstractVariationalFamily):
    """Directed Gaussian Markov path with dense affine dynamics and diagonal noise."""

    initial_location: Array
    initial_raw_scale: Array
    transitions: Array
    offsets: Array
    innovation_raw_scale: Array
    step_valid: Array
    case_shape: tuple[int, ...] = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    num_steps: int = eqx.field(static=True)
    state_size: int = eqx.field(static=True)
    scale_floor: float = eqx.field(static=True)

    def __init__(
        self,
        initial_location: Array,
        initial_raw_scale: Array,
        transitions: Array,
        offsets: Array,
        innovation_raw_scale: Array,
        step_valid: Array,
        /,
        *,
        case_shape: tuple[int, ...],
        state_shape: tuple[int, ...],
        scale_floor: float = 1e-6,
    ):
        cases = tuple(int(size) for size in case_shape)
        state = tuple(int(size) for size in state_shape)
        state_size = prod(state) if state else 1
        initial = jnp.asarray(initial_location)
        initial_scale = jnp.asarray(initial_raw_scale)
        matrices = jnp.asarray(transitions)
        additions = jnp.asarray(offsets)
        innovation_scale = jnp.asarray(innovation_raw_scale)
        valid = jnp.asarray(step_valid, dtype=bool)
        if initial.shape != cases + state or initial_scale.shape != initial.shape:
            raise ValueError("Initial Gaussian Markov parameters have invalid shapes.")
        if valid.ndim != len(cases) + 1 or valid.shape[: len(cases)] != cases:
            raise ValueError("step_valid must have shape case_shape + (time,).")
        steps = int(valid.shape[-1])
        if steps < 1:
            raise ValueError("Gaussian Markov paths require at least one transition.")
        if matrices.shape != cases + (steps, state_size, state_size):
            raise ValueError("transitions have an invalid Gaussian Markov shape.")
        if additions.shape != cases + (steps,) + state:
            raise ValueError("offsets have an invalid Gaussian Markov shape.")
        if innovation_scale.shape != additions.shape:
            raise ValueError("innovation_raw_scale must match offsets.")
        arrays = (initial, initial_scale, matrices, additions, innovation_scale)
        if any(not jnp.issubdtype(array.dtype, jnp.floating) for array in arrays):
            raise TypeError("Gaussian Markov parameters must be real floating arrays.")
        floor = float(scale_floor)
        if not isfinite(floor) or floor <= 0.0:
            raise ValueError("scale_floor must be positive and finite.")
        self.initial_location = initial
        self.initial_raw_scale = initial_scale
        self.transitions = matrices
        self.offsets = additions
        self.innovation_raw_scale = innovation_scale
        self.step_valid = valid
        self.case_shape = cases
        self.state_shape = state
        self.num_steps = steps
        self.state_size = state_size
        self.scale_floor = floor

    @classmethod
    def from_problem(
        cls,
        problem: StateSpaceProblem,
        /,
        *,
        initial_scale: float = 0.5,
        scale_floor: float = 1e-6,
    ) -> "GaussianMarkovVariationalFamily":
        if not isinstance(problem, StateSpaceProblem):
            raise TypeError("problem must be a StateSpaceProblem.")
        scale = float(initial_scale)
        floor = float(scale_floor)
        if not isfinite(scale) or scale <= floor:
            raise ValueError("initial_scale must be finite and exceed scale_floor.")
        case_shape = problem.observations.case_shape
        state_shape = problem.model.state_shape
        state_size = prod(state_shape) if state_shape else 1
        steps = problem.observations.num_steps
        location = jnp.asarray(problem.model.prior.location)
        dtype = location.dtype
        raw = _inverse_softplus(scale - floor).astype(dtype)
        return cls(
            location,
            jnp.full_like(location, raw),
            jnp.zeros(
                case_shape + (steps, state_size, state_size),
                dtype=dtype,
            ),
            jnp.broadcast_to(
                jnp.expand_dims(location, axis=len(case_shape)),
                case_shape + (steps,) + state_shape,
            ),
            jnp.full(case_shape + (steps,) + state_shape, raw, dtype=dtype),
            problem.observations.step_valid,
            case_shape=case_shape,
            state_shape=state_shape,
            scale_floor=floor,
        )

    @property
    def family_id(self) -> str:
        return "gaussian-markov-path"

    @property
    def initial_scale(self) -> Array:
        return jax.nn.softplus(self.initial_raw_scale) + self.scale_floor

    @property
    def innovation_scale(self) -> Array:
        return jax.nn.softplus(self.innovation_raw_scale) + self.scale_floor

    def _flat_parameters(self):
        case_count = prod(self.case_shape) if self.case_shape else 1
        return (
            self.initial_location.reshape((case_count, self.state_size)),
            self.initial_scale.reshape((case_count, self.state_size)),
            self.transitions.reshape(
                (case_count, self.num_steps, self.state_size, self.state_size)
            ),
            self.offsets.reshape((case_count, self.num_steps, self.state_size)),
            self.innovation_scale.reshape((case_count, self.num_steps, self.state_size)),
            self.step_valid.reshape((case_count, self.num_steps)),
        )

    def sample_and_log_prob(
        self,
        key: Array,
        /,
        *,
        sample_shape: tuple[int, ...] = (),
    ) -> tuple[Array, Array]:
        shape = tuple(int(size) for size in sample_shape)
        if any(size <= 0 for size in shape):
            raise ValueError("sample_shape dimensions must be positive.")
        sample_count = prod(shape) if shape else 1
        (
            initial_location,
            initial_scale,
            transitions,
            offsets,
            innovation_scale,
            valid,
        ) = self._flat_parameters()
        case_count = initial_location.shape[0]
        initial_key, innovation_key = jr.split(key)
        initial_noise = jr.normal(
            initial_key,
            (sample_count, case_count, self.state_size),
            dtype=initial_location.dtype,
        )
        innovation_noise = jr.normal(
            innovation_key,
            (sample_count, case_count, self.num_steps, self.state_size),
            dtype=initial_location.dtype,
        )
        initial_states = (
            initial_location[None, ...] + initial_scale[None, ...] * initial_noise
        )
        identity = jnp.eye(self.state_size, dtype=initial_location.dtype)
        effective_transitions = jnp.where(
            valid[..., None, None],
            transitions,
            identity,
        )
        effective_offsets = jnp.where(
            valid[..., None],
            offsets[None, ...] + innovation_scale[None, ...] * innovation_noise,
            0.0,
        )
        effective_transitions = jnp.broadcast_to(
            effective_transitions,
            (sample_count,) + effective_transitions.shape,
        )
        first_values = (
            ein.contract(
                "nctij,ncj->ncti",
                effective_transitions[:, :, :1],
                initial_states,
            )[:, :, 0]
            + effective_offsets[:, :, 0]
        )
        scan_transitions = effective_transitions.at[:, :, 0].set(
            jnp.zeros_like(effective_transitions[:, :, 0])
        )
        scan_offsets = effective_offsets.at[:, :, 0].set(first_values)

        def one_path(path_transitions, path_offsets):
            return associative_affine_solve(path_transitions, path_offsets)

        later_states = jax.vmap(
            jax.vmap(one_path, in_axes=(0, 0)),
            in_axes=(0, 0),
        )(scan_transitions, scan_offsets)
        flat_path = jnp.concatenate((initial_states[:, :, None, :], later_states), axis=2)
        path = flat_path.reshape(
            shape + self.case_shape + (self.num_steps + 1,) + self.state_shape
        )
        if not shape:
            path = path.reshape(
                self.case_shape + (self.num_steps + 1,) + self.state_shape
            )
        return path, self.log_prob(path)

    def log_prob(self, value: Array, /) -> Array:
        path = jnp.asarray(value)
        event_shape = self.case_shape + (self.num_steps + 1,) + self.state_shape
        if path.ndim < len(event_shape) or path.shape[-len(event_shape) :] != event_shape:
            raise ValueError("value has an incompatible Gaussian Markov path shape.")
        sample_shape = path.shape[: path.ndim - len(event_shape)]
        sample_count = prod(sample_shape) if sample_shape else 1
        case_count = prod(self.case_shape) if self.case_shape else 1
        flat_path = path.reshape(
            (sample_count, case_count, self.num_steps + 1, self.state_size)
        )
        (
            initial_location,
            initial_scale,
            transitions,
            offsets,
            innovation_scale,
            valid,
        ) = self._flat_parameters()
        log_two_pi = jnp.log(jnp.asarray(2.0 * jnp.pi, dtype=path.dtype))
        initial_standardized = (
            flat_path[:, :, 0] - initial_location[None, ...]
        ) / initial_scale[None, ...]
        initial_log_prob = -0.5 * jnp.sum(
            jnp.square(initial_standardized)
            + log_two_pi
            + 2.0 * jnp.log(initial_scale[None, ...]),
            axis=-1,
        )
        means = (
            ein.contract(
                "ctij,nctj->ncti",
                transitions,
                flat_path[:, :, :-1],
            )
            + offsets[None, ...]
        )
        standardized = (flat_path[:, :, 1:] - means) / innovation_scale[None, ...]
        transition_log_prob = -0.5 * jnp.sum(
            jnp.square(standardized)
            + log_two_pi
            + 2.0 * jnp.log(innovation_scale[None, ...]),
            axis=-1,
        )
        transition_log_prob = jnp.where(valid[None, ...], transition_log_prob, 0.0)
        total = jnp.sum(initial_log_prob, axis=-1) + jnp.sum(
            transition_log_prob,
            axis=(-2, -1),
        )
        return total.reshape(sample_shape) if sample_shape else total.reshape(())

    def log_prob_terms(self, value: Array, /) -> tuple[Array, Array]:
        """Return normalized initial and per-transition log-density terms."""
        path = jnp.asarray(value)
        event_shape = self.case_shape + (self.num_steps + 1,) + self.state_shape
        if path.ndim < len(event_shape) or path.shape[-len(event_shape) :] != event_shape:
            raise ValueError("value has an incompatible Gaussian Markov path shape.")
        sample_shape = path.shape[: path.ndim - len(event_shape)]
        sample_count = prod(sample_shape) if sample_shape else 1
        case_count = prod(self.case_shape) if self.case_shape else 1
        flat_path = path.reshape(
            (sample_count, case_count, self.num_steps + 1, self.state_size)
        )
        (
            initial_location,
            initial_scale,
            transitions,
            offsets,
            innovation_scale,
            valid,
        ) = self._flat_parameters()
        log_two_pi = jnp.log(jnp.asarray(2.0 * jnp.pi, dtype=path.dtype))
        initial_standardized = (
            flat_path[:, :, 0] - initial_location[None, ...]
        ) / initial_scale[None, ...]
        initial = -0.5 * jnp.sum(
            jnp.square(initial_standardized)
            + log_two_pi
            + 2.0 * jnp.log(initial_scale[None, ...]),
            axis=-1,
        )
        means = (
            ein.contract(
                "ctij,nctj->ncti",
                transitions,
                flat_path[:, :, :-1],
            )
            + offsets[None, ...]
        )
        standardized = (flat_path[:, :, 1:] - means) / innovation_scale[None, ...]
        transition = -0.5 * jnp.sum(
            jnp.square(standardized)
            + log_two_pi
            + 2.0 * jnp.log(innovation_scale[None, ...]),
            axis=-1,
        )
        transition = jnp.where(valid[None, ...], transition, 0.0)
        initial_shape = sample_shape + self.case_shape
        transition_shape = sample_shape + self.case_shape + (self.num_steps,)
        return initial.reshape(initial_shape), transition.reshape(transition_shape)


class StateSpaceVariationalConfig(StrictModule):
    """Full-path Gaussian Markov initialization and optimization controls."""

    optimization: VariationalConfig
    initial_scale: float = eqx.field(static=True)
    scale_floor: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        optimization: VariationalConfig | None = None,
        initial_scale: float = 0.5,
        scale_floor: float = 1e-6,
    ):
        optimization_ = VariationalConfig() if optimization is None else optimization
        if not isinstance(optimization_, VariationalConfig):
            raise TypeError("optimization must be VariationalConfig or None.")
        scale = float(initial_scale)
        floor = float(scale_floor)
        if not isfinite(scale) or not isfinite(floor) or scale <= floor or floor <= 0.0:
            raise ValueError("initial_scale must be finite and exceed a positive floor.")
        self.optimization = optimization_
        self.initial_scale = scale
        self.scale_floor = floor


class StateSpaceVariationalResult(StrictModule):
    """Fitted full latent paths and normalized model/family densities."""

    problem: StateSpaceProblem
    family: GaussianMarkovVariationalFamily
    states: Array
    log_model: Array
    log_variational: Array
    diagnostics: VariationalDiagnostics
    root_key: Array
    config: StateSpaceVariationalConfig
    duration_seconds: float = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)

    @property
    def num_draws(self) -> int:
        return int(self.log_model.shape[0])


def fit_state_space_variational(
    problem: StateSpaceProblem,
    /,
    *,
    key: Array,
    family: GaussianMarkovVariationalFamily | None = None,
    config: StateSpaceVariationalConfig | None = None,
    num_samples: int = 1000,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int | None = None,
    checkpoint_id: str | None = None,
    resume_from: str | Path | None = None,
) -> StateSpaceVariationalResult:
    """Fit a normalized full-path Gaussian Markov posterior approximation."""

    if not isinstance(problem, StateSpaceProblem):
        raise TypeError("problem must be a StateSpaceProblem.")
    config_ = StateSpaceVariationalConfig() if config is None else config
    if not isinstance(config_, StateSpaceVariationalConfig):
        raise TypeError("config must be StateSpaceVariationalConfig or None.")
    family_ = (
        GaussianMarkovVariationalFamily.from_problem(
            problem,
            initial_scale=config_.initial_scale,
            scale_floor=config_.scale_floor,
        )
        if family is None
        else family
    )
    if not isinstance(family_, GaussianMarkovVariationalFamily):
        raise TypeError("family must be GaussianMarkovVariationalFamily or None.")
    initial_path, _ = family_.sample_and_log_prob(jr.fold_in(key, 0x51A7E))
    path_space = ParameterSpace(
        initial_path,
        log_prior=lambda _: jnp.zeros((), dtype=initial_path.dtype),
    )
    path_problem = PosteriorProblem(
        path_space,
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
    return StateSpaceVariationalResult(
        problem=problem,
        family=fitted.family,
        states=fitted.unconstrained_samples,
        log_model=log_model,
        log_variational=fitted.log_variational,
        diagnostics=fitted.diagnostics,
        root_key=fitted.root_key,
        config=config_,
        duration_seconds=fitted.duration_seconds,
        approximation_id="reverse-kl/gaussian-markov-state-space-path",
    )


__all__ = [
    "fit_state_space_variational",
    "GaussianMarkovVariationalFamily",
    "StateSpaceVariationalConfig",
    "StateSpaceVariationalResult",
]
