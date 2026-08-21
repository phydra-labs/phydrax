#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from time import perf_counter

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jaxtyping import Array

from .._strict import StrictModule
from ..stochastic import StateSpaceProblem
from ._state_space_amortized import AmortizedGaussianMarkovFamily
from ._state_space_path_density import state_space_path_log_density
from ._variational import _tree_all_finite, VariationalConfig


class StateSpaceWindowBatch(StrictModule):
    """One target interval, conditioning context, and inclusion probabilities."""

    target_start: Array
    context_start: Array
    context_end: Array
    target_mask: Array
    context_mask: Array
    inclusion_probability: Array


class StateSpaceWindowPlan(StrictModule):
    """Uniform fixed-length target windows with explicit edge probabilities."""

    inclusion_probability: Array
    num_steps: int = eqx.field(static=True)
    target_length: int = eqx.field(static=True)
    left_buffer: int = eqx.field(static=True)
    right_buffer: int = eqx.field(static=True)
    num_starts: int = eqx.field(static=True)

    def __init__(
        self,
        num_steps: int,
        /,
        *,
        target_length: int,
        left_buffer: int = 0,
        right_buffer: int = 0,
    ):
        steps = int(num_steps)
        target = int(target_length)
        left = int(left_buffer)
        right = int(right_buffer)
        if steps < 1 or target < 1:
            raise ValueError("num_steps and target_length must be positive.")
        if target > steps:
            raise ValueError("target_length cannot exceed num_steps.")
        if left < 0 or right < 0:
            raise ValueError("Window buffer lengths cannot be negative.")
        starts = steps - target + 1
        indices = jnp.arange(steps)
        lower = jnp.maximum(0, indices - target + 1)
        upper = jnp.minimum(indices, starts - 1)
        counts = jnp.maximum(0, upper - lower + 1)
        inclusion = counts.astype(float) / float(starts)
        self.inclusion_probability = inclusion
        self.num_steps = steps
        self.target_length = target
        self.left_buffer = left
        self.right_buffer = right
        self.num_starts = starts

    def sample(self, key: Array, /) -> StateSpaceWindowBatch:
        start = jr.randint(
            key,
            (),
            minval=0,
            maxval=self.num_starts,
            dtype=jnp.int32,
        )
        indices = jnp.arange(self.num_steps, dtype=jnp.int32)
        target_end = start + self.target_length
        context_start = jnp.maximum(0, start - self.left_buffer)
        context_end = jnp.minimum(
            self.num_steps,
            target_end + self.right_buffer,
        )
        return StateSpaceWindowBatch(
            target_start=start,
            context_start=context_start,
            context_end=context_end,
            target_mask=(indices >= start) & (indices < target_end),
            context_mask=(indices >= context_start) & (indices < context_end),
            inclusion_probability=self.inclusion_probability,
        )


class BufferedStateSpaceVariationalConfig(StrictModule):
    """Buffered target/context geometry and amortized optimization controls."""

    optimization: VariationalConfig
    target_length: int = eqx.field(static=True)
    left_buffer: int = eqx.field(static=True)
    right_buffer: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    scale_floor: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        target_length: int,
        left_buffer: int = 0,
        right_buffer: int = 0,
        hidden_size: int = 64,
        scale_floor: float = 1e-6,
        optimization: VariationalConfig | None = None,
    ):
        optimization_ = VariationalConfig() if optimization is None else optimization
        if not isinstance(optimization_, VariationalConfig):
            raise TypeError("optimization must be VariationalConfig or None.")
        target = int(target_length)
        left = int(left_buffer)
        right = int(right_buffer)
        hidden = int(hidden_size)
        floor = float(scale_floor)
        if target < 1 or hidden < 1:
            raise ValueError("target_length and hidden_size must be positive.")
        if left < 0 or right < 0:
            raise ValueError("Window buffer lengths cannot be negative.")
        if not isfinite(floor) or floor <= 0.0:
            raise ValueError("scale_floor must be positive and finite.")
        self.optimization = optimization_
        self.target_length = target
        self.left_buffer = left
        self.right_buffer = right
        self.hidden_size = hidden
        self.scale_floor = floor


class BufferedStateSpaceVariationalDiagnostics(StrictModule):
    """Window starts, context bounds, ELBO estimates, and gradient norms."""

    steps: Array
    target_start: Array
    context_start: Array
    context_end: Array
    elbo: Array
    gradient_norm: Array
    finite: Array


class BufferedStateSpaceVariationalResult(StrictModule):
    """Buffered-trained reusable encoder with full-context posterior draws."""

    problem: StateSpaceProblem
    family: AmortizedGaussianMarkovFamily
    states: Array
    log_model: Array
    log_variational: Array
    diagnostics: BufferedStateSpaceVariationalDiagnostics
    window_plan: StateSpaceWindowPlan
    root_key: Array
    config: BufferedStateSpaceVariationalConfig
    duration_seconds: float = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)

    @property
    def num_draws(self) -> int:
        return int(self.log_model.shape[0])


def fit_buffered_state_space_variational(
    problem: StateSpaceProblem,
    /,
    *,
    key: Array,
    config: BufferedStateSpaceVariationalConfig,
    family: AmortizedGaussianMarkovFamily | None = None,
    num_samples: int = 1000,
) -> BufferedStateSpaceVariationalResult:
    """Fit an inverse-inclusion-weighted buffered path ELBO approximation."""

    if not isinstance(problem, StateSpaceProblem):
        raise TypeError("problem must be StateSpaceProblem.")
    if not isinstance(config, BufferedStateSpaceVariationalConfig):
        raise TypeError("config must be BufferedStateSpaceVariationalConfig.")
    draws = int(num_samples)
    if draws < 1:
        raise ValueError("num_samples must be positive.")
    plan = StateSpaceWindowPlan(
        problem.observations.num_steps,
        target_length=config.target_length,
        left_buffer=config.left_buffer,
        right_buffer=config.right_buffer,
    )
    family_ = (
        AmortizedGaussianMarkovFamily.from_problem(
            problem,
            hidden_size=config.hidden_size,
            scale_floor=config.scale_floor,
            key=jr.fold_in(key, 0xB0FFE2),
        )
        if family is None
        else family
    )
    if not isinstance(family_, AmortizedGaussianMarkovFamily):
        raise TypeError("family must be AmortizedGaussianMarkovFamily or None.")
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.optimization.gradient_clip),
        optax.adam(config.optimization.learning_rate),
    )
    dynamic_family, static_family = eqx.partition(family_, eqx.is_inexact_array)
    optimizer_state = optimizer.init(dynamic_family)
    case_shape = problem.observations.case_shape
    step_valid = problem.observations.step_valid

    def loss_function(current_dynamic, sample_key, window):
        current_family = eqx.combine(current_dynamic, static_family)
        context_mask = (
            jnp.broadcast_to(
                window.context_mask,
                case_shape + (plan.num_steps,),
            )
            & step_valid
        )
        window_family = eqx.tree_at(
            lambda value: value.context_mask,
            current_family,
            context_mask,
        )
        conditional = window_family.conditional_family
        paths, _ = conditional.sample_and_log_prob(
            sample_key,
            sample_shape=(config.optimization.samples_per_step,),
        )
        q_initial, q_transition = conditional.log_prob_terms(paths)
        model = jax.vmap(lambda path: state_space_path_log_density(problem, path))(paths)
        target_weight = (
            window.target_mask.astype(paths.dtype) / window.inclusion_probability
        )
        target_weight = jnp.broadcast_to(
            target_weight,
            case_shape + (plan.num_steps,),
        )
        step_terms = (model.transition + model.observation - q_transition) * target_weight
        initial_weight = target_weight[..., 0]
        initial_terms = (model.prior - q_initial) * initial_weight
        elbo_samples = initial_terms.reshape((paths.shape[0], -1)).sum(axis=-1)
        elbo_samples = elbo_samples + step_terms.reshape((paths.shape[0], -1)).sum(
            axis=-1
        )
        loss = -jnp.mean(elbo_samples)
        finite = jnp.isfinite(loss) & jnp.all(model.valid) & _tree_all_finite(paths)
        return loss, finite

    @eqx.filter_jit
    def update(current_dynamic, current_optimizer_state, sample_key, window):
        (loss, finite), gradient = eqx.filter_value_and_grad(
            loss_function,
            has_aux=True,
        )(current_dynamic, sample_key, window)
        gradient_norm = optax.tree.norm(gradient)
        updates, next_optimizer_state = optimizer.update(
            gradient,
            current_optimizer_state,
            current_dynamic,
        )
        next_dynamic = eqx.apply_updates(current_dynamic, updates)
        finite = finite & jnp.isfinite(gradient_norm) & _tree_all_finite(next_dynamic)
        return next_dynamic, next_optimizer_state, loss, gradient_norm, finite

    recorded_steps = []
    starts = []
    context_starts = []
    context_ends = []
    elbo_history = []
    gradient_history = []
    finite_history = []
    started = perf_counter()
    for step in range(config.optimization.num_steps):
        window_key = jr.fold_in(jr.fold_in(key, 0x710D0), step)
        sample_key = jr.fold_in(jr.fold_in(key, 0x5A4F1E), step)
        window = plan.sample(window_key)
        dynamic_family, optimizer_state, loss, gradient_norm, finite = update(
            dynamic_family,
            optimizer_state,
            sample_key,
            window,
        )
        jax.block_until_ready(loss)
        if not bool(finite):
            raise FloatingPointError(
                f"Buffered variational optimization became nonfinite at step {step + 1}."
            )
        completed = step + 1
        if (
            completed % config.optimization.record_every == 0
            or completed == config.optimization.num_steps
        ):
            recorded_steps.append(completed)
            starts.append(window.target_start)
            context_starts.append(window.context_start)
            context_ends.append(window.context_end)
            elbo_history.append(-loss)
            gradient_history.append(gradient_norm)
            finite_history.append(finite)

    fitted_family = eqx.combine(dynamic_family, static_family)
    states, log_variational = fitted_family.sample_and_log_prob(
        jr.fold_in(key, 0xF17A1),
        sample_shape=(draws,),
    )
    log_model = jax.vmap(
        lambda path: state_space_path_log_density(problem, path).log_density
    )(states)
    jax.block_until_ready(log_model)
    diagnostics = BufferedStateSpaceVariationalDiagnostics(
        steps=jnp.asarray(recorded_steps, dtype=jnp.int32),
        target_start=jnp.asarray(starts, dtype=jnp.int32),
        context_start=jnp.asarray(context_starts, dtype=jnp.int32),
        context_end=jnp.asarray(context_ends, dtype=jnp.int32),
        elbo=jnp.asarray(elbo_history),
        gradient_norm=jnp.asarray(gradient_history),
        finite=jnp.asarray(finite_history, dtype=bool),
    )
    return BufferedStateSpaceVariationalResult(
        problem=problem,
        family=fitted_family,
        states=states,
        log_model=log_model,
        log_variational=log_variational,
        diagnostics=diagnostics,
        window_plan=plan,
        root_key=jnp.asarray(key),
        config=config,
        duration_seconds=perf_counter() - started,
        approximation_id="buffered-amortized-gaussian-markov-path",
    )


__all__ = [
    "BufferedStateSpaceVariationalConfig",
    "BufferedStateSpaceVariationalDiagnostics",
    "BufferedStateSpaceVariationalResult",
    "fit_buffered_state_space_variational",
    "StateSpaceWindowBatch",
    "StateSpaceWindowPlan",
]
