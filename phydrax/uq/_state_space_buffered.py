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

from .._differentiation import ComponentAuthority, DerivativeRoute, ObjectiveKind
from .._fingerprint import canonical_fingerprint
from .._sampling import derive_key, SampleAddress
from .._strict import StrictModule
from .._trainable import combine_parameters
from .._training_kernel import (
    KernelObjective,
    OptaxUpdateRule,
    prepare_training_kernel,
    run_training_attempt,
    TrainingKernelSpec,
    TrainingRejectionBudgetError,
)
from .._training_objective import _ObjectiveContribution
from ..stochastic import StateSpaceProblem
from ._state_space_amortized import AmortizedGaussianMarkovFamily
from ._state_space_path_density import state_space_path_log_density
from ._variational import _tree_all_finite, VariationalConfig


_FAMILY_ADDRESS = SampleAddress(
    "uq.state-space-buffered", "family-initialization", role="initialization"
)
_FINAL_DRAWS_ADDRESS = SampleAddress(
    "uq.state-space-buffered", "final-draws", role="posterior-draws"
)


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
        inclusion = counts.astype("float64") / float(starts)
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
        return self.log_model.shape[0]


class _BufferedPathObjective(StrictModule):
    """Inverse-inclusion-weighted buffered path ELBO; the payload is the problem.

    The target window is accepted-addressed (a retried attempt reuses it) and the
    path draws are attempt-addressed. A nonfinite draw, density, or loss makes
    the numerator nonfinite, so the kernel rolls the attempt back.
    """

    plan: StateSpaceWindowPlan
    samples_per_step: int = eqx.field(static=True)

    def __call__(self, parameters, model_state, fixed, problem, keys):
        plan = self.plan
        window = plan.sample(keys.accepted_key("window"))
        case_shape = problem.observations.case_shape
        family = combine_parameters(parameters, model_state, fixed)
        context_mask = (
            jnp.broadcast_to(window.context_mask, case_shape + (plan.num_steps,))
            & problem.observations.step_valid
        )
        window_family = eqx.tree_at(
            lambda value: value.context_mask, family, context_mask
        )
        conditional = window_family.conditional_family
        paths, _ = conditional.sample_and_log_prob(
            keys.attempt_key("path-sample"),
            sample_shape=(self.samples_per_step,),
        )
        q_initial, q_transition = conditional.log_prob_terms(paths)
        model = jax.vmap(lambda path: state_space_path_log_density(problem, path))(paths)
        target_weight = (
            window.target_mask.astype(paths.dtype) / window.inclusion_probability
        )
        target_weight = jnp.broadcast_to(target_weight, case_shape + (plan.num_steps,))
        step_terms = (model.transition + model.observation - q_transition) * target_weight
        initial_weight = target_weight[..., 0]
        initial_terms = (model.prior - q_initial) * initial_weight
        elbo_samples = initial_terms.reshape((paths.shape[0], -1)).sum(axis=-1)
        elbo_samples = elbo_samples + step_terms.reshape((paths.shape[0], -1)).sum(
            axis=-1
        )
        loss = -jnp.mean(elbo_samples)
        finite = jnp.isfinite(loss) & jnp.all(model.valid) & _tree_all_finite(paths)
        numerator = jnp.where(finite, loss, jnp.full_like(loss, jnp.nan))
        diagnostics = (
            loss,
            window.target_start,
            window.context_start,
            window.context_end,
        )
        return _ObjectiveContribution(numerator, 1.0), model_state, diagnostics


def fit_buffered_state_space_variational(
    problem: StateSpaceProblem,
    /,
    *,
    key: Array,
    config: BufferedStateSpaceVariationalConfig,
    family: AmortizedGaussianMarkovFamily | None = None,
    num_samples: int = 1000,
) -> BufferedStateSpaceVariationalResult:
    """Fit an inverse-inclusion-weighted buffered path ELBO approximation.

    Every step is one attempt of the shared training kernel (`MODEL` root
    authority, one data-fit objective, clipped Adam). A nonfinite step rolls
    back and raises `FloatingPointError`.
    """

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
            key=derive_key(key, _FAMILY_ADDRESS),
        )
        if family is None
        else family.condition(problem)
    )
    if not isinstance(family_, AmortizedGaussianMarkovFamily):
        raise TypeError("family must be AmortizedGaussianMarkovFamily or None.")
    optimization = config.optimization
    kernel = prepare_training_kernel(
        family_,
        (
            KernelObjective(
                objective_id="buffered-path-elbo",
                kind=ObjectiveKind.DATA_FIT,
                route=DerivativeRoute.DIRECT,
                fn=_BufferedPathObjective(plan, optimization.samples_per_step),
            ),
        ),
        TrainingKernelSpec(
            OptaxUpdateRule(
                optax.chain(
                    optax.clip_by_global_norm(optimization.gradient_clip),
                    optax.adam(optimization.learning_rate),
                ),
                rule_id=canonical_fingerprint(
                    {
                        "kind": "buffered-path-elbo-clipped-adam",
                        "gradient_clip": optimization.gradient_clip.hex(),
                        "learning_rate": optimization.learning_rate.hex(),
                    }
                ),
            ),
            context="fit_buffered_state_space_variational",
            rejection_budget=0,
        ),
        root_authority=ComponentAuthority.MODEL,
    )
    state = kernel.init(family_, key)

    recorded_steps = []
    starts = []
    context_starts = []
    context_ends = []
    elbo_history = []
    gradient_history = []
    finite_history = []
    started = perf_counter()
    for step in range(optimization.num_steps):
        try:
            state, evidence = run_training_attempt(kernel, state, problem)
        except TrainingRejectionBudgetError as error:
            raise FloatingPointError(
                f"Buffered variational optimization became nonfinite at step {step + 1}; "
                "the encoder was rolled back to its last accepted state."
            ) from error
        completed = step + 1
        if (
            completed % optimization.record_every == 0
            or completed == optimization.num_steps
        ):
            loss, target_start, context_start, context_end = evidence.diagnostics[0]
            recorded_steps.append(completed)
            starts.append(target_start)
            context_starts.append(context_start)
            context_ends.append(context_end)
            elbo_history.append(-loss)
            gradient_history.append(evidence.gradient_norm)
            finite_history.append(evidence.finite)

    fitted_family = kernel.tree(state)
    states, log_variational = fitted_family.sample_and_log_prob(
        derive_key(key, _FINAL_DRAWS_ADDRESS),
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
        finite=jnp.asarray(finite_history, dtype=jnp.bool_),
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
