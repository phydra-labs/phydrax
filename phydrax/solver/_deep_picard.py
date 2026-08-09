#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from math import prod
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jaxtyping import Array, ArrayLike, Key

from phydrax.domain import DomainFunction

from .._strict import StrictModule
from ..stochastic._bsde import (
    _pointwise_autodiff_control,
    _pointwise_values,
    _predictor_value,
    BSDEPathBatch,
    BSDEProblem,
)
from ..stochastic._feynman_kac import (
    _resolve_queries,
    _terminal_values,
    FeynmanKacLabelBatch,
    FeynmanKacSamplingPlan,
    query_feynman_kac_labels,
    trajectory_node_feynman_kac_labels,
)
from ..terms._feynman_kac import FeynmanKacRegressionTerm
from ._functional_solver import FunctionalSolver


DeepPicardInitialSource: TypeAlias = Literal["zero", "current"]
PicardPredictor: TypeAlias = Callable | DomainFunction


class PicardSourceContext(StrictModule):
    """Matrix-free differential contractions of one frozen Picard value field."""

    value_predictor: PicardPredictor
    problem: BSDEProblem
    key: Array

    def value(self, time: Array, state: Array, /) -> Array:
        value = _predictor_value(
            self.value_predictor,
            jnp.asarray(time),
            jnp.asarray(state),
            self.problem,
            key=self.key,
        )
        if value.shape != self.problem.output_shape:
            raise ValueError("Picard value predictor returned an incompatible shape.")
        return value

    def gradient(self, time: Array, state: Array, /) -> Array:
        state_value = jnp.asarray(state)
        return jax.jacrev(lambda argument: self.value(time, argument))(state_value)

    def control(self, time: Array, state: Array, /) -> Array:
        gradient = self.gradient(time, state).reshape(
            (prod(self.problem.output_shape), prod(self.problem.state_shape))
        )
        diffusion = jnp.asarray(
            self.problem.diffusion(time, state, self.problem.args)
        ).reshape((prod(self.problem.state_shape), prod(self.problem.noise_shape)))
        return (gradient @ diffusion).reshape(
            self.problem.output_shape + self.problem.noise_shape
        )

    def directional_hessian(
        self,
        time: Array,
        state: Array,
        direction: Array,
        /,
    ) -> Array:
        state_value = jnp.asarray(state)
        direction_value = jnp.asarray(direction)
        if state_value.shape != self.problem.state_shape:
            raise ValueError("state must have problem.state_shape.")
        if direction_value.shape != self.problem.state_shape:
            raise ValueError("direction must have problem.state_shape.")
        gradient_fn = lambda argument: self.gradient(time, argument)
        _, hessian_direction = jax.jvp(
            gradient_fn,
            (state_value,),
            (direction_value,),
        )
        contracted = hessian_direction.reshape(
            (prod(self.problem.output_shape), prod(self.problem.state_shape))
        ) @ direction_value.reshape((-1,))
        return contracted.reshape(self.problem.output_shape)

    def covariance_trace(self, time: Array, state: Array, /) -> Array:
        """Return tr(sigma sigmaᵀ Hess u) through factor-HVP contractions."""
        diffusion = jnp.asarray(
            self.problem.diffusion(time, state, self.problem.args)
        )
        expected = self.problem.state_shape + self.problem.noise_shape
        if diffusion.shape != expected:
            raise ValueError(f"diffusion must have shape {expected}.")
        factors = jnp.moveaxis(
            diffusion.reshape(
                (prod(self.problem.state_shape), prod(self.problem.noise_shape))
            ),
            1,
            0,
        ).reshape((prod(self.problem.noise_shape),) + self.problem.state_shape)
        contractions = jax.vmap(
            lambda direction: self.directional_hessian(time, state, direction)
        )(factors)
        return jnp.sum(contractions, axis=0)


class StructuredPicardSource(StrictModule):
    """Explicit fully-nonlinear frozen source without a dense-Hessian interface."""

    evaluator: Callable[[Array, Array, PicardSourceContext, Any], Array]
    source_id: str = eqx.field(static=True)

    def __init__(
        self,
        evaluator: Callable[[Array, Array, PicardSourceContext, Any], Array],
        /,
        *,
        source_id: str,
    ):
        if not callable(evaluator):
            raise TypeError("evaluator must be callable.")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("source_id must be a non-empty string.")
        self.evaluator = evaluator
        self.source_id = source_id

    def __call__(
        self,
        time: Array,
        state: Array,
        context: PicardSourceContext,
        args: Any,
        /,
    ) -> Array:
        return jnp.asarray(self.evaluator(time, state, context, args))


StructuredSourceBuilder: TypeAlias = Callable[
    [PicardSourceContext], StructuredPicardSource
]


class DeepPicardDiagnostics(StrictModule):
    steps: Array
    iterate_rmse: Array
    target_rmse: Array
    relative_target_rmse: Array
    control_target_rmse: Array
    terminal_rmse: Array
    target_variance: Array
    valid_fraction: Array
    contraction_rate: Array
    finite: Array

    @property
    def passed(self) -> bool:
        return bool(jnp.all(self.finite)) and bool(self.steps.shape[0] > 0)


class DeepPicardResult(StrictModule):
    solver: FunctionalSolver
    diagnostics: DeepPicardDiagnostics
    completed_iterations: int = eqx.field(static=True)
    converged: bool = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


def _zero_predictor(problem: BSDEProblem, /) -> Callable[[Array, Array], Array]:
    return lambda time, state: jnp.zeros(problem.output_shape, dtype=jnp.asarray(state).dtype)


def _source_problem(
    problem: BSDEProblem,
    predictor: PicardPredictor,
    source_builder: StructuredSourceBuilder,
    /,
    *,
    key: Key[Array, ""],
) -> BSDEProblem:
    context = PicardSourceContext(predictor, problem, key)
    source = source_builder(context)
    if not isinstance(source, StructuredPicardSource):
        raise TypeError("source_builder must return StructuredPicardSource.")

    def generator(time, state, value, control, args):
        del value, control
        output = source(time, state, context, args)
        if output.shape != problem.output_shape:
            raise ValueError("Structured Picard source returned an incompatible shape.")
        return output

    return BSDEProblem(
        problem.forward_sampler,
        problem.drift,
        problem.diffusion,
        generator,
        problem.terminal,
        state_shape=problem.state_shape,
        noise_shape=problem.noise_shape,
        output_shape=problem.output_shape,
        problem_id=problem.problem_id,
        process_id=problem.process_id,
        args=problem.args,
        time_label=problem.time_label,
        state_label=problem.state_label,
    )


def _iteration_labels(
    problem: BSDEProblem,
    plan: FeynmanKacSamplingPlan,
    /,
    *,
    source_value: PicardPredictor | None,
    source_control: PicardPredictor | None,
    source_builder: StructuredSourceBuilder | None,
    query_times: ArrayLike | None,
    query_states: ArrayLike | None,
    query_weights: ArrayLike | None,
    query_sampler: Callable[[Key[Array, ""]], Any] | None,
    paths: BSDEPathBatch | None,
    key: Key[Array, ""],
) -> FeynmanKacLabelBatch:
    label_problem = problem
    label_value = source_value
    label_control = source_control
    if source_builder is not None:
        predictor = _zero_predictor(problem) if source_value is None else source_value
        label_problem = _source_problem(
            problem,
            predictor,
            source_builder,
            key=jr.fold_in(key, 11),
        )
        label_value = None
        label_control = None
    if plan.sampling_mode == "trajectory_nodes":
        path_batch = problem.sample(jr.fold_in(key, 0)) if paths is None else paths
        return trajectory_node_feynman_kac_labels(
            label_problem,
            path_batch,
            plan,
            source_value=label_value,
            source_control=label_control,
            key=jr.fold_in(key, 1),
        )
    labels = query_feynman_kac_labels(
        label_problem,
        plan,
        query_times=query_times,
        query_states=query_states,
        query_weights=query_weights,
        query_sampler=query_sampler,
        source_value=label_value,
        source_control=label_control,
        key=key,
    )
    if not isinstance(labels, FeynmanKacLabelBatch):
        raise RuntimeError("Internal Feynman-Kac label generation returned paths.")
    return labels


def _model_predictions(
    problem: BSDEProblem,
    solver: FunctionalSolver,
    labels: FeynmanKacLabelBatch,
    /,
    *,
    value_name: str,
    control_name: str | None,
    key: Key[Array, ""],
) -> tuple[Array, Array | None]:
    functions = solver.ansatz_functions()
    if value_name not in functions:
        raise KeyError(f"Missing Picard value function {value_name!r}.")
    value_key, control_key = jr.split(key)
    values = _pointwise_values(
        functions[value_name],
        labels.query_times,
        labels.query_states,
        problem,
        key=value_key,
        output_shape=problem.output_shape,
    )
    if labels.control_targets is None:
        return values, None
    if control_name is None:
        controls = _pointwise_autodiff_control(
            functions[value_name],
            labels.query_times,
            labels.query_states,
            problem,
            key=control_key,
        )
    else:
        if control_name not in functions:
            raise KeyError(f"Missing Picard control function {control_name!r}.")
        controls = _pointwise_values(
            functions[control_name],
            labels.query_times,
            labels.query_states,
            problem,
            key=control_key,
            output_shape=problem.output_shape + problem.noise_shape,
        )
    return values, controls


def _damped_labels(
    problem: BSDEProblem,
    solver: FunctionalSolver,
    labels: FeynmanKacLabelBatch,
    /,
    *,
    value_name: str,
    control_name: str | None,
    damping: float,
    key: Key[Array, ""],
) -> FeynmanKacLabelBatch:
    if damping == 1.0:
        return labels
    values, controls = _model_predictions(
        problem,
        solver,
        labels,
        value_name=value_name,
        control_name=control_name,
        key=key,
    )
    damped_values = (1.0 - damping) * values + damping * labels.value_targets
    out = eqx.tree_at(lambda batch: batch.value_targets, labels, damped_values)
    out = eqx.tree_at(
        lambda batch: batch.value_standard_errors,
        out,
        damping * labels.value_standard_errors,
    )
    if labels.control_targets is not None:
        if controls is None:
            raise RuntimeError("Control targets exist but current controls are unavailable.")
        damped_controls = (
            (1.0 - damping) * controls + damping * labels.control_targets
        )
        out = eqx.tree_at(lambda batch: batch.control_targets, out, damped_controls)
        if labels.control_standard_errors is not None:
            out = eqx.tree_at(
                lambda batch: batch.control_standard_errors,
                out,
                damping * labels.control_standard_errors,
            )
    return out


def _masked_rmse(
    residual: Array,
    valid: Array,
    event_shape: tuple[int, ...],
    /,
) -> Array:
    squared = jnp.abs(residual) ** 2
    axes = tuple(range(squared.ndim - len(event_shape), squared.ndim))
    squared = jnp.sum(squared, axis=axes)
    count = jnp.sum(valid)
    return jnp.sqrt(
        jnp.sum(jnp.where(valid, squared, 0.0)) / jnp.maximum(count, 1)
    )


def _validation_queries(
    problem: BSDEProblem,
    plan: FeynmanKacSamplingPlan,
    /,
    *,
    query_times: ArrayLike | None,
    query_states: ArrayLike | None,
    query_weights: ArrayLike | None,
    query_sampler: Callable[[Key[Array, ""]], Any] | None,
    key: Key[Array, ""],
) -> tuple[Array | None, Array | None, Array | None]:
    if plan.sampling_mode == "trajectory_nodes":
        return None, None, None
    return _resolve_queries(
        problem,
        plan,
        key=key,
        query_times=query_times,
        query_states=query_states,
        query_weights=query_weights,
        query_sampler=query_sampler,
    )


def solve_deep_picard(
    solver: FunctionalSolver,
    problem: BSDEProblem,
    /,
    *,
    value_name: str,
    sampling_plan: FeynmanKacSamplingPlan,
    num_picard_steps: int,
    inner_num_iter: int,
    optim: Any = None,
    control_name: str | None = None,
    query_times: ArrayLike | None = None,
    query_states: ArrayLike | None = None,
    query_weights: ArrayLike | None = None,
    query_sampler: Callable[[Key[Array, ""]], Any] | None = None,
    validation_query_times: ArrayLike | None = None,
    validation_query_states: ArrayLike | None = None,
    validation_query_weights: ArrayLike | None = None,
    validation_query_sampler: Callable[[Key[Array, ""]], Any] | None = None,
    source_builder: StructuredSourceBuilder | None = None,
    initial_source: DeepPicardInitialSource = "zero",
    target_damping: float = 1.0,
    convergence_tolerance: float = 1e-3,
    relative_tolerance: float = 1e-3,
    minimum_picard_steps: int = 1,
    value_weight: ArrayLike = 1.0,
    control_weight: ArrayLike = 0.0,
    interior_weight: ArrayLike = 1.0,
    terminal_weight: ArrayLike = 1.0,
    seed: int = 0,
    jit: bool = True,
    keep_best: bool = True,
    log_every: int = 0,
) -> DeepPicardResult:
    """Train one global value/control field by frozen Feynman--Kac Picard steps."""
    if not isinstance(solver, FunctionalSolver) or not isinstance(problem, BSDEProblem):
        raise TypeError("solver and problem must be FunctionalSolver and BSDEProblem.")
    if not isinstance(sampling_plan, FeynmanKacSamplingPlan):
        raise TypeError("sampling_plan must be a FeynmanKacSamplingPlan.")
    outer_steps = int(num_picard_steps)
    inner_steps = int(inner_num_iter)
    minimum_steps = int(minimum_picard_steps)
    if outer_steps < 1 or inner_steps < 1:
        raise ValueError("Picard and inner iteration counts must be positive.")
    if minimum_steps < 1 or minimum_steps > outer_steps:
        raise ValueError("minimum_picard_steps must lie in [1, num_picard_steps].")
    if initial_source not in ("zero", "current"):
        raise ValueError("initial_source must be 'zero' or 'current'.")
    damping = float(target_damping)
    absolute_tolerance = float(convergence_tolerance)
    relative_tolerance_value = float(relative_tolerance)
    if not 0.0 < damping <= 1.0:
        raise ValueError("target_damping must lie in (0, 1].")
    if absolute_tolerance < 0.0 or relative_tolerance_value < 0.0:
        raise ValueError("Convergence tolerances must be nonnegative.")
    if value_name not in solver.ansatz_functions():
        raise KeyError(f"Missing Picard value function {value_name!r}.")
    if control_name is not None and control_name not in solver.ansatz_functions():
        raise KeyError(f"Missing Picard control function {control_name!r}.")
    if sampling_plan.sampling_mode == "queries" and (
        query_sampler is None and (query_times is None or query_states is None)
    ):
        raise ValueError("Query-mode Picard requires explicit queries or query_sampler.")
    if sampling_plan.sampling_mode == "trajectory_nodes" and any(
        value is not None
        for value in (query_times, query_states, query_weights, query_sampler)
    ):
        raise ValueError("Trajectory-node Picard does not accept query arguments.")
    if source_builder is not None and not callable(source_builder):
        raise TypeError("source_builder must be callable or None.")
    if optim is None:
        optim = optax.adam(1e-3)
    root_key = jr.key(int(seed))
    validation_sampler = (
        validation_query_sampler
        if validation_query_sampler is not None
        else query_sampler
    )
    validation_times_input = (
        validation_query_times
        if validation_query_times is not None
        else query_times
    )
    validation_states_input = (
        validation_query_states
        if validation_query_states is not None
        else query_states
    )
    validation_weights_input = (
        validation_query_weights
        if validation_query_weights is not None
        else query_weights
    )
    validation_times, validation_states, validation_weights = _validation_queries(
        problem,
        sampling_plan,
        query_times=validation_times_input,
        query_states=validation_states_input,
        query_weights=validation_weights_input,
        query_sampler=validation_sampler,
        key=jr.fold_in(root_key, 7001),
    )
    validation_paths = (
        problem.sample(jr.fold_in(root_key, 7002))
        if sampling_plan.sampling_mode == "trajectory_nodes"
        else None
    )
    current = solver
    step_values: list[Array] = []
    iterate_errors: list[Array] = []
    target_errors: list[Array] = []
    relative_errors: list[Array] = []
    control_errors: list[Array] = []
    terminal_errors: list[Array] = []
    target_variances: list[Array] = []
    valid_fractions: list[Array] = []
    contractions: list[Array] = []
    finite_flags: list[Array] = []
    previous_target_error: Array | None = None
    converged = False

    for outer in range(outer_steps):
        old_functions = current.ansatz_functions()
        source_value = (
            None
            if outer == 0 and initial_source == "zero"
            else old_functions[value_name]
        )
        source_control = (
            None
            if source_value is None or control_name is None
            else old_functions[control_name]
        )
        stochastic_key = (
            jr.fold_in(root_key, 100)
            if sampling_plan.refresh_mode == "fixed"
            else jr.fold_in(root_key, 100 + outer)
        )
        training_labels = _iteration_labels(
            problem,
            sampling_plan,
            source_value=source_value,
            source_control=source_control,
            source_builder=source_builder,
            query_times=query_times,
            query_states=query_states,
            query_weights=query_weights,
            query_sampler=query_sampler,
            paths=None,
            key=stochastic_key,
        )
        training_labels = _damped_labels(
            problem,
            current,
            training_labels,
            value_name=value_name,
            control_name=control_name,
            damping=damping,
            key=jr.fold_in(root_key, 200 + outer),
        )
        objective = FeynmanKacRegressionTerm(
            problem,
            sampling_plan,
            value_name=value_name,
            control_name=control_name,
            labels=training_labels,
            value_weight=value_weight,
            control_weight=control_weight,
            interior_weight=interior_weight,
            terminal_weight=terminal_weight,
            label=f"deep-picard-{outer + 1}",
        )
        base_term_count = len(current.terms)
        temporary = current._append_training_terms(
            objective,
            key=jr.fold_in(root_key, 3000 + outer),
        )
        trained = temporary.solve(
            num_iter=inner_steps,
            optim=optim,
            seed=int(seed) + outer + 1,
            jit=jit,
            keep_best=keep_best,
            log_every=log_every,
        )
        current = trained._retain_training_prefix(base_term_count)
        validation_labels = _iteration_labels(
            problem,
            sampling_plan,
            source_value=source_value,
            source_control=source_control,
            source_builder=source_builder,
            query_times=validation_times,
            query_states=validation_states,
            query_weights=validation_weights,
            query_sampler=None,
            paths=validation_paths,
            key=jr.fold_in(root_key, 8000),
        )
        old_values, _ = _model_predictions(
            problem,
            temporary,
            validation_labels,
            value_name=value_name,
            control_name=control_name,
            key=jr.fold_in(root_key, 300 + outer),
        )
        new_values, new_controls = _model_predictions(
            problem,
            current,
            validation_labels,
            value_name=value_name,
            control_name=control_name,
            key=jr.fold_in(root_key, 400 + outer),
        )
        iterate_rmse = _masked_rmse(
            new_values - old_values,
            validation_labels.valid,
            problem.output_shape,
        )
        target_rmse = _masked_rmse(
            new_values - validation_labels.value_targets,
            validation_labels.valid,
            problem.output_shape,
        )
        target_norm = _masked_rmse(
            validation_labels.value_targets,
            validation_labels.valid,
            problem.output_shape,
        )
        relative_rmse = target_rmse / jnp.maximum(target_norm, 1e-12)
        if validation_labels.control_targets is None or new_controls is None:
            control_rmse = jnp.asarray(jnp.nan)
        else:
            control_rmse = _masked_rmse(
                new_controls - validation_labels.control_targets,
                validation_labels.control_valid,
                problem.output_shape + problem.noise_shape,
            )
        terminal_times = jnp.full_like(
            validation_labels.query_times,
            sampling_plan.terminal_time,
        )
        terminal_targets = _terminal_values(problem, validation_labels.query_states)
        terminal_labels = eqx.tree_at(
            lambda batch: (batch.query_times, batch.value_targets),
            validation_labels,
            (terminal_times, terminal_targets),
        )
        terminal_predictions, _ = _model_predictions(
            problem,
            current,
            terminal_labels,
            value_name=value_name,
            control_name=control_name,
            key=jr.fold_in(root_key, 500 + outer),
        )
        terminal_rmse = _masked_rmse(
            terminal_predictions - terminal_targets,
            validation_labels.valid,
            problem.output_shape,
        )
        finite_standard_errors = jnp.where(
            jnp.isfinite(validation_labels.value_standard_errors),
            validation_labels.value_standard_errors,
            0.0,
        )
        target_variance = jnp.mean(
            finite_standard_errors**2 * float(validation_labels.source_path_count)
        )
        contraction = (
            jnp.asarray(jnp.nan)
            if previous_target_error is None
            else target_rmse / jnp.maximum(previous_target_error, 1e-12)
        )
        finite = (
            jnp.isfinite(iterate_rmse)
            & jnp.isfinite(target_rmse)
            & jnp.isfinite(relative_rmse)
            & jnp.isfinite(terminal_rmse)
            & (jnp.isnan(control_rmse) | jnp.isfinite(control_rmse))
        )
        step_values.append(jnp.asarray(outer + 1))
        iterate_errors.append(iterate_rmse)
        target_errors.append(target_rmse)
        relative_errors.append(relative_rmse)
        control_errors.append(control_rmse)
        terminal_errors.append(terminal_rmse)
        target_variances.append(target_variance)
        valid_fractions.append(jnp.mean(validation_labels.valid))
        contractions.append(contraction)
        finite_flags.append(finite)
        previous_target_error = target_rmse
        if (
            outer + 1 >= minimum_steps
            and bool(finite)
            and float(target_rmse) <= absolute_tolerance
            and float(relative_rmse) <= relative_tolerance_value
        ):
            converged = True
            break

    diagnostics = DeepPicardDiagnostics(
        steps=jnp.stack(step_values),
        iterate_rmse=jnp.stack(iterate_errors),
        target_rmse=jnp.stack(target_errors),
        relative_target_rmse=jnp.stack(relative_errors),
        control_target_rmse=jnp.stack(control_errors),
        terminal_rmse=jnp.stack(terminal_errors),
        target_variance=jnp.stack(target_variances),
        valid_fraction=jnp.stack(valid_fractions),
        contraction_rate=jnp.stack(contractions),
        finite=jnp.stack(finite_flags),
    )
    return DeepPicardResult(
        solver=current,
        diagnostics=diagnostics,
        completed_iterations=int(diagnostics.steps.shape[0]),
        converged=converged,
        problem_id=problem.problem_id,
        process_id=problem.process_id,
        plan_id=sampling_plan.plan_id,
    )


__all__ = [
    "DeepPicardDiagnostics",
    "DeepPicardInitialSource",
    "DeepPicardResult",
    "PicardSourceContext",
    "solve_deep_picard",
    "StructuredPicardSource",
    "StructuredSourceBuilder",
]
