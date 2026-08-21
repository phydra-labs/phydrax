#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.flatten_util import ravel_pytree

from .._frozendict import frozendict
from .._trainable import combine_trainable, partition_trainable
from .._training import (
    log_training_signal_stop as _log_training_signal_stop,
    tensorboard_every as _tensorboard_every,
    TensorBoardLogger,
    TrainingSignalGuard as _TrainingSignalGuard,
)
from ..optim._kfac._blocks import (
    solve_block_direction,
    update_block_state_from_observations,
)
from ..optim._kfac._config import KFAC
from ..optim._kfac._types import KFACMetrics, KFACState
from ..optim._iterative._globalization import (
    ArmijoLineSearch,
    armijo_backtracking,
)
from ._functional_objective import (
    evaluate_prepared_objective,
    prepared_data_metrics,
)
from ._functional_reporting import (
    log_tensorboard_scalars as _log_tensorboard_scalars,
    metric_suffix as _metric_suffix,
    term_label as _term_label,
)
from ._functional_run import (
    expand_train_terms as _expanded_train_terms,
    replace_solver_state,
    select_train_terms as _active_train_terms,
    validate_term_sample_size as _train_term_sample_size,
)
from ._kfac_layout import build_kfac_plan
from ._kfac_problem import (
    frozen_loss,
    frozen_loss_and_flat_gradient,
    materialize_frozen_terms,
    term_block_curvature_observations,
    validate_derivative_coverage,
)
from ._model_losses import function_model_loss_labels


def _armijo_search(
    flat_parameters,
    direction,
    gradient,
    initial_loss,
    loss_function,
    /,
    *,
    learning_rate: float,
    shrink: float,
    c1: float,
    max_steps: int,
) -> tuple[Any, Any, Any, Any]:
    directional_derivative = jnp.vdot(gradient, direction).real
    if float(jnp.linalg.norm(direction)) == 0.0:
        return flat_parameters, initial_loss, jnp.asarray(0.0), jnp.asarray(0)
    if (
        not bool(jnp.isfinite(directional_derivative))
        or float(directional_derivative) <= 0.0
    ):
        raise ValueError("KFAC produced a non-descent or nonfinite update direction.")

    minimum_rate = float(learning_rate) * float(shrink) ** (int(max_steps) - 1)
    if minimum_rate == 0.0:
        minimum_rate = float(learning_rate)
    policy = ArmijoLineSearch(
        initial_rate=learning_rate,
        contraction=shrink,
        sufficient_decrease=c1,
        maximum_steps=max_steps,
        minimum_rate=minimum_rate,
    )
    descent = -direction
    result = armijo_backtracking(
        loss_function,
        flat_parameters,
        initial_loss,
        descent,
        -directional_derivative,
        step=lambda base, tangent, rate: base + rate * tangent,
        contains=lambda candidate: jnp.all(jnp.isfinite(candidate)),
        policy=policy,
    )
    if not bool(result.finite_candidate_seen):
        raise FloatingPointError("Every KFAC line-search candidate was nonfinite.")
    return result.parameters, result.value, result.rate, result.evaluations


def _fixed_step(flat_parameters, direction, loss_function, /, *, learning_rate: float):
    candidate = flat_parameters - float(learning_rate) * direction
    candidate_loss = loss_function(candidate)
    if not bool(jnp.isfinite(candidate_loss)):
        raise FloatingPointError("KFAC fixed-step update produced a nonfinite loss.")
    return candidate, candidate_loss, jnp.asarray(float(learning_rate)), jnp.asarray(0)


def _quadratic_norm_and_clip(direction, gradient, /, *, maximum: float | None):
    quadratic_norm = jnp.sqrt(jnp.maximum(jnp.vdot(gradient, direction).real, 0.0))
    if maximum is not None:
        ratio = float(maximum) / jnp.maximum(quadratic_norm, 1e-30)
        scale = jnp.minimum(1.0, ratio)
        direction = scale * direction
        quadratic_norm = scale * quadratic_norm
    return direction, quadratic_norm


def _regularized_psd_condition(matrix, /, *, damping: float):
    eigenvalues = jnp.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    regularizer = jnp.sqrt(float(damping))
    return (jnp.max(eigenvalues) + regularizer) / (
        jnp.maximum(jnp.min(eigenvalues), 0.0) + regularizer
    )


def _factor_condition_estimate(curvature, /, *, damping: float):
    maximum = jnp.asarray(1.0)
    for block_terms in curvature.affine:
        for factor in block_terms:
            condition = _regularized_psd_condition(
                factor.activation,
                damping=damping,
            ) * _regularized_psd_condition(
                factor.sensitivity,
                damping=damping,
            )
            maximum = jnp.maximum(maximum, condition)
    for factor in curvature.uncovered:
        if factor.value.ndim == 2:
            condition = _regularized_psd_condition(
                factor.value,
                damping=damping**2,
            )
        else:
            diagonal = factor.value + float(damping)
            condition = jnp.max(diagonal) / jnp.min(diagonal)
        maximum = jnp.maximum(maximum, condition)
    return maximum


def solve_kfac(
    self,
    *,
    num_iter: int,
    optim: KFAC,
    evaluation_parameters,
    seed: int,
    jit: bool,
    keep_best: bool,
    log_every: int,
    log_terms: bool,
    log_path: str | Path | None,
    tensorboard_log_dir: str | Path | None,
    tensorboard_every: int | None,
    tensorboard_flush_every: int,
    profile_adaptive: bool,
    train_term_sample_size: int | None,
):
    """Run Phydrax-native KFAC over frozen residual terms."""

    if int(num_iter) < 0:
        raise ValueError("num_iter must be nonnegative.")
    if int(num_iter) == 0:
        return self
    if evaluation_parameters is not None:
        raise ValueError("evaluation_parameters is not supported by KFAC.")
    if int(log_every) < 0:
        raise ValueError("log_every must be >= 0.")
    if int(tensorboard_flush_every) <= 0:
        raise ValueError("tensorboard_flush_every must be positive.")
    coverage_functions = (
        self.functions
        if self.enforcement is None
        else self.enforcement.apply(self.functions)
    )
    validate_derivative_coverage(self.terms, coverage_functions)
    model_loss_labels = function_model_loss_labels(self.functions)
    if model_loss_labels:
        raise ValueError(
            "KFAC does not support attached model losses because they do not provide "
            f"residual roots; found {', '.join(model_loss_labels)}."
        )

    params, non_trainable = partition_trainable(self.functions)
    plan = build_kfac_plan(
        optim,
        self.functions,
        params,
        num_terms=len(self.terms),
    )
    state = plan.initialize(params)
    term_sample_size = _train_term_sample_size(
        train_term_sample_size,
        num_terms=len(self.terms),
    )
    term_names = tuple(_term_label(term) for term in self.terms)
    evaluation_term_names = tuple(_term_label(term) for term in self.evaluation_terms)
    root_key = jr.key(int(seed))
    objective = self.objective
    best_loss = float("inf")
    best_params = params
    completed = 0
    refresh_wall_time = 0.0
    optimizer_wall_time = 0.0
    gradient_wall_time = 0.0
    factor_wall_time = 0.0
    linear_solve_wall_time = 0.0
    line_search_wall_time = 0.0
    first_optimizer_step_wall_time = 0.0
    steady_optimizer_step_wall_time = 0.0
    last_metrics = KFACMetrics(
        factor_updates=jnp.asarray(0, dtype=jnp.int32),
        cg_iterations_max=jnp.asarray(0, dtype=jnp.int32),
        cg_relative_residual_max=jnp.asarray(0.0),
        quadratic_update_norm=jnp.asarray(0.0),
        accepted_step_size=jnp.asarray(0.0),
        line_search_steps=jnp.asarray(0, dtype=jnp.int32),
    )

    log_context = (
        open(Path(log_path), "w", encoding="utf-8")
        if log_path is not None
        else nullcontext(None)
    )
    tensorboard_context = (
        TensorBoardLogger(tensorboard_log_dir)
        if tensorboard_log_dir is not None
        else nullcontext(None)
    )
    tensorboard_period = _tensorboard_every(
        tensorboard_log_dir=tensorboard_log_dir,
        tensorboard_every=tensorboard_every,
        log_every=int(log_every),
    )

    with (
        log_context as log_file,
        tensorboard_context as tensorboard_writer,
        _TrainingSignalGuard() as signal_guard,
    ):
        for epoch in range(int(num_iter)):
            if signal_guard.stop_requested:
                _log_training_signal_stop(
                    "kfac",
                    signal_guard,
                    completed=epoch,
                    total=int(num_iter),
                    file=log_file,
                )
                break
            iteration_started = time.perf_counter()
            iteration = epoch + 1
            term_iteration = jnp.asarray(iteration, dtype=float)
            iteration_key = jr.fold_in(root_key, epoch)
            functions_snapshot = combine_trainable(params, non_trainable)
            refresh_started = time.perf_counter()
            objective = objective.refresh(
                functions_snapshot,
                key=jr.fold_in(iteration_key, 101),
                iter_=term_iteration,
            )
            if profile_adaptive:
                jax.block_until_ready(objective)
                refresh_wall_time += time.perf_counter() - refresh_started

            if epoch == 0:
                active_indices = tuple(range(len(self.terms)))
                term_scale = jnp.asarray(1.0)
            else:
                _, active_indices, term_scale = _active_train_terms(
                    self.terms,
                    sample_size=term_sample_size,
                    key=jr.fold_in(iteration_key, 17),
                )
            prepared = objective.prepare_training(
                active_indices,
                scale=term_scale,
                evaluation_key=jr.fold_in(iteration_key, 31),
                sampling_key=jr.fold_in(iteration_key, 31),
                iteration=term_iteration,
            )
            frozen_terms = materialize_frozen_terms(prepared)
            optimizer_started = time.perf_counter()
            gradient_started = time.perf_counter()
            loss, gradient, unravel = frozen_loss_and_flat_gradient(
                params,
                non_trainable,
                self,
                frozen_terms,
                iter_=term_iteration,
            )
            if profile_adaptive:
                jax.block_until_ready((loss, gradient))
                gradient_wall_time += time.perf_counter() - gradient_started
            if not bool(jnp.isfinite(loss)) or not bool(jnp.all(jnp.isfinite(gradient))):
                raise FloatingPointError("KFAC encountered a nonfinite loss or gradient.")
            if keep_best and float(loss) < best_loss:
                best_loss = float(loss)
                best_params = params
            refresh_factors = epoch % optim.factor_update_period == 0
            curvature = state.curvature
            factor_updates = state.factor_updates
            factor_started = time.perf_counter()
            if refresh_factors:
                factor_terms = tuple(replace(term, scale=1.0) for term in frozen_terms)
                flat_parameters, observations = term_block_curvature_observations(
                    params,
                    non_trainable,
                    self,
                    factor_terms,
                    plan.layout,
                    approximation=optim.approximation,
                    chunk_size=optim.factor_chunk_size,
                    iter_=term_iteration,
                )
                curvature = update_block_state_from_observations(
                    curvature,
                    observations,
                    factor_decay=optim.factor_decay,
                    term_indices=active_indices,
                )
                factor_updates = factor_updates + 1
            else:
                flat_parameters, _ = ravel_pytree(params)
            if profile_adaptive:
                jax.block_until_ready((flat_parameters, curvature))
                factor_wall_time += time.perf_counter() - factor_started

            linear_solve_started = time.perf_counter()
            direction, cg_iterations, cg_relative_residual = solve_block_direction(
                curvature,
                plan.layout,
                gradient,
                damping=optim.damping,
                cg_max_steps=optim.cg_max_steps,
                cg_relative_tolerance=optim.cg_relative_tolerance,
            )
            if not bool(jnp.all(jnp.isfinite(direction))):
                raise FloatingPointError("KFAC produced a nonfinite update direction.")
            direction, quadratic_norm = _quadratic_norm_and_clip(
                direction,
                gradient,
                maximum=optim.max_update_norm,
            )
            if profile_adaptive:
                jax.block_until_ready(
                    (direction, cg_iterations, cg_relative_residual, quadratic_norm)
                )
                linear_solve_wall_time += time.perf_counter() - linear_solve_started

            def candidate_loss(flat_candidate):
                return frozen_loss(
                    unravel(flat_candidate),
                    non_trainable,
                    self,
                    frozen_terms,
                    iter_=term_iteration,
                )

            line_search_started = time.perf_counter()
            if optim.line_search:
                new_flat, accepted_loss, step_size, line_search_steps = _armijo_search(
                    flat_parameters,
                    direction,
                    gradient,
                    loss,
                    candidate_loss,
                    learning_rate=optim.learning_rate,
                    shrink=optim.line_search_shrink,
                    c1=optim.line_search_c1,
                    max_steps=optim.line_search_max_steps,
                )
            else:
                new_flat, accepted_loss, step_size, line_search_steps = _fixed_step(
                    flat_parameters,
                    direction,
                    candidate_loss,
                    learning_rate=optim.learning_rate,
                )
            if profile_adaptive:
                jax.block_until_ready(
                    (new_flat, accepted_loss, step_size, line_search_steps)
                )
                line_search_wall_time += time.perf_counter() - line_search_started
            params = unravel(new_flat)
            state = KFACState(
                step=state.step + 1,
                curvature=curvature,
                factor_updates=factor_updates,
            )
            objective = objective.record_training_evaluations(term_indices=active_indices)
            completed = iteration
            if profile_adaptive:
                jax.block_until_ready((params, accepted_loss, state))
                optimizer_step_wall_time = time.perf_counter() - optimizer_started
                optimizer_wall_time += optimizer_step_wall_time
                if epoch == 0:
                    first_optimizer_step_wall_time = optimizer_step_wall_time
                else:
                    steady_optimizer_step_wall_time += optimizer_step_wall_time

            accepted_loss_float = float(accepted_loss)
            if keep_best and accepted_loss_float < best_loss:
                best_loss = accepted_loss_float
                best_params = params
            elif not keep_best:
                best_loss = accepted_loss_float
                best_params = params
            last_metrics = KFACMetrics(
                factor_updates=state.factor_updates,
                cg_iterations_max=cg_iterations,
                cg_relative_residual_max=cg_relative_residual,
                quadratic_update_norm=quadratic_norm,
                accepted_step_size=step_size,
                line_search_steps=line_search_steps,
            )

            console_step = int(log_every) > 0 and iteration % int(log_every) == 0
            tensorboard_step = (
                tensorboard_writer is not None
                and tensorboard_period is not None
                and iteration % int(tensorboard_period) == 0
            )
            elapsed = time.perf_counter() - iteration_started
            train_terms = jnp.zeros((len(self.terms),), dtype=float)
            train_data_metrics = tuple({} for _ in self.terms)
            eval_terms = jnp.zeros((len(self.evaluation_terms),), dtype=float)
            eval_data_metrics = tuple({} for _ in self.evaluation_terms)
            if log_terms and (console_step or tensorboard_step):
                evaluation_functions = combine_trainable(params, non_trainable)
                active_values = evaluate_prepared_objective(
                    prepared,
                    evaluation_functions,
                    include_model_losses=False,
                ).term_values
                train_terms = _expanded_train_terms(
                    active_values,
                    active_term_indices=active_indices,
                    num_terms=len(self.terms),
                )
                active_metrics = prepared_data_metrics(
                    prepared,
                    evaluation_functions,
                )
                expanded_metrics = [{} for _ in self.terms]
                for term_index, metrics in zip(
                    active_indices,
                    active_metrics,
                    strict=True,
                ):
                    expanded_metrics[term_index] = metrics
                collocation_metrics = objective.collocation_data_metrics()
                train_data_metrics = tuple(
                    data_metrics | adaptive_metrics
                    for data_metrics, adaptive_metrics in zip(
                        expanded_metrics,
                        collocation_metrics,
                        strict=True,
                    )
                )
                prepared_evaluation = objective.prepare_evaluation(
                    key=jr.fold_in(iteration_key, 402),
                    iteration=term_iteration,
                )
                eval_terms = evaluate_prepared_objective(
                    prepared_evaluation,
                    evaluation_functions,
                    include_model_losses=False,
                ).term_values
                eval_data_metrics = prepared_data_metrics(
                    prepared_evaluation,
                    evaluation_functions,
                )
            if console_step:
                print(
                    f"step={iteration} loss={accepted_loss_float:.6e} "
                    f"best={best_loss:.6e} time={elapsed:.3f}s",
                    file=log_file,
                )
                if log_terms:
                    for term_index, (name, value) in enumerate(
                        zip(term_names, train_terms, strict=True)
                    ):
                        suffix = _metric_suffix(train_data_metrics[term_index])
                        print(
                            f"  [train {term_index}] {name}: {float(value):.6e}{suffix}",
                            file=log_file,
                        )
                    for term_index, (name, value) in enumerate(
                        zip(evaluation_term_names, eval_terms, strict=True)
                    ):
                        suffix = _metric_suffix(eval_data_metrics[term_index])
                        print(
                            f"  [eval {term_index}] {name}: {float(value):.6e}{suffix}",
                            file=log_file,
                        )
            if tensorboard_step and tensorboard_writer is not None:
                _log_tensorboard_scalars(
                    tensorboard_writer,
                    step=iteration,
                    loss=accepted_loss,
                    best_loss=best_loss,
                    evaluation_loss=None,
                    iter_time_s=elapsed,
                    train_term_names=term_names,
                    train_terms=train_terms,
                    train_data_metrics=train_data_metrics,
                    train_model_loss_names=(),
                    train_model_loss_terms=jnp.zeros((0,), dtype=float),
                    evaluation_term_names=evaluation_term_names,
                    eval_terms=eval_terms,
                    eval_data_metrics=eval_data_metrics,
                    log_terms=log_terms,
                )
                tensorboard_writer.scalar(
                    "optimizer/kfac/step_size", step_size, iteration
                )
                tensorboard_writer.scalar(
                    "optimizer/kfac/factor_updates", state.factor_updates, iteration
                )
                tensorboard_writer.scalar(
                    "optimizer/kfac/cg_iterations_max", cg_iterations, iteration
                )
                tensorboard_writer.scalar(
                    "optimizer/kfac/cg_relative_residual_max",
                    cg_relative_residual,
                    iteration,
                )
                tensorboard_writer.scalar(
                    "optimizer/kfac/quadratic_update_norm",
                    quadratic_norm,
                    iteration,
                )
                tensorboard_writer.scalar(
                    "optimizer/kfac/line_search_steps",
                    line_search_steps,
                    iteration,
                )
                tensorboard_writer.scalar(
                    "optimizer/kfac/factor_condition_estimate_max",
                    _factor_condition_estimate(
                        state.curvature,
                        damping=optim.damping,
                    ),
                    iteration,
                )
                tensorboard_writer.scalar(
                    "optimizer/kfac/damping",
                    optim.damping,
                    iteration,
                )
                if profile_adaptive:
                    tensorboard_writer.scalar(
                        "optimizer/kfac/gradient_wall_time_seconds",
                        gradient_wall_time,
                        iteration,
                    )
                    tensorboard_writer.scalar(
                        "optimizer/kfac/factor_wall_time_seconds",
                        factor_wall_time,
                        iteration,
                    )
                    tensorboard_writer.scalar(
                        "optimizer/kfac/linear_solve_wall_time_seconds",
                        linear_solve_wall_time,
                        iteration,
                    )
                    tensorboard_writer.scalar(
                        "optimizer/kfac/line_search_wall_time_seconds",
                        line_search_wall_time,
                        iteration,
                    )
                if iteration % int(tensorboard_flush_every) == 0:
                    tensorboard_writer.flush()

    chosen = best_params if keep_best else params
    functions = combine_trainable(chosen, non_trainable)
    settle_started = time.perf_counter()
    objective = objective.settle(
        functions,
        key=jr.fold_in(root_key, 991),
        iter_=completed + 1,
    )
    if profile_adaptive:
        jax.block_until_ready(objective)
        refresh_wall_time += time.perf_counter() - settle_started
    result = replace_solver_state(
        self,
        functions=functions,
        objective=objective,
    )
    diagnostics = frozendict(
        {
            "profile_enabled": jnp.asarray(profile_adaptive),
            "refresh_wall_time_seconds": jnp.asarray(refresh_wall_time),
            "optimizer_wall_time_seconds": jnp.asarray(optimizer_wall_time),
            "optimizer/kfac/gradient_wall_time_seconds": jnp.asarray(gradient_wall_time),
            "optimizer/kfac/factor_wall_time_seconds": jnp.asarray(factor_wall_time),
            "optimizer/kfac/linear_solve_wall_time_seconds": jnp.asarray(
                linear_solve_wall_time
            ),
            "optimizer/kfac/line_search_wall_time_seconds": jnp.asarray(
                line_search_wall_time
            ),
            "optimizer/kfac/first_step_wall_time_seconds": jnp.asarray(
                first_optimizer_step_wall_time
            ),
            "optimizer/kfac/steady_step_wall_time_seconds": jnp.asarray(
                steady_optimizer_step_wall_time / max(completed - 1, 1)
            ),
            "optimizer/kfac/step_size": last_metrics.accepted_step_size,
            "optimizer/kfac/factor_updates": last_metrics.factor_updates,
            "optimizer/kfac/cg_iterations_max": last_metrics.cg_iterations_max,
            "optimizer/kfac/cg_relative_residual_max": (
                last_metrics.cg_relative_residual_max
            ),
            "optimizer/kfac/quadratic_update_norm": (last_metrics.quadratic_update_norm),
            "optimizer/kfac/factor_condition_estimate_max": (
                _factor_condition_estimate(state.curvature, damping=optim.damping)
            ),
            "optimizer/kfac/damping": jnp.asarray(optim.damping),
            "optimizer/kfac/line_search_steps": last_metrics.line_search_steps,
            "optimizer/kfac/num_parameters": jnp.asarray(plan.layout.parameter_count),
            "optimizer/kfac/num_affine_blocks": jnp.asarray(
                len(plan.layout.affine_blocks)
            ),
            "optimizer/kfac/factor_chunk_size": jnp.asarray(optim.factor_chunk_size),
            "optimizer/kfac/jit_requested": jnp.asarray(bool(jit)),
        }
    )
    return eqx.tree_at(lambda solver: solver.training_diagnostics, result, diagnostics)


__all__ = ["solve_kfac"]
