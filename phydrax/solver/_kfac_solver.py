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
    TrainingProgress,
    TrainingSignalGuard as _TrainingSignalGuard,
    update_training_selection,
)
from ..optim._iterative._globalization import (
    armijo_backtracking,
    ArmijoLineSearch,
)
from ..optim._kfac._blocks import (
    solve_block_direction,
    update_block_state_from_observations,
)
from ..optim._kfac._config import KFAC
from ..optim._kfac._types import KFACMetrics, KFACState
from ._functional_checkpoint import (
    load_functional_training_checkpoint,
    save_functional_training_checkpoint,
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
from ._functional_residual import materialize_prepared_residual_terms
from ._functional_run import (
    expand_train_terms as _expanded_train_terms,
    replace_solver_state,
    select_train_terms as _active_train_terms,
    validate_term_sample_size as _train_term_sample_size,
)
from ._functional_surrogate import (
    _functional_ntk_diagnostic_values,
    _functional_ntk_diagnostics,
    prepare_functional_update,
)
from ._functional_training import FunctionalTrainingPlan, FunctionalTrainingState
from ._kfac_layout import build_kfac_plan
from ._kfac_problem import (
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
    direction_norm = jnp.linalg.norm(direction)
    zero_direction = direction_norm == 0.0
    directional_derivative = eqx.error_if(
        directional_derivative,
        (~zero_direction)
        & (~jnp.isfinite(directional_derivative) | (directional_derivative <= 0.0)),
        "KFAC produced a non-descent or nonfinite update direction.",
    )
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

    def zero_step(_):
        return (
            flat_parameters,
            initial_loss,
            jnp.asarray(
                0.0,
                dtype=jnp.result_type(initial_loss, directional_derivative, float),
            ),
            jnp.asarray(0, dtype=jnp.int32),
        )

    def search_step(_):
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
        parameters = eqx.error_if(
            result.parameters,
            ~result.finite_candidate_seen,
            "Every KFAC line-search candidate was nonfinite.",
        )
        return parameters, result.value, result.rate, result.evaluations

    return jax.lax.cond(zero_direction, zero_step, search_step, operand=None)


def _fixed_step(flat_parameters, direction, loss_function, /, *, learning_rate: float):
    candidate = flat_parameters - float(learning_rate) * direction
    candidate_loss = loss_function(candidate)
    candidate = eqx.error_if(
        candidate,
        ~jnp.isfinite(candidate_loss),
        "KFAC fixed-step update produced a nonfinite loss.",
    )
    return (
        candidate,
        candidate_loss,
        jnp.asarray(float(learning_rate), dtype=candidate_loss.dtype),
        jnp.asarray(0, dtype=jnp.int32),
    )


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
    training: FunctionalTrainingPlan | None = None,
    resume: bool = False,
):
    """Run Phydrax-native KFAC over frozen residual terms."""

    if int(num_iter) < 0:
        raise ValueError("num_iter must be non-negative.")
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

    resume_state = self.training_state if resume else None
    source_functions = (
        self.functions if resume_state is None else resume_state.current_functions
    )
    params, non_trainable = partition_trainable(source_functions)
    sharding_policy = None if training is None else training.sharding
    if sharding_policy is not None:
        params = sharding_policy.place_parameters(params)
        non_trainable = sharding_policy.place_tree(non_trainable)
    plan = build_kfac_plan(
        optim,
        source_functions,
        params,
        num_terms=len(self.terms),
    )
    state = (
        plan.initialize(params) if resume_state is None else resume_state.optimizer_state
    )
    term_sample_size = _train_term_sample_size(
        train_term_sample_size,
        num_terms=len(self.terms),
    )
    term_names = tuple(_term_label(term) for term in self.terms)
    evaluation_term_names = tuple(_term_label(term) for term in self.evaluation_terms)
    root_key = jr.key(int(seed)) if resume_state is None else resume_state.key
    objective = self.objective
    if (
        resume
        and resume_state is None
        and training is not None
        and training.checkpoint is not None
    ):
        state_template = FunctionalTrainingState(
            current_functions=source_functions,
            best_functions=source_functions,
            previous_functions=(source_functions if training.pseudo_transient else None),
            optimizer_state=state,
            key=root_key,
            pseudo_inverse_steps=tuple(
                policy.initial_inverse_step for policy in training.pseudo_transient
            ),
            term_multipliers=jnp.ones(
                (
                    0
                    if training.term_balance is None
                    else len(training.term_balance.blocks)
                ),
                dtype=float,
            ),
            progress=TrainingProgress(),
            run_id=training.plan_id,
        )
        restored = load_functional_training_checkpoint(
            training.checkpoint.path,
            self,
            state_template,
            training,
        )
        resume_state = restored.state
        source_functions = resume_state.current_functions
        params, non_trainable = partition_trainable(source_functions)
        if sharding_policy is not None:
            params = sharding_policy.place_parameters(params)
            non_trainable = sharding_policy.place_tree(non_trainable)
        plan = build_kfac_plan(
            optim,
            source_functions,
            params,
            num_terms=len(self.terms),
        )
        state = resume_state.optimizer_state
        if sharding_policy is not None:
            state = sharding_policy.place_tree(state)
        root_key = resume_state.key
        objective = restored.objective
    selection_policy = None if training is None else training.selection
    selection_progress = (
        TrainingProgress() if resume_state is None else resume_state.progress
    )
    best_loss = (
        float("inf")
        if selection_progress.best_value is None
        else float(selection_progress.best_value)
    )
    best_params = (
        params
        if resume_state is None
        else partition_trainable(resume_state.best_functions)[0]
    )
    if keep_best and selection_policy is not None and resume_state is None:
        initial_selection = objective.prepare_evaluation(
            key=jr.fold_in(root_key, 1200),
            iteration=jnp.asarray(0.0),
        )
        initial_functions = combine_trainable(params, non_trainable)
        initial_value = evaluate_prepared_objective(
            initial_selection, initial_functions
        ).total
        selection_progress, _ = update_training_selection(
            selection_progress,
            float(initial_value),
            step=0,
            mode=selection_policy.mode,
            min_delta=selection_policy.min_delta,
            patience=selection_policy.patience,
        )
        best_loss = float(initial_value)
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
    training_started = time.perf_counter()
    latest_ntk_diagnostics = None
    update = None
    start_step = 0 if resume_state is None else resume_state.progress.update_step
    if start_step >= int(num_iter):
        resumed_result = replace_solver_state(
            self,
            functions=resume_state.best_functions,
            objective=objective,
        )
        return eqx.tree_at(
            lambda solver: solver.training_state,
            resumed_result,
            resume_state,
            is_leaf=lambda value: value is None,
        )
    previous_functions = (
        None
        if training is None or not training.pseudo_transient
        else source_functions
        if resume_state is None or resume_state.previous_functions is None
        else resume_state.previous_functions
    )
    pseudo_inverse_steps = (
        ()
        if training is None
        else resume_state.pseudo_inverse_steps
        if resume_state is not None
        else tuple(policy.initial_inverse_step for policy in training.pseudo_transient)
    )
    term_multipliers = (
        jnp.zeros((0,), dtype=float)
        if training is None or training.term_balance is None
        else resume_state.term_multipliers
        if resume_state is not None
        else jnp.ones((len(training.term_balance.blocks),), dtype=float)
    )
    previous_gradient = None if resume_state is None else resume_state.previous_gradient

    def make_training_state(current_params, selected_params):
        if training is None:
            raise RuntimeError("Functional training state requires a training plan.")
        current_functions_ = combine_trainable(current_params, non_trainable)
        selected_functions_ = combine_trainable(selected_params, non_trainable)
        return FunctionalTrainingState(
            current_functions=current_functions_,
            best_functions=selected_functions_,
            previous_functions=previous_functions,
            optimizer_state=state,
            key=root_key,
            pseudo_inverse_steps=pseudo_inverse_steps,
            term_multipliers=term_multipliers,
            previous_gradient=previous_gradient,
            progress=selection_progress,
            run_id=training.plan_id,
            training_seconds=(
                (0.0 if resume_state is None else resume_state.training_seconds)
                + time.perf_counter()
                - training_started
            ),
            resumed_from_step=start_step,
        )

    log_context = (
        open(Path(log_path), "w", encoding="utf-8")
        if log_path is not None
        else nullcontext(None)
    )

    def publish_checkpoint(checkpoint_solver, checkpoint_state):
        if training is None or training.checkpoint is None:
            return
        if sharding_policy is not None:
            sharding_policy.synchronize(
                f"functional-kfac-checkpoint-before-{checkpoint_state.progress.update_step}"
            )
        if sharding_policy is None or sharding_policy.is_primary_process:
            save_functional_training_checkpoint(
                training.checkpoint.path,
                checkpoint_solver,
                checkpoint_state,
                training,
            )
        if sharding_policy is not None:
            sharding_policy.synchronize(
                f"functional-kfac-checkpoint-after-{checkpoint_state.progress.update_step}"
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
        for epoch in range(start_step, int(num_iter)):
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
            if sharding_policy is not None:
                prepared = sharding_policy.place_prepared(prepared)
            update = (
                None
                if training is None
                else prepare_functional_update(
                    prepared,
                    params,
                    non_trainable,
                    self.enforcement,
                    training=training,
                    previous_functions=previous_functions,
                    pseudo_inverse_steps=pseudo_inverse_steps,
                    term_multipliers=term_multipliers,
                    previous_gradient=previous_gradient,
                )
            )
            if update is not None:
                pseudo_inverse_steps = update.pseudo_inverse_steps
                term_multipliers = update.term_multipliers
                if update.diagnostic_gradient is not None:
                    previous_gradient = update.diagnostic_gradient
                if (
                    training is not None
                    and training.diagnostics is not None
                    and training.diagnostics.ntk
                    and training.diagnostics.due(iteration)
                ):
                    if update.residual is None:
                        raise ValueError("NTK diagnostics require residual roots.")
                    latest_ntk_diagnostics = _functional_ntk_diagnostics(
                        update.residual,
                        params,
                        training.diagnostics,
                        jr.fold_in(iteration_key, 1701),
                    )
            residual_terms = (
                materialize_prepared_residual_terms(prepared, require_all=True)
                if update is None
                else update.residual.terms
            )
            optimizer_started = time.perf_counter()
            gradient_started = time.perf_counter()
            flat_parameters, unravel = ravel_pytree(params)
            if update is None:

                def physical_loss(flat, _prepared=prepared, _unravel=unravel):
                    functions = combine_trainable(_unravel(flat), non_trainable)
                    return evaluate_prepared_objective(
                        _prepared,
                        functions,
                    ).total

                loss, gradient = jax.value_and_grad(physical_loss)(flat_parameters)
            else:

                def surrogate_loss(flat, _update=update, _unravel=unravel):
                    return _update.surrogate_loss(_unravel(flat), non_trainable)

                loss, gradient = jax.value_and_grad(surrogate_loss)(flat_parameters)
            if profile_adaptive:
                jax.block_until_ready((loss, gradient))
                gradient_wall_time += time.perf_counter() - gradient_started
            if not bool(jnp.isfinite(loss)) or not bool(jnp.all(jnp.isfinite(gradient))):
                raise FloatingPointError("KFAC encountered a nonfinite loss or gradient.")
            if keep_best and selection_policy is None and float(loss) < best_loss:
                best_loss = float(loss)
                best_params = params
            refresh_factors = epoch % optim.factor_update_period == 0
            curvature = state.curvature
            factor_updates = state.factor_updates
            factor_started = time.perf_counter()
            if refresh_factors:
                flat_parameters, observations = term_block_curvature_observations(
                    params,
                    non_trainable,
                    self,
                    residual_terms,
                    plan.layout,
                    approximation=optim.approximation,
                    chunk_size=optim.factor_chunk_size,
                    iter_=term_iteration,
                    functional_residual=(None if update is None else update.residual),
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

            def candidate_loss(
                flat_candidate,
                _update=update,
                _unravel=unravel,
                _prepared=prepared,
            ):
                if _update is not None:
                    return _update.surrogate_loss(_unravel(flat_candidate), non_trainable)
                functions = combine_trainable(
                    _unravel(flat_candidate),
                    non_trainable,
                )
                return evaluate_prepared_objective(_prepared, functions).total

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
            if training is not None and training.pseudo_transient:
                previous_functions = functions_snapshot
            objective = objective.record_training_evaluations(term_indices=active_indices)
            completed = iteration
            selection_progress = replace(selection_progress, update_step=iteration)
            if profile_adaptive:
                jax.block_until_ready((params, accepted_loss, state))
                optimizer_step_wall_time = time.perf_counter() - optimizer_started
                optimizer_wall_time += optimizer_step_wall_time
                if epoch == 0:
                    first_optimizer_step_wall_time = optimizer_step_wall_time
                else:
                    steady_optimizer_step_wall_time += optimizer_step_wall_time

            accepted_loss_float = float(accepted_loss)
            selection_evaluation_loss = None
            selection_stopped = False
            if (
                keep_best
                and selection_policy is not None
                and selection_policy.due(iteration)
            ):
                selection_prepared = objective.prepare_evaluation(
                    key=jr.fold_in(iteration_key, 1201),
                    iteration=term_iteration,
                )
                selection_functions = combine_trainable(params, non_trainable)
                selection_evaluation_loss = evaluate_prepared_objective(
                    selection_prepared, selection_functions
                ).total
                selection_progress, improved = update_training_selection(
                    selection_progress,
                    float(selection_evaluation_loss),
                    step=iteration,
                    mode=selection_policy.mode,
                    min_delta=selection_policy.min_delta,
                    patience=selection_policy.patience,
                )
                if improved:
                    best_loss = float(selection_evaluation_loss)
                    best_params = params
                selection_stopped = selection_progress.stopped_early
            elif (
                keep_best and selection_policy is None and accepted_loss_float < best_loss
            ):
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
            if (
                training is not None
                and training.checkpoint is not None
                and training.checkpoint.due(iteration)
            ):
                checkpoint_selected = best_params if keep_best else params
                checkpoint_state = make_training_state(params, checkpoint_selected)
                checkpoint_solver = replace_solver_state(
                    self,
                    functions=checkpoint_state.best_functions,
                    objective=objective,
                )
                checkpoint_solver = eqx.tree_at(
                    lambda solver: solver.training_state,
                    checkpoint_solver,
                    checkpoint_state,
                    is_leaf=lambda value: value is None,
                )
                publish_checkpoint(checkpoint_solver, checkpoint_state)
            if selection_stopped:
                break

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
                    evaluation_loss=selection_evaluation_loss,
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
    if training is not None:
        training_state = make_training_state(params, chosen)
        result = eqx.tree_at(
            lambda solver: solver.training_state,
            result,
            training_state,
            is_leaf=lambda value: value is None,
        )
        if training.checkpoint is not None and training.checkpoint.save_final:
            publish_checkpoint(result, training_state)
    objective_plane_diagnostics: dict[str, Any] = {}
    if update is not None:
        objective_plane_diagnostics = {
            "objective/physical": update.physical_values(functions).total,
            "objective/surrogate": update.surrogate_loss(chosen, non_trainable),
            "gradient_alignment/intra": update.intra_gradient_alignment,
            "gradient_alignment/inter": update.inter_gradient_alignment,
        }
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
        | objective_plane_diagnostics
        | _functional_ntk_diagnostic_values(latest_ntk_diagnostics)
    )
    return eqx.tree_at(lambda solver: solver.training_diagnostics, result, diagnostics)


__all__ = ["solve_kfac"]
