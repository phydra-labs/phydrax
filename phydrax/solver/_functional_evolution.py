#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from .._frozendict import frozendict
from .._iteration import IterationSession
from .._sampling import derive_key, SampleAddress
from .._training import (
    emit_training_signal_stop as _emit_training_signal_stop,
    tensorboard_every as _tensorboard_every,
    TensorBoardLogger as _TensorBoardLogger,
    TrainingController,
    TrainingIterationKind,
    TrainingProgress,
    TrainingSignalGuard as _TrainingSignalGuard,
)
from .._training_kernel import (
    prepare_training_kernel,
    run_training_attempt,
    TrainingAttemptOutcome,
    TrainingKernelSpec,
)
from ..logging import is_enabled as _logging_enabled
from ..optim._evolution_strategy import (
    AbstractDistributionEvolutionMethod,
    DistributionEvolutionPayload,
    DistributionEvolutionUpdateRule,
)
from ._functional_kernel import (
    functional_kernel_objective,
    functional_lanes,
    functional_parameter_lane,
    FUNCTIONAL_REJECTION_BUDGET,
    FUNCTIONAL_ROOT_AUTHORITY,
    functional_site_key,
    functional_training_tree,
)
from ._functional_objective import (
    evaluate_prepared_objective,
    prepared_data_metrics,
)
from ._functional_reporting import (
    best_display_value as _best_display_value,
    emit_training_scalars as _emit_training_scalars,
    term_label as _term_label,
    training_scalars as _training_scalars,
    write_tensorboard_scalars as _write_tensorboard_scalars,
)
from ._functional_run import replace_solver_state
from ._model_losses import function_model_loss_labels


if TYPE_CHECKING:
    from ._functional_solver import FunctionalSolver


_ALGORITHM_INITIALIZATION_ADDRESS = SampleAddress(
    "functional", "distribution-evolution", target="algorithm", role="initialization"
)


def _prepared_objective_total(
    trained: Any, held: Any, payload: DistributionEvolutionPayload, keys: Any, /
) -> tuple[jax.Array, tuple[()]]:
    """Ordered total of one generation's prepared training objective."""
    del keys
    functions = eqx.combine(trained, held)
    return evaluate_prepared_objective(payload.objective, functions).total, ()


def _solve_distribution_evolution(
    self: "FunctionalSolver",
    *,
    num_iter: int,
    algo: AbstractDistributionEvolutionMethod,
    seed: int,
    jit: bool,
    keep_best: bool,
    log_every: int,
    log_terms: bool,
    session: IterationSession | None = None,
    session_every: int = 1,
    tensorboard_log_dir: str | Path | None = None,
    tensorboard_every: int | None = None,
    tensorboard_flush_every: int = 10,
    profile_adaptive: bool = False,
    tensorboard_writer: _TensorBoardLogger | None = None,
    train_term_sample_size: int | None = None,
) -> "FunctionalSolver":

    if train_term_sample_size is not None:
        raise NotImplementedError(
            "train_term_sample_size is currently supported only for Optax optimizers."
        )

    log_every_ = int(log_every)
    if log_every_ < 0:
        raise ValueError("log_every must be >= 0.")
    session_every_ = int(session_every)
    if session_every_ <= 0:
        raise ValueError("session_every must be positive.")
    tb_every_ = _tensorboard_every(
        tensorboard_log_dir=tensorboard_log_dir,
        tensorboard_every=tensorboard_every,
        log_every=log_every_,
    )
    tb_flush_every_ = int(tensorboard_flush_every)
    if tb_flush_every_ <= 0:
        raise ValueError("tensorboard_flush_every must be positive.")
    log_terms_ = bool(log_terms)
    term_names = tuple(_term_label(c) for c in self.terms)
    model_loss_names = function_model_loss_labels(self.functions)
    evaluation_term_names = tuple(_term_label(c) for c in self.evaluation_terms)

    def _values_for_params(p, held_, prepared_):
        functions = eqx.combine(p, held_)
        return evaluate_prepared_objective(prepared_, functions).flat_values

    def _evaluation_term_values_for_params(p, held_, prepared_):
        functions = eqx.combine(p, held_)
        return evaluate_prepared_objective(
            prepared_,
            functions,
            include_model_losses=False,
        ).term_values

    def _data_metrics_for_terms(p, held_, prepared_):
        functions = eqx.combine(p, held_)
        return prepared_data_metrics(prepared_, functions)

    terms_fn = eqx.filter_jit(_values_for_params) if jit else _values_for_params

    tree = functional_training_tree(self.functions)
    root_key = jr.key(seed)
    kernel = prepare_training_kernel(
        tree,
        (functional_kernel_objective(_prepared_objective_total),),
        TrainingKernelSpec(
            DistributionEvolutionUpdateRule(
                algo,
                derive_key(root_key, _ALGORITHM_INITIALIZATION_ADDRESS),
                rule_id="functional-distribution-evolution",
            ),
            context="FunctionalSolver distribution evolution",
            rejection_budget=FUNCTIONAL_REJECTION_BUDGET,
        ),
        root_authority=FUNCTIONAL_ROOT_AUTHORITY,
    )
    state = kernel.init(tree, root_key)
    params, held = functional_lanes(state.parameters, state.model_state, kernel.fixed)
    # One generation evaluates the prepared objective at the committed mean (the
    # kernel's attempt value), at every population member, and at the proposal.
    evaluations_per_generation = algo.population_size + 2

    total_steps = int(num_iter)
    control = TrainingController(
        total_steps=total_steps,
        algorithm_id="functional-evolution-training",
        progress=TrainingProgress(),
        session=session,
    )
    control.best_payload = params
    control.emit(TrainingIterationKind.RUN_START, metrics={"total_steps": total_steps})
    objective = self.objective

    tb_ctx = (
        _TensorBoardLogger(tensorboard_log_dir)
        if tensorboard_writer is None and tensorboard_log_dir is not None
        else nullcontext(tensorboard_writer)
    )

    with tb_ctx as tb_writer, _TrainingSignalGuard() as signal_guard:
        refresh_wall_time = 0.0
        optimizer_wall_time = 0.0

        completed = control.progress.update_step
        while completed < total_steps:
            if control.stop_requested:
                break
            if signal_guard.stop_requested:
                _emit_training_signal_stop(
                    "native-evolution",
                    signal_guard,
                    completed=completed,
                    total=total_steps,
                )
                break
            try:
                iter_start = time.perf_counter()
                step = completed + 1
                iter_ = jnp.asarray(step, dtype=jnp.float64)
                # Every host site of this attempt is addressed by its cursors.
                attempt_state = state
                refresh_started = time.perf_counter() if profile_adaptive else 0.0
                attempt_objective = objective.refresh(
                    eqx.combine(control.selected(params), held),
                    key=functional_site_key(attempt_state, "refresh"),
                    iter_=step,
                )
                if profile_adaptive:
                    jax.block_until_ready(attempt_objective)
                    refresh_wall_time += time.perf_counter() - refresh_started
                optimizer_started = time.perf_counter() if profile_adaptive else 0.0

                # Common random numbers: prepare every stochastic term once per
                # generation and reuse the payload across the population.
                prepared = attempt_objective.prepare_training(
                    range(len(attempt_objective.training)),
                    scale=1.0,
                    evaluation_key=functional_site_key(attempt_state, "evaluation"),
                    sampling_key=functional_site_key(attempt_state, "sampling"),
                    iteration=iter_,
                )
                state, evidence = run_training_attempt(
                    kernel,
                    state,
                    DistributionEvolutionPayload(
                        prepared, functional_site_key(attempt_state, "ask")
                    ),
                    jit=jit,
                )
                outcome, cand_loss = jax.device_get(
                    (evidence.outcome, state.rule_state.value)
                )
                if profile_adaptive:
                    optimizer_wall_time += time.perf_counter() - optimizer_started
                if outcome != TrainingAttemptOutcome.ACCEPTED:
                    # A rejected generation commits nothing, including the
                    # refreshed objective: the retry refreshes and asks afresh.
                    continue
                params = functional_parameter_lane(state.parameters)
                objective = attempt_objective.record_training_evaluations(
                    multiplier=evaluations_per_generation,
                )
                control.complete_update(step)
                loss_f = float(cand_loss)
                if keep_best:
                    control.select(loss_f, params, step=step)
                else:
                    control.best_payload = params
                completed = step
                iter_time_s = time.perf_counter() - iter_start
                log_step = (
                    _logging_enabled() and log_every_ > 0 and step % log_every_ == 0
                )
                tensorboard_step = tb_every_ is not None and (step % tb_every_ == 0)
                session_step = session is not None and step % session_every_ == 0
                report_step = log_step or tensorboard_step or session_step
                train_data_metrics: tuple[dict[str, Any], ...] = tuple(
                    {} for _ in self.terms
                )
                eval_terms = jnp.zeros((0,), dtype=jnp.float64)
                eval_data_metrics: tuple[dict[str, Any], ...] = tuple(
                    {} for _ in self.evaluation_terms
                )
                values_arr = jnp.zeros((0,), dtype=jnp.float64)
                train_term_values = values_arr[: len(term_names)]
                train_model_loss_terms = values_arr[len(term_names) :]
                if log_terms_ and report_step:
                    values_arr = jnp.asarray(
                        terms_fn(params, held, prepared),
                        dtype=jnp.float64,
                    )
                    train_term_values = values_arr[: len(term_names)]
                    train_model_loss_terms = values_arr[len(term_names) :]
                    train_data_metrics = _data_metrics_for_terms(params, held, prepared)
                    prepared_evaluation = objective.prepare_evaluation(
                        key=functional_site_key(attempt_state, "report-evaluation"),
                        iteration=iter_,
                    )
                    eval_terms = _evaluation_term_values_for_params(
                        params,
                        held,
                        prepared_evaluation,
                    )
                    eval_data_metrics = _data_metrics_for_terms(
                        params,
                        held,
                        prepared_evaluation,
                    )

                if report_step:
                    best_display = _best_display_value(
                        control.progress.best_value,
                        loss_f,
                        keep_best=keep_best,
                    )
                    scalars = _training_scalars(
                        loss=loss_f,
                        best_loss=best_display,
                        evaluation_loss=None,
                        iter_time_s=iter_time_s,
                        train_term_names=term_names,
                        train_terms=train_term_values,
                        train_data_metrics=train_data_metrics,
                        train_model_loss_names=model_loss_names,
                        train_model_loss_terms=train_model_loss_terms,
                        evaluation_term_names=evaluation_term_names,
                        eval_terms=eval_terms,
                        eval_data_metrics=eval_data_metrics,
                        log_terms=log_terms_,
                    )
                    if session_step:
                        control.deliver(
                            TrainingIterationKind.UPDATE,
                            metrics=scalars,
                        )
                    if log_step:
                        _emit_training_scalars(
                            scalars,
                            backend="native-evolution",
                            step=step,
                            total_steps=total_steps,
                        )
                    if tensorboard_step and tb_writer is not None:
                        _write_tensorboard_scalars(tb_writer, scalars, step=step)
                        if step % tb_flush_every_ == 0:
                            tb_writer.flush()
                if control.stop_requested:
                    break
                if signal_guard.stop_requested:
                    _emit_training_signal_stop(
                        "native-evolution",
                        signal_guard,
                        completed=step,
                        total=total_steps,
                    )
                    break
            except (KeyboardInterrupt, InterruptedError) as exc:
                signal_guard.request_stop_from_exception(exc)
                _emit_training_signal_stop(
                    "native-evolution",
                    signal_guard,
                    completed=completed,
                    total=total_steps,
                )
                break

        functions = eqx.combine(control.selected(params), held)
        settle_started = time.perf_counter() if profile_adaptive else 0.0
        objective = objective.settle(
            functions,
            key=functional_site_key(state, "settle"),
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
            }
        )
        control.emit(
            TrainingIterationKind.RUN_TERMINAL,
            metrics={"completed_steps": completed},
        )
        return eqx.tree_at(lambda s: s.training_diagnostics, result, diagnostics)


__all__ = ["_solve_distribution_evolution"]
