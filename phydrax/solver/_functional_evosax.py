#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from evosax.algorithms.distribution_based.base import DistributionBasedAlgorithm
from jax import core as jcore

from .._frozendict import frozendict
from .._trainable import combine_trainable, partition_trainable
from .._training import (
    log_training_signal_stop as _log_training_signal_stop,
    tensorboard_every as _tensorboard_every,
    TensorBoardLogger as _TensorBoardLogger,
    TrainingController,
    TrainingProgress,
    TrainingSignalGuard as _TrainingSignalGuard,
)
from ._functional_objective import (
    evaluate_prepared_objective,
    prepared_data_metrics,
)
from ._functional_reporting import (
    best_display_value as _best_display_value,
    log_tensorboard_scalars as _log_tensorboard_scalars,
    metric_suffix as _metric_suffix,
    term_label as _term_label,
)
from ._functional_run import replace_solver_state
from ._model_losses import function_model_loss_labels


if TYPE_CHECKING:
    from ._functional_solver import FunctionalSolver


def _solve_evosax_distribution(
    self: "FunctionalSolver",
    *,
    num_iter: int,
    algo: DistributionBasedAlgorithm,
    seed: int,
    jit: bool,
    keep_best: bool,
    log_every: int,
    log_terms: bool,
    log_path: str | Path | None,
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

    params, non_trainable = partition_trainable(self.functions)
    log_every_ = int(log_every)
    if log_every_ < 0:
        raise ValueError("log_every must be >= 0.")
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

    algo_runtime = cast(Any, algo)
    algo_params = algo_runtime.default_params

    def _loss_for_params(p, non_trainable_, prepared_):
        functions = combine_trainable(p, non_trainable_)
        return evaluate_prepared_objective(prepared_, functions).total

    def _values_for_params(p, non_trainable_, prepared_):
        functions = combine_trainable(p, non_trainable_)
        return evaluate_prepared_objective(prepared_, functions).flat_values

    def _evaluation_term_values_for_params(p, non_trainable_, prepared_):
        functions = combine_trainable(p, non_trainable_)
        return evaluate_prepared_objective(
            prepared_,
            functions,
            include_model_losses=False,
        ).term_values

    def _data_metrics_for_terms(p, non_trainable_, prepared_):
        functions = combine_trainable(p, non_trainable_)
        return prepared_data_metrics(prepared_, functions)

    loss_fn = eqx.filter_jit(_loss_for_params) if jit else _loss_for_params
    terms_fn = eqx.filter_jit(_values_for_params) if jit else _values_for_params

    key = jr.key(seed)
    evo_state = algo_runtime.init(key, mean=params, params=algo_params)

    control = TrainingController(
        total_steps=int(num_iter),
        key=key,
        progress=TrainingProgress(best_value=float("inf")),
    )
    control.best_payload = params
    objective = self.objective

    log_ctx = (
        open(Path(log_path), "w", encoding="utf-8")
        if log_path is not None
        else nullcontext(None)
    )
    tb_ctx = (
        _TensorBoardLogger(tensorboard_log_dir)
        if tensorboard_writer is None and tensorboard_log_dir is not None
        else nullcontext(tensorboard_writer)
    )

    with log_ctx as log_fp, tb_ctx as tb_writer, _TrainingSignalGuard() as signal_guard:
        out_file = log_fp if log_fp is not None else None
        refresh_wall_time = 0.0
        optimizer_wall_time = 0.0

        for epoch in range(int(num_iter)):
            if signal_guard.stop_requested:
                _log_training_signal_stop(
                    "evosax",
                    signal_guard,
                    completed=epoch,
                    total=int(num_iter),
                    file=out_file,
                )
                break
            completed = epoch
            try:
                iter_start = time.perf_counter()
                control.key, ask_key, eval_key, tell_key, cand_key = jr.split(
                    control.key, 5
                )
                population, evo_state = algo_runtime.ask(ask_key, evo_state, algo_params)
                popsize = None
                for leaf in jax.tree_util.tree_leaves(population):
                    if (
                        isinstance(leaf, (jax.Array, jcore.Tracer))
                        and len(leaf.shape) > 0
                    ):
                        popsize = int(leaf.shape[0])
                        break
                if popsize is None:
                    raise ValueError(
                        "Could not infer population size from evosax population."
                    )

                iter_ = jnp.asarray(epoch + 1, dtype=float)
                functions_snapshot = combine_trainable(
                    control.selected(params),
                    non_trainable,
                )
                refresh_started = time.perf_counter() if profile_adaptive else 0.0
                objective = objective.refresh(
                    functions_snapshot,
                    key=jr.fold_in(eval_key, 101),
                    iter_=epoch + 1,
                )
                if profile_adaptive:
                    jax.block_until_ready(objective)
                    refresh_wall_time += time.perf_counter() - refresh_started
                optimizer_started = time.perf_counter() if profile_adaptive else 0.0

                # Common random numbers: prepare every stochastic term once per
                # generation and reuse the payloads across the population.
                batch_key = jr.fold_in(eval_key, 0)
                eval_key_shared = jr.fold_in(eval_key, 1)
                prepared = objective.prepare_training(
                    range(len(objective.training)),
                    scale=1.0,
                    evaluation_key=eval_key_shared,
                    sampling_key=batch_key,
                    iteration=iter_,
                )
                losses = jax.vmap(
                    lambda p: loss_fn(
                        p,
                        non_trainable,
                        prepared,
                    )
                )(population)
                evo_state, _ = algo_runtime.tell(
                    tell_key, population, losses, evo_state, algo_params
                )
                cand_params = algo_runtime.get_mean(evo_state)
                cand_loss = loss_fn(
                    cand_params,
                    non_trainable,
                    prepared,
                )
                if profile_adaptive:
                    jax.block_until_ready((evo_state, cand_params, cand_loss))
                    optimizer_wall_time += time.perf_counter() - optimizer_started
                objective = objective.record_training_evaluations(
                    multiplier=popsize + 1,
                )
                step = epoch + 1
                control.complete_update(step)
                if keep_best:
                    control.select(float(cand_loss), cand_params, step=step)
                else:
                    control.best_payload = cand_params
                completed = step
                iter_time_s = time.perf_counter() - iter_start
                if signal_guard.stop_requested:
                    _log_training_signal_stop(
                        "evosax",
                        signal_guard,
                        completed=step,
                        total=int(num_iter),
                        file=out_file,
                    )
                    break
                console_step = log_every_ > 0 and (step % log_every_ == 0)
                tensorboard_step = tb_every_ is not None and (step % tb_every_ == 0)
                train_data_metrics: tuple[dict[str, Any], ...] = tuple(
                    {} for _ in self.terms
                )
                eval_terms = jnp.zeros((0,), dtype=float)
                eval_data_metrics: tuple[dict[str, Any], ...] = tuple(
                    {} for _ in self.evaluation_terms
                )
                values_arr = jnp.zeros((0,), dtype=float)
                train_term_values = values_arr[: len(term_names)]
                train_model_loss_terms = values_arr[len(term_names) :]
                if log_terms_ and (console_step or tensorboard_step):
                    values_arr = jnp.asarray(
                        terms_fn(
                            cand_params,
                            non_trainable,
                            prepared,
                        ),
                        dtype=float,
                    )
                    train_term_values = values_arr[: len(term_names)]
                    train_model_loss_terms = values_arr[len(term_names) :]
                    train_data_metrics = _data_metrics_for_terms(
                        cand_params,
                        non_trainable,
                        prepared,
                    )
                    prepared_evaluation = objective.prepare_evaluation(
                        key=jr.fold_in(eval_key_shared, 2),
                        iteration=iter_,
                    )
                    eval_terms = _evaluation_term_values_for_params(
                        cand_params,
                        non_trainable,
                        prepared_evaluation,
                    )
                    eval_data_metrics = _data_metrics_for_terms(
                        cand_params,
                        non_trainable,
                        prepared_evaluation,
                    )

                if console_step:
                    loss_f = float(cand_loss)
                    best_display = _best_display_value(
                        control.progress.best_value,
                        loss_f,
                        keep_best=keep_best,
                    )
                    print(
                        f"[phydrax][evosax] iter {step}/{int(num_iter)} "
                        f"loss={loss_f:.6e} best={best_display:.6e} "
                        f"iter_time={iter_time_s:.3f}s",
                        file=out_file,
                    )
                    if log_terms_:
                        for i, (name, val) in enumerate(
                            zip(
                                term_names,
                                list(map(float, train_term_values)),
                                strict=True,
                            )
                        ):
                            suffix = _metric_suffix(train_data_metrics[i])
                            print(
                                f"  [train {i}] {name}: {val:.6e}{suffix}",
                                file=out_file,
                            )
                        for i, (name, val) in enumerate(
                            zip(
                                model_loss_names,
                                list(map(float, train_model_loss_terms)),
                                strict=True,
                            )
                        ):
                            print(
                                f"  [model {i}] {name}: {val:.6e}",
                                file=out_file,
                            )
                        eval_terms_arr = jnp.asarray(eval_terms, dtype=float)
                        for i, (name, val) in enumerate(
                            zip(
                                evaluation_term_names,
                                list(map(float, eval_terms_arr)),
                                strict=True,
                            )
                        ):
                            suffix = _metric_suffix(eval_data_metrics[i])
                            print(
                                f"  [eval {i}] {name}: {val:.6e}{suffix}",
                                file=out_file,
                            )
                if tensorboard_step and tb_writer is not None:
                    loss_f = float(cand_loss)
                    best_display = _best_display_value(
                        control.progress.best_value,
                        loss_f,
                        keep_best=keep_best,
                    )
                    _log_tensorboard_scalars(
                        tb_writer,
                        step=step,
                        loss=cand_loss,
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
                    if step % tb_flush_every_ == 0:
                        tb_writer.flush()
            except (KeyboardInterrupt, InterruptedError) as exc:
                signal_guard.request_stop_from_exception(exc)
                _log_training_signal_stop(
                    "evosax",
                    signal_guard,
                    completed=completed,
                    total=int(num_iter),
                    file=out_file,
                )
                break

        functions = combine_trainable(control.selected(params), non_trainable)
        settle_started = time.perf_counter() if profile_adaptive else 0.0
        objective = objective.settle(
            functions,
            key=jr.fold_in(control.key, 991),
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
        return eqx.tree_at(lambda s: s.training_diagnostics, result, diagnostics)


__all__ = ["_solve_evosax_distribution"]
