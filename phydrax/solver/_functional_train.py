#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Protocol, runtime_checkable, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from evosax.algorithms.distribution_based.base import DistributionBasedAlgorithm
from evosax.algorithms.population_based.base import PopulationBasedAlgorithm
from jax import core as jcore

from .._frozendict import frozendict
from .._term import AbstractSamplingTerm, evaluate
from .._trainable import combine_trainable, partition_trainable
from .._training import (
    EvaluationParametersFn,
    log_training_signal_stop as _log_training_signal_stop,
    resolve_evaluation_parameters,
    tensorboard_every as _tensorboard_every,
    TensorBoardLogger as _TensorBoardLogger,
    TrainingController,
    TrainingProgress,
    TrainingSignalGuard as _TrainingSignalGuard,
)
from ..integration import AdaptiveIntegration
from ..operators.differential._runtime import derivative_runtime_context
from ..optim._riemannian import (
    AbstractRiemannianLineSearchOptimizer,
    AbstractRiemannianOptimizer,
)
from ..sampling.collocation import ControlledCollocationPolicy
from ..terms._residual import ResidualPenalty
from ._model_losses import function_model_loss_labels, function_model_loss_values


if TYPE_CHECKING:
    from ._functional_solver import FunctionalSolver


@runtime_checkable
class _SupportsDataMetrics(Protocol):
    def data_metrics(
        self,
        functions: Any,
        /,
        *,
        key: Any,
        **kwargs: Any,
    ) -> dict[str, Any]: ...


def _sample_term_batches(
    terms: tuple[Any, ...],
    populations: tuple[Any | None, ...],
    /,
    *,
    key: Any,
):
    keys = jr.split(key, len(terms))
    return tuple(
        term.sample(key=term_key)
        if population is None and isinstance(term, AbstractSamplingTerm)
        else None
        for term, population, term_key in zip(
            terms,
            populations,
            keys,
            strict=True,
        )
    )


def _adaptive_policy(term: Any, /):
    if not isinstance(term, ResidualPenalty) or not isinstance(
        term.source, AdaptiveIntegration
    ):
        raise TypeError(
            "Adaptive collocation requires ResidualPenalty with an "
            "AdaptiveIntegration source."
        )
    return term.policy


def _term_label(term: Any, /) -> str:
    return term.label or type(term).__name__


def _clean_tag_part(value: str, /) -> str:
    cleaned = "".join(
        ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value)
    ).strip("_")
    return cleaned or "term"


def _term_tag(index: int, name: str, /) -> str:
    return f"terms/{index:03d}_{_clean_tag_part(name)}"


def _model_loss_tag(index: int, name: str, /) -> str:
    return f"model_losses/{index:03d}_{_clean_tag_part(name)}"


def _metric_suffix(metrics: dict[str, Any], /) -> str:
    if not metrics:
        return ""
    parts = []
    for name, value in metrics.items():
        value_f = float(jnp.asarray(value, dtype=float).reshape(()))
        parts.append(f"{name}={value_f:.6e}")
    return " " + " ".join(parts)


def _best_display_value(
    best_value: int | float | None,
    loss: float,
    /,
    *,
    keep_best: bool,
) -> float:
    if keep_best and best_value is not None:
        return float(best_value)
    return loss


def _train_term_sample_size(
    value: int | None,
    /,
    *,
    num_terms: int,
) -> int | None:
    if value is None:
        return None
    count = int(num_terms)
    if count <= 0:
        raise ValueError("train_term_sample_size requires at least one training term.")
    sample_size = int(value)
    if sample_size <= 0:
        raise ValueError("train_term_sample_size must be positive.")
    if sample_size >= count:
        return None
    return sample_size


def _active_train_terms(
    terms: tuple[Any, ...],
    /,
    *,
    sample_size: int | None,
    key: Any,
) -> tuple[tuple[Any, ...], tuple[int, ...], Any]:
    count = len(terms)
    if sample_size is None:
        return terms, tuple(range(count)), jnp.asarray(1.0, dtype=float)
    sampled = jr.choice(
        key,
        count,
        shape=(int(sample_size),),
        replace=False,
    )
    active_indices = tuple(int(i) for i in np.asarray(sampled, dtype=np.int32))
    active = tuple(terms[i] for i in active_indices)
    scale = jnp.asarray(count / int(sample_size), dtype=float)
    return active, active_indices, scale


def _expanded_train_terms(
    active_terms: Any,
    /,
    *,
    active_term_indices: tuple[int, ...],
    num_terms: int,
) -> Any:
    active_arr = jnp.asarray(active_terms, dtype=float).reshape((-1,))
    if int(active_arr.shape[0]) == 0:
        return jnp.zeros((int(num_terms),), dtype=float)
    out = jnp.full((int(num_terms),), jnp.nan, dtype=float)
    for local_i, term_i in enumerate(active_term_indices):
        out = out.at[int(term_i)].set(active_arr[int(local_i)])
    return out


def _write_term_tensorboard_scalars(
    writer: _TensorBoardLogger,
    *,
    step: int,
    namespace: str,
    term_names: tuple[str, ...],
    terms: Any,
    data_metrics: tuple[dict[str, Any], ...],
) -> None:
    terms_arr = jnp.asarray(terms, dtype=float)
    for i, (name, val) in enumerate(
        zip(term_names, list(map(float, terms_arr)), strict=True)
    ):
        prefix = f"{namespace}/{_term_tag(i, name)}"
        writer.scalar(f"{prefix}/value", val, step)
        for metric_name, metric_value in data_metrics[i].items():
            writer.scalar(f"{prefix}/{metric_name}", metric_value, step)


def _write_model_loss_tensorboard_scalars(
    writer: _TensorBoardLogger,
    *,
    step: int,
    model_loss_names: tuple[str, ...],
    terms: Any,
) -> None:
    terms_arr = jnp.asarray(terms, dtype=float)
    for i, (name, val) in enumerate(
        zip(model_loss_names, list(map(float, terms_arr)), strict=True)
    ):
        writer.scalar(f"train/{_model_loss_tag(i, name)}/loss", val, step)


def _log_tensorboard_scalars(
    writer: _TensorBoardLogger,
    *,
    step: int,
    loss: Any,
    best_loss: float,
    evaluation_loss: Any | None,
    iter_time_s: float,
    train_term_names: tuple[str, ...],
    train_terms: Any,
    train_data_metrics: tuple[dict[str, Any], ...],
    train_model_loss_names: tuple[str, ...],
    train_model_loss_terms: Any,
    evaluation_term_names: tuple[str, ...],
    eval_terms: Any,
    eval_data_metrics: tuple[dict[str, Any], ...],
    log_terms: bool,
) -> None:
    writer.scalar("train/loss", loss, step)
    writer.scalar("train/best_loss", best_loss, step)
    if evaluation_loss is not None:
        writer.scalar("eval/loss", evaluation_loss, step)
    writer.scalar("train/iter_time_s", iter_time_s, step)

    if not log_terms:
        return

    _write_term_tensorboard_scalars(
        writer,
        step=step,
        namespace="train",
        term_names=train_term_names,
        terms=train_terms,
        data_metrics=train_data_metrics,
    )
    _write_model_loss_tensorboard_scalars(
        writer,
        step=step,
        model_loss_names=train_model_loss_names,
        terms=train_model_loss_terms,
    )
    _write_term_tensorboard_scalars(
        writer,
        step=step,
        namespace="eval",
        term_names=evaluation_term_names,
        terms=eval_terms,
        data_metrics=eval_data_metrics,
    )


def _term_value(
    term,
    population: Any | None,
    materialized_batch: Any | None,
    functions,
    /,
    *,
    key,
    iter_,
):
    evaluation_kwargs: dict[str, Any] = {}
    if population is not None:
        policy = _adaptive_policy(term)
        batch, local_weight = policy.loss_batch_and_weight(population)
        evaluation_kwargs["realization"] = term._adaptive_realization(
            batch,
            local_weight,
            key=key,
        )
    elif materialized_batch is not None:
        evaluation_kwargs["batch"] = materialized_batch
    return evaluate(
        term,
        functions,
        key=key,
        step=iter_,
        **evaluation_kwargs,
    ).value


def _refresh_collocation(
    solver,
    functions,
    collocation: tuple[Any | None, ...],
    /,
    *,
    key,
    iter_,
) -> tuple[Any | None, ...]:
    if solver.enforcement is None:
        enforced = functions
    else:
        enforced = solver.enforcement.apply(functions)
    keys = jr.split(key, len(solver.terms))
    refreshed: list[Any | None] = []
    for term, population, term_key in zip(solver.terms, collocation, keys, strict=True):
        if population is None:
            refreshed.append(None)
            continue
        policy = _adaptive_policy(term)
        if bool(policy.should_refresh(population, iter_)):
            population = policy.refresh(
                term,
                enforced,
                population,
                key=term_key,
                iter_=iter_,
            )
        refreshed.append(population)
    return tuple(refreshed)


def _settle_collocation(
    solver,
    functions,
    collocation: tuple[Any | None, ...],
    /,
    *,
    key,
    iter_,
) -> tuple[Any | None, ...]:
    if solver.enforcement is None:
        enforced = functions
    else:
        enforced = solver.enforcement.apply(functions)
    keys = jr.split(key, len(solver.terms))
    settled: list[Any | None] = []
    for term, population, term_key in zip(solver.terms, collocation, keys, strict=True):
        if population is None:
            settled.append(None)
            continue
        policy = _adaptive_policy(term)
        if isinstance(policy, ControlledCollocationPolicy):
            population = policy.settle(
                term,
                enforced,
                population,
                key=term_key,
                iter_=iter_,
            )
        settled.append(population)
    return tuple(settled)


def _record_collocation_training_evaluations(
    solver,
    collocation: tuple[Any | None, ...],
    /,
    *,
    multiplier: int = 1,
    term_indices: tuple[int, ...] | None = None,
) -> tuple[Any | None, ...]:
    selected = (
        None if term_indices is None else frozenset(int(index) for index in term_indices)
    )
    recorded: list[Any | None] = []
    for index, (term, population) in enumerate(
        zip(solver.terms, collocation, strict=True)
    ):
        if population is None or (selected is not None and index not in selected):
            recorded.append(population)
            continue
        policy = _adaptive_policy(term)
        if isinstance(policy, ControlledCollocationPolicy):
            population = policy.record_training_evaluation(
                population,
                multiplier=multiplier,
            )
        recorded.append(population)
    return tuple(recorded)


def _collocation_data_metrics(
    solver,
    collocation: tuple[Any | None, ...],
    /,
) -> tuple[dict[str, jax.Array], ...]:
    metrics: list[dict[str, jax.Array]] = []
    for term, population in zip(
        solver.terms,
        collocation,
        strict=True,
    ):
        if population is None:
            metrics.append({})
            continue
        policy = _adaptive_policy(term)
        metrics.append(policy.data_metrics(population))
    return tuple(metrics)


def solve(
    self: "FunctionalSolver",
    *,
    num_iter: int,
    optim: AbstractRiemannianOptimizer
    | optax.GradientTransformation
    | optax.GradientTransformationExtraArgs
    | Any = optax.rprop(1e-3),
    evaluation_parameters: EvaluationParametersFn | None = None,
    seed: int = 0,
    jit: bool = True,
    keep_best: bool = True,
    log_every: int = 1,
    log_terms: bool = True,
    log_path: str | Path | None = None,
    tensorboard_log_dir: str | Path | None = None,
    tensorboard_every: int | None = None,
    tensorboard_flush_every: int = 10,
    profile_adaptive: bool = False,
    train_term_sample_size: int | None = None,
) -> "FunctionalSolver":
    if num_iter == 0:
        return self

    if isinstance(optim, str):
        raise TypeError(
            "optim must be an optimizer object (e.g. phydrax.optim.riemannian_sgd(...), "
            "optax.adam(...), optax.lbfgs(...), or an evosax distribution-based "
            "algorithm instance), not a string."
        )

    _opt_linesearch: optax.GradientTransformationExtraArgs | None = None
    _opt_standard: optax.GradientTransformation | None = None

    _opt_riemannian: AbstractRiemannianOptimizer | None = None
    if isinstance(optim, AbstractRiemannianOptimizer):
        if evaluation_parameters is not None:
            raise ValueError(
                "evaluation_parameters is unsupported for Riemannian optimizers "
                "because an ambient transform need not preserve manifold membership."
            )
        _opt_riemannian = optim
    elif isinstance(optim, optax.GradientTransformationExtraArgs):
        _opt_linesearch = optim
    elif isinstance(optim, optax.GradientTransformation):
        _opt_standard = optim
    elif isinstance(optim, PopulationBasedAlgorithm):
        if evaluation_parameters is not None:
            raise ValueError(
                "evaluation_parameters is supported only for Optax optimizers."
            )
        raise NotImplementedError(
            "FunctionalSolver does not accept Evosax population-based algorithms: "
            "they require an explicit initial population and finite search-space "
            "semantics. For bounded geometry design, use "
            "DesignConstraintSystem.search(...)."
        )
    elif isinstance(optim, DistributionBasedAlgorithm):
        if evaluation_parameters is not None:
            raise ValueError(
                "evaluation_parameters is supported only for Optax optimizers."
            )
        return _solve_evosax_distribution(
            self,
            num_iter=num_iter,
            algo=optim,
            seed=seed,
            jit=jit,
            keep_best=keep_best,
            log_every=log_every,
            log_terms=log_terms,
            log_path=log_path,
            tensorboard_log_dir=tensorboard_log_dir,
            tensorboard_every=tensorboard_every,
            tensorboard_flush_every=tensorboard_flush_every,
            profile_adaptive=profile_adaptive,
            train_term_sample_size=train_term_sample_size,
        )
    else:
        raise TypeError(
            "optim must be a Phydrax Riemannian optimizer, an Optax transformation, "
            "or an Evosax distribution-based algorithm instance."
        )
    optimizer_label = (
        _opt_riemannian.optimizer_id if _opt_riemannian is not None else "optax"
    )

    log_ctx = (
        open(Path(log_path), "w", encoding="utf-8")
        if log_path is not None
        else nullcontext(None)
    )

    tb_ctx = (
        _TensorBoardLogger(tensorboard_log_dir)
        if tensorboard_log_dir is not None
        else nullcontext(None)
    )

    with log_ctx as log_fp, tb_ctx as tb_writer, _TrainingSignalGuard() as signal_guard:
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
        term_sample_size = _train_term_sample_size(
            train_term_sample_size,
            num_terms=len(self.terms),
        )

        def _loss_wrt_params(
            params_,
            non_trainable_,
            solver,
            active_terms_,
            term_scale,
            key,
            iter_,
            collocation_,
            materialized_batches_,
        ):
            functions = combine_trainable(params_, non_trainable_)
            if solver.enforcement is None:
                enforced = functions
            else:
                enforced = solver.enforcement.apply(functions)
            term_keys = jr.split(key, len(active_terms_))
            total = jnp.array(0.0, dtype=float)
            scale = jnp.asarray(term_scale, dtype=float).reshape(())
            evaluated: list[jax.Array] = []
            with derivative_runtime_context():
                for term, population, batch, term_key in zip(
                    active_terms_,
                    collocation_,
                    materialized_batches_,
                    term_keys,
                    strict=True,
                ):
                    value = _term_value(
                        term,
                        population,
                        batch,
                        enforced,
                        key=term_key,
                        iter_=iter_,
                    )
                    scaled_value = scale * jnp.asarray(value, dtype=float).reshape(())
                    total = total + scaled_value
                    if log_terms_:
                        evaluated.append(scaled_value)
                for value in function_model_loss_values(
                    functions,
                    key=jr.fold_in(key, len(active_terms_)),
                    iter_=iter_,
                ):
                    scalar_value = jnp.asarray(value, dtype=float).reshape(())
                    total = total + scalar_value
                    if log_terms_:
                        evaluated.append(scalar_value)
            if evaluated:
                return total, jnp.stack(evaluated, axis=0)
            return total, jnp.zeros((0,), dtype=float)

        def _enforced_functions_wrt_params(params_, non_trainable_, solver):
            functions = combine_trainable(params_, non_trainable_)
            if solver.enforcement is None:
                return functions
            return solver.enforcement.apply(functions)

        def _term_values_wrt_params(
            params_,
            non_trainable_,
            solver,
            requested_terms,
            key,
            iter_,
        ):
            enforced = _enforced_functions_wrt_params(params_, non_trainable_, solver)
            keys = jr.split(key, len(requested_terms))
            batches = _sample_term_batches(
                requested_terms,
                (None,) * len(requested_terms),
                key=jr.fold_in(key, 1),
            )
            values: list[jax.Array] = []
            with derivative_runtime_context():
                for term, batch, term_key in zip(
                    requested_terms, batches, keys, strict=True
                ):
                    value = _term_value(
                        term,
                        None,
                        batch,
                        enforced,
                        key=term_key,
                        iter_=iter_,
                    )
                    values.append(jnp.asarray(value, dtype=float).reshape(()))
            if values:
                return jnp.stack(values, axis=0)
            return jnp.zeros((0,), dtype=float)

        def _data_metrics_wrt_terms(
            params_,
            non_trainable_,
            solver,
            requested_terms,
            key,
            iter_,
        ):
            enforced = _enforced_functions_wrt_params(params_, non_trainable_, solver)
            keys = jr.split(key, len(requested_terms))
            metrics: list[dict[str, Any]] = []
            with derivative_runtime_context():
                for term, term_key in zip(requested_terms, keys, strict=True):
                    if isinstance(term, _SupportsDataMetrics):
                        metrics.append(
                            term.data_metrics(enforced, key=term_key, iter_=iter_)
                        )
                    else:
                        metrics.append({})
            return tuple(metrics)

        loss_fn = eqx.filter_value_and_grad(_loss_wrt_params, has_aux=True)

        is_linesearch = _opt_linesearch is not None
        is_riemannian = _opt_riemannian is not None
        is_riemannian_linesearch = isinstance(
            _opt_riemannian, AbstractRiemannianLineSearchOptimizer
        )

        def solve_step_terms(
            params_,
            non_trainable_,
            opt_state,
            solver,
            active_terms_,
            term_scale,
            key,
            iter_,
            collocation_,
            materialized_batches_,
        ):
            if is_riemannian:
                (loss_val, terms), grads = loss_fn(
                    params_,
                    non_trainable_,
                    solver,
                    active_terms_,
                    term_scale,
                    key,
                    iter_,
                    collocation_,
                    materialized_batches_,
                )
                assert _opt_riemannian is not None
                if is_riemannian_linesearch:

                    def _riemannian_value_fn(p):
                        return _loss_wrt_params(
                            p,
                            non_trainable_,
                            solver,
                            active_terms_,
                            term_scale,
                            key,
                            iter_,
                            collocation_,
                            materialized_batches_,
                        )[0]

                    params_, opt_state = _opt_riemannian.update(
                        grads,
                        opt_state,
                        params_,
                        value=loss_val,
                        value_fn=_riemannian_value_fn,
                    )
                    loss_val, terms = _loss_wrt_params(
                        params_,
                        non_trainable_,
                        solver,
                        active_terms_,
                        term_scale,
                        key,
                        iter_,
                        collocation_,
                        materialized_batches_,
                    )
                else:
                    params_, opt_state = _opt_riemannian.update(
                        grads,
                        opt_state,
                        params_,
                    )
                return params_, opt_state, loss_val, terms

            if is_linesearch:
                import jax.tree_util as jtu

                def _value_fn(p):
                    return _loss_wrt_params(
                        p,
                        non_trainable_,
                        solver,
                        active_terms_,
                        term_scale,
                        key,
                        iter_,
                        collocation_,
                        materialized_batches_,
                    )[0]

                (value, _term_values0), grads = loss_fn(
                    params_,
                    non_trainable_,
                    solver,
                    active_terms_,
                    term_scale,
                    key,
                    iter_,
                    collocation_,
                    materialized_batches_,
                )
                grads = jtu.tree_map(
                    lambda a: (
                        jnp.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
                        if eqx.is_inexact_array(a)
                        else a
                    ),
                    grads,
                    is_leaf=eqx.is_inexact_array,
                )
                assert _opt_linesearch is not None
                updates, opt_state = _opt_linesearch.update(
                    grads,
                    opt_state,
                    params_,
                    value=value,
                    grad=grads,
                    value_fn=_value_fn,
                )
                params_ = eqx.apply_updates(params_, updates)
                loss_val, term_values = _loss_wrt_params(
                    params_,
                    non_trainable_,
                    solver,
                    active_terms_,
                    term_scale,
                    key,
                    iter_,
                    collocation_,
                    materialized_batches_,
                )
                return params_, opt_state, loss_val, term_values

            (loss_val, term_values), grads = loss_fn(
                params_,
                non_trainable_,
                solver,
                active_terms_,
                term_scale,
                key,
                iter_,
                collocation_,
                materialized_batches_,
            )
            assert _opt_standard is not None
            updates, opt_state = _opt_standard.update(grads, opt_state, params_)
            params_ = eqx.apply_updates(params_, updates)
            return params_, opt_state, loss_val, term_values

        solve_step = (
            eqx.filter_jit(solve_step_terms)
            if jit and not is_linesearch
            else solve_step_terms
        )
        selection_loss_fn = (
            eqx.filter_jit(_loss_wrt_params)
            if jit and evaluation_parameters is not None
            else _loss_wrt_params
        )

        opt = (
            _opt_riemannian
            if is_riemannian
            else (_opt_linesearch if is_linesearch else _opt_standard)
        )
        if opt is None:
            raise ValueError("Optimizer is not configured.")
        opt_state = opt.init(params)
        current_evaluation_params = resolve_evaluation_parameters(
            evaluation_parameters,
            opt_state,
            params,
        )
        control = TrainingController(
            total_steps=int(num_iter),
            key=jr.key(seed),
            progress=TrainingProgress(best_value=float("inf")),
        )
        control.best_payload = current_evaluation_params
        collocation = self.collocation
        out_file = log_fp if log_fp is not None else None
        refresh_wall_time = 0.0
        optimizer_wall_time = 0.0
        first_optimizer_step_wall_time = 0.0
        steady_optimizer_step_wall_time = 0.0

        for epoch in range(int(num_iter)):
            if signal_guard.stop_requested:
                _log_training_signal_stop(
                    optimizer_label,
                    signal_guard,
                    completed=epoch,
                    total=int(num_iter),
                    file=out_file,
                )
                break
            completed = epoch
            try:
                iter_start = time.perf_counter()
                subkey = control.split_key()
                iter_ = jnp.asarray(epoch + 1, dtype=float)
                functions_snapshot = combine_trainable(params, non_trainable)
                refresh_started = time.perf_counter() if profile_adaptive else 0.0
                collocation = _refresh_collocation(
                    self,
                    functions_snapshot,
                    collocation,
                    key=jr.fold_in(subkey, 101),
                    iter_=epoch + 1,
                )
                if profile_adaptive:
                    jax.block_until_ready(collocation)
                    refresh_wall_time += time.perf_counter() - refresh_started
                optimizer_started = time.perf_counter() if profile_adaptive else 0.0
                active_terms, active_term_indices, term_scale = _active_train_terms(
                    self.terms,
                    sample_size=term_sample_size,
                    key=jr.fold_in(subkey, 17),
                )
                active_collocation = tuple(
                    collocation[index] for index in active_term_indices
                )
                materialized_batches = _sample_term_batches(
                    active_terms,
                    active_collocation,
                    key=jr.fold_in(subkey, 211),
                )
                pre_update_params = params
                params, opt_state, loss_val, term_values = solve_step(
                    params,
                    non_trainable,
                    opt_state,
                    self,
                    active_terms,
                    term_scale,
                    subkey,
                    iter_,
                    active_collocation,
                    materialized_batches,
                )
                if profile_adaptive:
                    jax.block_until_ready((params, opt_state, loss_val))
                    optimizer_step_wall_time = time.perf_counter() - optimizer_started
                    optimizer_wall_time += optimizer_step_wall_time
                    if epoch == 0:
                        first_optimizer_step_wall_time = optimizer_step_wall_time
                    else:
                        steady_optimizer_step_wall_time += optimizer_step_wall_time
                collocation = _record_collocation_training_evaluations(
                    self,
                    collocation,
                    term_indices=active_term_indices,
                )
                completed = epoch + 1
                values_arr = jnp.asarray(term_values, dtype=float)
                active_term_count = len(active_terms)
                train_term_values = _expanded_train_terms(
                    values_arr[:active_term_count],
                    active_term_indices=active_term_indices,
                    num_terms=len(term_names),
                )
                train_model_loss_terms = values_arr[active_term_count:]
                step = epoch + 1
                control.complete_update(step)
                current_evaluation_params = resolve_evaluation_parameters(
                    evaluation_parameters,
                    opt_state,
                    params,
                )
                evaluation_loss = None
                if keep_best:
                    if evaluation_parameters is None:
                        selection_parameters = (
                            params
                            if is_linesearch or is_riemannian_linesearch
                            else pre_update_params
                        )
                        selection_loss = loss_val
                    else:
                        evaluation_loss, _ = selection_loss_fn(
                            current_evaluation_params,
                            non_trainable,
                            self,
                            active_terms,
                            term_scale,
                            subkey,
                            iter_,
                            active_collocation,
                            materialized_batches,
                        )
                        selection_parameters = current_evaluation_params
                        selection_loss = evaluation_loss
                    control.select(
                        float(selection_loss),
                        selection_parameters,
                        step=step,
                    )
                iter_time_s = time.perf_counter() - iter_start
                if signal_guard.stop_requested:
                    _log_training_signal_stop(
                        optimizer_label,
                        signal_guard,
                        completed=step,
                        total=int(num_iter),
                        file=out_file,
                    )
                    break
                console_step = log_every_ > 0 and (step % log_every_ == 0)
                tensorboard_step = tb_every_ is not None and (step % tb_every_ == 0)
                riemannian_step_metrics = None
                riemannian_constraint_residual = None
                if _opt_riemannian is not None and (console_step or tensorboard_step):
                    riemannian_step_metrics = _opt_riemannian.step_metrics(opt_state)
                    riemannian_constraint_residual = (
                        _opt_riemannian.parameter_geometry.maximum_constraint_residual(
                            current_evaluation_params
                        )
                    )
                train_data_metrics: tuple[dict[str, Any], ...] = tuple(
                    {} for _ in self.terms
                )
                eval_terms = jnp.zeros((0,), dtype=float)
                eval_data_metrics: tuple[dict[str, Any], ...] = tuple(
                    {} for _ in self.evaluation_terms
                )
                if log_terms_ and (console_step or tensorboard_step):
                    train_data_metrics = _data_metrics_wrt_terms(
                        current_evaluation_params,
                        non_trainable,
                        self,
                        self.terms,
                        jr.fold_in(subkey, 1),
                        iter_,
                    )
                    collocation_metrics = _collocation_data_metrics(
                        self,
                        collocation,
                    )
                    train_data_metrics = tuple(
                        data_metrics | adaptive_metrics
                        for data_metrics, adaptive_metrics in zip(
                            train_data_metrics,
                            collocation_metrics,
                            strict=True,
                        )
                    )
                    eval_terms = _term_values_wrt_params(
                        current_evaluation_params,
                        non_trainable,
                        self,
                        self.evaluation_terms,
                        jr.fold_in(subkey, 2),
                        iter_,
                    )
                    eval_data_metrics = _data_metrics_wrt_terms(
                        current_evaluation_params,
                        non_trainable,
                        self,
                        self.evaluation_terms,
                        jr.fold_in(subkey, 3),
                        iter_,
                    )

                if console_step:
                    loss_f = float(loss_val)
                    best_display = _best_display_value(
                        control.progress.best_value,
                        loss_f,
                        keep_best=keep_best,
                    )
                    evaluation_suffix = (
                        ""
                        if evaluation_loss is None
                        else f" eval_loss={float(evaluation_loss):.6e}"
                    )
                    optimizer_suffix = ""
                    if riemannian_step_metrics is not None:
                        if riemannian_constraint_residual is None:
                            raise RuntimeError(
                                "Riemannian diagnostics require a constraint residual."
                            )
                        optimizer_suffix = (
                            " rgrad="
                            f"{float(riemannian_step_metrics.gradient_norm):.6e}"
                            " step_norm="
                            f"{float(riemannian_step_metrics.tangent_step_norm):.6e}"
                            " constraint="
                            f"{float(riemannian_constraint_residual):.6e}"
                        )
                        if (
                            _opt_riemannian is not None
                            and _opt_riemannian.optimizer_id == "riemannian-momentum"
                        ):
                            optimizer_suffix += (
                                " momentum="
                                f"{float(riemannian_step_metrics.momentum_norm):.6e}"
                            )
                        if (
                            _opt_riemannian is not None
                            and _opt_riemannian.optimizer_id
                            in (
                                "riemannian-conjugate-gradient",
                                "riemannian-lbfgs",
                            )
                        ):
                            optimizer_suffix += (
                                " line_search="
                                f"{int(riemannian_step_metrics.line_search_evaluations)}"
                                " accepted="
                                f"{int(riemannian_step_metrics.line_search_accepted)}"
                                " reduction="
                                f"{float(riemannian_step_metrics.line_search_reduction):.6e}"
                            )
                    print(
                        f"[phydrax][{optimizer_label}] iter {step}/{int(num_iter)} "
                        f"loss={loss_f:.6e}{evaluation_suffix} "
                        f"best={best_display:.6e} iter_time={iter_time_s:.3f}s"
                        f"{optimizer_suffix}",
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
                    loss_f = float(loss_val)
                    best_display = _best_display_value(
                        control.progress.best_value,
                        loss_f,
                        keep_best=keep_best,
                    )
                    _log_tensorboard_scalars(
                        tb_writer,
                        step=step,
                        loss=loss_val,
                        best_loss=best_display,
                        evaluation_loss=evaluation_loss,
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
                    if riemannian_step_metrics is not None:
                        tb_writer.scalar(
                            "optimizer/riemannian/learning_rate",
                            riemannian_step_metrics.learning_rate,
                            step,
                        )
                        tb_writer.scalar(
                            "optimizer/riemannian/gradient_norm",
                            riemannian_step_metrics.gradient_norm,
                            step,
                        )
                        tb_writer.scalar(
                            "optimizer/riemannian/clipping_scale",
                            riemannian_step_metrics.clipping_scale,
                            step,
                        )
                        tb_writer.scalar(
                            "optimizer/riemannian/tangent_step_norm",
                            riemannian_step_metrics.tangent_step_norm,
                            step,
                        )
                        tb_writer.scalar(
                            "optimizer/riemannian/momentum_norm",
                            riemannian_step_metrics.momentum_norm,
                            step,
                        )
                        tb_writer.scalar(
                            "optimizer/riemannian/constraint_residual_max",
                            riemannian_constraint_residual,
                            step,
                        )
                        tb_writer.scalar(
                            "optimizer/riemannian/line_search_evaluations",
                            riemannian_step_metrics.line_search_evaluations,
                            step,
                        )
                        tb_writer.scalar(
                            "optimizer/riemannian/line_search_accepted",
                            riemannian_step_metrics.line_search_accepted,
                            step,
                        )
                        tb_writer.scalar(
                            "optimizer/riemannian/line_search_reduction",
                            riemannian_step_metrics.line_search_reduction,
                            step,
                        )
                        tb_writer.scalar(
                            "optimizer/riemannian/conjugacy_beta",
                            riemannian_step_metrics.conjugacy_beta,
                            step,
                        )
                        tb_writer.scalar(
                            "optimizer/riemannian/history_pair_count",
                            riemannian_step_metrics.history_pair_count,
                            step,
                        )
                        tb_writer.scalar(
                            "optimizer/riemannian/tangent_residual",
                            riemannian_step_metrics.tangent_residual,
                            step,
                        )
                        tb_writer.scalar(
                            "optimizer/riemannian/transported_tangent_residual",
                            riemannian_step_metrics.transported_tangent_residual,
                            step,
                        )
                        tb_writer.scalar(
                            "optimizer/riemannian/transport_metric_distortion",
                            riemannian_step_metrics.transport_metric_distortion,
                            step,
                        )
                    if step % tb_flush_every_ == 0:
                        tb_writer.flush()
            except (KeyboardInterrupt, InterruptedError) as exc:
                signal_guard.request_stop_from_exception(exc)
                _log_training_signal_stop(
                    optimizer_label,
                    signal_guard,
                    completed=completed,
                    total=int(num_iter),
                    file=out_file,
                )
                break

        current_evaluation_params = resolve_evaluation_parameters(
            evaluation_parameters,
            opt_state,
            params,
        )
        chosen = (
            control.selected(current_evaluation_params)
            if keep_best
            else current_evaluation_params
        )
        riemannian_diagnostics: dict[str, Any] = {}
        if _opt_riemannian is not None:
            geometry = _opt_riemannian.parameter_geometry
            if not bool(geometry.contains(chosen)):
                raise ValueError(
                    "Returned parameters are outside their declared ParameterGeometry."
                )
            final_metrics = _opt_riemannian.step_metrics(opt_state)
            riemannian_diagnostics = {
                "optimizer/riemannian/num_manifold_leaves": jnp.asarray(
                    geometry.num_manifold_leaves
                ),
                "optimizer/riemannian/learning_rate": final_metrics.learning_rate,
                "optimizer/riemannian/gradient_norm": final_metrics.gradient_norm,
                "optimizer/riemannian/clipping_scale": final_metrics.clipping_scale,
                "optimizer/riemannian/tangent_step_norm": (
                    final_metrics.tangent_step_norm
                ),
                "optimizer/riemannian/momentum_norm": final_metrics.momentum_norm,
                "optimizer/riemannian/constraint_residual_max": (
                    geometry.maximum_constraint_residual(chosen)
                ),
                "optimizer/riemannian/tangent_residual": (final_metrics.tangent_residual),
                "optimizer/riemannian/transported_tangent_residual": (
                    final_metrics.transported_tangent_residual
                ),
                "optimizer/riemannian/transport_metric_distortion": (
                    final_metrics.transport_metric_distortion
                ),
                "optimizer/riemannian/line_search_evaluations": (
                    final_metrics.line_search_evaluations
                ),
                "optimizer/riemannian/line_search_accepted": (
                    final_metrics.line_search_accepted
                ),
                "optimizer/riemannian/line_search_reduction": (
                    final_metrics.line_search_reduction
                ),
                "optimizer/riemannian/conjugacy_beta": (final_metrics.conjugacy_beta),
                "optimizer/riemannian/history_pair_count": (
                    final_metrics.history_pair_count
                ),
            }
        functions = combine_trainable(chosen, non_trainable)
        settle_started = time.perf_counter() if profile_adaptive else 0.0
        collocation = _settle_collocation(
            self,
            functions,
            collocation,
            key=jr.fold_in(control.key, 991),
            iter_=completed + 1,
        )
        if profile_adaptive:
            jax.block_until_ready(collocation)
            refresh_wall_time += time.perf_counter() - settle_started
        result = eqx.tree_at(lambda s: s.functions, self, functions)
        result = eqx.tree_at(lambda s: s.collocation, result, collocation)
        diagnostics = frozendict(
            {
                "profile_enabled": jnp.asarray(profile_adaptive),
                "refresh_wall_time_seconds": jnp.asarray(refresh_wall_time),
                "optimizer_wall_time_seconds": jnp.asarray(optimizer_wall_time),
                "optimizer_first_step_wall_time_seconds": jnp.asarray(
                    first_optimizer_step_wall_time
                ),
                "optimizer_steady_step_wall_time_seconds": jnp.asarray(
                    steady_optimizer_step_wall_time / max(completed - 1, 1)
                ),
            }
            | riemannian_diagnostics
        )
        return eqx.tree_at(lambda s: s.training_diagnostics, result, diagnostics)


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

    algo_params = algo.default_params

    def _loss_for_params(
        p,
        non_trainable_,
        solver,
        key,
        iter_,
        populations,
        materialized_batches,
    ):
        functions = combine_trainable(p, non_trainable_)
        if solver.enforcement is None:
            enforced = functions
        else:
            enforced = solver.enforcement.apply(functions)
        term_keys = jr.split(key, len(solver.terms))
        total = jnp.array(0.0, dtype=float)
        with derivative_runtime_context():
            for term, population, batch, term_key in zip(
                solver.terms,
                populations,
                materialized_batches,
                term_keys,
                strict=True,
            ):
                value = _term_value(
                    term,
                    population,
                    batch,
                    enforced,
                    key=term_key,
                    iter_=iter_,
                )
                total = total + jnp.asarray(value, dtype=float).reshape(())
            for value in function_model_loss_values(
                functions,
                key=jr.fold_in(key, len(solver.terms)),
                iter_=iter_,
            ):
                total = total + jnp.asarray(value, dtype=float).reshape(())
        return total

    def _values_for_params(
        p,
        non_trainable_,
        solver,
        key,
        iter_,
        populations,
        materialized_batches,
    ):
        functions = combine_trainable(p, non_trainable_)
        if solver.enforcement is None:
            enforced = functions
        else:
            enforced = solver.enforcement.apply(functions)
        term_keys = jr.split(key, len(solver.terms))
        values: list[jax.Array] = []
        with derivative_runtime_context():
            for term, population, batch, term_key in zip(
                solver.terms,
                populations,
                materialized_batches,
                term_keys,
                strict=True,
            ):
                value = _term_value(
                    term,
                    population,
                    batch,
                    enforced,
                    key=term_key,
                    iter_=iter_,
                )
                values.append(jnp.asarray(value, dtype=float).reshape(()))
            for value in function_model_loss_values(
                functions,
                key=jr.fold_in(key, len(solver.terms)),
                iter_=iter_,
            ):
                values.append(jnp.asarray(value, dtype=float).reshape(()))
        if values:
            return jnp.stack(values, axis=0)
        return jnp.zeros((0,), dtype=float)

    def _enforced_functions_for_params(p, non_trainable_, solver):
        functions = combine_trainable(p, non_trainable_)
        if solver.enforcement is None:
            return functions
        return solver.enforcement.apply(functions)

    def _evaluation_term_values_for_params(
        p, non_trainable_, solver, requested_terms, key, iter_
    ):
        enforced = _enforced_functions_for_params(p, non_trainable_, solver)
        keys = jr.split(key, len(requested_terms))
        batches = _sample_term_batches(
            requested_terms,
            (None,) * len(requested_terms),
            key=jr.fold_in(key, 1),
        )
        values: list[jax.Array] = []
        with derivative_runtime_context():
            for term, batch, term_key in zip(requested_terms, batches, keys, strict=True):
                value = _term_value(
                    term,
                    None,
                    batch,
                    enforced,
                    key=term_key,
                    iter_=iter_,
                )
                values.append(jnp.asarray(value, dtype=float).reshape(()))
        if values:
            return jnp.stack(values, axis=0)
        return jnp.zeros((0,), dtype=float)

    def _data_metrics_for_terms(p, non_trainable_, solver, requested_terms, key, iter_):
        enforced = _enforced_functions_for_params(p, non_trainable_, solver)
        keys = jr.split(key, len(requested_terms))
        metrics: list[dict[str, Any]] = []
        with derivative_runtime_context():
            for term, term_key in zip(requested_terms, keys, strict=True):
                if isinstance(term, _SupportsDataMetrics):
                    metrics.append(term.data_metrics(enforced, key=term_key, iter_=iter_))
                else:
                    metrics.append({})
        return tuple(metrics)

    loss_fn = eqx.filter_jit(_loss_for_params) if jit else _loss_for_params
    terms_fn = eqx.filter_jit(_values_for_params) if jit else _values_for_params

    key = jr.key(seed)
    evo_state = algo.init(key, mean=params, params=algo_params)

    control = TrainingController(
        total_steps=int(num_iter),
        key=key,
        progress=TrainingProgress(best_value=float("inf")),
    )
    control.best_payload = params
    collocation = self.collocation

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
                population, evo_state = algo.ask(ask_key, evo_state, algo_params)
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
                collocation = _refresh_collocation(
                    self,
                    functions_snapshot,
                    collocation,
                    key=jr.fold_in(eval_key, 101),
                    iter_=epoch + 1,
                )
                if profile_adaptive:
                    jax.block_until_ready(collocation)
                    refresh_wall_time += time.perf_counter() - refresh_started
                optimizer_started = time.perf_counter() if profile_adaptive else 0.0

                # Common random numbers: sample every stochastic term once per
                # generation and reuse the materialized batches across the population.
                batch_key = jr.fold_in(eval_key, 0)
                materialized_batches = _sample_term_batches(
                    self.terms,
                    collocation,
                    key=batch_key,
                )

                eval_key_shared = jr.fold_in(eval_key, 1)
                losses = jax.vmap(
                    lambda p: loss_fn(
                        p,
                        non_trainable,
                        self,
                        eval_key_shared,
                        iter_,
                        collocation,
                        materialized_batches,
                    )
                )(population)
                evo_state, _ = algo.tell(
                    tell_key, population, losses, evo_state, algo_params
                )
                cand_params = algo.get_mean(evo_state)
                cand_loss = loss_fn(
                    cand_params,
                    non_trainable,
                    self,
                    eval_key_shared,
                    iter_,
                    collocation,
                    materialized_batches,
                )
                if profile_adaptive:
                    jax.block_until_ready((evo_state, cand_params, cand_loss))
                    optimizer_wall_time += time.perf_counter() - optimizer_started
                collocation = _record_collocation_training_evaluations(
                    self,
                    collocation,
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
                            self,
                            eval_key_shared,
                            iter_,
                            collocation,
                            materialized_batches,
                        ),
                        dtype=float,
                    )
                    train_term_values = values_arr[: len(term_names)]
                    train_model_loss_terms = values_arr[len(term_names) :]
                    train_data_metrics = _data_metrics_for_terms(
                        cand_params,
                        non_trainable,
                        self,
                        self.terms,
                        jr.fold_in(eval_key_shared, 1),
                        iter_,
                    )
                    eval_terms = _evaluation_term_values_for_params(
                        cand_params,
                        non_trainable,
                        self,
                        self.evaluation_terms,
                        jr.fold_in(eval_key_shared, 2),
                        iter_,
                    )
                    eval_data_metrics = _data_metrics_for_terms(
                        cand_params,
                        non_trainable,
                        self,
                        self.evaluation_terms,
                        jr.fold_in(eval_key_shared, 3),
                        iter_,
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
        collocation = _settle_collocation(
            self,
            functions,
            collocation,
            key=jr.fold_in(control.key, 991),
            iter_=completed + 1,
        )
        if profile_adaptive:
            jax.block_until_ready(collocation)
            refresh_wall_time += time.perf_counter() - settle_started
        result = eqx.tree_at(lambda s: s.functions, self, functions)
        result = eqx.tree_at(lambda s: s.collocation, result, collocation)
        diagnostics = frozendict(
            {
                "profile_enabled": jnp.asarray(profile_adaptive),
                "refresh_wall_time_seconds": jnp.asarray(refresh_wall_time),
                "optimizer_wall_time_seconds": jnp.asarray(optimizer_wall_time),
            }
        )
        return eqx.tree_at(lambda s: s.training_diagnostics, result, diagnostics)
