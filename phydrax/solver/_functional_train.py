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
from .._objective import AbstractSamplingObjectiveTerm
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
from ..constraints._adaptive_control import ControlledCollocationPolicy
from ..constraints._functional import FunctionalConstraint
from ..operators.differential._runtime import derivative_runtime_context
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


def _sample_objective_batches(objectives: tuple[Any, ...], /, *, key: Any):
    keys = jr.split(key, len(objectives))
    return tuple(
        objective.sample(key=objective_key)
        if isinstance(objective, AbstractSamplingObjectiveTerm)
        else None
        for objective, objective_key in zip(objectives, keys, strict=True)
    )


def _constraint_label(constraint: Any, /) -> str:
    label = getattr(constraint, "label", None)
    if label:
        return str(label)
    return type(constraint).__name__


def _clean_tag_part(value: str, /) -> str:
    cleaned = "".join(
        ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value)
    ).strip("_")
    return cleaned or "constraint"


def _constraint_tag(index: int, name: str, /) -> str:
    return f"constraints/{index:03d}_{_clean_tag_part(name)}"


def _model_loss_tag(index: int, name: str, /) -> str:
    return f"model_losses/{index:03d}_{_clean_tag_part(name)}"


def _objective_tag(index: int, name: str, /) -> str:
    return f"objectives/{index:03d}_{_clean_tag_part(name)}"


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


def _train_constraint_sample_size(
    value: int | None,
    /,
    *,
    num_constraints: int,
) -> int | None:
    if value is None:
        return None
    n_constraints = int(num_constraints)
    if n_constraints <= 0:
        raise ValueError(
            "train_constraint_sample_size requires at least one training constraint."
        )
    sample_size = int(value)
    if sample_size <= 0:
        raise ValueError("train_constraint_sample_size must be positive.")
    if sample_size >= n_constraints:
        return None
    return sample_size


def _active_train_constraints(
    constraints: tuple[Any, ...],
    /,
    *,
    sample_size: int | None,
    key: Any,
) -> tuple[tuple[Any, ...], tuple[int, ...], Any]:
    n_constraints = len(constraints)
    if sample_size is None:
        return constraints, tuple(range(n_constraints)), jnp.asarray(1.0, dtype=float)
    sampled = jr.choice(
        key,
        n_constraints,
        shape=(int(sample_size),),
        replace=False,
    )
    active_indices = tuple(int(i) for i in np.asarray(sampled, dtype=np.int32))
    active = tuple(constraints[i] for i in active_indices)
    scale = jnp.asarray(n_constraints / int(sample_size), dtype=float)
    return active, active_indices, scale


def _expanded_train_terms(
    active_terms: Any,
    /,
    *,
    active_constraint_indices: tuple[int, ...],
    num_constraints: int,
) -> Any:
    active_arr = jnp.asarray(active_terms, dtype=float).reshape((-1,))
    if int(active_arr.shape[0]) == 0:
        return jnp.zeros((int(num_constraints),), dtype=float)
    out = jnp.full((int(num_constraints),), jnp.nan, dtype=float)
    for local_i, constraint_i in enumerate(active_constraint_indices):
        out = out.at[int(constraint_i)].set(active_arr[int(local_i)])
    return out


def _write_constraint_tensorboard_scalars(
    writer: _TensorBoardLogger,
    *,
    step: int,
    namespace: str,
    constraint_names: tuple[str, ...],
    terms: Any,
    data_metrics: tuple[dict[str, Any], ...],
    write_legacy: bool = False,
) -> None:
    terms_arr = jnp.asarray(terms, dtype=float)
    for i, (name, val) in enumerate(
        zip(constraint_names, list(map(float, terms_arr)), strict=True)
    ):
        base = _constraint_tag(i, name)
        prefix = f"{namespace}/{base}"
        writer.scalar(f"{prefix}/loss", val, step)
        if write_legacy:
            writer.scalar(f"{base}/loss", val, step)
        for metric_name, metric_value in data_metrics[i].items():
            writer.scalar(f"{prefix}/{metric_name}", metric_value, step)
            if write_legacy:
                writer.scalar(f"{base}/{metric_name}", metric_value, step)


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


def _write_objective_tensorboard_scalars(
    writer: _TensorBoardLogger,
    *,
    step: int,
    objective_names: tuple[str, ...],
    terms: Any,
) -> None:
    terms_arr = jnp.asarray(terms, dtype=float)
    for i, (name, val) in enumerate(
        zip(objective_names, list(map(float, terms_arr)), strict=True)
    ):
        writer.scalar(f"train/{_objective_tag(i, name)}/value", val, step)


def _log_tensorboard_scalars(
    writer: _TensorBoardLogger,
    *,
    step: int,
    loss: Any,
    best_loss: float,
    evaluation_loss: Any | None,
    iter_time_s: float,
    train_constraint_names: tuple[str, ...],
    train_terms: Any,
    train_data_metrics: tuple[dict[str, Any], ...],
    train_objective_names: tuple[str, ...],
    train_objective_terms: Any,
    train_model_loss_names: tuple[str, ...],
    train_model_loss_terms: Any,
    eval_constraint_names: tuple[str, ...],
    eval_terms: Any,
    eval_data_metrics: tuple[dict[str, Any], ...],
    log_constraints: bool,
) -> None:
    writer.scalar("train/loss", loss, step)
    writer.scalar("train/best_loss", best_loss, step)
    if evaluation_loss is not None:
        writer.scalar("eval/loss", evaluation_loss, step)
    writer.scalar("train/iter_time_s", iter_time_s, step)

    if not log_constraints:
        return

    _write_constraint_tensorboard_scalars(
        writer,
        step=step,
        namespace="train",
        constraint_names=train_constraint_names,
        terms=train_terms,
        data_metrics=train_data_metrics,
        write_legacy=True,
    )
    _write_objective_tensorboard_scalars(
        writer,
        step=step,
        objective_names=train_objective_names,
        terms=train_objective_terms,
    )
    _write_model_loss_tensorboard_scalars(
        writer,
        step=step,
        model_loss_names=train_model_loss_names,
        terms=train_model_loss_terms,
    )
    _write_constraint_tensorboard_scalars(
        writer,
        step=step,
        namespace="eval",
        constraint_names=eval_constraint_names,
        terms=eval_terms,
        data_metrics=eval_data_metrics,
    )


def _adaptive_constraint_loss(
    constraint,
    population: Any | None,
    functions,
    /,
    *,
    key,
    iter_,
):
    if population is None:
        return constraint.loss(functions, key=key, iter_=iter_)
    if not isinstance(constraint, FunctionalConstraint):
        raise TypeError("Adaptive collocation is only valid for FunctionalConstraint.")
    policy = constraint.collocation_policy
    if policy is None:
        raise ValueError("Adaptive population requires a collocation policy.")
    batch, batch_weight = policy.loss_batch_and_weight(population)
    return constraint.loss(
        functions,
        key=key,
        iter_=iter_,
        batch=batch,
        batch_weight=batch_weight,
    )


def _refresh_collocation(
    solver,
    functions,
    collocation: tuple[Any | None, ...],
    /,
    *,
    key,
    iter_,
) -> tuple[Any | None, ...]:
    if solver.constraint_pipelines is None:
        enforced = functions
    else:
        enforced = solver.constraint_pipelines.apply(functions)
    keys = jr.split(key, len(solver.constraints))
    refreshed: list[Any | None] = []
    for constraint, population, constraint_key in zip(
        solver.constraints, collocation, keys, strict=True
    ):
        if population is None:
            refreshed.append(None)
            continue
        if not isinstance(constraint, FunctionalConstraint):
            raise TypeError(
                "Adaptive collocation is only valid for FunctionalConstraint."
            )
        policy = constraint.collocation_policy
        if policy is None:
            raise ValueError("Adaptive population requires a collocation policy.")
        if bool(policy.should_refresh(population, iter_)):
            population = policy.refresh(
                constraint,
                enforced,
                population,
                key=constraint_key,
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
    if solver.constraint_pipelines is None:
        enforced = functions
    else:
        enforced = solver.constraint_pipelines.apply(functions)
    keys = jr.split(key, len(solver.constraints))
    settled: list[Any | None] = []
    for constraint, population, constraint_key in zip(
        solver.constraints, collocation, keys, strict=True
    ):
        if population is None:
            settled.append(None)
            continue
        if not isinstance(constraint, FunctionalConstraint):
            raise TypeError(
                "Adaptive collocation is only valid for FunctionalConstraint."
            )
        policy = constraint.collocation_policy
        if isinstance(policy, ControlledCollocationPolicy):
            population = policy.settle(
                constraint,
                enforced,
                population,
                key=constraint_key,
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
    constraint_indices: tuple[int, ...] | None = None,
) -> tuple[Any | None, ...]:
    selected = (
        None
        if constraint_indices is None
        else frozenset(int(index) for index in constraint_indices)
    )
    recorded: list[Any | None] = []
    for index, (constraint, population) in enumerate(
        zip(solver.constraints, collocation, strict=True)
    ):
        if population is None or (selected is not None and index not in selected):
            recorded.append(population)
            continue
        if not isinstance(constraint, FunctionalConstraint):
            raise TypeError(
                "Adaptive collocation is only valid for FunctionalConstraint."
            )
        policy = constraint.collocation_policy
        if policy is None:
            raise ValueError("Adaptive population requires a collocation policy.")
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
    for constraint, population in zip(
        solver.constraints,
        collocation,
        strict=True,
    ):
        if population is None:
            metrics.append({})
            continue
        if not isinstance(constraint, FunctionalConstraint):
            raise TypeError(
                "Adaptive collocation is only valid for FunctionalConstraint."
            )
        policy = constraint.collocation_policy
        if policy is None:
            raise ValueError("Adaptive population requires a collocation policy.")
        metrics.append(policy.data_metrics(population))
    return tuple(metrics)


def solve(
    self: "FunctionalSolver",
    *,
    num_iter: int,
    optim: optax.GradientTransformation
    | optax.GradientTransformationExtraArgs
    | Any = optax.rprop(1e-3),
    evaluation_parameters: EvaluationParametersFn | None = None,
    seed: int = 0,
    jit: bool = True,
    keep_best: bool = True,
    log_every: int = 1,
    log_constraints: bool = True,
    log_path: str | Path | None = None,
    tensorboard_log_dir: str | Path | None = None,
    tensorboard_every: int | None = None,
    tensorboard_flush_every: int = 10,
    profile_adaptive: bool = False,
    train_constraint_sample_size: int | None = None,
) -> "FunctionalSolver":
    if num_iter == 0:
        return self

    if isinstance(optim, str):
        raise TypeError(
            "optim must be an optimizer object (e.g. optax.adam(...), optax.lbfgs(...), "
            "or an evosax distribution-based algorithm instance), not a string."
        )

    _opt_linesearch: optax.GradientTransformationExtraArgs | None = None
    _opt_standard: optax.GradientTransformation | None = None

    if isinstance(optim, optax.GradientTransformationExtraArgs):
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
            log_constraints=log_constraints,
            log_path=log_path,
            tensorboard_log_dir=tensorboard_log_dir,
            tensorboard_every=tensorboard_every,
            tensorboard_flush_every=tensorboard_flush_every,
            profile_adaptive=profile_adaptive,
            train_constraint_sample_size=train_constraint_sample_size,
        )
    else:
        raise TypeError(
            "optim must be an Optax transformation or an Evosax distribution-based "
            "algorithm instance."
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
        log_constraints_ = bool(log_constraints)
        constraint_names = tuple(_constraint_label(c) for c in self.constraints)
        objective_names = tuple(_constraint_label(term) for term in self.objectives)
        model_loss_names = function_model_loss_labels(self.functions)
        eval_constraint_names = tuple(_constraint_label(c) for c in self.eval_constraints)
        constraint_sample_size = _train_constraint_sample_size(
            train_constraint_sample_size,
            num_constraints=len(self.constraints),
        )

        def _loss_wrt_params(
            params_,
            non_trainable_,
            solver,
            constraints,
            constraint_scale,
            key,
            iter_,
            collocation_,
            objective_batches_,
        ):
            functions = combine_trainable(params_, non_trainable_)
            if solver.constraint_pipelines is None:
                enforced = functions
            else:
                enforced = solver.constraint_pipelines.apply(functions)
            num_constraints = len(constraints)
            num_objectives = len(solver.objectives)
            num_terms = num_constraints + num_objectives
            keys = jr.split(key, num_terms)
            constraint_keys = keys[:num_constraints]
            objective_keys = keys[num_constraints:]
            total = jnp.array(0.0, dtype=float)
            scale = jnp.asarray(constraint_scale, dtype=float).reshape(())
            with derivative_runtime_context():
                if not log_constraints_:
                    for c, population, k in zip(
                        constraints, collocation_, constraint_keys, strict=True
                    ):
                        term = _adaptive_constraint_loss(
                            c, population, enforced, key=k, iter_=iter_
                        )
                        total = total + scale * jnp.asarray(term, dtype=float).reshape(())
                    for objective, objective_key, objective_batch in zip(
                        solver.objectives,
                        objective_keys,
                        objective_batches_,
                        strict=True,
                    ):
                        if objective_batch is None:
                            term = objective.loss(
                                enforced, key=objective_key, iter_=iter_
                            )
                        else:
                            term = objective.loss(
                                enforced,
                                key=objective_key,
                                iter_=iter_,
                                batch=objective_batch,
                            )
                        total = total + jnp.asarray(term, dtype=float).reshape(())
                    for term in function_model_loss_values(
                        functions,
                        key=jr.fold_in(key, num_terms),
                        iter_=iter_,
                    ):
                        total = total + jnp.asarray(term, dtype=float).reshape(())
                    return total, jnp.zeros((0,), dtype=float)

                terms: list[jax.Array] = []
                for c, population, k in zip(
                    constraints, collocation_, constraint_keys, strict=True
                ):
                    term = _adaptive_constraint_loss(
                        c, population, enforced, key=k, iter_=iter_
                    )
                    raw_term = jnp.asarray(term, dtype=float).reshape(())
                    scaled_term = scale * raw_term
                    terms.append(scaled_term)
                    total = total + scaled_term
                for objective, objective_key, objective_batch in zip(
                    solver.objectives,
                    objective_keys,
                    objective_batches_,
                    strict=True,
                ):
                    if objective_batch is None:
                        term = objective.loss(enforced, key=objective_key, iter_=iter_)
                    else:
                        term = objective.loss(
                            enforced,
                            key=objective_key,
                            iter_=iter_,
                            batch=objective_batch,
                        )
                    term = jnp.asarray(term, dtype=float).reshape(())
                    terms.append(term)
                    total = total + term
                for term in function_model_loss_values(
                    functions,
                    key=jr.fold_in(key, num_terms),
                    iter_=iter_,
                ):
                    term = jnp.asarray(term, dtype=float).reshape(())
                    terms.append(term)
                    total = total + term
                if terms:
                    return total, jnp.stack(terms, axis=0)
                return total, jnp.zeros((0,), dtype=float)

        def _enforced_functions_wrt_params(params_, non_trainable_, solver):
            functions = combine_trainable(params_, non_trainable_)
            if solver.constraint_pipelines is None:
                return functions
            return solver.constraint_pipelines.apply(functions)

        def _terms_wrt_constraints(
            params_,
            non_trainable_,
            solver,
            constraints,
            key,
            iter_,
        ):
            enforced = _enforced_functions_wrt_params(params_, non_trainable_, solver)
            keys = jr.split(key, len(constraints))
            terms: list[jax.Array] = []
            with derivative_runtime_context():
                for c, k in zip(constraints, keys, strict=True):
                    term = c.loss(enforced, key=k, iter_=iter_)
                    terms.append(jnp.asarray(term, dtype=float).reshape(()))
            if terms:
                return jnp.stack(terms, axis=0)
            return jnp.zeros((0,), dtype=float)

        def _data_metrics_wrt_constraints(
            params_,
            non_trainable_,
            solver,
            constraints,
            key,
            iter_,
        ):
            enforced = _enforced_functions_wrt_params(params_, non_trainable_, solver)
            keys = jr.split(key, len(constraints))
            metrics: list[dict[str, Any]] = []
            with derivative_runtime_context():
                for c, k in zip(constraints, keys, strict=True):
                    if isinstance(c, _SupportsDataMetrics):
                        metrics.append(c.data_metrics(enforced, key=k, iter_=iter_))
                    else:
                        metrics.append({})
            return tuple(metrics)

        loss_fn = eqx.filter_value_and_grad(_loss_wrt_params, has_aux=True)

        is_linesearch = _opt_linesearch is not None

        def solve_step_constraints(
            params_,
            non_trainable_,
            opt_state,
            solver,
            constraints,
            constraint_scale,
            key,
            iter_,
            collocation_,
            objective_batches_,
        ):
            if is_linesearch:
                import jax.tree_util as jtu

                def _value_fn(p):
                    return _loss_wrt_params(
                        p,
                        non_trainable_,
                        solver,
                        constraints,
                        constraint_scale,
                        key,
                        iter_,
                        collocation_,
                        objective_batches_,
                    )[0]

                (value, _terms0), grads = loss_fn(
                    params_,
                    non_trainable_,
                    solver,
                    constraints,
                    constraint_scale,
                    key,
                    iter_,
                    collocation_,
                    objective_batches_,
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
                loss_val, terms = _loss_wrt_params(
                    params_,
                    non_trainable_,
                    solver,
                    constraints,
                    constraint_scale,
                    key,
                    iter_,
                    collocation_,
                    objective_batches_,
                )
                return params_, opt_state, loss_val, terms

            (loss_val, terms), grads = loss_fn(
                params_,
                non_trainable_,
                solver,
                constraints,
                constraint_scale,
                key,
                iter_,
                collocation_,
                objective_batches_,
            )
            assert _opt_standard is not None
            updates, opt_state = _opt_standard.update(grads, opt_state, params_)
            params_ = eqx.apply_updates(params_, updates)
            return params_, opt_state, loss_val, terms

        solve_step = (
            eqx.filter_jit(solve_step_constraints)
            if jit and not is_linesearch
            else solve_step_constraints
        )
        selection_loss_fn = (
            eqx.filter_jit(_loss_wrt_params)
            if jit and evaluation_parameters is not None
            else _loss_wrt_params
        )

        opt = _opt_linesearch if is_linesearch else _opt_standard
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
                    "optax",
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
                active_constraints, active_constraint_indices, constraint_scale = (
                    _active_train_constraints(
                        self.constraints,
                        sample_size=constraint_sample_size,
                        key=jr.fold_in(subkey, 17),
                    )
                )
                active_collocation = tuple(
                    collocation[index] for index in active_constraint_indices
                )
                objective_batches = _sample_objective_batches(
                    self.objectives,
                    key=jr.fold_in(subkey, 211),
                )
                pre_update_params = params
                params, opt_state, loss_val, terms = solve_step(
                    params,
                    non_trainable,
                    opt_state,
                    self,
                    active_constraints,
                    constraint_scale,
                    subkey,
                    iter_,
                    active_collocation,
                    objective_batches,
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
                    constraint_indices=active_constraint_indices,
                )
                completed = epoch + 1
                terms_arr = jnp.asarray(terms, dtype=float)
                active_constraint_count = len(active_constraints)
                train_constraint_terms = _expanded_train_terms(
                    terms_arr[:active_constraint_count],
                    active_constraint_indices=active_constraint_indices,
                    num_constraints=len(constraint_names),
                )
                objective_count = len(objective_names)
                train_objective_terms = terms_arr[
                    active_constraint_count : active_constraint_count + objective_count
                ]
                train_model_loss_terms = terms_arr[
                    active_constraint_count + objective_count :
                ]
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
                            params if is_linesearch else pre_update_params
                        )
                        selection_loss = loss_val
                    else:
                        evaluation_loss, _ = selection_loss_fn(
                            current_evaluation_params,
                            non_trainable,
                            self,
                            active_constraints,
                            constraint_scale,
                            subkey,
                            iter_,
                            active_collocation,
                            objective_batches,
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
                        "optax",
                        signal_guard,
                        completed=step,
                        total=int(num_iter),
                        file=out_file,
                    )
                    break
                console_step = log_every_ > 0 and (step % log_every_ == 0)
                tensorboard_step = tb_every_ is not None and (step % tb_every_ == 0)
                train_data_metrics: tuple[dict[str, Any], ...] = tuple(
                    {} for _ in self.constraints
                )
                eval_terms = jnp.zeros((0,), dtype=float)
                eval_data_metrics: tuple[dict[str, Any], ...] = tuple(
                    {} for _ in self.eval_constraints
                )
                if log_constraints_ and (console_step or tensorboard_step):
                    train_data_metrics = _data_metrics_wrt_constraints(
                        current_evaluation_params,
                        non_trainable,
                        self,
                        self.constraints,
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
                    eval_terms = _terms_wrt_constraints(
                        current_evaluation_params,
                        non_trainable,
                        self,
                        self.eval_constraints,
                        jr.fold_in(subkey, 2),
                        iter_,
                    )
                    eval_data_metrics = _data_metrics_wrt_constraints(
                        current_evaluation_params,
                        non_trainable,
                        self,
                        self.eval_constraints,
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
                    print(
                        f"[phydrax][optax] iter {step}/{int(num_iter)} "
                        f"loss={loss_f:.6e}{evaluation_suffix} "
                        f"best={best_display:.6e} iter_time={iter_time_s:.3f}s",
                        file=out_file,
                    )
                    if log_constraints_:
                        for i, (name, val) in enumerate(
                            zip(
                                constraint_names,
                                list(map(float, train_constraint_terms)),
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
                                objective_names,
                                list(map(float, train_objective_terms)),
                                strict=True,
                            )
                        ):
                            print(
                                f"  [objective {i}] {name}: {val:.6e}",
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
                                eval_constraint_names,
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
                        train_constraint_names=constraint_names,
                        train_terms=train_constraint_terms,
                        train_data_metrics=train_data_metrics,
                        train_objective_names=objective_names,
                        train_objective_terms=train_objective_terms,
                        train_model_loss_names=model_loss_names,
                        train_model_loss_terms=train_model_loss_terms,
                        eval_constraint_names=eval_constraint_names,
                        eval_terms=eval_terms,
                        eval_data_metrics=eval_data_metrics,
                        log_constraints=log_constraints_,
                    )
                    if step % tb_flush_every_ == 0:
                        tb_writer.flush()
            except (KeyboardInterrupt, InterruptedError) as exc:
                signal_guard.request_stop_from_exception(exc)
                _log_training_signal_stop(
                    "optax",
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
    log_constraints: bool,
    log_path: str | Path | None,
    tensorboard_log_dir: str | Path | None = None,
    tensorboard_every: int | None = None,
    tensorboard_flush_every: int = 10,
    profile_adaptive: bool = False,
    tensorboard_writer: _TensorBoardLogger | None = None,
    train_constraint_sample_size: int | None = None,
) -> "FunctionalSolver":
    from ..constraints._base import AbstractSamplingConstraint

    if train_constraint_sample_size is not None:
        raise NotImplementedError(
            "train_constraint_sample_size is currently supported only for Optax "
            "optimizers."
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
    log_constraints_ = bool(log_constraints)
    constraint_names = tuple(_constraint_label(c) for c in self.constraints)
    objective_names = tuple(_constraint_label(term) for term in self.objectives)
    model_loss_names = function_model_loss_labels(self.functions)
    eval_constraint_names = tuple(_constraint_label(c) for c in self.eval_constraints)

    algo_params = algo.default_params

    def _loss_for_params(
        p,
        non_trainable_,
        solver,
        key,
        iter_,
        batches,
        objective_batches,
    ):
        functions = combine_trainable(p, non_trainable_)
        if solver.constraint_pipelines is None:
            enforced = functions
        else:
            enforced = solver.constraint_pipelines.apply(functions)
        num_constraints = len(solver.constraints)
        num_objectives = len(solver.objectives)
        num_terms = num_constraints + num_objectives
        keys = jr.split(key, num_terms)
        constraint_keys = keys[:num_constraints]
        objective_keys = keys[num_constraints:]
        total = jnp.array(0.0, dtype=float)
        with derivative_runtime_context():
            for c, k, batch_info in zip(
                solver.constraints, constraint_keys, batches, strict=True
            ):
                if batch_info is None:
                    total = total + c.loss(enforced, key=k, iter_=iter_)
                else:
                    batch, batch_weight = batch_info
                    if batch_weight is None:
                        total = total + c.loss(enforced, key=k, iter_=iter_, batch=batch)
                    else:
                        total = total + c.loss(
                            enforced,
                            key=k,
                            iter_=iter_,
                            batch=batch,
                            batch_weight=batch_weight,
                        )
            for objective, objective_key, objective_batch in zip(
                solver.objectives,
                objective_keys,
                objective_batches,
                strict=True,
            ):
                if objective_batch is None:
                    term = objective.loss(enforced, key=objective_key, iter_=iter_)
                else:
                    term = objective.loss(
                        enforced,
                        key=objective_key,
                        iter_=iter_,
                        batch=objective_batch,
                    )
                total = total + jnp.asarray(term, dtype=float).reshape(())
            for term in function_model_loss_values(
                functions,
                key=jr.fold_in(key, num_terms),
                iter_=iter_,
            ):
                total = total + jnp.asarray(term, dtype=float).reshape(())
        return total

    def _terms_for_params(
        p,
        non_trainable_,
        solver,
        key,
        iter_,
        batches,
        objective_batches,
    ):
        functions = combine_trainable(p, non_trainable_)
        if solver.constraint_pipelines is None:
            enforced = functions
        else:
            enforced = solver.constraint_pipelines.apply(functions)
        num_constraints = len(solver.constraints)
        num_objectives = len(solver.objectives)
        num_terms = num_constraints + num_objectives
        keys = jr.split(key, num_terms)
        constraint_keys = keys[:num_constraints]
        objective_keys = keys[num_constraints:]
        terms: list[jax.Array] = []
        with derivative_runtime_context():
            for c, k, batch_info in zip(
                solver.constraints, constraint_keys, batches, strict=True
            ):
                if batch_info is None:
                    term = c.loss(enforced, key=k, iter_=iter_)
                else:
                    batch, batch_weight = batch_info
                    if batch_weight is None:
                        term = c.loss(enforced, key=k, iter_=iter_, batch=batch)
                    else:
                        term = c.loss(
                            enforced,
                            key=k,
                            iter_=iter_,
                            batch=batch,
                            batch_weight=batch_weight,
                        )
                terms.append(jnp.asarray(term, dtype=float).reshape(()))
            for objective, objective_key, objective_batch in zip(
                solver.objectives,
                objective_keys,
                objective_batches,
                strict=True,
            ):
                if objective_batch is None:
                    term = objective.loss(enforced, key=objective_key, iter_=iter_)
                else:
                    term = objective.loss(
                        enforced,
                        key=objective_key,
                        iter_=iter_,
                        batch=objective_batch,
                    )
                terms.append(jnp.asarray(term, dtype=float).reshape(()))
            for term in function_model_loss_values(
                functions,
                key=jr.fold_in(key, num_terms),
                iter_=iter_,
            ):
                terms.append(jnp.asarray(term, dtype=float).reshape(()))
        if terms:
            return jnp.stack(terms, axis=0)
        return jnp.zeros((0,), dtype=float)

    def _enforced_functions_for_params(p, non_trainable_, solver):
        functions = combine_trainable(p, non_trainable_)
        if solver.constraint_pipelines is None:
            return functions
        return solver.constraint_pipelines.apply(functions)

    def _terms_for_constraints(p, non_trainable_, solver, constraints, key, iter_):
        enforced = _enforced_functions_for_params(p, non_trainable_, solver)
        keys = jr.split(key, len(constraints))
        terms: list[jax.Array] = []
        with derivative_runtime_context():
            for c, k in zip(constraints, keys, strict=True):
                term = c.loss(enforced, key=k, iter_=iter_)
                terms.append(jnp.asarray(term, dtype=float).reshape(()))
        if terms:
            return jnp.stack(terms, axis=0)
        return jnp.zeros((0,), dtype=float)

    def _data_metrics_for_constraints(p, non_trainable_, solver, constraints, key, iter_):
        enforced = _enforced_functions_for_params(p, non_trainable_, solver)
        keys = jr.split(key, len(constraints))
        metrics: list[dict[str, Any]] = []
        with derivative_runtime_context():
            for c, k in zip(constraints, keys, strict=True):
                if isinstance(c, _SupportsDataMetrics):
                    metrics.append(c.data_metrics(enforced, key=k, iter_=iter_))
                else:
                    metrics.append({})
        return tuple(metrics)

    loss_fn = eqx.filter_jit(_loss_for_params) if jit else _loss_for_params
    terms_fn = eqx.filter_jit(_terms_for_params) if jit else _terms_for_params

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

                # Common random numbers (CRN): sample each constraint batch once per
                # generation and reuse it across the full population to reduce variance
                # and avoid vmapping through host callbacks.
                batch_key = jr.fold_in(eval_key, 0)
                batch_keys = jr.split(batch_key, len(self.constraints))
                batches: list[Any] = []
                for c, adaptive_population, k in zip(
                    self.constraints, collocation, batch_keys, strict=True
                ):
                    if adaptive_population is not None:
                        if not isinstance(c, FunctionalConstraint):
                            raise TypeError(
                                "Adaptive collocation is only valid for FunctionalConstraint."
                            )
                        policy = c.collocation_policy
                        if policy is None:
                            raise ValueError(
                                "Adaptive population requires a collocation policy."
                            )
                        batches.append(policy.loss_batch_and_weight(adaptive_population))
                    elif isinstance(c, AbstractSamplingConstraint):
                        batches.append((c.sample(key=k), None))
                    else:
                        batches.append(None)
                batches_tuple = tuple(batches)
                objective_batches_tuple = _sample_objective_batches(
                    self.objectives,
                    key=jr.fold_in(batch_key, 1),
                )

                eval_key_shared = jr.fold_in(eval_key, 1)
                losses = jax.vmap(
                    lambda p: loss_fn(
                        p,
                        non_trainable,
                        self,
                        eval_key_shared,
                        iter_,
                        batches_tuple,
                        objective_batches_tuple,
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
                    batches_tuple,
                    objective_batches_tuple,
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
                    {} for _ in self.constraints
                )
                eval_terms = jnp.zeros((0,), dtype=float)
                eval_data_metrics: tuple[dict[str, Any], ...] = tuple(
                    {} for _ in self.eval_constraints
                )
                terms_arr = jnp.zeros((0,), dtype=float)
                train_constraint_terms = terms_arr[: len(constraint_names)]
                objective_count = len(objective_names)
                train_objective_terms = terms_arr[
                    len(constraint_names) : len(constraint_names) + objective_count
                ]
                train_model_loss_terms = terms_arr[
                    len(constraint_names) + objective_count :
                ]
                if log_constraints_ and (console_step or tensorboard_step):
                    terms_arr = jnp.asarray(
                        terms_fn(
                            cand_params,
                            non_trainable,
                            self,
                            eval_key_shared,
                            iter_,
                            batches_tuple,
                            objective_batches_tuple,
                        ),
                        dtype=float,
                    )
                    train_constraint_terms = terms_arr[: len(constraint_names)]
                    train_objective_terms = terms_arr[
                        len(constraint_names) : len(constraint_names) + objective_count
                    ]
                    train_model_loss_terms = terms_arr[
                        len(constraint_names) + objective_count :
                    ]
                    train_data_metrics = _data_metrics_for_constraints(
                        cand_params,
                        non_trainable,
                        self,
                        self.constraints,
                        jr.fold_in(eval_key_shared, 1),
                        iter_,
                    )
                    eval_terms = _terms_for_constraints(
                        cand_params,
                        non_trainable,
                        self,
                        self.eval_constraints,
                        jr.fold_in(eval_key_shared, 2),
                        iter_,
                    )
                    eval_data_metrics = _data_metrics_for_constraints(
                        cand_params,
                        non_trainable,
                        self,
                        self.eval_constraints,
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
                    if log_constraints_:
                        for i, (name, val) in enumerate(
                            zip(
                                constraint_names,
                                list(map(float, train_constraint_terms)),
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
                                objective_names,
                                list(map(float, train_objective_terms)),
                                strict=True,
                            )
                        ):
                            print(
                                f"  [objective {i}] {name}: {val:.6e}",
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
                                eval_constraint_names,
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
                        train_constraint_names=constraint_names,
                        train_terms=train_constraint_terms,
                        train_data_metrics=train_data_metrics,
                        train_objective_names=objective_names,
                        train_objective_terms=train_objective_terms,
                        train_model_loss_names=model_loss_names,
                        train_model_loss_terms=train_model_loss_terms,
                        eval_constraint_names=eval_constraint_names,
                        eval_terms=eval_terms,
                        eval_data_metrics=eval_data_metrics,
                        log_constraints=log_constraints_,
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
