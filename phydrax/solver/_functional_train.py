#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import inspect
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Protocol, runtime_checkable, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jax import core as jcore

from .._trainable import combine_trainable, partition_trainable
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


def _constraint_label(constraint: Any, /) -> str:
    label = getattr(constraint, "label", None)
    if label:
        return str(label)
    return type(constraint).__name__


class _TensorBoardLogger:
    _event_cls: Any
    _summary_cls: Any
    _writer: Any

    def __init__(self, log_dir: str | Path):
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary
        from tensorboard.summary.writer.event_file_writer import EventFileWriter

        path = Path(log_dir)
        path.mkdir(parents=True, exist_ok=True)
        self._event_cls = Event
        self._summary_cls = Summary
        self._writer = EventFileWriter(str(path))

    def __enter__(self) -> "_TensorBoardLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.flush()
        self._writer.close()

    def scalar(self, tag: str, value: Any, step: int) -> None:
        summary = self._summary_cls(
            value=[
                self._summary_cls.Value(
                    tag=str(tag),
                    simple_value=float(jnp.asarray(value, dtype=float).reshape(())),
                )
            ]
        )
        event = self._event_cls(
            wall_time=time.time(),
            step=int(step),
            summary=summary,
        )
        self._writer.add_event(event)

    def flush(self) -> None:
        self._writer.flush()


def _clean_tag_part(value: str, /) -> str:
    cleaned = "".join(
        ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value)
    ).strip("_")
    return cleaned or "constraint"


def _constraint_tag(index: int, name: str, /) -> str:
    return f"constraints/{index:03d}_{_clean_tag_part(name)}"


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


def _tensorboard_every(
    *,
    tensorboard_log_dir: str | Path | None,
    tensorboard_every: int | None,
    log_every: int,
) -> int | None:
    if tensorboard_log_dir is None:
        return None
    if tensorboard_every is None:
        return log_every if log_every > 0 else 1
    every = int(tensorboard_every)
    if every <= 0:
        raise ValueError(
            "tensorboard_every must be positive when TensorBoard is enabled."
        )
    return every


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


def _log_tensorboard_scalars(
    writer: _TensorBoardLogger,
    *,
    step: int,
    loss: Any,
    best_loss: float,
    iter_time_s: float,
    train_constraint_names: tuple[str, ...],
    train_terms: Any,
    train_data_metrics: tuple[dict[str, Any], ...],
    train_model_loss_names: tuple[str, ...],
    train_model_loss_terms: Any,
    eval_constraint_names: tuple[str, ...],
    eval_terms: Any,
    eval_data_metrics: tuple[dict[str, Any], ...],
    log_constraints: bool,
) -> None:
    writer.scalar("train/loss", loss, step)
    writer.scalar("train/best_loss", best_loss, step)
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


def solve(
    self: "FunctionalSolver",
    *,
    num_iter: int,
    optim: optax.GradientTransformation
    | optax.GradientTransformationExtraArgs
    | Any = optax.rprop(1e-3),
    seed: int = 0,
    jit: bool = True,
    keep_best: bool = True,
    log_every: int = 1,
    log_constraints: bool = True,
    log_path: str | Path | None = None,
    tensorboard_log_dir: str | Path | None = None,
    tensorboard_every: int | None = None,
    tensorboard_flush_every: int = 10,
) -> "FunctionalSolver":
    if num_iter == 0:
        return self

    if isinstance(optim, str):
        raise TypeError(
            "optim must be an optimizer object (e.g. optax.adam(...), optax.lbfgs(...), "
            "or an evosax algorithm instance), not a string."
        )

    _opt_linesearch: optax.GradientTransformationExtraArgs | None = None
    _opt_standard: optax.GradientTransformation | None = None

    if isinstance(optim, optax.GradientTransformationExtraArgs):
        _opt_linesearch = optim
    elif isinstance(optim, optax.GradientTransformation):
        _opt_standard = optim
    else:
        return _solve_evosax(
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

    with log_ctx as log_fp, tb_ctx as tb_writer:
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
        model_loss_names = function_model_loss_labels(self.functions)
        eval_constraint_names = tuple(_constraint_label(c) for c in self.eval_constraints)

        def _loss_wrt_params(params_, non_trainable_, solver, key, iter_):
            functions = combine_trainable(params_, non_trainable_)
            if solver.constraint_pipelines is None:
                enforced = functions
            else:
                enforced = solver.constraint_pipelines.apply(functions)
            keys = jr.split(key, len(solver.constraints))
            total = jnp.array(0.0, dtype=float)
            with derivative_runtime_context():
                if not log_constraints_:
                    for c, k in zip(solver.constraints, keys, strict=True):
                        term = c.loss(enforced, key=k, iter_=iter_)
                        total = total + jnp.asarray(term, dtype=float).reshape(())
                    for term in function_model_loss_values(
                        functions,
                        key=jr.fold_in(key, len(solver.constraints)),
                        iter_=iter_,
                    ):
                        total = total + jnp.asarray(term, dtype=float).reshape(())
                    return total, jnp.zeros((0,), dtype=float)

                terms: list[jax.Array] = []
                for c, k in zip(solver.constraints, keys, strict=True):
                    term = c.loss(enforced, key=k, iter_=iter_)
                    term = jnp.asarray(term, dtype=float).reshape(())
                    terms.append(term)
                    total = total + term
                for term in function_model_loss_values(
                    functions,
                    key=jr.fold_in(key, len(solver.constraints)),
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

        def solve_step(params_, non_trainable_, opt_state, solver, key, iter_):
            if is_linesearch:
                import jax.tree_util as jtu

                def _value_fn(p):
                    return _loss_wrt_params(p, non_trainable_, solver, key, iter_)[0]

                (value, _terms0), grads = loss_fn(
                    params_, non_trainable_, solver, key, iter_
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
                    params_, non_trainable_, solver, key, iter_
                )
                return params_, opt_state, loss_val, terms

            (loss_val, terms), grads = loss_fn(
                params_, non_trainable_, solver, key, iter_
            )
            assert _opt_standard is not None
            updates, opt_state = _opt_standard.update(grads, opt_state, params_)
            params_ = eqx.apply_updates(params_, updates)
            return params_, opt_state, loss_val, terms

        if jit and not is_linesearch:
            solve_step = eqx.filter_jit(solve_step)

        opt = _opt_linesearch if is_linesearch else _opt_standard
        if opt is None:
            raise ValueError("Optimizer is not configured.")
        opt_state = opt.init(params)
        rng = jr.key(seed)

        best_loss = float("inf")
        best_params = params
        out_file = log_fp if log_fp is not None else None

        for epoch in range(int(num_iter)):
            iter_start = time.perf_counter()
            rng, subkey = jr.split(rng)
            iter_ = jnp.asarray(epoch + 1, dtype=float)
            params, opt_state, loss_val, terms = solve_step(
                params, non_trainable, opt_state, self, subkey, iter_
            )
            terms_arr = jnp.asarray(terms, dtype=float)
            train_constraint_terms = terms_arr[: len(constraint_names)]
            train_model_loss_terms = terms_arr[len(constraint_names) :]
            if keep_best:
                loss_f = float(loss_val)
                if loss_f < best_loss:
                    best_loss = loss_f
                    best_params = params
            iter_time_s = time.perf_counter() - iter_start
            step = epoch + 1
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
                    params,
                    non_trainable,
                    self,
                    self.constraints,
                    jr.fold_in(subkey, 1),
                    iter_,
                )
                eval_terms = _terms_wrt_constraints(
                    params,
                    non_trainable,
                    self,
                    self.eval_constraints,
                    jr.fold_in(subkey, 2),
                    iter_,
                )
                eval_data_metrics = _data_metrics_wrt_constraints(
                    params,
                    non_trainable,
                    self,
                    self.eval_constraints,
                    jr.fold_in(subkey, 3),
                    iter_,
                )

            if console_step:
                loss_f = float(loss_val)
                best_display = best_loss if keep_best else loss_f
                print(
                    f"[phydrax][optax] iter {step}/{int(num_iter)} "
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
                best_display = best_loss if keep_best else loss_f
                _log_tensorboard_scalars(
                    tb_writer,
                    step=step,
                    loss=loss_val,
                    best_loss=best_display,
                    iter_time_s=iter_time_s,
                    train_constraint_names=constraint_names,
                    train_terms=train_constraint_terms,
                    train_data_metrics=train_data_metrics,
                    train_model_loss_names=model_loss_names,
                    train_model_loss_terms=train_model_loss_terms,
                    eval_constraint_names=eval_constraint_names,
                    eval_terms=eval_terms,
                    eval_data_metrics=eval_data_metrics,
                    log_constraints=log_constraints_,
                )
                if step % tb_flush_every_ == 0:
                    tb_writer.flush()

        chosen = best_params if keep_best else params
        functions = combine_trainable(chosen, non_trainable)
        return eqx.tree_at(lambda s: s.functions, self, functions)


def _solve_evosax(
    self: "FunctionalSolver",
    *,
    num_iter: int,
    algo: Any,
    seed: int,
    jit: bool,
    keep_best: bool,
    log_every: int,
    log_constraints: bool,
    log_path: str | Path | None,
    tensorboard_log_dir: str | Path | None = None,
    tensorboard_every: int | None = None,
    tensorboard_flush_every: int = 10,
    tensorboard_writer: _TensorBoardLogger | None = None,
) -> "FunctionalSolver":
    from ..constraints._base import AbstractSamplingConstraint

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
    model_loss_names = function_model_loss_labels(self.functions)
    eval_constraint_names = tuple(_constraint_label(c) for c in self.eval_constraints)

    algo_params = algo.default_params

    def _loss_for_params(p, non_trainable_, solver, key, iter_, batches):
        functions = combine_trainable(p, non_trainable_)
        if solver.constraint_pipelines is None:
            enforced = functions
        else:
            enforced = solver.constraint_pipelines.apply(functions)
        keys = jr.split(key, len(solver.constraints))
        total = jnp.array(0.0, dtype=float)
        with derivative_runtime_context():
            for c, k, b in zip(solver.constraints, keys, batches, strict=True):
                if b is None:
                    total = total + c.loss(enforced, key=k, iter_=iter_)
                else:
                    total = total + c.loss(enforced, key=k, iter_=iter_, batch=b)
            for term in function_model_loss_values(
                functions,
                key=jr.fold_in(key, len(solver.constraints)),
                iter_=iter_,
            ):
                total = total + jnp.asarray(term, dtype=float).reshape(())
        return total

    def _terms_for_params(p, non_trainable_, solver, key, iter_, batches):
        functions = combine_trainable(p, non_trainable_)
        if solver.constraint_pipelines is None:
            enforced = functions
        else:
            enforced = solver.constraint_pipelines.apply(functions)
        keys = jr.split(key, len(solver.constraints))
        terms: list[jax.Array] = []
        with derivative_runtime_context():
            for c, k, b in zip(solver.constraints, keys, batches, strict=True):
                if b is None:
                    term = c.loss(enforced, key=k, iter_=iter_)
                else:
                    term = c.loss(enforced, key=k, iter_=iter_, batch=b)
                terms.append(jnp.asarray(term, dtype=float).reshape(()))
            for term in function_model_loss_values(
                functions,
                key=jr.fold_in(key, len(solver.constraints)),
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
    init_sig = inspect.signature(algo.init)
    init_params = init_sig.parameters
    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in init_params.values()
    )
    init_kwargs = {}
    if accepts_var_kwargs or ("params" in init_params):
        init_kwargs["params"] = algo_params
    if accepts_var_kwargs or ("mean" in init_params):
        init_kwargs["mean"] = params
    evo_state = algo.init(key, **init_kwargs)

    best_loss = float("inf")
    best_params = params

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

    with log_ctx as log_fp, tb_ctx as tb_writer:
        out_file = log_fp if log_fp is not None else None

        for epoch in range(int(num_iter)):
            iter_start = time.perf_counter()
            key, ask_key, eval_key, tell_key, cand_key = jr.split(key, 5)
            population, evo_state = algo.ask(ask_key, evo_state, algo_params)
            popsize = None
            for leaf in jax.tree_util.tree_leaves(population):
                if isinstance(leaf, (jax.Array, jcore.Tracer)) and len(leaf.shape) > 0:
                    popsize = int(leaf.shape[0])
                    break
            if popsize is None:
                raise ValueError(
                    "Could not infer population size from evosax population."
                )

            iter_ = jnp.asarray(epoch + 1, dtype=float)

            # Common random numbers (CRN): sample each constraint batch once per generation
            # and reuse it across the full population to reduce variance and avoid
            # vmapping through host callbacks.
            batch_key = jr.fold_in(eval_key, 0)
            batch_keys = jr.split(batch_key, len(self.constraints))
            batches: list[Any] = []
            for c, k in zip(self.constraints, batch_keys, strict=True):
                if isinstance(c, AbstractSamplingConstraint):
                    batches.append(c.sample(key=k))
                else:
                    batches.append(None)
            batches_tuple = tuple(batches)

            eval_key_shared = jr.fold_in(eval_key, 1)
            losses = jax.vmap(
                lambda p: loss_fn(
                    p,
                    non_trainable,
                    self,
                    eval_key_shared,
                    iter_,
                    batches_tuple,
                )
            )(population)
            evo_state, _ = algo.tell(tell_key, population, losses, evo_state, algo_params)
            cand_params = algo.get_mean(evo_state)
            cand_loss = loss_fn(
                cand_params,
                non_trainable,
                self,
                eval_key_shared,
                iter_,
                batches_tuple,
            )

            if keep_best:
                loss_f = float(cand_loss)
                if loss_f < best_loss:
                    best_loss = loss_f
                    best_params = cand_params
            else:
                best_params = cand_params
            iter_time_s = time.perf_counter() - iter_start
            step = epoch + 1
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
            train_model_loss_terms = terms_arr[len(constraint_names) :]
            if log_constraints_ and (console_step or tensorboard_step):
                terms_arr = jnp.asarray(
                    terms_fn(
                        cand_params,
                        non_trainable,
                        self,
                        eval_key_shared,
                        iter_,
                        batches_tuple,
                    ),
                    dtype=float,
                )
                train_constraint_terms = terms_arr[: len(constraint_names)]
                train_model_loss_terms = terms_arr[len(constraint_names) :]
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
                best_display = best_loss if keep_best else loss_f
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
                best_display = best_loss if keep_best else loss_f
                _log_tensorboard_scalars(
                    tb_writer,
                    step=step,
                    loss=cand_loss,
                    best_loss=best_display,
                    iter_time_s=iter_time_s,
                    train_constraint_names=constraint_names,
                    train_terms=train_constraint_terms,
                    train_data_metrics=train_data_metrics,
                    train_model_loss_names=model_loss_names,
                    train_model_loss_terms=train_model_loss_terms,
                    eval_constraint_names=eval_constraint_names,
                    eval_terms=eval_terms,
                    eval_data_metrics=eval_data_metrics,
                    log_constraints=log_constraints_,
                )
                if step % tb_flush_every_ == 0:
                    tb_writer.flush()

        functions = combine_trainable(best_params, non_trainable)
        return eqx.tree_at(lambda s: s.functions, self, functions)
