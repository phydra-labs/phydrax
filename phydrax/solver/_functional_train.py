#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import inspect
from contextlib import nullcontext
from pathlib import Path
from typing import Any, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jax import core as jcore


if TYPE_CHECKING:
    from ._functional_solver import FunctionalSolver


def _constraint_label(constraint: Any, /) -> str:
    label = getattr(constraint, "label", None)
    if label:
        return str(label)
    return type(constraint).__name__


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
) -> "FunctionalSolver":
    if num_iter == 0:
        return self

    if isinstance(optim, str):
        raise TypeError(
            "optim must be an optimizer object (e.g. optax.adam(...), optax.lbfgs(...), "
            "or an evosax algorithm instance), not a string."
        )

    log_ctx = (
        open(Path(log_path), "w", encoding="utf-8")
        if log_path is not None
        else nullcontext(None)
    )

    with log_ctx as log_fp:
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
            )

        params, static = eqx.partition(self.functions, eqx.is_inexact_array)
        log_every_ = int(log_every)
        if log_every_ < 0:
            raise ValueError("log_every must be >= 0.")
        log_constraints_ = bool(log_constraints)
        constraint_names = tuple(_constraint_label(c) for c in self.constraints)

        def _loss_wrt_params(params_, solver, key, iter_):
            functions = eqx.combine(params_, static)
            if solver.constraint_pipelines is None:
                enforced = functions
            else:
                enforced = solver.constraint_pipelines.apply(functions)
            keys = jr.split(key, len(solver.constraints))
            total = jnp.array(0.0, dtype=float)
            if not log_constraints_:
                for c, k in zip(solver.constraints, keys, strict=True):
                    term = c.loss(enforced, key=k, iter_=iter_)
                    total = total + jnp.asarray(term, dtype=float).reshape(())
                return total, jnp.zeros((0,), dtype=float)

            terms: list[jax.Array] = []
            for c, k in zip(solver.constraints, keys, strict=True):
                term = c.loss(enforced, key=k, iter_=iter_)
                term = jnp.asarray(term, dtype=float).reshape(())
                terms.append(term)
                total = total + term
            if terms:
                return total, jnp.stack(terms, axis=0)
            return total, jnp.zeros((0,), dtype=float)

        loss_fn = eqx.filter_value_and_grad(_loss_wrt_params, has_aux=True)

        is_linesearch = _opt_linesearch is not None

        def solve_step(params_, opt_state, solver, key, iter_):
            if is_linesearch:
                import jax.tree_util as jtu

                def _value_fn(p):
                    return _loss_wrt_params(p, solver, key, iter_)[0]

                (value, _terms0), grads = loss_fn(params_, solver, key, iter_)
                grads = jtu.tree_map(
                    lambda a: jnp.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
                    if eqx.is_inexact_array(a)
                    else a,
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
                loss_val, terms = _loss_wrt_params(params_, solver, key, iter_)
                return params_, opt_state, loss_val, terms

            (loss_val, terms), grads = loss_fn(params_, solver, key, iter_)
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
            rng, subkey = jr.split(rng)
            iter_ = jnp.asarray(epoch + 1, dtype=float)
            params, opt_state, loss_val, terms = solve_step(
                params, opt_state, self, subkey, iter_
            )
            if keep_best:
                loss_f = float(loss_val)
                if loss_f < best_loss:
                    best_loss = loss_f
                    best_params = params
            if log_every_ > 0 and ((epoch + 1) % log_every_ == 0):
                loss_f = float(loss_val)
                best_display = best_loss if keep_best else loss_f
                print(
                    f"[phydrax][optax] iter {epoch + 1}/{int(num_iter)} "
                    f"loss={loss_f:.6e} best={best_display:.6e}",
                    file=out_file,
                )
                if log_constraints_:
                    terms_arr = jnp.asarray(terms, dtype=float)
                    for i, (name, val) in enumerate(
                        zip(constraint_names, list(map(float, terms_arr)), strict=True)
                    ):
                        print(f"  [{i}] {name}: {val:.6e}", file=out_file)

        chosen = best_params if keep_best else params
        functions = eqx.combine(chosen, static)
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
) -> "FunctionalSolver":
    from ..constraints._base import AbstractSamplingConstraint

    params, static = eqx.partition(self.functions, eqx.is_inexact_array)
    log_every_ = int(log_every)
    if log_every_ < 0:
        raise ValueError("log_every must be >= 0.")
    log_constraints_ = bool(log_constraints)
    constraint_names = tuple(_constraint_label(c) for c in self.constraints)

    algo_params = algo.default_params

    def _loss_for_params(p, solver, key, iter_, batches):
        functions = eqx.combine(p, static)
        if solver.constraint_pipelines is None:
            enforced = functions
        else:
            enforced = solver.constraint_pipelines.apply(functions)
        keys = jr.split(key, len(solver.constraints))
        total = jnp.array(0.0, dtype=float)
        for c, k, b in zip(solver.constraints, keys, batches, strict=True):
            if b is None:
                total = total + c.loss(enforced, key=k, iter_=iter_)
            else:
                total = total + c.loss(enforced, key=k, iter_=iter_, batch=b)
        return total

    def _terms_for_params(p, solver, key, iter_, batches):
        functions = eqx.combine(p, static)
        if solver.constraint_pipelines is None:
            enforced = functions
        else:
            enforced = solver.constraint_pipelines.apply(functions)
        keys = jr.split(key, len(solver.constraints))
        terms: list[jax.Array] = []
        for c, k, b in zip(solver.constraints, keys, batches, strict=True):
            if b is None:
                term = c.loss(enforced, key=k, iter_=iter_)
            else:
                term = c.loss(enforced, key=k, iter_=iter_, batch=b)
            terms.append(jnp.asarray(term, dtype=float).reshape(()))
        if terms:
            return jnp.stack(terms, axis=0)
        return jnp.zeros((0,), dtype=float)

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

    with log_ctx as log_fp:
        out_file = log_fp if log_fp is not None else None

        for epoch in range(int(num_iter)):
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
                lambda p: loss_fn(p, self, eval_key_shared, iter_, batches_tuple)
            )(population)
            evo_state, _ = algo.tell(tell_key, population, losses, evo_state, algo_params)
            cand_params = algo.get_mean(evo_state)
            cand_loss = loss_fn(cand_params, self, eval_key_shared, iter_, batches_tuple)

            if keep_best:
                loss_f = float(cand_loss)
                if loss_f < best_loss:
                    best_loss = loss_f
                    best_params = cand_params
            else:
                best_params = cand_params
            if log_every_ > 0 and ((epoch + 1) % log_every_ == 0):
                loss_f = float(cand_loss)
                best_display = best_loss if keep_best else loss_f
                print(
                    f"[phydrax][evosax] iter {epoch + 1}/{int(num_iter)} "
                    f"loss={loss_f:.6e} best={best_display:.6e}",
                    file=out_file,
                )
                if log_constraints_:
                    terms_arr = jnp.asarray(
                        terms_fn(
                            cand_params, self, eval_key_shared, iter_, batches_tuple
                        ),
                        dtype=float,
                    )
                    for i, (name, val) in enumerate(
                        zip(constraint_names, list(map(float, terms_arr)), strict=True)
                    ):
                        print(f"  [{i}] {name}: {val:.6e}", file=out_file)

        functions = eqx.combine(best_params, static)
        return eqx.tree_at(lambda s: s.functions, self, functions)
