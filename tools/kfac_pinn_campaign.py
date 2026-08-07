#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jax.flatten_util import ravel_pytree

import phydrax as phx
from phydrax._trainable import combine_trainable, partition_trainable
from phydrax.constraints import FunctionalConstraint
from phydrax.operators.differential import laplacian, partial_n
from phydrax.solver._kfac_problem import (
    frozen_loss,
    frozen_loss_and_flat_gradient,
    materialize_constraint_terms,
    term_residual_jacobians,
)


CASES = (
    "poisson-1d",
    "poisson-2d",
    "heat-1d",
    "burgers-1d",
    "poisson-100d",
    "coupled-1d",
    "inverse-1d",
)
OPTIMIZERS = ("kfac-expand", "kfac-reduce", "adam", "lbfgs", "exact-ggn")


def _network(domain, *, in_size, width, depth, key):
    model = phx.nn.MLP(
        in_size=in_size,
        out_size="scalar",
        hidden_sizes=(int(width),) * int(depth),
        activation=jnp.tanh,
        rwf=False,
        key=key,
    )
    return domain.Model(*domain.labels)(model)


def _fixed_constraint(component, operator, variables, *, samples, key, weight=1.0):
    return FunctionalConstraint.from_operator(
        component=component,
        operator=operator,
        constraint_vars=variables,
        sampling=phx.domain.PointSampling(
            int(samples),
            layout=phx.domain.SampleLayout((component.domain.labels,)),
        ),
        sampling_mode="fixed",
        fixed_batch_key=key,
        weight=weight,
    )


def _poisson_1d(width, depth, samples, key):
    domain = phx.domain.Interval1d(-1.0, 1.0)
    u = _network(domain, in_size=1, width=width, depth=depth, key=key)

    @domain.Function("x")
    def forcing(x):
        return (jnp.pi**2) * jnp.sin(jnp.pi * x[0])

    residual = _fixed_constraint(
        domain.component(),
        lambda field: laplacian(field, var="x") + forcing,
        "u",
        samples=samples,
        key=jr.fold_in(key, 1),
    )
    boundary_component = domain.component({"x": phx.domain.Boundary()})
    boundary = _fixed_constraint(
        boundary_component,
        lambda field: field,
        "u",
        samples=max(4, samples // 2),
        key=jr.fold_in(key, 2),
        weight=5.0,
    )
    return phx.solver.FunctionalSolver(
        functions={"u": u}, constraints=(residual, boundary)
    )


def _poisson_2d(width, depth, samples, key):
    domain = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )
    u = _network(domain, in_size=2, width=width, depth=depth, key=key)
    residual = _fixed_constraint(
        domain.component(),
        lambda field: laplacian(field, var="x") - 4.0,
        "u",
        samples=samples,
        key=jr.fold_in(key, 1),
    )
    return phx.solver.FunctionalSolver(functions={"u": u}, constraints=residual)


def _spacetime_problem(width, depth, samples, key, *, burgers):
    domain = phx.domain.Interval1d(0.0, 1.0) @ phx.domain.TimeInterval(0.0, 1.0)
    u = _network(domain, in_size=2, width=width, depth=depth, key=key)
    if burgers:

        @domain.Function("x", "t")
        def forcing(x, t):
            return 1.0 + x[0] + t

        def operator(field):
            return (
                partial_n(field, var="t", order=1)
                + field * partial_n(field, var="x", order=1)
                - 0.1 * partial_n(field, var="x", order=2)
                - forcing
            )
    else:

        @domain.Function("x", "t")
        def forcing(x, t):
            return (jnp.pi**2 - 1.0) * jnp.sin(jnp.pi * x[0]) * jnp.exp(-t)

        def operator(field):
            return (
                partial_n(field, var="t", order=1) - laplacian(field, var="x") - forcing
            )

    residual = _fixed_constraint(
        domain.component(),
        operator,
        "u",
        samples=samples,
        key=jr.fold_in(key, 1),
    )
    return phx.solver.FunctionalSolver(functions={"u": u}, constraints=residual)


def _poisson_high_dim(width, depth, samples, key, *, dimension=100):
    factors = [
        phx.domain.Interval1d(-1.0, 1.0).relabel(f"x{index}")
        for index in range(int(dimension))
    ]
    domain = factors[0]
    for factor in factors[1:]:
        domain = domain @ factor
    u = _network(domain, in_size=dimension, width=width, depth=depth, key=key)

    def operator(field):
        total = sum(
            (partial_n(field, var=label, order=2) for label in domain.labels),
            field * 0.0,
        )
        return total - 2.0 * float(dimension)

    residual = _fixed_constraint(
        domain.component(),
        operator,
        "u",
        samples=samples,
        key=jr.fold_in(key, 1),
    )
    return phx.solver.FunctionalSolver(functions={"u": u}, constraints=residual)


def _coupled(width, depth, samples, key):
    domain = phx.domain.Interval1d(-1.0, 1.0)
    key_u, key_v = jr.split(key)
    u = _network(domain, in_size=1, width=width, depth=depth, key=key_u)
    v = _network(domain, in_size=1, width=width, depth=depth, key=key_v)

    @domain.Function("x")
    def first_forcing(x):
        return 2.0 + x[0]

    first = _fixed_constraint(
        domain.component(),
        lambda u_field, v_field: laplacian(u_field, var="x") + v_field - first_forcing,
        ("u", "v"),
        samples=samples,
        key=jr.fold_in(key, 1),
    )
    second = _fixed_constraint(
        domain.component(),
        lambda u_field, v_field: u_field - v_field,
        ("u", "v"),
        samples=samples,
        key=jr.fold_in(key, 2),
    )
    return phx.solver.FunctionalSolver(
        functions={"u": u, "v": v}, constraints=(first, second)
    )


def _inverse(width, depth, samples, key):
    domain = phx.domain.Interval1d(0.0, 1.0)
    u = _network(domain, in_size=1, width=width, depth=depth, key=key)
    coefficient = domain.Parameter(0.5)

    @domain.Function("x")
    def state_target(x):
        return x[0]

    @domain.Function("x")
    def equation_target(x):
        return 2.0 * x[0]

    state = _fixed_constraint(
        domain.component(),
        lambda field: field - state_target,
        "u",
        samples=samples,
        key=jr.fold_in(key, 1),
    )
    equation = _fixed_constraint(
        domain.component(),
        lambda field, parameter: parameter * field - equation_target,
        ("u", "coefficient"),
        samples=samples,
        key=jr.fold_in(key, 2),
    )
    return phx.solver.FunctionalSolver(
        functions={"u": u, "coefficient": coefficient},
        constraints=(state, equation),
    )


def make_solver(case, *, width, depth, samples, seed):
    key = jr.key(int(seed))
    if case == "poisson-1d":
        return _poisson_1d(width, depth, samples, key)
    if case == "poisson-2d":
        return _poisson_2d(width, depth, samples, key)
    if case == "heat-1d":
        return _spacetime_problem(width, depth, samples, key, burgers=False)
    if case == "burgers-1d":
        return _spacetime_problem(width, depth, samples, key, burgers=True)
    if case == "poisson-100d":
        return _poisson_high_dim(width, depth, samples, key)
    if case == "coupled-1d":
        return _coupled(width, depth, samples, key)
    if case == "inverse-1d":
        return _inverse(width, depth, samples, key)
    raise ValueError(f"Unknown benchmark case {case!r}.")


def _solve_exact_ggn(solver, *, steps, seed, damping=1e-3):
    params, non_trainable = partition_trainable(solver.functions)
    step_times: list[float] = []
    for step in range(int(steps)):
        step_started = time.perf_counter()
        key = jr.fold_in(jr.key(seed), step)
        terms = materialize_constraint_terms(
            solver.constraints,
            solver.collocation,
            key=key,
        )
        loss, gradient, unravel = frozen_loss_and_flat_gradient(
            params,
            non_trainable,
            solver,
            terms,
            objective_key=jr.fold_in(key, 1),
            iter_=step + 1,
        )
        flat, jacobians, _ = term_residual_jacobians(
            params,
            non_trainable,
            solver,
            terms,
            iter_=step + 1,
        )
        curvature = float(damping) * jnp.eye(flat.size, dtype=flat.dtype)
        for jacobian in jacobians:
            curvature = curvature + jacobian.T @ jacobian
        direction = jnp.linalg.solve(curvature, gradient)
        step_size = 1.0
        for _ in range(10):
            candidate = flat - step_size * direction
            candidate_loss = frozen_loss(
                unravel(candidate),
                non_trainable,
                solver,
                terms,
                objective_key=jr.fold_in(key, 1),
                iter_=step + 1,
            )
            if bool(candidate_loss < loss):
                flat = candidate
                break
            step_size *= 0.5
        params = unravel(flat)
        jax.block_until_ready(params)
        step_times.append(time.perf_counter() - step_started)
    result = eqx.tree_at(
        lambda item: item.functions,
        solver,
        combine_trainable(params, non_trainable),
    )
    first_step = step_times[0] if step_times else 0.0
    steady_step = (
        sum(step_times[1:]) / len(step_times[1:]) if len(step_times) > 1 else 0.0
    )
    return result, first_step, steady_step


def _peak_device_memory_bytes():
    statistics = jax.devices()[0].memory_stats()
    if statistics is None:
        return None
    value = statistics.get("peak_bytes_in_use")
    return None if value is None else int(value)


def _sampling_evaluations(sampling):
    if isinstance(sampling, tuple):
        return sum(_sampling_evaluations(item) for item in sampling)
    counts = (
        (int(sampling.count),)
        if isinstance(sampling.count, int)
        else tuple(int(count) for count in sampling.count)
    )
    total = 1
    for count in counts:
        total *= count
    return total


def _diagnostic_float(diagnostics, name):
    return float(diagnostics[name]) if name in diagnostics else None


def _run_one(args):
    samples = min(args.samples, 8) if args.smoke else args.samples
    solver = make_solver(
        args.case,
        width=args.width,
        depth=args.depth,
        samples=samples,
        seed=args.seed,
    )
    initial_parameters, _ = ravel_pytree(solver.trainable_functions())
    if args.optimizer == "exact-ggn" and int(initial_parameters.size) > 512:
        raise ValueError(
            "The explicit exact-GGN baseline is limited to at most 512 parameters."
        )
    initial_loss = float(solver.loss(key=jr.key(args.seed + 1)))
    started = time.perf_counter()
    external_first_step_time = None
    external_steady_step_time = None
    if args.optimizer == "kfac-expand":
        optimizer = phx.optim.kfac(
            approximation="expand",
            damping=args.damping,
            factor_update_period=args.factor_update_period,
            factor_chunk_size=args.factor_chunk_size,
        )
        trained = solver.solve(
            num_iter=args.steps,
            optim=optimizer,
            seed=args.seed,
            log_every=0,
            keep_best=False,
            profile_adaptive=True,
        )
    elif args.optimizer == "kfac-reduce":
        optimizer = phx.optim.kfac(
            approximation="reduce",
            damping=args.damping,
            factor_update_period=args.factor_update_period,
            factor_chunk_size=args.factor_chunk_size,
        )
        trained = solver.solve(
            num_iter=args.steps,
            optim=optimizer,
            seed=args.seed,
            log_every=0,
            keep_best=False,
            profile_adaptive=True,
        )
    elif args.optimizer == "adam":
        trained = solver.solve(
            num_iter=args.steps,
            optim=optax.adam(1e-3),
            seed=args.seed,
            log_every=0,
            keep_best=False,
            profile_adaptive=True,
        )
    elif args.optimizer == "lbfgs":
        trained = solver.solve(
            num_iter=args.steps,
            optim=optax.lbfgs(),
            seed=args.seed,
            log_every=0,
            keep_best=False,
            jit=False,
            profile_adaptive=True,
        )
    else:
        (
            trained,
            external_first_step_time,
            external_steady_step_time,
        ) = _solve_exact_ggn(
            solver,
            steps=args.steps,
            seed=args.seed,
            damping=args.damping,
        )
    jax.block_until_ready(trained.functions)
    elapsed = time.perf_counter() - started
    final_loss = float(trained.loss(key=jr.key(args.seed + 1)))
    parameters, _ = ravel_pytree(trained.trainable_functions())
    diagnostics = trained.training_diagnostics
    first_step_time = _diagnostic_float(
        diagnostics,
        "optimizer/kfac/first_step_wall_time_seconds",
    )
    steady_step_time = _diagnostic_float(
        diagnostics,
        "optimizer/kfac/steady_step_wall_time_seconds",
    )
    if first_step_time is None:
        first_step_time = _diagnostic_float(
            diagnostics,
            "optimizer_first_step_wall_time_seconds",
        )
    if steady_step_time is None:
        steady_step_time = _diagnostic_float(
            diagnostics,
            "optimizer_steady_step_wall_time_seconds",
        )
    if first_step_time is None:
        first_step_time = external_first_step_time
    if steady_step_time is None:
        steady_step_time = external_steady_step_time
    comparable_step_schedule = (
        not args.optimizer.startswith("kfac-") or int(args.factor_update_period) == 1
    )
    first_step_overhead = (
        max(first_step_time - steady_step_time, 0.0)
        if first_step_time is not None
        and steady_step_time is not None
        and int(args.steps) > 1
        and comparable_step_schedule
        else None
    )
    result = {
        "case": args.case,
        "optimizer": args.optimizer,
        "steps": int(args.steps),
        "width": int(args.width),
        "depth": int(args.depth),
        "samples": int(samples),
        "seed": int(args.seed),
        "damping": float(args.damping),
        "factor_update_period": int(args.factor_update_period),
        "factor_chunk_size": int(args.factor_chunk_size),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "relative_loss": final_loss / max(initial_loss, 1e-30),
        "wall_time_seconds": elapsed,
        "parameter_count": int(parameters.size),
        "peak_device_memory_bytes": _peak_device_memory_bytes(),
        "first_step_wall_time_seconds": first_step_time,
        "steady_step_wall_time_seconds": steady_step_time,
        "first_step_overhead_seconds": first_step_overhead,
        "gradient_wall_time_seconds": _diagnostic_float(
            diagnostics,
            "optimizer/kfac/gradient_wall_time_seconds",
        ),
        "factor_wall_time_seconds": _diagnostic_float(
            diagnostics,
            "optimizer/kfac/factor_wall_time_seconds",
        ),
        "linear_solve_wall_time_seconds": _diagnostic_float(
            diagnostics,
            "optimizer/kfac/linear_solve_wall_time_seconds",
        ),
        "line_search_wall_time_seconds": _diagnostic_float(
            diagnostics,
            "optimizer/kfac/line_search_wall_time_seconds",
        ),
        "factor_updates": _diagnostic_float(
            diagnostics,
            "optimizer/kfac/factor_updates",
        ),
        "factor_condition_estimate_max": _diagnostic_float(
            diagnostics,
            "optimizer/kfac/factor_condition_estimate_max",
        ),
        "collocation_point_steps": int(args.steps)
        * sum(_sampling_evaluations(term.sampling) for term in solver.constraints),
    }
    print(json.dumps(result, sort_keys=True))


def _isolated_command(args, *, case, optimizer, width, depth, seed):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--case",
        str(case),
        "--optimizer",
        str(optimizer),
        "--steps",
        str(args.steps),
        "--width",
        str(width),
        "--depth",
        str(depth),
        "--samples",
        str(args.samples),
        "--seed",
        str(seed),
        "--damping",
        str(args.damping),
        "--factor-update-period",
        str(args.factor_update_period),
        "--factor-chunk-size",
        str(args.factor_chunk_size),
    ]
    if args.smoke:
        command.append("--smoke")
    return command


def run(args):
    if int(args.steps) < 0:
        raise ValueError("steps must be nonnegative.")
    if int(args.width) <= 0 or (
        args.widths is not None and any(int(width) <= 0 for width in args.widths)
    ):
        raise ValueError("network widths must be positive.")
    if int(args.depth) < 0 or (
        args.depths is not None and any(int(depth) < 0 for depth in args.depths)
    ):
        raise ValueError("network depths must be nonnegative.")
    if int(args.samples) <= 0:
        raise ValueError("samples must be positive.")

    cases = (args.case,) if args.cases is None else tuple(args.cases)
    optimizers = (args.optimizer,) if args.optimizers is None else tuple(args.optimizers)
    widths = (args.width,) if args.widths is None else tuple(args.widths)
    depths = (args.depth,) if args.depths is None else tuple(args.depths)
    seeds = (args.seed,) if args.seeds is None else tuple(args.seeds)
    configurations = tuple(product(cases, optimizers, widths, depths, seeds))
    if len(configurations) == 1:
        case, optimizer, width, depth, seed = configurations[0]
        local_args = argparse.Namespace(
            **{
                **vars(args),
                "case": case,
                "optimizer": optimizer,
                "width": width,
                "depth": depth,
                "seed": seed,
            }
        )
        _run_one(local_args)
        return

    for case, optimizer, width, depth, seed in configurations:
        subprocess.run(
            _isolated_command(
                args,
                case=case,
                optimizer=optimizer,
                width=width,
                depth=depth,
                seed=seed,
            ),
            check=True,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Phydrax PINN optimizers.")
    parser.add_argument("--case", choices=CASES, default="poisson-1d")
    parser.add_argument("--optimizer", choices=OPTIMIZERS, default="kfac-expand")
    parser.add_argument("--cases", nargs="+", choices=CASES)
    parser.add_argument("--optimizers", nargs="+", choices=OPTIMIZERS)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--widths", nargs="+", type=int)
    parser.add_argument("--depths", nargs="+", type=int)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--damping", type=float, default=1e-3)
    parser.add_argument("--factor-update-period", type=int, default=1)
    parser.add_argument("--factor-chunk-size", type=int, default=32)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
