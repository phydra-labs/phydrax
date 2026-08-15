#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import platform
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy as jsp

import phydrax as phx


_METHODS = ("hard-quantile", "soft-quantile", "logsumexp", "hard-cvar")
_DTYPES = {"float32": jnp.float32, "float64": jnp.float64}


@dataclass(frozen=True)
class _Problem:
    name: str
    initial_parameters: jax.Array
    residuals: Callable[[jax.Array], jax.Array]
    physical_metrics: Callable[[jax.Array], dict[str, jax.Array]]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare training objectives for high empirical residual quantiles."
    )
    parser.add_argument("--methods", nargs="+", choices=_METHODS, default=_METHODS)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--quantile", type=float, default=0.9)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--sinkhorn-iterations", type=int, default=100)
    parser.add_argument("--logsumexp-temperature", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--dtype", choices=tuple(_DTYPES), default="float64")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser


def _validate(arguments: argparse.Namespace) -> None:
    if arguments.size < 8:
        raise ValueError("size must be at least eight.")
    if not 0.0 < arguments.quantile < 1.0:
        raise ValueError("quantile must lie strictly inside (0, 1).")
    if arguments.epsilon <= 0.0 or not jnp.isfinite(arguments.epsilon):
        raise ValueError("epsilon must be finite and positive.")
    if arguments.sinkhorn_iterations < 1:
        raise ValueError("sinkhorn-iterations must be positive.")
    if arguments.logsumexp_temperature <= 0.0:
        raise ValueError("logsumexp-temperature must be positive.")
    if arguments.learning_rate <= 0.0:
        raise ValueError("learning-rate must be positive.")
    if arguments.steps < 1:
        raise ValueError("steps must be positive.")


def _heavy_tailed_problem(size: int, dtype: jnp.dtype) -> _Problem:
    coordinate = jnp.linspace(-1.0, 1.0, size, dtype=dtype)
    design = jnp.stack(
        (
            jnp.ones_like(coordinate),
            coordinate,
            jnp.sin(jnp.pi * coordinate),
            jnp.cos(2.0 * jnp.pi * coordinate),
        ),
        axis=-1,
    )
    baseline = (
        0.15
        + 0.08 * jnp.sin(5.0 * coordinate)
        + 1.8 / (1.0 + (18.0 * (coordinate - 0.62)) ** 2)
        - 1.2 / (1.0 + (24.0 * (coordinate + 0.48)) ** 2)
    )

    def signed(parameters):
        return baseline + design @ parameters

    def residuals(parameters):
        return jnp.abs(signed(parameters))

    def metrics(parameters):
        errors = signed(parameters)
        return {
            "signed_bias": jnp.mean(errors),
            "residual_rms": jnp.sqrt(jnp.mean(errors**2)),
            "parameter_norm": jnp.linalg.norm(parameters),
        }

    return _Problem(
        "deterministic-heavy-tail",
        jnp.zeros((design.shape[-1],), dtype=dtype),
        residuals,
        metrics,
    )


def _pde_problem(size: int, dtype: jnp.dtype) -> _Problem:
    coordinate = jnp.linspace(0.0, 1.0, size + 2, dtype=dtype)[1:-1]
    mode = jnp.arange(1, 9, dtype=dtype)
    basis = jnp.sin(jnp.pi * coordinate[:, None] * mode[None, :])
    second_derivative = -((jnp.pi * mode) ** 2) * basis
    forcing = jnp.pi**2 * jnp.sin(jnp.pi * coordinate) + 16.0 * jnp.exp(
        -(((coordinate - 0.72) / 0.055) ** 2)
    )
    boundary_basis = jnp.sin(
        jnp.pi * jnp.asarray([[0.0], [1.0]], dtype=dtype) * mode[None, :]
    )

    def signed(parameters):
        return second_derivative @ parameters + forcing

    def residuals(parameters):
        return jnp.abs(signed(parameters))

    def metrics(parameters):
        equation_residual = signed(parameters)
        boundary_values = boundary_basis @ parameters
        return {
            "pde_residual_l2": jnp.sqrt(jnp.mean(equation_residual**2)),
            "pde_residual_linf": jnp.max(jnp.abs(equation_residual)),
            "boundary_linf": jnp.max(jnp.abs(boundary_values)),
        }

    return _Problem(
        "localized-poisson-residual",
        jnp.zeros((mode.shape[0],), dtype=dtype),
        residuals,
        metrics,
    )


def _ensemble_problem(size: int, dtype: jnp.dtype) -> _Problem:
    case_count = max(4, size // 4)
    member_count = 4
    coordinate = jnp.linspace(-1.0, 1.0, case_count, dtype=dtype)
    member = jnp.arange(member_count, dtype=dtype)
    true_mean = 0.4 - 0.7 * coordinate + 0.5 * coordinate**2
    scale = 0.08 + 0.5 * (coordinate + 1.0) ** 2
    noise = jnp.sin(
        1.7 + 2.3 * jnp.arange(case_count, dtype=dtype)[:, None] + 3.1 * member[None, :]
    )
    observations = true_mean[:, None] + scale[:, None] * noise
    design = jnp.stack(
        (jnp.ones_like(coordinate), coordinate, coordinate**2),
        axis=-1,
    )

    def signed(parameters):
        prediction = design @ parameters
        return observations - prediction[:, None]

    def residuals(parameters):
        return jnp.abs(signed(parameters)).reshape((-1,))

    def metrics(parameters):
        errors = signed(parameters)
        split = case_count // 2
        return {
            "ensemble_rmse": jnp.sqrt(jnp.mean(errors**2)),
            "low_variance_rmse": jnp.sqrt(jnp.mean(errors[:split] ** 2)),
            "high_variance_rmse": jnp.sqrt(jnp.mean(errors[split:] ** 2)),
        }

    return _Problem(
        "heteroskedastic-ensemble-output",
        jnp.zeros((design.shape[-1],), dtype=dtype),
        residuals,
        metrics,
    )


def _problems(size: int, dtype: jnp.dtype) -> tuple[_Problem, ...]:
    return (
        _heavy_tailed_problem(size, dtype),
        _pde_problem(size, dtype),
        _ensemble_problem(size, dtype),
    )


def _hard_cvar(residuals: jax.Array, quantile: float) -> jax.Array:
    threshold = jax.lax.stop_gradient(jnp.quantile(residuals, quantile))
    selected = residuals >= threshold
    count = jnp.maximum(jnp.sum(selected), 1)
    return jnp.sum(jnp.where(selected, residuals, 0.0)) / count


def _objective(
    method: str,
    problem: _Problem,
    *,
    quantile: float,
    epsilon: float,
    solver_iterations: int,
    logsumexp_temperature: float,
):
    solver = phx.transport.Sinkhorn(
        epsilon,
        max_iterations=solver_iterations,
        tolerance=1e-7,
        check_every=5,
        early_stop=False,
    )

    def objective(parameters):
        residuals = problem.residuals(parameters)
        if method == "hard-quantile":
            tail = jnp.quantile(residuals, quantile)
        elif method == "soft-quantile":
            tail = phx.transport.soft_quantile(
                residuals,
                quantile,
                solver=solver,
            )
        elif method == "logsumexp":
            temperature = jnp.asarray(logsumexp_temperature, dtype=residuals.dtype)
            tail = temperature * (
                jsp.special.logsumexp(residuals / temperature) - jnp.log(residuals.size)
            )
        else:
            tail = _hard_cvar(residuals, quantile)
        return tail + 1e-4 * jnp.sum(parameters**2)

    return objective


def _memory(compiled) -> dict[str, int | str]:
    analysis = compiled.memory_analysis()
    if analysis is None:
        return {"status": "unavailable"}
    return {
        "status": "available",
        "argument_bytes": int(analysis.argument_size_in_bytes),
        "output_bytes": int(analysis.output_size_in_bytes),
        "temporary_bytes": int(analysis.temp_size_in_bytes),
        "alias_bytes": int(analysis.alias_size_in_bytes),
    }


def _block(value: Any) -> Any:
    return jax.tree.map(lambda leaf: leaf.block_until_ready(), value)


def _adam_step(objective, learning_rate: float):
    beta1 = 0.9
    beta2 = 0.999
    stability = 1e-8

    def step(state):
        parameters, first_moment, second_moment, iteration = state
        value, gradient = jax.value_and_grad(objective)(parameters)
        next_iteration = iteration + 1
        first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
        second_moment = beta2 * second_moment + (1.0 - beta2) * gradient**2
        corrected_first = first_moment / (1.0 - beta1**next_iteration)
        corrected_second = second_moment / (1.0 - beta2**next_iteration)
        parameters = parameters - learning_rate * corrected_first / (
            jnp.sqrt(corrected_second) + stability
        )
        return (
            (
                parameters,
                first_moment,
                second_moment,
                next_iteration,
            ),
            value,
            gradient,
        )

    return step


def _transport_diagnostics(
    residuals: jax.Array,
    epsilon: float,
    solver_iterations: int,
) -> dict[str, Any]:
    solver = phx.transport.Sinkhorn(
        epsilon,
        max_iterations=solver_iterations,
        tolerance=1e-7,
        check_every=5,
        early_stop=False,
    )
    result = phx.transport.soft_order_transport(residuals, solver=solver)
    assert isinstance(result, phx.transport.SinkhornResult)
    return {
        "converged": bool(result.converged),
        "normalized_marginal_residual": float(
            result.diagnostics.normalized_marginal_residual
        ),
        "physical_marginal_residual": float(
            result.diagnostics.physical_marginal_residual
        ),
    }


def _sensitivity(
    residuals: jax.Array,
    quantile: float,
    epsilon: float,
    solver_iterations: int,
) -> dict[str, float]:
    values = {}
    for candidate in (0.5 * epsilon, epsilon, 2.0 * epsilon):
        solver = phx.transport.Sinkhorn(
            candidate,
            max_iterations=solver_iterations,
            tolerance=1e-7,
            check_every=5,
            early_stop=False,
        )
        estimate = phx.transport.soft_quantile(
            residuals,
            quantile,
            solver=solver,
        )
        values[f"{candidate:.8g}"] = float(estimate)
    return values


def _train(
    problem: _Problem,
    method: str,
    *,
    seed: int,
    quantile: float,
    epsilon: float,
    solver_iterations: int,
    logsumexp_temperature: float,
    learning_rate: float,
    steps: int,
) -> dict[str, Any]:
    key = jax.random.fold_in(jax.random.key(seed), len(problem.name))
    parameters = problem.initial_parameters + 0.02 * jax.random.normal(
        key,
        problem.initial_parameters.shape,
        dtype=problem.initial_parameters.dtype,
    )
    initial_parameters = parameters
    objective = _objective(
        method,
        problem,
        quantile=quantile,
        epsilon=epsilon,
        solver_iterations=solver_iterations,
        logsumexp_temperature=logsumexp_temperature,
    )
    step = _adam_step(objective, learning_rate)
    state = (
        parameters,
        jnp.zeros_like(parameters),
        jnp.zeros_like(parameters),
        jnp.asarray(0, dtype=jnp.int32),
    )

    started = time.perf_counter_ns()
    compiled = jax.jit(step).lower(state).compile()
    compile_ms = (time.perf_counter_ns() - started) / 1e6

    started = time.perf_counter_ns()
    state, value, gradient = _block(compiled(state))
    first_step_ms = (time.perf_counter_ns() - started) / 1e6
    started = time.perf_counter_ns()
    for _ in range(steps - 1):
        state, value, gradient = _block(compiled(state))
    remaining_ms = (time.perf_counter_ns() - started) / 1e6

    parameters = state[0]
    final_value, final_gradient = _block(jax.value_and_grad(objective)(parameters))
    residuals = _block(problem.residuals(parameters))
    initial_residuals = _block(problem.residuals(initial_parameters))
    hard_cvar = _hard_cvar(residuals, quantile)
    initial_hard_cvar = _hard_cvar(initial_residuals, quantile)
    physical_metrics = {
        name: float(metric)
        for name, metric in problem.physical_metrics(parameters).items()
    }
    initial_physical_metrics = {
        name: float(metric)
        for name, metric in problem.physical_metrics(initial_parameters).items()
    }
    return {
        "status": "ok",
        "workload": problem.name,
        "method": method,
        "seed": seed,
        "steps": steps,
        "compile_ms": compile_ms,
        "first_step_ms": first_step_ms,
        "remaining_training_ms": remaining_ms,
        "total_training_ms": first_step_ms + remaining_ms,
        "compiled_memory": _memory(compiled),
        "initial_hard_quantile": float(jnp.quantile(initial_residuals, quantile)),
        "initial_maximum_residual": float(jnp.max(initial_residuals)),
        "initial_hard_cvar": float(initial_hard_cvar),
        "initial_mean_residual": float(jnp.mean(initial_residuals)),
        "final_objective": float(final_value),
        "final_hard_quantile": float(jnp.quantile(residuals, quantile)),
        "final_maximum_residual": float(jnp.max(residuals)),
        "final_hard_cvar": float(hard_cvar),
        "final_mean_residual": float(jnp.mean(residuals)),
        "gradient_norm": float(jnp.linalg.norm(final_gradient)),
        "nonfinite_gradient_count": int(jnp.sum(~jnp.isfinite(final_gradient))),
        "soft_quantile_sensitivity": _sensitivity(
            residuals,
            quantile,
            epsilon,
            solver_iterations,
        ),
        "transport_diagnostics": _transport_diagnostics(
            residuals,
            epsilon,
            solver_iterations,
        ),
        "physical_metrics": physical_metrics,
        "initial_physical_metrics": initial_physical_metrics,
        "parameters": [float(value) for value in parameters],
    }


def _summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = sorted(
        {
            (record["workload"], record["method"])
            for record in records
            if record["status"] == "ok"
        }
    )
    result = []
    for workload, method in groups:
        group = [
            record
            for record in records
            if record["status"] == "ok"
            and record["workload"] == workload
            and record["method"] == method
        ]
        hard_quantiles = [record["final_hard_quantile"] for record in group]
        initial_quantiles = [record["initial_hard_quantile"] for record in group]
        hard_cvars = [record["final_hard_cvar"] for record in group]
        maxima = [record["final_maximum_residual"] for record in group]
        training_times = [record["total_training_ms"] for record in group]
        count = len(group)
        mean_quantile = sum(hard_quantiles) / count
        variance = sum((value - mean_quantile) ** 2 for value in hard_quantiles) / count
        physical_metric_means = {
            name: sum(record["physical_metrics"][name] for record in group) / count
            for name in group[0]["physical_metrics"]
        }
        result.append(
            {
                "workload": workload,
                "method": method,
                "successful_seeds": count,
                "mean_initial_hard_quantile": sum(initial_quantiles) / count,
                "mean_final_hard_quantile": mean_quantile,
                "mean_hard_quantile_improvement": (
                    sum(initial_quantiles) / count - mean_quantile
                ),
                "std_final_hard_quantile": variance**0.5,
                "mean_final_hard_cvar": sum(hard_cvars) / count,
                "mean_final_maximum_residual": sum(maxima) / count,
                "mean_training_ms": sum(training_times) / count,
                "mean_physical_metrics": physical_metric_means,
                "nonfinite_gradient_count": sum(
                    record["nonfinite_gradient_count"] for record in group
                ),
            }
        )
    return result


def _metadata() -> dict[str, Any]:
    device = jax.devices()[0]
    return {
        "python": platform.python_version(),
        "jax": jax.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": {
            "platform": device.platform,
            "device_kind": device.device_kind,
            "id": device.id,
        },
    }


def main() -> None:
    arguments = _parser().parse_args()
    _validate(arguments)
    size = 12 if arguments.smoke else int(arguments.size)
    steps = 2 if arguments.smoke else int(arguments.steps)
    seeds = (0,) if arguments.smoke else tuple(arguments.seeds)
    dtype = _DTYPES[arguments.dtype]
    records: list[dict[str, Any]] = []

    for problem in _problems(size, dtype):
        for method in arguments.methods:
            for seed in seeds:
                records.append(
                    _train(
                        problem,
                        method,
                        seed=seed,
                        quantile=float(arguments.quantile),
                        epsilon=float(arguments.epsilon),
                        solver_iterations=int(arguments.sinkhorn_iterations),
                        logsumexp_temperature=float(arguments.logsumexp_temperature),
                        learning_rate=float(arguments.learning_rate),
                        steps=steps,
                    )
                )

    payload = {
        "schema": "phydrax.soft-quantile-objectives.v1",
        "metadata": _metadata(),
        "configuration": {
            "methods": list(arguments.methods),
            "size": size,
            "quantile": float(arguments.quantile),
            "epsilon": float(arguments.epsilon),
            "sinkhorn_iterations": int(arguments.sinkhorn_iterations),
            "logsumexp_temperature": float(arguments.logsumexp_temperature),
            "learning_rate": float(arguments.learning_rate),
            "steps": steps,
            "seeds": list(seeds),
            "dtype": arguments.dtype,
            "smoke": bool(arguments.smoke),
        },
        "records": records,
        "summary": _summary(records),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
