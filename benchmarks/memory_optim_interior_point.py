#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Phase-separated interior-point trial workspace and fixed-point action scaling.

The original solver selects on the host after materializing every trial's
derivatives. Its baseline kernel therefore returns the complete batch.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from phydrax import optim
from phydrax.linalg import prepare_linearization
from phydrax.optim._constrained_model import (
    ConstrainedModelEvaluation,
    prepare_constrained_model_evaluation,
    PreparedConstrainedModel,
)
from phydrax.optim._filter_ipm import _select_derivative_trial

from ._io import write_json_atomic
from ._memory import (
    evaluate_memory_gate,
    fitted_scaling_exponent,
    host_allocation_peak,
    MemoryGatePolicy,
    process_peak_rss_bytes,
)
from ._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
)


def _problem(size: int) -> optim.MinimizationProblem:
    def objective(point: Array, args: Any) -> Array:
        del args
        return 0.5 * jnp.sum((point - 0.2) ** 2) + 0.01 * jnp.sum(jnp.sin(point) ** 2)

    def constraints(point: Array, args: Any) -> Array:
        del args
        return jnp.sin(point) + point

    return optim.MinimizationProblem(
        objective,
        constraints=(
            optim.NonlinearConstraint(
                constraints,
                lower=-1.0,
                upper=2.0,
                constraint_id="nonlinear-rows",
            ),
        ),
        problem_id=f"interior-memory-{size}",
    )


def _baseline_trials(
    model: PreparedConstrainedModel,
    point: Array,
    alphas: Array,
    initial: ConstrainedModelEvaluation,
    batch_size: int,
) -> ConstrainedModelEvaluation:
    del initial, batch_size
    return eqx.filter_vmap(lambda alpha: model.evaluate(point * (1.0 - alpha), None))(
        alphas
    )


def _candidate_trials(
    model: PreparedConstrainedModel,
    point: Array,
    alphas: Array,
    initial: ConstrainedModelEvaluation,
    batch_size: int,
) -> tuple[Array, ConstrainedModelEvaluation, Array]:
    values = eqx.filter_vmap(
        lambda alpha: model.evaluate_values(point * (1.0 - alpha), None)
    )(alphas)
    return _select_derivative_trial(
        model, values, values.finite, initial, None, batch_size
    )


_BASELINE_TRIALS = eqx.filter_jit(_baseline_trials)
_CANDIDATE_TRIALS = eqx.filter_jit(_candidate_trials)


def _actions(point: Array, products: int, prepared: bool) -> Array:
    def constraints(value: Array) -> Array:
        return jnp.sin(value) + value * value

    def lagrangian(value: Array) -> Array:
        return jnp.sum(jnp.cos(value)) + jnp.sum(constraints(value))

    if prepared:
        derivative = prepare_linearization(constraints, point)
        hessian = prepare_linearization(jax.grad(lagrangian), point)

        def action(tangent: Array) -> Array:
            return hessian.jvp(tangent) + derivative.vjp(derivative.jvp(tangent))
    else:

        def action(tangent: Array) -> Array:
            hessian_tangent = jax.jvp(jax.grad(lagrangian), (point,), (tangent,))[1]
            constraint_tangent = jax.jvp(constraints, (point,), (tangent,))[1]
            _, pullback = jax.vjp(constraints, point)
            return hessian_tangent + pullback(constraint_tangent)[0]

    def step(tangent: Array, _: None) -> tuple[Array, None]:
        return action(tangent) / 8.0, None

    return jax.lax.scan(step, jnp.ones_like(point), None, length=products)[0]


_ACTIONS = eqx.filter_jit(_actions)


def _measure(
    entry: Any, arguments: tuple[Any, ...], repeats: int
) -> tuple[Any, dict[str, Any]]:
    executable, timing = measure_lower_and_compile(
        lambda: entry.lower(*arguments),
        lambda lowered: lowered.compile(),
    )
    output, warm = measure_repeated(
        lambda: executable(*arguments), warmup=1, repeats=repeats
    )
    compiler = compiler_evidence(
        executable.compiled.cost_analysis(),
        executable.compiled.memory_analysis(),
        source="jax-interior-point-memory",
        unavailable_reason="Compiler estimates unavailable.",
    )
    return output, {
        "lowering_seconds": timing.lowering_seconds,
        "compilation_seconds": timing.compilation_seconds,
        "warm": warm.to_seconds_dict(),
        "compiler": asdict(compiler),
        "logical_output_bytes": logical_array_bytes(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[8, 16])
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--phase", choices=("baseline", "candidate"), default="candidate")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    if options.smoke:
        options.sizes = [4, 8]
        options.repeats = 1
    cases: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    for size in options.sizes:
        point = jnp.full((size,), 0.8, dtype=jnp.float64)
        problem = _problem(size)
        (model, initial), host_peak = host_allocation_peak(
            lambda: prepare_constrained_model_evaluation(problem, point)
        )
        alphas = 0.5 ** jnp.arange(options.trials, dtype=point.dtype)
        baseline_batch, evidence = _measure(
            _BASELINE_TRIALS,
            (model, point, alphas, initial, options.trials),
            options.repeats,
        )
        baseline_index = jnp.argmax(baseline_batch.finite).astype(jnp.int32)
        baseline = (
            baseline_index,
            jax.tree.map(lambda leaf: leaf[baseline_index], baseline_batch),
            jnp.asarray(options.trials, dtype=jnp.int32),
        )
        routes = [("all-derivative-trials", options.trials, baseline, evidence)]
        if options.phase == "candidate":
            for batch in (1, 4, options.trials):
                output, evidence = _measure(
                    _CANDIDATE_TRIALS,
                    (model, point, alphas, initial, batch),
                    options.repeats,
                )
                np.testing.assert_allclose(
                    output[1].gradient, baseline[1].gradient, atol=1e-12
                )
                np.testing.assert_allclose(
                    output[1].constraint_jacobian,
                    baseline[1].constraint_jacobian,
                    atol=1e-12,
                )
                np.testing.assert_array_equal(output[0], baseline[0])
                routes.append(("screened-bounded-derivatives", batch, output, evidence))
        for route, batch, output, evidence in routes:
            cases.append(
                {
                    "size": size,
                    "route": route,
                    "batch_size": batch,
                    "trials": options.trials,
                    **evidence,
                    "work_unit": "one-line-search",
                    "host_preparation_peak_bytes": host_peak,
                    "logical_derivative_workspace_bytes": batch
                    * (size * size + size)
                    * point.dtype.itemsize,
                    "derivative_evaluations": int(output[2]),
                    "selected_index": int(output[0]),
                    "finite": bool(output[1].finite),
                }
            )
        reference, action_evidence = _measure(
            _ACTIONS, (point, 16, False), options.repeats
        )
        actions.append(
            {
                "size": size,
                "route": "reconstructed-actions",
                "products": 16,
                "logical_constraint_primal_evaluations": 32,
                "logical_lagrangian_gradient_evaluations": 16,
                "jvp_actions": 16,
                "vjp_actions": 16,
                "hvp_actions": 16,
                **action_evidence,
            }
        )
        if options.phase == "candidate":
            output, action_evidence = _measure(
                _ACTIONS, (point, 16, True), options.repeats
            )
            np.testing.assert_allclose(output, reference, rtol=1e-12, atol=1e-12)
            actions.append(
                {
                    "size": size,
                    "route": "prepared-actions",
                    "products": 16,
                    "logical_constraint_primal_evaluations": 1,
                    "logical_lagrangian_gradient_evaluations": 1,
                    "jvp_actions": 16,
                    "vjp_actions": 16,
                    "hvp_actions": 16,
                    **action_evidence,
                }
            )
    record: dict[str, Any] = {
        "identity": capture_benchmark_identity(
            Path(__file__).resolve().parents[1],
            Path(__file__),
            ("environment", "cases", "actions", "phase"),
        ).to_dict(),
        "environment": capture_environment().to_dict(),
        "phase": options.phase,
        "cases": cases,
        "actions": actions,
        "process_peak_rss_bytes": process_peak_rss_bytes(),
    }
    if options.phase == "candidate":
        baselines = [
            case["compiler"]["temporary_bytes"]
            for case in cases
            if case["route"] == "all-derivative-trials"
        ]
        gates = []
        for batch in (1, 4, options.trials):
            candidates = [
                case
                for case in cases
                if case["route"] == "screened-bounded-derivatives"
                and case["batch_size"] == batch
            ]
            temporary = [case["compiler"]["temporary_bytes"] for case in candidates]
            gates.append(
                {
                    "batch_size": batch,
                    **evaluate_memory_gate(
                        options.sizes, baselines, temporary, MemoryGatePolicy()
                    ).to_dict(),
                    "compiler_exponent": fitted_scaling_exponent(options.sizes, temporary)
                    if all(temporary)
                    else None,
                    "logical_workspace_exponent": fitted_scaling_exponent(
                        options.sizes,
                        [
                            case["logical_derivative_workspace_bytes"]
                            for case in candidates
                        ],
                    ),
                }
            )
        record["gates"] = gates
    output_path = options.output or Path(__file__).with_name(
        f"memory_optim_interior_point{'_baseline' if options.phase == 'baseline' else ''}.json"
    )
    write_json_atomic(output_path, record)
    print(json.dumps({"output": str(output_path), "gates": record.get("gates")}))


if __name__ == "__main__":
    main()
