#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Sequence
from typing import Any

import coordax as cx
import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx
from benchmarks._runtime import measure_repeated


def _measure(
    operation: Callable[[], phx.integration.IntegrationEstimate],
    /,
    *,
    repeats: int,
) -> tuple[phx.integration.IntegrationEstimate, float]:
    estimate, distribution = measure_repeated(
        operation,
        warmup=1,
        repeats=repeats,
    )
    return estimate, 1_000.0 * float(distribution.mean_seconds)


def _record(
    method: str,
    budget: int,
    reference: float,
    estimate: phx.integration.IntegrationEstimate,
    elapsed_ms: float,
    /,
) -> dict[str, Any]:
    value = float(estimate.value.data)
    reported_error = (
        None if estimate.error_estimate is None else float(estimate.error_estimate)
    )
    return {
        "method": method,
        "requested_budget": int(budget),
        "num_evaluations": int(estimate.num_evaluations),
        "value": value,
        "absolute_error": abs(value - reference),
        "reported_error": reported_error,
        "error_kind": estimate.error_kind,
        "status": int(estimate.status),
        "mean_wall_ms": float(elapsed_ms),
    }


def _measure_compiled(
    operation: Callable[[jax.Array], phx.integration.IntegrationEstimate],
    /,
    *,
    repeats: int,
) -> tuple[phx.integration.IntegrationEstimate, float]:
    compiled = jax.jit(operation)
    scale = jnp.asarray(1.0)
    estimate, distribution = measure_repeated(
        lambda: compiled(scale),
        warmup=1,
        repeats=repeats,
    )
    return estimate, 1_000.0 * float(distribution.mean_seconds)


def _interoperability_record(
    method: str,
    budget: int,
    operation: Callable[[jax.Array], phx.integration.IntegrationEstimate],
    /,
    *,
    repeats: int,
    working_set_bytes: int,
    weight_layout: str,
) -> dict[str, Any]:
    scale = jnp.asarray(1.0)
    estimate, eager_ms = _measure(lambda: operation(scale), repeats=repeats)
    compiled_estimate, compiled_ms = _measure_compiled(operation, repeats=repeats)
    values = jnp.asarray(compiled_estimate.value.data)
    statuses = jnp.asarray(compiled_estimate.status)
    evaluations = jnp.asarray(compiled_estimate.num_evaluations)
    return {
        "method": method,
        "requested_budget": int(budget),
        "value_shape": list(values.shape),
        "status_shape": list(statuses.shape),
        "num_evaluations_shape": list(evaluations.shape),
        "max_num_evaluations": int(jnp.max(evaluations)),
        "value_checksum": float(jnp.sum(jnp.nan_to_num(values))),
        "successful": bool(jnp.all(compiled_estimate.successful)),
        "error_kind": compiled_estimate.error_kind,
        "mean_eager_ms": float(eager_ms),
        "mean_compiled_ms": float(compiled_ms),
        "working_set_bytes": int(working_set_bytes),
        "weight_layout": weight_layout,
    }


def _interoperability_benchmarks(
    budgets: tuple[int, ...],
    /,
    *,
    repeats: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for budget in budgets:
        particle = jnp.linspace(-2.0, 2.0, budget)
        weighted_samples = cx.Field(
            jnp.arange(4.0)[:, None] + particle[None, :] ** 2,
            dims=("case", "particle"),
        )
        weighted_log_weights = cx.Field(
            jnp.broadcast_to(-0.5 * particle[None, :] ** 2, (4, budget)),
            dims=("case", "particle"),
        )
        weighted_target = phx.integration.weighted(
            weighted_samples,
            weighted_log_weights,
            sample_axes="particle",
            independent=True,
        )
        weighted_operation = lambda scale, target=weighted_target: (
            phx.integration.integrate(
                lambda values: scale * values,
                target,
            )
        )
        records.append(
            _interoperability_record(
                "batched-weighted-case-particle",
                budget,
                weighted_operation,
                repeats=repeats,
                working_set_bytes=(
                    jnp.asarray(weighted_samples.data).nbytes
                    + jnp.asarray(weighted_log_weights.data).nbytes
                ),
                weight_layout="batched-log-weights",
            )
        )

        num_times = 16
        num_space = 16
        times = jnp.linspace(0.0, 1.0, num_times)
        axis = phx.discretization.FourierAxisSpec(num_space).materialize(0.0, 1.0)
        spatial = phx.discretization.TensorSpectralDiscretization.from_axes((axis,))
        phase = jnp.linspace(0.0, 2.0 * jnp.pi, budget, endpoint=False)
        states = (
            jnp.sin(phase[:, None, None] + 2.0 * jnp.pi * axis.nodes[None, None, :])
            + times[None, :, None]
        )
        valid = jnp.ones((budget, num_times), dtype=bool)
        valid = valid.at[-1, num_times // 2 :].set(False)
        trajectory = phx.stochastic.StochasticTrajectory(
            times,
            states,
            valid=valid,
            realization_axes=("path",),
            realization_shape=(budget,),
            state_axes=("space",),
            realizations=(None,),
        )
        marginal_target = phx.stochastic.trajectory_measure(
            trajectory,
            mode="marginal",
        )
        path_target = phx.stochastic.trajectory_measure(trajectory, mode="path")
        time_target = phx.stochastic.time_measure(trajectory)
        space_target = phx.integration.spatial_measure(
            spatial,
            spatial_dims="space",
        )
        state_bytes = states.nbytes + valid.nbytes
        marginal_operation = lambda scale, target=marginal_target: (
            phx.integration.integrate(
                lambda values: scale * values,
                target,
            )
        )
        records.append(
            _interoperability_record(
                "trajectory-marginal",
                budget,
                marginal_operation,
                repeats=repeats,
                working_set_bytes=state_bytes,
                weight_layout="masked-path-time",
            )
        )
        space_operation = lambda scale, values=path_target.samples: (
            phx.integration.integrate(
                scale * values,
                space_target,
            )
        )
        records.append(
            _interoperability_record(
                "separable-spatial-path-time",
                budget,
                space_operation,
                repeats=repeats,
                working_set_bytes=state_bytes,
                weight_layout="separable-spatial",
            )
        )

        def staged_operation(
            scale: jax.Array,
            values: cx.Field = path_target.samples,
        ) -> phx.integration.IntegrationEstimate:
            space_estimate = phx.integration.integrate(
                scale * values,
                space_target,
            )
            time_estimate = phx.integration.integrate(
                space_estimate.value,
                time_target,
            )
            return phx.integration.integrate(time_estimate.value, path_target)

        records.append(
            _interoperability_record(
                "staged-space-time-path",
                budget,
                staged_operation,
                repeats=repeats,
                working_set_bytes=state_bytes,
                weight_layout="separable-space-then-fixed-time-then-weighted-path",
            )
        )
    return records


def run_integration_benchmarks(
    budgets: Sequence[int] = (64, 256, 1024),
    /,
    *,
    repeats: int = 3,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare integration accuracy, diagnostics, and end-to-end wall time."""
    if repeats < 1:
        raise ValueError("repeats must be at least one.")
    budgets_ = tuple(int(value) for value in budgets)
    if not budgets_ or any(value < 8 for value in budgets_):
        raise ValueError("benchmark budgets must contain integers of at least eight.")

    domain = phx.domain.ScalarInterval(0.0, 1.0, label="x")
    component = domain.component()
    target = phx.integration.over(component)
    integrand = domain.Function("x")(lambda x: jax.numpy.exp(x))
    reference_estimate = phx.integration.integrate(
        integrand,
        target,
        phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(128)),
    )
    reference = float(reference_estimate.value.data)
    root_key = jr.key(seed)
    records: list[dict[str, Any]] = []

    adaptive_plan = phx.integration.AdaptiveQuadraturePlan(
        phx.integration.GaussKronrodRule(21),
        absolute_tolerance=1e-12,
        relative_tolerance=1e-12,
        max_intervals=256,
    )
    adaptive, adaptive_ms = _measure(
        lambda: phx.integration.integrate(integrand, target, adaptive_plan),
        repeats=repeats,
    )
    records.append(
        _record("adaptive-gauss-kronrod", 256, reference, adaptive, adaptive_ms)
    )

    for index, budget in enumerate(budgets_):
        fixed_order = min(budget, 128)
        fixed_plan = phx.integration.FixedQuadraturePlan(
            phx.integration.GaussLegendreRule(fixed_order)
        )
        fixed, fixed_ms = _measure(
            lambda: phx.integration.integrate(integrand, target, fixed_plan),
            repeats=repeats,
        )
        records.append(
            _record("fixed-gauss-legendre", budget, reference, fixed, fixed_ms)
        )

        iid_plan = phx.integration.MonteCarloPlan(budget)
        iid_key = jr.fold_in(root_key, 4 * index)
        iid, iid_ms = _measure(
            lambda plan=iid_plan, key=iid_key: phx.integration.integrate(
                integrand, target, plan, key=key
            ),
            repeats=repeats,
        )
        records.append(_record("iid-monte-carlo", budget, reference, iid, iid_ms))

        antithetic_count = budget - budget % 2
        antithetic_plan = phx.integration.MonteCarloPlan(
            antithetic_count,
            design=phx.integration.AntitheticDesign(),
        )
        antithetic_key = jr.fold_in(root_key, 4 * index + 1)
        antithetic, antithetic_ms = _measure(
            lambda plan=antithetic_plan, key=antithetic_key: phx.integration.integrate(
                integrand, target, plan, key=key
            ),
            repeats=repeats,
        )
        records.append(
            _record(
                "antithetic-monte-carlo",
                budget,
                reference,
                antithetic,
                antithetic_ms,
            )
        )

        qmc_count = 1 << int(math.floor(math.log2(budget)))
        qmc_plan = phx.integration.QuasiMonteCarloPlan(
            qmc_count,
            sequence="sobol",
            num_replicates=4,
        )
        qmc_key = jr.fold_in(root_key, 4 * index + 2)
        qmc, qmc_ms = _measure(
            lambda plan=qmc_plan, key=qmc_key: phx.integration.integrate(
                integrand, target, plan, key=key
            ),
            repeats=repeats,
        )
        records.append(_record("randomized-sobol", budget, reference, qmc, qmc_ms))

        level = max(2, int(math.floor(math.log2(budget))) - 1)
        sparse_plan = phx.integration.SparseGridPlan(1, level)
        sparse, sparse_ms = _measure(
            lambda plan=sparse_plan: phx.integration.integrate(integrand, target, plan),
            repeats=repeats,
        )
        records.append(
            _record("smolyak-clenshaw-curtis", budget, reference, sparse, sparse_ms)
        )

    return {
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "integrand": "exp(x)",
        "domain": [0.0, 1.0],
        "reference_method": "Gauss-Legendre-128",
        "reference_value": reference,
        "analytic_value": math.e - 1.0,
        "reference_absolute_error": abs(reference - (math.e - 1.0)),
        "repeats": int(repeats),
        "seed": int(seed),
        "records": records,
        "interoperability_records": _interoperability_benchmarks(
            budgets_,
            repeats=repeats,
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark unified Phydrax integration methods."
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=(64, 256, 1024),
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    result = run_integration_benchmarks(
        arguments.budgets,
        repeats=arguments.repeats,
        seed=arguments.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
