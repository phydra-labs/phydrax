#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


@dataclass(frozen=True, slots=True)
class TrefftzBenchmarkRecord:
    problem_id: str
    method_id: str
    dimension: int
    trial_rank: int
    parameter_count: int
    boundary_points: int
    evaluation_points: int
    compile_ms: float
    solve_ms: float
    evaluation_ms: float
    boundary_residual: float
    relative_l2: float
    maximum_pde_residual: float
    certificate_id: str
    passed: bool


def _ready(value):
    return jax.block_until_ready(value)


def _harmonic_record(
    dimension: int,
    *,
    boundary_points: int,
    evaluation_points: int,
    key,
) -> TrefftzBenchmarkRecord:
    basis = phx.equations.HarmonicPolynomialBasis(dimension, 1)
    domain = phx.domain.HyperRectangle((-1.0,) * dimension, (1.0,) * dimension)
    field = domain.Model("x")(phx.equations.LinearTrefftzField(basis))
    coefficients = jnp.linspace(0.1, 0.1 * dimension, dimension)
    target = domain.Function("x")(lambda x: 0.25 + jnp.dot(coefficients, x))
    boundary = domain.component({"x": phx.domain.Boundary()})
    condition = phx.conditions.Dirichlet("u", boundary, target=target)
    boundary_key, solve_key, evaluation_key = jr.split(key, 3)
    source = phx.integration.fixed(
        phx.integration.materialize(
            phx.integration.mean_over(boundary),
            phx.domain.PointSampling(boundary_points),
            key=boundary_key,
        )
    )
    solver = phx.solver.FunctionalSolver(
        functions={"u": field},
        terms=(phx.terms.ResidualPenalty(condition, source),),
    )
    started = time.perf_counter()
    result = phx.solver.solve_linear_trial_space(solver, key=solve_key)
    _ready(result.final_residual_norm)
    solve_ms = 1e3 * (time.perf_counter() - started)

    batch = domain.component().sample(
        phx.domain.PointSampling(evaluation_points), key=evaluation_key
    )
    evaluate = eqx.filter_jit(lambda model: model(batch).data)
    started = time.perf_counter()
    first = evaluate(result.solver["u"])
    _ready(first)
    first_ms = 1e3 * (time.perf_counter() - started)
    started = time.perf_counter()
    predicted = evaluate(result.solver["u"])
    _ready(predicted)
    evaluation_ms = 1e3 * (time.perf_counter() - started)
    compile_ms = max(first_ms - evaluation_ms, 0.0)
    expected = jnp.asarray(target(batch).data)
    relative_l2 = jnp.linalg.norm(predicted - expected) / jnp.linalg.norm(expected)
    audit = phx.equations.audit_trial_space(result.solver["u"], batch)
    certificate = phx.equations.trial_space_certificate(result.solver["u"])
    passed = bool(
        result.valid
        & audit.valid
        & (result.final_residual_norm <= 1e-8)
        & (relative_l2 <= 1e-8)
    )
    return TrefftzBenchmarkRecord(
        problem_id=f"harmonic-affine-{dimension}d",
        method_id="direct-harmonic-polynomial",
        dimension=dimension,
        trial_rank=basis.rank,
        parameter_count=result.coefficient_count,
        boundary_points=boundary_points,
        evaluation_points=evaluation_points,
        compile_ms=compile_ms,
        solve_ms=solve_ms,
        evaluation_ms=evaluation_ms,
        boundary_residual=float(result.final_residual_norm),
        relative_l2=float(relative_l2),
        maximum_pde_residual=float(audit.maximum_residual),
        certificate_id=certificate.certificate_id,
        passed=passed,
    )


def run_trefftz_benchmarks(
    dimensions: tuple[int, ...] = (2, 4, 8),
    /,
    *,
    boundary_points: int = 256,
    evaluation_points: int = 128,
    seed: int = 0,
) -> dict[str, Any]:
    """Run deterministic nD harmonic boundary-fit workloads."""

    dimensions_ = tuple(int(value) for value in dimensions)
    if not dimensions_ or any(value < 2 for value in dimensions_):
        raise ValueError("Benchmark dimensions must be at least two.")
    if boundary_points <= 0 or evaluation_points <= 0:
        raise ValueError("Benchmark point counts must be positive.")
    keys = jr.split(jr.key(seed), len(dimensions_))
    records = tuple(
        _harmonic_record(
            dimension,
            boundary_points=int(boundary_points),
            evaluation_points=int(evaluation_points),
            key=key,
        )
        for dimension, key in zip(dimensions_, keys, strict=True)
    )
    device = jax.devices()[0]
    return {
        "schema_version": 1,
        "dimensions": list(dimensions_),
        "boundary_points": int(boundary_points),
        "evaluation_points": int(evaluation_points),
        "seed": int(seed),
        "environment": {
            "platform": device.platform,
            "device_kind": device.device_kind,
            "jax_enable_x64": bool(jax.config.x64_enabled),
        },
        "records": [asdict(record) for record in records],
        "passed": all(record.passed for record in records),
    }


def _dimensions(value: str, /) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark exact nD Trefftz trial spaces.")
    parser.add_argument("--dimensions", type=_dimensions, default=(2, 4, 8))
    parser.add_argument("--boundary-points", type=int, default=256)
    parser.add_argument("--evaluation-points", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    result = run_trefftz_benchmarks(
        arguments.dimensions,
        boundary_points=arguments.boundary_points,
        evaluation_points=arguments.evaluation_points,
        seed=arguments.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
