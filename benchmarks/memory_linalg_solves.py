# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Least-squares JVP workspace and host sparse factor reuse; small default sweep."""

from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

import phydrax.linalg as la
from phydrax.linalg._dense_pseudoinverse import _pseudoinverse_tangent
from phydrax.linalg.backends._jax_sparse import _host_factor, _host_solve
from phydrax.observation import (
    CoordinateLayout,
    DiagonalCovarianceAction,
    LinearNuisancePlan,
)

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
    measure_host,
    measure_lower_and_compile,
    measure_repeated,
)


def _derivative(matrix: jax.Array, rhs: jax.Array) -> jax.Array:
    policy = la.LinearSolvePolicy(
        la.DenseSVD(damping=0.3),
        tolerance=la.TolerancePolicy(relative=1e-10, absolute=1e-12, max_steps=32),
    )

    def solution(values: jax.Array) -> jax.Array:
        return la.solve(
            la.LeastSquaresProblem(la.DenseLinearOperator(matrix)),
            values,
            policy=policy,
            rhs_layout=la.RHSLayout((values.shape[-1],)),
        ).value

    return jax.jvp(solution, (rhs,), (jnp.ones_like(rhs),))[1]


def _sparse_cases(size: int, counts: list[int]) -> list[dict[str, Any]]:
    matrix = sp.diags(
        (-np.ones(size - 1), np.full(size, 4.0), -np.ones(size - 1)),
        (-1, 0, 1),
        format="csc",
    )
    providers = ["scipy-superlu"]
    if (
        importlib.util.find_spec("scikits") is not None
        and importlib.util.find_spec("scikits.umfpack") is not None
    ):
        providers.append("umfpack")
    if importlib.util.find_spec("sparseqr") is not None:
        providers.append("spqr")
    if importlib.util.find_spec("sksparse") is not None:
        providers.append("cholmod")
    records = []
    for provider in providers:
        factor, seconds = measure_host(
            lambda: _host_factor(provider, matrix, np.arange(size))
        )
        for count in counts:
            rhs = np.arange(size * count, dtype=np.float64).reshape(size, count) / size
            solution, warm = measure_repeated(
                lambda: _host_solve(provider, factor, rhs), warmup=1, repeats=2
            )
            _, peak = host_allocation_peak(lambda: _host_solve(provider, factor, rhs))
            records.append(
                {
                    "provider": provider,
                    "rhs_count": count,
                    "factorization_seconds": seconds,
                    "solve": warm.to_seconds_dict(),
                    "host_peak_bytes": peak,
                    "relative_residual": float(
                        np.linalg.norm(matrix @ solution - rhs) / np.linalg.norm(rhs)
                    ),
                    "factorizations_per_preparation": 1
                    if provider in ("scipy-superlu", "cholmod")
                    or provider == "umfpack"
                    and type(factor).__name__ != "csc_matrix"
                    else 0,
                    "factorizations_per_solve": count
                    if provider == "umfpack" and type(factor).__name__ == "csc_matrix"
                    else (1 if provider == "spqr" else 0),
                    "route": "per-column-refactor"
                    if provider == "umfpack" and type(factor).__name__ == "csc_matrix"
                    else (
                        "provider-matrix-rhs" if provider == "spqr" else "reusable-factor"
                    ),
                }
            )
    return records


def _auxiliary_cases() -> dict[str, list[dict[str, Any]]]:
    rng = np.random.default_rng(76)
    tangent_records: list[dict[str, Any]] = []
    nuisance_records: list[dict[str, Any]] = []
    determinant_records: list[dict[str, Any]] = []
    for size in (32, 64, 128):
        for rows, columns in ((size, 4), (4, size)):
            matrix = rng.normal(size=(rows, columns))
            direction = rng.normal(size=matrix.shape)
            pseudoinverse = np.linalg.pinv(matrix)
            arguments = tuple(
                jnp.asarray(value) for value in (matrix, pseudoinverse, direction)
            )
            entry = jax.jit(_pseudoinverse_tangent)
            compiled, timing = measure_lower_and_compile(
                lambda: entry.lower(*arguments), lambda lowered: lowered.compile()
            )
            value, warm = measure_repeated(
                lambda: compiled(*arguments), warmup=1, repeats=2
            )
            step = 1e-6
            reference = (
                np.linalg.pinv(matrix + step * direction)
                - np.linalg.pinv(matrix - step * direction)
            ) / (2 * step)
            evidence = compiler_evidence(
                compiled.cost_analysis(),
                compiled.memory_analysis(),
                source="pseudoinverse-tangent",
            )
            tangent_records.append(
                {
                    "rows": rows,
                    "columns": columns,
                    "compiler": asdict(evidence),
                    "lowering_seconds": timing.lowering_seconds,
                    "compilation_seconds": timing.compilation_seconds,
                    "warm": warm.to_seconds_dict(),
                    "max_error": float(np.max(np.abs(np.asarray(value) - reference))),
                    "largest_square_entries": min(rows, columns) ** 2,
                }
            )
        layout = CoordinateLayout(tuple(f"observation-{i}" for i in range(size + 4)))
        covariance = DiagonalCovarianceAction(np.ones(size + 4, dtype=np.float64), layout)
        design = rng.normal(size=(size + 4, size))
        nuisance, peak = host_allocation_peak(
            lambda: LinearNuisancePlan(
                design, covariance, layout, tuple(f"parameter-{i}" for i in range(size))
            )
        )
        result, warm = measure_repeated(
            lambda: nuisance.evaluate(
                jnp.ones(size + 4, dtype=jnp.float64),
                jnp.zeros(size + 4, dtype=jnp.float64),
            ),
            warmup=1,
            repeats=2,
        )
        nuisance_records.append(
            {
                "size": size,
                "host_peak_bytes": peak,
                "retained_bytes": logical_array_bytes(nuisance),
                "factor_bytes": logical_array_bytes(nuisance.normal_lower),
                "warm": warm.to_seconds_dict(),
                "successful": bool(result.successful),
                "normal_residual": float(jnp.max(jnp.abs(result.normal_residual))),
            }
        )
        matrix = design.T @ design + np.eye(size)
        properties = la.OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "construction",
                "positive_semidefinite": "construction",
            },
        )
        factorization = la.factorize(
            la.DenseLinearOperator(jnp.asarray(matrix), properties=properties),
            la.FactorizationPolicy("cholesky"),
        )
        logdet, warm = measure_repeated(
            factorization.log_abs_determinant, warmup=1, repeats=2
        )
        determinant_records.append(
            {
                "size": size,
                "warm": warm.to_seconds_dict(),
                "max_error": abs(float(logdet) - float(np.linalg.slogdet(matrix)[1])),
                "diagonal_workspace_bytes": size * np.dtype(np.float64).itemsize,
                "retained_bytes": logical_array_bytes(factorization),
            }
        )
        jax.clear_caches()
    return {
        "pseudoinverse": tangent_records,
        "nuisance": nuisance_records,
        "logdet": determinant_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--size", type=int, default=12)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_suffix(".json")
    )
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    baseline = None if args.baseline is None else json.loads(args.baseline.read_text())
    if (
        baseline is not None
        and [case["rhs_count"] for case in baseline["cases"]] != args.counts
    ):
        raise ValueError("The baseline must use the same RHS capacities.")
    rng = np.random.default_rng(42)
    matrix = jnp.asarray(rng.normal(size=(args.size + 4, args.size)))
    cases = []
    temporary_bytes: list[int] = []
    for count in args.counts:
        rhs = jnp.asarray(rng.normal(size=(args.size + 4, count)))
        entry = jax.jit(_derivative)
        compiled, timing = measure_lower_and_compile(
            lambda: entry.lower(matrix, rhs), lambda lowered: lowered.compile()
        )
        value, warm = measure_repeated(lambda: compiled(matrix, rhs), warmup=1, repeats=2)
        evidence = compiler_evidence(
            compiled.cost_analysis(),
            compiled.memory_analysis(),
            source="least-squares-jvp",
        )
        if evidence.temporary_bytes is None:
            raise RuntimeError("The compiler did not report temporary workspace.")
        temporary_bytes.append(evidence.temporary_bytes)
        reference = np.linalg.solve(
            np.asarray(matrix).T @ np.asarray(matrix) + 0.09 * np.eye(args.size),
            np.asarray(matrix).T @ np.ones(rhs.shape),
        )
        cases.append(
            {
                "rhs_count": count,
                "compiler": asdict(evidence),
                "lowering_seconds": timing.lowering_seconds,
                "compilation_seconds": timing.compilation_seconds,
                "warm": warm.to_seconds_dict(),
                "output_bytes": logical_array_bytes(value),
                "max_error": float(np.max(np.abs(np.asarray(value) - reference))),
            }
        )
        jax.clear_caches()
    fields = (
        "environment",
        "identity",
        "cases",
        "sparse",
        "auxiliary",
        "scaling_exponent",
        "gate",
        "peak_rss_bytes",
    )
    record = {
        "environment": capture_environment().to_dict(),
        "identity": asdict(
            capture_benchmark_identity(Path.cwd(), Path(__file__), fields)
        ),
        "cases": cases,
        "sparse": _sparse_cases(args.size, args.counts),
        "auxiliary": _auxiliary_cases(),
        "scaling_exponent": fitted_scaling_exponent(args.counts, temporary_bytes),
        "gate": None,
        "peak_rss_bytes": process_peak_rss_bytes(),
    }
    if baseline is not None:
        record["gate"] = evaluate_memory_gate(
            args.counts,
            [case["compiler"]["temporary_bytes"] for case in baseline["cases"]],
            temporary_bytes,
            MemoryGatePolicy(),
        ).to_dict()
    write_json_atomic(args.output, record)
    print(
        json.dumps(
            {"scaling_exponent": record["scaling_exponent"], "gate": record["gate"]}
        )
    )


if __name__ == "__main__":
    main()
