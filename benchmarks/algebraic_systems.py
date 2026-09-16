#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from phydrax import algebraic
from phydrax.algebraic._quotient import solve_quotient_roots
from phydrax.geometry import polynomial_image
from phydrax.optim import polynomial as polynomial_optim
from phydrax.tensor_decomposition import (
    normalize_waring_components,
    reconstruct_symmetric_tensor,
    solve_symmetric_waring,
    SymmetricWaringProblem,
    SymmetricWaringRefinement,
)


def _time(call):
    start = time.perf_counter()
    value = call()
    jax.block_until_ready(value)
    return value, time.perf_counter() - start


def _system(variable_count):
    variables = tuple(f"x{index}" for index in range(variable_count))
    equations = tuple(f"f{index}" for index in range(variable_count))
    rows = []
    exponents = []
    coefficients = []
    for equation in range(variable_count):
        constant = [0] * variable_count
        quadratic = [0] * variable_count
        quadratic[equation] = 2
        rows.extend((equation, equation))
        exponents.extend((constant, quadratic))
        coefficients.extend((-1.0, 1.0))
    return algebraic.SparsePolynomialSystem.from_coo(
        variables,
        equations,
        rows,
        exponents,
        jnp.asarray(coefficients),
    )


def _core_row(variable_count, iterations):
    system = _system(variable_count)
    points = jnp.linspace(-1.0, 1.0, 256 * variable_count).reshape((256, variable_count))
    evaluate = eqx.filter_jit(system.evaluate)
    jacobian = eqx.filter_jit(system.jacobian)
    _, first_evaluate = _time(lambda: evaluate(points))
    _, first_jacobian = _time(lambda: jacobian(points))
    start = time.perf_counter()
    value = None
    for _ in range(iterations):
        value = evaluate(points)
    jax.block_until_ready(value)
    warm = (time.perf_counter() - start) / iterations
    quotient_start = time.perf_counter()
    quotient = solve_quotient_roots(system)
    quotient_seconds = time.perf_counter() - quotient_start
    return {
        "kind": "sparse-polynomial-core",
        "variables": variable_count,
        "terms": system.support.term_count,
        "batch": 256,
        "first_evaluate_seconds": first_evaluate,
        "first_jacobian_seconds": first_jacobian,
        "warm_evaluate_seconds": warm,
        "quotient_seconds": quotient_seconds,
        "quotient_status": int(quotient.status),
        "quotient_roots": quotient.root_count,
        "maximum_residual": float(
            jnp.max(jnp.abs(quotient.original_residuals), initial=0.0)
        ),
    }


def _image_row():
    mapping = polynomial_image.SparsePolynomialMap(
        algebraic.SparsePolynomialSystem.from_coo(
            ("t",),
            ("x", "y", "z"),
            (0, 1, 2),
            ((1,), (2,), (3,)),
            jnp.ones((3,)),
        )
    )
    target = polynomial_image.TargetMonomialSupport(
        ("x", "y", "z"),
        tuple(
            (a, b, c) for a in range(3) for b in range(3 - a) for c in range(3 - a - b)
        ),
    )
    plan_start = time.perf_counter()
    plan = polynomial_image.plan_polynomial_image_analysis(
        mapping,
        target,
        discovery_sample_count=64,
        heldout_sample_count=16,
        key=jr.PRNGKey(11),
        policy=polynomial_image.PolynomialImageAnalysisPolicy(
            rank_relative_tolerance=1.0e-5,
            heldout_absolute_tolerance=2.0e-5,
            heldout_relative_tolerance=2.0e-5,
        ),
    )
    plan_seconds = time.perf_counter() - plan_start
    prepare_start = time.perf_counter()
    prepared = plan.prepare()
    prepare_seconds = time.perf_counter() - prepare_start
    return {
        "kind": "polynomial-image",
        "source_dimension": mapping.source_dimension,
        "target_dimension": mapping.target_dimension,
        "target_monomials": target.monomial_count,
        "plan_seconds": plan_seconds,
        "analysis_seconds": prepare_seconds,
        "status": prepared.result.status.value,
        "relations": prepared.result.relations.relation_count,
    }


def _moment_row():
    objective = algebraic.SparsePolynomialSystem.from_coo(
        ("x", "y"), ("objective",), (0, 0), ((2, 0), (0, 2)), jnp.asarray((1.0, 1.0))
    )
    domain = algebraic.SparsePolynomialSystem.from_coo(
        ("x", "y"),
        ("unit disk",),
        (0, 0, 0),
        ((0, 0), (2, 0), (0, 2)),
        jnp.asarray((1.0, -1.0, -1.0)),
    )
    problem = polynomial_optim.PolynomialOptimizationProblem(
        objective, inequalities=domain
    )
    start = time.perf_counter()
    prepared = polynomial_optim.prepare_polynomial_relaxation(problem, 2)
    elapsed = time.perf_counter() - start
    return {
        "kind": "dense-moment-relaxation",
        "variables": problem.variable_count,
        "order": prepared.plan.order,
        "moments": prepared.template.moment_basis.moment_count,
        "conic_rows": prepared.plan.estimate.conic_rows,
        "dense_constraint_entries": prepared.plan.estimate.dense_constraint_entries,
        "prepare_seconds": elapsed,
    }


def _tensor_row():
    weights, factors = normalize_waring_components(
        jnp.asarray((1.7, -0.65)),
        jnp.asarray(((1.0, 0.35), (0.55, 1.0))),
        3,
    )
    tensor = reconstruct_symmetric_tensor(weights, factors, 3)
    start = time.perf_counter()
    result = solve_symmetric_waring(
        SymmetricWaringProblem(tensor, 2),
        refinement=SymmetricWaringRefinement(enabled=False),
    )
    elapsed = time.perf_counter() - start
    return {
        "kind": "symmetric-waring",
        "dimension": 2,
        "order": 3,
        "rank": 2,
        "solve_seconds": elapsed,
        "status": int(result.status),
        "relative_residual": float(result.relative_residual),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    rows = [
        _core_row(1, arguments.iterations),
        _core_row(2, arguments.iterations),
        _image_row(),
        _moment_row(),
        _tensor_row(),
    ]
    payload = {
        "claim": "representative-native-algebraic-costs-not-provider-or-scaling-guarantees",
        "rows": rows,
    }
    encoded = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True)
    print(encoded)
    if arguments.output is not None:
        arguments.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
