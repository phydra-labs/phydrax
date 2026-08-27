#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx
from phydrax._fingerprint import canonical_fingerprint
from phydrax._trainable import partition_trainable


def _parameter_count(model: Any, /) -> int:
    trainable, _ = partition_trainable(model)
    return sum(int(leaf.size) for leaf in jax.tree.leaves(trainable))


def _cauchy_riemann_residual(model: Any, point: jax.Array, /) -> jax.Array:
    def real_map(value):
        output = jnp.asarray(model(value[0] + 1j * value[1])).reshape((-1,))
        return jnp.concatenate((jnp.real(output), jnp.imag(output)))

    jacobian = jax.jacfwd(real_map)(point)
    output_count = jacobian.shape[0] // 2
    first = jacobian[:output_count, 0] - jacobian[output_count:, 1]
    second = jacobian[:output_count, 1] + jacobian[output_count:, 0]
    return jnp.maximum(jnp.max(jnp.abs(first)), jnp.max(jnp.abs(second)))


def _polynomial(coefficients: jax.Array, /):
    values = jnp.asarray(coefficients, dtype=float)
    potential = phx.equations.HolomorphicPolynomialPotential(
        values.shape[0],
        values.shape[1] - 1,
    )
    return eqx.tree_at(
        lambda item: (item.coefficient_real, item.coefficient_imag),
        potential,
        (values, jnp.zeros_like(values)),
    )


def run_holomorphic_separability_benchmarks() -> dict[str, Any]:
    point = jnp.asarray([0.2, -0.15])
    z = point[0] + 1j * point[1]
    dense = phx.nn.models.HolomorphicMLP(
        in_size=1,
        out_size=1,
        hidden_sizes=(8, 8),
        key=jr.key(0),
    )
    factorized = phx.nn.models.HolomorphicMLP(
        in_size=1,
        out_size=1,
        hidden_sizes=(8, 8),
        linear_ranks=(1, 2, 1),
        key=jr.key(0),
    )
    first = _polynomial(jnp.asarray([[1.0, 1.0], [2.0, -1.0]]))
    second = _polynomial(jnp.asarray([[3.0, 1.0], [1.0, 2.0]]))
    product = phx.equations.HolomorphicProductPotential(
        (first, second),
        latent_rank=2,
        branches=1,
    )
    bundle = phx.equations.HolomorphicBranchBundle((first, second))
    constraint_plan = phx.equations.HolomorphicPolynomialConstraintPlan(
        3,
        (
            phx.equations.HolomorphicPointConstraint.dirichlet(-1.0, 0.0),
            phx.equations.HolomorphicPointConstraint.dirichlet(1.0, 0.0),
        ),
    )
    started = time.perf_counter()
    prepared_constraints = constraint_plan.prepare()
    constraint_preparation_seconds = time.perf_counter() - started
    constrained = phx.equations.ConstrainedHolomorphicPolynomialPotential(
        prepared_constraints,
        initial_free_coordinates=jnp.linspace(
            -0.2,
            0.3,
            prepared_constraints.evidence.nullity,
        ),
    )

    timings = {}
    outputs = {}
    for name, model in (
        ("dense_hmlp", dense),
        ("factorized_hmlp", factorized),
        ("product_potential", product),
        ("branch_bundle", bundle),
        ("constrained_polynomial", constrained),
    ):
        evaluate = jax.jit(model)
        started = time.perf_counter()
        output = evaluate(z)
        jax.block_until_ready(output)
        first_seconds = time.perf_counter() - started
        started = time.perf_counter()
        output = evaluate(z)
        jax.block_until_ready(output)
        steady_seconds = time.perf_counter() - started
        timings[name] = {
            "first_seconds": first_seconds,
            "steady_seconds": steady_seconds,
        }
        outputs[name] = output

    product_jet = product.jet(z, 4)

    def scalar(value):
        return product(value)[0]

    derivative = scalar
    jet_error = jnp.asarray(0.0)
    for order in range(1, 5):
        derivative = jax.jacfwd(derivative, holomorphic=True)
        jet_error = jnp.maximum(
            jet_error,
            jnp.abs(product_jet.derivative(order)[0] - derivative(z)),
        )

    harmonic = phx.equations.HarmonicPotential2D(product)
    laplace_residual = jnp.abs(jnp.trace(jax.hessian(harmonic)(point)))
    cr_residuals = {
        "dense_hmlp": float(_cauchy_riemann_residual(dense, point)),
        "factorized_hmlp": float(_cauchy_riemann_residual(factorized, point)),
        "product_potential": float(_cauchy_riemann_residual(product, point)),
        "branch_bundle": float(_cauchy_riemann_residual(bundle, point)),
        "constrained_polynomial": float(_cauchy_riemann_residual(constrained, point)),
    }
    parameters = {
        "dense_hmlp": _parameter_count(dense),
        "factorized_hmlp": _parameter_count(factorized),
        "product_potential": _parameter_count(product),
        "branch_bundle": _parameter_count(bundle),
        "constrained_polynomial": _parameter_count(constrained),
    }
    constraint_residual = float(jnp.linalg.norm(constrained.constraint_residual()))
    constrained_harmonic = phx.equations.HarmonicPotential2D(constrained)
    constrained_laplace_residual = float(
        jnp.abs(jnp.trace(jax.hessian(constrained_harmonic)(point)))
    )
    passed = bool(
        all(value < 1e-10 for value in cr_residuals.values())
        and float(jet_error) < 1e-10
        and float(laplace_residual) < 1e-10
        and constrained_laplace_residual < 1e-10
        and constraint_residual <= float(prepared_constraints.evidence.lift_tolerance)
        and all(jnp.all(jnp.isfinite(value)) for value in outputs.values())
    )
    payload = {
        "kind": "holomorphic-separability-benchmark",
        "passed": passed,
        "parameters": parameters,
        "cauchy_riemann_residuals": cr_residuals,
        "product_jet_error": float(jet_error),
        "laplace_residual": float(laplace_residual),
        "factor_gauge_imbalance": float(product.gauge_report(z).imbalance_ratio),
        "constraint_evidence": {
            "rank": prepared_constraints.evidence.rank,
            "nullity": prepared_constraints.evidence.nullity,
            "residual": constraint_residual,
            "tolerance": float(prepared_constraints.evidence.lift_tolerance),
            "preparation_seconds": constraint_preparation_seconds,
        },
        "constrained_laplace_residual": constrained_laplace_residual,
        "timings": timings,
    }
    payload["benchmark_id"] = canonical_fingerprint(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark certified separable holomorphic potential models."
    )
    parser.add_argument("--output", type=str, default=None)
    arguments = parser.parse_args()
    payload = run_holomorphic_separability_benchmarks()
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if arguments.output is None:
        print(rendered)
    else:
        with open(arguments.output, "w", encoding="utf-8") as stream:
            stream.write(rendered + "\n")


if __name__ == "__main__":
    main()
