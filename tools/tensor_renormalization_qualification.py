#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from benchmarks._runtime import capture_environment


_CRITICAL_BETA = 0.5 * math.log(1.0 + math.sqrt(2.0))


def _onsager_log_partition(beta: float) -> float:
    nodes, weights = np.polynomial.legendre.leggauss(512)
    angles = 0.5 * math.pi * (nodes + 1.0)
    kappa = 2.0 * math.sinh(2.0 * beta) / math.cosh(2.0 * beta) ** 2
    radicand = np.maximum(0.0, 1.0 - kappa**2 * np.sin(angles) ** 2)
    integrand = np.log(0.5 * (1.0 + np.sqrt(radicand)))
    integral = 0.5 * math.pi * float(np.sum(weights * integrand))
    return math.log(2.0 * math.cosh(2.0 * beta)) + integral / (2.0 * math.pi)


def _ising_tensor(beta: float):
    spins = jnp.asarray((-1.0, 1.0), dtype=jnp.float64)
    pair_weight = jnp.exp(beta * spins[:, None] * spins[None, :])
    return phx.tensor_network.build_uniform_pair_partition_tensor(pair_weight).tensor


def _method_case(name: str, beta: float, bond: int, steps: int):
    tn = phx.tensor_network
    method = tn.TRGMethod() if name == "trg" else tn.HOTRGMethod()
    problem = tn.TensorRenormalizationProblem(_ising_tensor(beta))
    policy = tn.TensorRenormalizationPolicy(
        method,
        maximum_bond_dimension=bond,
        steps=steps,
    )
    plan = tn.plan_tensor_renormalization(problem, policy)
    result = tn.run_tensor_renormalization(
        tn.prepare_tensor_renormalization(problem, plan)
    )
    computed = float(result.log_partition_density)
    reference = _onsager_log_partition(beta)
    absolute_error = abs(computed - reference)
    return {
        "case": f"ising-beta-{beta:.12g}",
        "method": name,
        "maximum_bond_dimension": bond,
        "steps": steps,
        "computed_log_partition_density": computed,
        "reference_log_partition_density": reference,
        "absolute_error": absolute_error,
        "first_discarded_weight_sum": float(
            jnp.sum(result.diagnostics.first_discarded_weight_history)
        ),
        "second_discarded_weight_sum": float(
            jnp.sum(result.diagnostics.second_discarded_weight_history)
        ),
        "terminal_correction": float(result.terminal_correction),
        "admitted_peak_bytes": plan.cost.peak_workspace_bytes,
        "status": int(result.diagnostics.status),
        "finite": bool(jnp.all(result.diagnostics.finite_history)),
        "passed": bool(result.successful) and absolute_error < 5e-3,
    }


def _q2_potts_case(bond: int, steps: int):
    tn = phx.tensor_network
    beta = 2.0 * _CRITICAL_BETA
    pair_weight = jnp.asarray(
        ((math.exp(beta), 1.0), (1.0, math.exp(beta))),
        dtype=jnp.float64,
    )
    tensor = tn.build_uniform_pair_partition_tensor(pair_weight).tensor
    result = tn.run_tensor_renormalization(
        tn.TensorRenormalizationProblem(tensor),
        tn.TensorRenormalizationPolicy(
            tn.TRGMethod(),
            maximum_bond_dimension=bond,
            steps=steps,
        ),
    )
    computed = float(result.log_partition_density)
    reference = beta + _onsager_log_partition(0.5 * beta)
    error = abs(computed - reference)
    return {
        "case": "q2-potts-to-ising-mapping",
        "method": "trg",
        "computed_log_partition_density": computed,
        "reference_log_partition_density": reference,
        "absolute_error": error,
        "status": int(result.diagnostics.status),
        "passed": bool(result.successful) and error < 5e-3,
    }


def _product_case():
    tn = phx.tensor_network
    tensor = tn.UniformSquareTensor(jnp.asarray([[[[2.0]]]], dtype=jnp.float64))
    result = tn.run_tensor_renormalization(
        tn.TensorRenormalizationProblem(tensor),
        tn.TensorRenormalizationPolicy(
            tn.TRGMethod(),
            maximum_bond_dimension=1,
            steps=4,
        ),
    )
    error = abs(float(result.log_partition_density) - math.log(2.0))
    return {
        "case": "unit-bond-product",
        "computed_log_partition_density": float(result.log_partition_density),
        "reference_log_partition_density": math.log(2.0),
        "absolute_error": error,
        "status": int(result.diagnostics.status),
        "exact": bool(result.diagnostics.exact),
        "passed": bool(result.successful)
        and bool(result.diagnostics.exact)
        and error < 1e-12,
    }


def _nonfinite_case():
    tn = phx.tensor_network
    tensor = tn.UniformSquareTensor(jnp.asarray([[[[jnp.nan]]]], dtype=jnp.float64))
    result = tn.run_tensor_renormalization(
        tn.TensorRenormalizationProblem(tensor),
        tn.TensorRenormalizationPolicy(
            tn.TRGMethod(),
            maximum_bond_dimension=1,
            steps=1,
        ),
    )
    passed = int(result.diagnostics.status) == int(
        tn.TensorRenormalizationStatus.NONFINITE_INPUT
    ) and not bool(result.successful)
    return {
        "case": "nonfinite-input-refusal",
        "status": int(result.diagnostics.status),
        "accepted": bool(result.successful),
        "passed": passed,
    }


def _resource_refusal_case():
    tn = phx.tensor_network
    problem = tn.TensorRenormalizationProblem(_ising_tensor(0.2))
    refused = False
    try:
        tn.plan_tensor_renormalization(
            problem,
            tn.TensorRenormalizationPolicy(
                tn.TRGMethod(),
                maximum_bond_dimension=4,
                steps=1,
                resources=tn.TensorRenormalizationResourcePolicy(
                    maximum_factorization_elements=1
                ),
            ),
        )
    except MemoryError:
        refused = True
    return {
        "case": "factorization-resource-refusal",
        "refused_before_execution": refused,
        "passed": refused,
    }


def qualification(*, maximum_bond_dimension: int = 6, steps: int = 8):
    betas = (0.2, 0.3, _CRITICAL_BETA, 0.6)
    cases = [
        _method_case(method, beta, maximum_bond_dimension, steps)
        for method in ("trg", "hotrg")
        for beta in betas
    ]
    cases.extend(
        (
            _q2_potts_case(maximum_bond_dimension, steps),
            _product_case(),
            _nonfinite_case(),
            _resource_refusal_case(),
        )
    )
    return {
        "environment": capture_environment().to_dict(),
        "configuration": {
            "maximum_bond_dimension": maximum_bond_dimension,
            "steps": steps,
        },
        "cases": cases,
        "passed": all(case["passed"] for case in cases),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maximum-bond-dimension", type=int, default=6)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.maximum_bond_dimension < 1 or arguments.steps < 1:
        raise ValueError("Qualification bond dimension and steps must be positive.")
    payload = qualification(
        maximum_bond_dimension=arguments.maximum_bond_dimension,
        steps=arguments.steps,
    )
    encoded = json.dumps(payload, indent=2)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n")
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
