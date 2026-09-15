#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import math

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _onsager_log_partition(beta: float) -> float:
    nodes, weights = np.polynomial.legendre.leggauss(512)
    angles = 0.5 * math.pi * (nodes + 1.0)
    kappa = 2.0 * math.sinh(2.0 * beta) / math.cosh(2.0 * beta) ** 2
    radicand = np.maximum(0.0, 1.0 - kappa**2 * np.sin(angles) ** 2)
    integrand = np.log(0.5 * (1.0 + np.sqrt(radicand)))
    integral = 0.5 * math.pi * float(np.sum(weights * integrand))
    return math.log(2.0 * math.cosh(2.0 * beta)) + integral / (2.0 * math.pi)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("trg", "hotrg"), default="hotrg")
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5 * math.log(1.0 + math.sqrt(2.0)),
    )
    parser.add_argument("--maximum-bond-dimension", type=int, default=6)
    parser.add_argument("--steps", type=int, default=8)
    arguments = parser.parse_args()

    spins = jnp.asarray((-1.0, 1.0), dtype=jnp.float64)
    pair_weight = jnp.exp(arguments.beta * spins[:, None] * spins[None, :])
    tn = phx.tensor_network
    built = tn.build_uniform_pair_partition_tensor(pair_weight)
    method = tn.TRGMethod() if arguments.method == "trg" else tn.HOTRGMethod()
    problem = tn.TensorRenormalizationProblem(built.tensor)
    policy = tn.TensorRenormalizationPolicy(
        method,
        maximum_bond_dimension=arguments.maximum_bond_dimension,
        steps=arguments.steps,
    )
    plan = tn.plan_tensor_renormalization(problem, policy)
    result = tn.run_tensor_renormalization(
        tn.prepare_tensor_renormalization(problem, plan)
    )
    reference = _onsager_log_partition(arguments.beta)
    payload = {
        "method": arguments.method,
        "beta": arguments.beta,
        "maximum_bond_dimension": arguments.maximum_bond_dimension,
        "steps": arguments.steps,
        "log_partition_density": float(result.log_partition_density),
        "onsager_reference": reference,
        "absolute_error": abs(float(result.log_partition_density) - reference),
        "terminal_correction": float(result.terminal_correction),
        "admitted_peak_bytes": plan.cost.peak_workspace_bytes,
        "status": int(result.diagnostics.status),
        "accepted": bool(result.successful),
    }
    print(json.dumps(payload, indent=2))
    if not result.successful:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
