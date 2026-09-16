#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json

import jax.numpy as jnp

import phydrax as phx
from benchmarks._runtime import measure_repeated
from phydrax.linalg import DensePropertyVerificationPolicy, verify_dense_properties


def main() -> None:
    matrix = jnp.eye(16) + 0.01 * jnp.ones((16, 16))
    _, property_timing = measure_repeated(
        lambda: verify_dense_properties(
            matrix,
            policy=DensePropertyVerificationPolicy(require_positive_definite=True),
        ),
        warmup=1,
        repeats=3,
    )
    plan = phx.nonlinear.VectorLocalRootPlan(2, plan_id="benchmark-local-root")
    _, root_timing = measure_repeated(
        lambda: plan.solve(
            lambda value: jnp.asarray((value[0] ** 2 - 4.0, value[1] - 3.0)),
            jnp.asarray((1.0, 0.0)),
        ),
        warmup=1,
        repeats=3,
    )
    evolution = phx.dynamics.PreparedAffineLinearEvolution(
        jnp.zeros((2, 2)), jnp.ones((2,))
    )
    _, evolution_timing = measure_repeated(
        lambda: evolution.step(jnp.zeros((2,)), jnp.asarray(0.5)),
        warmup=1,
        repeats=3,
    )
    print(
        json.dumps(
            {
                "matrix_property_median_seconds": property_timing.median_seconds,
                "local_root_median_seconds": root_timing.median_seconds,
                "affine_evolution_median_seconds": evolution_timing.median_seconds,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
