#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import itertools
import json

import jax.numpy as jnp

import phydrax as phx


def main() -> int:
    velocities = jnp.asarray(
        tuple(itertools.product((-2.0, -1.0, 0.0, 1.0, 2.0), repeat=3))
    )
    quadrature = phx.equations.MolecularVelocityQuadrature(
        velocities, jnp.ones((velocities.shape[0],)), 1
    )
    multipliers = jnp.asarray((-2.0, 0.1, -0.05, 0.02, -0.8))
    population = jnp.exp(quadrature.moment_features @ multipliers)
    moments = quadrature.moments(population)
    maxwellian = phx.equations.PositiveDiscreteMaxwellianPlan(quadrature).solve(moments)
    collision = phx.equations.MonatomicBGKCollisionPlan(
        quadrature, dynamic_viscosity=0.01
    ).advance(population, jnp.asarray(1.0e-3))
    maximum_maxwellian_residual = float(jnp.max(jnp.abs(maxwellian.moment_residual)))
    maximum_bgk_defect = float(jnp.max(jnp.abs(collision.moment_defect)))
    report = {
        "velocity_count": quadrature.velocity_count,
        "maximum_maxwellian_moment_residual": maximum_maxwellian_residual,
        "maximum_bgk_moment_defect": maximum_bgk_defect,
        "entropy_change": float(collision.entropy_change),
        "successful": bool(
            maxwellian.successful
            & collision.successful
            & jnp.isfinite(maximum_maxwellian_residual)
            & jnp.isfinite(maximum_bgk_defect)
            & (maximum_maxwellian_residual <= 1.0e-8)
            & (maximum_bgk_defect <= 1.0e-8)
            & (collision.entropy_change <= 1.0e-12)
        ),
    }
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
