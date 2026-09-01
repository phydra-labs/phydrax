#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Resolved-subtraction lubrication curve for two approaching immersed spheres."""

import jax.numpy as jnp

import phydrax as phx


def main() -> None:
    plan = phx.discretization.ResolvedLubricationCorrectionPlan(
        1.0e-3,
        5.0e-2,
        1.0e-4,
    )
    gaps = jnp.geomspace(2.0e-4, 4.0e-2, 32)
    normal_speed = -jnp.full_like(gaps, 1.0e-2)
    result = plan.evaluate(
        gaps,
        jnp.broadcast_to(jnp.asarray([1.0, 0.0, 0.0]), gaps.shape + (3,)),
        normal_speed,
        jnp.full_like(gaps, 1.0e-2),
        resolved_resistance=jnp.full_like(gaps, 2.0e-6),
    )
    if not bool(result.finite):
        raise RuntimeError("Lubrication evaluation produced nonfinite evidence.")
    if not bool(jnp.all(result.dissipation_rate >= 0.0)):
        raise RuntimeError("Lubrication correction violated dissipation.")
    print(
        {
            "minimum_gap": float(gaps[0]),
            "maximum_resistance": float(jnp.max(result.resistance)),
            "integrated_dissipation_rate": float(jnp.sum(result.dissipation_rate)),
        }
    )


if __name__ == "__main__":
    main()
