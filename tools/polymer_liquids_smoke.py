from __future__ import annotations

import json

import jax.numpy as jnp

from phydrax.applications import polymer_liquids as pl


def main() -> int:
    transform = pl.IsotropicRadialTransformPlan(16, 8.0).prepare()
    prepared = pl.PRISMPlan(
        pl.PRISMClosurePlan(pl.PRISMClosureKind.HNC), maximum_iterations=16
    ).prepare(
        transform,
        pl.SiteMixturePlan(("A",), [0.1]),
        pl.SequenceFormFactorPlan("gaussian-chain", [0], 1.0, site_count=1),
        pl.SitePairPotentialPlan(
            transform.radii, jnp.zeros((1, 1, 16)), source_id="ideal-smoke"
        ),
    )
    result = pl.solve_prism(prepared)
    successful = bool(result.successful)
    print(
        json.dumps(
            {
                "successful": successful,
                "maximum_residual": float(jnp.max(jnp.abs(result.evaluation.residual))),
            }
        )
    )
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
