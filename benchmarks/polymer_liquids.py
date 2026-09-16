from __future__ import annotations

import json

import equinox as eqx
import jax.numpy as jnp

from benchmarks._runtime import capture_environment, logical_array_bytes, measure_repeated
from phydrax.applications import polymer_liquids as pl


def main() -> int:
    count = 256
    transform = pl.IsotropicRadialTransformPlan(count, 32.0).prepare()
    prepared = pl.PRISMPlan(pl.PRISMClosurePlan(pl.PRISMClosureKind.HNC)).prepare(
        transform,
        pl.SiteMixturePlan(("A",), [0.1]),
        pl.SequenceFormFactorPlan("gaussian-chain", [0], 1.0, site_count=1),
        pl.SitePairPotentialPlan(
            transform.radii, jnp.zeros((1, 1, count)), source_id="benchmark"
        ),
    )
    gamma = jnp.zeros((1, 1, count))
    operation = eqx.filter_jit(prepared.evaluate)
    result, timing = measure_repeated(lambda: operation(gamma), warmup=1, repeats=3)
    payload = {
        "benchmark": "polymer-prism-evaluation",
        "radial_count": count,
        "site_count": 1,
        "timing": timing.to_milliseconds_dict(),
        "logical_bytes": logical_array_bytes(result),
        "successful": bool(result.successful),
        "environment": capture_environment().to_dict(),
        "release_claim": False,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
