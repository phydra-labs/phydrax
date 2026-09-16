from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from benchmarks._runtime import capture_environment, logical_array_bytes, measure_repeated


def main() -> int:
    count = 1024
    layout = phx.atomistic.PolymerChainLayoutPlan(
        np.arange(count)[None, :], np.ones((1, count), dtype=bool), maximum_frames=1
    )
    positions = jnp.stack(
        (jnp.arange(count, dtype=float), jnp.zeros(count), jnp.zeros(count)), axis=-1
    )[None, ...]
    operation = jax.jit(lambda value: phx.atomistic.polymer_conformation(layout, value))
    result, timing = measure_repeated(lambda: operation(positions), warmup=1, repeats=3)
    payload = {
        "benchmark": "polymer-particle-conformation",
        "particle_count": count,
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
