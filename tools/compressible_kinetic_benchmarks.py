#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp

from phydrax.discretization.discrete_velocity import guided_d3q39_plan


def benchmark(size: int, repetitions: int) -> dict[str, object]:
    shape = (size, size, size)
    model = guided_d3q39_plan()
    state = model.initialize(jnp.ones(shape), jnp.zeros(shape + (3,)), jnp.ones(shape))
    collision = jax.jit(lambda value: model.collide(value, 1.0))
    lower_started = time.perf_counter()
    lowered = collision.lower(state)
    lowering_seconds = time.perf_counter() - lower_started
    compile_started = time.perf_counter()
    executable = lowered.compile()
    compilation_seconds = time.perf_counter() - compile_started
    warmup = executable(state)
    jax.block_until_ready(warmup.accepted.populations[0])
    started = time.perf_counter()
    result = warmup
    for _ in range(repetitions):
        result = executable(result.accepted)
    jax.block_until_ready(result.accepted.populations[0])
    elapsed = time.perf_counter() - started
    cells = size**3 * repetitions
    state_bytes = sum(value.size * value.dtype.itemsize for value in state.populations)
    state_bytes += state.equilibrium_dual.size * state.equilibrium_dual.dtype.itemsize
    return {
        "kind": "compressible-kinetic-benchmark",
        "backend": jax.default_backend(),
        "grid_shape": list(shape),
        "repetitions": repetitions,
        "lowering_seconds": lowering_seconds,
        "compilation_seconds": compilation_seconds,
        "warmed_seconds": elapsed,
        "million_cell_collisions_per_second": cells / elapsed / 1.0e6,
        "logical_state_bytes": state_bytes,
        "successful": bool(jnp.all(result.successful)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.size < 1 or args.repetitions < 1:
        raise ValueError("size and repetitions must be positive.")
    print(json.dumps(benchmark(args.size, args.repetitions), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
