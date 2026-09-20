#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
import time

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx


def _elapsed(callable_, repetitions: int = 10) -> float:
    started = time.perf_counter()
    value = None
    for _ in range(repetitions):
        value = callable_()
    jax.block_until_ready(value)
    return (time.perf_counter() - started) / repetitions


def main() -> None:
    mf = phx.applications.microfluidics
    geometry = mf.DLDGeometryPlan(
        mf.DLDTopology(
            row_count=16,
            column_count=4,
            period_rows=4,
            outlet_count=2,
            particle_capacity=4096,
        ),
        mf.DLDDesign(
            post_radius=0.08,
            axial_pitch=0.5,
            lateral_pitch=0.5,
            row_shift=0.125,
            channel_lower=0.0,
            channel_upper=3.0,
            first_row_x=0.5,
            outlet_x=9.0,
            depth=0.1,
            length_unit_id="mm",
        ),
    )
    key = jax.random.key(0)
    points = jax.random.uniform(key, (4096, 2), minval=0.0, maxval=3.0)
    radii = jnp.full((4096,), 0.01)
    geometry_call = eqx.filter_jit(geometry.evaluate)
    jax.block_until_ready(geometry_call(points, radii).clearance)

    species = phx.discretization.dsmc.DSMCSpeciesPlan(
        ("A",),
        jnp.asarray((1.0,)),
        jnp.asarray((1.0,)),
        jnp.asarray((0.5,)),
        jnp.asarray((0.0,)),
        jnp.asarray((0.0,)),
        jnp.asarray((0.0,)),
    )
    collision = phx.discretization.dsmc.DSMCVHSCollisionPlan(
        species,
        phx.discretization.dsmc.DSMCPairCollisionParameters(
            jnp.asarray(((1.0,),)),
            jnp.asarray(((0.5,),)),
            boltzmann_constant=1.0,
        ),
    )
    capacity = 1024
    state = phx.discretization.dsmc.DSMCParticleState(
        jnp.zeros((capacity, 3)),
        jax.random.normal(jax.random.key(1), (capacity, 3)),
        jnp.zeros((capacity,), dtype=jnp.int32),
        jnp.zeros((capacity,)),
        jnp.zeros((capacity,)),
        jnp.ones((capacity,)),
        jnp.zeros((capacity,), dtype=jnp.int32),
        jnp.ones((capacity,), dtype="bool"),
        jnp.zeros((capacity,), dtype=jnp.int32),
    )
    event_count = 1024
    first = jnp.arange(event_count, dtype=jnp.int32) % capacity
    second = (first + 1) % capacity
    valid = jnp.ones((event_count,), dtype="bool")
    uniforms = jax.random.uniform(jax.random.key(2), (event_count, 3))
    majorant = jnp.full((event_count,), 100.0)
    collision_call = eqx.filter_jit(collision.collide)
    jax.block_until_ready(
        collision_call(state, first, second, valid, uniforms, majorant).state.velocity
    )

    result = {
        "kind": "flow-regime-benchmark",
        "timing_is_not_a_release_gate": True,
        "dld_geometry_seconds": _elapsed(lambda: geometry_call(points, radii).clearance),
        "dsmc_1024_events_seconds": _elapsed(
            lambda: (
                collision_call(
                    state, first, second, valid, uniforms, majorant
                ).state.velocity
            )
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
