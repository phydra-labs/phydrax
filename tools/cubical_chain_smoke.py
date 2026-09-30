#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Exercise actual cubical PIC transfer, conservative current, and phase loads."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import phydrax.discretization as D
import phydrax.discretization.pic as PIC
from phydrax.discretization._cubical_whitney import CubicalSplineWhitneyKernel


def main() -> None:
    grid = D.TensorGridPlan(
        tuple(D.UniformCellAxisSpec(8) for _ in range(3)), axis_names=("x", "y", "z")
    ).prepare(jnp.asarray(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), dtype=jnp.float64))
    bridge = D.StructuredCochainBridge(grid)
    particles = D.ParticleSetPlan(
        jnp.arange(2), jnp.ones((2,), dtype=jnp.float64), ambient_dimension=3
    ).prepare()
    species = D.ChargedParticlePlan(
        jnp.asarray((1.7, -0.8), dtype=jnp.float64), "cubical-chain-smoke"
    ).prepare(particles)
    transfer = PIC.PICParticleCochainTransferPlan(bridge, shape_order=3).prepare(species)
    current = PIC.ChargeConservingCurrentPlan(transfer, maximum_segments_per_particle=12)
    first = jnp.asarray(((0.31, 0.36, 0.41), (0.64, 0.59, 0.53)), dtype=jnp.float64)
    last = jnp.asarray(((0.66, 0.61, 0.57), (0.34, 0.39, 0.43)), dtype=jnp.float64)
    result = jax.jit(lambda a, b: current.deposit(a, b, 0.2))(first, last)
    jax.block_until_ready(result)
    if not bool(result.successful):
        raise RuntimeError(
            f"Current fixture failed: continuity={result.maximum_continuity_defect}, overflow={result.capacity_overflow}"
        )
    shapes = bridge.orientation_shapes[1]
    electric = bridge.pack_edge_circulation(
        (jnp.full(shapes[0], 0.4), jnp.full(shapes[1], -1.2), jnp.full(shapes[2], 2.1))
    )
    gathered = transfer.gather_electric(transfer.build(first), electric)
    np.testing.assert_allclose(
        gathered.values, np.broadcast_to((0.4, -1.2, 2.1), (2, 3)), atol=3e-13
    )
    load = bridge.cochain.hodge_star(1, result.current)
    grid_work = jnp.vdot(electric, load) * 0.2
    physical_work = jnp.sum(
        species.charges[:, None] * (last - first) * jnp.asarray((0.4, -1.2, 2.1))
    )
    np.testing.assert_allclose(grid_work, physical_work, atol=3e-13)
    kernel = CubicalSplineWhitneyKernel(bridge, 3)
    phase = kernel.integrate_segments(
        first, last, weight="phase", phase_rate=7.0, maximum_segments=12
    )
    expected = np.asarray(last - first) @ np.asarray((0.4, -1.2, 2.1)) * np.expm1(7j) / 7j
    np.testing.assert_allclose(phase.gather(electric), expected, atol=3e-13)
    print(
        {
            "continuity_defect": float(result.maximum_continuity_defect),
            "grid_work": float(grid_work),
            "phase_integral": np.asarray(phase.gather(electric)).tolist(),
            "segments": int(result.segment_count),
            "overflow": bool(result.capacity_overflow),
        }
    )


if __name__ == "__main__":
    main()
