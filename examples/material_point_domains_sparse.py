#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Build uGIMP/CPDI routes and compact active-block storage."""

import jax.numpy as jnp

import phydrax as phx


def run():
    grid_plan = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformAxisSpec(16, periodic=True, endpoint=False)
            for _ in range(2)
        ),
        axis_names=("x", "y"),
    )
    bounds = jnp.asarray([[0.0, 0.0], [1.0, 1.0]])
    index = grid_plan.prepare_index_space(bounds)
    position = jnp.asarray([[0.27, 0.31], [0.43, 0.38], [0.36, 0.52]])
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(3), jnp.ones((3,)), ambient_dimension=2
    ).prepare()
    gimp = phx.discretization.UniformGIMPSplatAssignment(jnp.full((3, 2), 0.025))
    prepared = phx.discretization.ParticleGridSplatPlan(index, assignment=gimp).prepare(
        particles
    )
    routes = prepared.build(position)
    topology = phx.discretization.SparseBlockTopologyPlan(
        index, (4, 4), 16, layout=index.vertices()
    )
    storage = phx.discretization.BlockSparseMPMNodalStoragePlan(topology)
    active = storage.build(routes)
    dense = jnp.arange(16 * 16, dtype=jnp.float64).reshape((16, 16))
    compact = storage.pack(dense, active)
    restored = storage.unpack(compact, active)
    parity = jnp.max(
        jnp.abs(
            jnp.where(
                active.materialize_support(),
                restored - dense,
                0.0,
            )
        )
    )
    return {
        "partition_defect": float(jnp.max(jnp.abs(routes.partition_sums - 1.0))),
        "gradient_defect": float(jnp.max(jnp.abs(routes.gradient_sums))),
        "active_blocks": int(active.evidence.required_blocks),
        "dense_sparse_parity": float(parity),
    }


if __name__ == "__main__":
    print(run())
