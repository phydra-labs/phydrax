#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import trimesh

import phydrax as phx
from phydrax.constraints._enforced import _enforced_constraint_weight_fn


def test_enforced_constraint_weight_fn_mesh_normals_jittable():
    geom = phx.domain.GeometryDomain(
        phx.geometry.mesh_region_from_source(
            trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        ).compile()
    )

    def where(point):
        return point[0] > 0.0

    w_fn = _enforced_constraint_weight_fn(
        geom,
        where,
        num_reference=2_000,
        sampler="uniform",
        key=jr.key(0),
    )

    points = jnp.array(
        [
            [0.6, 0.0, 0.0],
            [-0.6, 0.0, 0.0],
            [0.0, 0.6, 0.0],
            [0.0, 0.0, -0.6],
            [0.2, 0.3, 0.4],
        ],
        dtype=float,
    )
    values = jax.jit(jax.vmap(w_fn))(points)
    assert values.shape == (points.shape[0],)
    assert np.all(np.isfinite(np.asarray(values)))
