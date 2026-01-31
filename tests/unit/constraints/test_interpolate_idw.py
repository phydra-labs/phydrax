#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp

from phydrax._frozendict import frozendict
from phydrax.constraints._interpolate import idw_interpolant
from phydrax.domain import Interval1d, PointsBatch, ProductStructure


def test_idw_interpolant_snaps_at_anchors_and_interpolates_midpoint():
    geom = Interval1d(0.0, 1.0)
    anchors = {"x": jnp.array([[0.0], [1.0]], dtype=float)}
    values = jnp.array([0.0, 1.0], dtype=float)
    interp = idw_interpolant(geom, anchors=anchors, values=values, eps_snap=1e-12)

    structure = ProductStructure((("x",),)).canonicalize(geom.labels)
    axis_names = structure.axis_names
    assert axis_names is not None
    axis = axis_names[0]
    query = jnp.array([[0.0], [0.5], [1.0]], dtype=float)
    batch = PointsBatch(
        points=frozendict({"x": cx.Field(query, dims=(axis, None))}), structure=structure
    )

    out = jnp.asarray(interp(batch).data).reshape((-1,))
    assert jnp.allclose(out[0], 0.0)
    assert jnp.allclose(out[2], 1.0)
    assert jnp.allclose(out[1], 0.5, atol=1e-6)
