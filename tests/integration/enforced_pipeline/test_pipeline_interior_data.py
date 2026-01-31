#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp

from phydrax._frozendict import frozendict
from phydrax.domain import Interval1d, PointsBatch, ProductStructure, TimeInterval
from phydrax.solver import EnforcedConstraintPipelines, EnforcedInteriorData


def _paired_batch(domain, xs, ts):
    structure = ProductStructure((("x", "t"),)).canonicalize(domain.labels)
    axis_names = structure.axis_names
    assert axis_names is not None
    axis = axis_names[0]
    points = frozendict(
        {
            "x": cx.Field(
                jnp.asarray(xs, dtype=float).reshape((-1, 1)), dims=(axis, None)
            ),
            "t": cx.Field(jnp.asarray(ts, dtype=float).reshape((-1,)), dims=(axis,)),
        }
    )
    return PointsBatch(points=points, structure=structure)


def test_unified_interior_data_tracks_and_scattered():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    domain = geom @ time

    @domain.Function("x", "t")
    def u(x, t):
        return x[0] * 0.0 + t * 0.0

    scattered = EnforcedInteriorData(
        "u",
        points={
            "x": jnp.array([[0.2], [0.8]], dtype=float),
            "t": jnp.array([0.3, 0.7], dtype=float),
        },
        values=jnp.array([1.0, 2.0], dtype=float),
        use_envelope=True,
        envelope_scale=0.5,
    )

    tracks = EnforcedInteriorData(
        "u",
        sensors=jnp.array([[0.5]], dtype=float),
        times=jnp.array([0.2, 0.6], dtype=float),
        sensor_values=jnp.array([[3.0, 4.0]], dtype=float),
    )

    pipelines = EnforcedConstraintPipelines.build(
        functions={"u": u},
        interior_data=[scattered, tracks],
    )
    u_enforced = pipelines.apply({"u": u})["u"]

    xs = jnp.array([0.2, 0.8, 0.5, 0.5], dtype=float)
    ts = jnp.array([0.3, 0.7, 0.2, 0.6], dtype=float)
    expected = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=float)

    batch = _paired_batch(domain, xs=xs, ts=ts)
    out = jnp.asarray(u_enforced(batch).data).reshape((-1,))
    assert jnp.allclose(out, expected, atol=1e-3)


def test_enforced_interior_data_hermite_track_matches_curve():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    domain = geom @ time

    @domain.Function("x", "t")
    def u(x, t):
        return x[0] * 0.0 + t * 0.0

    tracks = EnforcedInteriorData(
        "u",
        sensors=jnp.array([[0.5]], dtype=float),
        times=jnp.array([0.0, 1.0], dtype=float),
        sensor_values=jnp.array([[0.0, 1.0]], dtype=float),
        time_interp="hermite",
    )

    pipelines = EnforcedConstraintPipelines.build(
        functions={"u": u},
        interior_data=[tracks],
    )
    u_enforced = pipelines.apply({"u": u})["u"]

    xs = jnp.array([0.5, 0.5], dtype=float)
    ts = jnp.array([0.0, 0.5], dtype=float)
    expected = jnp.array([0.0, 0.5], dtype=float)

    batch = _paired_batch(domain, xs=xs, ts=ts)
    out = jnp.asarray(u_enforced(batch).data).reshape((-1,))
    assert jnp.allclose(out, expected, atol=1e-3)
