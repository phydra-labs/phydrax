#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp
import pytest

from phydrax._frozendict import frozendict
from phydrax.constraints import (
    enforce_blend,
    enforce_dirichlet,
    enforce_neumann,
    enforce_robin,
    enforce_sommerfeld,
    enforce_traction,
)
from phydrax.domain import (
    Boundary,
    FourierAxisSpec,
    Interval1d,
    PointsBatch,
    ProductStructure,
    TimeInterval,
)
from phydrax.operators.differential._domain_ops import directional_derivative


def _points_on_interval(geom: Interval1d, xs: jnp.ndarray) -> PointsBatch:
    structure = ProductStructure((("x",),)).canonicalize(geom.labels)
    axis_names = structure.axis_names
    assert axis_names is not None
    axis = axis_names[0]
    pts = xs.reshape((-1, 1))
    points = frozendict({"x": cx.Field(pts, dims=(axis, None))})
    return PointsBatch(points=points, structure=structure)


def test_enforce_dirichlet_enforces_values_on_boundary():
    geom = Interval1d(0.0, 1.0)
    component = geom.component({"x": Boundary()})

    @geom.Function("x")
    def u(x):
        return x[0]

    u_enforced = enforce_dirichlet(u, component, target=2.0)
    batch = _points_on_interval(geom, jnp.array([0.0, 1.0], dtype=float))
    out = jnp.asarray(u_enforced(batch).data).reshape((-1,))
    assert jnp.allclose(out, 2.0)


def test_enforce_neumann_enforces_normal_derivative_on_boundary():
    geom = Interval1d(0.0, 1.0)
    component = geom.component({"x": Boundary()})

    @geom.Function("x")
    def u(x):
        return x[0]

    u_enforced = enforce_neumann(u, component, target=0.0)
    n = component.normal(var="x")
    dd = directional_derivative(u_enforced, n, var="x")

    batch = _points_on_interval(geom, jnp.array([0.0, 1.0], dtype=float))
    out = jnp.asarray(dd(batch).data).reshape((-1,))
    assert jnp.allclose(out, 0.0, atol=1e-5)


def test_enforce_neumann_rejects_coord_separable_evaluation():
    geom = Interval1d(0.0, 1.0)
    boundary = geom.component({"x": Boundary()})

    @geom.Function("x")
    def u(x):
        return x[0]

    u_enforced = enforce_neumann(u, boundary, target=0.0)
    interior = geom.component()
    batch = interior.sample_coord_separable({"x": FourierAxisSpec(8)})
    with pytest.raises(
        ValueError, match="enforce_neumann does not support coord-separable"
    ):
        _ = u_enforced(batch)


def test_enforce_robin_rejects_coord_separable_evaluation():
    geom = Interval1d(0.0, 1.0)
    boundary = geom.component({"x": Boundary()})

    @geom.Function("x")
    def u(x):
        return x[0]

    u_enforced = enforce_robin(
        u,
        boundary,
        dirichlet_coeff=1.0,
        neumann_coeff=1.0,
        target=0.0,
    )
    interior = geom.component()
    batch = interior.sample_coord_separable({"x": FourierAxisSpec(8)})
    with pytest.raises(
        ValueError, match="enforce_robin does not support coord-separable"
    ):
        _ = u_enforced(batch)


def test_enforce_traction_rejects_coord_separable_evaluation():
    geom = Interval1d(0.0, 1.0)
    boundary = geom.component({"x": Boundary()})

    @geom.Function("x")
    def u(x):
        return jnp.asarray([x[0]], dtype=float)

    u_enforced = enforce_traction(
        u,
        boundary,
        target=jnp.zeros((1,), dtype=float),
        lambda_=1.0,
        mu=1.0,
    )
    interior = geom.component()
    batch = interior.sample_coord_separable({"x": FourierAxisSpec(8)})
    with pytest.raises(
        ValueError, match="enforce_traction does not support coord-separable"
    ):
        _ = u_enforced(batch)


def test_enforce_sommerfeld_rejects_coord_separable_evaluation():
    geom = Interval1d(0.0, 1.0)
    time = TimeInterval(0.0, 1.0)
    domain = geom @ time

    boundary = domain.component({"x": Boundary()})

    @domain.Function("x", "t")
    def u(x, t):
        return x[0] + t

    u_enforced = enforce_sommerfeld(
        u,
        boundary,
        var="x",
        time_var="t",
        wavespeed=1.0,
        target=0.0,
    )
    interior = domain.component()
    batch = interior.sample_coord_separable(
        {"x": FourierAxisSpec(8)},
        num_points=2,
        dense_structure=ProductStructure((("t",),)),
    )
    with pytest.raises(
        ValueError, match="enforce_sommerfeld does not support coord-separable"
    ):
        _ = u_enforced(batch)


def test_enforce_blend_combines_subset_pieces_without_leakage():
    geom = Interval1d(0.0, 1.0)

    def left_where(x):
        return x[0] < 0.5

    def right_where(x):
        return x[0] >= 0.5

    base = geom.Function()(0.0)
    left_component = geom.component({"x": Boundary()}, where={"x": left_where})
    right_component = geom.component({"x": Boundary()}, where={"x": right_where})

    left_piece = enforce_dirichlet(base, left_component, target=1.0)
    right_piece = enforce_dirichlet(base, right_component, target=2.0)

    blended = enforce_blend(
        base,
        [(left_component, left_piece), (right_component, right_piece)],
        num_reference=256,
    )

    batch = _points_on_interval(geom, jnp.array([0.0, 1.0], dtype=float))
    out = jnp.asarray(blended(batch).data).reshape((-1,))
    assert jnp.allclose(out[0], 1.0, atol=1e-3)
    assert jnp.allclose(out[1], 2.0, atol=1e-3)
