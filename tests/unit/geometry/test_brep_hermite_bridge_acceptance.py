# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Exact, whole-span contracts for the fixed-branch rational bridge."""

from fractions import Fraction as F
from math import factorial
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.geometry.brep._patches import BSplineSurfacePatch
from phydrax.geometry.brep._hermite_bridge import build_hermite_bridge
from phydrax.geometry.brep._bridge_embedding_proof import (
    BridgeEmbeddingEvidence,
    BridgeEmbeddingRefusal,
    BridgeSeparation,
    certify_bridge_embedding,
)


from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry.brep._constructors import _Builder, assemble_brep_model
from phydrax.geometry.brep._patches import BSplineCurve, ExactLineCurve
from phydrax.geometry.brep._model import BRepModel
from phydrax.discretization._exact_algebraic_geometry import ExactScalar


def _rational(value: ExactScalar) -> F:
    assert isinstance(value, F)
    return value


def _rational_banks(
    patch: BSplineSurfacePatch,
) -> tuple[tuple[tuple[tuple[F, F, F], ...], ...], tuple[tuple[F, ...], ...]]:
    points, weights = patch.exact_control_point_bank, patch.exact_weight_bank
    assert points is not None and weights is not None
    rational_points = tuple(
        tuple((_rational(p[0]), _rational(p[1]), _rational(p[2])) for p in row)
        for row in points
    )
    rational_weights = tuple(tuple(_rational(w) for w in row) for row in weights)
    return rational_points, rational_weights


def _curved_sources() -> tuple[BSplineSurfacePatch, BSplineSurfacePatch]:
    """Distinct cubic rational graph patches with nonconstant positive weights."""
    u = (F(0),) * 4 + (F(1),) * 4
    v = (F(0),) * 3 + (F(1),) * 3
    weights = tuple((w,) * 3 for w in (F(1), F(2), F(1), F(1)))

    def patch(offset: int, heights: tuple[F, ...]) -> BSplineSurfacePatch:
        points = tuple(
            tuple(
                (F(i, 3), F(offset) + F(j, 2), heights[i] + F(j * j, 20))
                for j in range(3)
            )
            for i in range(4)
        )
        return BSplineSurfacePatch(
            np.asarray(points, dtype=np.float64),
            np.asarray(weights, dtype=np.float64),
            np.asarray(u, dtype=np.float64),
            np.asarray(v, dtype=np.float64),
            3,
            2,
            exact_control_points=points,
            exact_weights=weights,
            exact_u_knots=u,
            exact_v_knots=v,
        )

    return patch(-1, (F(0), F(1, 4), F(-1, 5), F(1, 3))), patch(
        1, (F(1, 5), F(-1, 3), F(2, 5), F(0))
    )


def _bridge(
    left: BSplineSurfacePatch,
    right: BSplineSurfacePatch,
    continuity: int = 1,
    *,
    left_along: tuple[F, F] = (F(0), F(1)),
    right_along: tuple[F, F] = (F(0), F(1)),
    left_collar: F = F(1),
    right_collar: F = F(1),
) -> BSplineSurfacePatch:
    return build_hermite_bridge(
        left,
        right,
        left_side="v1",
        right_side="v0",
        left_along=left_along,
        right_along=right_along,
        left_collar=left_collar,
        right_collar=right_collar,
        continuity=continuity,
    )


def _source_model(patch: BSplineSurfacePatch, source_id: str) -> BRepModel:
    """One unplaced, open rectangular face with exact spline boundary lifts."""
    points, weights = _rational_banks(patch)
    u_knots, v_knots = patch.exact_u_knot_bank, patch.exact_v_knot_bank
    assert u_knots is not None and v_knots is not None
    builder = _Builder()
    corners = (points[0][0], points[-1][0], points[-1][-1], points[0][-1])
    vertices = tuple(builder.vertex(tuple(map(float, p)), exact_point=p) for p in corners)
    paths = (
        (
            tuple((i, 0) for i in range(4)),
            (F(0), F(0)),
            (F(1), F(0)),
            u_knots,
            3,
        ),
        (
            tuple((3, j) for j in range(3)),
            (F(1), F(0)),
            (F(1), F(1)),
            v_knots,
            2,
        ),
        (
            tuple((i, 2) for i in reversed(range(4))),
            (F(1), F(1)),
            (F(0), F(1)),
            u_knots,
            3,
        ),
        (
            tuple((0, j) for j in reversed(range(3))),
            (F(0), F(1)),
            (F(0), F(0)),
            v_knots,
            2,
        ),
    )
    loop = []
    for index, (indices, first, last, knots, degree) in enumerate(paths):
        cp = tuple(points[i][j] for i, j in indices)
        w = tuple(weights[i][j] for i, j in indices)
        curve = BSplineCurve(
            np.asarray(cp, dtype=np.float64),
            np.asarray(w, dtype=np.float64),
            np.asarray(knots, dtype=np.float64),
            degree,
            exact_control_points=cp,
            exact_weights=w,
            exact_knots=knots,
        )
        edge = builder.edge(curve, 0.0, 1.0, vertices[index], vertices[(index + 1) % 4])
        loop.append(builder.coedge(edge, 1, ExactLineCurve(first, last)))
    builder.face(patch, np.asarray(((0.0, 0.0), (1.0, 1.0))), 1, source_id, [loop])
    geometry, patches, bounds, orientations, tags, trims = builder.geometry(solid=False)
    return assemble_brep_model(
        geometry,
        patches,
        bounds,
        orientations,
        tags,
        coordinate_contract=SpatialCoordinateContract.si(),
        source_id=source_id,
        trim_domains=trims,
    )


def _curved_source_models() -> tuple[BRepModel, BRepModel]:
    left, right = _curved_sources()
    return _source_model(left, "bridge-left"), _source_model(right, "bridge-right")


def _homogeneous(patch: BSplineSurfacePatch) -> tuple[tuple[tuple[F, ...], ...], ...]:
    points, weights = _rational_banks(patch)
    return tuple(
        tuple(tuple(x * w for x in p) + (w,) for p, w in zip(row, wr))
        for row, wr in zip(points, weights)
    )


def _polynomial_product(a: list[F], b: list[F]) -> list[F]:
    result = [F(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            result[i + j] += x * y
    return result


def _rank(rows: list[list[F]]) -> int:
    rows = [list(row) for row in rows]
    rank = 0
    for column in range(len(rows[0])):
        pivot = next((i for i in range(rank, len(rows)) if rows[i][column]), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        scale = rows[rank][column]
        rows[rank] = [x / scale for x in rows[rank]]
        for i in range(rank + 1, len(rows)):
            factor = rows[i][column]
            rows[i] = [x - factor * y for x, y in zip(rows[i], rows[rank])]
        rank += 1
        if rank == len(rows):
            break
    return rank


def test_source_seams_are_distinct_nonconic_rational_curves() -> None:
    left, right = _curved_sources()
    for source, endpoint in ((left, -1), (right, 0)):
        h = tuple(row[endpoint] for row in _homogeneous(source))

        # Convert cubic Bernstein coordinates to the power basis exactly.
        def power(component: int) -> list[F]:
            a, b, c, d = (p[component] for p in h)
            return [a, 3 * (b - a), 3 * (a - 2 * b + c), -a + 3 * b - 3 * c + d]

        x, z, w = (power(i) for i in (0, 2, 3))
        assert len(set(p[3] for p in h)) > 1
        # A conic would give a nonzero linear relation among X², XZ, Z²,
        # XW, ZW, W². Full exact rank excludes every such relation.
        products = [
            _polynomial_product(a, b)
            for a, b in ((x, x), (x, z), (z, z), (x, w), (z, w), (w, w))
        ]
        assert _rank(products) == 6
    assert tuple(row[-1] for row in _homogeneous(left)) != tuple(
        row[0] for row in _homogeneous(right)
    )


def _endpoint_jet(
    row: tuple[tuple[F, ...], ...], order: int, high: bool
) -> tuple[F, ...]:
    """Exact Bezier endpoint derivative, independently of the builder."""
    degree = len(row) - 1
    values = list(row)
    for _ in range(order):
        values = [tuple(b - a for a, b in zip(x, y)) for x, y in zip(values, values[1:])]
    scale = factorial(degree) // factorial(degree - order)
    return tuple(scale * x for x in values[-1 if high else 0])


def _along_derivative(rows: tuple[tuple[F, ...], ...]) -> tuple[tuple[F, ...], ...]:
    degree = len(rows) - 1
    return tuple(
        tuple(degree * (b - a) for a, b in zip(x, y)) for x, y in zip(rows, rows[1:])
    )


@pytest.mark.parametrize("continuity", [1, 2])
def test_exact_full_span_seam_and_mixed_jets(continuity: int) -> None:
    left, right = _curved_sources()
    before = tuple(
        (
            np.array(p.control_points),
            np.array(p.weights),
            p.exact_control_point_bank,
            p.exact_weight_bank,
        )
        for p in (left, right)
    )
    bridge = _bridge(left, right, continuity)
    _, bridge_weights = _rational_banks(bridge)
    assert all(w > 0 for row in bridge_weights for w in row)
    assert bridge.v_degree == 2 * continuity + 1
    # Equality of the complete along-Bernstein coefficient banks proves the
    # identities over the entire seam, not just at a collection of samples.
    for source, high in ((left, False), (right, True)):
        sh, bh = _homogeneous(source), _homogeneous(bridge)
        for order in range(continuity + 1):
            source_jets = tuple(_endpoint_jet(row, order, not high) for row in sh)
            bridge_jets = tuple(_endpoint_jet(row, order, high) for row in bh)
            assert bridge_jets == source_jets
            for _ in range(2):
                source_jets = _along_derivative(source_jets)
                bridge_jets = _along_derivative(bridge_jets)
                assert bridge_jets == source_jets
    for patch, snapshot in zip((left, right), before):
        np.testing.assert_array_equal(patch.control_points, snapshot[0])
        np.testing.assert_array_equal(patch.weights, snapshot[1])
        assert (patch.exact_control_point_bank, patch.exact_weight_bank) == snapshot[2:]


def _embedding_args() -> dict[str, Any]:
    projection = ((F(1), F(0), F(0)), (F(0), F(1), F(0)))
    return dict(
        projection=projection,
        source_projections=(projection, projection),
        seams=(
            BridgeSeparation((1, 0), (1, 1), (F(0), F(1), F(0)), F(0)),
            BridgeSeparation((1, 1), (1, 0), (F(0), F(-1), F(0)), F(-1)),
        ),
        source_separator=((F(0), F(-1), F(0)), F(-1, 2)),
    )


@pytest.mark.parametrize("continuity", [1, 2])
def test_exact_embedding_and_source_separation(continuity: int) -> None:
    left, right = _curved_sources()
    evidence = certify_bridge_embedding(
        _bridge(left, right, continuity), left, right, **_embedding_args()
    )
    assert isinstance(evidence, BridgeEmbeddingEvidence), evidence


def test_embedding_refuses_singular_projection_and_wrong_seam() -> None:
    left, right = _curved_sources()
    bridge = _bridge(left, right)
    args = _embedding_args()
    args["projection"] = ((F(1), F(0), F(0)), (F(1), F(0), F(0)))
    assert isinstance(
        certify_bridge_embedding(bridge, left, right, **args), BridgeEmbeddingRefusal
    )
    args = _embedding_args()
    args["seams"] = (
        BridgeSeparation((1, 0), (1, 0), (F(0), F(1), F(0)), F(0)),
        args["seams"][1],
    )
    assert isinstance(
        certify_bridge_embedding(bridge, left, right, **args), BridgeEmbeddingRefusal
    )


def test_branch_and_refinement_refusal() -> None:
    left, right = _curved_sources()
    with pytest.raises(ValueError, match="full along-domain"):
        _bridge(left, right, left_along=(F(0), F(1, 2)))
    with pytest.raises(ValueError, match="positive"):
        _bridge(left, right, left_collar=F(0))
    # Reversal is an explicit branch, not an inferred seam correspondence.
    reversed_bridge = _bridge(left, right, right_along=(F(1), F(0)))
    assert _homogeneous(reversed_bridge)[0][-1] == _homogeneous(right)[-1][0]
    # Same geometry with a different along degree must not be silently refined.
    points, weights = _rational_banks(right)
    cp = points[:3]
    w = weights[:3]
    knots = (F(0),) * 3 + (F(1),) * 3
    other = BSplineSurfacePatch(
        np.asarray(cp, dtype=np.float64),
        np.asarray(w, dtype=np.float64),
        np.asarray(knots, dtype=np.float64),
        right.v_knots,
        2,
        2,
        exact_control_points=cp,
        exact_weights=w,
        exact_u_knots=knots,
        exact_v_knots=right.exact_v_knot_bank,
    )
    with pytest.raises(ValueError, match="refinement"):
        _bridge(left, other)


def test_unprovable_positive_denominator_is_refused() -> None:
    left, right = _curved_sources()
    _, source_weights = _rational_banks(left)
    weights = tuple((row[0], 10 * row[1], row[2]) for row in source_weights)
    steep = BSplineSurfacePatch(
        left.control_points,
        np.asarray(weights, dtype=np.float64),
        left.u_knots,
        left.v_knots,
        left.u_degree,
        left.v_degree,
        exact_control_points=left.exact_control_point_bank,
        exact_weights=weights,
        exact_u_knots=left.exact_u_knot_bank,
        exact_v_knots=left.exact_v_knot_bank,
    )
    with pytest.raises(ValueError, match="positive Bernstein weights"):
        _bridge(steep, right, left_collar=F(4))


def test_fixed_branch_jvp_vjp_transpose_identity() -> None:
    left, right = _curved_sources()
    direction = (
        jnp.arange(left.control_points.size, dtype=left.control_points.dtype).reshape(
            left.control_points.shape
        )
        / 37
    )

    def operation(points: jax.Array) -> jax.Array:
        dynamic_left = eqx.tree_at(lambda p: p.control_points, left, points)
        return _bridge(dynamic_left, right, 2).control_points

    primal, tangent = jax.jvp(operation, (left.control_points,), (direction,))
    np.testing.assert_allclose(
        tangent[:, 0, :], direction[:, -1, :], rtol=1e-12, atol=1e-12
    )
    cotangent = jnp.arange(primal.size, dtype=primal.dtype).reshape(primal.shape) / 29
    _, pullback = jax.vjp(operation, left.control_points)
    (gradient,) = pullback(cotangent)
    assert float(jnp.linalg.norm(tangent)) > 0
    np.testing.assert_allclose(
        jnp.vdot(tangent, cotangent),
        jnp.vdot(direction, gradient),
        rtol=1e-10,
        atol=1e-10,
    )


@pytest.mark.parametrize("continuity", [1, 2])
def test_public_bridge_publishes_exact_curved_sources(continuity: int) -> None:
    from phydrax.geometry.brep import brep_surface_bridge

    left, right = _curved_source_models()
    result = brep_surface_bridge(
        left,
        right,
        left.face_ids[0],
        left.edge_ids[2],
        right.face_ids[0],
        right.edge_ids[0],
        left_side="v1",
        right_side="v0",
        left_along=(F(0), F(1)),
        right_along=(F(0), F(1)),
        left_collar=F(1),
        right_collar=F(1),
        continuity=continuity,
        **_embedding_args(),
    )
    assert result.model is not None, result.reason
    assert result.evidence is not None
    assert len(result.model.patches) == 3


def test_public_bridge_refuses_incompatible_selected_boundary() -> None:
    from phydrax.geometry.brep import brep_surface_bridge

    left, right = _curved_source_models()
    # Edge 0 is v0, not the requested v1; its corners are incompatible.
    result = brep_surface_bridge(
        left,
        right,
        left.face_ids[0],
        left.edge_ids[0],
        right.face_ids[0],
        right.edge_ids[0],
        left_side="v1",
        right_side="v0",
        left_along=(F(0), F(1)),
        right_along=(F(0), F(1)),
        left_collar=F(1),
        right_collar=F(1),
        continuity=1,
        **_embedding_args(),
    )
    assert result.model is None
    assert result.reason


def test_public_bridge_refuses_rank_deficient_source() -> None:
    from phydrax.geometry.brep import brep_surface_bridge

    # x(u) = (u - 1/2)^3, y(v) = v - 1, z = 0. At u=1/2 the
    # exact u derivative vanishes while the v derivative remains nonzero.
    x = (F(-1, 8), F(1, 8), F(-1, 8), F(1, 8))
    points = tuple(
        tuple((x[i], F(-1) + F(j, 2), F(0)) for j in range(3)) for i in range(4)
    )
    weights = tuple((F(1),) * 3 for _ in range(4))
    u = (F(0),) * 4 + (F(1),) * 4
    v = (F(0),) * 3 + (F(1),) * 3
    derivative_controls = tuple(3 * (b - a) for a, b in zip(x, x[1:]))
    assert (
        derivative_controls[0] + 2 * derivative_controls[1] + derivative_controls[2]
    ) / 4 == 0
    patch = BSplineSurfacePatch(
        np.asarray(points, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
        np.asarray(u, dtype=np.float64),
        np.asarray(v, dtype=np.float64),
        3,
        2,
        exact_control_points=points,
        exact_weights=weights,
        exact_u_knots=u,
        exact_v_knots=v,
    )
    left = _source_model(patch, "singular-bridge-left")
    _, right = _curved_source_models()
    result = brep_surface_bridge(
        left,
        right,
        left.face_ids[0],
        left.edge_ids[2],
        right.face_ids[0],
        right.edge_ids[0],
        left_side="v1",
        right_side="v0",
        left_along=(F(0), F(1)),
        right_along=(F(0), F(1)),
        left_collar=F(1),
        right_collar=F(1),
        continuity=1,
        **_embedding_args(),
    )
    assert result.model is None
    assert result.reason


def test_public_bridge_refuses_opposing_corner_correspondence() -> None:
    from phydrax.geometry.brep import brep_surface_bridge

    left, right = _curved_source_models()
    # Reversing only the right map pairs opposite u corners and forces the
    # along tangent to reverse across the bridge. An injective oriented chart
    # cannot have the demanded nonfolding projection on both seams.
    result = brep_surface_bridge(
        left,
        right,
        left.face_ids[0],
        left.edge_ids[2],
        right.face_ids[0],
        right.edge_ids[0],
        left_side="v1",
        right_side="v0",
        left_along=(F(0), F(1)),
        right_along=(F(1), F(0)),
        left_collar=F(1),
        right_collar=F(1),
        continuity=1,
        **_embedding_args(),
    )
    assert result.model is None
    assert result.reason
