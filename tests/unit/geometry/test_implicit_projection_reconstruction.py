#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from fractions import Fraction
from typing import Iterator

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from phydrax.discretization._coordinate_enclosure import ExactScalar
from phydrax.discretization._exact_rational_geometry import (
    affine_image_geometry_from_routes,
)
from phydrax.geometry._embedding_state import MeshCertificateLimits
from phydrax.geometry.implicit import _projection_cover as cover
from phydrax.geometry.implicit._analytic_profile import AnalyticImplicitProfile
from phydrax.units import METER


def test_projection_reconstruction_preserves_retained_inputs_and_binding() -> None:
    geometry = phx.geometry.Sphere((0.0, 0.0, 0.0), 1.0, feature_id="sphere").compile(
        coordinate_unit=METER
    )
    plan = phx.geometry.ImplicitPointProjectionPlan(
        geometry,
        jnp.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        jnp.asarray([0.1, 0.2]),
        source_id="sphere-anchors",
        plan_id="retained-projection",
    )

    restored = phx.geometry.ImplicitPointProjectionPlan.from_stored_fields(
        plan, geometry=geometry
    )

    assert restored is not plan
    assert restored.kernel is geometry.kernel
    assert restored.schema is geometry.schema
    assert restored.tolerance is geometry.tolerance
    assert restored.policy is plan.policy
    assert restored.source_id == plan.source_id
    assert restored.plan_id == plan.plan_id
    assert jnp.array_equal(restored.anchors, plan.anchors)
    assert jnp.array_equal(restored.trust_radii, plan.trust_radii)
    assert eqx.tree_equal(restored.realize(geometry.state), plan.realize(geometry.state))


def test_projection_reconstruction_revalidates_stored_trust_radii() -> None:
    geometry = phx.geometry.Sphere((0.0, 0.0, 0.0), 1.0, feature_id="sphere").compile(
        coordinate_unit=METER
    )
    plan = phx.geometry.ImplicitPointProjectionPlan(
        geometry,
        jnp.asarray([[1.0, 0.0, 0.0]]),
        0.1,
        source_id="sphere-anchor",
    )
    invalid = eqx.tree_at(lambda value: value.trust_radii, plan, jnp.asarray([-0.1]))

    with pytest.raises(ValueError, match="trust_radii must be positive"):
        phx.geometry.ImplicitPointProjectionPlan.from_stored_fields(
            invalid, geometry=geometry
        )


def test_projection_reconstruction_validates_anchors_against_supplied_geometry() -> None:
    geometry = phx.geometry.Sphere((0.0, 0.0, 0.0), 1.0, feature_id="sphere").compile(
        coordinate_unit=METER
    )
    plan = phx.geometry.ImplicitPointProjectionPlan(
        geometry,
        jnp.asarray([[1.0, 0.0, 0.0]]),
        0.1,
        source_id="sphere-anchor",
    )
    other_geometry = phx.geometry.Sphere(
        (0.0, 0.0, 0.0), 2.0, feature_id="sphere"
    ).compile(coordinate_unit=METER)

    with pytest.raises(ValueError, match="discovery zero set"):
        phx.geometry.ImplicitPointProjectionPlan.from_stored_fields(
            plan, geometry=other_geometry
        )


def _memo_profile(outer: float = 3.0) -> AnalyticImplicitProfile:
    return AnalyticImplicitProfile(
        phx.geometry.Torus((0.0, 0.0, 0.0), 1.0, outer).compile(
            coordinate_unit=METER
        ),
        phx.SpatialCoordinateContract.si(),
    )


def _memo_boundary() -> cover._Boundary:
    angles = np.asarray(((0.0, 0.0), (0.08, 0.0), (0.0, 0.08)))
    u, v = angles.T
    points = np.stack(
        ((2.0 + np.cos(v)) * np.cos(u),
         (2.0 + np.cos(v)) * np.sin(u), np.sin(v)),
        axis=1,
    )
    triangles = np.stack((points, points[::-1]))
    vertices = np.arange(6, dtype=np.int64).reshape(2, 3)
    return cover._Boundary(
        triangles, vertices, vertices, np.arange(2), np.zeros(2, dtype=np.int64), ()
    )


@contextmanager
def _memo_budget(
    work: int = 1_000_000, memory: int = 64 << 20
) -> Iterator[CoordinateEnclosureBudget]:
    with HostBatchTicket(work, bytes_upper=memory, queries_upper=100_000) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket, maximum_work_units=work, maximum_memory_bytes=memory
        )
        with budget.activate():
            yield budget


def test_torus_corner_memo_charges_only_actual_corner_work_and_full_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile, boundary = _memo_profile(), _memo_boundary()
    original = cover._torus_corner_bounds
    calls = []

    def counted(
        weights: np.ndarray, facet: tuple[cover._RationalPoint, ...],
        center: cover._RationalPoint, major: Fraction, minor: Fraction,
    ) -> cover._TorusCorner:
        result = original(weights, facet, center, major, minor)
        calls.append(result)
        return result

    monkeypatch.setattr(cover, "_torus_corner_bounds", counted)
    with _memo_budget() as budget:
        with budget.live_storage() as storage:
            storage.set_bound(4096)
            memo = cover._TorusCornerMemo(boundary, profile, 3, budget, storage)
            first = tuple(memo.corner(0, weights) for weights in np.eye(3))
            assert budget.work_units == 120 + 3
            assert len(calls) == 3
            again = tuple(memo.corner(0, weights) for weights in np.eye(3))
            assert all(a is b for a, b in zip(first, again, strict=True))
            assert budget.work_units == 120 + 3 + 3
            assert len(calls) == 3
            low, high = cover._normal_bounds(boundary.triangles, None)
            before = budget.work_units
            cover._torus_piece_bounds(
                boundary, profile, np.asarray([0]), np.eye(3)[None],
                np.asarray([[-np.inf, np.inf]]), np.asarray([[-np.inf, np.inf]]),
                low, high, 0.1, budget=budget, corner_memo=memo,
            )
            assert budget.work_units - before == 3 + 120
            assert len(calls) == 3
            assert storage.bound > 4096
            memo.corners.clear()
        assert budget.temporary_bytes_upper == 0


@pytest.mark.parametrize("exact_source", (False, True))
def test_torus_corner_memo_preserves_bitwise_whole_facet_bounds(
    exact_source: bool,
) -> None:
    boundary, profile = _memo_boundary(), _memo_profile()
    if exact_source:
        exact = tuple(
            tuple(tuple(Fraction(float(v)) for v in p) for p in triangle)
            for triangle in boundary.triangles
        )
        shifted = list(exact)
        facet = list(shifted[0])
        point = list(facet[0])
        point[0] += Fraction(1, 10**25)
        facet[0] = tuple(point)
        shifted[0] = tuple(facet)
        boundary = replace(boundary, exact_triangles=tuple(shifted))
    limits = MeshCertificateLimits(maximum_subdivision_depth=3)
    cached = cover._pieces(boundary, profile, 0.0001, limits)
    uncached = cover._pieces_impl(
        boundary, profile, 0.0001, limits, budget=None,
        retire_violated_facets=False, corner_memo=None,
    )
    assert cached.queries > boundary.triangles.shape[0]
    for name in (
        "triangles", "parameters", "coordinates", "owners", "fields",
        "gradients", "normals",
    ):
        first, second = getattr(cached, name), getattr(uncached, name)
        assert first.dtype == second.dtype and first.shape == second.shape
        assert first.tobytes() == second.tobytes()
    assert (cached.upper, cached.lower, cached.queries, cached.findings) == (
        uncached.upper, uncached.lower, uncached.queries, uncached.findings
    )
    # The reverse facet remains reverse; reuse is not an orientation proof.
    assert np.all(cached.normals[cached.owners == 1, 1] < 0)


def test_torus_corner_memo_shares_unit_points_but_keeps_nonunit_owner_laws() -> None:
    boundary, profile = _memo_boundary(), _memo_profile()
    with _memo_budget() as budget, budget.live_storage() as storage:
        storage.set_bound(4096)
        memo = cover._TorusCornerMemo(boundary, profile, 3, budget, storage)
        first = memo.corner(0, np.asarray([1.0, 0.0, 0.0]))
        changed = memo.corner(0, np.asarray([0.5, 0.5, 0.0]))
        other_owner = memo.corner(1, np.asarray([0.0, 0.0, 1.0]))
        assert first[0] != changed[0]
        assert first is other_owner  # Pure point values, not facet identity.
        assert budget.work_units == 2 * 41 + 1
        other_middle = memo.corner(1, np.asarray([0.0, 0.5, 0.5]))
        assert changed == other_middle and changed is not other_middle
        assert budget.work_units == 3 * 41 + 1


def test_torus_corner_memo_reearns_changed_source_and_profile() -> None:
    boundary, profile = _memo_boundary(), _memo_profile()
    first_point = tuple(Fraction(float(v)) for v in boundary.triangles[0, 0])
    exact = tuple(
        tuple(tuple(Fraction(float(v)) for v in p) for p in triangle)
        for triangle in boundary.triangles
    )
    changed_point = (first_point[0] + Fraction(1, 10**25), *first_point[1:])
    changed = replace(
        boundary,
        exact_triangles=((changed_point, *exact[0][1:]), exact[1]),
    )
    assert float(changed_point[0]) == float(first_point[0])
    results = []
    with _memo_budget() as budget:
        for source, current in (
            (boundary, profile), (boundary, profile),
            (changed, profile), (boundary, _memo_profile(3.1)),
        ):
            before = budget.work_units
            with budget.live_storage() as storage:
                storage.set_bound(4096)
                memo = cover._TorusCornerMemo(source, current, 3, budget, storage)
                results.append(memo.corner(0, np.asarray([1.0, 0.0, 0.0])))
                memo.corners.clear()
            assert budget.work_units - before == 41
            assert budget.temporary_bytes_upper == 0
    assert results[0] == results[1] and results[0] is not results[1]
    assert results[0][0] != results[2][0]
    assert results[0][1] != results[3][1]


def test_torus_corner_memo_exact_work_boundary_releases_refused_storage() -> None:
    boundary, profile = _memo_boundary(), _memo_profile()
    for allowance in (126, 125):
        with _memo_budget(allowance) as budget:
            def run() -> None:
                with budget.live_storage() as storage:
                    storage.set_bound(4096)
                    memo = cover._TorusCornerMemo(boundary, profile, 3, budget, storage)
                    try:
                        for _ in range(2):
                            for weights in np.eye(3):
                                memo.corner(0, weights)
                    finally:
                        memo.corners.clear()

            if allowance == 126:
                run()
                assert budget.work_units == 126
            else:
                with pytest.raises(CoordinateEnclosureResourceError) as caught:
                    run()
                assert caught.value.resource == "coefficient_work"
                assert caught.value.requested == 126
                assert caught.value.completed == budget.work_units == 125
            assert budget.work_units <= allowance
            assert budget.temporary_bytes_upper == 0
            assert budget._live_bytes_upper == 0


def test_torus_pieces_call_local_work_and_exact_refusal_release() -> None:
    boundary, profile = _memo_boundary(), _memo_profile()
    limits = MeshCertificateLimits(maximum_subdivision_depth=3)
    with _memo_budget() as budget:
        cover._pieces(boundary, profile, 0.0001, limits, budget=budget)
        required = budget.work_units
        assert budget.temporary_bytes_upper == 0
        cover._pieces(boundary, profile, 0.0001, limits, budget=budget)
        # Quartic proofs re-earn source-dependent arithmetic. Immutable
        # reference-only Bernstein weight/lattice tables may already be owned.
        assert required < budget.work_units <= 2 * required
        peak = budget.peak_bytes_upper
        assert peak > 4096 and budget.temporary_bytes_upper == 0
    with _memo_budget(required) as budget:
        cover._pieces(boundary, profile, 0.0001, limits, budget=budget)
        assert budget.work_units == required
        assert budget.temporary_bytes_upper == 0
    with _memo_budget(required - 1) as budget:
        with pytest.raises(CoordinateEnclosureResourceError) as caught:
            cover._pieces(boundary, profile, 0.0001, limits, budget=budget)
        assert caught.value.resource == "coefficient_work"
        assert caught.value.requested == required
        assert budget.work_units <= required - 1
        assert budget.temporary_bytes_upper == 0
        assert budget._live_bytes_upper == 0
    with _memo_budget(memory=peak - 1) as budget:
        with pytest.raises(CoordinateEnclosureResourceError) as caught:
            cover._pieces(boundary, profile, 0.0001, limits, budget=budget)
        assert caught.value.resource == "polynomial_storage"
        assert budget.temporary_bytes_upper == 0
        assert budget._live_bytes_upper == 0


@pytest.mark.parametrize(
    ("points", "expected"),
    (
        (((0, 0, 0), (2, 0, 0), (1, 2, 0)), Fraction(25, 16)),
        (((0, 0, 0), (2, 0, 0), (0, 2, 0)), Fraction(2)),
        (((0, 0, 0), (4, 0, 0), (1, 1, 0)), Fraction(4)),
        (((0, 0, 0), (1, 0, 0), (2, 0, 0)), Fraction(1)),
        (((1, 0, 0), (0, 1, 0), (0, 0, 1)), Fraction(2, 3)),
        (((0, 0, 0), (0, 0, 0), (0, 0, 0)), Fraction(0)),
    ),
)
def test_torus_exact_triangle_variance_bound(
    points: tuple[tuple[int, int, int], ...], expected: Fraction,
) -> None:
    corners = tuple((Fraction(x), Fraction(y), Fraction(z)) for x, y, z in points)
    variance = cover._triangle_variance_upper(corners)
    assert variance == expected
    for i in range(17):
        for j in range(17 - i):
            weights = (Fraction(i, 16), Fraction(j, 16), Fraction(16 - i - j, 16))
            point = tuple(
                sum((weight * corner[k] for weight, corner in zip(weights, corners)))
                for k in range(3)
            )
            actual = sum(
                weight * sum((a - b) ** 2 for a, b in zip(corner, point))
                for weight, corner in zip(weights, corners)
            )
            assert actual <= variance


@pytest.mark.parametrize("v_offset", (0.0, np.pi / 2, np.pi))
def test_torus_asymmetric_shape_bounds_enclose_whole_facet(
    v_offset: float,
) -> None:
    profile = _memo_profile()
    u = np.asarray((0.0, 0.08, 0.0))
    v = v_offset + np.asarray((0.0, 0.0, 0.08))
    triangle = np.stack(
        ((2 + np.cos(v)) * np.cos(u), (2 + np.cos(v)) * np.sin(u), np.sin(v)),
        axis=1,
    )
    boundary = replace(_memo_boundary(), triangles=triangle[None])
    low, high = cover._normal_bounds(boundary.triangles, None)
    fields, oriented, witnessed = cover._torus_piece_bounds(
        boundary, profile, np.asarray([0]), np.eye(3)[None],
        np.asarray([[-np.inf, np.inf]]), np.asarray([[-np.inf, np.inf]]),
        low, high, 0.1, budget=None,
    )
    assert not witnessed[0]
    assert np.all(np.isfinite(fields)) and np.all(np.isfinite(oriented))
    weights = np.asarray(
        [(i / 32, j / 32, 1 - (i + j) / 32)
         for i in range(1, 32) for j in range(1, 32 - i)]
    )
    points = weights @ triangle
    rho = np.linalg.norm(points[:, :2], axis=1)
    tube = np.sqrt((rho - 2) ** 2 + points[:, 2] ** 2)
    values = tube - 1
    gradients = points.copy()
    gradients[:, :2] *= ((rho - 2) / rho)[:, None]
    gradients /= tube[:, None]
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    slopes = gradients @ normal
    assert np.all(values >= fields[0, 0]) and np.all(values <= fields[0, 1])
    assert np.all(slopes >= oriented[0, 0]) and np.all(slopes <= oriented[0, 1])
    if v_offset == 0:
        facet = tuple(cover._fraction_point(point) for point in triangle)
        corner_values = [
            cover._torus_corner_bounds(
                weights, facet, (Fraction(0), Fraction(0), Fraction(0)),
                Fraction(2), Fraction(1),
            )[1]
            for weights in np.eye(3)
        ]
        bounds = []
        for value in corner_values:
            assert value is not None
            bounds.append(float(value[1]))
        assert fields[0, 1] <= max(bounds)


@pytest.mark.parametrize(
    ("coefficients", "expected"),
    (
        ((1, 1, 1), Fraction(1, 3)),
        ((1, 1, 2), Fraction(1, 2)),
        ((5, -2, -1), Fraction(5, 4)),
        ((-1, -1, -1), Fraction(0)),
        ((0, 1, 1), Fraction(1, 4)),
    ),
)
def test_torus_signed_barycentric_error_maximum(
    coefficients: tuple[int, int, int], expected: Fraction,
) -> None:
    a, b, c = (Fraction(value) for value in coefficients)
    bound = cover._barycentric_pair_upper(a, b, c)
    assert bound == expected
    for i in range(25):
        for j in range(25 - i):
            x, y, z = Fraction(i, 24), Fraction(j, 24), Fraction(24 - i - j, 24)
            assert a * x * y + b * y * z + c * x * z <= bound


def test_torus_anisotropic_tube_floor_keeps_toroidal_curvature() -> None:
    angles = np.asarray((0.0, 0.1, 0.2))
    points = np.stack((3 * np.cos(angles), 3 * np.sin(angles), np.zeros(3)), axis=1)
    corners = tuple(cover._fraction_point(point) for point in points)
    entries = [
        cover._torus_corner_bounds(
            weights, corners, (Fraction(0), Fraction(0), Fraction(0)),
            Fraction(2), Fraction(1),
        )
        for weights in np.eye(3)
    ]
    tubes, axes = [entry[2] for entry in entries], [entry[3] for entry in entries]
    floor = cover._torus_tube_floor(corners, tubes, axes, Fraction(2))
    variance = cover._triangle_variance_upper(corners)
    # Even a zero-reach isotropic 1/t bound is substantially less sharp.
    isotropic = min(tube[0] for tube in tubes) - variance / 2
    assert floor > isotropic + Fraction(1, 50)
    for i in range(33):
        for j in range(33 - i):
            weights = np.asarray((i / 32, j / 32, 1 - (i + j) / 32))
            point = weights @ points
            actual = abs(np.linalg.norm(point[:2]) - 2)
            assert actual >= float(floor) - 1e-14


@pytest.mark.parametrize(
    ("points", "ray", "inside"),
    (
        (((1, 0, 0), (1, 1, 0), (2, 1, 0)), (2, 1), True),
        (((1, 0, 0), (1, 1, 0), (2, 1, 0)), (1, -1), False),
        (((1, 0, 0), (1, 1, 0), (2, 1, 0)), (-1, -1), False),
        (((-2, 1, 0), (-2, -1, 0), (-3, 0, 0)), (-1, 0), True),
        (((-2, 1, 0), (-2, -1, 0), (-3, 0, 0)), (1, 0), False),
        (((1, 0, 0), (2, 0, 0), (3, 0, 0)), (1, 0), True),
        (((1, 0, 0), (2, 0, 0), (3, 0, 0)), (-1, 0), False),
    ),
)
def test_torus_exact_planar_direction_cone(
    points: tuple[tuple[int, int, int], ...], ray: tuple[int, int], inside: bool,
) -> None:
    corners = tuple((Fraction(x), Fraction(y), Fraction(z)) for x, y, z in points)
    direction = (Fraction(ray[0]), Fraction(ray[1]))
    assert cover._corner_cone_contains(corners, direction) is inside


@pytest.mark.parametrize("u_offset", (0.0, np.pi))
@pytest.mark.parametrize("v_offset", (0.0, np.pi / 2, np.pi))
def test_torus_exact_normal_interval_retains_full_normal_width(
    u_offset: float, v_offset: float,
) -> None:
    u = u_offset + np.asarray((0.0, 0.08, 0.0))
    v = v_offset + np.asarray((0.0, 0.0, 0.08))
    points = np.stack(
        ((2 + np.cos(v)) * np.cos(u), (2 + np.cos(v)) * np.sin(u), np.sin(v)),
        axis=1,
    )
    corners = tuple(cover._fraction_point(point) for point in points)
    entries = [
        cover._torus_corner_bounds(
            weights, corners, (Fraction(0), Fraction(0), Fraction(0)),
            Fraction(2), Fraction(1),
        )
        for weights in np.eye(3)
    ]
    axes = [entry[3] for entry in entries]
    low, high = np.asarray((-0.5, -0.3, -1.2)), np.asarray((0.8, 0.4, 2.0))
    interval = cover._torus_normal_interval(
        corners, axes, Fraction(2), Fraction(4, 5), Fraction(6, 5), low, high,
    )
    for i in range(9):
        for j in range(9 - i):
            point = np.asarray((i / 8, j / 8, 1 - (i + j) / 8)) @ points
            rho = np.linalg.norm(point[:2])
            tube = np.sqrt((rho - 2) ** 2 + point[2] ** 2)
            gradient = point.copy()
            gradient[:2] *= (rho - 2) / rho
            gradient /= tube
            for mask in range(8):
                normal = np.where(np.asarray([bool(mask & (1 << k)) for k in range(3)]), low, high)
                slope = normal @ gradient
                assert float(interval[0]) <= slope <= float(interval[1])


@pytest.mark.parametrize(
    ("values", "edges", "expected"),
    (
        ((0, 0, 0), (1, 1, 1), Fraction(-1, 3)),
        ((0, 0, 0), (1, 1, 2), Fraction(-1, 2)),
        ((0, 0, 0), (-1, -1, -1), Fraction(0)),
        ((0, 1, 1), (1, 1, 1), Fraction(0)),
        ((0, 1, 1), (2, 2, 2), Fraction(-1, 6)),
        ((1, 2, 2), (2, 2, 2), Fraction(5, 6)),
        ((0, 0, 1), (1, -1, -1), Fraction(-1, 4)),
        ((0, 0, 0), (1, 1, 4), Fraction(-1)),
    ),
)
def test_torus_exact_correlated_barycentric_quadratic_floor(
    values: Sequence[int | Fraction], edges: Sequence[int | Fraction], expected: Fraction,
) -> None:
    values = (Fraction(values[0]), Fraction(values[1]), Fraction(values[2]))
    edges = [Fraction(edge) for edge in edges]
    floor = cover._barycentric_quadratic_floor(values, edges)
    assert floor == expected
    assert floor >= min(values) - cover._barycentric_pair_upper(*edges)
    for i in range(25):
        for j in range(25 - i):
            x, y, z = Fraction(i, 24), Fraction(j, 24), Fraction(24 - i - j, 24)
            actual = (
                x * values[0] + y * values[1] + z * values[2]
                - edges[0] * x * y - edges[1] * y * z - edges[2] * x * z
            )
            assert actual >= floor


def _shared_source_case(
    budget: CoordinateEnclosureBudget, *, seam: bool = False,
) -> tuple[
    phx.discretization.CellMesh, phx.discretization.CellGeometrySpec,
    cover._Boundary, tuple[tuple[Fraction, ...], ...],
]:
    local = (
        (Fraction(1, 3), Fraction(0), Fraction(0)),
        (Fraction(4, 3), Fraction(0), Fraction(0)),
        (Fraction(1, 3), Fraction(1), Fraction(0)),
        (Fraction(4, 3), Fraction(1), Fraction(0)),
    )
    vertices = np.asarray(((0, 1, 3), (0, 3, 2)), dtype=np.int32)
    points = np.asarray([[float(value) for value in point] for point in local])
    mesh = phx.discretization.CellMesh(
        points, (phx.discretization.CellBlock("surface", "triangle", vertices),)
    )
    route = vertices.copy()
    if seam:
        local = (*local, (local[0][0] + Fraction(1, 10**25), *local[0][1:]))
        route[1, 0] = 4
    element = phx.discretization.CellGeometrySpec.affine(mesh).elements[0]
    geometry = affine_image_geometry_from_routes(
        np.eye(4), local, {"surface": element}, {"surface": route}, budget=budget
    )
    boundary = cover._Boundary(
        points[vertices], vertices,
        np.asarray(mesh.vertex_global_ids)[vertices],
        np.asarray(mesh.blocks[0].global_ids), np.zeros(2, dtype=np.int64), (),
    )
    return mesh, geometry, boundary, local


def test_affine_source_projection_converts_each_exact_shared_image_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = cover.rational_coefficient
    calls = []

    def counted(value: ExactScalar, *, role: str) -> Fraction:
        if role == "projection_cover.source_corner":
            calls.append(value)
        return original(value, role=role)

    monkeypatch.setattr(cover, "rational_coefficient", counted)
    with _memo_budget() as budget:
        mesh, geometry, boundary, local = _shared_source_case(budget)
        with budget.temporary_scope():
            result = cover._source_affine_boundary(
                mesh, geometry, boundary, budget=budget
            )
        assert not result.findings
        assert len(calls) == 3 * 4  # Six cell corners, but four exact image laws.
        assert result.exact_triangles == tuple(
            tuple(local[int(vertex)] for vertex in face)
            for face in boundary.local_vertices
        )
        assert result.exact_triangles[0][0][0] == Fraction(1, 3)
        assert result.exact_triangles[0][0][0] != Fraction(float(mesh.coordinates[0, 0]))
        with budget.temporary_scope():
            again = cover._source_affine_boundary(
                mesh, geometry, boundary, budget=budget
            )
        assert len(calls) == 2 * 3 * 4
        assert again.exact_triangles == result.exact_triangles
        assert again.coordinate_bounds is not None
        assert result.coordinate_bounds is not None
        assert again.coordinate_bounds.tobytes() == result.coordinate_bounds.tobytes()


def test_affine_source_projection_compares_distinct_exact_routes_not_vertex_ids() -> None:
    with _memo_budget() as budget:
        mesh, geometry, boundary, _ = _shared_source_case(budget)
        with budget.temporary_scope():
            good = cover._source_affine_boundary(mesh, geometry, boundary, budget=budget)
        assert not good.findings
        changed_mesh, changed, changed_boundary, local = _shared_source_case(
            budget, seam=True
        )
        assert np.asarray(changed_mesh.coordinates).tobytes() == np.asarray(mesh.coordinates).tobytes()
        assert float(local[0][0]) == float(local[4][0]) and local[0] != local[4]
        with budget.temporary_scope():
            result = cover._source_affine_boundary(
                changed_mesh, changed, changed_boundary, budget=budget
            )
        assert any(
            finding.check == "projection_source_vertex_continuity"
            for finding in result.findings
        )


def test_affine_source_projection_retains_bitwise_carrier_rounding_check() -> None:
    with _memo_budget() as budget:
        mesh, geometry, boundary, _ = _shared_source_case(budget)
        changed = np.asarray(mesh.coordinates).copy()
        changed[0, 0] = np.nextafter(changed[0, 0], np.inf)
        carrier = mesh.with_coordinates(changed, numeric_version="changed-carrier")
        with budget.temporary_scope(), pytest.raises(
            ValueError, match="carrier|coordinates|source image"
        ):
            cover._source_affine_boundary(carrier, geometry, boundary, budget=budget)


def test_affine_source_projection_exact_work_refusal_releases_scratch() -> None:
    def run(budget: CoordinateEnclosureBudget) -> cover._Boundary:
        with budget.temporary_scope():
            mesh, geometry, boundary, _ = _shared_source_case(budget)
            return cover._source_affine_boundary(mesh, geometry, boundary, budget=budget)

    with _memo_budget() as budget:
        assert not run(budget).findings
        required = budget.work_units
    with _memo_budget(required) as budget:
        assert not run(budget).findings
        assert budget.work_units == required
    with _memo_budget(required - 1) as budget:
        with pytest.raises(CoordinateEnclosureResourceError) as caught:
            run(budget)
        assert caught.value.resource == "coefficient_work"
        assert caught.value.requested == required
        assert budget.work_units <= required - 1
        assert budget.temporary_bytes_upper == 0


@pytest.mark.parametrize("exact_source", (False, True))
def test_torus_shared_unit_points_skip_only_actual_corner_work(
    exact_source: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    boundary, profile = _memo_boundary(), _memo_profile()
    if exact_source:
        exact = tuple(
            tuple(tuple(Fraction(float(value)) for value in point) for point in triangle)
            for triangle in boundary.triangles
        )
        boundary = replace(boundary, exact_triangles=exact)
    calls = []
    original = cover._torus_corner_bounds

    def counted(
        weights: np.ndarray, facet: tuple[cover._RationalPoint, ...],
        center: cover._RationalPoint, major: Fraction, minor: Fraction,
    ) -> cover._TorusCorner:
        result = original(weights, facet, center, major, minor)
        calls.append(result)
        return result

    monkeypatch.setattr(cover, "_torus_corner_bounds", counted)
    low, high = cover._normal_bounds(boundary.triangles, None)
    with _memo_budget() as budget:
        with budget.live_storage() as storage:
            storage.set_bound(4096)
            memo = cover._TorusCornerMemo(boundary, profile, 3, budget, storage)
            before = budget.work_units
            first = cover._torus_piece_bounds(
                boundary, profile, np.asarray((0, 1)), np.tile(np.eye(3), (2, 1, 1)),
                np.tile((-np.inf, np.inf), (2, 1)), np.tile((-np.inf, np.inf), (2, 1)),
                low, high, 0.1, budget=budget, corner_memo=memo,
            )
            assert len(calls) == 3
            assert budget.work_units - before == 3 * 40 + 6 + 2 * 120
            before = budget.work_units
            again = cover._torus_piece_bounds(
                boundary, profile, np.asarray((0, 1)), np.tile(np.eye(3), (2, 1, 1)),
                np.tile((-np.inf, np.inf), (2, 1)), np.tile((-np.inf, np.inf), (2, 1)),
                low, high, 0.1, budget=budget, corner_memo=memo,
            )
            assert len(calls) == 3
            assert budget.work_units - before == 6 + 2 * 120
            assert first[1][0, 0] > 0 and first[1][1, 1] < 0
            for cold, warm in zip(first, again, strict=True):
                assert cold.tobytes() == warm.tobytes()
            memo.corners.clear()
        assert budget.temporary_bytes_upper == budget._live_bytes_upper == 0


def test_torus_shared_unit_point_keys_keep_exact_nonbinary_laws_distinct() -> None:
    boundary, profile = _memo_boundary(), _memo_profile()
    exact = [
        [tuple(Fraction(float(value)) for value in point) for point in triangle]
        for triangle in boundary.triangles
    ]
    point = exact[1][2]
    exact[1][2] = (point[0] + Fraction(1, 10**25), *point[1:])
    assert tuple(float(value) for value in exact[1][2]) == tuple(float(value) for value in point)
    boundary = replace(boundary, exact_triangles=tuple(tuple(row) for row in exact))
    with _memo_budget() as budget, budget.live_storage() as storage:
        storage.set_bound(4096)
        memo = cover._TorusCornerMemo(boundary, profile, 3, budget, storage)
        first = memo.corner(0, np.asarray((1.0, 0.0, 0.0)))
        changed = memo.corner(1, np.asarray((0.0, 0.0, 1.0)))
        assert first is not changed and first[0] != changed[0]
        assert budget.work_units == 2 * 41
        assert len(memo.corners) == 2


def test_torus_binary_unit_point_keys_preserve_signed_zero_bytes() -> None:
    boundary, profile = _memo_boundary(), _memo_profile()
    triangles = boundary.triangles.copy()
    triangles[1, 2, 1] = -0.0
    boundary = replace(boundary, triangles=triangles)
    with _memo_budget() as budget, budget.live_storage() as storage:
        storage.set_bound(4096)
        memo = cover._TorusCornerMemo(boundary, profile, 3, budget, storage)
        first = memo.corner(0, np.asarray((1.0, 0.0, 0.0)))
        signed = memo.corner(1, np.asarray((0.0, 0.0, 1.0)))
        assert first == signed and first is not signed
        assert budget.work_units == 2 * 41


def test_torus_shared_invalid_unit_point_remains_an_invalid_corner() -> None:
    boundary, profile = _memo_boundary(), _memo_profile()
    triangles = boundary.triangles.copy()
    triangles[0, 0] = triangles[1, 2] = 0.0
    boundary = replace(boundary, triangles=triangles)
    with _memo_budget() as budget, budget.live_storage() as storage:
        storage.set_bound(4096)
        memo = cover._TorusCornerMemo(boundary, profile, 3, budget, storage)
        first = memo.corner(0, np.asarray((1.0, 0.0, 0.0)))
        again = memo.corner(1, np.asarray((0.0, 0.0, 1.0)))
        assert first is again and first[1] is None
        assert budget.work_units == 40 + 2


def test_torus_shared_unit_point_exact_work_boundary_releases_storage() -> None:
    boundary, profile = _memo_boundary(), _memo_profile()
    low, high = cover._normal_bounds(boundary.triangles, None)
    for allowance in (612, 611):
        with _memo_budget(allowance) as budget:
            def run() -> None:
                with budget.live_storage() as storage:
                    storage.set_bound(4096)
                    memo = cover._TorusCornerMemo(boundary, profile, 3, budget, storage)
                    try:
                        for _ in range(2):
                            cover._torus_piece_bounds(
                                boundary, profile, np.asarray((0, 1)),
                                np.tile(np.eye(3), (2, 1, 1)),
                                np.tile((-np.inf, np.inf), (2, 1)),
                                np.tile((-np.inf, np.inf), (2, 1)),
                                low, high, 0.1, budget=budget, corner_memo=memo,
                            )
                    finally:
                        memo.corners.clear()

            if allowance == 612:
                run()
                assert budget.work_units == 612
            else:
                with pytest.raises(CoordinateEnclosureResourceError) as caught:
                    run()
                assert caught.value.requested == 612
                assert budget.work_units == caught.value.completed == 372
            assert budget.work_units <= allowance
            assert budget.temporary_bytes_upper == budget._live_bytes_upper == 0


@pytest.mark.parametrize("side", (-1, 1))
@pytest.mark.parametrize("rotate", (False, True))
def test_torus_quartic_band_certifies_exact_inner_outer_boundary_triangles(
    side: int, rotate: bool,
) -> None:
    major, minor, band = Fraction(2), Fraction(1), Fraction(1, 8)
    low, high = minor - band, minor + band
    points = (
        (major + side * low, Fraction(0), Fraction(0)),
        (major + side * high, Fraction(0), Fraction(0)),
        (major + side * minor, Fraction(0), Fraction(1, 16)),
    )
    if rotate:
        points = tuple((-y, -x, z) for x, y, z in points)
    with _memo_budget() as budget:
        assert cover._torus_quartic_band(
            points, major, minor, band, budget=budget
        ) == (-band, band)
        assert budget.work_units > cover._TORUS_SHAPE_WORK
        assert budget.temporary_bytes_upper == budget._live_bytes_upper == 0
    # Independent exact tube distances at the two closed-boundary corners.
    for point, expected in zip(points[:2], (low, high), strict=True):
        rho = abs(point[1] if rotate else point[0])
        assert (rho - major) ** 2 + point[2] ** 2 == expected ** 2


@pytest.mark.parametrize(
    ("rho", "height", "inside"),
    (
        (Fraction(25, 8), Fraction(0), True),   # outer U boundary
        (Fraction(23, 8), Fraction(0), True),   # outer L boundary
        (Fraction(7, 8), Fraction(0), True),    # inner U boundary
        (Fraction(9, 8), Fraction(0), True),    # inner L boundary
        (Fraction(2), Fraction(9, 8), True),    # top U boundary
        (Fraction(2), Fraction(-7, 8), True),   # bottom L boundary
        (Fraction(25, 8) + Fraction(1, 2**70), Fraction(0), False),
        (Fraction(23, 8) - Fraction(1, 2**70), Fraction(0), False),
        (Fraction(7, 8) - Fraction(1, 2**70), Fraction(0), False),
        (Fraction(9, 8) + Fraction(1, 2**70), Fraction(0), False),
        (Fraction(0), Fraction(0), False),      # axis / inner hole
        (Fraction(1, 8), Fraction(0), False),   # genuine inner hole
        (Fraction(2), Fraction(0), False),      # medial core circle
        (Fraction(2), Fraction(1, 2**70), False),
        (Fraction(2) + Fraction(1, 2**70), Fraction(0), False),
    ),
)
def test_torus_quartic_band_exact_signs_and_medial_counterexamples(
    rho: Fraction, height: Fraction, inside: bool,
) -> None:
    major, minor, band = Fraction(2), Fraction(1), Fraction(1, 8)
    # This rational 3-4-5 direction gives rho exactly, independently of the
    # implementation's polynomials or any rounded square-root enclosure.
    point = (3 * rho / 5, -4 * rho / 5, height)
    tube_squared = (rho - major) ** 2 + height ** 2
    assert ((minor - band) ** 2 <= tube_squared <= (minor + band) ** 2) is inside
    with _memo_budget() as budget:
        result = cover._torus_quartic_band(
            (point,) * 3, major, minor, band, budget=budget
        )
        assert (result is not None) is inside


@pytest.mark.parametrize(
    ("major", "minor", "band"),
    (
        (2, 1, 0),
        (2, 1, 1),        # L=0
        (2, 1, 2),        # L<0
        (2, 1, -1),
        (2, 1, Fraction(3, 2)),
        (2, Fraction(3, 2), Fraction(1, 2)),  # U=R with L>0
        (1, 2, Fraction(1, 4)),  # B_a can be negative
    ),
)
def test_torus_quartic_band_unsupported_side_conditions_remain_unknown(
    major: int | Fraction, minor: int | Fraction, band: int | Fraction,
) -> None:
    point = (Fraction(2), Fraction(0), Fraction(1))
    with _memo_budget() as budget:
        assert cover._torus_quartic_band(
            (point,) * 3, Fraction(major), Fraction(minor), Fraction(band),
            budget=budget,
        ) is None
        assert budget.temporary_bytes_upper == budget._live_bytes_upper == 0


def _quartic_top_boundary() -> cover._Boundary:
    # A broad toroidal/very narrow radial facet at z=r. Its exact t>=r
    # theorem is much sharper than the existing isotropic curvature bound.
    points = np.asarray(
        ((2.0, -0.25, 1.0), (2.0 + 1 / 64, 0.0, 1.0), (2.0, 0.25, 1.0))
    )
    return replace(_memo_boundary(), triangles=np.stack((points, points[::-1])))


def test_torus_quartic_band_owns_exact_degree_four_bernstein_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corners = tuple(cover._fraction_point(p) for p in _quartic_top_boundary().triangles[0])
    band = Fraction(1, 4096)
    original = cover.bernstein_coefficients
    calls = []

    def recorded(
        polynomial: Mapping[tuple[int, ...], ExactScalar], domain: str, dimension: int,
        *, budget: CoordinateEnclosureBudget,
    ) -> tuple[ExactScalar, ...]:
        before = budget.work_units
        coefficients = original(polynomial, domain, dimension, budget=budget)
        calls.append((dict(polynomial), coefficients, budget.work_units - before))
        assert domain == "simplex" and dimension == 2
        assert len(coefficients) == 15
        assert len(polynomial) <= 15 and max(map(sum, polynomial)) == 4
        return coefficients

    monkeypatch.setattr(cover, "bernstein_coefficients", recorded)
    with _memo_budget() as budget:
        assert cover._torus_quartic_band(
            corners, Fraction(2), Fraction(1), band, budget=budget
        ) == (-band, band)
        assert len(calls) == 2 and all(call[2] > 15 for call in calls)
        assert budget.work_units > 2 * cover._TORUS_SHAPE_WORK
        assert budget.temporary_bytes_upper == budget._live_bytes_upper == 0
    assert max(calls[0][1]) <= 0 and min(calls[1][1]) >= 0
    # Check the actual earned polynomials against the independent full-field
    # quartic identity; dense samples are supplemental, not their sign proof.
    for i in range(17):
        for j in range(17 - i):
            u, v = Fraction(i, 16), Fraction(j, 16)
            point = tuple(
                u * corners[0][axis] + v * corners[1][axis]
                + (1 - u - v) * corners[2][axis]
                for axis in range(3)
            )
            squared = sum(value * value for value in point)
            rho_squared = point[0] ** 2 + point[1] ** 2
            for (polynomial, _controls, _work), endpoint in zip(
                calls, (1 + band, 1 - band), strict=True
            ):
                actual = sum(value * u**a * v**b for (a, b), value in polynomial.items())
                assert actual == (squared + 4 - endpoint**2) ** 2 - 16 * rho_squared


def test_torus_quartic_band_tightens_only_fields_and_preserves_reversed_normals() -> None:
    boundary, profile = _quartic_top_boundary(), _memo_profile()
    low, high = cover._normal_bounds(boundary.triangles, None)
    args = (
        boundary, profile, np.asarray((0, 1)), np.tile(np.eye(3), (2, 1, 1)),
        np.tile((-np.inf, np.inf), (2, 1)),
        np.tile((-np.inf, np.inf), (2, 1)), low, high, 1 / 4096,
    )
    generic = cover._torus_piece_bounds(*args, budget=None)
    assert np.max(np.abs(generic[0][0])) > 1 / 4096
    with _memo_budget() as budget:
        exact = cover._torus_piece_bounds(*args, budget=budget)
        assert budget.work_units > 2 * (cover._TORUS_CORNER_WORK + cover._TORUS_SHAPE_WORK)
        assert budget.temporary_bytes_upper == budget._live_bytes_upper == 0
    assert np.max(np.abs(exact[0])) <= 1 / 4096
    assert exact[1].tobytes() == generic[1].tobytes()
    assert exact[2].tobytes() == generic[2].tobytes()
    assert exact[1][0, 0] > 0 and exact[1][1, 1] < 0


def test_torus_quartic_band_unknown_keeps_generic_enclosures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boundary, profile = _quartic_top_boundary(), _memo_profile()
    low, high = cover._normal_bounds(boundary.triangles, None)
    args = (
        boundary, profile, np.asarray((0, 1)), np.tile(np.eye(3), (2, 1, 1)),
        np.tile((-np.inf, np.inf), (2, 1)),
        np.tile((-np.inf, np.inf), (2, 1)), low, high, 1 / 4096,
    )
    generic = cover._torus_piece_bounds(*args, budget=None)
    monkeypatch.setattr(cover, "_torus_quartic_band", lambda *args, **kwargs: None)
    with _memo_budget() as budget:
        unknown = cover._torus_piece_bounds(*args, budget=budget)
    assert all(a.tobytes() == b.tobytes() for a, b in zip(generic, unknown, strict=True))


def test_torus_quartic_band_refuses_corner_only_medial_cover() -> None:
    # Every corner has t=r, yet the midpoint of the first edge is on the
    # medial core circle. Ancestor/corner rows cannot stand in for this facet.
    corners = (
        (Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(3), Fraction(0), Fraction(0)),
        (Fraction(2), Fraction(0), Fraction(1)),
    )
    assert all((p[0] - 2)**2 + p[2]**2 == 1 for p in corners)
    assert tuple((a + b) / 2 for a, b in zip(corners[0], corners[1])) == (
        Fraction(2), Fraction(0), Fraction(0),
    )
    with _memo_budget() as budget:
        assert cover._torus_quartic_band(
            corners, Fraction(2), Fraction(1), Fraction(1, 8), budget=budget
        ) is None


def test_torus_quartic_band_reearns_changed_exact_source_and_profile() -> None:
    band = Fraction(1, 8)
    boundary_point = (Fraction(2), Fraction(0), 1 + band)
    changed_point = (*boundary_point[:2], boundary_point[2] + Fraction(1, 2**70))
    assert tuple(map(float, boundary_point)) == tuple(map(float, changed_point))
    with _memo_budget() as budget:
        assert cover._torus_quartic_band(
            (boundary_point,) * 3, Fraction(2), Fraction(1), band, budget=budget
        ) == (-band, band)
        before = budget.work_units
        assert cover._torus_quartic_band(
            (changed_point,) * 3, Fraction(2), Fraction(1), band, budget=budget
        ) is None
        assert budget.work_units > before
        before = budget.work_units
        assert cover._torus_quartic_band(
            (boundary_point,) * 3, Fraction(2), Fraction(1, 2), band, budget=budget
        ) is None
        assert budget.work_units > before
    # Center and R/r are taken from the actual new profile, not old band keys.
    boundary = _quartic_top_boundary()
    low, high = cover._normal_bounds(boundary.triangles, None)
    shifted = AnalyticImplicitProfile(
        phx.geometry.Torus((0.0, 0.0, 0.25), 1.0, 3.0).compile(coordinate_unit=METER),
        phx.SpatialCoordinateContract.si(),
    )
    with _memo_budget() as budget:
        _fields, _normals, witnessed = cover._torus_piece_bounds(
            boundary, shifted, np.asarray((0, 1)), np.tile(np.eye(3), (2, 1, 1)),
            np.tile((-np.inf, np.inf), (2, 1)),
            np.tile((-np.inf, np.inf), (2, 1)), low, high, 1 / 4096,
            budget=budget,
        )
    assert np.all(witnessed)


def test_torus_quartic_band_exact_work_memory_refusal_and_cold_reearning() -> None:
    corners = tuple(cover._fraction_point(p) for p in _quartic_top_boundary().triangles[0])

    def prove(budget: CoordinateEnclosureBudget) -> tuple[Fraction, Fraction] | None:
        return cover._torus_quartic_band(
            corners, Fraction(2), Fraction(1), Fraction(1, 4096), budget=budget
        )

    with _memo_budget() as budget:
        assert prove(budget) is not None
        required, peak = budget.work_units, budget.peak_bytes_upper
        assert budget.temporary_bytes_upper == budget._live_bytes_upper == 0
    with _memo_budget(required) as budget:
        assert prove(budget) is not None
        assert budget.work_units == required
        assert budget.temporary_bytes_upper == budget._live_bytes_upper == 0
    with _memo_budget(required - 1) as budget:
        with pytest.raises(CoordinateEnclosureResourceError) as caught:
            prove(budget)
        assert caught.value.resource == "coefficient_work"
        assert caught.value.requested == required
        assert budget.temporary_bytes_upper == budget._live_bytes_upper == 0
    with _memo_budget(memory=peak - 1) as budget:
        with pytest.raises(CoordinateEnclosureResourceError) as caught:
            prove(budget)
        assert caught.value.resource == "polynomial_storage"
        assert budget.temporary_bytes_upper == budget._live_bytes_upper == 0


def test_torus_quartic_band_eliminates_descendant_queries_and_nested_root_corners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boundary = _quartic_top_boundary()
    boundary = replace(
        boundary,
        triangles=boundary.triangles[:1],
        local_vertices=boundary.local_vertices[:1],
        global_vertices=boundary.global_vertices[:1],
        identifiers=boundary.identifiers[:1],
        components=boundary.components[:1],
    )
    profile = _memo_profile()
    limits = MeshCertificateLimits(maximum_subdivision_depth=3)
    original = cover._torus_corner_bounds
    calls = []

    def counted(
        weights: np.ndarray, facet: tuple[cover._RationalPoint, ...],
        center: cover._RationalPoint, major: Fraction, minor: Fraction,
    ) -> cover._TorusCorner:
        result = original(weights, facet, center, major, minor)
        calls.append(result)
        return result

    monkeypatch.setattr(cover, "_torus_corner_bounds", counted)
    generic = cover._pieces(boundary, profile, 1 / 4096, limits)
    generic_corner_count = len(calls)
    calls.clear()
    with _memo_budget() as budget:
        exact = cover._pieces(boundary, profile, 1 / 4096, limits, budget=budget)
        assert budget.temporary_bytes_upper == budget._live_bytes_upper == 0
    assert exact.queries == 2 and generic.queries > exact.queries
    assert len(calls) == 3 and generic_corner_count > len(calls)
    assert exact.triangles.shape[0] == 1 and not exact.findings
    assert np.max(np.abs(exact.fields)) <= 1 / 4096
    assert np.all(exact.normals[:, 0] > 0)
