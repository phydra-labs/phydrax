#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import itertools
import math
from typing import Any

import numpy as np
import pytest

from phydrax._meshcore import (
    clip_box_halfspaces,
    IntersectionClass,
    meshcore_available,
    triangle_intersection_classes,
)
from phydrax.geometry.surface import (
    surface_boolean,
    SurfaceArrangementLimits,
    SurfaceBooleanError,
    SurfaceBooleanOperation,
)
from tests._support.surface_boolean_reference import (
    clipped_volume_reference,
    cone_tetrahedra,
    cube_surface,
    icosphere,
    rotation_matrix,
    signed_volume_reference,
    surface_area_reference,
    surface_model,
)


# Empirical binary64 allowances retained from the pre-remediation unit-cube,
# rotated-cube, tangent-contact, and icosphere fixtures. Closed-form volumes,
# rational oriented volume, and independent cone clipping supply the oracles.
_REFERENCE_ATOL_12 = 1e-12
_REFERENCE_ATOL_13 = 1e-13
_REFERENCE_ATOL_14 = 1e-14
_REFERENCE_ATOL_15 = 1e-15


pytestmark = [
    pytest.mark.meshcore,
    pytest.mark.skipif(
        not meshcore_available(),
        reason="native phydrax-meshcore library is not available",
    ),
]

UNION = SurfaceBooleanOperation.UNION
INTERSECTION = SurfaceBooleanOperation.INTERSECTION
DIFFERENCE = SurfaceBooleanOperation.DIFFERENCE


def _assert_watertight_embedding(result: Any, *, manifold: bool = True) -> None:
    """Check half-edge pairing and pairwise non-intersection independently."""

    faces = result.triangles
    directed = np.concatenate((faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]))
    edges, uses = np.unique(directed, axis=0, return_counts=True)
    reversed_uses = dict(
        zip(map(tuple, edges[:, ::-1].tolist()), uses.tolist(), strict=True)
    )
    # Closed: every directed edge is matched by as many reversed uses.
    assert all(
        reversed_uses.get(tuple(edge), 0) == count
        for edge, count in zip(edges.tolist(), uses.tolist(), strict=True)
    )
    undirected, uses = np.unique(np.sort(directed, axis=1), axis=0, return_counts=True)
    assert np.all(uses == 2) == manifold
    first, second = np.triu_indices(faces.shape[0], k=1)
    classes, status = triangle_intersection_classes(
        result.vertices[faces[first]],
        result.vertices[faces[second]],
        first_ids=faces[first],
        second_ids=faces[second],
    )
    assert np.all(status == 0)
    assert np.all(
        np.isin(
            classes,
            (
                IntersectionClass.DISJOINT,
                IntersectionClass.SHARED_VERTEX,
                IntersectionClass.SHARED_EDGE,
            ),
        )
    )
    assert undirected.shape[0] > 0


def _assert_ancestry(
    result: Any, operands: tuple[tuple[np.ndarray, np.ndarray], ...]
) -> None:
    """Every output triangle lies in its source face with the declared orientation."""

    corners = result.vertices[result.triangles]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    for operand, (vertices, faces) in enumerate(operands):
        rows = result.source_operand == operand
        # Operand models keep their input face order and number cells 0..n-1.
        source = vertices[faces[result.source_cell_global_ids[rows]]]
        source_normal = np.cross(source[:, 1] - source[:, 0], source[:, 2] - source[:, 0])
        unit = source_normal / np.linalg.norm(source_normal, axis=1, keepdims=True)
        offsets = np.abs(
            np.sum((corners[rows] - source[:, None, 0]) * unit[:, None], axis=-1)
        )
        assert np.max(offsets, initial=0.0) <= _REFERENCE_ATOL_12
        basis = np.stack((source[:, 1] - source[:, 0], source[:, 2] - source[:, 0]), -1)
        gram = np.swapaxes(basis, 1, 2) @ basis
        moments = (corners[rows] - source[:, None, 0]) @ basis
        local = np.linalg.solve(gram[:, None], moments[..., None])[..., 0]
        barycentric = np.concatenate((1.0 - np.sum(local, -1, keepdims=True), local), -1)
        assert np.min(barycentric, initial=0.0) >= -_REFERENCE_ATOL_12
        agrees = np.sum(normals[rows] * source_normal, axis=1) > 0.0
        np.testing.assert_array_equal(agrees, ~result.reversed[rows])


def _run(
    first: tuple[np.ndarray, np.ndarray],
    second: tuple[np.ndarray, np.ndarray],
    operation: SurfaceBooleanOperation,
) -> Any:
    result = surface_boolean(
        surface_model(*first, "first"), surface_model(*second, "second"), operation
    )
    _assert_ancestry(result, (first, second))
    return result


_OVERLAP = 0.5 * 0.75 * 0.75


@pytest.mark.parametrize(
    ("operation", "volume"),
    [(UNION, 2.0 - _OVERLAP), (INTERSECTION, _OVERLAP), (DIFFERENCE, 1.0 - _OVERLAP)],
    ids=["union", "intersection", "difference"],
)
def test_overlapping_cubes_match_box_volumes(
    operation: SurfaceBooleanOperation, volume: float
) -> None:
    result = _run(cube_surface(), cube_surface((0.5, 0.25, 0.25)), operation)

    assert signed_volume_reference(result.vertices, result.triangles) == pytest.approx(
        volume, abs=_REFERENCE_ATOL_14
    )
    _assert_watertight_embedding(result)
    assert result.closure.closed and result.closure.vertex_manifold
    assert result.closure.component_count == 1
    assert result.surface is not None
    surface_faces = np.concatenate(
        [np.asarray(block.vertices) for block in result.surface.mesh.blocks]
    )
    assert surface_faces.shape[0] == result.triangles.shape[0]


@pytest.mark.parametrize(
    ("operation", "volume", "area"),
    [(UNION, 1.5, 8.0), (INTERSECTION, 0.5, 4.0), (DIFFERENCE, 0.5, 4.0)],
    ids=["union", "intersection", "difference"],
)
def test_coplanar_faces_are_emitted_once(
    operation: SurfaceBooleanOperation, volume: float, area: float
) -> None:
    result = _run(cube_surface(), cube_surface((0.5, 0.0, 0.0)), operation)

    # Box areas fail if a shared coplanar region is duplicated or dropped.
    assert signed_volume_reference(result.vertices, result.triangles) == pytest.approx(
        volume, abs=_REFERENCE_ATOL_14
    )
    assert surface_area_reference(result.vertices, result.triangles) == pytest.approx(
        area, abs=_REFERENCE_ATOL_13
    )
    _assert_watertight_embedding(result)


def test_face_tangent_cubes_merge_and_touch() -> None:
    first, second = cube_surface(), cube_surface((1.0, 0.0, 0.0))

    union = _run(first, second, UNION)
    intersection = _run(first, second, INTERSECTION)
    difference = _run(first, second, DIFFERENCE)

    assert signed_volume_reference(union.vertices, union.triangles) == pytest.approx(2.0)
    assert surface_area_reference(union.vertices, union.triangles) == pytest.approx(10.0)
    _assert_watertight_embedding(union)
    assert union.closure.component_count == 1
    assert intersection.empty and intersection.surface is None
    assert signed_volume_reference(
        difference.vertices, difference.triangles
    ) == pytest.approx(1.0)
    assert set(difference.source_operand.tolist()) == {0}


def test_point_and_edge_tangent_unions_report_nonmanifold_contacts() -> None:
    vertex_touch = _run(cube_surface(), cube_surface((1.0, 1.0, 1.0)), UNION)
    edge_touch = _run(cube_surface(), cube_surface((1.0, 1.0, 0.0)), UNION)

    assert signed_volume_reference(
        vertex_touch.vertices, vertex_touch.triangles
    ) == pytest.approx(2.0)
    assert vertex_touch.closure.closed and vertex_touch.closure.edge_manifold
    assert vertex_touch.closure.nonmanifold_vertices == 1
    assert signed_volume_reference(
        edge_touch.vertices, edge_touch.triangles
    ) == pytest.approx(2.0)
    assert edge_touch.closure.closed
    assert edge_touch.closure.nonmanifold_edges == 1
    assert edge_touch.surface is None and not edge_touch.empty
    _assert_watertight_embedding(edge_touch, manifold=False)


def test_rotated_square_prism_matches_regular_octagon() -> None:
    rotated = cube_surface(rotation=rotation_matrix((0.0, 0.0, 1.0), math.pi / 4.0))
    octagon = 8.0 * 0.25 * math.tan(math.pi / 8.0)

    volumes = {
        operation: signed_volume_reference(result.vertices, result.triangles)
        for operation in (UNION, INTERSECTION, DIFFERENCE)
        for result in (_run(cube_surface(), rotated, operation),)
    }

    assert volumes[INTERSECTION] == pytest.approx(octagon, abs=_REFERENCE_ATOL_14)
    assert volumes[UNION] == pytest.approx(2.0 - octagon, abs=_REFERENCE_ATOL_14)
    assert volumes[DIFFERENCE] == pytest.approx(1.0 - octagon, abs=_REFERENCE_ATOL_14)


@pytest.mark.parametrize(
    "operation",
    [UNION, INTERSECTION, DIFFERENCE],
    ids=["union", "intersection", "difference"],
)
def test_obliquely_rotated_cubes_match_tetrahedral_clipping(
    operation: SurfaceBooleanOperation,
) -> None:
    first = cube_surface()
    second = cube_surface((0.3, 0.1, 0.2), rotation=rotation_matrix((1.0, 2.0, 0.5), 0.7))
    common = clipped_volume_reference(cone_tetrahedra(*first), cone_tetrahedra(*second))
    expected = {UNION: 2.0 - common, INTERSECTION: common, DIFFERENCE: 1.0 - common}

    result = _run(first, second, operation)

    assert signed_volume_reference(result.vertices, result.triangles) == pytest.approx(
        expected[operation], abs=_REFERENCE_ATOL_13
    )
    _assert_watertight_embedding(result)


@pytest.mark.parametrize(
    "operation",
    [UNION, INTERSECTION, DIFFERENCE],
    ids=["union", "intersection", "difference"],
)
def test_tessellated_spheres_match_tetrahedral_clipping(
    operation: SurfaceBooleanOperation,
) -> None:
    vertices, faces = icosphere(2)
    rotation = rotation_matrix((0.3, -1.0, 0.8), 0.9)
    first = (vertices, faces)
    second = (vertices @ rotation.T + np.asarray((0.7, 0.35, 0.2)), faces)
    sphere = signed_volume_reference(*first)
    common = clipped_volume_reference(cone_tetrahedra(*first), cone_tetrahedra(*second))
    expected = {
        UNION: 2.0 * sphere - common,
        INTERSECTION: common,
        DIFFERENCE: sphere - common,
    }

    result = _run(first, second, operation)

    assert signed_volume_reference(result.vertices, result.triangles) == pytest.approx(
        expected[operation], abs=_REFERENCE_ATOL_12
    )
    _assert_watertight_embedding(result)
    assert result.arrangement.maximum_construction_bound < _REFERENCE_ATOL_14
    assert set(result.source_operand.tolist()) == {0, 1}


def test_nested_operands_give_inner_solid_and_cavity() -> None:
    outer, inner = cube_surface(), cube_surface((0.25, 0.25, 0.25), size=0.5)

    union = _run(outer, inner, UNION)
    intersection = _run(outer, inner, INTERSECTION)
    cavity = _run(outer, inner, DIFFERENCE)

    assert set(union.source_operand.tolist()) == {0}
    assert signed_volume_reference(union.vertices, union.triangles) == pytest.approx(1.0)
    assert set(intersection.source_operand.tolist()) == {1}
    assert signed_volume_reference(
        intersection.vertices, intersection.triangles
    ) == pytest.approx(0.125)
    assert cavity.closure.component_count == 2
    assert signed_volume_reference(cavity.vertices, cavity.triangles) == pytest.approx(
        0.875
    )
    np.testing.assert_array_equal(cavity.reversed, cavity.source_operand == 1)


def test_disjoint_operands_give_disconnected_union_and_empty_intersection() -> None:
    first, second = cube_surface(), cube_surface((2.0, 0.5, 0.0))

    union = _run(first, second, UNION)
    intersection = _run(first, second, INTERSECTION)

    assert union.closure.component_count == 2
    assert signed_volume_reference(union.vertices, union.triangles) == pytest.approx(2.0)
    assert intersection.empty
    assert intersection.surface is None
    assert intersection.vertices.shape == (0, 3)
    assert intersection.closure.closed


def test_identical_operands_resolve_by_coplanar_orientation() -> None:
    union = _run(cube_surface(), cube_surface(), UNION)
    difference = _run(cube_surface(), cube_surface(), DIFFERENCE)

    assert union.triangles.shape[0] == 12
    assert set(union.source_operand.tolist()) == {0}
    assert difference.empty


def test_source_corner_values_reproduce_affine_fields() -> None:
    first, second = cube_surface(), cube_surface((0.5, 0.25, 0.25))
    first_values = 2.0 * first[0] @ np.asarray((1.0, -1.0, 0.5))
    second_values = second[0] @ np.asarray((0.0, 3.0, 1.0)) - 7.0

    result = _run(first, second, UNION)
    values = result.source_corner_values(first_values, second_values)

    corners = result.vertices[result.triangles]
    expected = np.where(
        result.source_operand[:, None] == 0,
        2.0 * corners @ np.asarray((1.0, -1.0, 0.5)),
        corners @ np.asarray((0.0, 3.0, 1.0)) - 7.0,
    )
    np.testing.assert_allclose(values, expected, atol=_REFERENCE_ATOL_12)
    assert set(result.source_operand.tolist()) == {0, 1}
    with pytest.raises(ValueError, match=r"SurfaceBooleanResult\.operand_values"):
        result.source_corner_values(first_values, np.zeros((8, 2)))


def test_boolean_is_deterministic() -> None:
    vertices, faces = icosphere(1)
    first = surface_model(vertices, faces, "first")
    second = surface_model(vertices + 0.6, faces, "second")

    results = [surface_boolean(first, second, DIFFERENCE) for _ in range(2)]

    assert results[0].result_id == results[1].result_id
    np.testing.assert_array_equal(results[0].triangles, results[1].triangles)
    np.testing.assert_array_equal(results[0].vertices, results[1].vertices)


def test_open_operand_is_refused() -> None:
    vertices, faces = cube_surface()
    sheet = surface_model(vertices, faces[:-1], "sheet")

    with pytest.raises(
        SurfaceBooleanError, match=r"surface_boolean\.operands"
    ) as refused:
        surface_boolean(
            surface_model(*cube_surface((0.5, 0.0, 0.0)), "solid"), sheet, UNION
        )

    assert refused.value.status == "open_operand"
    assert refused.value.operand == 1


def test_inverted_operand_is_refused() -> None:
    vertices, faces = cube_surface()

    with pytest.raises(SurfaceBooleanError) as refused:
        surface_boolean(
            surface_model(vertices, faces[:, ::-1].copy(), "inverted"),
            surface_model(*cube_surface((0.5, 0.0, 0.0)), "solid"),
            UNION,
        )

    assert refused.value.status == "operand_not_solid"
    assert refused.value.operand == 0


def test_hollow_operand_requires_inward_cavity_shell() -> None:
    outer_vertices, outer_faces = cube_surface((-1.0, -1.0, -1.0), size=3.0)
    inner_vertices, inner_faces = cube_surface()
    vertices = np.concatenate((outer_vertices, inner_vertices))
    hollow = surface_model(
        vertices, np.concatenate((outer_faces, inner_faces[:, ::-1] + 8)), "hollow"
    )
    nested = surface_model(
        vertices, np.concatenate((outer_faces, inner_faces + 8)), "nested"
    )
    probe = surface_model(*cube_surface((0.5, 0.25, 0.25)), "probe")

    result = surface_boolean(hollow, probe, INTERSECTION)

    assert signed_volume_reference(result.vertices, result.triangles) == pytest.approx(
        1.0 - _OVERLAP
    )
    with pytest.raises(SurfaceBooleanError) as refused:
        surface_boolean(nested, probe, INTERSECTION)
    assert refused.value.status == "operand_not_solid"


def test_classification_budget_is_refused() -> None:
    with pytest.raises(SurfaceBooleanError) as refused:
        surface_boolean(
            surface_model(*cube_surface(), "first"),
            surface_model(*cube_surface((0.5, 0.25, 0.25)), "second"),
            UNION,
            limits=SurfaceArrangementLimits(maximum_winding_evaluations=1),
        )

    assert refused.value.status == "limit_exceeded"


def _convex_membership(
    points: np.ndarray, vertices: np.ndarray, faces: np.ndarray
) -> np.ndarray:
    """Independent half-space oracle for outward oriented convex operands."""
    corners = vertices[faces]
    normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    return np.all(
        np.sum((points[:, None] - corners[None, :, 0]) * normal[None], axis=-1) < 0.0,
        axis=1,
    )


def _ray_membership(
    points: np.ndarray, vertices: np.ndarray, faces: np.ndarray
) -> np.ndarray:
    """Odd forward intersections, independently of generalized winding numbers."""
    corners = vertices[faces]
    first = corners[:, 1] - corners[:, 0]
    second = corners[:, 2] - corners[:, 0]
    direction = np.asarray((1.0, 0.371391, 0.173219), dtype=np.float64)
    cross = np.cross(direction, second)
    determinant = np.sum(first * cross, axis=1)
    eligible = np.abs(determinant) > _REFERENCE_ATOL_13
    denominator = np.where(eligible, determinant, 1.0)
    offset = points[:, None] - corners[None, :, 0]
    u = np.sum(offset * cross[None], axis=-1) / denominator
    q = np.cross(offset, first[None])
    v = np.sum(q * direction, axis=-1) / denominator
    t = np.sum(q * second[None], axis=-1) / denominator
    hits = eligible[None] & (u > 0.0) & (v > 0.0) & (u + v < 1.0) & (t > 0.0)
    return np.count_nonzero(hits, axis=1) % 2 == 1


@pytest.mark.parametrize(
    "operation",
    [UNION, INTERSECTION, DIFFERENCE],
    ids=["union", "intersection", "difference"],
)
def test_nary_rotated_cube_sphere_regions_preserve_original_ancestry(
    operation: SurfaceBooleanOperation,
) -> None:
    first = cube_surface(rotation=rotation_matrix((1.0, 2.0, 3.0), 0.19))
    second = cube_surface(
        (0.37, 0.17, -0.09), rotation=rotation_matrix((3.0, -1.0, 2.0), -0.23)
    )
    sphere_points, sphere_faces = icosphere(1)
    third = (sphere_points * 0.65 + np.asarray((0.65, 0.59, 0.47)), sphere_faces)
    inputs = (first, second, third)
    models = tuple(surface_model(*data, f"original-{i}") for i, data in enumerate(inputs))
    result = surface_boolean(models[0], models[1], operation, operands=(models[2],))
    probes = np.stack(
        np.meshgrid(
            np.linspace(-0.23, 1.61, 7),
            np.linspace(-0.17, 1.49, 6),
            np.linspace(-0.31, 1.43, 5),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    occupancy = np.stack([_convex_membership(probes, *data) for data in inputs], axis=1)
    match operation:
        case SurfaceBooleanOperation.UNION:
            expected = np.any(occupancy, axis=1)
        case SurfaceBooleanOperation.INTERSECTION:
            expected = np.all(occupancy, axis=1)
        case SurfaceBooleanOperation.DIFFERENCE:
            expected = occupancy[:, 0] & ~np.any(occupancy[:, 1:], axis=1)
    np.testing.assert_array_equal(
        _ray_membership(probes, result.vertices, result.triangles), expected
    )
    _assert_ancestry(result, inputs)
    assert result.closure.closed and result.closure.edge_manifold
    assert result.operand_ids == tuple(model.model_id for model in models)
    assert set(result.source_operand.tolist()) == {0, 1, 2}
    assert result.arrangement.triple_plane_vertices > 0
    assert np.max(result.vertex_bounds) <= result.arrangement.maximum_construction_bound
    coefficients = (
        np.asarray((1.0, 2.0, -0.7)),
        np.asarray((-0.4, 0.2, 1.9)),
        np.asarray((0.7, -0.1, 0.3)),
    )
    values = tuple(
        data[0] @ coefficient
        for data, coefficient in zip(inputs, coefficients, strict=True)
    )
    transferred = result.source_corner_values(
        values[0], values[1], operand_values=(values[2],)
    )
    expected_values = np.sum(
        result.vertices[result.triangles]
        * np.stack(coefficients)[result.source_operand, None],
        axis=-1,
    )
    np.testing.assert_allclose(transferred, expected_values, atol=_REFERENCE_ATOL_12)


def test_nary_coplanar_duplicate_faces_emit_once_and_difference_is_empty() -> None:
    models = tuple(surface_model(*cube_surface(), f"coincident-{i}") for i in range(3))
    union = surface_boolean(models[0], models[1], UNION, operands=(models[2],))
    difference = surface_boolean(models[0], models[1], DIFFERENCE, operands=(models[2],))
    assert signed_volume_reference(union.vertices, union.triangles) == pytest.approx(1.0)
    assert union.closure.closed and union.closure.edge_manifold
    np.testing.assert_array_equal(
        union.source_operand, np.zeros(union.triangles.shape[0], dtype=np.int64)
    )
    assert difference.empty


def test_chained_csg_reuses_original_features_without_accumulating_roundoff() -> None:
    inputs = (
        cube_surface(rotation=rotation_matrix((1.0, 2.0, 3.0), 0.19)),
        cube_surface(
            (0.37, 0.17, -0.09), rotation=rotation_matrix((3.0, -1.0, 2.0), -0.23)
        ),
        cube_surface(
            (0.19, -0.11, 0.33), rotation=rotation_matrix((2.0, 1.0, -1.0), 0.31)
        ),
    )
    models = tuple(
        surface_model(*data, f"chain-source-{i}") for i, data in enumerate(inputs)
    )
    first = surface_boolean(models[0], models[1], UNION)
    chained = surface_boolean(first, models[2], UNION)
    simultaneous = surface_boolean(models[0], models[1], UNION, operands=(models[2],))
    np.testing.assert_array_equal(chained.vertices, simultaneous.vertices)
    np.testing.assert_array_equal(chained.triangles, simultaneous.triangles)
    np.testing.assert_array_equal(chained.vertex_bounds, simultaneous.vertex_bounds)
    assert chained.operand_ids == tuple(model.model_id for model in models)
    _assert_ancestry(chained, inputs)
    assert chained.closure.closed and chained.closure.edge_manifold


def test_chained_difference_preserves_nonassociative_region_expression() -> None:
    inputs = (
        cube_surface(),
        cube_surface((0.5, 0.0, 0.0)),
        cube_surface((0.75, 0.25, 0.25), size=0.5),
    )
    models = tuple(
        surface_model(*data, f"nested-csg-{i}") for i, data in enumerate(inputs)
    )
    united = surface_boolean(models[0], models[1], UNION)
    cavity = surface_boolean(united, models[2], DIFFERENCE)
    assert signed_volume_reference(cavity.vertices, cavity.triangles) == pytest.approx(
        1.5 - 0.125, abs=_REFERENCE_ATOL_13
    )
    assert cavity.closure.closed and cavity.closure.edge_manifold
    assert cavity.closure.component_count == 2
    np.testing.assert_array_equal(cavity.reversed, cavity.source_operand == 2)
    _assert_ancestry(cavity, inputs)


def test_empty_chained_operand_cannot_steal_surviving_source_properties() -> None:
    models = tuple(
        surface_model(*cube_surface(), name)
        for name in ("canceled-a", "canceled-b", "surviving-c")
    )
    empty = surface_boolean(models[0], models[1], DIFFERENCE)
    surviving = surface_boolean(empty, models[2], UNION)
    assert empty.empty
    assert signed_volume_reference(
        surviving.vertices, surviving.triangles
    ) == pytest.approx(1.0)
    np.testing.assert_array_equal(
        surviving.source_operand,
        np.full(surviving.triangles.shape[0], 2, dtype=np.int64),
    )
    transferred = surviving.source_corner_values(
        np.full(8, 2.0, dtype=np.float64),
        np.full(8, 3.0, dtype=np.float64),
        operand_values=(np.full(8, 9.0, dtype=np.float64),),
    )
    np.testing.assert_array_equal(transferred, np.full_like(transferred, 9.0))
    if surviving.surface is None:
        raise RuntimeError(
            "surviving.surface must be present for a nonempty solid; got None"
        )
    assert set(surviving.surface.metadata.cell_tags) == {"surviving-c"}


def _convex_common_volume(operands: tuple[tuple[np.ndarray, np.ndarray], ...]) -> float:
    """Independent reference: one box clipped by every operand face halfspace."""

    normals = []
    offsets = []
    for vertices, faces in operands:
        corners = vertices[faces]
        normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        normals.append(normal)
        offsets.append(np.sum(normal * corners[:, 0], axis=1))
    stacked = np.concatenate(normals)[None]
    lower = np.min([vertices.min(axis=0) for vertices, _ in operands], axis=0) - 1.0
    upper = np.max([vertices.max(axis=0) for vertices, _ in operands], axis=0) + 1.0
    clipped = clip_box_halfspaces(
        lower,
        upper,
        stacked,
        np.concatenate(offsets)[None],
        np.asarray((stacked.shape[1],), dtype=np.int32),
    )
    volume, status = clipped.volume, clipped.status
    assert status[0] == 0
    return float(volume[0])


def _convex_union_volume(operands: tuple[tuple[np.ndarray, np.ndarray], ...]) -> float:
    """Inclusion-exclusion over every nonempty subset of convex operands."""

    return sum(
        (-1.0) ** (size + 1) * _convex_common_volume(subset)
        for size in range(1, len(operands) + 1)
        for subset in itertools.combinations(operands, size)
    )


def _rotated_cubes(count: int) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    return tuple(
        cube_surface(
            (0.1 * index, 0.07 * index, -0.05 * index),
            rotation=rotation_matrix(
                (1.0 + index, 2.0 - index, 0.5 + 0.3 * index), 0.17 + 0.29 * index
            ),
        )
        for index in range(count)
    )


def test_union_of_five_rotated_cubes_matches_inclusion_exclusion() -> None:
    inputs = _rotated_cubes(5)
    models = tuple(
        surface_model(*data, f"rotated-{index}") for index, data in enumerate(inputs)
    )

    result = surface_boolean(models[0], models[1], UNION, operands=models[2:])

    assert signed_volume_reference(result.vertices, result.triangles) == pytest.approx(
        _convex_union_volume(inputs), abs=_REFERENCE_ATOL_13
    )
    _assert_watertight_embedding(result)
    _assert_ancestry(result, inputs)
    assert result.closure.component_count == 1
    assert result.arrangement.triple_plane_vertices > 0
    assert set(result.source_operand.tolist()) == set(range(5))


def test_union_minus_third_with_triple_contacts_matches_inclusion_exclusion() -> None:
    first, second, third = _rotated_cubes(3)
    models = tuple(
        surface_model(*data, f"triple-{index}")
        for index, data in enumerate((first, second, third))
    )
    expected = (
        _convex_union_volume((first, second))
        - _convex_common_volume((first, third))
        - _convex_common_volume((second, third))
        + _convex_common_volume((first, second, third))
    )

    result = surface_boolean(
        surface_boolean(models[0], models[1], UNION), models[2], DIFFERENCE
    )

    assert signed_volume_reference(result.vertices, result.triangles) == pytest.approx(
        expected, abs=_REFERENCE_ATOL_13
    )
    _assert_watertight_embedding(result)
    _assert_ancestry(result, (first, second, third))
    assert result.arrangement.triple_plane_vertices > 0
    # The subtracted operand bounds the result only through reversed faces.
    np.testing.assert_array_equal(result.reversed, result.source_operand == 2)


def test_union_of_six_spheres_matches_inclusion_exclusion_deterministically() -> None:
    vertices, faces = icosphere(1)
    centers = (
        (0.0, 0.0, 0.0),
        (0.7, 0.1, 0.0),
        (0.3, 0.6, 0.1),
        (0.2, 0.2, 0.65),
        (-0.5, 0.4, -0.2),
        (0.4, -0.5, 0.3),
    )
    inputs = tuple((vertices * 0.6 + np.asarray(center), faces) for center in centers)
    models = tuple(
        surface_model(*data, f"sphere-{index}") for index, data in enumerate(inputs)
    )

    runs = [
        surface_boolean(models[0], models[1], UNION, operands=models[2:])
        for _ in range(2)
    ]

    result = runs[0]
    assert signed_volume_reference(result.vertices, result.triangles) == pytest.approx(
        _convex_union_volume(inputs), abs=_REFERENCE_ATOL_13
    )
    _assert_watertight_embedding(result)
    _assert_ancestry(result, inputs)
    assert result.arrangement.triple_plane_vertices > 0
    assert runs[0].result_id == runs[1].result_id
    np.testing.assert_array_equal(runs[0].vertices, runs[1].vertices)
    np.testing.assert_array_equal(runs[0].triangles, runs[1].triangles)


@pytest.mark.parametrize("exponent", [-30, -45, -52], ids=["2^-30", "2^-45", "2^-52"])
def test_nearly_coincident_faces_are_classified_exactly(exponent: int) -> None:
    gap = float(np.ldexp(np.float64(1.0), exponent))
    first, second = cube_surface(), cube_surface((0.5, 0.25, gap))

    result = _run(first, second, INTERSECTION)

    # The common solid is bounded below by the second cube's bottom face, which
    # lies inside the first cube, and above by the first cube's top face.
    corners = result.vertices[result.triangles]
    bottom = np.all(corners[..., 2] == gap, axis=1)
    top = np.all(corners[..., 2] == 1.0, axis=1)
    assert np.any(bottom) and np.any(top)
    np.testing.assert_array_equal(result.source_operand[bottom], 1)
    np.testing.assert_array_equal(result.source_operand[top], 0)
    _assert_watertight_embedding(result)
    assert signed_volume_reference(result.vertices, result.triangles) == pytest.approx(
        0.5 * 0.75 * (1.0 - gap), abs=_REFERENCE_ATOL_15
    )
