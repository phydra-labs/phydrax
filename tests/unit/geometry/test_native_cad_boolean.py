#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import math
from fractions import Fraction
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry.brep._boolean import (
    boolean_brep,
    BRepBooleanFailure,
    BRepBooleanOperation,
    BRepBooleanPolicy,
)
from phydrax.geometry.brep._boolean_coincidence import prove_surface_correspondence
from phydrax.geometry.brep._constructors import (
    brep_box,
    brep_cylinder,
    brep_sphere,
    BRepTessellationPolicy,
)
from phydrax.geometry.brep._containment import certify_solid_containment
from phydrax.geometry.brep._intersection import (
    NativePeriodEndpoint,
    ParametricIntersectionPolicy,
)
from phydrax.geometry.brep._intersection_curve import IntersectionCurve
from phydrax.geometry.brep._model import BRepModel
from phydrax.geometry.brep._patches import CylinderPatch, PlanePatch, SpherePatch
from phydrax.geometry.brep._placed import PlacedSurface
from phydrax.geometry.brep._query import prepare_brep_query
from phydrax.interchange._cad import CadInterchangeError
from phydrax.interchange._cad_archive import load_brep_archive, save_brep_archive
from phydrax.interchange._step import write_step
from phydrax.units import MILLIMETER


_COORDINATES = SpatialCoordinateContract(MILLIMETER)
_EXACT_SOURCE = BRepTessellationPolicy(realize=False)


def _box(
    lower: tuple[float, float, float], upper: tuple[float, float, float]
) -> BRepModel:
    return brep_box(lower, upper, coordinate_contract=_COORDINATES)


def _volume(model: BRepModel) -> float:
    """Independent oriented boundary integral over exact planar face loops."""
    geometry = model.geometry
    if geometry is None:
        raise ValueError("The test requires authoritative exact topology.")
    points = np.asarray(geometry.vertex_points)
    volume = 0.0
    for faces, shell_signs in zip(
        model.topology.solid_faces, model.topology.solid_face_orientations, strict=True
    ):
        for face, shell_sign in zip(faces, shell_signs, strict=True):
            sign = shell_sign * float(model.orientation[face])
            for loop in geometry.face_loops[face]:
                vertices = []
                for coedge in loop:
                    edge = geometry.coedge_edges[coedge]
                    first, last = geometry.edge_vertices[edge]
                    vertices.append(
                        points[first if geometry.coedge_senses[coedge] > 0 else last]
                    )
                for index in range(1, len(vertices) - 1):
                    volume += (
                        sign
                        * float(
                            vertices[0] @ np.cross(vertices[index], vertices[index + 1])
                        )
                        / 6.0
                    )
    return volume


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        pytest.param("union", 15.0, id="union"),
        pytest.param("intersection", 1.0, id="intersection"),
        pytest.param("difference", 7.0, id="difference"),
    ],
)
def test_overlapping_boxes_have_exact_oriented_volume(
    operation: BRepBooleanOperation,
    expected: float,
) -> None:
    first = _box((0.0, 0.0, 0.0), (2.0, 2.0, 2.0))
    second = _box((1.0, 1.0, 1.0), (3.0, 3.0, 3.0))
    original = first.model_id, second.model_id
    result = boolean_brep(first, second, operation)
    assert _volume(result.model) == pytest.approx(expected, abs=1e-12)
    assert (first.model_id, second.model_id) == original
    assert result.association_graph.transaction.coverage.source_exhaustive
    assert result.association_graph.transaction.coverage.target_exhaustive


def test_difference_reconstructs_oriented_cavity_shell() -> None:
    outer = _box((0.0, 0.0, 0.0), (3.0, 3.0, 3.0))
    inner = _box((1.0, 1.0, 1.0), (2.0, 2.0, 2.0))
    result = boolean_brep(outer, inner, "difference")
    geometry = result.model.geometry
    assert geometry is not None
    assert result.model.topology.num_solids == 1
    assert len(geometry.solid_shells[0]) == 2
    assert _volume(result.model) == pytest.approx(26.0, abs=1e-12)


def test_disconnected_union_preserves_distinct_solid_topology() -> None:
    result = boolean_brep(
        _box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        _box((2.0, 0.0, 0.0), (3.0, 1.0, 1.0)),
        "union",
    )
    assert result.model.topology.num_solids == 2
    assert _volume(result.model) == pytest.approx(2.0, abs=1e-12)


@pytest.mark.parametrize("operation", ["intersection", "difference"])
def test_regularized_empty_boolean_has_exhaustive_deletion(
    operation: BRepBooleanOperation,
) -> None:
    first = _box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    second = (
        first if operation == "difference" else _box((1.0, 0.0, 0.0), (2.0, 1.0, 1.0))
    )
    result = boolean_brep(first, second, operation)
    assert result.empty
    assert result.model.topology.num_faces == 0
    assert not result.association_graph.target_revision.occurrences
    assert set(result.deleted_source_ids) == {
        occurrence.occurrence_id
        for occurrence in result.association_graph.source_revision.occurrences
    }


def test_coincident_union_removes_duplicate_boundary_faces() -> None:
    box = _box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    result = boolean_brep(box, box, "union")
    assert result.model.topology.num_faces == 6
    assert _volume(result.model) == pytest.approx(1.0, abs=1e-12)
    mapped = {
        edge.source_occurrence_id
        for edge in result.association_graph.transaction.correspondences
    }
    assert mapped == {
        occurrence.occurrence_id
        for occurrence in result.association_graph.source_revision.occurrences
    }


def test_native_boolean_model_archive_retains_exact_topology(tmp_path: Path) -> None:
    result = boolean_brep(
        _box((0.0, 0.0, 0.0), (2.0, 2.0, 2.0)),
        _box((1.0, 1.0, 1.0), (3.0, 3.0, 3.0)),
        "difference",
    )
    path = tmp_path / "boolean.phx"
    save_brep_archive(result.model, path)
    restored = load_brep_archive(path)
    assert restored.model_id == result.model.model_id
    assert restored.geometry is not None
    assert result.model.geometry is not None
    assert restored.geometry.geometry_id == result.model.geometry.geometry_id
    assert _volume(restored) == pytest.approx(7.0, abs=1e-12)


def test_arrangement_budget_refusal_preserves_source_revision() -> None:
    first = _box((0.0, 0.0, 0.0), (2.0, 2.0, 2.0))
    second = _box((1.0, 1.0, 1.0), (3.0, 3.0, 3.0))
    original = first.model_id
    with pytest.raises(BRepBooleanFailure, match="cell budget exhausted"):
        boolean_brep(first, second, "union", policy=BRepBooleanPolicy(maximum_cells=1))
    assert first.model_id == original


def test_curved_intersection_budget_preserves_source_revision() -> None:
    sphere = brep_sphere(1.0, coordinate_contract=_COORDINATES)
    original = sphere.model_id
    policy = BRepBooleanPolicy(
        intersection=ParametricIntersectionPolicy(maximum_boxes=1),
    )
    with pytest.raises(BRepBooleanFailure):
        boolean_brep(
            sphere,
            _box((0.3, -2.0, -2.0), (2.0, 2.0, 2.0)),
            "intersection",
            policy=policy,
        )
    assert sphere.model_id == original


@pytest.mark.parametrize("family", ["sphere", "cylinder"])
def test_analytic_coincidence_retains_one_exact_solid(family: str) -> None:
    model = (
        brep_sphere(1.0, coordinate_contract=_COORDINATES)
        if family == "sphere"
        else brep_cylinder(1.0, 2.0, coordinate_contract=_COORDINATES)
    )
    result = boolean_brep(model, model, "union")
    assert result.model.topology.num_solids == 1
    assert result.model.geometry is not None
    assert model.geometry is not None
    assert result.model.geometry.geometry_id == model.geometry.geometry_id
    graph = result.association_graph
    mapped = {edge.source_occurrence_id for edge in graph.transaction.correspondences}
    assert mapped == {
        occurrence.occurrence_id for occurrence in graph.source_revision.occurrences
    }
    empty = boolean_brep(model, model, "difference")
    assert empty.empty
    assert not empty.association_graph.target_revision.occurrences


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        pytest.param("intersection", 5.0 * math.pi / 12.0, id="lens"),
        pytest.param("union", 9.0 * math.pi / 4.0, id="union"),
        pytest.param("difference", 11.0 * math.pi / 12.0, id="difference"),
    ],
)
def test_overlapping_spheres_retain_exact_coupled_boundary(
    operation: BRepBooleanOperation,
    expected: float,
) -> None:
    first = brep_sphere(1.0, coordinate_contract=_COORDINATES, tessellation=_EXACT_SOURCE)
    second = brep_sphere(
        1.0,
        center=(0.0, 1.0, 0.0),
        coordinate_contract=_COORDINATES,
        tessellation=_EXACT_SOURCE,
    )
    result = boolean_brep(
        first, second, operation, policy=BRepBooleanPolicy(tessellation=_EXACT_SOURCE)
    )
    assert result.model.geometry is not None
    assert any(
        isinstance(curve, IntersectionCurve) for curve in result.model.geometry.curves
    )
    measures = prepare_brep_query(result.model).measures
    assert float(np.sum(np.asarray(measures.solid_volumes))) == pytest.approx(
        expected, rel=1e-6
    )
    assert result.association_graph.transaction.coverage.source_exhaustive
    assert result.association_graph.transaction.coverage.target_exhaustive


def test_cylinder_plane_cut_keeps_irrational_root_bindings() -> None:
    cylinder = brep_cylinder(
        1.0,
        2.0,
        base_center=(0.0, 0.0, -1.0),
        coordinate_contract=_COORDINATES,
        tessellation=_EXACT_SOURCE,
    )
    cutter = brep_box(
        (0.3, -2.0, -2.0),
        (2.0, 2.0, 2.0),
        coordinate_contract=_COORDINATES,
        tessellation=_EXACT_SOURCE,
    )
    result = boolean_brep(
        cylinder,
        cutter,
        "intersection",
        policy=BRepBooleanPolicy(tessellation=_EXACT_SOURCE),
    )
    geometry = result.model.geometry
    assert geometry is not None
    assert any(root is not None for root in geometry.vertex_roots)
    expected = 2.0 * (math.acos(0.3) - 0.3 * math.sqrt(0.91))
    actual = float(
        np.sum(np.asarray(prepare_brep_query(result.model).measures.solid_volumes))
    )
    assert actual == pytest.approx(expected, rel=1e-6)


def test_containment_on_a_face_plane_is_outside_or_on_boundary() -> None:
    cylinder = brep_cylinder(
        1.0,
        2.0,
        base_center=(0.0, 0.0, -1.0),
        coordinate_contract=_COORDINATES,
        tessellation=_EXACT_SOURCE,
    )
    # On the cap plane, inside the bounding box, outside the disk: every ray
    # starts on that plane, and the solid is still certified not to contain it.
    outside = certify_solid_containment(
        cylinder,
        np.asarray([0.9, 0.9, -1.0]),
        0,
        maximum_boxes=200_000,
        maximum_depth=24,
    )
    assert outside.complete and not outside.inside
    assert outside.ray_start < 0.0
    # On the cap face itself the point is boundary: parity is never claimed.
    boundary = certify_solid_containment(
        cylinder,
        np.asarray([0.25, 0.125, -1.0]),
        0,
        maximum_boxes=200_000,
        maximum_depth=24,
    )
    assert not boundary.complete


def test_closest_point_lower_bound_ignores_face_plane_outside_the_trim() -> None:
    cylinder = brep_cylinder(
        1.0, 2.0, base_center=(0.0, 0.0, -1.0), coordinate_contract=_COORDINATES
    )
    query = prepare_brep_query(cylinder)
    # On the cap's supporting plane, inside its square chart, outside its disk.
    closest = query.closest_point(np.asarray([[0.9, 0.9, -1.0]]))
    distance = float(np.asarray(closest.distances)[0])
    lower = float(np.asarray(closest.distance_lower_bounds)[0])
    assert distance == pytest.approx(math.hypot(0.9, 0.9) - 1.0, abs=1e-6)
    assert 0.0 < lower <= distance


def test_nonrational_quadric_boolean_archive_preserves_authority(tmp_path: Path) -> None:
    first = brep_cylinder(
        1.0,
        4.0,
        base_center=(0.0, 0.0, -2.0),
        coordinate_contract=_COORDINATES,
        tessellation=_EXACT_SOURCE,
    )
    second = brep_cylinder(
        0.8,
        4.0,
        base_center=(-2.0, 0.0, 0.0),
        axis=(1.0, 0.0, 0.0),
        coordinate_contract=_COORDINATES,
        tessellation=_EXACT_SOURCE,
    )
    result = boolean_brep(
        first,
        second,
        "intersection",
        policy=BRepBooleanPolicy(tessellation=_EXACT_SOURCE),
    )
    geometry = result.model.geometry
    assert geometry is not None
    branches = tuple(
        curve for curve in geometry.curves if isinstance(curve, IntersectionCurve)
    )
    assert branches
    branch = branches[0]
    parameters = np.linspace(0.2, branch.num_charts - 0.2, 11, dtype=np.float64)
    evaluated = branch.evaluate(parameters)
    first_points = np.asarray(branch.first.patch.evaluate(evaluated.first_parameters))
    second_points = np.asarray(branch.second.patch.evaluate(evaluated.second_parameters))
    np.testing.assert_allclose(first_points, second_points, atol=1e-10, rtol=0.0)
    nodes, weights = np.polynomial.legendre.leggauss(96)
    angle = 0.5 * math.pi * nodes
    expected = (
        0.5
        * math.pi
        * np.sum(
            weights
            * 4.0
            * 0.8**2
            * np.cos(angle) ** 2
            * np.sqrt(1.0 - 0.8**2 * np.sin(angle) ** 2)
        )
    )
    actual = float(
        np.sum(np.asarray(prepare_brep_query(result.model).measures.solid_volumes))
    )
    assert actual == pytest.approx(expected, rel=1e-6)
    path = tmp_path / "nonrational.phx"
    save_brep_archive(result.model, path)
    restored = load_brep_archive(path)
    assert restored.model_id == result.model.model_id
    assert restored.geometry is not None
    restored_branches = tuple(
        curve
        for curve in restored.geometry.curves
        if isinstance(curve, IntersectionCurve)
    )
    assert tuple(curve.branch_id for curve in restored_branches) == tuple(
        curve.branch_id for curve in branches
    )
    with pytest.raises(CadInterchangeError, match="bounded approximation"):
        write_step(restored, tmp_path / "nonrational.step")


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        pytest.param("union", 3.0 * math.pi, id="union"),
        pytest.param("intersection", math.pi, id="intersection"),
        pytest.param("difference", math.pi, id="difference"),
    ],
)
def test_coincident_cylinder_support_preserves_region_volume(
    operation: BRepBooleanOperation,
    expected: float,
    tmp_path: Path,
) -> None:
    first = brep_cylinder(1.0, 2.0, coordinate_contract=_COORDINATES)
    second = brep_cylinder(
        1.0, 2.0, base_center=(0.0, 0.0, 1.0), coordinate_contract=_COORDINATES
    )
    result = boolean_brep(first, second, operation)
    actual = float(
        np.sum(np.asarray(prepare_brep_query(result.model).measures.solid_volumes))
    )
    assert actual == pytest.approx(expected, rel=1e-10)
    assert result.model.topology.num_solids == 1
    assert result.association_graph.transaction.coverage.source_exhaustive
    assert result.association_graph.transaction.coverage.target_exhaustive
    geometry = result.model.geometry
    assert geometry is not None
    path = tmp_path / f"coincident-cylinder-{operation}.phx"
    save_brep_archive(result.model, path)
    restored = load_brep_archive(path)
    assert restored.model_id == result.model.model_id
    assert restored.geometry is not None
    authored = tuple(
        (coedge, endpoint, root)
        for coedge, roots in enumerate(geometry.coedge_endpoint_roots)
        for endpoint, root in enumerate(roots)
        if isinstance(root, NativePeriodEndpoint)
    )
    assert authored
    for coedge, endpoint, root in authored:
        retained = restored.geometry.coedge_endpoint_roots[coedge][endpoint]
        assert isinstance(retained, NativePeriodEndpoint)
        assert (retained.rational, retained.turns, retained.axis) == (
            root.rational,
            root.turns,
            root.axis,
        )
        np.testing.assert_allclose(
            np.asarray(
                retained.carrier.evaluate(
                    jnp.asarray(retained.parameter, dtype=jnp.float64)
                )
            ),
            np.asarray(
                geometry.pcurves[coedge].evaluate(
                    jnp.asarray(root.parameter, dtype=jnp.float64)
                )
            ),
            rtol=0.0,
            atol=1.0e-12,
        )


def test_tangent_coaxial_cylinders_have_regularized_empty_intersection() -> None:
    first = brep_cylinder(1.0, 2.0, coordinate_contract=_COORDINATES)
    second = brep_cylinder(
        1.0, 2.0, base_center=(0.0, 0.0, 2.0), coordinate_contract=_COORDINATES
    )
    result = boolean_brep(first, second, "intersection")
    assert result.empty
    assert not result.association_graph.target_revision.occurrences


@pytest.mark.parametrize("family", ["plane", "cylinder", "sphere"])
def test_nonbinary_placed_coincidence_requires_exact_source_identity(family: str) -> None:
    rotation = np.asarray(
        (
            (math.cos(0.37), -math.sin(0.37), 0.0),
            (math.sin(0.37), math.cos(0.37), 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    shift = np.asarray((0.1, 0.2, 0.3), dtype=np.float64)
    origin = (0.1, 0.2, 0.3)
    different = (0.1, 0.2, float(np.nextafter(0.3, np.inf)))
    if family == "plane":
        source = PlanePatch(origin, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        other = PlanePatch(origin, (2.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        distinct = PlanePatch(different, (2.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        expected = ((Fraction(1, 2), Fraction(0)), (Fraction(0), Fraction(1)))
    elif family == "cylinder":
        source = CylinderPatch(
            origin, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), 0.1
        )
        other = source
        distinct = CylinderPatch(
            origin,
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            float(np.nextafter(0.1, np.inf)),
        )
        expected = ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)))
    else:
        source = SpherePatch(
            origin, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), 0.1
        )
        other = source
        distinct = SpherePatch(
            different, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), 0.1
        )
        expected = ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)))
    first = PlacedSurface(source, rotation, shift)
    second = PlacedSurface(
        PlacedSurface(other, rotation, shift),
        np.eye(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
    )
    proof = prove_surface_correspondence(first, second)
    assert proof is not None
    assert proof.matrix == expected
    assert proof.offset == (Fraction(0), Fraction(0))
    assert proof.first is first and proof.second is second
    assert (
        prove_surface_correspondence(
            first,
            PlacedSurface(distinct, rotation, shift),
        )
        is None
    )
