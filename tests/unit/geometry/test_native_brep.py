import subprocess
import sys
import textwrap
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax.geometry.brep import (
    BRepQueryBudget,
    BRepQueryResourceError,
    BSplineCurve,
    ExtrusionSurface,
)
from phydrax.units import METER
from tests._support.cad_models import native_solid


# Empirical `test_native_projection_reports_unique_seam_and_continuum_outcomes`: matched max |delta|=1.11022e-16; joint allowance fraction=1.19726e-09, margin=100%.
_CYLINDER_PROJECTION_PARAMETERS_ATOL = 1e-12
# Empirical `test_native_projection_reports_unique_seam_and_continuum_outcomes`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
_CYLINDER_PROJECTION_NORMAL_ATOL = 1e-12
# Empirical `test_native_projection_reports_unique_seam_and_continuum_outcomes`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
_CYLINDER_NORMAL_TANGENT_PAIRING_ATOL = 1e-12
# Empirical `test_native_edge_projection_handles_seams_and_circle_centers`: matched max |delta|=6.12323e-17; joint allowance fraction=6.12323e-05, margin=99.9939%.
_CIRCLE_PROJECTION_TANGENT_ATOL = 1e-12
# Empirical `test_native_sphere_projection_distinguishes_pole_and_center`: matched max |delta|=6.12323e-17; joint allowance fraction=6.12323e-08, margin=100%.
_SPHERE_POLE_LIMIT_NORMAL_ATOL = 1e-9
# Empirical `test_native_boundary_closest_point_reports_multiple_minima`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
_TORUS_AMBIGUOUS_CLOSEST_POINT_ATOL = 1e-12
# Empirical `test_cylinder_global_closest_retains_far_stationary_and_rim_candidates`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
_CYLINDER_GLOBAL_CLOSEST_POINT_ATOL = 1e-8
# Empirical `test_cylinder_global_closest_retains_far_stationary_and_rim_candidates`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
_CYLINDER_GLOBAL_CLOSEST_DISTANCE_ATOL = 1e-8
# Empirical `test_authored_curved_occurrence_queries_use_world_metric`: matched max |delta|=1.77636e-14; joint allowance fraction=1.77636e-06, margin=99.9998%.
_OCCURRENCE_WORLD_CLOSEST_POINT_ATOL = 1e-8
# Empirical `test_authored_curved_occurrence_queries_use_world_metric`: matched max |delta|=1.9762e-14; joint allowance fraction=1.9762e-06, margin=99.9998%.
_OCCURRENCE_WORLD_CLOSEST_DISTANCE_ATOL = 1e-8
# Empirical `test_authored_curved_occurrence_queries_use_world_metric`: matched max |delta|=1.77694e-14; joint allowance fraction=1.77694e-06, margin=99.9998%.
_OCCURRENCE_WORLD_NORMAL_ATOL = 1e-8
# Empirical `test_reflected_live_world_radius_has_independent_jvp_vjp`: matched max |delta|=1.11022e-16; joint allowance fraction=1.11022e-08, margin=100%.
_REFLECTED_WORLD_DISTANCE_VALUE_ATOL = 1e-8
# Empirical `test_reflected_live_world_radius_has_independent_jvp_vjp`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
_REFLECTED_WORLD_RADIUS_TANGENT_ATOL = 1e-8
# Empirical `test_reflected_live_world_radius_has_independent_jvp_vjp`: matched max |delta|=1.11022e-16; joint allowance fraction=1.11022e-08, margin=100%.
_REFLECTED_WORLD_RADIUS_COTANGENT_ATOL = 1e-8
# Empirical `test_native_planar_face_projection_respects_trim_holes`: matched max |delta|=4.55946e-12; joint allowance fraction=0.00310862, margin=99.6891%.
_PLANAR_TRIM_PROJECTION_RESIDUAL_ATOL = 1e-12
# Empirical `test_native_brep_source_answers_region_queries_exactly`: matched max |delta|=1.78181e-14; joint allowance fraction=0.0178181, margin=98.2182%.
_SPHERE_SOURCE_CLOSEST_POINT_ATOL = 1e-12
# Empirical `test_native_brep_boundary_field_compiles_with_exact_source_point_jvp`: matched max |delta|=6.12323e-17; joint allowance fraction=6.12323e-05, margin=99.9939%.
_SPHERE_BOUNDARY_FIELD_VALUE_ATOL = 1e-12
# Empirical `test_native_brep_boundary_field_compiles_with_exact_source_point_jvp`: matched max |delta|=1.77636e-14; joint allowance fraction=0.0177636, margin=98.2236%.
_SPHERE_BOUNDARY_FIELD_TANGENT_ATOL = 1e-12
# Empirical `test_large_translated_source_queries_preserve_distance_and_boundary_proof`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
_TRANSLATED_SOURCE_DISTANCE_ATOL = 1e-14
# Empirical `test_compiled_spline_extrusion_tracks_dynamic_knot_endpoints`: matched max |delta|=2.77556e-17; joint allowance fraction=0.000253476, margin=99.9747%.
_DYNAMIC_KNOT_EXTRUSION_POINT_RTOL = 1e-13
# Empirical `test_compiled_spline_extrusion_tracks_dynamic_knot_endpoints`: matched max |delta|=2.77556e-17; joint allowance fraction=0.000253476, margin=99.9747%.
_DYNAMIC_KNOT_EXTRUSION_POINT_ATOL = 1e-13
# Empirical `test_compiled_spline_extrusion_tracks_dynamic_knot_endpoints`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
_DYNAMIC_KNOT_EXTRUSION_TANGENT_RTOL = 1e-13
# Empirical `test_compiled_spline_extrusion_tracks_dynamic_knot_endpoints`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
_DYNAMIC_KNOT_EXTRUSION_TANGENT_ATOL = 1e-13
# Empirical `test_rational_first_jet_bounds_include_closed_knot_endpoint`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
_RATIONAL_ENDPOINT_FIRST_JET_ATOL = 1e-15
# Empirical `test_native_solid_measures_match_analytic_values`: matched max |delta|=3.55271e-15; joint allowance fraction=3.56355e-05, margin=99.9964%.
_NATIVE_SOLID_VOLUME_RTOL = 1e-11
# Empirical `test_native_solid_measures_match_analytic_values`: matched max |delta|=1.42109e-14; joint allowance fraction=3.75957e-05, margin=99.9962%.
_NATIVE_SOLID_SURFACE_AREA_RTOL = 1e-11
# Empirical `test_native_brep_source_answers_region_queries_exactly`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
_SPHERE_SOURCE_VOLUME_RTOL = 1e-11
# Empirical `test_native_brep_source_answers_region_queries_exactly`: matched max |delta|=3.55271e-15; joint allowance fraction=2.80484e-05, margin=99.9972%.
_SPHERE_SOURCE_SURFACE_AREA_RTOL = 1e-11
# Empirical `test_fixed_topology_native_measure_is_exact_and_differentiable`: matched max |delta|=8.88178e-16; joint allowance fraction=2.07093e-05, margin=99.9979%.
_DYNAMIC_SPHERE_VOLUME_RTOL = 1e-11
# Empirical `test_fixed_topology_native_measure_is_exact_and_differentiable`: matched max |delta|=3.55271e-15; joint allowance fraction=2.82491e-06, margin=99.9997%.
_DYNAMIC_SPHERE_VOLUME_DERIVATIVE_RTOL = 1e-10
# Empirical `test_rational_bezier_pieces_reproduce_the_spline_exactly`: matched max |delta|=4.44089e-16; joint allowance fraction=2.95519e-08, margin=100%.
_RATIONAL_BEZIER_POINT_ATOL = 1e-13


_CONTRACT = phx.SpatialCoordinateContract.si()
_COARSE = phx.geometry.BRepTessellationPolicy(
    linear_deflection=0.05, angular_deflection=0.4
)
_STATUS = phx.geometry.BRepProjectionStatus
_PI = np.pi


def _plate_profile() -> Any:
    return phx.geometry.PlanarProfile(
        phx.geometry.ProfilePlane(),
        phx.geometry.ProfileLoop.polygon(((-2, -2), (2, -2), (2, 2), (-2, 2))),
        (phx.geometry.ProfileLoop.circle((0.0, 0.0), 1.0),),
    )


_MODELS: dict[str, Any] = {}
_QUERIES: dict[str, Any] = {}


def _model(name: str) -> Any:
    if name not in _MODELS:
        _MODELS[name] = native_solid(name, catalog="geometry", tessellation=_COARSE)
    return _MODELS[name]


def _query(name: str) -> Any:
    if name not in _QUERIES:
        _QUERIES[name] = phx.geometry.prepare_brep_query(_model(name))
    return _QUERIES[name]


# Independent analytic references: (V, non-degenerate E, F,
# V - E + F - (L - F)), volume, area.
_SOLIDS = {
    "box": ((8, 12, 6, 2), 6.0, 22.0),
    "cylinder": ((2, 3, 3, 2), 2 * _PI, 6 * _PI),
    "cone": ((2, 2, 2, 2), 2 * _PI / 3, _PI + _PI * np.sqrt(5.0)),
    "sphere": ((2, 1, 1, 2), 4 * _PI / 3, 4 * _PI),
    "torus": ((1, 2, 1, 0), 2 * _PI**2 * 2 * 0.25, 4 * _PI**2 * 2 * 0.5),
    "extruded_plate": ((10, 15, 7, 0), 0.5 * (16 - _PI), 2 * (16 - _PI) + 8 + _PI),
    "revolved_ring": ((4, 6, 4, 0), 3 * _PI, 4 * _PI + 2 * _PI + 6 * _PI),
    "revolved_quarter": ((8, 12, 6, 2), 0.75 * _PI, 3 * _PI + 2),
}


@pytest.mark.parametrize("name", sorted(_SOLIDS), ids=sorted(_SOLIDS))
def test_native_solid_topology_is_exact_and_closed(name: str) -> None:
    model = _model(name)
    geometry = model.geometry
    (vertices, edges, faces, euler), _, _ = _SOLIDS[name]
    counted_edges = sum(1 for index in geometry.edge_curves if index != -1)

    assert (model.topology.num_vertices, counted_edges, model.topology.num_faces) == (
        vertices,
        edges,
        faces,
    )
    assert geometry.euler_characteristic() == euler
    assert model.topology.num_solids == 1
    # Every non-degenerate edge is used once in each direction by the
    # consistently oriented closed shell.
    balance = geometry.edge_use_balance(model.orientation)
    assert all(
        value == 0
        for value, degenerate in zip(balance, geometry.degenerate_edges, strict=True)
        if not degenerate
    )
    mesh = phx.geometry.TriangleMesh(model.mesh_vertices, model.mesh_faces)
    assert mesh.topology.watertight
    assert (
        np.max(np.asarray(model.tessellation_deviation_bounds))
        <= _COARSE.linear_deflection
    )
    assert (
        np.max(np.asarray(model.tessellation_normal_bounds)) <= _COARSE.angular_deflection
    )
    assert model.report.source_format == "native"


@pytest.mark.parametrize("name", sorted(_SOLIDS), ids=sorted(_SOLIDS))
def test_native_solid_measures_match_analytic_values(name: str) -> None:
    measures = _query(name).measures
    _, volume, area = _SOLIDS[name]

    assert float(measures.solid_volumes[0]) == pytest.approx(
        volume, rel=_NATIVE_SOLID_VOLUME_RTOL
    )
    assert float(jnp.sum(measures.face_areas)) == pytest.approx(
        area, rel=_NATIVE_SOLID_SURFACE_AREA_RTOL
    )
    assert float(measures.solid_volume_error_estimates[0]) <= 1e-10
    assert float(jnp.max(measures.face_area_error_estimates)) <= 1e-10


def test_native_model_identity_is_independent_of_tessellation() -> None:
    coarse = phx.geometry.brep_cylinder(
        1.0, 2.0, coordinate_contract=_CONTRACT, tessellation=_COARSE
    )
    fine = phx.geometry.brep_cylinder(
        1.0,
        2.0,
        coordinate_contract=_CONTRACT,
        tessellation=phx.geometry.BRepTessellationPolicy(
            linear_deflection=0.01, angular_deflection=0.2
        ),
    )
    taller = phx.geometry.brep_cylinder(
        1.0, 2.5, coordinate_contract=_CONTRACT, tessellation=_COARSE
    )

    assert fine.mesh_faces.shape[0] > coarse.mesh_faces.shape[0]
    assert coarse.model_id == fine.model_id
    assert coarse.source_revision == fine.source_revision
    assert coarse.geometry is not None and fine.geometry is not None
    assert coarse.geometry.geometry_id == fine.geometry.geometry_id
    assert coarse.tessellation_id != fine.tessellation_id
    assert taller.model_id != coarse.model_id
    assert taller.source_revision != coarse.source_revision


def test_native_projection_reports_unique_seam_and_continuum_outcomes() -> None:
    model = _model("cylinder")
    projection = phx.geometry.prepare_brep_projection(model)
    lateral = model.physical_tags.index("cylinder")
    points = np.asarray(
        [[0.3, 0.4, 1.0], [1.5, 1.5, 0.25], [2.0, 0.0, 0.5], [0.0, 0.0, 1.0]]
    )
    result = projection.project(points, np.full(4, 2), np.full(4, lateral))

    np.testing.assert_allclose(
        np.asarray(result.points)[:2], [[0.6, 0.8, 1.0], [2**-0.5, 2**-0.5, 0.25]]
    )
    np.testing.assert_allclose(
        np.asarray(result.parameters)[:3],
        [[np.arctan2(0.8, 0.6), 1.0], [_PI / 4, 0.25], [0.0, 0.5]],
        atol=_CYLINDER_PROJECTION_PARAMETERS_ATOL,
    )
    np.testing.assert_allclose(
        result.residuals, [0.5, np.hypot(1.5, 1.5) - 1.0, 1.0, 1.0]
    )
    # A point on the seam has two parameter representatives; a point on the
    # axis has a whole circle of closest points.
    assert np.asarray(result.status).tolist() == [
        _STATUS.UNIQUE,
        _STATUS.UNIQUE,
        _STATUS.SEAM,
        _STATUS.NON_ISOLATED,
    ]
    assert np.all(np.isnan(np.asarray(result.points)[3]))
    assert np.isnan(np.asarray(result.parameters)[3, 0])
    assert int(result.indices[3]) == lateral
    assert not bool(result.resolved[3])
    assert np.isfinite(float(result.residuals[3]))
    assert float(result.residuals[3]) >= 1.0
    np.testing.assert_allclose(
        np.asarray(result.normals)[:2],
        [[0.6, 0.8, 0.0], [2**-0.5, 2**-0.5, 0.0]],
        atol=_CYLINDER_PROJECTION_NORMAL_ATOL,
    )
    tangents = np.asarray(result.tangents)[:2]
    np.testing.assert_allclose(
        np.sum(tangents * np.asarray(result.normals)[:2, None, :], axis=-1),
        0.0,
        atol=_CYLINDER_NORMAL_TANGENT_PAIRING_ATOL,
    )


def test_native_edge_projection_handles_seams_and_circle_centers() -> None:
    projection = phx.geometry.prepare_brep_projection(_model("cylinder"))
    assert isinstance(projection, phx.geometry.NativeBRepProjection)
    vertices = np.asarray(projection.vertex_points)
    edge_vertices = np.asarray(projection.edge_vertices)
    closed = np.flatnonzero(np.asarray(projection.edge_closed))
    rim = int(closed[np.isclose(vertices[edge_vertices[closed, 0], 2], 2.0)][0])
    points = np.asarray([[0.0, 2.0, 2.0], [3.0, 0.0, 2.0], [0.0, 0.0, 2.0]])
    result = projection.project(points, np.full(3, 1), np.full(3, rim))

    assert np.asarray(result.status).tolist() == [
        _STATUS.UNIQUE,
        _STATUS.SEAM,
        _STATUS.AMBIGUOUS,
    ]
    np.testing.assert_allclose(np.asarray(result.parameters)[0, 0], _PI / 2)
    np.testing.assert_allclose(result.residuals, [1.0, 2.0, 1.0])
    np.testing.assert_allclose(
        np.abs(np.asarray(result.tangents)[0, 0]),
        [1.0, 0.0, 0.0],
        atol=_CIRCLE_PROJECTION_TANGENT_ATOL,
    )


def test_native_sphere_projection_distinguishes_pole_and_center() -> None:
    projection = phx.geometry.prepare_brep_projection(_model("sphere"))
    query = np.asarray([[0.3, 0.4, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 0.0]])
    result = projection.project(query, np.full(3, 2), np.zeros(3, dtype=np.int64))
    radius = np.linalg.norm(query[0])

    np.testing.assert_allclose(np.asarray(result.points)[0], query[0] / radius)
    np.testing.assert_allclose(
        np.asarray(result.parameters)[0],
        [np.arctan2(0.4, 0.3), np.arcsin(1.0 / radius)],
    )
    np.testing.assert_allclose(result.residuals, [radius - 1.0, 1.0, 1.0])
    assert np.asarray(result.status).tolist() == [
        _STATUS.UNIQUE,
        _STATUS.SEAM,
        _STATUS.AMBIGUOUS,
    ]
    # The limit normal at the pole is the outward axis direction.
    np.testing.assert_allclose(
        np.asarray(result.normals)[1],
        [0.0, 0.0, 1.0],
        atol=_SPHERE_POLE_LIMIT_NORMAL_ATOL,
    )


def test_native_boundary_closest_point_reports_multiple_minima() -> None:
    box = _query("box")
    torus = _query("torus")
    closest = box.closest_point(np.asarray([[0.5, 1.0, 1.2], [0.2, 1.0, 1.5]]))
    ring = torus.closest_point(np.asarray([[0.0, 0.0, 1.0], [3.0, 0.0, 0.0]]))

    # Equidistant walls x = 0 and x = 1 give two isolated minima.
    assert np.asarray(closest.status).tolist() == [_STATUS.AMBIGUOUS, _STATUS.UNIQUE]
    np.testing.assert_allclose(closest.distances, [0.5, 0.2])
    np.testing.assert_allclose(np.asarray(closest.points)[1], [0.0, 1.0, 1.5])
    # On the torus axis the closest points form a circle.
    assert int(ring.status[0]) == _STATUS.AMBIGUOUS
    np.testing.assert_allclose(ring.distances, [np.sqrt(5.0) - 0.5, 0.5])
    np.testing.assert_allclose(
        np.asarray(ring.points)[1],
        [2.5, 0.0, 0.0],
        atol=_TORUS_AMBIGUOUS_CLOSEST_POINT_ATOL,
    )


@pytest.mark.parametrize(
    ("name", "inside", "outside"),
    [
        (
            "cylinder",
            [[0, 0, 1], [0.9, 0, 0.1]],
            [[0, 0, 2.5], [1.2, 0, 1], [1.1, 1.1, 2.1]],
        ),
        ("sphere", [[0, 0, 0], [0, 0, 0.99]], [[0, 0, 1.01], [1, 1, 1]]),
        ("torus", [[2.0, 0, 0], [0, -2.3, 0.2]], [[0, 0, 0], [1.2, 0, 0], [2, 0, 0.6]]),
        ("box", [[0.5, 1, 1.5], [0.2, 0.2, 0.2]], [[1.1, 2.1, 3.1], [-0.1, 1, 1]]),
        ("extruded_plate", [[1.5, 1.5, 0.25]], [[0, 0, 0.25], [0.5, 0.5, 0.25]]),
    ],
    ids=["cylinder", "sphere", "torus", "box", "extruded_plate"],
)
def test_native_containment_certifies_inside_and_outside(
    name: str, inside: Any, outside: Any
) -> None:
    points = np.asarray(inside + outside, dtype=np.float64)
    result = _query(name).contains(points)

    assert np.asarray(result.inside).tolist() == [True] * len(inside) + [False] * len(
        outside
    )
    assert np.all(np.asarray(result.status) == _STATUS.UNIQUE)


def test_native_containment_reports_boundary_points_as_ambiguous() -> None:
    result = _query("cylinder").contains(np.asarray([[1.0, 0.0, 1.0], [0.0, 0.5, 2.0]]))

    assert np.asarray(result.inside).tolist() == [True, True]
    assert np.all(np.asarray(result.status) == _STATUS.AMBIGUOUS)


def test_native_classification_and_solid_location() -> None:
    projection = phx.geometry.prepare_brep_projection(_model("cylinder"))
    points = np.asarray(
        [[0.0, 1.0, 1.0], [0.0, 1.0, 2.0], [1.0, 0.0, 2.0], [0.2, 0.1, 1.0]]
    )
    classified = projection.classify(points, tolerance=1e-9)
    located = projection.locate_solids(points)

    assert np.asarray(classified.dimensions).tolist() == [2, 1, 0, -1]
    assert np.asarray(classified.status)[-1] == _STATUS.FAILED
    assert np.asarray(located.status).tolist() == [
        _STATUS.AMBIGUOUS,
        _STATUS.AMBIGUOUS,
        _STATUS.AMBIGUOUS,
        _STATUS.UNIQUE,
    ]
    rim = int(classified.indices[1])
    corner = int(classified.indices[2])
    assert np.asarray(
        projection.contains(
            np.asarray([2, 1, 1]),
            np.asarray([1, rim, rim]),
            np.asarray([1, 0, 2]),
            np.asarray([rim, corner, 0]),
        )
    ).tolist() == [True, True, False]


def test_cylinder_global_closest_retains_far_stationary_and_rim_candidates() -> None:
    points = np.asarray(((101.0, 0.0, 1.0), (2.0, 0.0, 3.0), (0.25, 0.0, 1.75)))
    result = _query("cylinder").closest_point(points)
    # Independent minimization over the Cartesian finite cylinder: the first
    # point selects the near lateral stationary point (not its opposite),
    # the second the rim, and the interior point the top-cap interior.
    expected = np.asarray(((1.0, 0.0, 1.0), (1.0, 0.0, 2.0), (0.25, 0.0, 2.0)))
    distances = np.asarray((100.0, np.sqrt(2.0), 0.25))
    np.testing.assert_allclose(
        result.points, expected, atol=_CYLINDER_GLOBAL_CLOSEST_POINT_ATOL, rtol=0.0
    )
    np.testing.assert_allclose(
        result.distances, distances, atol=_CYLINDER_GLOBAL_CLOSEST_DISTANCE_ATOL, rtol=0.0
    )
    assert not np.any(np.asarray(result.unresolved))
    assert np.all(np.asarray(result.distance_lower_bounds) <= distances)
    assert np.all(np.asarray(result.distances) >= distances - 1e-14)
    assert np.all(np.asarray(result.query_operations) > 0)


def _query_occurrence_model(
    name: str, rotation: np.ndarray, translation: np.ndarray
) -> Any:
    from phydrax.geometry.brep import BRepGeometry, BRepOccurrence
    from phydrax.geometry.brep._constructors import assemble_brep_model

    source = _model(name)
    original = source.geometry
    assert original is not None
    geometry = BRepGeometry(
        vertex_points=original.vertex_points,
        curves=original.curves,
        edge_curves=original.edge_curves,
        edge_ranges=original.edge_ranges,
        edge_vertices=original.edge_vertices,
        pcurves=original.pcurves,
        coedge_edges=original.coedge_edges,
        coedge_senses=original.coedge_senses,
        face_loops=original.face_loops,
        shell_faces=original.shell_faces,
        shell_orientations=original.shell_orientations,
        solid_shells=original.solid_shells,
        occurrences=(BRepOccurrence(("reflected",), 0, rotation, translation),),
    )
    return assemble_brep_model(
        geometry,
        source.patches,
        source.parameter_bounds,
        source.orientation,
        source.physical_tags,
        coordinate_contract=_CONTRACT,
        source_id="reflected-query-oracle",
        source_format="native",
        source_digest=source.source_digest,
        import_policy_id=source.import_policy_id,
        tessellation=phx.geometry.BRepTessellationPolicy(realize=False),
    )


def test_reflected_assembly_location_uses_world_source_and_row_scope() -> None:
    reflection = np.diag((-1.0, 1.0, 1.0))
    translation = np.asarray((2.0**30, -(2.0**30), 2.0**30))
    model = _query_occurrence_model("box", reflection, translation)
    # Independent Cartesian membership oracle: reflection maps x in [0,1]
    # to world x in [translation_x-1, translation_x].
    offsets = np.asarray(((-0.5, 1.0, 1.5), (0.0, 1.0, 1.5), (0.5, 1.0, 1.5)))
    points = translation + offsets
    projection = phx.geometry.prepare_brep_projection(model)
    result = projection.locate_solids(
        points,
        occurrence_paths=(("reflected",),) * len(points),
    )
    np.testing.assert_array_equal(result.dimensions, (3, 3, -1))
    np.testing.assert_array_equal(
        result.status,
        (_STATUS.UNIQUE, _STATUS.AMBIGUOUS, _STATUS.FAILED),
    )
    assert result.occurrence_paths == projection.occurrence_paths
    with pytest.raises(BRepQueryResourceError):
        limited = phx.geometry.prepare_brep_projection(
            model,
            query_policy=phx.geometry.BRepQueryPolicy(maximum_points=2),
        )
        limited.locate_solids(
            points,
            occurrence_paths=(("reflected",),) * len(points),
        )


@pytest.mark.parametrize("reflection", (False, True))
def test_authored_curved_occurrence_queries_use_world_metric(reflection: bool) -> None:
    # Accepted binary64 authored coefficients are deliberately not normalized:
    # this source occurrence is a slightly stretched sphere, not a sphere.
    scale = np.nextafter(1.0, 2.0)
    rotation = np.diag(((-scale if reflection else scale), 1.0, 1.0))
    translation = np.asarray((4.0, -3.0, 2.0))
    model = _query_occurrence_model("sphere", rotation, translation)
    query = phx.geometry.prepare_brep_query(model)
    points = translation + np.asarray(((3.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 2.0)))
    membership = query.contains(points, path=("reflected",))
    np.testing.assert_array_equal(membership.inside, (False, True, False))
    assert not np.any(np.asarray(membership.unresolved))
    closest = query.closest_point(points[[0, 2]], path=("reflected",))
    expected = translation + np.asarray(((scale, 0.0, 0.0), (0.0, 0.0, 1.0)))
    np.testing.assert_allclose(
        closest.points, expected, atol=_OCCURRENCE_WORLD_CLOSEST_POINT_ATOL, rtol=0.0
    )
    np.testing.assert_allclose(
        closest.distances,
        (3.0 - scale, 1.0),
        atol=_OCCURRENCE_WORLD_CLOSEST_DISTANCE_ATOL,
        rtol=0.0,
    )
    assert not np.any(np.asarray(closest.unresolved))
    np.testing.assert_allclose(
        closest.normals,
        ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        atol=_OCCURRENCE_WORLD_NORMAL_ATOL,
        rtol=0.0,
    )


def test_reflected_live_world_radius_has_independent_jvp_vjp() -> None:
    translation = np.asarray((4.0, -3.0, 2.0))
    model = _query_occurrence_model("sphere", np.diag((-1.0, 1.0, 1.0)), translation)
    geometry = phx.geometry.FixedTopologyBRepSource(model).compile(coordinate_unit=METER)
    radius_index = next(
        index
        for index, spec in enumerate(geometry.schema.specs)
        if spec.parameter_id.name == "radius"
    )
    point = jnp.asarray(translation + np.asarray((2.0, 0.3, 0.2)))

    @jax.jit
    def field(radius: Array) -> Array:
        state = geometry.state.replace_at(radius_index, radius)
        return geometry.kernel.boundary_field(state, point)

    radius = jnp.asarray(1.25, dtype=jnp.float64)
    value, tangent = jax.jvp(field, (radius,), (jnp.asarray(2.0),))
    _, transpose = jax.vjp(field, radius)
    np.testing.assert_allclose(
        value,
        np.linalg.norm((2.0, 0.3, 0.2)) - 1.25,
        atol=_REFLECTED_WORLD_DISTANCE_VALUE_ATOL,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        tangent, -2.0, atol=_REFLECTED_WORLD_RADIUS_TANGENT_ATOL, rtol=0.0
    )
    np.testing.assert_allclose(
        transpose(jnp.ones_like(value))[0],
        -1.0,
        atol=_REFLECTED_WORLD_RADIUS_COTANGENT_ATOL,
        rtol=0.0,
    )


def test_native_planar_face_projection_respects_trim_holes() -> None:
    plate = phx.geometry.brep_planar_face(
        _plate_profile(), coordinate_contract=_CONTRACT, tessellation=_COARSE
    )
    embedding = phx.geometry.PlanarEmbedding((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
    projection = phx.geometry.prepare_brep_projection(plate, embedding=embedding)
    assert isinstance(projection, phx.geometry.NativeBRepProjection)
    hole = int(np.flatnonzero(np.asarray(projection.edge_closed))[0])
    edge = projection.project(
        np.asarray([[0.5, 0.5]]), np.asarray([1]), np.asarray([hole])
    )
    face = projection.project(
        np.asarray([[0.5, 0.5], [1.5, 0.0]]), np.asarray([2, 2]), np.asarray([0, 0])
    )

    assert plate.topology.num_solids == 0
    np.testing.assert_allclose(edge.points, [[2**-0.5, 2**-0.5]])
    np.testing.assert_allclose(face.points, [[2**-0.5, 2**-0.5], [1.5, 0.0]])
    np.testing.assert_allclose(
        face.residuals, [1.0 - 2**-0.5, 0.0], atol=_PLANAR_TRIM_PROJECTION_RESIDUAL_ATOL
    )
    assert np.all(np.isnan(np.asarray(face.normals)))


def test_native_brep_source_answers_region_queries_exactly() -> None:
    geometry = phx.geometry.BRepSource(_model("sphere")).compile(coordinate_unit=METER)
    closest = geometry.closest_point(jnp.asarray([[0.0, 0.0, 0.5], [2.0, 0.0, 0.0]]))

    assert float(geometry.measure) == pytest.approx(
        4 * _PI / 3, rel=_SPHERE_SOURCE_VOLUME_RTOL
    )
    assert float(geometry.boundary_measure) == pytest.approx(
        4 * _PI, rel=_SPHERE_SOURCE_SURFACE_AREA_RTOL
    )
    assert np.asarray(
        geometry.contains(jnp.asarray([[0.0, 0.0, 0.5], [2.0, 0.0, 0.0]]))
    ).tolist() == [True, False]
    np.testing.assert_allclose(closest.normal_coordinate, [-0.5, 1.0])
    np.testing.assert_allclose(
        closest.closest_point,
        [[0, 0, 1], [1, 0, 0]],
        atol=_SPHERE_SOURCE_CLOSEST_POINT_ATOL,
    )
    assert closest.exact_to_physical
    assert np.asarray(closest.unique).tolist() == [True, True]
    bounds = np.asarray(geometry.bounds)
    assert np.all(bounds[0] <= -1.0) and np.all(bounds[1] >= 1.0)


def test_native_brep_boundary_field_compiles_with_exact_source_point_jvp() -> None:
    geometry = phx.geometry.BRepSource(_model("sphere")).compile(coordinate_unit=METER)
    points = jnp.asarray(
        ((2.0, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 1.0)), dtype=jnp.float64
    )
    directions = jnp.asarray(
        ((3.0, 4.0, 0.0), (0.0, -2.0, 5.0), (-4.0, 1.0, 2.0)), dtype=jnp.float64
    )

    @jax.jit
    def evaluate(point: Array, direction: Array) -> tuple[Array, Array]:
        return jax.jvp(geometry.boundary_field, (point,), (direction,))

    values, tangents = evaluate(points, directions)
    # The source unit sphere determines the inside sign and radial normal,
    # including the point exactly on its authored boundary.
    np.testing.assert_allclose(
        values, (1.0, -0.5, 0.0), rtol=0.0, atol=_SPHERE_BOUNDARY_FIELD_VALUE_ATOL
    )
    np.testing.assert_allclose(
        tangents, (3.0, -2.0, 2.0), rtol=0.0, atol=_SPHERE_BOUNDARY_FIELD_TANGENT_ATOL
    )


def test_fixed_topology_native_measure_is_exact_and_differentiable() -> None:
    geometry = phx.geometry.FixedTopologyBRepSource(_model("sphere")).compile(
        coordinate_unit=METER
    )
    radius_index = next(
        index
        for index, spec in enumerate(geometry.schema.specs)
        if spec.parameter_id.name == "radius"
    )

    def volume(radius: Any) -> Any:
        return geometry.kernel.measure(geometry.state.replace_at(radius_index, radius))

    assert float(volume(jnp.asarray(1.0))) == pytest.approx(
        4 * _PI / 3, rel=_DYNAMIC_SPHERE_VOLUME_RTOL
    )
    assert float(jax.grad(volume)(jnp.asarray(1.0))) == pytest.approx(
        4 * _PI, rel=_DYNAMIC_SPHERE_VOLUME_DERIVATIVE_RTOL
    )


def _carriers() -> dict[str, tuple[Any, Any]]:
    geometry = phx.geometry
    arc = geometry.CircleCurve((0.5, 0.0, 0.0), (1, 0, 0), (0, 0, 1), 0.4)
    rational = geometry.BSplineCurve(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 0.5],
            [2.0, -1.0, 1.0],
            [3.0, 1.0, 0.0],
            [4.0, 0.0, 1.0],
        ],
        [1.0, 0.4, 2.5, 0.7, 1.0],
        [0.0, 0.0, 0.0, 0.3, 0.6, 1.0, 1.0, 1.0],
        2,
    )
    return {
        "ellipse": (
            geometry.EllipseCurve((1, 2), (0.6, 0.8), (-0.8, 0.6), 2.0, 0.5),
            (0.3, 4.1),
        ),
        "rational_bspline": (rational, (0.1, 0.8)),
        "extrusion": (
            geometry.ExtrusionSurface(rational, (0.2, 0.1, 1.0)),
            ((0.2, -1.0), (0.9, 2.0)),
        ),
        "revolution": (
            geometry.RevolutionSurface(arc, (0, 0, 0), (0, 0, 1)),
            ((0.3, 0.5), (4.0, 2.5)),
        ),
        "ruled": (
            geometry.RuledSurface(rational, arc),
            ((0.0, 0.0), (1.0, 1.0)),
        ),
        "torus": (
            geometry.TorusPatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 2.0, 0.5),
            ((0.4, 1.0), (2.0, 5.5)),
        ),
    }


@pytest.mark.parametrize("name", sorted(_carriers()), ids=sorted(_carriers()))
def test_carrier_bounding_boxes_enclose_dense_samples(name: str) -> None:
    carrier, domain = _carriers()[name]
    if isinstance(carrier, phx.geometry.AbstractCurve):
        box = carrier.bounding_box(*domain)
        samples = np.asarray(carrier.evaluate(jnp.linspace(domain[0], domain[1], 2001)))
    else:
        box = carrier.bounding_box(np.asarray(domain))
        grid = np.stack(
            np.meshgrid(
                np.linspace(domain[0][0], domain[1][0], 97),
                np.linspace(domain[0][1], domain[1][1], 97),
                indexing="ij",
            ),
            axis=-1,
        ).reshape(-1, 2)
        samples = np.asarray(carrier.evaluate(jnp.asarray(grid)))

    assert np.all(samples >= box[0]) and np.all(samples <= box[1])


def test_rational_bezier_pieces_reproduce_the_spline_exactly() -> None:
    curve, _ = _carriers()["rational_bspline"]
    pieces = curve.bezier_pieces()

    assert [piece.parameter_bounds for piece in pieces] == [
        ((0.0, 0.3),),
        ((0.3, 0.6),),
        ((0.6, 1.0),),
    ]
    for piece in pieces:
        ((lower, upper),) = piece.parameter_bounds
        local = np.linspace(0.0, 1.0, 7)
        bernstein = np.stack(
            [(1 - local) ** 2, 2 * local * (1 - local), local**2], axis=1
        )
        homogeneous = bernstein @ piece.homogeneous_controls
        expected = np.asarray(
            curve.evaluate(jnp.asarray(lower + local * (upper - lower)))
        )
        np.testing.assert_allclose(
            homogeneous[:, :3] / homogeneous[:, 3:],
            expected,
            atol=_RATIONAL_BEZIER_POINT_ATOL,
        )
    with pytest.raises(ValueError, match="multiplicities"):
        phx.geometry.BSplineCurve(
            np.zeros((6, 2)),
            np.ones(6),
            np.asarray([0, 0, 0, 0.3, 0.3, 0.3, 1, 1, 1], dtype=np.float64),
            2,
        )


def test_native_construction_and_queries_never_import_occt() -> None:
    script = textwrap.dedent(
        """
        import sys
        import numpy as np
        import phydrax as phx

        contract = phx.SpatialCoordinateContract.si()
        policy = phx.geometry.BRepTessellationPolicy(
            linear_deflection=0.1, angular_deflection=0.5
        )
        model = phx.geometry.brep_torus(
            2.0, 0.5, coordinate_contract=contract, tessellation=policy
        )
        query = phx.geometry.prepare_brep_query(model)
        inside = query.contains(np.asarray([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
        projection = phx.geometry.prepare_brep_projection(model)
        projection.project(np.asarray([[3.0, 0.0, 0.0]]), np.asarray([2]), np.asarray([0]))
        assert np.asarray(inside.inside).tolist() == [True, False]
        loaded = sorted(name for name in sys.modules if name.split(".")[0] in ("OCP", "build123d"))
        print(",".join(loaded))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == ""


def test_large_translated_source_queries_preserve_distance_and_boundary_proof() -> None:
    upper = float(2**30)
    source = phx.geometry.brep_box(
        (upper - 1.0, 0.0, 0.0),
        (upper, 1.0, 1.0),
        coordinate_contract=_CONTRACT,
        tessellation=_COARSE,
    )
    query = phx.geometry.prepare_brep_query(source)
    points = np.asarray(
        [
            [upper, 0.25, 0.25],
            [upper - 0.25, 0.3, 0.4],
            [upper + 0.25, 0.2, 0.3],
            [upper - 0.5, 0.6, 0.7],
        ],
        dtype=np.float64,
    )
    closest = query.closest_point(points)
    expected = np.asarray([0.0, 0.25, 0.25, 1.0 - points[3, 2]], dtype=np.float64)
    np.testing.assert_allclose(
        closest.distances, expected, atol=_TRANSLATED_SOURCE_DISTANCE_ATOL, rtol=0.0
    )
    assert np.all(np.asarray(closest.distance_lower_bounds) <= expected)
    assert np.all(np.asarray(closest.distances) >= expected)
    assert np.all(np.asarray(closest.status) == _STATUS.UNIQUE)
    np.testing.assert_array_equal(np.asarray(closest.points)[0], points[0])
    membership = query.contains(points)
    np.testing.assert_array_equal(membership.inside, [True, True, False, True])
    np.testing.assert_array_equal(
        membership.status,
        [_STATUS.AMBIGUOUS, _STATUS.UNIQUE, _STATUS.UNIQUE, _STATUS.UNIQUE],
    )
    assert np.all(np.asarray(membership.distance_lower_bounds) <= expected)
    assert np.all(np.asarray(membership.distance_upper_bounds) >= expected)


@pytest.mark.parametrize(
    ("operations", "points", "scratch", "resource"),
    [
        (1, None, None, "operations"),
        (10000, 1, None, "points"),
        (10000, None, 1, "scratch_bytes"),
    ],
)
def test_native_membership_refuses_before_consuming_shared_allowance(
    operations: int,
    points: int | None,
    scratch: int | None,
    resource: str,
) -> None:
    from phydrax.geometry.brep._query_types import BRepQueryBudget, BRepQueryResourceError

    query = _query("box")
    budget = BRepQueryBudget(operations, points, scratch)
    query_points = np.asarray([[0.1, 0.2, 0.3], [2.0, 1.0, 1.0]], dtype=np.float64)
    with pytest.raises(BRepQueryResourceError) as refusal:
        query.contains(query_points, budget=budget)
    assert refusal.value.resource == resource
    assert (budget.operations, budget.points, budget.subdivisions) == (0, 0, 0)
    accepted = BRepQueryBudget(10000)
    membership = query.contains(query_points, budget=accepted)
    np.testing.assert_array_equal(membership.inside, [True, False])
    assert np.all(np.asarray(membership.status) == _STATUS.UNIQUE)
    assert accepted.points == 2
    before = accepted.operations
    query.contains(query_points, budget=accepted)
    assert accepted.points == 4
    assert accepted.operations == 2 * before


def test_compiled_spline_extrusion_tracks_dynamic_knot_endpoints() -> None:
    def evaluate_surface(surface: ExtrusionSurface, parameters: Array) -> Array:
        return surface.evaluate(parameters)

    def zero_tangent(leaf: object) -> None:
        return None

    controls = jnp.asarray(
        ((0.0, 0.0, 0.0), (0.5, 1.0, 0.0), (1.0, 0.0, 0.0)), dtype=jnp.float64
    )
    curve = BSplineCurve(
        controls,
        jnp.ones(3, dtype=jnp.float64),
        jnp.asarray((0.0, 0.0, 0.0, 1.0, 1.0, 1.0), dtype=jnp.float64),
        2,
    )
    surface = ExtrusionSurface(curve, jnp.asarray((0.0, 0.0, 1.0), dtype=jnp.float64))
    parameters = jnp.asarray(
        ((-0.5, 0.2), (0.25, 0.3), (0.75, 0.4), (1.5, 0.5)), dtype=jnp.float64
    )
    compiled = eqx.filter_jit(evaluate_surface)
    t = np.asarray((0.0, 0.25, 0.75, 1.0), dtype=np.float64)
    expected = np.stack((t, 2 * t * (1 - t), np.asarray(parameters)[:, 1]), axis=1)
    np.testing.assert_allclose(
        compiled(surface, parameters),
        expected,
        rtol=_DYNAMIC_KNOT_EXTRUSION_POINT_RTOL,
        atol=_DYNAMIC_KNOT_EXTRUSION_POINT_ATOL,
    )
    tangent_surface = jax.tree_util.tree_map(zero_tangent, surface)
    _, tangent = eqx.filter_jvp(
        evaluate_surface,
        (surface, parameters),
        (tangent_surface, jnp.ones_like(parameters)),
    )
    np.testing.assert_allclose(
        tangent,
        ((0.0, 0.0, 1.0), (1.0, 1.0, 1.0), (1.0, -1.0, 1.0), (0.0, 0.0, 1.0)),
        rtol=_DYNAMIC_KNOT_EXTRUSION_TANGENT_RTOL,
        atol=_DYNAMIC_KNOT_EXTRUSION_TANGENT_ATOL,
    )
    shifted = ExtrusionSurface(
        BSplineCurve(
            controls,
            jnp.ones(3, dtype=jnp.float64),
            jnp.asarray((0.2, 0.2, 0.2, 1.2, 1.2, 1.2), dtype=jnp.float64),
            2,
        ),
        surface.direction,
    )
    t = np.clip(np.asarray(parameters)[:, 0] - 0.2, 0.0, 1.0)
    expected = np.stack((t, 2 * t * (1 - t), np.asarray(parameters)[:, 1]), axis=1)
    np.testing.assert_allclose(
        compiled(shifted, parameters),
        expected,
        rtol=_DYNAMIC_KNOT_EXTRUSION_POINT_RTOL,
        atol=_DYNAMIC_KNOT_EXTRUSION_POINT_ATOL,
    )


@pytest.mark.parametrize(
    ("operations", "points", "subdivisions"),
    ((-1, 0, 0), (0, -1, 0), (0, 0, -1)),
    ids=("operations", "points", "subdivisions"),
)
def test_query_budget_rejects_negative_consumption_without_credit(
    operations: int,
    points: int,
    subdivisions: int,
) -> None:
    budget = BRepQueryBudget(10, maximum_points=4, maximum_scratch_bytes=64)
    budget.consume(4, points=2, subdivisions=1)
    with pytest.raises(ValueError, match="(operations|points|subdivisions).*nonnegative"):
        budget.consume(operations, points=points, subdivisions=subdivisions)
    assert (budget.operations, budget.points, budget.subdivisions) == (4, 2, 1)
    budget.consume(6, points=2)
    with pytest.raises(BRepQueryResourceError) as refusal:
        budget.consume(1)
    assert (refusal.value.resource, refusal.value.requested, refusal.value.remaining) == (
        "operations",
        1,
        0,
    )
    assert (budget.operations, budget.points, budget.subdivisions) == (10, 4, 1)


def test_query_budget_rejects_negative_scratch_admission_without_consuming() -> None:
    budget = BRepQueryBudget(10, maximum_points=4, maximum_scratch_bytes=64)
    budget.consume(4, points=2, subdivisions=1)
    with pytest.raises(ValueError, match="scratch_bytes.*nonnegative"):
        budget.admit(0, 0, -1)
    assert (budget.operations, budget.points, budget.subdivisions) == (4, 2, 1)
    budget.admit(6, 2, 64)
    with pytest.raises(BRepQueryResourceError) as refusal:
        budget.admit(0, 0, 65)
    assert (refusal.value.resource, refusal.value.requested, refusal.value.remaining) == (
        "scratch_bytes",
        65,
        64,
    )
    assert (budget.operations, budget.points, budget.subdivisions) == (4, 2, 1)


@eqx.filter_jit
def _extrusion_jet_samples(
    surface: ExtrusionSurface, parameters: Array, order: int
) -> Array:
    derivative = jax.jacfwd(surface.evaluate)
    if order == 2:
        derivative = jax.jacfwd(derivative)
    return jax.vmap(derivative)(parameters)


@pytest.mark.parametrize(
    "middle_weight", (0.75, 1.0), ids=("rational-profile", "polynomial-profile")
)
def test_extrusion_source_jets_bound_whole_profile_and_fixed_rulings(
    middle_weight: float,
) -> None:
    curve = BSplineCurve.bezier(
        ((0.0, 0.0, 0.0), (0.0, 0.5, 0.2), (0.0, 1.0, 0.0)),
        (1.0, middle_weight, 1.0),
    )
    surface = ExtrusionSurface(curve, (1.0, 0.0, 0.0))
    boxes = np.asarray((((0.0, 0.0), (1.0, 1.0)), ((0.25, 0.0), (0.25, 1.0))))
    # Knot clamping has a separate nonsmooth AD convention exactly at an
    # endpoint. These source jets enclose the mathematical profile derivative,
    # so compare the source numerical map strictly inside its knot domain.
    parameters = jnp.asarray(
        np.column_stack((np.linspace(1e-4, 1.0 - 1e-4, 65), np.linspace(0.0, 1.0, 65)))
    )
    for order in (1, 2):
        actual = np.asarray(_extrusion_jet_samples(surface, parameters, order))
        ruling = np.asarray(
            _extrusion_jet_samples(surface, jnp.asarray(((0.25, 0.5),)), order)
        )[0]
        lower, upper = surface.derivative_bounds_batch(boxes, order=order)
        assert np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))
        rounding = 64 * np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(actual))))
        assert np.all(actual >= lower[0] - rounding)
        assert np.all(actual <= upper[0] + rounding)
        assert np.all(ruling >= lower[1] - rounding)
        assert np.all(ruling <= upper[1] + rounding)
        if order == 1:
            np.testing.assert_array_equal(lower[:, :, 1], ((1.0, 0.0, 0.0),) * 2)
            np.testing.assert_array_equal(upper[:, :, 1], lower[:, :, 1])
        else:
            np.testing.assert_array_equal(lower[:, :, 1, :], np.zeros((2, 3, 2)))
            np.testing.assert_array_equal(upper[:, :, :, 1], np.zeros((2, 3, 2)))


@pytest.mark.parametrize(
    "interval",
    ((0.8, 1.0), (1.0, 1.2), (0.8, 1.2)),
    ids=("ends-at-node", "starts-at-node", "crosses-node"),
)
def test_native_branch_chord_bound_remains_continuous_at_c0_chart_nodes(
    interval: tuple[float, float],
) -> None:
    from phydrax.geometry._domain_numerics import physical_curve_chord_bound
    from phydrax.geometry.brep._coordinate_request import coordinate_source_request
    from phydrax.geometry.brep._intersection_curve import IntersectionCurve
    from tests._support.cad_models import native_intersection_cap

    model = native_intersection_cap(realize=False)
    assert model.geometry is not None
    curve = model.geometry.curves[0]
    assert isinstance(curve, IntersectionCurve)
    first, last = interval
    with coordinate_source_request((curve, first, last)) as budget:
        bound = physical_curve_chord_bound(curve, first, last, budget=budget)
    assert 0.0 < bound < 0.01
    points = np.asarray(curve.evaluate(jnp.linspace(first, last, 33)).point)
    chord = points[-1] - points[0]
    along = np.clip((points - points[0]) @ chord / (chord @ chord), 0.0, 1.0)
    distances = np.linalg.norm(points - points[0] - along[:, None] * chord, axis=1)
    assert float(np.max(distances)) <= bound


@pytest.mark.parametrize(
    "interval",
    ((0.5, 1.0), (1.0, 1.5), (0.5, 1.5)),
    ids=("ends-at-knot", "starts-at-knot", "crosses-knot"),
)
def test_spline_chord_bound_remains_continuous_at_c0_knots(
    interval: tuple[float, float],
) -> None:
    from phydrax.geometry._domain_numerics import physical_curve_chord_bound
    from phydrax.geometry.brep._coordinate_request import coordinate_source_request

    # Cubic with a C0 (multiplicity = degree) knot at 1: the shape of the
    # export fit's per-chart segments. Closed arcs touching the knot use the
    # per-chart knot admission with finite second-jet bounds.
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (0.4, 0.6, 0.0),
            (0.8, 0.9, 0.1),
            (1.0, 1.0, 0.2),
            (1.5, 0.7, 0.4),
            (1.8, 0.2, 0.3),
            (2.0, 0.0, 0.0),
        )
    )
    knots = np.asarray((0.0,) * 4 + (1.0,) * 3 + (2.0,) * 4)
    curve = BSplineCurve(
        points, np.asarray((1.0, 0.9, 1.1, 1.0, 0.8, 1.2, 1.0)), knots, 3
    )
    first, last = interval
    assert not curve.is_c1_on(first, last)
    with coordinate_source_request((curve, first, last)) as budget:
        bound = physical_curve_chord_bound(curve, first, last, budget=budget)
    assert 0.0 < bound < np.inf
    samples = np.asarray(curve.evaluate(jnp.linspace(first, last, 129)))
    chord = samples[-1] - samples[0]
    along = np.clip((samples - samples[0]) @ chord / (chord @ chord), 0.0, 1.0)
    distances = np.linalg.norm(samples - samples[0] - along[:, None] * chord, axis=1)
    assert float(np.max(distances)) <= bound
    assert curve.is_c1_on(0.2, 0.8)


@pytest.mark.parametrize(
    "interval",
    ((0.5, 1.0), (1.0, 1.5), (1.0, 1.0)),
    ids=("ends-at-knot", "starts-at-knot", "point-at-knot"),
)
def test_rational_first_jet_bounds_include_closed_knot_endpoint(
    interval: tuple[float, float],
) -> None:
    curve = BSplineCurve(
        [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]],
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 1.0, 2.0, 2.0],
        1,
    )
    lower, upper = curve.derivative_bounds(*interval, order=1)
    # A rational linear segment's endpoint derivative is the control leg
    # times the ratio of its endpoint weights. Both source spans own this
    # closed knot, while ordinary evaluation chooses the right-hand span.
    for exact in (np.asarray([0.5, 0.5]), np.asarray([0.5, -0.5])):
        assert np.all(lower <= exact)
        assert np.all(exact <= upper)
    actual = np.asarray(jax.jacfwd(curve.evaluate)(jnp.asarray(1.0, dtype=jnp.float64)))
    np.testing.assert_allclose(
        actual, [0.5, -0.5], rtol=0.0, atol=_RATIONAL_ENDPOINT_FIRST_JET_ATOL
    )
    assert np.all(lower <= actual) and np.all(actual <= upper)


def test_repeated_rational_source_bounds_preserve_closed_knots_under_one_owner() -> None:
    from phydrax._resource_ticket import HostBatchTicket
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget

    curve = BSplineCurve(
        [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]],
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 1.0, 2.0, 2.0],
        1,
    )
    intervals = ((0.0, 1.0), (1.0, 2.0), (0.5, 1.5), (1.0, 1.0))
    expected = [
        (curve.bounding_box(*interval), curve.derivative_bounds(*interval))
        for interval in intervals
    ]
    ticket = HostBatchTicket(1_000_000, bytes_upper=16_000_000)
    owner = CoordinateEnclosureBudget(
        ticket, maximum_work_units=1_000_000, maximum_memory_bytes=16_000_000
    )
    with ticket, owner.activate():
        for _ in range(3):
            for interval, (box, jet) in zip(intervals, expected, strict=True):
                with owner.temporary_scope():
                    np.testing.assert_array_equal(
                        curve.bounding_box(*interval, budget=owner), box
                    )
                    actual = curve.derivative_bounds(*interval, budget=owner)
                    np.testing.assert_array_equal(actual[0], jet[0])
                    np.testing.assert_array_equal(actual[1], jet[1])


def test_repeated_unclamped_periodic_source_bounds_keep_original_knot_domain() -> None:
    from phydrax._resource_ticket import HostBatchTicket
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget

    # Expanded unclamped periodic knots: preparation must not clamp or
    # normalize the source to manufacture a reusable span bank.
    curve = BSplineCurve(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0]],
        [1.0, 0.8, 1.0, 0.8, 1.0, 0.8],
        [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        2,
    )
    intervals = ((0.0, 4.0), (0.0, 0.25), (1.0, 2.0), (3.75, 4.0))
    expected = [curve.bounding_box(*interval) for interval in intervals]
    ticket = HostBatchTicket(2_000_000, bytes_upper=16_000_000)
    owner = CoordinateEnclosureBudget(
        ticket, maximum_work_units=2_000_000, maximum_memory_bytes=16_000_000
    )
    with ticket, owner.activate():
        for _ in range(3):
            for interval, box in zip(intervals, expected, strict=True):
                with owner.temporary_scope():
                    actual = curve.bounding_box(*interval, budget=owner)
                    np.testing.assert_array_equal(actual, box)
                    samples = np.asarray(curve.evaluate(jnp.linspace(*interval, 33)))
                    assert np.all(samples >= actual[0]) and np.all(samples <= actual[1])


@pytest.mark.parametrize(
    "order", (1, 2), ids=("first-quotient-jet", "mixed-second-quotient-jet")
)
def test_rational_tensor_source_jets_enclose_whole_positive_weight_span(
    order: int,
) -> None:
    from phydrax._resource_ticket import HostBatchTicket
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget

    surface = phx.geometry.BSplineSurfacePatch(
        [
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.2]],
            [[0.6, 0.0, 0.3], [0.6, 1.0, -0.1]],
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.5]],
        ],
        [[1.0, 1.2], [0.6, 0.8], [1.1, 1.3]],
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        2,
        1,
    )
    box = np.asarray(((0.0, 0.0), (1.0, 1.0)))
    lower, upper = surface.derivative_bounds(box, order=order)
    assert np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))
    parameters = jnp.asarray(
        [(u, v) for u in (0.2, 0.5, 0.8) for v in (0.15, 0.55, 0.85)]
    )
    derivative = surface.evaluate
    for _ in range(order):
        derivative = jax.jacfwd(derivative)
    actual = np.asarray(jax.vmap(derivative)(parameters))
    assert np.all(actual >= lower) and np.all(actual <= upper)
    changed = eqx.tree_at(
        lambda value: value.weights, surface, surface.weights.at[1, 0].set(0.4)
    )
    expected = changed.derivative_bounds(box, order=order)
    ticket = HostBatchTicket(2_000_000, bytes_upper=16_000_000)
    owner = CoordinateEnclosureBudget(
        ticket, maximum_work_units=2_000_000, maximum_memory_bytes=16_000_000
    )
    with ticket, owner.activate():
        for source, bounds in (
            (surface, (lower, upper)),
            (changed, expected),
            (surface, (lower, upper)),
        ):
            with owner.temporary_scope():
                enclosed = source.derivative_bounds(box, order=order, budget=owner)
                np.testing.assert_array_equal(enclosed[0], bounds[0])
                np.testing.assert_array_equal(enclosed[1], bounds[1])


@pytest.mark.parametrize("axis", (0, 1), ids=("constant-seam-u", "constant-boundary-v"))
def test_spline_pcurve_constant_coordinate_keeps_actual_boundary_source_normal(
    axis: int,
) -> None:
    from phydrax.geometry.brep._patches import surface_differential

    end = 6.28318530718
    if axis == 0:
        controls = [[end, 0.0], [end, 1.0], [end, 2.0]]
        knots = [0.0, 0.0, 0.0, 2.0, 2.0, 2.0]
        last, constant = 2.0, end
    else:
        values = [
            0.0,
            1.047197551197,
            2.094395102393,
            3.14159265359,
            4.188790204786,
            5.235987755983,
            end,
        ]
        controls = [[value, 2.0] for value in values]
        knots = [
            0.0,
            0.0,
            0.0,
            2.094395102393,
            2.094395102393,
            4.188790204786,
            4.188790204786,
            end,
            end,
            end,
        ]
        last, constant = end, 2.0
    curve = BSplineCurve(controls, np.ones(len(controls)), knots, 2)
    parameters = jnp.linspace(0.0, last, 128)
    charts = np.asarray(curve.evaluate(parameters))
    np.testing.assert_array_equal(charts[:, axis], np.full(128, constant))
    tangent = np.asarray(jax.vmap(jax.jacfwd(curve.evaluate))(parameters))
    np.testing.assert_array_equal(tangent[:, axis], np.zeros(128))
    surface = phx.geometry.BSplineSurfacePatch(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]],
            [[1.0, 0.5, 0.0], [1.0, 0.5, 2.0]],
            [[2.0, 0.0, 0.0], [2.0, 0.0, 2.0]],
        ],
        np.ones((3, 2)),
        [0.0, 0.0, 0.0, end, end, end],
        [0.0, 0.0, 2.0, 2.0],
        2,
        1,
    )
    differential = np.asarray(surface_differential(surface, jnp.asarray(charts)))
    cross = np.cross(differential[..., 0], differential[..., 1])
    assert np.all(np.linalg.norm(cross, axis=1) > 0.1)


def test_exact_restriction_bank_retains_cold_arrays_and_scoped_storage() -> None:
    from phydrax._resource_ticket import HostBatchTicket
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
    from phydrax.geometry.brep import _patches as bounds

    source = phx.geometry.BSplineSurfacePatch(
        [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.2]], [[1.0, 0.0, 0.3], [1.0, 1.0, 0.5]]],
        [[1.0, 0.9], [1.1, 1.0]],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        1,
        1,
    )
    piece = source.bezier_pieces()[0]
    ticket = HostBatchTicket(1_000_000, bytes_upper=16_000_000)
    owner = CoordinateEnclosureBudget(
        ticket, maximum_work_units=1_000_000, maximum_memory_bytes=16_000_000
    )
    with (
        ticket,
        owner.activate(),
        bounds.source_bernstein_restriction_scope(budget=owner) as bank,
    ):
        assert bank is not None
        key = (bank.prefix(piece, 1), (0, 0), b"original-exact-test-interval")
        cold = bounds.restrict_bernstein_bounds(
            piece.homogeneous_lower, piece.homogeneous_upper, 0.2, 0.8, 0
        )
        with owner.temporary_scope():
            retained = bank.restrict(
                key, piece.homogeneous_lower, piece.homogeneous_upper, 0.2, 0.8, 0
            )
        live = owner.temporary_bytes_upper
        assert live > 0
        before = owner.work_units
        with owner.temporary_scope():
            reused = bank.restrict(
                key, piece.homogeneous_lower, piece.homogeneous_upper, 0.2, 0.8, 0
            )
            assert reused[0] is retained[0] and reused[1] is retained[1]
            for actual, expected in zip(reused, cold, strict=True):
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(
                    np.min(actual, axis=(0, 1)), np.min(expected, axis=(0, 1))
                )
                np.testing.assert_array_equal(
                    np.max(actual, axis=(0, 1)), np.max(expected, axis=(0, 1))
                )
                assert not actual.flags.writeable
        assert owner.work_units > before
        assert owner.temporary_bytes_upper == live
        del actual, expected, retained, reused, cold
    assert owner.temporary_bytes_upper == 0


@pytest.mark.parametrize("resource", ("work", "storage"))
def test_exact_restriction_bank_refuses_before_retaining_unadmitted_result(
    resource: str,
) -> None:
    from phydrax._resource_ticket import HostBatchTicket
    from phydrax.discretization._coordinate_budget import (
        CoordinateEnclosureBudget,
        CoordinateEnclosureResourceError,
    )
    from phydrax.geometry.brep import _patches as bounds

    work, storage = (0, 1_000_000) if resource == "work" else (1_000_000, 0)
    ticket = HostBatchTicket(work, bytes_upper=storage)
    owner = CoordinateEnclosureBudget(
        ticket, maximum_work_units=work, maximum_memory_bytes=storage
    )
    controls = np.ones((2, 2, 4))
    with (
        ticket,
        owner.activate(),
        bounds.source_bernstein_restriction_scope(budget=owner) as bank,
    ):
        assert bank is not None
        with pytest.raises(CoordinateEnclosureResourceError):
            bank.restrict(
                ("actual-constant-coefficients", controls.tobytes()),
                controls,
                controls,
                0.2,
                0.8,
                0,
            )
        assert not bank.entries


def test_restriction_prefix_storage_refusal_is_atomic() -> None:
    from phydrax._resource_ticket import HostBatchTicket
    from phydrax.discretization._coordinate_budget import (
        CoordinateEnclosureBudget,
        CoordinateEnclosureResourceError,
    )
    from phydrax.geometry.brep import _patches as bounds

    source = phx.geometry.BSplineSurfacePatch(
        [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]],
        np.ones((2, 2)),
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        1,
        1,
    )
    piece = source.bezier_pieces()[0]
    ticket = HostBatchTicket(1_000_000, bytes_upper=0)
    owner = CoordinateEnclosureBudget(
        ticket, maximum_work_units=1_000_000, maximum_memory_bytes=0
    )
    with (
        ticket,
        owner.activate(),
        bounds.source_bernstein_restriction_scope(budget=owner) as bank,
    ):
        assert bank is not None
        with pytest.raises(CoordinateEnclosureResourceError):
            bank.prefix(piece, 2)
        assert owner.work_units == 0 and owner.temporary_bytes_upper == 0
        assert not bank.prefixes and not bank.entries


def test_tensor_derivative_preparation_reuses_only_exact_request_content() -> None:
    from phydrax._resource_ticket import HostBatchTicket
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget

    source = phx.geometry.BSplineSurfacePatch(
        [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.2]], [[1.0, 0.0, 0.3], [1.0, 1.0, 0.5]]],
        [[1.0, 0.9], [1.1, 1.0]],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        1,
        1,
    )
    boxes = np.asarray([[[0.2, 0.25], [0.8, 0.75]]], dtype=np.float64)
    ticket = HostBatchTicket(640_000, bytes_upper=81_920_000)
    shared = CoordinateEnclosureBudget(
        ticket,
        maximum_work_units=640_000,
        maximum_memory_bytes=81_920_000,
    )
    with (
        ticket,
        shared.activate(),
        shared.temporary_scope(),
    ):
        first = source.derivative_bounds_batch(boxes, order=1, budget=shared)
        second = source.derivative_bounds_batch(boxes, order=2, budget=shared)
        neighboring = boxes.copy()
        neighboring[0, 1, 1] = np.nextafter(neighboring[0, 1, 1], np.inf)
        third = source.derivative_bounds_batch(neighboring, order=2, budget=shared)
        changed_points = np.asarray(source.control_points).copy()
        changed_points[1, 1, 2] += 0.1
        changed_source = phx.geometry.BSplineSurfacePatch(
            changed_points, source.weights, source.u_knots, source.v_knots, 1, 1
        )
        changed = changed_source.derivative_bounds_batch(boxes, order=2, budget=shared)
        assert not np.array_equal(changed[0], second[0])
    cold_work = 0
    for patch, order, queries, result in (
        (source, 1, boxes, first),
        (source, 2, boxes, second),
        (source, 2, neighboring, third),
        (changed_source, 2, boxes, changed),
    ):
        cold_ticket = HostBatchTicket(640_000, bytes_upper=81_920_000)
        cold = CoordinateEnclosureBudget(
            cold_ticket,
            maximum_work_units=640_000,
            maximum_memory_bytes=81_920_000,
        )
        with cold_ticket, cold.activate():
            expected = patch.derivative_bounds_batch(queries, order=order, budget=cold)
        for value, reference in zip(result, expected, strict=True):
            np.testing.assert_array_equal(value, reference)
        cold_work += cold.work_units
    assert 0 < shared.work_units < cold_work < 640_000
    assert shared.temporary_bytes_upper == 0


def test_tensor_box_validation_reuses_only_exact_source_ordered_boxes_and_order() -> None:
    from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
    from phydrax._resource_ticket import HostBatchTicket
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
    from phydrax.geometry.brep._patches import (
        _prepared_tensor_derivative_boxes,
        _validate_tensor_derivative_boxes,
        BSplineSurfacePatch,
    )

    source = BSplineSurfacePatch(
        np.asarray(
            (
                ((0.0, 0.0, 0.0), (0.0, 1.0, 0.2)),
                ((0.5, 0.0, 0.1), (0.5, 1.0, 0.3)),
                ((1.0, 0.0, 0.3), (1.0, 1.0, 0.5)),
            ),
            dtype=np.float64,
        ),
        np.asarray(((1.0, 0.9), (0.95, 1.05), (1.1, 1.0)), dtype=np.float64),
        (0.0, 0.0, 0.5, 1.0, 1.0),
        (0.0, 0.0, 1.0, 1.0),
        1,
        1,
    )
    boxes = np.asarray(
        (((0.2, 0.1), (0.7, 0.9)), ((0.7, 0.2), (0.9, 0.8))),
        dtype=np.float64,
    )

    def source_key(patch: BSplineSurfacePatch) -> str:
        return canonical_fingerprint(
            (
                f"{type(patch).__module__}.{type(patch).__qualname__}",
                patch.u_degree,
                patch.v_degree,
                array_tree_fingerprint(patch),
            )
        )

    ticket = HostBatchTicket(640_000, bytes_upper=81_920_000)
    budget = CoordinateEnclosureBudget(
        ticket, maximum_work_units=640_000, maximum_memory_bytes=81_920_000
    )
    key = source_key(source)
    with ticket, budget.activate(), budget.temporary_scope():
        first = _prepared_tensor_derivative_boxes(source, boxes, 1, key, budget=budget)
        first_work = budget.work_units
        np.testing.assert_array_equal(first, (False, False))
        first[:] = True
        before = budget.work_units
        repeated = _prepared_tensor_derivative_boxes(
            source, boxes.copy(), 1, key, budget=budget
        )
        hit_work = boxes.size + 1 + repeated.size
        assert budget.work_units - before == hit_work < first_work
        np.testing.assert_array_equal(repeated, (False, False))
        before = budget.work_units
        second_order = _prepared_tensor_derivative_boxes(
            source, boxes, 2, key, budget=budget
        )
        assert budget.work_units - before > hit_work
        np.testing.assert_array_equal(second_order, (True, False))
        changed_box = boxes.copy()
        changed_box[0, 1, 0] = 0.4
        before = budget.work_units
        changed = _prepared_tensor_derivative_boxes(
            source, changed_box, 2, key, budget=budget
        )
        assert budget.work_units - before > hit_work
        np.testing.assert_array_equal(changed, (False, False))
        before = budget.work_units
        reversed_order = _prepared_tensor_derivative_boxes(
            source, boxes[::-1], 2, key, budget=budget
        )
        assert budget.work_units - before > hit_work
        np.testing.assert_array_equal(reversed_order, (False, True))
        changed_weights = np.asarray(source.weights).copy()
        changed_weights[0, 0] = np.nextafter(changed_weights[0, 0], np.inf)
        changed_source = BSplineSurfacePatch(
            source.control_points,
            changed_weights,
            source.u_knots,
            source.v_knots,
            1,
            1,
        )
        before = budget.work_units
        changed_source_result = _prepared_tensor_derivative_boxes(
            changed_source, boxes, 2, source_key(changed_source), budget=budget
        )
        assert budget.work_units - before > hit_work
        np.testing.assert_array_equal(changed_source_result, second_order)
        invalid = boxes.copy()
        invalid[0, 0, 0] = -0.1
        with pytest.raises(ValueError, match="active domain"):
            _prepared_tensor_derivative_boxes(source, invalid, 2, key, budget=budget)
    cold_ticket = HostBatchTicket(640_000, bytes_upper=81_920_000)
    cold = CoordinateEnclosureBudget(
        cold_ticket, maximum_work_units=640_000, maximum_memory_bytes=81_920_000
    )
    with cold_ticket, cold.activate():
        expected = _validate_tensor_derivative_boxes(source, boxes, 2, budget=cold)
    np.testing.assert_array_equal(second_order, expected)


def test_tensor_derivative_mixed_batch_matches_initialized_legacy_bits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from numpy.typing import DTypeLike

    from phydrax._resource_ticket import HostBatchTicket
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget

    source = phx.geometry.BSplineSurfacePatch(
        [
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.2]],
            [[0.5, 0.0, 0.1], [0.5, 1.0, 0.3]],
            [[1.0, 0.0, 0.3], [1.0, 1.0, 0.5]],
        ],
        [[1.0, 0.9], [0.95, 1.05], [1.1, 1.0]],
        [0.0, 0.0, 0.5, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        1,
        1,
    )
    boxes = np.asarray(
        [
            [[0.6, 0.1], [0.8, 0.4]],
            [[0.6, 0.1], [0.8, 0.4]],
            [[0.4, 0.2], [0.6, 0.7]],
            [[0.1, 0.2], [0.3, 0.7]],
            [[-1.0e-13, 0.2], [-1.0e-13, 0.3]],
        ],
        dtype=np.float64,
    )
    original_empty = np.empty
    poisoned_outputs = 0

    def poison_outputs(
        shape: tuple[int, ...], dtype: DTypeLike = np.float64
    ) -> np.ndarray:
        nonlocal poisoned_outputs
        result = original_empty(shape, dtype=dtype)
        if result.shape == (5, 3, 2, 2):
            result.fill(np.nan)
            poisoned_outputs += 1
        return result

    ticket = HostBatchTicket(640_000, bytes_upper=81_920_000)
    budget = CoordinateEnclosureBudget(
        ticket, maximum_work_units=640_000, maximum_memory_bytes=81_920_000
    )
    with ticket, budget.activate(), budget.temporary_scope():
        source.derivative_bounds_batch(boxes[:1], order=2, budget=budget)
        before = budget.work_units
        monkeypatch.setattr(np, "empty", poison_outputs)
        actual = source.derivative_bounds_batch(boxes, order=2, budget=budget)
        # Legacy 1481 minus 48 duplicate fills, 28 repeated source scans
        # and one already-owned full-content key object profile.
        assert budget.work_units - before == 1404
    assert poisoned_outputs == 2
    # Frozen from the preceding fully initialized implementation. These are
    # raw float64 bits, not a looser mathematical interval comparison.
    finite_rows = (
        (
            "-0x1.37dddf6212cb3p-1",
            "-0x1.319099ef1b8c3p-3",
            "-0x1.319099ef1b8c3p-3",
            "-0x1.2c990dea34615p-8",
            "0x1.949cb924ab6d5p-8",
            "-0x1.614e306fa7903p-2",
            "-0x1.614e306fa7903p-2",
            "-0x1.09087bacb69ffp-3",
            "-0x1.dfcf3caa9ed17p-3",
            "-0x1.f62cf0f635836p-4",
            "-0x1.f62cf0f635836p-4",
            "-0x1.b57bad706b4dbp-6",
        ),
        (
            "-0x1.e2ae7d1ee3205p-3",
            "0x1.77a5fc7fe1c80p-4",
            "0x1.77a5fc7fe1c80p-4",
            "0x1.7201251c3a98dp-7",
            "0x1.47f560e4a84cep-4",
            "-0x1.bc3708d457101p-5",
            "-0x1.bc3708d457101p-5",
            "0x1.6160a4e648d72p-5",
            "-0x1.722338461edaep-4",
            "0x1.9bccd6a06e2aap-7",
            "0x1.9bccd6a06e2aap-7",
            "0x1.23a7c8f59ce03p-7",
        ),
        (
            "-0x1.a50ae4d914cd0p-2",
            "-0x1.03db4deba47bap-4",
            "-0x1.03db4deba47bap-4",
            "-0x1.1fffb4dabd73ep-9",
            "-0x1.708c6ca31e6ddp-4",
            "-0x1.a1e8ed8ad33b3p-3",
            "-0x1.a1e8ed8ad33b3p-3",
            "-0x1.78f5c506ec46bp-5",
            "-0x1.9a8b3334b0538p-4",
            "-0x1.b645439a1dfafp-5",
            "-0x1.b645439a1dfafp-5",
            "-0x1.3bf7cd1079b19p-7",
        ),
        (
            "0x1.7642928812800p-5",
            "0x1.11727d30ad6bap-3",
            "0x1.11727d30ad6bap-3",
            "0x1.afff8f481c2a5p-8",
            "0x1.d5cc5238050b0p-6",
            "0x1.194ba79e3e6dep-2",
            "0x1.194ba79e3e6dep-2",
            "0x1.1ab853c53132ep-3",
            "0x1.6ced82d9804ccp-7",
            "0x1.4e6a51c5441c2p-4",
            "0x1.4e6a51c5441c2p-4",
            "0x1.d9f3b398b6867p-6",
        ),
    )
    for bound, hit_hex, enclosed_hex, c0, empty in (
        (actual[0], finite_rows[0], finite_rows[2], -np.inf, np.inf),
        (actual[1], finite_rows[1], finite_rows[3], np.inf, -np.inf),
    ):
        hit = np.asarray(tuple(map(float.fromhex, hit_hex))).reshape(3, 2, 2)
        enclosed = np.asarray(tuple(map(float.fromhex, enclosed_hex))).reshape(3, 2, 2)
        expected = np.stack(
            (hit, hit, np.full_like(hit, c0), enclosed, np.full_like(hit, empty))
        )
        np.testing.assert_array_equal(bound.view(np.uint64), expected.view(np.uint64))


def test_tensor_source_fingerprint_reuses_only_strongly_owned_immutable_leaves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from copy import copy

    from phydrax._resource_ticket import HostBatchTicket
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
    from phydrax.geometry.brep import _patches as bounds

    source = phx.geometry.BSplineSurfacePatch(
        [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.2]], [[1.0, 0.0, 0.3], [1.0, 1.0, 0.5]]],
        [[1.0, 0.9], [1.1, 1.0]],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        1,
        1,
    )
    equal = phx.geometry.BSplineSurfacePatch(
        np.asarray(source.control_points).copy(),
        np.asarray(source.weights).copy(),
        np.asarray(source.u_knots).copy(),
        np.asarray(source.v_knots).copy(),
        1,
        1,
    )
    original_fingerprint = bounds.array_tree_fingerprint
    legacy_key = bounds.canonical_fingerprint(
        {
            "kind": "source-bspline-surface-derivative-bounds",
            "type": f"{type(source).__module__}.{type(source).__qualname__}",
            "degrees": (source.u_degree, source.v_degree),
            "coefficients": original_fingerprint(source),
        }
    )
    scanned: list[phx.geometry.BSplineSurfacePatch] = []

    def fingerprint(patch: phx.geometry.BSplineSurfacePatch) -> dict[str, Any]:
        scanned.append(patch)
        return original_fingerprint(patch)

    monkeypatch.setattr(bounds, "array_tree_fingerprint", fingerprint)
    ticket = HostBatchTicket(640_000, bytes_upper=81_920_000)
    budget = CoordinateEnclosureBudget(
        ticket, maximum_work_units=640_000, maximum_memory_bytes=81_920_000
    )
    with ticket, budget.activate(), budget.temporary_scope():
        first = bounds._bspline_source_key(
            source, kind="source-bspline-surface-derivative-bounds", budget=budget
        )
        assert first == legacy_key
        retained = budget.retained_basis_bytes
        assert retained > 0
        before = budget.work_units
        assert (
            bounds._bspline_source_key(
                source, kind="source-bspline-surface-derivative-bounds", budget=budget
            )
            == first
        )
        assert budget.work_units - before == 5
        assert budget.retained_basis_bytes == retained
        assert len(scanned) == 1
        for leaf in (
            source.control_points,
            source.weights,
            source.u_knots,
            source.v_knots,
        ):
            assert budget.retained_owner_objects[id(leaf)] is leaf

        assert (
            bounds._bspline_source_key(
                equal, kind="source-bspline-surface-derivative-bounds", budget=budget
            )
            == first
        )
        assert len(scanned) == 2
        assert all(
            old is not new
            for old, new in zip(
                (source.control_points, source.weights, source.u_knots, source.v_knots),
                (equal.control_points, equal.weights, equal.u_knots, equal.v_knots),
                strict=True,
            )
        )
        replaced = eqx.tree_at(
            lambda patch: patch.weights,
            source,
            source.weights.at[0, 0].set(np.nextafter(1.0, np.inf)),
        )
        assert (
            bounds._bspline_source_key(
                replaced, kind="source-bspline-surface-derivative-bounds", budget=budget
            )
            != first
        )
        assert len(scanned) == 3

        # Alter only the static degree to exercise key invalidation, not a
        # numerical query on a constructor-invalid knot/degree combination.
        changed_degree = copy(source)
        object.__setattr__(changed_degree, "u_degree", 2)
        assert (
            bounds._bspline_source_key(
                changed_degree,
                kind="source-bspline-surface-derivative-bounds",
                budget=budget,
            )
            != first
        )
        assert len(scanned) == 4

        mutable = copy(source)
        mutable_weights = np.asarray(source.weights).copy()
        object.__setattr__(mutable, "weights", mutable_weights)
        entries = len(budget.immutable_bspline_fingerprint_cache)
        assert (
            bounds._bspline_source_key(
                mutable, kind="source-bspline-surface-derivative-bounds", budget=budget
            )
            == first
        )
        assert (
            bounds._bspline_source_key(
                mutable, kind="source-bspline-surface-derivative-bounds", budget=budget
            )
            == first
        )
        assert len(scanned) == 6
        assert len(budget.immutable_bspline_fingerprint_cache) == entries
        mutable_weights[0, 0] = np.nextafter(1.0, np.inf)
        assert (
            bounds._bspline_source_key(
                mutable, kind="source-bspline-surface-derivative-bounds", budget=budget
            )
            != first
        )
        assert len(scanned) == 7


@pytest.mark.parametrize(
    "kind", ("source-bspline-spans", "source-bspline-curve-value-bounds")
)
def test_curve_source_fingerprint_reuses_only_strongly_owned_immutable_leaves(
    kind: Literal["source-bspline-spans", "source-bspline-curve-value-bounds"],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from copy import copy

    from phydrax._resource_ticket import HostBatchTicket
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
    from phydrax.geometry.brep import _patches as bounds

    source = BSplineCurve(
        [[0.0, 0.0], [0.5, 0.2], [1.0, 0.0]],
        [1.0, 0.9, 1.1],
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        2,
    )
    equal = BSplineCurve(
        np.asarray(source.control_points).copy(),
        np.asarray(source.weights).copy(),
        np.asarray(source.knots).copy(),
        source.degree,
    )
    original_fingerprint = bounds.array_tree_fingerprint
    legacy_key = bounds.canonical_fingerprint(
        {
            "kind": kind,
            "type": f"{type(source).__module__}.{type(source).__qualname__}",
            "degree": source.degree,
            "coefficients": original_fingerprint(source),
        }
    )
    scans = 0

    def fingerprint(curve: BSplineCurve) -> dict[str, Any]:
        nonlocal scans
        scans += 1
        return original_fingerprint(curve)

    monkeypatch.setattr(bounds, "array_tree_fingerprint", fingerprint)
    ticket = HostBatchTicket(640_000, bytes_upper=81_920_000)
    budget = CoordinateEnclosureBudget(
        ticket, maximum_work_units=640_000, maximum_memory_bytes=81_920_000
    )
    with ticket, budget.activate(), budget.temporary_scope():
        first = bounds._bspline_source_key(source, kind=kind, budget=budget)
        assert first == legacy_key and scans == 1
        retained = budget.retained_basis_bytes
        assert retained > 0
        before = budget.work_units
        assert bounds._bspline_source_key(source, kind=kind, budget=budget) == first
        assert budget.work_units - before == 4
        assert budget.retained_basis_bytes == retained
        assert scans == 1
        for leaf in (source.control_points, source.weights, source.knots):
            assert budget.retained_owner_objects[id(leaf)] is leaf

        assert bounds._bspline_source_key(equal, kind=kind, budget=budget) == first
        assert scans == 2
        assert all(
            old is not new
            for old, new in zip(
                (source.control_points, source.weights, source.knots),
                (equal.control_points, equal.weights, equal.knots),
                strict=True,
            )
        )
        replaced = eqx.tree_at(
            lambda curve: curve.weights,
            source,
            source.weights.at[0].set(np.nextafter(1.0, np.inf)),
        )
        assert bounds._bspline_source_key(replaced, kind=kind, budget=budget) != first
        assert scans == 3
        changed_degree = copy(source)
        object.__setattr__(changed_degree, "degree", 1)
        assert (
            bounds._bspline_source_key(changed_degree, kind=kind, budget=budget) != first
        )
        assert scans == 4

        mutable = copy(source)
        mutable_weights = np.asarray(source.weights).copy()
        object.__setattr__(mutable, "weights", mutable_weights)
        entries = len(budget.immutable_bspline_fingerprint_cache)
        assert bounds._bspline_source_key(mutable, kind=kind, budget=budget) == first
        assert bounds._bspline_source_key(mutable, kind=kind, budget=budget) == first
        assert scans == 6
        assert len(budget.immutable_bspline_fingerprint_cache) == entries
        mutable_weights[0] = np.nextafter(1.0, np.inf)
        assert bounds._bspline_source_key(mutable, kind=kind, budget=budget) != first
        assert scans == 7
