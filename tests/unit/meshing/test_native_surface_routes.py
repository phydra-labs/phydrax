#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization._cell_complex import PolygonalConnectivity
from phydrax.discretization._cell_geometry import ScalarCoordinateElement
from phydrax.discretization._cell_geometry_validity import (
    certify_cell_geometry_validity,
)
from phydrax.discretization._surface_chart_records import SurfaceChartWitness
from phydrax.geometry._curve_boundary import ParametricCurveBoundarySource
from phydrax.geometry._domain_boundary import MeshingDomainBoundarySource
from phydrax.geometry._meshing_domain import (
    MeshingDomain,
    MeshingDomainCurve,
    MeshingDomainRegion,
    MeshingSourceAccuracy,
    MeshingSurfacePatch,
    PatchCurveUse,
)
from phydrax.geometry._source_boundary_records import (
    SourceBoundaryChartWitness,
    SourceBoundaryPointWitness,
)
from phydrax.geometry._source_fidelity import (
    certify_source_fidelity,
    validate_source_fidelity_witness,
)
from phydrax.geometry.brep import brep_cone, brep_cylinder, brep_sphere
from phydrax.geometry.brep._patches import (
    BSplineCurve,
    BSplineSurfacePatch,
    LineCurve,
)
from phydrax.geometry.reconstruction._core import (
    ReconstructedGeometrySource,
    ReconstructionReport,
)
from phydrax.geometry.simplicial._regions import MeshRegion
from phydrax.geometry.surface import SurfaceModel
from phydrax.meshing._association import GeometrySourceEntityRole
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    SurfaceMeshingSpec,
)
from phydrax.meshing._controls import FeatureKind, ProtectedFeature
from phydrax.meshing._native_construction import NativeMeshConstruction
from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
from phydrax.meshing._sizing import SizeControlStrength, UniformSizeControl
from phydrax.meshing.providers._native_options import NativeMeshingOptions
from phydrax.meshing.providers._native_schedules import (
    NativeCurveSchedule,
    NativeSurfaceSchedule,
)
from phydrax.meshing.providers._native_surface import prepare_native_surface


def _warped_surface() -> BSplineSurfacePatch:
    u_parameters = tuple(Fraction(index, 2) for index in range(3))
    v_parameters = tuple(Fraction(index, 2) for index in range(3))
    quadratic_u = Fraction.from_float(0.25)
    quadratic_v = Fraction.from_float(0.15)
    cross = Fraction.from_float(0.1)
    exact_control_points = tuple(
        tuple(
            (
                u,
                v,
                quadratic_u * (u_index == 2)
                + quadratic_v * (v_index == 2)
                + cross * u * v,
            )
            for v_index, v in enumerate(v_parameters)
        )
        for u_index, u in enumerate(u_parameters)
    )
    knots = (Fraction(0),) * 3 + (Fraction(1),) * 3
    controls = np.asarray(
        [
            [[float(value) for value in point] for point in row]
            for row in exact_control_points
        ],
        dtype=np.float64,
    )
    return BSplineSurfacePatch(
        controls,
        np.ones((3, 3), dtype=np.float64),
        np.asarray(knots, dtype=np.float64),
        np.asarray(knots, dtype=np.float64),
        2,
        2,
        exact_control_points=exact_control_points,
        exact_u_knots=knots,
        exact_v_knots=knots,
        control_point_ids=np.arange(9, dtype=np.int64).reshape((3, 3)),
    )


def _square_domain(
    *,
    accuracy: MeshingSourceAccuracy = "exact_represented",
    tolerance: float = 0.0,
    authored_corners: bool = False,
) -> MeshingDomain:
    chart_points = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    pairs = ((0, 1), (1, 2), (2, 3), (3, 0))
    curves = tuple(MeshingDomainCurve(first, last) for first, last in pairs)
    uses = []
    for curve, (first, last) in enumerate(pairs):
        start = np.asarray(chart_points[first], dtype=np.float64)
        stop = np.asarray(chart_points[last], dtype=np.float64)
        uses.append(
            PatchCurveUse(
                curve,
                LineCurve(start, stop - start),
                0.0,
                1.0,
                sense=1,
            )
        )
    return MeshingDomain(
        (
            MeshingSurfacePatch(
                _warped_surface(),
                (tuple(uses),),
            ),
        ),
        curves,
        4,
        source_id="native-parametric-square",
        source_revision="warped-square-r1",
        regions=(MeshingDomainRegion("body", ((0, 1),)),),
        tolerance=tolerance,
        accuracy=accuracy,
        corner_points=(
            np.asarray(
                ((0.0, 0.0, 0.0), (1.0, 0.0, 0.25), (1.0, 1.0, 0.5), (0.0, 1.0, 0.15)),
                dtype=np.float64,
            )
            if authored_corners
            else None
        ),
    )


def _flat_nurbs_surface(
    lower_x: float, upper_x: float, control_id: int, /
) -> BSplineSurfacePatch:
    exact_knots = (Fraction(0), Fraction(0), Fraction(1), Fraction(1))
    exact_control_points = tuple(
        tuple(
            (
                Fraction.from_float(x),
                Fraction.from_float(float(v)),
                Fraction(0),
            )
            for v in (0.0, 1.0)
        )
        for x in (lower_x, upper_x)
    )
    return BSplineSurfacePatch(
        np.asarray(
            [
                [[float(value) for value in point] for point in row]
                for row in exact_control_points
            ],
            dtype=np.float64,
        ),
        np.ones((2, 2), dtype=np.float64),
        np.asarray(exact_knots, dtype=np.float64),
        np.asarray(exact_knots, dtype=np.float64),
        1,
        1,
        exact_control_points=exact_control_points,
        exact_u_knots=exact_knots,
        exact_v_knots=exact_knots,
        control_point_ids=np.arange(control_id, control_id + 4, dtype=np.int64).reshape(
            (2, 2)
        ),
    )


def _curved_trim_domain() -> MeshingDomain:
    bottom = BSplineCurve(
        np.asarray(((0.0, 0.0), (0.5, 0.2), (1.0, 0.0)), dtype=np.float64),
        np.ones((3,), dtype=np.float64),
        np.asarray((0.0, 0.0, 0.0, 1.0, 1.0, 1.0), dtype=np.float64),
        2,
    )
    curves = tuple(
        MeshingDomainCurve(first, last)
        for first, last in (
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (1, 4),
            (4, 5),
            (5, 2),
        )
    )
    left_uses = (
        PatchCurveUse(0, bottom, 0.0, 1.0),
        PatchCurveUse(1, LineCurve((1.0, 0.0), (0.0, 1.0)), 0.0, 1.0),
        PatchCurveUse(2, LineCurve((1.0, 1.0), (-1.0, 0.0)), 0.0, 1.0),
        PatchCurveUse(3, LineCurve((0.0, 1.0), (0.0, -1.0)), 0.0, 1.0),
    )
    right_uses = (
        PatchCurveUse(4, LineCurve((0.0, 0.0), (1.0, 0.0)), 0.0, 1.0),
        PatchCurveUse(5, LineCurve((1.0, 0.0), (0.0, 1.0)), 0.0, 1.0),
        PatchCurveUse(6, LineCurve((1.0, 1.0), (-1.0, 0.0)), 0.0, 1.0),
        PatchCurveUse(
            1,
            LineCurve((0.0, 0.0), (0.0, 1.0)),
            0.0,
            1.0,
            sense=-1,
        ),
    )
    return MeshingDomain(
        (
            MeshingSurfacePatch(_flat_nurbs_surface(0.0, 0.5, 0), (left_uses,)),
            MeshingSurfacePatch(_flat_nurbs_surface(0.5, 1.0, 4), (right_uses,)),
        ),
        curves,
        6,
        source_id="native-parametric-curved-trim",
        source_revision="curved-trim-r1",
        regions=(MeshingDomainRegion("body", ((0, 1), (1, 1))),),
    )


def _surface_specification(
    domain: MeshingDomain,
    *,
    maximum_deviation: float | None = 0.015,
) -> SurfaceMeshingSpec:
    scope = MeshingScope(
        domain.source_id,
        domain.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        domain.scope_entity_set_id(2),
        domain.scope_entity_ids(2),
    )
    target = CellMeshingTarget(
        2,
        3,
        CellFamilyPolicy(required=("triangle",)),
    )
    return SurfaceMeshingSpec(
        target,
        scope,
        size_controls=(
            UniformSizeControl(scope, 0.8, strength=SizeControlStrength.HARD),
        ),
        maximum_deviation=maximum_deviation,
    )


def _curve_point(domain: MeshingDomain, curve: int, parameter: float, /) -> np.ndarray:
    patch, loop, use_row = (int(value) for value in domain.curve_owners[curve])
    use = domain.patches[patch].loops[loop][use_row]
    native_parameter = use.first + parameter * (use.last - use.first)
    uv = np.asarray(use.pcurve.evaluate(jnp.asarray(native_parameter)), dtype=np.float64)
    return np.asarray(
        domain.patches[patch].surface.evaluate(jnp.asarray(uv)), dtype=np.float64
    )


def _point_segment_distance(
    point: np.ndarray,
    first: np.ndarray,
    last: np.ndarray,
    /,
) -> float:
    direction = last - first
    denominator = float(np.dot(direction, direction))
    parameter = (
        0.0
        if denominator == 0.0
        else float(np.clip(np.dot(point - first, direction) / denominator, 0.0, 1.0))
    )
    return float(np.linalg.norm(point - (first + parameter * direction)))


def _segments_distance(
    point: np.ndarray, first: np.ndarray, last: np.ndarray, /
) -> np.ndarray:
    direction = last - first
    denominator = np.einsum("ij,ij->i", direction, direction)
    numerator = np.einsum("ij,ij->i", point - first, direction)
    parameter = np.clip(
        np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0.0,
        ),
        0.0,
        1.0,
    )
    return np.linalg.norm(point - (first + parameter[:, None] * direction), axis=1)


def _point_mesh_distance(point: np.ndarray, triangles: np.ndarray, /) -> float:
    """Euclidean distance from one point to a triangle soup."""
    first, second, third = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    edge_first, edge_second = second - first, third - first
    normal = np.cross(edge_first, edge_second)
    normal_squared = np.einsum("ij,ij->i", normal, normal)
    relative = point - first
    signed = np.einsum("ij,ij->i", relative, normal)
    safe_normal = np.where(normal_squared > 0.0, normal_squared, 1.0)
    projection = relative - (signed / safe_normal)[:, None] * normal
    dot_first_first = np.einsum("ij,ij->i", edge_first, edge_first)
    dot_first_second = np.einsum("ij,ij->i", edge_first, edge_second)
    dot_second_second = np.einsum("ij,ij->i", edge_second, edge_second)
    dot_relative_first = np.einsum("ij,ij->i", projection, edge_first)
    dot_relative_second = np.einsum("ij,ij->i", projection, edge_second)
    denominator = dot_first_first * dot_second_second - dot_first_second**2
    safe_denominator = np.where(denominator > 0.0, denominator, 1.0)
    first_weight = (
        dot_second_second * dot_relative_first - dot_first_second * dot_relative_second
    ) / safe_denominator
    second_weight = (
        dot_first_first * dot_relative_second - dot_first_second * dot_relative_first
    ) / safe_denominator
    inside = (
        (normal_squared > 0.0)
        & (denominator > 0.0)
        & (first_weight >= 0.0)
        & (second_weight >= 0.0)
        & (first_weight + second_weight <= 1.0)
    )
    face = np.abs(signed) / np.sqrt(safe_normal)
    edges = np.minimum(
        np.minimum(
            _segments_distance(point, first, second),
            _segments_distance(point, second, third),
        ),
        _segments_distance(point, third, first),
    )
    return float(np.min(np.where(inside, face, edges)))


def _source_to_mesh_sampled_distance(
    domain: MeshingDomain,
    construction: NativeMeshConstruction,
    /,
) -> float:
    patch_rows = []
    parameter_rows = []
    for patch, record in enumerate(domain.patches):
        parameter_domain = record.surface.parameter_domain
        if parameter_domain is None:
            trim_boxes = np.asarray(
                tuple(
                    use.pcurve.bounding_box(*sorted((use.first, use.last)))
                    for loop in record.loops
                    for use in loop
                ),
                dtype=np.float64,
            )
            lower = np.min(trim_boxes[:, 0], axis=0)
            upper = np.max(trim_boxes[:, 1], axis=0)
        else:
            lower, upper = parameter_domain
        first = np.linspace(lower[0], upper[0], 17, dtype=np.float64)
        last = np.linspace(lower[1], upper[1], 13, dtype=np.float64)
        first_grid, last_grid = np.meshgrid(first, last, indexing="ij")
        parameter_rows.append(np.stack((first_grid, last_grid), axis=-1).reshape((-1, 2)))
        patch_rows.extend([patch] * first_grid.size)
    source_points = domain.evaluate(
        np.asarray(patch_rows, dtype=np.int64),
        np.concatenate(parameter_rows, axis=0),
    )
    coordinates = np.asarray(construction.mesh.coordinates, dtype=np.float64)
    triangles = coordinates[
        np.concatenate(
            tuple(
                np.asarray(block.vertices, dtype=np.int64)
                for block in construction.mesh.blocks
            )
        )
    ]
    return max(
        _point_mesh_distance(point, triangles)
        for point in np.asarray(source_points, dtype=np.float64)
    )


def _assert_vertices_on_domain(
    domain: MeshingDomain,
    construction: NativeMeshConstruction,
    /,
) -> None:
    mesh = construction.mesh
    actual = np.asarray(mesh.coordinates, dtype=np.float64)
    vertex_association = next(
        association
        for association in construction.organization.associations
        if association.target_entity_set_id == mesh.entity_set(0).entity_set_id
    )
    vertex_rows = {
        int(identifier): row
        for row, identifier in enumerate(mesh.vertex_global_ids.tolist())
    }
    dimensions = np.asarray(vertex_association.source_dimensions, dtype=np.int64)
    parameters = np.asarray(vertex_association.parameters, dtype=np.float64)
    for association_row, identifier in enumerate(
        np.asarray(vertex_association.target_global_ids, dtype=np.int64).tolist()
    ):
        dimension = int(dimensions[association_row])
        source_row = domain.resolve_entity_id(
            dimension,
            vertex_association.source_entity_ids[association_row],
            occurrence_path=vertex_association.source_occurrence_paths[association_row],
        )
        if dimension == 0:
            expected = domain.corner_points[source_row]
        elif dimension == 1:
            expected = _curve_point(domain, source_row, parameters[association_row, 0])
        else:
            expected = domain.evaluate(
                np.asarray((source_row,), dtype=np.int64),
                parameters[association_row, :2][None, :],
            )[0]
        np.testing.assert_allclose(
            actual[vertex_rows[int(identifier)]], expected, atol=2e-12
        )


def test_authored_domain_corner_points_have_collapsed_source_bounds() -> None:
    domain = _square_domain(authored_corners=True)
    for row, point in enumerate(domain.corner_points):
        lower, upper = domain.source_corner_point_bounds(row)
        exact = tuple(Fraction.from_float(float(value)) for value in point)
        assert lower == exact
        assert upper == exact


def test_surface_size_is_not_an_inferred_fidelity_request() -> None:
    domain = _square_domain()
    specification = _surface_specification(domain, maximum_deviation=None)
    construction = prepare_native_surface(
        domain,
        specification,
        NativeMeshingOptions("parametric_surface"),
        SpatialCoordinateContract.si(),
    ).execute()

    assert construction.certification.fidelity_tolerance is None
    assert not any(
        name == "maximum_deviation" for name, _ in construction.compliance.requested
    )
    assert construction.compliance.passed
    assert "maximum_edge_length" in dict(construction.compliance.requested)


def test_parametric_surface_route_maps_source_charts_and_trim_curves() -> None:
    domain = _square_domain()
    specification = _surface_specification(domain)
    options = NativeMeshingOptions(
        "parametric_surface",
        surface_schedule=NativeSurfaceSchedule(
            quality_angle_degrees=20.0,
            spacing_fraction=0.45,
            minimum_edge_samples=4,
            curve_schedule=NativeCurveSchedule(fidelity_samples=5),
        ),
    )

    construction = prepare_native_surface(
        domain,
        specification,
        options,
        SpatialCoordinateContract.si(),
    ).execute()
    assert isinstance(construction, NativeMeshConstruction)

    mesh = construction.mesh
    assert mesh.topological_dimension == 2
    assert mesh.ambient_dimension == 3
    assert all(block.cell_kind == "triangle" for block in mesh.blocks)
    np.testing.assert_array_equal(
        np.asarray(construction.geometry.coordinates), np.asarray(mesh.coordinates)
    )

    element = construction.geometry.elements[0]
    if not isinstance(element, ScalarCoordinateElement):
        raise AssertionError("Surface geometry element must support scalar tabulation.")
    basis, _ = element.tabulate(np.asarray(((1.0 / 3.0, 1.0 / 3.0),)))
    dofs = np.asarray(construction.geometry.geometry_dofs[0], dtype=np.int64)[0]
    mapped_center = (
        np.asarray(basis, dtype=np.float64)[0]
        @ np.asarray(construction.geometry.coordinates, dtype=np.float64)[dofs]
    )
    np.testing.assert_allclose(
        mapped_center, np.mean(mesh.coordinates[mesh.blocks[0].vertices[0]], axis=0)
    )
    assert isinstance(construction.organization.boundary, SurfaceModel)
    source_charts = construction.organization.surface_source
    assert source_charts is not None
    source_charts.require_current()
    corner_lower, corner_upper = domain.source_corner_point_bounds(0)
    assert all(
        lower <= upper for lower, upper in zip(corner_lower, corner_upper, strict=True)
    )
    assert all(
        lower <= Fraction.from_float(float(value)) <= upper
        for lower, upper, value in zip(
            corner_lower,
            corner_upper,
            domain.corner_points[0],
            strict=True,
        )
    )
    exact_cover = source_charts.boundary_source.boundary_chart_cover()
    assert exact_cover.complete
    assert exact_cover.semantics == "certified"

    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    expected_height = (
        0.25 * coordinates[:, 0] ** 2
        + 0.15 * coordinates[:, 1] ** 2
        + 0.1 * coordinates[:, 0] * coordinates[:, 1]
    )
    np.testing.assert_allclose(coordinates[:, 2], expected_height, atol=2e-12)

    block = mesh.blocks[0]
    faces = np.asarray(block.vertices, dtype=np.int64)
    face_association = next(
        association
        for association in construction.organization.associations
        if association.target_entity_set_id == mesh.entity_set(2).entity_set_id
    )
    assert set(face_association.source_entity_ids) == {domain.entity_id(2, 0)}
    assert all(
        role is GeometrySourceEntityRole.FACET for role in face_association.source_roles
    )
    face_charts = np.asarray(face_association.parameters, dtype=np.float64)
    assert face_charts.shape == (faces.shape[0], 2)
    assert np.all(np.isfinite(face_charts))

    edge_association = next(
        association
        for association in construction.organization.associations
        if association.target_entity_set_id == mesh.entity_set(1).entity_set_id
    )
    assert all(
        role is GeometrySourceEntityRole.EDGE for role in edge_association.source_roles
    )
    intervals = edge_association.source_parameter_intervals
    assert intervals is not None
    expected_curves = {
        domain.source_entity_key_for_scope(1, int(identifier))[0]
        for identifier in domain.scope_entity_ids(1)
    }
    assert set(edge_association.source_entity_ids) == expected_curves
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolygonalConnectivity):
        raise AssertionError("Surface route must retain polygonal connectivity.")
    edge_vertices = np.asarray(connectivity.edges, dtype=np.int64)
    edge_ids = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
    edge_by_id = {
        int(identifier): edge
        for identifier, edge in zip(
            edge_ids.tolist(), edge_vertices.tolist(), strict=True
        )
    }
    curve_by_key = {}
    for identifier in domain.scope_entity_ids(1).tolist():
        source_key, occurrence = domain.source_entity_key_for_scope(1, int(identifier))
        curve_by_key[source_key] = domain.resolve_entity_id(
            1, source_key, occurrence_path=occurrence
        )
    interval_rows: dict[str, list[tuple[Fraction, Fraction]]] = {}
    for row, (source_key, interval) in enumerate(
        zip(edge_association.source_entity_ids, intervals, strict=True)
    ):
        assert interval is not None
        lower, upper = interval
        assert isinstance(lower, Fraction) and isinstance(upper, Fraction)
        assert lower < upper
        interval_rows.setdefault(source_key, []).append(interval)
        curve = curve_by_key[source_key]
        expected_endpoints = np.asarray(
            (
                _curve_point(domain, curve, float(lower)),
                _curve_point(domain, curve, float(upper)),
            )
        )
        edge_id = int(edge_association.target_global_ids[row])
        actual_endpoints = coordinates[np.asarray(edge_by_id[edge_id])]
        direct = np.linalg.norm(actual_endpoints - expected_endpoints, axis=1).max()
        reverse = np.linalg.norm(
            actual_endpoints[::-1] - expected_endpoints, axis=1
        ).max()
        assert min(direct, reverse) < 2e-11
    for rows in interval_rows.values():
        ordered = sorted(rows)
        assert ordered[0][0] == 0
        assert ordered[-1][1] == 1
        assert all(first[1] == second[0] for first, second in zip(ordered, ordered[1:]))

        sample_parameters = {parameter for interval in rows for parameter in interval}
        surface_schedule = options.surface_schedule
        assert surface_schedule is not None
        assert len(sample_parameters) >= surface_schedule.minimum_edge_samples

    vertices = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    vertex_association = next(
        association
        for association in construction.organization.associations
        if association.target_entity_set_id == mesh.entity_set(0).entity_set_id
    )
    assert set(vertex_association.source_entity_ids) >= {
        domain.entity_id(0, row) for row in range(domain.corner_count)
    }
    assert vertices.size == coordinates.shape[0]
    assert construction.certification.fidelity_source is source_charts.boundary_source
    assert construction.certification.fidelity_tolerance == 0.015
    assert construction.compliance.passed


def test_parametric_surface_route_certifies_hard_curve_and_corner_features() -> None:
    domain = _square_domain(authored_corners=True)
    base = _surface_specification(domain)
    curve_id = int(domain.scope_entity_ids(1)[0])
    point_id = int(domain.scope_entity_ids(0)[0])
    curve_key = domain.source_entity_key_for_scope(1, curve_id)
    curve_scope = MeshingScope(
        domain.source_id,
        domain.source_revision,
        MeshingEntityKind.GEOMETRY,
        1,
        domain.scope_entity_set_id(1),
        np.asarray((curve_id,), dtype=np.int64),
    )
    point_scope = MeshingScope(
        domain.source_id,
        domain.source_revision,
        MeshingEntityKind.GEOMETRY,
        0,
        domain.scope_entity_set_id(0),
        np.asarray((point_id,), dtype=np.int64),
    )
    specification = SurfaceMeshingSpec(
        base.target,
        base.scope,
        size_controls=base.size_controls,
        protected_features=(
            ProtectedFeature(
                curve_scope,
                FeatureKind.CURVE,
                maximum_deviation=0.01,
                hard=True,
            ),
            ProtectedFeature(
                point_scope,
                FeatureKind.CORNER,
                maximum_deviation=0.01,
                hard=True,
            ),
        ),
        maximum_deviation=base.maximum_deviation,
    )
    construction = prepare_native_surface(
        domain,
        specification,
        NativeMeshingOptions(
            "parametric_surface",
            surface_schedule=NativeSurfaceSchedule(spacing_fraction=0.45),
        ),
        SpatialCoordinateContract.si(),
    ).execute()
    scoped = construction.certification.scoped_fidelity
    assert len(scoped) == 2
    curve_query, curve_witness, curve_tolerance = next(
        row for row in scoped if isinstance(row[1], SourceBoundaryChartWitness)
    )
    point_query, point_witness, point_tolerance = next(
        row for row in scoped if isinstance(row[1], SourceBoundaryPointWitness)
    )
    assert isinstance(curve_query, ParametricCurveBoundarySource)
    assert isinstance(curve_witness, SourceBoundaryChartWitness)
    assert isinstance(point_query, MeshingDomainBoundarySource)
    assert point_query is construction.certification.fidelity_source
    assert isinstance(point_witness, SourceBoundaryPointWitness)
    assert curve_tolerance == point_tolerance == 0.01
    curve_query.validate_scope(
        domain, np.asarray((curve_id,), dtype=np.int64), (curve_key[1],)
    )
    assert set(curve_witness.source_chart_ids) == set(curve_query.chart_ids)
    assert point_witness.source_query_id == point_query.source_scope_id
    assert point_witness.source_entity_ids == (
        domain.source_entity_key_for_scope(0, point_id)[0],
    )


@pytest.mark.parametrize("hard", (True, False), ids=("hard", "soft"))
def test_parametric_surface_feature_preserves_requested_fidelity(hard: bool) -> None:
    domain = _square_domain()
    base = _surface_specification(domain)
    tolerance = 0.003 if hard else 0.0
    feature = ProtectedFeature(
        base.scope, FeatureKind.SURFACE, maximum_deviation=tolerance, hard=hard
    )
    specification = SurfaceMeshingSpec(
        base.target,
        base.scope,
        size_controls=base.size_controls,
        protected_features=(feature,),
        maximum_deviation=base.maximum_deviation,
    )
    construction = prepare_native_surface(
        domain,
        specification,
        NativeMeshingOptions("parametric_surface"),
        SpatialCoordinateContract.si(),
    ).execute()
    key = f"feature:{feature.feature_id}:maximum_deviation"
    assert dict(construction.compliance.requested)[key] == tolerance
    observed = dict(construction.compliance.achieved)[key]
    if hard:
        assert observed <= tolerance
        query, witness, limit = construction.certification.scoped_fidelity[0]
        certificate = certify_source_fidelity(
            construction.mesh,
            construction.geometry,
            query,
            tolerance=limit,
            witness=witness,
        )
        assert certificate.status == "certified"
        assert certificate.mesh_to_source_upper <= tolerance
        assert certificate.source_to_mesh_upper <= tolerance
    else:
        assert observed > tolerance
        assert base.maximum_deviation is not None
        assert observed <= base.maximum_deviation


def test_public_parametric_surface_protected_feature() -> None:
    from phydrax.meshing import NativeMeshingOptions, NativeMeshingProvider

    domain = _square_domain()
    base = _surface_specification(domain)
    feature = ProtectedFeature(
        base.scope, FeatureKind.SURFACE, maximum_deviation=0.003
    )
    specification = SurfaceMeshingSpec(
        base.target,
        base.scope,
        size_controls=base.size_controls,
        maximum_deviation=base.maximum_deviation,
        protected_features=(feature,),
    )
    result = (
        NativeMeshingProvider(NativeMeshingOptions("parametric_surface"))
        .plan(domain, specification, coordinate_contract=SpatialCoordinateContract.si())
        .execute()
    )
    key = f"feature:{feature.feature_id}:maximum_deviation"
    assert dict(result.compliance.requested)[key] == 0.003
    assert dict(result.compliance.achieved)[key] <= 0.003


def test_surface_feature_scope_does_not_tighten_an_unselected_curved_patch() -> None:
    uv = np.asarray(((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.float64)
    surfaces = (_flat_nurbs_surface(-2.0, -1.0, 100), _warped_surface())
    domain = MeshingDomain(
        tuple(
            MeshingSurfacePatch(
                surface,
                (
                    tuple(
                        PatchCurveUse(
                            4 * patch + edge,
                            LineCurve(uv[edge], uv[(edge + 1) % 4] - uv[edge]),
                            0.0,
                            1.0,
                        )
                        for edge in range(4)
                    ),
                ),
            )
            for patch, surface in enumerate(surfaces)
        ),
        tuple(
            MeshingDomainCurve(4 * patch + edge, 4 * patch + (edge + 1) % 4)
            for patch in range(2)
            for edge in range(4)
        ),
        8,
        source_id="independent-protected-surface-patches",
        source_revision="authored",
    )
    base = _surface_specification(domain)
    selected_key = (domain.entity_id(2, 0), domain.source_occurrences[2][0])
    face_id = next(
        int(identifier)
        for identifier in domain.scope_entity_ids(2)
        if domain.source_entity_key_for_scope(2, int(identifier)) == selected_key
    )
    feature_scope = MeshingScope(
        domain.source_id,
        domain.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        domain.scope_entity_set_id(2),
        np.asarray((face_id,), dtype=np.int64),
    )
    feature = ProtectedFeature(feature_scope, FeatureKind.SURFACE, maximum_deviation=0.0)
    specification = SurfaceMeshingSpec(
        base.target,
        base.scope,
        size_controls=base.size_controls,
        maximum_deviation=base.maximum_deviation,
        protected_features=(feature,),
    )
    construction = prepare_native_surface(
        domain,
        specification,
        NativeMeshingOptions("parametric_surface"),
        SpatialCoordinateContract.si(),
    ).execute()
    query, witness, tolerance = construction.certification.scoped_fidelity[0]
    assert isinstance(query, MeshingDomainBoundarySource)
    assert isinstance(witness, SurfaceChartWitness)
    assert query.patches == (0,)
    certificate = certify_source_fidelity(
        construction.mesh,
        construction.geometry,
        query,
        tolerance=tolerance,
        witness=witness,
    )
    assert certificate.status == "certified", certificate.findings
    assert certificate.mesh_to_source_upper == certificate.source_to_mesh_upper == 0.0
    assert set(np.asarray(certificate.source_cell_global_ids).tolist()) == set(
        np.asarray(witness.cell_global_ids)[np.asarray(witness.patches) == 0].tolist()
    )
    assert dict(construction.compliance.achieved)["maximum_deviation"] > 0.0
    incomplete = SurfaceChartWitness(
        geometry_id=witness.geometry_id,
        topology_id=witness.topology_id,
        domain_id=witness.domain_id,
        cell_global_ids=witness.cell_global_ids[:-1],
        patches=witness.patches[:-1],
        charts=witness.charts[:-1],
        source_root_cell_ids=witness.source_root_cell_ids[:-1],
        source_piece_ids=witness.source_piece_ids[:-1],
        geometry_entity_ids=witness.geometry_entity_ids[:-1],
        occurrence_paths=witness.occurrence_paths[:-1],
        chart_rational_tokens=tuple(witness.chart_rational_tokens[:-1]),
        root_mesh_cell_bindings=witness.root_mesh_cell_bindings[:-1],
    )
    with pytest.raises(ValueError, match="every target cell global ID"):
        validate_source_fidelity_witness(
            construction.mesh, construction.geometry, query, incomplete
        )


def test_parametric_surface_route_retains_bounded_source_accuracy() -> None:
    domain = _square_domain(accuracy="bounded_approximation", tolerance=0.002)
    construction = prepare_native_surface(
        domain,
        _surface_specification(domain),
        NativeMeshingOptions(
            "parametric_surface",
            surface_schedule=NativeSurfaceSchedule(spacing_fraction=0.45),
        ),
        SpatialCoordinateContract.si(),
    ).execute()

    source_charts = construction.organization.surface_source
    assert source_charts is not None
    assert source_charts.domain.accuracy == "bounded_approximation"
    assert source_charts.domain.tolerance == 0.002
    cover = source_charts.boundary_source.boundary_chart_cover()
    assert cover.complete
    assert cover.semantics == "sampled"


def test_parametric_surface_route_adapts_exact_reconstructed_meshregion() -> None:
    coordinates = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    faces = np.asarray(
        ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)),
        dtype=np.int64,
    )
    region = MeshRegion(
        jnp.asarray(coordinates),
        jnp.asarray(faces),
        feature_id="reconstructed-tetra-boundary",
    )
    report = ReconstructionReport(
        source_kind="exact_mesh_region",
        algorithm="represented_triangle_mesh",
        input_digest="test-input-digest",
        input_points=coordinates.shape[0],
        retained_points=coordinates.shape[0],
        output_vertices=coordinates.shape[0],
        output_cells=faces.shape[0],
        connected_components=1,
        watertight=True,
        winding_consistent=True,
        recenter_offset=(0.0, 0.0, 0.0),
        parameters=(),
    )
    source = ReconstructedGeometrySource(region, report)
    domain = source.meshing_domain
    assert domain is not None
    specification = _surface_specification(domain)
    construction = prepare_native_surface(
        source,
        specification,
        NativeMeshingOptions(
            "parametric_surface",
            surface_schedule=NativeSurfaceSchedule(spacing_fraction=0.6),
        ),
        SpatialCoordinateContract.si(),
    ).execute()

    assert construction.mesh.topological_dimension == 2
    assert construction.organization.surface_source is not None
    assert construction.organization.surface_source.domain.domain_id == domain.domain_id
    actual = np.asarray(construction.mesh.coordinates, dtype=np.float64)
    vertex_association = next(
        association
        for association in construction.organization.associations
        if association.target_entity_set_id
        == construction.mesh.entity_set(0).entity_set_id
    )
    vertex_rows = {
        int(identifier): row
        for row, identifier in enumerate(construction.mesh.vertex_global_ids.tolist())
    }
    vertex_dimensions = np.asarray(vertex_association.source_dimensions, dtype=np.int64)
    source_entity_ids = vertex_association.source_entity_ids
    source_occurrences = vertex_association.source_occurrence_paths
    vertex_parameters = np.asarray(vertex_association.parameters, dtype=np.float64)
    for association_row, identifier in enumerate(
        np.asarray(vertex_association.target_global_ids, dtype=np.int64).tolist()
    ):
        dimension = int(vertex_dimensions[association_row])
        source_entity_id = source_entity_ids[association_row]
        occurrence = source_occurrences[association_row]
        source_row = domain.resolve_entity_id(
            dimension, source_entity_id, occurrence_path=occurrence
        )
        assert domain.entity_id(dimension, source_row) == source_entity_id
        assert domain.source_occurrences[dimension][source_row] == occurrence
        if dimension == 0:
            expected_point = domain.corner_points[source_row]
        elif dimension == 1:
            expected_point = _curve_point(
                domain, source_row, vertex_parameters[association_row, 0]
            )
        else:
            expected_point = domain.evaluate(
                np.asarray((source_row,), dtype=np.int64),
                vertex_parameters[association_row, :2][None, :],
            )[0]
        np.testing.assert_allclose(
            actual[vertex_rows[int(identifier)]], expected_point, atol=2e-12
        )
    face_association = next(
        association
        for association in construction.organization.associations
        if (
            association.target_entity_set_id
            == construction.mesh.entity_set(2).entity_set_id
        )
    )
    assert set(face_association.source_entity_ids) == {
        domain.entity_id(2, row) for row in range(len(domain.patches))
    }


def test_parametric_surface_route_meshes_shared_curved_nurbs_trims() -> None:
    domain = _curved_trim_domain()
    specification = _surface_specification(domain, maximum_deviation=0.04)
    construction = prepare_native_surface(
        domain,
        specification,
        NativeMeshingOptions(
            "parametric_surface",
            surface_schedule=NativeSurfaceSchedule(
                spacing_fraction=0.6,
                minimum_edge_samples=2,
                curve_schedule=NativeCurveSchedule(fidelity_samples=5),
            ),
        ),
        SpatialCoordinateContract.si(),
    ).execute()

    assert construction.compliance.passed
    source_charts = construction.organization.surface_source
    assert source_charts is not None
    source_cover = source_charts.boundary_source.boundary_chart_cover()
    assert source_cover.complete
    assert source_cover.semantics == "certified"
    _assert_vertices_on_domain(domain, construction)
    validity = certify_cell_geometry_validity(construction.mesh)
    assert validity.invalid_count == 0
    assert validity.unresolved_count == 0
    assert validity.certified_valid_count == construction.mesh.entity_set(2).count

    edge_association = next(
        association
        for association in construction.organization.associations
        if association.target_entity_set_id
        == construction.mesh.entity_set(1).entity_set_id
    )
    intervals = edge_association.source_parameter_intervals
    assert intervals is not None
    bottom_id = domain.entity_id(1, 0)
    bottom_rows = [
        row
        for row, source_id in enumerate(edge_association.source_entity_ids)
        if source_id == bottom_id
    ]
    raw_bottom_intervals = tuple(intervals[row] for row in bottom_rows)
    assert all(interval is not None for interval in raw_bottom_intervals)
    bottom_intervals = sorted(
        interval for interval in raw_bottom_intervals if interval is not None
    )
    assert len(bottom_intervals) >= 2
    assert bottom_intervals[0][0] == 0
    assert bottom_intervals[-1][1] == 1
    assert all(
        first[1] == second[0]
        for first, second in zip(bottom_intervals[:-1], bottom_intervals[1:], strict=True)
    )
    connectivity = construction.mesh.connectivity
    assert isinstance(connectivity, PolygonalConnectivity)
    edge_by_id = {
        int(identifier): tuple(int(value) for value in vertices)
        for identifier, vertices in zip(
            construction.mesh.entity_set(1).entity_ids,
            connectivity.edges,
            strict=True,
        )
    }
    coordinates = np.asarray(construction.mesh.coordinates, dtype=np.float64)
    for row in bottom_rows:
        interval = intervals[row]
        assert interval is not None
        expected = np.asarray(
            (
                _curve_point(domain, 0, float(interval[0])),
                _curve_point(domain, 0, float(interval[1])),
            ),
            dtype=np.float64,
        )
        actual = coordinates[
            np.asarray(
                edge_by_id[int(edge_association.target_global_ids[row])],
                dtype=np.int64,
            )
        ]
        assert (
            min(
                np.max(np.linalg.norm(actual - expected, axis=1)),
                np.max(np.linalg.norm(actual[::-1] - expected, axis=1)),
            )
            < 2e-12
        )

    seam_id = domain.entity_id(1, 1)
    seam_edge_ids = tuple(
        int(edge_association.target_global_ids[row])
        for row, source_id in enumerate(edge_association.source_entity_ids)
        if source_id == seam_id
    )
    assert seam_edge_ids
    assert len(set(seam_edge_ids)) == len(seam_edge_ids)

    parameters = []
    patch_rows = []
    for patch, lower_v in ((0, 0.2), (1, 0.05)):
        u = np.linspace(0.0, 1.0, 15, dtype=np.float64)
        v = np.linspace(lower_v, 0.95, 11, dtype=np.float64)
        u_grid, v_grid = np.meshgrid(u, v, indexing="ij")
        parameters.append(np.stack((u_grid, v_grid), axis=-1).reshape((-1, 2)))
        patch_rows.extend([patch] * u_grid.size)
    source_points = domain.evaluate(
        np.asarray(patch_rows, dtype=np.int64),
        np.concatenate(parameters, axis=0),
    )
    faces = np.concatenate(
        tuple(
            np.asarray(block.vertices, dtype=np.int64)
            for block in construction.mesh.blocks
        )
    )
    physical_triangles = coordinates[faces]
    sampled_distance = max(
        _point_mesh_distance(point, physical_triangles)
        for point in np.asarray(source_points, dtype=np.float64)
    )
    assert sampled_distance <= 0.041


@pytest.mark.parametrize("shape", ("sphere", "cone"), ids=("sphere-poles", "cone-apex"))
def test_parametric_surface_route_meshes_collapsed_poles(shape: str) -> None:
    contract = SpatialCoordinateContract.si()
    source = (
        brep_sphere(1.0, coordinate_contract=contract)
        if shape == "sphere"
        else brep_cone(1.0, 0.0, 1.0, coordinate_contract=contract)
    )
    domain = MeshingDomain.from_brep(source)
    specification = _surface_specification(domain, maximum_deviation=0.06)
    construction = prepare_native_surface(
        domain,
        specification,
        NativeMeshingOptions(
            "parametric_surface",
            surface_schedule=NativeSurfaceSchedule(
                spacing_fraction=0.65,
                minimum_edge_samples=3,
            ),
        ),
        contract,
    ).execute()

    assert construction.compliance.passed
    _assert_vertices_on_domain(domain, construction)
    validity = certify_cell_geometry_validity(construction.mesh)
    assert validity.invalid_count == 0
    assert validity.unresolved_count == 0
    assert validity.certified_valid_count == construction.mesh.entity_set(2).count
    coordinates = np.asarray(construction.mesh.coordinates, dtype=np.float64)
    faces = np.concatenate(
        tuple(
            np.asarray(block.vertices, dtype=np.int64)
            for block in construction.mesh.blocks
        )
    )
    triangles = coordinates[faces]
    areas = np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        axis=1,
    )
    assert np.all(areas > 0.0)
    assert _source_to_mesh_sampled_distance(domain, construction) <= 0.061

    centers = np.mean(triangles, axis=1)
    if shape == "sphere":
        source_distances = np.abs(np.linalg.norm(centers, axis=1) - 1.0)
    else:
        source_distances = np.empty((centers.shape[0],), dtype=np.float64)
        for row in range(centers.shape[0]):
            point = np.asarray(centers[row], dtype=np.float64).reshape((3,))
            radial = float(np.linalg.norm(point[:2]))
            side = _point_segment_distance(
                np.asarray((point[2], radial), dtype=np.float64),
                np.asarray((0.0, 1.0), dtype=np.float64),
                np.asarray((1.0, 0.0), dtype=np.float64),
            )
            base = float(np.hypot(point[2], max(radial - 1.0, 0.0)))
            source_distances[row] = min(side, base)
    assert float(np.max(source_distances, initial=0.0)) <= 0.061


def test_parametric_surface_route_grades_planar_disk_poles_without_thin_bands() -> None:
    # A cylinder cap is a revolution of a radial line, so its polar chart
    # never becomes isotropic and graded rings run up to their extent. A ring
    # placed just inside the extent left a thin last band whose short radial gap
    # forced a dense ring beside the coarse chart beyond, i.e. cap slivers of
    # hundredths of a degree that no volume refinement can repair.
    contract = SpatialCoordinateContract.si()
    domain = MeshingDomain.from_brep(
        brep_cylinder(0.5, 1.0, coordinate_contract=contract)
    )
    construction = prepare_native_surface(
        domain,
        _surface_specification(domain, maximum_deviation=0.05),
        NativeMeshingOptions("parametric_surface"),
        contract,
    ).execute()

    assert construction.compliance.passed
    coordinates = np.asarray(construction.mesh.coordinates, dtype=np.float64)
    triangles = coordinates[
        np.concatenate(
            tuple(
                np.asarray(block.vertices, dtype=np.int64)
                for block in construction.mesh.blocks
            )
        )
    ]
    caps = np.all(triangles[:, :, 2] == triangles[:, :1, 2], axis=1)
    assert np.any(caps)
    corners = triangles[caps]
    angles = []
    for corner in range(3):
        first = corners[:, (corner + 1) % 3] - corners[:, corner]
        second = corners[:, (corner + 2) % 3] - corners[:, corner]
        cosine = np.sum(first * second, axis=1) / (
            np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
        )
        angles.append(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    assert float(np.min(angles)) >= 10.0


def test_parametric_surface_route_flips_anisotropic_chart_cells_to_physical_shape() -> (
    None
):
    # A sphere chart is anisotropic away from its equator: a chart triangle of
    # good chart angles near the seam at latitude 62 degrees mapped to a physical
    # cell of 12.5 degrees, beyond the default tetrahedral radius-edge bound 2 on
    # a fixed boundary face. Interior chart edges flip to the physically better
    # diagonal before each refinement round.
    contract = SpatialCoordinateContract.si()
    domain = MeshingDomain.from_brep(brep_sphere(1.0, coordinate_contract=contract))
    scope = MeshingScope(
        domain.source_id,
        domain.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        domain.scope_entity_set_id(2),
        domain.scope_entity_ids(2),
    )
    construction = prepare_native_surface(
        domain,
        SurfaceMeshingSpec(
            CellMeshingTarget(2, 3, CellFamilyPolicy(required=("triangle",))),
            scope,
            size_controls=(
                UniformSizeControl(scope, 0.7, strength=SizeControlStrength.SOFT),
            ),
            maximum_deviation=0.08,
        ),
        NativeMeshingOptions("parametric_surface"),
        contract,
    ).execute()

    assert construction.compliance.passed
    coordinates = np.asarray(construction.mesh.coordinates, dtype=np.float64)
    corners = coordinates[
        np.concatenate(
            tuple(
                np.asarray(block.vertices, dtype=np.int64)
                for block in construction.mesh.blocks
            )
        )
    ]
    lengths = np.stack(
        tuple(
            np.linalg.norm(corners[:, (corner + 1) % 3] - corners[:, corner], axis=1)
            for corner in range(3)
        ),
        axis=1,
    )
    areas = 0.5 * np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1
    )
    circumradii = np.prod(lengths, axis=1) / (4.0 * areas)
    assert float(np.max(circumradii / np.min(lengths, axis=1))) <= 2.0


def test_parametric_surface_route_samples_seams_at_the_curvature_rate() -> None:
    # A first-order chord bound samples a curved seam down to edges near the
    # deviation tolerance itself and leaves needle cells along the seam. The
    # chord deviation of a C2 seam decays quadratically with its interval, so
    # seam edges stay near sqrt(8 R tolerance) and the cells stay well shaped.
    contract = SpatialCoordinateContract.si()
    domain = MeshingDomain.from_brep(brep_sphere(0.5, coordinate_contract=contract))
    construction = prepare_native_surface(
        domain,
        _surface_specification(domain, maximum_deviation=0.005),
        NativeMeshingOptions("parametric_surface"),
        contract,
    ).execute()

    assert construction.compliance.passed
    coordinates = np.asarray(construction.mesh.coordinates, dtype=np.float64)
    triangles = coordinates[
        np.concatenate(
            tuple(
                np.asarray(block.vertices, dtype=np.int64)
                for block in construction.mesh.blocks
            )
        )
    ]
    lengths = np.stack(
        tuple(
            np.linalg.norm(triangles[:, (side + 1) % 3] - triangles[:, side], axis=1)
            for side in range(3)
        ),
        axis=1,
    )
    areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    quality = 2.0 * np.sqrt(3.0) * areas / np.sum(lengths**2, axis=1)
    assert float(np.min(quality)) >= 0.2
    assert float(np.max(np.max(lengths, axis=1) / np.min(lengths, axis=1))) <= 5.0
