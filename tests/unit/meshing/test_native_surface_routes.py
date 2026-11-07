#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization._cell_complex import PolygonalConnectivity
from phydrax.discretization._cell_geometry import ScalarCoordinateElement
from phydrax.discretization._cell_geometry_validity import (
    certify_cell_geometry_validity,
)
from phydrax.discretization._coordinate_enclosure import ExactScalar
from phydrax.discretization._surface_chart_records import SurfaceChartWitness
from phydrax.geometry._curve_boundary import (
    _endpoint_interval,
    ParametricCurveBoundarySource,
)
from phydrax.geometry._domain_boundary import MeshingDomainBoundarySource
from phydrax.geometry._meshing_domain import (
    MeshingDomain,
    MeshingDomainCurve,
    MeshingDomainRegion,
    MeshingSourceAccuracy,
    MeshingSurfacePatch,
    PatchCurveUse,
    SourcePreparation,
)
from phydrax.geometry._source_boundary_records import (
    SourceBoundaryChartWitness,
    SourceBoundaryPointWitness,
)
from phydrax.geometry._source_fidelity import (
    _surface_jacobian_bounds,
    certify_source_fidelity,
    validate_source_fidelity_witness,
)
from phydrax.geometry.brep import brep_cone, brep_cylinder, brep_sphere
from phydrax.geometry.brep._patches import (
    BSplineCurve,
    BSplineSurfacePatch,
    LineCurve,
    PlanePatch,
)
from phydrax.geometry.reconstruction._core import (
    ReconstructedGeometrySource,
    ReconstructionReport,
)
from phydrax.geometry.simplicial._regions import MeshRegion
from phydrax.geometry.surface import SurfaceModel
from phydrax.meshing._association import GeometryAssociationKind, GeometrySourceEntityRole
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
from phydrax.meshing.providers._native_surface import (
    _domain_curve_source_query,
    prepare_native_surface,
)


def test_curve_interval_expands_intersected_inward_root_bracket() -> None:
    brackets = ((0.0, 0.0), (0.875, 1.0))
    assert _endpoint_interval(0.9, 0.95, brackets) == (0.875, 1.0)
    assert _endpoint_interval(0.5, 0.75, brackets) == (0.5, 0.75)


def test_capped_native_curve_endpoint_source_and_tampering() -> None:
    domain = MeshingDomain.from_brep(
        brep_cylinder(
            0.8,
            1.5,
            coordinate_contract=SpatialCoordinateContract.si(),
            source_id="capped-curve-endpoint",
        )
    )
    physical = {
        use.curve
        for patch in domain.patches
        for loop in patch.loops
        for use in loop
        if isinstance(use, PatchCurveUse)
    }
    selected_ids = []
    for identifier in domain.scope_entity_ids(1):
        entity, occurrence = domain.source_entity_key_for_scope(1, int(identifier))
        if domain.resolve_entity_id(1, entity, occurrence_path=occurrence) in physical:
            selected_ids.append(int(identifier))
    selected = np.asarray(selected_ids, dtype=np.int64)
    query = _domain_curve_source_query(domain, selected)
    domain.validate_curve_boundary_query(query, selected)
    bracket_row, bracket_side = next(
        (row, side)
        for row, pair in enumerate(query.endpoint_parameter_bounds)
        for side, (lower, upper) in enumerate(pair)
        if lower < upper
    )
    bracket = query.endpoint_parameter_bounds[bracket_row][bracket_side]
    assert bracket[1] > query.parameter_ranges[bracket_row][bracket_side]
    shadow = query.parameter_ranges[bracket_row][bracket_side]
    endpoint_box = query.chart_point_bounds(query.chart_ids[bracket_row], shadow)
    expected_box = query.curves[bracket_row].bounding_box(*bracket)
    assert np.array_equal(endpoint_box[0], expected_box[0])
    assert np.array_equal(endpoint_box[1], expected_box[1])
    assert query.boundary_cover(0.5).complete
    from phydrax._model._structure import model_archive_projection
    from phydrax.meshing._model_codec import restore_model_archive_value

    recipe = model_archive_projection(query)
    restored = restore_model_archive_value(recipe)
    restored.require_current()
    assert restored.query_id == query.query_id
    assert restored.endpoint_parameter_bounds == query.endpoint_parameter_bounds
    mismatches = []
    for row, (curve, parameters, bounds) in enumerate(
        zip(
            query.curves, query.parameter_ranges, query.endpoint_point_bounds, strict=True
        )
    ):
        for side, (parameter, enclosure) in enumerate(
            zip(parameters, bounds, strict=True)
        ):
            assert enclosure is not None
            value = np.asarray(curve.evaluate(jnp.asarray(parameter, dtype=jnp.float64)))
            if np.any(value < enclosure[0]) or np.any(value > enclosure[1]):
                mismatches.append((row, side, value))
        midpoint = (parameters[0] + parameters[1]) / 2
        lower, upper = query.chart_point_bounds(query.chart_ids[row], midpoint)
        value = np.asarray(curve.evaluate(jnp.asarray(midpoint, dtype=jnp.float64)))
        assert np.all(lower <= value) and np.all(value <= upper)
    assert mismatches
    row, side, value = mismatches[0]

    def rebuild(
        *,
        parameter_ranges: tuple[tuple[float, float], ...] | None = None,
        endpoint_source_keys: tuple[
            tuple[tuple[str, tuple[str, ...]] | None, tuple[str, tuple[str, ...]] | None],
            ...,
        ]
        | None = None,
        endpoint_point_bounds: tuple[
            tuple[tuple[tuple[float, ...], tuple[float, ...]] | None, ...], ...
        ]
        | None = None,
        endpoint_parameter_bounds: tuple[
            tuple[tuple[float, float], tuple[float, float]], ...
        ]
        | None = None,
    ) -> ParametricCurveBoundarySource:
        return ParametricCurveBoundarySource(
            query.curves,
            query.parameter_ranges if parameter_ranges is None else parameter_ranges,
            source_id=query.source_id,
            source_revision=query.source_revision,
            curve_entity_ids=query.curve_entity_ids,
            occurrence_paths=query.occurrence_paths,
            chart_ids=query.chart_ids,
            endpoint_source_keys=query.endpoint_source_keys
            if endpoint_source_keys is None
            else endpoint_source_keys,
            endpoint_point_bounds=query.endpoint_point_bounds
            if endpoint_point_bounds is None
            else endpoint_point_bounds,
            endpoint_parameter_bounds=query.endpoint_parameter_bounds
            if endpoint_parameter_bounds is None
            else endpoint_parameter_bounds,
        )

    bounds = [list(pair) for pair in query.endpoint_point_bounds]
    bounds[row][side] = (tuple(value), tuple(value))
    with pytest.raises(ValueError):
        forged = rebuild(endpoint_point_bounds=tuple(tuple(pair) for pair in bounds))
        domain.validate_curve_boundary_query(forged, selected)

    bounds[row][side] = ((100.0,) * 3, (100.0,) * 3)
    with pytest.raises(ValueError):
        rebuild(endpoint_point_bounds=tuple(tuple(pair) for pair in bounds))

    ranges = list(query.parameter_ranges)
    ranges[row] = (ranges[row][0], (ranges[row][0] + ranges[row][1]) / 2)
    with pytest.raises(ValueError):
        forged = rebuild(parameter_ranges=tuple(ranges))
        domain.validate_curve_boundary_query(forged, selected)

    keys = [list(pair) for pair in query.endpoint_source_keys]
    key = keys[row][side]
    assert key is not None
    keys[row][side] = (key[0], key[1] + ("forged-occurrence",))
    with pytest.raises(ValueError):
        forged = rebuild(endpoint_source_keys=tuple((pair[0], pair[1]) for pair in keys))
        domain.validate_curve_boundary_query(forged, selected)

    brackets = [list(pair) for pair in query.endpoint_parameter_bounds]
    shadow = query.parameter_ranges[bracket_row][bracket_side]
    brackets[bracket_row][bracket_side] = (shadow, shadow)
    with pytest.raises(ValueError):
        forged = rebuild(
            endpoint_parameter_bounds=tuple((pair[0], pair[1]) for pair in brackets)
        )
        domain.validate_curve_boundary_query(forged, selected)
    with pytest.raises(ValueError):
        restore_model_archive_value(
            recipe._replace(
                endpoint_parameter_bounds=tuple((pair[0], pair[1]) for pair in brackets)
            )
        )


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
            parameter = parameters[association_row, 0]
            if vertex_association.association_kind is GeometryAssociationKind.BREP:
                patch, loop, use_row = (
                    int(value) for value in domain.curve_owners[source_row]
                )
                use = domain.patches[patch].loops[loop][use_row]
                uv = np.asarray(
                    use.pcurve.evaluate(jnp.asarray(parameter)), dtype=np.float64
                )
                expected = np.asarray(
                    domain.patches[patch].surface.evaluate(jnp.asarray(uv)),
                    dtype=np.float64,
                )
            else:
                expected = _curve_point(domain, source_row, parameter)
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


def test_exact_planar_surface_proves_zero_without_inventing_a_control() -> None:
    from phydrax.geometry.brep import brep_box
    from phydrax.meshing import NativeMeshingProvider

    contract = SpatialCoordinateContract.si()
    model = brep_box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), coordinate_contract=contract)
    scope = MeshingScope.whole(model, 2)
    specification = SurfaceMeshingSpec(
        CellMeshingTarget(2, 3, CellFamilyPolicy(required=("triangle",))),
        scope,
        size_controls=(
            UniformSizeControl(scope, 0.8, strength=SizeControlStrength.SOFT),
        ),
        maximum_deviation=None,
    )
    result = (
        NativeMeshingProvider(NativeMeshingOptions("parametric_surface"))
        .plan(model, specification, coordinate_contract=contract)
        .execute()
    )
    assert specification.maximum_deviation is None
    assert not any(name == "maximum_deviation" for name, _ in result.compliance.requested)
    report = result.certification
    assert report is not None and report.fidelity is not None
    assert report.inputs.fidelity_tolerance == 0.0
    assert report.fidelity.status == "certified"
    assert report.fidelity.mesh_to_source_upper == 0.0
    assert report.fidelity.source_to_mesh_upper == 0.0

    from phydrax.discretization._surface_chart_records import _exact_chart_from_tokens
    from phydrax.geometry._source_fidelity import (
        _exact_affine_target_charts,
        _exact_triangle2,
        _partition_findings,
    )
    from phydrax.geometry._source_interpolation import (
        exact_surface_coordinate_expressions,
    )

    witness = report.inputs.fidelity_witness
    source = report.inputs.source
    assert isinstance(witness, SurfaceChartWitness)
    assert isinstance(source, MeshingDomainBoundarySource)
    ids = tuple(int(value) for value in np.asarray(witness.cell_global_ids))
    patches = tuple(int(value) for value in np.asarray(witness.patches))
    original = tuple(
        _exact_triangle2(_exact_chart_from_tokens(witness.chart_rational_tokens, row))
        for row in range(len(ids))
    )
    positions = np.asarray(result.mesh.coordinates, dtype=np.float64)
    target: dict[int, tuple[tuple[ExactScalar, ...], ...]] = {
        int(cell_id): tuple(
            tuple(Fraction.from_float(float(value)) for value in positions[int(vertex)])
            for vertex in vertices
        )
        for block in result.mesh.blocks
        for cell_id, vertices in zip(block.global_ids, block.vertices, strict=True)
    }
    actual = _exact_affine_target_charts(source, ids, patches, original, target, set(ids))
    assert actual is not None
    assert not _partition_findings(
        source, witness, SourcePreparation(source.domain), exact_target_charts=actual
    )
    frame = exact_surface_coordinate_expressions(
        source.domain.patches[patches[0]].surface
    )
    assert frame is not None
    first = tuple(expression.get((1, 0), Fraction(0)) for expression in frame)
    second = tuple(expression.get((0, 1), Fraction(0)) for expression in frame)
    normal = (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )
    axis = next(index for index, value in enumerate(normal) if value != 0)
    changed = dict(target)
    triangle = [list(point) for point in changed[ids[0]]]
    triangle[0][axis] += Fraction(1, 1000)
    changed[ids[0]] = tuple(tuple(point) for point in triangle)
    assert (
        _exact_affine_target_charts(source, ids, patches, original, changed, set(ids))
        is None
    )
    same_patch = next(
        index for index in range(1, len(ids)) if patches[index] == patches[0]
    )
    overlapping = list(actual)
    overlapping[same_patch] = actual[0]
    assert _partition_findings(
        source,
        witness,
        SourcePreparation(source.domain),
        exact_target_charts=tuple(overlapping),
    )


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
    feature = ProtectedFeature(base.scope, FeatureKind.SURFACE, maximum_deviation=0.003)
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


def test_reconstructed_triangle_chart_box_encloses_exact_affine_image() -> None:
    from phydrax.geometry.reconstruction._core import _ReconstructedTrianglePatch

    corners = np.asarray(
        ((0.1, -0.3, 0.7), (1.3, 0.2, -0.1), (-0.4, 0.9, 0.3)), dtype=np.float64
    )
    patch = _ReconstructedTrianglePatch(corners)
    parameter_box = np.asarray(((0.1, 0.2), (0.3, 0.7)), dtype=np.float64)
    box = patch.bounding_box(parameter_box)
    exact = tuple(
        tuple(Fraction.from_float(float(value)) for value in row) for row in corners
    )
    images = tuple(
        tuple(
            exact[0][axis]
            + u * (exact[1][axis] - exact[0][axis])
            + v * (exact[2][axis] - exact[0][axis])
            for axis in range(3)
        )
        for u in (Fraction.from_float(0.1), Fraction.from_float(0.3))
        for v in (Fraction.from_float(0.2), Fraction.from_float(0.7))
    )
    for axis in range(3):
        lowest = min(image[axis] for image in images)
        highest = max(image[axis] for image in images)
        assert Fraction.from_float(float(box[0, axis])) <= lowest
        assert highest <= Fraction.from_float(float(box[1, axis]))
        # Outward rounding of the exact extremes, never a wider sampled box.
        assert float(box[0, axis]) >= np.nextafter(float(lowest), -np.inf)
        assert float(box[1, axis]) <= np.nextafter(float(highest), np.inf)


def test_rational_surface_jacobian_uses_positive_source_spans() -> None:
    domain = _curved_trim_domain()
    surface = domain.patches[0].surface
    preparation = SourcePreparation(domain)
    triangle = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )
    bounds = _surface_jacobian_bounds(surface, triangle, preparation)
    assert bounds is not None
    lower, upper = bounds
    exact = np.asarray(((0.5, 0.0), (0.0, 1.0), (0.0, 0.0)))
    assert np.all(lower <= exact)
    assert np.all(exact <= upper)
    outside = ((Fraction(-1), Fraction(0)), *triangle[1:])
    assert _surface_jacobian_bounds(surface, outside, preparation) is None


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
    from phydrax.geometry._surface_source_support import (
        _cad_root_edges,
        validate_cad_curve_ancestry,
    )

    charts = construction.organization.surface_source
    assert charts is not None
    bindings = validate_cad_curve_ancestry(charts, 0.05)
    edges, _ = _cad_root_edges(charts)
    vertex_table = next(
        value
        for value in charts.associations
        if value.target_entity_set_id == charts.mesh.entity_set(0).entity_set_id
    )
    vertex_rows = {
        int(identifier): row
        for row, identifier in enumerate(vertex_table.target_global_ids)
    }
    vertex_ids = np.asarray(charts.mesh.vertex_global_ids, dtype=np.int64)
    native_checked = 0
    for edge_id, uses in bindings.items():
        endpoints = edges[edge_id]
        for binding in uses:
            for vertex, parameter in zip(endpoints, binding.parameters, strict=True):
                if parameter in (0, 1):
                    continue
                expected = domain.curve_native_parameter(binding.curve_row, parameter)
                patch, loop, use = (
                    int(v) for v in domain.curve_owners[binding.curve_row]
                )
                owner = domain.patches[patch].loops[loop][use]
                if max(abs(owner.first), abs(owner.last)) <= 1.0:
                    continue
                row = vertex_rows[int(vertex_ids[vertex])]
                assert float(vertex_table.parameters[row, 0]) == float(expected)
                native_checked += 1
    assert native_checked > 0

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


def test_cad_surface_curve_ancestry_replays_root_edge_chains() -> None:
    from phydrax.geometry._surface_source_support import (
        SurfaceSourceCharts,
        validate_cad_curve_ancestry,
    )
    from phydrax.geometry.brep import brep_box

    contract = SpatialCoordinateContract.si()
    model = brep_box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), coordinate_contract=contract)
    domain = MeshingDomain.from_brep(model)
    scope = MeshingScope.whole(model, 2)
    specification = SurfaceMeshingSpec(
        CellMeshingTarget(2, 3, CellFamilyPolicy(required=("triangle",))),
        scope,
        size_controls=(
            UniformSizeControl(scope, 0.7, strength=SizeControlStrength.SOFT),
        ),
        maximum_deviation=0.1,
    )
    construction = prepare_native_surface(
        domain, specification, NativeMeshingOptions("parametric_surface"), contract
    ).execute()
    charts = construction.organization.surface_source
    assert isinstance(charts, SurfaceSourceCharts)
    bindings = validate_cad_curve_ancestry(charts, 0.1)
    assert {binding.curve_row for rows in bindings.values() for binding in rows} == set(
        range(len(domain.curves))
    )
    forged_occurrences = np.array(charts.boundary_occurrences, copy=True)
    row, slot = np.argwhere(forged_occurrences[:, :, 0] >= 0)[0]
    patch, loop, use = (int(value) for value in forged_occurrences[row, slot])
    forged_occurrences[row, slot, 2] = (use + 1) % len(domain.patches[patch].loops[loop])
    forged = SurfaceSourceCharts(
        domain,
        charts.mesh,
        charts.geometry,
        charts.coordinate_contract,
        charts.root_parameters,
        charts.root_patches,
        charts.associations,
        charts.boundary_source,
        source_root_cell_ids=charts.source_root_cell_ids,
        source_piece_ids=charts.source_piece_ids,
        root_parameter_exact=charts.root_parameter_exact,
        boundary_occurrences=forged_occurrences,
    )
    with pytest.raises(ValueError, match="coedge occurrence"):
        validate_cad_curve_ancestry(forged, 0.1)


def test_surface_curve_ancestry_replays_chart_and_coedge_authority() -> None:
    from phydrax.geometry._surface_source_support import (
        SurfaceSourceCharts,
        validate_surface_curve_ancestry,
    )

    domain = _square_domain(authored_corners=True)
    construction = prepare_native_surface(
        domain,
        _surface_specification(domain),
        NativeMeshingOptions("parametric_surface"),
        SpatialCoordinateContract.si(),
    ).execute()
    charts = construction.organization.surface_source
    assert isinstance(charts, SurfaceSourceCharts)
    bindings = validate_surface_curve_ancestry(charts, 0.05)
    assert {use.curve_row for uses in bindings.values() for use in uses} == set(
        range(len(domain.curves))
    )
    occurrences = np.array(charts.boundary_occurrences, copy=True)
    row, slot = np.argwhere(occurrences[:, :, 0] >= 0)[0]
    patch, loop, use = (int(value) for value in occurrences[row, slot])
    occurrences[row, slot, 2] = (use + 1) % len(domain.patches[patch].loops[loop])
    forged = SurfaceSourceCharts(
        domain,
        charts.mesh,
        charts.geometry,
        charts.coordinate_contract,
        charts.root_parameters,
        charts.root_patches,
        charts.associations,
        charts.boundary_source,
        source_root_cell_ids=charts.source_root_cell_ids,
        source_piece_ids=charts.source_piece_ids,
        root_parameter_exact=charts.root_parameter_exact,
        boundary_occurrences=occurrences,
        trim_roots=charts.trim_roots,
    )
    with pytest.raises(ValueError, match="coedge occurrence"):
        validate_surface_curve_ancestry(forged, 0.05)

    from phydrax.meshing._association import GeometryAssociation

    def replace_table(
        original: GeometryAssociation,
        *,
        parameters: np.ndarray | None = None,
        intervals: tuple[tuple[Fraction, Fraction] | None, ...] | None = None,
        dimensions: np.ndarray | None = None,
        indices: np.ndarray | None = None,
        roles: tuple[GeometrySourceEntityRole | None, ...] | None = None,
        source_ids: tuple[str, ...] | None = None,
        paths: tuple[tuple[str, ...], ...] | None = None,
    ) -> GeometryAssociation:
        return GeometryAssociation(
            original.association_kind,
            original.source_id,
            original.source_revision,
            original.target_entity_set_id,
            original.target_global_ids,
            original.source_entity_ids if source_ids is None else source_ids,
            original.residuals,
            resolved=original.resolved,
            ambiguous=original.ambiguous,
            exact=original.exact,
            source_dimensions=original.source_dimensions
            if dimensions is None
            else dimensions,
            source_indices=original.source_indices if indices is None else indices,
            source_roles=original.source_roles if roles is None else roles,
            source_occurrence_paths=original.source_occurrence_paths
            if paths is None
            else paths,
            source_region_sides=original.source_region_sides,
            source_parameter_intervals=original.source_parameter_intervals
            if intervals is None
            else intervals,
            source_parameter_points=original.source_parameter_points,
            parameters=original.parameters if parameters is None else parameters,
            orientations=original.orientations,
            parent_dimensions=original.parent_dimensions,
            parent_ids=original.parent_ids,
            parent_association_id=original.parent_association_id,
            provenance=original.provenance,
        )

    def rebind(
        original: GeometryAssociation, replacement: GeometryAssociation
    ) -> SurfaceSourceCharts:
        return SurfaceSourceCharts(
            domain,
            charts.mesh,
            charts.geometry,
            charts.coordinate_contract,
            charts.root_parameters,
            charts.root_patches,
            tuple(
                replacement if value is original else value
                for value in charts.associations
            ),
            charts.boundary_source,
            source_root_cell_ids=charts.source_root_cell_ids,
            source_piece_ids=charts.source_piece_ids,
            root_parameter_exact=charts.root_parameter_exact,
            boundary_occurrences=charts.boundary_occurrences,
            trim_roots=charts.trim_roots,
        )

    vertex_table = next(
        value
        for value in charts.associations
        if value.association_kind is GeometryAssociationKind.SURFACE
        and value.target_entity_set_id == charts.mesh.entity_set(0).entity_set_id
    )
    parameters = np.asarray(vertex_table.parameters).copy()
    curve_vertex = np.flatnonzero(np.asarray(vertex_table.source_dimensions) == 1)[0]
    parameters[curve_vertex, 0] += 0.01
    with pytest.raises(ValueError, match="source parameter"):
        validate_surface_curve_ancestry(
            rebind(vertex_table, replace_table(vertex_table, parameters=parameters)), 0.05
        )
    dimensions = np.asarray(vertex_table.source_dimensions).copy()
    indices = np.asarray(vertex_table.source_indices).copy()
    roles = list(vertex_table.source_roles)
    source_ids = list(vertex_table.source_entity_ids)
    paths = list(vertex_table.source_occurrence_paths)
    dimensions[curve_vertex] = 2
    indices[curve_vertex] = domain.source_indices[2][0]
    roles[curve_vertex] = GeometrySourceEntityRole.FACET
    source_ids[curve_vertex] = domain.entity_id(2, 0)
    paths[curve_vertex] = domain.source_occurrences[2][0]
    with pytest.raises(ValueError, match="source occurrence"):
        validate_surface_curve_ancestry(
            rebind(
                vertex_table,
                replace_table(
                    vertex_table,
                    dimensions=dimensions,
                    indices=indices,
                    roles=tuple(roles),
                    source_ids=tuple(source_ids),
                    paths=tuple(paths),
                ),
            ),
            0.05,
        )
    edge_table = next(
        value
        for value in charts.associations
        if value.association_kind is GeometryAssociationKind.CURVE
    )
    intervals = list(edge_table.source_parameter_intervals)
    assert intervals[0] is not None
    first, last = intervals[0]
    intervals[0] = (first, (first + last) / 2)
    with pytest.raises(ValueError):
        validate_surface_curve_ancestry(
            rebind(edge_table, replace_table(edge_table, intervals=tuple(intervals))),
            0.05,
        )
    from phydrax.geometry._surface_source_support import _cad_root_edges

    original_intervals = edge_table.source_parameter_intervals
    first_row, second_row = next(
        (i, j)
        for i, left in enumerate(original_intervals)
        for j, right in enumerate(original_intervals)
        if left is not None
        and right is not None
        and edge_table.source_entity_ids[i] == edge_table.source_entity_ids[j]
        and edge_table.source_occurrence_paths[i] == edge_table.source_occurrence_paths[j]
        and left[1] == right[0]
    )
    left, right = original_intervals[first_row], original_intervals[second_row]
    assert left is not None and right is not None
    shifted = left[1] + (right[1] - left[1]) / 100
    altered = list(original_intervals)
    altered[first_row] = (left[0], shifted)
    altered[second_row] = (shifted, right[1])
    edges, _ = _cad_root_edges(charts)
    first_edge = edges[int(edge_table.target_global_ids[first_row])]
    second_edge = edges[int(edge_table.target_global_ids[second_row])]
    common = set(first_edge) & set(second_edge)
    assert len(common) == 1
    vertex = next(iter(common))
    vertex_id = int(charts.mesh.vertex_global_ids[vertex])
    vertex_row = next(
        row
        for row, value in enumerate(vertex_table.target_global_ids)
        if int(value) == vertex_id
    )
    adjusted_parameters = np.asarray(vertex_table.parameters).copy()
    adjusted_parameters[vertex_row, 0] = float(shifted)
    forged_edge = replace_table(edge_table, intervals=tuple(altered))
    forged_vertex = replace_table(vertex_table, parameters=adjusted_parameters)
    forged_both = SurfaceSourceCharts(
        domain,
        charts.mesh,
        charts.geometry,
        charts.coordinate_contract,
        charts.root_parameters,
        charts.root_patches,
        tuple(
            forged_edge
            if value is edge_table
            else forged_vertex
            if value is vertex_table
            else value
            for value in charts.associations
        ),
        charts.boundary_source,
        source_root_cell_ids=charts.source_root_cell_ids,
        source_piece_ids=charts.source_piece_ids,
        root_parameter_exact=charts.root_parameter_exact,
        boundary_occurrences=charts.boundary_occurrences,
        trim_roots=charts.trim_roots,
    )
    with pytest.raises(ValueError, match="exact root interpolation proof"):
        validate_surface_curve_ancestry(forged_both, 0.05)


def test_non_affine_surface_trim_replays_exact_root_lineage(tmp_path: Path) -> None:
    from phydrax.geometry._surface_source_support import validate_surface_curve_ancestry

    domain = _curved_trim_domain()
    construction = prepare_native_surface(
        domain,
        _surface_specification(domain, maximum_deviation=0.04),
        NativeMeshingOptions(
            "parametric_surface",
            surface_schedule=NativeSurfaceSchedule(
                spacing_fraction=0.6,
                minimum_edge_samples=4,
                curve_schedule=NativeCurveSchedule(fidelity_samples=5),
            ),
        ),
        SpatialCoordinateContract.si(),
    ).execute()
    charts = construction.organization.surface_source
    assert charts is not None and charts.trim_roots
    assert any(
        root.first.denominator == 3 or root.last.denominator == 3
        for root in charts.trim_roots
    )
    edge_table = next(
        value
        for value in charts.associations
        if value.association_kind is GeometryAssociationKind.CURVE
    )
    assert any(
        endpoint.denominator == 3
        for interval in edge_table.source_parameter_intervals
        if interval is not None
        for endpoint in interval
    )
    bindings = validate_surface_curve_ancestry(charts, 0.04)
    assert {use.curve_row for uses in bindings.values() for use in uses} == set(
        range(len(domain.curves))
    )
    import subprocess
    import sys

    from phydrax.geometry._surface_source_support import SurfaceSourceCharts
    from phydrax.lifecycle import Composition, CompositionEntry
    from phydrax.meshing._model_codec import MODEL_ARCHIVE_LIMITS, write_model_artifact

    def rechart(roots: tuple) -> SurfaceSourceCharts:
        return SurfaceSourceCharts(
            charts.domain,
            charts.mesh,
            charts.geometry,
            charts.coordinate_contract,
            charts.root_parameters,
            charts.root_patches,
            charts.associations,
            charts.boundary_source,
            source_root_cell_ids=charts.source_root_cell_ids,
            source_piece_ids=charts.source_piece_ids,
            root_parameter_exact=charts.root_parameter_exact,
            boundary_occurrences=charts.boundary_occurrences,
            trim_roots=roots,
        )

    def persist(value: SurfaceSourceCharts, filename: str) -> Path:
        entry = CompositionEntry(
            value,
            entry_id="source/charts",
            role="model-state",
            owner_id=value.source_id,
            structure_id=value.source_chart_id,
            revision_id=value.source_revision,
            semantics_id="surface-source-charts",
        )
        return write_model_artifact(
            tmp_path / filename,
            Composition((entry,), boundary_id="surface-roots"),
            limits=MODEL_ARCHIVE_LIMITS,
        )

    script = (
        "import sys; from phydrax.lifecycle import Composition; "
        "from phydrax.meshing._model_codec import MODEL_ARCHIVE_LIMITS, read_model_artifact, restore_model_archive_value; "
        "from phydrax.geometry._surface_source_support import validate_surface_curve_ancestry; "
        "c=read_model_artifact(sys.argv[1], expected_type=Composition, restore=restore_model_archive_value, limits=MODEL_ARCHIVE_LIMITS); "
        "s=c.value('source/charts'); assert s.source_chart_id==sys.argv[2]; "
        "validate_surface_curve_ancestry(s,0.04)"
    )
    valid = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(persist(charts, "valid.phx")),
            charts.source_chart_id,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert valid.returncode == 0, valid.stderr
    roots = list(charts.trim_roots)
    first = roots[0]
    roots[0] = first._replace(
        first_uv=(first.first_uv[0] + Fraction(1, 100), first.first_uv[1])
    )
    forged = rechart(tuple(roots))
    assert forged.source_chart_id != charts.source_chart_id
    for value, filename in ((forged, "forged.phx"), (rechart(()), "empty-roots.phx")):
        rejected = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(persist(value, filename)),
                value.source_chart_id,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert rejected.returncode != 0, "unproved trim lineage was accepted"


def test_non_brep_surface_pole_ancestry_replays_exact_collapse() -> None:
    from phydrax.geometry._meshing_domain import PatchPoleUse
    from phydrax.geometry._surface_source_support import validate_surface_curve_ancestry

    contract = SpatialCoordinateContract.si()
    original = MeshingDomain.from_brep(
        brep_cylinder(0.5, 1.0, coordinate_contract=contract)
    )
    domain = MeshingDomain(
        original.patches,
        original.curves,
        len(original.corner_points),
        source_id="surface-pole-proof",
        source_revision="surface-pole-proof-r1",
        regions=original.regions,
        corner_points=original.corner_points,
    )
    assert any(
        isinstance(use, PatchPoleUse)
        for patch in domain.patches
        for loop in patch.loops
        for use in loop
    )
    construction = prepare_native_surface(
        domain,
        _surface_specification(domain, maximum_deviation=0.05),
        NativeMeshingOptions("parametric_surface"),
        contract,
    ).execute()
    charts = construction.organization.surface_source
    assert charts is not None
    bindings = validate_surface_curve_ancestry(charts, 0.05)
    # A collapsed pole side is a corner use, not a coedge: its degenerate
    # source curve has no authored chart owner and no mesh edge to bind. Every
    # other curve is bound, and every pole side was proved by the replay.
    pole_corners = {
        use.corner
        for patch in domain.patches
        for loop in patch.loops
        for use in loop
        if isinstance(use, PatchPoleUse)
    }
    unowned = {
        row
        for row in range(len(domain.curves))
        if tuple(int(value) for value in domain.curve_owners[row]) == (-1, -1, -1)
    }
    assert unowned == {
        row
        for row, curve in enumerate(domain.curves)
        if curve.start == curve.end and curve.start in pole_corners
    }
    assert unowned
    assert {use.curve_row for uses in bindings.values() for use in uses} == (
        set(range(len(domain.curves))) - unowned
    )
    # The harmonic collapse proof refuses a ring side that does not collapse.
    from phydrax.geometry._surface_source_support import _surface_pole_proven

    pole_patch, pole = next(
        (row, use)
        for row, patch in enumerate(domain.patches)
        for loop in patch.loops
        for use in loop
        if isinstance(use, PatchPoleUse)
    )
    assert _surface_pole_proven(charts, pole_patch, pole)
    ring = next(
        use
        for loop in domain.patches[pole_patch].loops
        for use in loop
        if isinstance(use, PatchCurveUse)
        and domain.curves[use.curve].start == domain.curves[use.curve].end
    )
    ring_corner = domain.curves[ring.curve].start
    forged_pole = PatchPoleUse(ring_corner, ring.pcurve, ring.first, ring.last)
    assert not _surface_pole_proven(charts, pole_patch, forged_pole)


def test_non_brep_seam_corner_is_the_exact_whole_turn_image() -> None:
    contract = SpatialCoordinateContract.si()
    original = MeshingDomain.from_brep(
        brep_cylinder(0.5, 1.0, coordinate_contract=contract)
    )
    seam_end = original.patches[1].surface.evaluate(
        jnp.asarray((2.0 * np.pi, 0.0), dtype=jnp.float64)
    )
    corner = 1
    assert np.array_equal(original.corner_points[corner], (0.0, 0.5, 0.0))
    # The rounded binary64 turn misses the seam corner; the exact turn does not.
    assert not np.array_equal(np.asarray(seam_end), original.corner_points[corner])
    forged = np.array(original.corner_points, copy=True)
    forged[corner] = np.asarray(seam_end)
    with pytest.raises(ValueError, match="endpoints disagree at corner 1"):
        MeshingDomain(
            original.patches,
            original.curves,
            len(original.corner_points),
            source_id="surface-pole-proof",
            source_revision="surface-pole-proof-r1",
            regions=original.regions,
            corner_points=forged,
        )


def test_curve_chord_endpoint_box_refusal_is_a_typed_meshing_failure() -> None:
    from phydrax.meshing._contracts import MeshingFailure, MeshingFailureCategory
    from phydrax.meshing.providers._native_surface import _CurveChordBounds

    domain = _curved_trim_domain()
    # A curved p-curve leaving the patch chart at its midpoint: its endpoint
    # chart box cannot be admitted by the source surface.
    escaping = PatchCurveUse(
        0,
        BSplineCurve(
            np.asarray(((0.0, 0.0), (0.5, -0.4), (1.0, 0.0)), dtype=np.float64),
            np.ones((3,), dtype=np.float64),
            np.asarray((0.0, 0.0, 0.0, 1.0, 1.0, 1.0), dtype=np.float64),
            2,
        ),
        0.0,
        1.0,
    )
    chords = _CurveChordBounds(SourcePreparation(domain))
    with pytest.raises(MeshingFailure) as failure:
        chords._point_bounds(0, escaping, Fraction(1, 2))
    assert failure.value.category is MeshingFailureCategory.PROVIDER_EXECUTION_FAILED
    assert failure.value.entity_ids == (0,)
    assert "patch 0 at native parameter 1/2" in str(failure.value)
    assert isinstance(failure.value.__cause__, ValueError)


def test_hard_surface_angle_decides_represented_face_ratio_at_threshold() -> None:
    from phydrax.meshing._quality import _triangle_minimum_sine_violations

    threshold = np.sqrt(15.0)
    lower = float(np.nextafter(threshold, -np.inf))
    upper = float(np.nextafter(threshold, np.inf))
    assert Fraction.from_float(lower) ** 2 < 15 < Fraction.from_float(upper) ** 2
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, lower, 0.0), (0.0, upper, 0.0)),
        dtype=np.float64,
    )
    triangles = np.asarray(((0, 1, 2), (0, 1, 3)), dtype=np.int64)
    np.testing.assert_array_equal(
        _triangle_minimum_sine_violations(points, triangles, Fraction(1, 4)),
        np.asarray((False, True), dtype=np.bool_),
    )
    np.testing.assert_array_equal(
        _triangle_minimum_sine_violations(
            points, np.asarray(((0, 1, 1),), dtype=np.int64), Fraction(1, 4)
        ),
        np.asarray((True,), dtype=np.bool_),
    )


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


def test_prepared_surface_boxes_preserve_owner_and_refusals() -> None:
    preparation = SourcePreparation(_square_domain())
    surfaces = (
        PlanePatch((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        PlanePatch((2.0, 3.0, 4.0), (0.0, 2.0, 0.0), (0.0, 0.0, 3.0)),
    )
    boxes = (
        np.asarray(((0.0, 0.25), (0.5, 1.0)), dtype=np.float64),
        np.asarray(((0.25, 0.5), (1.0, 0.75)), dtype=np.float64),
    )
    expected = tuple(
        tuple(surface.bounding_box(box) for box in boxes) for surface in surfaces
    )
    for surface_index in (0, 1, 0, 1):
        for box_index, box in enumerate(boxes):
            actual = preparation.surface_box(surfaces[surface_index], box)
            assert np.array_equal(actual, expected[surface_index][box_index])
    assert not np.array_equal(expected[0][0], expected[1][0])

    invalid = np.asarray(((1.0, 0.0), (0.0, 1.0)), dtype=np.float64)
    with pytest.raises(ValueError) as original:
        surfaces[0].bounding_box(invalid)
    with pytest.raises(ValueError) as prepared:
        preparation.surface_box(surfaces[0], invalid)
    assert str(prepared.value) == str(original.value)
