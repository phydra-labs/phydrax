#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._interpolation._tspline import TMesh2D
from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry._source_boundary_records import (
    SourceBoundaryProjectionQuery,
    SourceBoundaryProjectionResult,
    SourceProjectionStatus,
)
from phydrax.geometry._source_identity import source_entity_id
from phydrax.geometry.brep._constructors import (
    assemble_brep_model,
    brep_planar_face,
    PlanarProfile,
    ProfileLoop,
    ProfilePlane,
)
from phydrax.geometry.brep._model import (
    _certify_edge_face_gap_bounds,
    BRepGeometry,
    BRepModel,
)
from phydrax.geometry.brep._patches import BSplineSurfacePatch
from phydrax.geometry.brep._projection import prepare_brep_projection
from phydrax.geometry.brep._projection_contracts import (
    brep_entity_id,
    BRepProjectionPolicy,
)
from phydrax.geometry.brep._tspline import TSplineSurfacePatch
from phydrax.geometry.implicit._analytic_profile import AnalyticImplicitProfile
from phydrax.units import METER


def _sphere_profile() -> AnalyticImplicitProfile:
    geometry = phx.geometry.Sphere((0.0, 0.0, 0.0), 2.0).compile()
    return AnalyticImplicitProfile(geometry, SpatialCoordinateContract(METER))


def _tspline_planar_model(*, curved: bool) -> BRepModel:
    contract = SpatialCoordinateContract.si()
    base = brep_planar_face(
        PlanarProfile(
            ProfilePlane(),
            ProfileLoop.polygon(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        ),
        coordinate_contract=contract,
    )
    mesh = TMesh2D.tensor_product(
        tuple(Fraction(index, 4) for index in range(5)),
        tuple(Fraction(index, 4) for index in range(5)),
    )
    exact_points = []
    for anchor in mesh.anchors:
        u, v = anchor.local_knots.greville_parameter
        height = u * (1 - u) * v * (1 - v) if curved else Fraction(0)
        exact_points.append((u, v, height))
    exact_weights = tuple(Fraction(1) for _ in mesh.anchors)
    patch = TSplineSurfacePatch(
        mesh,
        np.asarray([[float(value) for value in point] for point in exact_points]),
        np.ones((mesh.coefficient_count,), dtype=np.float64),
        exact_control_points=tuple(exact_points),
        exact_weights=exact_weights,
    )
    return assemble_brep_model(
        base.geometry,
        (patch,),
        np.asarray(base.parameter_bounds),
        np.asarray(base.orientation),
        base.physical_tags,
        coordinate_contract=contract,
        source_id="native-tspline-projection",
        trim_domains=base.trim_domains,
    )


def _declared_gap_model(
    base: BRepModel,
    patch: BSplineSurfacePatch,
    source_id: str,
) -> BRepModel:
    """Bind a spline patch to the unit-square edges with honestly declared gaps.

    The folded and near-singular test nets do not pass through the square's
    edges, so each coedge declares its recomputed exact face-gap bound.
    """
    geometry = base.geometry
    face_gaps = _certify_edge_face_gap_bounds(
        geometry,
        (patch,),
        parameter_bounds=base.parameter_bounds,
        exact_unsupported=True,
        certified_degenerate_coedges=True,
    )
    assert face_gaps is not None and max(face_gaps) > 0.0
    declared = BRepGeometry(
        vertex_points=geometry.vertex_points,
        vertex_exact_points=geometry.vertex_exact_points,
        vertex_roots=geometry.vertex_roots,
        edge_endpoint_roots=geometry.edge_endpoint_roots,
        coedge_endpoint_roots=geometry.coedge_endpoint_roots,
        edge_vertex_gap_bounds=geometry.edge_vertex_gap_bounds,
        edge_face_gap_bounds=face_gaps,
        curves=geometry.curves,
        edge_curves=geometry.edge_curves,
        edge_ranges=geometry.edge_ranges,
        edge_vertices=geometry.edge_vertices,
        pcurves=geometry.pcurves,
        coedge_edges=geometry.coedge_edges,
        coedge_senses=geometry.coedge_senses,
        face_loops=geometry.face_loops,
        shell_faces=geometry.shell_faces,
        shell_orientations=geometry.shell_orientations,
        solid_shells=geometry.solid_shells,
        assembly_graph=geometry.assembly_graph,
    )
    return assemble_brep_model(
        declared,
        (patch,),
        np.asarray(base.parameter_bounds),
        np.asarray(base.orientation),
        base.physical_tags,
        coordinate_contract=base.coordinate_contract,
        source_id=source_id,
        trim_domains=base.trim_domains,
    )


def _bspline_surface_model(
    *, competing: bool = False
) -> tuple[BRepModel, BSplineSurfacePatch]:
    contract = SpatialCoordinateContract.si()
    base = brep_planar_face(
        PlanarProfile(
            ProfilePlane(),
            ProfileLoop.polygon(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        ),
        coordinate_contract=contract,
    )
    exact_points = []
    exact_weights = []
    # The former height-bump net had one closest stationary root; the exact folded
    # Bernstein coordinate below gives two distinct parameter preimages.
    folded_x_controls = (Fraction(1, 4), Fraction(-1, 4), Fraction(1, 4))
    for i in range(3):
        point_row, weight_row = [], []
        for j in range(3):
            u, v = Fraction(i, 2), Fraction(j, 2)
            bump = u * (1 - u) * v * (1 - v)
            if competing:
                point = (folded_x_controls[i], v, Fraction(0))
                weight = Fraction(1)
            else:
                point = (u + bump / 4, v - bump / 4, bump)
                weight = Fraction(2) if (i, j) == (1, 1) else Fraction(1)
            point_row.append(point)
            weight_row.append(weight)
        exact_points.append(tuple(point_row))
        exact_weights.append(tuple(weight_row))
    patch = BSplineSurfacePatch(
        np.asarray(
            [
                [[float(value) for value in point] for point in row]
                for row in exact_points
            ],
            dtype=np.float64,
        ),
        np.asarray([[float(value) for value in row] for row in exact_weights]),
        np.asarray((0.0, 0.0, 0.0, 1.0, 1.0, 1.0)),
        np.asarray((0.0, 0.0, 0.0, 1.0, 1.0, 1.0)),
        2,
        2,
        exact_control_points=tuple(exact_points),
        exact_weights=tuple(exact_weights),
    )
    if competing:
        # The folded net leaves the square's x edges: its gaps are declared.
        return _declared_gap_model(
            base, patch, "native-bspline-competing-projection"
        ), patch
    model = assemble_brep_model(
        base.geometry,
        (patch,),
        np.asarray(base.parameter_bounds),
        np.asarray(base.orientation),
        base.physical_tags,
        coordinate_contract=contract,
        source_id="native-rational-bspline-projection",
        trim_domains=base.trim_domains,
    )
    return model, patch


def _nearby_stationary_root_surface_model() -> tuple[BRepModel, BSplineSurfacePatch]:
    contract = SpatialCoordinateContract.si()
    base = brep_planar_face(
        PlanarProfile(
            ProfilePlane(),
            ProfileLoop.polygon(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        ),
        coordinate_contract=contract,
    )
    x_scale = Fraction(1, 10**10)
    linear_height = Fraction(1, 10**9)
    x_controls = (Fraction(0), x_scale / 2, x_scale)
    z_controls = (
        Fraction(0),
        linear_height / 2,
        linear_height - 1,
    )
    exact_points = tuple(
        tuple((x_controls[i], Fraction(j, 2), z_controls[i]) for j in range(3))
        for i in range(3)
    )
    exact_weights = tuple(tuple(Fraction(1) for _ in range(3)) for _ in range(3))
    patch = BSplineSurfacePatch(
        np.asarray(
            [
                [[float(value) for value in point] for point in row]
                for row in exact_points
            ],
            dtype=np.float64,
        ),
        np.ones((3, 3), dtype=np.float64),
        np.asarray((0.0, 0.0, 0.0, 1.0, 1.0, 1.0)),
        np.asarray((0.0, 0.0, 0.0, 1.0, 1.0, 1.0)),
        2,
        2,
        exact_control_points=exact_points,
        exact_weights=exact_weights,
    )
    return _declared_gap_model(
        base, patch, "native-bspline-nearby-stationary-roots"
    ), patch


def _near_tied_folded_bspline_model() -> tuple[BRepModel, BSplineSurfacePatch]:
    contract = SpatialCoordinateContract.si()
    base = brep_planar_face(
        PlanarProfile(
            ProfilePlane(),
            ProfileLoop.polygon(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        ),
        coordinate_contract=contract,
    )
    x_controls = (
        Fraction(1, 4),
        Fraction(-1, 12),
        Fraction(-1, 12),
        Fraction(1, 4),
    )
    y_controls = (Fraction(0), Fraction(1, 3), Fraction(2, 3), Fraction(1))
    z_controls = (
        Fraction(-13, 384),
        Fraction(17, 384),
        Fraction(-17, 384),
        Fraction(13, 384),
    )
    exact_points = tuple(
        tuple((x_controls[i], y_controls[j], z_controls[i]) for j in range(4))
        for i in range(4)
    )
    exact_weights = tuple(tuple(Fraction(1) for _ in range(4)) for _ in range(4))
    knots = np.asarray((0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0))
    patch = BSplineSurfacePatch(
        np.asarray(
            [
                [[float(value) for value in point] for point in row]
                for row in exact_points
            ],
            dtype=np.float64,
        ),
        np.ones((4, 4), dtype=np.float64),
        knots,
        knots,
        3,
        3,
        exact_control_points=exact_points,
        exact_weights=exact_weights,
    )
    return _declared_gap_model(base, patch, "native-bspline-near-tied-minima"), patch


def _tspline_non_graph_model(
    *, competing: bool = False
) -> tuple[BRepModel, TSplineSurfacePatch]:
    contract = SpatialCoordinateContract.si()
    base = brep_planar_face(
        PlanarProfile(
            ProfilePlane(),
            ProfileLoop.polygon(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        ),
        coordinate_contract=contract,
    )
    mesh = TMesh2D.tensor_product(
        tuple(Fraction(index, 4) for index in range(5)),
        tuple(Fraction(index, 4) for index in range(5)),
    )
    exact_points = []
    for anchor in mesh.anchors:
        u, v = anchor.local_knots.greville_parameter
        bump = u * (1 - u) * v * (1 - v)
        exact_points.append(
            (u + (u - Fraction(1, 2)) * bump / 8, v, 8 * bump)
            if competing
            else (u + bump / 4, v - bump / 4, bump)
        )
    exact_weights = tuple(Fraction(1) for _ in mesh.anchors)
    patch = TSplineSurfacePatch(
        mesh,
        np.asarray([[float(value) for value in point] for point in exact_points]),
        np.ones((mesh.coefficient_count,), dtype=np.float64),
        exact_control_points=tuple(exact_points),
        exact_weights=exact_weights,
    )
    model = assemble_brep_model(
        base.geometry,
        (patch,),
        np.asarray(base.parameter_bounds),
        np.asarray(base.orientation),
        base.physical_tags,
        coordinate_contract=contract,
        source_id=(
            "native-tspline-competing-projection"
            if competing
            else "native-tspline-non-graph-projection"
        ),
        trim_domains=base.trim_domains,
    )
    return model, patch


def test_analytic_implicit_projection_binds_frame_source_and_certified_distance() -> None:
    profile = _sphere_profile()
    assert isinstance(profile, SourceBoundaryProjectionQuery)
    entity_id = source_entity_id(
        profile.source_id, profile.source_revision, "implicit-surface", "zero-set"
    )
    result = profile.project_boundary(
        np.asarray(((4.0, 0.0, 0.0),)), entity_ids=(entity_id,)
    )

    assert isinstance(result, SourceBoundaryProjectionResult)
    assert result.source_id == profile.source_id
    assert result.source_revision == profile.source_revision
    assert result.frame_id == profile.coordinate_contract.spatial_id
    assert result.entity_ids() == (entity_id,)
    assert result.occurrence_paths == ((),)
    assert int(np.asarray(result.status)[0]) == SourceProjectionStatus.UNIQUE
    np.testing.assert_allclose(result.points, ((2.0, 0.0, 0.0),), atol=1e-14)
    source_parameters = np.asarray(result.parameters)
    # Neighboring binary64 angles enclose the exact polar angle π/2 for this
    # axis-aligned query; certify chart reconstruction over that parameter box.
    source_lower, source_upper = profile._map_program.evaluate(
        np.nextafter(source_parameters, -np.inf),
        np.nextafter(source_parameters, np.inf),
    )
    point = np.asarray(result.points)
    assert np.all(np.asarray(source_lower) <= point)
    assert np.all(point <= np.asarray(source_upper))
    np.testing.assert_allclose(result.normals, ((1.0, 0.0, 0.0),), atol=0.0)
    np.testing.assert_allclose(result.residuals, (2.0,), atol=0.0)
    np.testing.assert_allclose(result.distance_lower_bounds, (2.0,), atol=0.0)
    tangents = np.asarray(result.tangents)
    np.testing.assert_allclose(np.linalg.norm(tangents, axis=-1), 1.0, atol=1e-15)
    np.testing.assert_allclose(
        np.einsum("nka,na->nk", tangents, result.normals), 0.0, atol=1e-15
    )


def test_implicit_nonisolated_minimum_keeps_source_identity_without_a_point() -> None:
    profile = _sphere_profile()
    result = profile.project_boundary(np.zeros((1, 3), dtype=np.float64))

    assert int(np.asarray(result.status)[0]) == SourceProjectionStatus.NON_ISOLATED
    assert result.entity_ids() == (
        source_entity_id(
            profile.source_id,
            profile.source_revision,
            "implicit-surface",
            "zero-set",
        ),
    )
    np.testing.assert_array_equal(result.residuals, (2.0,))
    np.testing.assert_array_equal(result.distance_lower_bounds, (2.0,))
    assert np.all(np.isnan(np.asarray(result.points)))
    assert np.all(np.isnan(np.asarray(result.parameters)))
    assert np.all(np.isnan(np.asarray(result.normals)))
    assert np.all(np.isnan(np.asarray(result.tangents)))


def test_torus_projection_handles_unique_and_nonisolated_loci() -> None:
    geometry = phx.geometry.Torus((0.0, 0.0, 0.0), 1.5, 2.5).compile()
    profile = AnalyticImplicitProfile(geometry, SpatialCoordinateContract(METER))
    unique = profile.project_boundary(np.asarray(((2.5, 0.0, 0.0),)))
    np.testing.assert_allclose(unique.points, ((2.5, 0.0, 0.0),), atol=0.0)
    np.testing.assert_allclose(unique.residuals, (0.0,), atol=0.0)
    assert int(np.asarray(unique.status)[0]) == SourceProjectionStatus.UNIQUE

    singular = profile.project_boundary(np.asarray(((0.0, 0.0, 0.0),)))
    assert int(np.asarray(singular.status)[0]) == SourceProjectionStatus.NON_ISOLATED
    assert singular.entity_ids()[0]
    assert np.all(np.isnan(np.asarray(singular.points)))
    assert np.all(np.isnan(np.asarray(singular.parameters)))
    np.testing.assert_allclose(singular.residuals, (1.5,), atol=0.0)
    np.testing.assert_allclose(singular.distance_lower_bounds, (1.5,), atol=0.0)


def test_implicit_projection_rejects_foreign_entities_and_occurrences() -> None:
    profile = _sphere_profile()
    with pytest.raises(ValueError, match="implicit source revision"):
        profile.project_boundary(
            np.asarray(((2.0, 0.0, 0.0),)), entity_ids=("foreign-entity",)
        )
    with pytest.raises(ValueError, match="definition-only empty occurrence path"):
        profile.project_boundary(
            np.asarray(((2.0, 0.0, 0.0),)), occurrence_paths=(("foreign-use",),)
        )


def test_native_brep_projection_uses_source_neutral_boundary_query() -> None:
    contract = SpatialCoordinateContract.si()
    model = brep_planar_face(
        PlanarProfile(
            ProfilePlane(),
            ProfileLoop.polygon(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        ),
        coordinate_contract=contract,
    )
    projection = phx.geometry.prepare_brep_projection(model)
    entity_id = brep_entity_id(model.source_id, model.source_revision, 2, 0)
    points = np.asarray(((0.5, 0.5, 0.25),))

    assert isinstance(projection, SourceBoundaryProjectionQuery)
    result = projection.project_boundary(
        points,
        entity_ids=(entity_id,),
        occurrence_paths=((),),
    )
    assert isinstance(result, SourceBoundaryProjectionResult)
    assert result.frame_id == contract.spatial_id
    assert result.entity_ids() == (entity_id,)
    assert result.occurrence_paths == ((),)
    assert int(np.asarray(result.status)[0]) == SourceProjectionStatus.UNIQUE
    np.testing.assert_allclose(result.points, ((0.5, 0.5, 0.0),), atol=1e-14)
    assert float(result.distance_lower_bounds[0]) <= 0.25 <= float(result.residuals[0])
    distance = projection.boundary_distance(points)
    assert distance.semantics == "certified"
    assert float(distance.lower[0]) <= 0.25 <= float(distance.upper[0])

    with pytest.raises(ValueError, match="do not belong to this source"):
        projection.project_boundary(points, entity_ids=("foreign-entity",))


def test_t_spline_projection_uses_exact_extraction_identity_without_tessellation() -> (
    None
):
    model = _tspline_planar_model(curved=False)
    projection = prepare_brep_projection(model)
    entity_id = brep_entity_id(model.source_id, model.source_revision, 2, 0)
    result = projection.project_boundary(
        np.asarray(((0.5, 0.5, 0.25),)), entity_ids=(entity_id,)
    )

    assert result.entity_ids() == (entity_id,)
    assert int(np.asarray(result.status)[0]) == SourceProjectionStatus.UNIQUE
    np.testing.assert_allclose(result.points, ((0.5, 0.5, 0.0),), atol=1e-14)
    np.testing.assert_allclose(result.parameters, ((0.5, 0.5),), atol=1e-14)
    assert float(result.distance_lower_bounds[0]) <= 0.25 <= float(result.residuals[0])


def test_curved_t_spline_projection_has_complete_strong_convexity_evidence() -> None:
    model = _tspline_planar_model(curved=True)
    projection = prepare_brep_projection(model)
    entity_id = brep_entity_id(model.source_id, model.source_revision, 2, 0)
    result = projection.project_boundary(
        np.asarray(((0.5, 0.5, 0.25),)), entity_ids=(entity_id,)
    )

    assert int(np.asarray(result.status)[0]) == SourceProjectionStatus.UNIQUE
    assert result.entity_ids() == (entity_id,)
    # Greville-sampled controls reproduce only the linear coordinates: the cubic
    # quasi-interpolant of u(1-u) is 11/48 at u=1/2, so the cap height is (11/48)^2.
    height = Fraction(11, 48) ** 2
    distance = float(Fraction(1, 4) - height)
    np.testing.assert_allclose(result.points, ((0.5, 0.5, float(height)),), atol=1e-14)
    np.testing.assert_allclose(result.parameters, ((0.5, 0.5),), atol=1e-14)
    assert (
        float(result.distance_lower_bounds[0]) <= distance <= float(result.residuals[0])
    )
    patch = model.patches[0]
    assert isinstance(patch, TSplineSurfacePatch)
    fixed_parameters = jnp.asarray(result.parameters)

    def evaluate_fixed_chart(control_points: jax.Array) -> jax.Array:
        return jnp.sum(
            patch.realization.evaluate_rational(
                fixed_parameters,
                control_points,
                patch.weights,
                maximum_order=0,
            ).values
        )

    gradient = np.asarray(jax.grad(evaluate_fixed_chart)(patch.control_points))
    assert np.all(np.isfinite(gradient)) and np.any(gradient != 0.0)
    selected = tuple(
        int(value)
        for value in np.unravel_index(np.argmax(np.abs(gradient)), gradient.shape)
    )
    step = 1.0e-6
    controls = np.asarray(patch.control_points)
    plus, minus = controls.copy(), controls.copy()
    plus[selected] += step
    minus[selected] -= step
    finite_difference = (
        float(evaluate_fixed_chart(jnp.asarray(plus)))
        - float(evaluate_fixed_chart(jnp.asarray(minus)))
    ) / (2.0 * step)
    np.testing.assert_allclose(
        gradient[selected], finite_difference, rtol=2e-6, atol=2e-8
    )


def test_curved_t_spline_without_a_stationary_uniqueness_proof_is_unresolved() -> None:
    model = _tspline_planar_model(curved=True)
    projection = prepare_brep_projection(model)
    entity_id = brep_entity_id(model.source_id, model.source_revision, 2, 0)
    result = projection.project_boundary(
        np.asarray(((0.4, 0.5, 0.25),)), entity_ids=(entity_id,)
    )

    assert int(np.asarray(result.status)[0]) == SourceProjectionStatus.UNRESOLVED
    assert result.entity_ids() == (entity_id,)
    assert np.all(np.isnan(np.asarray(result.points)))
    assert np.all(np.isnan(np.asarray(result.parameters)))


def test_rational_bspline_projection_certifies_curved_non_graph_minimum() -> None:
    model, patch = _bspline_surface_model()
    entity_id = brep_entity_id(model.source_id, model.source_revision, 2, 0)
    query = np.asarray(
        patch.evaluate(jnp.asarray((0.5, 0.5), dtype=jnp.float64)),
        dtype=np.float64,
    )
    result = prepare_brep_projection(model).project_boundary(
        query[None, :], entity_ids=(entity_id,)
    )

    assert int(np.asarray(result.status)[0]) == SourceProjectionStatus.UNIQUE
    assert result.entity_ids() == (entity_id,)
    np.testing.assert_allclose(result.parameters, ((0.5, 0.5),), atol=1.0e-8)
    np.testing.assert_allclose(result.points, query[None, :], atol=1.0e-9)
    assert 0.0 <= float(result.distance_lower_bounds[0])
    assert float(result.residuals[0]) < 1.0e-9


def test_rational_bspline_projection_compares_trim_boundary_minimum() -> None:
    model, _ = _bspline_surface_model()
    entity_id = brep_entity_id(model.source_id, model.source_revision, 2, 0)
    query = np.asarray(((-0.25, 0.5, 0.125),), dtype=np.float64)
    result = prepare_brep_projection(model).project_boundary(
        query, entity_ids=(entity_id,)
    )
    expected_distance = np.sqrt(0.25**2 + 0.125**2)

    assert int(np.asarray(result.status)[0]) == SourceProjectionStatus.SEAM
    assert result.entity_ids() == (entity_id,)
    np.testing.assert_allclose(result.points, ((0.0, 0.5, 0.0),), atol=1.0e-9)
    np.testing.assert_allclose(result.parameters, ((0.0, 0.5),), atol=1.0e-8)
    assert (
        float(result.distance_lower_bounds[0])
        <= expected_distance
        <= float(result.residuals[0])
    )


def test_known_boundary_minimum_does_not_hide_unresolved_stationary_roots() -> None:
    model, patch = _nearby_stationary_root_surface_model()
    entity_id = brep_entity_id(model.source_id, model.source_revision, 2, 0)
    query = np.asarray(
        patch.evaluate(jnp.asarray((0.0, 0.5), dtype=jnp.float64)),
        dtype=np.float64,
    )
    np.testing.assert_array_equal(query, (0.0, 0.5, 0.0))
    # Nearby interior roots remain unexcluded inside the policy-sized chart box.

    result = prepare_brep_projection(model).project_boundary(
        query[None, :], entity_ids=(entity_id,)
    )

    assert int(np.asarray(result.status)[0]) == SourceProjectionStatus.UNRESOLVED
    assert result.entity_ids() == (entity_id,)
    assert float(result.distance_lower_bounds[0]) == 0.0
    assert float(result.residuals[0]) < 1.0e-14
    assert np.all(np.isnan(np.asarray(result.points)))
    assert np.all(np.isnan(np.asarray(result.parameters)))


def test_non_graph_tspline_projection_uses_extracted_piece_roots() -> None:
    model, patch = _tspline_non_graph_model()
    entity_id = brep_entity_id(model.source_id, model.source_revision, 2, 0)
    query = np.asarray(
        patch.evaluate(jnp.asarray((0.5, 0.5), dtype=jnp.float64)),
        dtype=np.float64,
    )
    result = prepare_brep_projection(model).project_boundary(
        query[None, :], entity_ids=(entity_id,)
    )

    assert int(np.asarray(result.status)[0]) == SourceProjectionStatus.UNIQUE
    assert result.entity_ids() == (entity_id,)
    np.testing.assert_allclose(result.points, query[None, :], atol=1.0e-9)
    np.testing.assert_allclose(result.parameters, ((0.5, 0.5),), atol=1.0e-8)
    assert float(result.residuals[0]) < 1.0e-9


def _assert_spline_competing_minima(
    model: BRepModel,
    query: np.ndarray,
    feasible_distance: float,
) -> None:
    entity_id = brep_entity_id(model.source_id, model.source_revision, 2, 0)
    result = prepare_brep_projection(model).project_boundary(
        query[None, :], entity_ids=(entity_id,)
    )
    assert int(np.asarray(result.status)[0]) == SourceProjectionStatus.AMBIGUOUS
    assert result.entity_ids() == (entity_id,)
    assert (
        float(result.distance_lower_bounds[0])
        <= float(result.residuals[0])
        <= feasible_distance + 1.0e-12
    )


def test_rational_bspline_projection_preserves_competing_minima() -> None:
    model, patch = _bspline_surface_model(competing=True)
    query = np.asarray(
        patch.evaluate(jnp.asarray((0.375, 0.375), dtype=jnp.float64)),
        dtype=np.float64,
    )
    competing_point = np.asarray(
        patch.evaluate(jnp.asarray((0.625, 0.375), dtype=jnp.float64)),
        dtype=np.float64,
    )
    feasible_distance = float(np.linalg.norm(query - competing_point))
    _assert_spline_competing_minima(model, query, feasible_distance)


def test_tspline_projection_preserves_competing_minima() -> None:
    model, patch = _tspline_non_graph_model(competing=True)
    query = np.asarray((0.5, 0.5, 0.0), dtype=np.float64)
    left = np.asarray(
        patch.evaluate(jnp.asarray((0.5, 0.25), dtype=jnp.float64)),
        dtype=np.float64,
    )
    right = np.asarray(
        patch.evaluate(jnp.asarray((0.5, 0.75), dtype=jnp.float64)),
        dtype=np.float64,
    )
    left_distance = float(np.linalg.norm(left - query))
    right_distance = float(np.linalg.norm(right - query))
    np.testing.assert_allclose(left_distance, right_distance, atol=1.0e-14)
    _assert_spline_competing_minima(model, query, left_distance)


def test_near_but_distinct_spline_minima_honor_ambiguity_tolerance() -> None:
    model, patch = _near_tied_folded_bspline_model()
    tie_tolerance = BRepProjectionPolicy().ambiguity_tolerance
    source_distance = Fraction(1, 768)
    query_height = Fraction(1, 2**32)
    distance_gap = 2 * query_height
    assert 0 < float(distance_gap) < tie_tolerance
    query = np.asarray(
        (
            float(Fraction(1, 64)),
            float(Fraction(3, 8)),
            float(query_height),
        ),
        dtype=np.float64,
    )
    left = np.asarray(
        patch.evaluate(jnp.asarray((0.375, 0.375), dtype=jnp.float64)),
        dtype=np.float64,
    )
    right = np.asarray(
        patch.evaluate(jnp.asarray((0.625, 0.375), dtype=jnp.float64)),
        dtype=np.float64,
    )
    left_distance = float(source_distance - query_height)
    right_distance = float(source_distance + query_height)
    np.testing.assert_allclose(np.linalg.norm(left - query), left_distance, atol=1.0e-14)
    np.testing.assert_allclose(
        np.linalg.norm(right - query), right_distance, atol=1.0e-14
    )
    entity_id = brep_entity_id(model.source_id, model.source_revision, 2, 0)
    result = prepare_brep_projection(
        model, policy=BRepProjectionPolicy()
    ).project_boundary(query[None, :], entity_ids=(entity_id,))

    assert int(np.asarray(result.status)[0]) == SourceProjectionStatus.AMBIGUOUS
    assert result.entity_ids() == (entity_id,)
    assert float(result.residuals[0]) <= min(left_distance, right_distance) + 1.0e-12
