#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._array_archive import (
    ArrayArchiveLimits,
    read_array_archive,
    write_array_archive,
)
from phydrax._model._structure import (
    model_from_logical_array_recipe,
    model_structure_recipe,
    pack_model_array_tree,
)
from phydrax.discretization import (
    CellBlock,
    CellGeometrySpec,
    CellMesh,
    PolygonalConnectivity,
    PolynomialComposedCellGeometryElement,
    RestrictedCellGeometryElement,
    SplineCellGeometryElement,
)
from phydrax.discretization._cell_geometry import (
    coordinate_lagrange_element,
    require_scalar_coordinate_element,
)
from phydrax.discretization._cell_geometry_validity import (
    cell_geometry_id,
    CellValidityPolicy,
    certify_cell_geometry_validity,
)
from phydrax.discretization._coordinate_enclosure import (
    coordinate_corner_images,
    coordinate_expressions,
    RationalPolynomial,
    rounded_point,
    source_basis,
)
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.geometry._domain_coverage import DomainCoverageCertificate
from phydrax.geometry._surface_source_support import (
    SurfaceNativeRestrictionBoundarySource,
    SurfaceSourceRootAtlas,
)
from phydrax.geometry.brep import BSplineSurfacePatch
from phydrax.lifecycle._meshing_sources import (
    read_meshing_source_closure,
    register_meshing_source_artifacts,
    write_meshing_source_closure,
)
from phydrax.meshing._contracts import MeshingLimits
from phydrax.meshing._dual_extraction import DualExtraction
from phydrax.meshing._failures import MeshingFailure, MeshingFailureCategory
from phydrax.meshing._quad_generation import (
    extract_surface_quads,
    prepare_surface_cross_field,
)
from phydrax.meshing._result import CellMeshingResult
from phydrax.meshing.qualification import (
    generate_source_quads,
    solve_linear_diffusion,
)
from phydrax.typing import ConvertibleToArray
from tests._support.coordinate_references import (
    coordinate_reference_budget,
    native_meshing_request,
)
from tests._support.quad_generation_sources import rational_quad_source, tangent_triangle


# Match the default OptimizationTermination absolute optimality of 1e-8
# for unit-scale C4 alignment comparisons in prepare_surface_cross_field.
CROSS_FIELD_ALIGNMENT_ATOL = 1.0e-8
# The positive shoelace sums in these unit-scale fixtures use fewer than 4096
# roundoff contributions: gamma_4096 < 1e-12 in binary64 (Higham, Accuracy and
# Stability of Numerical Algorithms, 2nd ed., section 3.1).
AREA_CONSERVATION_RTOL = 1.0e-12
# Six-control rational source tabulation has at most 28 local contributions;
# gamma_28 / sqrt(0.5) < 1e-14 at unit scale. The frame normalization fixture
# has fewer contributions and a larger norm margin.
SOURCE_TABULATION_ATOL = 1.0e-14
# At the rational quarter-circle midpoint, dyadic factors require no rounding;
# the remaining weighted sum and division require at most eight unit roundoffs.
# 8 * eps64 < 2e-15. This retains the pre-split endpoint allowance.
RATIONAL_MIDPOINT_ATOL = 2.0e-15
# Reference solve and comparison targets from the pre-split spline archive
# regression's manufactured harmonic-x PDE and axial x-squared FV integral.
# The solve tolerances stay four orders tighter than the PDE acceptance limit;
# these named qualification targets retain every historical decimal value.
RESTORE_SOLVE_RTOL = 1.0e-12
RESTORE_SOLVE_ATOL = 1.0e-13
RESTORE_PDE_ERROR_LIMIT = 1.0e-8
RESTORE_QUADRATURE_RTOL = 1.0e-10


# Fixed empirical residual bounds from test_quad_generation.py
# at a6827a9e5; each comparison retains its strictness.
# Cross-field alignment residual at the four authored square features in each face
# frame.
_SQUARE_CROSS_FIELD_RESIDUAL = 1e-8


def _area(mesh: CellMesh, /) -> float:
    corners = np.asarray(mesh.coordinates)[np.asarray(mesh.blocks[0].vertices)]
    following = np.roll(corners, -1, axis=1)
    return float(
        0.5
        * np.sum(
            corners[..., 0] * following[..., 1] - corners[..., 1] * following[..., 0]
        )
    )


def test_quad_dual_closes_shared_edges_and_preserves_area() -> None:
    source = CellMesh.from_triangles(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float64),
        np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
    )
    with native_meshing_request(MeshingLimits()) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        result = extract_surface_quads(
            source, MeshingLimits(), scope=execution_scope, budget=coordinate_budget
        )
        assert result.mesh.blocks[0].cell_kind == "quadrilateral"
        assert result.validity.certified_valid_count == 6
        assert result.validity.invalid_count == result.validity.unresolved_count == 0
        assert _area(result.mesh) == pytest.approx(1.0)
        quads = np.asarray(result.mesh.blocks[0].vertices)
        edges = Counter(
            tuple(sorted((int(quad[a]), int(quad[b]))))
            for quad in quads
            for a, b in ((0, 1), (1, 2), (2, 3), (3, 0))
        )
        assert set(edges.values()) == {1, 2}
        assert sum(count == 1 for count in edges.values()) == 8
        assert result.mesh.coordinates.shape[0] - len(edges) + quads.shape[0] == 1
        np.testing.assert_array_equal(
            np.asarray(source.coordinates),
            np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        )


def test_surface_extraordinary_vertex_has_explicit_valence() -> None:
    angles = np.arange(5, dtype=np.float64) * 2.0 * np.pi / 5.0
    points = np.concatenate(
        (
            np.zeros((1, 2), dtype=np.float64),
            np.stack((np.cos(angles), np.sin(angles)), axis=1),
        )
    )
    source = CellMesh.from_triangles(
        points,
        np.asarray([(0, 1 + row, 1 + (row + 1) % 5) for row in range(5)], dtype=np.int64),
    )
    with native_meshing_request(MeshingLimits()) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        result = extract_surface_quads(
            source, MeshingLimits(), scope=execution_scope, budget=coordinate_budget
        )
        assert result.valences[0] == 5
        assert 0 in result.singular_vertices
        assert not result.boundary_vertices[0]
        np.testing.assert_allclose(_area(result.mesh), 2.5 * np.sin(2.0 * np.pi / 5.0))


def test_closed_embedded_surface_keeps_euler_characteristic() -> None:
    source = CellMesh.from_triangles(
        np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        ),
        np.asarray(((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)), dtype=np.int64),
    )
    with native_meshing_request(MeshingLimits()) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        result = extract_surface_quads(
            source, MeshingLimits(), scope=execution_scope, budget=coordinate_budget
        )
        assert (
            result.mesh.coordinates.shape[0]
            - result.mesh.topology.entities(1).count
            + result.mesh.topology.entities(2).count
            == 2
        )
        assert not np.any(result.boundary_vertices)
        assert result.validity.certified_valid_count == 12


def test_native_cross_field_aligns_square_features_in_each_face_frame() -> None:
    source = CellMesh.from_triangles(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float64),
        np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
    )
    with native_meshing_request(MeshingLimits()) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        field = prepare_surface_cross_field(
            source, scope=execution_scope, budget=coordinate_budget
        )
        assert np.max(field.feature_residuals) < _SQUARE_CROSS_FIELD_RESIDUAL
        mismatch = (
            field.angles[field.transport_edges[:, 1]]
            - field.angles[field.transport_edges[:, 0]]
            - field.transport_angles
        )
        np.testing.assert_allclose(
            np.cos(4.0 * mismatch),
            np.ones_like(mismatch),
            atol=CROSS_FIELD_ALIGNMENT_ATOL,
        )
        assert np.isfinite(np.asarray(field.optimization.objective))
        assert field.optimization.diagnostics.counts_complete
        assert int(np.asarray(field.optimization.diagnostics.objective_evaluations)) > 0
        assert int(np.asarray(field.optimization.diagnostics.gradient_evaluations)) > 0


def test_protected_crease_separates_field_patches_and_retains_quad_chain() -> None:
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.1, -0.7, 0.4)),
        dtype=np.float64,
    )
    triangles = np.asarray(((0, 1, 2), (1, 0, 3)), dtype=np.int64)
    source = CellMesh.from_triangles(points, triangles)
    assert isinstance(source.connectivity, PolygonalConnectivity)
    edges = np.asarray(source.connectivity.edges, dtype=np.int64)
    crease = np.flatnonzero(np.all(np.sort(edges, axis=1) == (0, 1), axis=1))
    isolated = CellMesh.from_triangles(points[:3], triangles[:1])
    with native_meshing_request(MeshingLimits()) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        reference = prepare_surface_cross_field(
            isolated, scope=execution_scope, budget=coordinate_budget
        )
        field = prepare_surface_cross_field(
            source, feature_edges=crease, scope=execution_scope, budget=coordinate_budget
        )
        np.testing.assert_array_equal(field.patch_ids, (0, 1))
        np.testing.assert_allclose(
            np.exp(4j * field.angles[0]),
            np.exp(4j * reference.angles[0]),
            atol=CROSS_FIELD_ALIGNMENT_ATOL,
        )
        # Changing another feature-separated patch cannot rotate this patch's field.
        modified = points.copy()
        modified[3] = (0.7, -0.2, 0.8)
        changed = prepare_surface_cross_field(
            CellMesh.from_triangles(modified, triangles),
            feature_edges=crease,
            scope=execution_scope,
            budget=coordinate_budget,
        )
        np.testing.assert_allclose(
            np.exp(4j * changed.angles[0]),
            np.exp(4j * reference.angles[0]),
            atol=CROSS_FIELD_ALIGNMENT_ATOL,
        )
        result = extract_surface_quads(
            source,
            MeshingLimits(),
            cross_field=field,
            scope=execution_scope,
            budget=coordinate_budget,
        )
        np.testing.assert_array_equal(
            field.patch_ids[result.parent_cells], (0, 0, 0, 1, 1, 1)
        )
        assert isinstance(result.mesh.connectivity, PolygonalConnectivity)
        child_edges = np.asarray(result.mesh.connectivity.edges, dtype=np.int64)
        selected = (result.entity_parent_dimensions[1] == 1) & (
            result.entity_parent_rows[1] == crease[0]
        )
        crease_edges = child_edges[selected]
        child_points = np.asarray(result.mesh.coordinates)[crease_edges]
        np.testing.assert_allclose(child_points[..., 1:], 0.0, atol=0.0)
        np.testing.assert_allclose(
            np.sum(np.linalg.norm(np.diff(child_points, axis=1)[:, 0], axis=1)), 1.0
        )
        quads = np.asarray(result.mesh.blocks[0].vertices)
        incidence = Counter(
            tuple(sorted((int(quad[a]), int(quad[b]))))
            for quad in quads
            for a, b in ((0, 1), (1, 2), (2, 3), (3, 0))
        )
        assert all(incidence[tuple(sorted(edge))] == 2 for edge in crease_edges.tolist())
        assert result.validity.certified_valid_count == 6


def test_immutable_odd_boundary_parity_is_explicit() -> None:
    source = CellMesh.from_triangles(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float64),
        np.asarray(((0, 1, 2),), dtype=np.int64),
    )
    with native_meshing_request(MeshingLimits()) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        with pytest.raises(MeshingFailure) as failed:
            extract_surface_quads(
                source,
                MeshingLimits(),
                boundary_subdivision=False,
                scope=execution_scope,
                budget=coordinate_budget,
            )
        assert failed.value.category is MeshingFailureCategory.PROVIDER_EXECUTION_FAILED


def test_quad_capacity_refusal_preserves_source() -> None:
    source = CellMesh.from_triangles(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float64),
        np.asarray(((0, 1, 2),), dtype=np.int64),
    )
    original = np.asarray(source.coordinates).copy()
    with native_meshing_request(MeshingLimits(maximum_cells=2)) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        with pytest.raises(MeshingFailure) as failed:
            extract_surface_quads(
                source,
                MeshingLimits(maximum_cells=2),
                scope=execution_scope,
                budget=coordinate_budget,
            )
        assert failed.value.category is MeshingFailureCategory.RESOURCE_EXHAUSTED
        np.testing.assert_array_equal(source.coordinates, original)


@pytest.mark.parametrize(
    "limit",
    (
        {"maximum_vertices": 6},
        {"maximum_connectivity_entries": 11},
        {"maximum_work_units": 31},
        {"maximum_scratch_bytes": 1},
        {"maximum_data_bytes": 1},
    ),
)
def test_quad_resource_refusal_keeps_original_scientific_source(
    limit: dict[str, int],
) -> None:
    source = CellMesh.from_triangles(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float64),
        np.asarray(((0, 1, 2),), dtype=np.int64),
    )
    identity = source.mesh_id
    coordinates = np.asarray(source.coordinates).copy()
    topology = np.asarray(source.blocks[0].vertices).copy()
    with native_meshing_request(MeshingLimits(**limit)) as (
        execution_scope,
        coordinate_budget,
        host_workspace,
    ):
        with pytest.raises(MeshingFailure) as failed:
            extract_surface_quads(
                source,
                MeshingLimits(**limit),
                scope=execution_scope,
                budget=coordinate_budget,
            )
        assert failed.value.category is MeshingFailureCategory.RESOURCE_EXHAUSTED
        assert source.mesh_id == identity
        np.testing.assert_array_equal(source.coordinates, coordinates)
        np.testing.assert_array_equal(source.blocks[0].vertices, topology)


def test_quad_tangent_source_refuses_without_replacing_source_geometry() -> None:
    source = tangent_triangle(0.0)
    coordinates = np.asarray(source.coordinates).copy()
    limits = MeshingLimits()
    with native_meshing_request(limits) as (scope, budget, _):
        with pytest.raises(MeshingFailure):
            extract_surface_quads(source, limits, scope=scope, budget=budget)
        np.testing.assert_array_equal(source.coordinates, coordinates)


def test_quad_thin_source_validity_does_not_replace_source_geometry() -> None:
    height = 2.0**-10
    source = tangent_triangle(height)
    coordinates = np.asarray(source.coordinates).copy()
    limits = MeshingLimits()
    with native_meshing_request(limits) as (scope, budget, _):
        extraction = extract_surface_quads(source, limits, scope=scope, budget=budget)
        assert (
            extraction.validity.invalid_count == extraction.validity.unresolved_count == 0
        )
        np.testing.assert_allclose(
            _area(extraction.mesh), height / 2.0, rtol=AREA_CONSERVATION_RTOL, atol=0.0
        )
        np.testing.assert_array_equal(source.coordinates, coordinates)


@pytest.mark.parametrize(
    "family_policy",
    (
        pytest.param(
            phx.meshing.CellFamilyPolicy(required=("quadrilateral",)), id="required-quads"
        ),
        pytest.param(
            phx.meshing.CellFamilyPolicy(
                preferred=("quadrilateral",),
                allowed_transitions=("triangle",),
                allow_mixed=True,
            ),
            id="preferred-quads-with-triangle-transitions",
        ),
    ),
)
def test_general_feature_quad_publication_preserves_concave_source_and_partition(
    family_policy: phx.meshing.CellFamilyPolicy,
) -> None:
    meshing = phx.meshing
    region = phx.geometry.PlanarMeshRegion(
        np.asarray(
            ((0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0), (1.0, 2.0), (0.0, 2.0)),
            dtype=np.float64,
        ),
        ((0, 1, 2, 3, 4, 5),),
        feature_id="feature-L",
    )
    partition = phx.geometry.SegmentMesh(
        jnp.asarray(((0.0, 1.0), (1.0, 1.0)), dtype=jnp.float64),
        jnp.asarray(((0, 1),), dtype=jnp.int64),
        source_id="partition",
    )
    source = meshing.NativePlanarSource(region, "original-r1", embedded=partition)
    scope = meshing.MeshingScope(
        source.source_id,
        source.source_revision,
        meshing.MeshingEntityKind.GEOMETRY,
        2,
        "L-region",
        np.asarray((0,), dtype=np.int64),
    )
    specification = meshing.SurfaceMeshingSpec(
        meshing.CellMeshingTarget(
            2,
            2,
            family_policy,
        ),
        scope,
        planar_embedding=phx.geometry.PlanarEmbedding(
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
        ),
        size_controls=(
            meshing.UniformSizeControl(
                scope, 0.8, strength=meshing.SizeControlStrength.SOFT
            ),
        ),
    )
    result = (
        meshing.NativeMeshingProvider(
            meshing.NativeMeshingOptions("planar_dual_quad"),
        )
        .plan(
            source, specification, coordinate_contract=phx.SpatialCoordinateContract.si()
        )
        .execute()
    )
    assert {block.cell_kind for block in result.mesh.blocks} == {"quadrilateral"}
    np.testing.assert_allclose(_area(result.mesh), 3.0, rtol=AREA_CONSERVATION_RTOL)
    assert result.audit.passed and result.compliance.passed
    certification = result.certification
    assert certification is not None and certification.passed
    coverage = certification.coverage
    assert coverage is not None
    assert isinstance(coverage, DomainCoverageCertificate)
    measures = tuple(
        value for value in coverage.achieved_region_measures if value is not None
    )
    assert len(measures) == len(coverage.achieved_region_measures)
    np.testing.assert_allclose(
        np.asarray(measures, dtype=np.float64), (3.0,), rtol=AREA_CONSERVATION_RTOL
    )
    binding = result.trace.binding
    assert binding is not None and binding.source_revision == "original-r1"
    (label,) = (label for label in result.labels if label.name == "embedded")
    entities = result.mesh.topology.entities(1)
    rows = np.isin(np.asarray(entities.entity_ids), np.asarray(label.scope.entity_ids))
    assert isinstance(result.mesh.connectivity, PolygonalConnectivity)
    edges = np.asarray(result.mesh.connectivity.edges, dtype=np.int64)[rows]
    points = np.asarray(result.mesh.coordinates)[edges]
    np.testing.assert_allclose(points[..., 1], 1.0, atol=0.0)
    np.testing.assert_allclose(
        np.sum(np.linalg.norm(np.diff(points, axis=1)[:, 0], axis=1)), 1.0
    )


def test_curved_quad_pullbacks_preserve_source_polynomials_and_rational_centers() -> None:

    parameter = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float64
    )
    triangles = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
    corners = parameter.copy()
    corners[:, 1] += 0.25 * corners[:, 0] ** 2
    mesh = CellMesh.from_triangles(corners, triangles)
    element = coordinate_lagrange_element("triangle", 2)
    nodes = np.asarray(element.reference_nodes)
    barycentric = np.column_stack((1.0 - np.sum(nodes, axis=1), nodes))
    coefficients = barycentric @ parameter[triangles]
    coefficients[..., 1] += 0.25 * coefficients[..., 0] ** 2
    name = mesh.blocks[0].name
    source = CellGeometrySpec(
        {name: element},
        {name: np.arange(12, dtype=np.int64).reshape(2, 6)},
        coefficients.reshape(-1, 2),
    )
    with (
        native_meshing_request(MeshingLimits()) as (
            execution_scope,
            coordinate_budget,
            host_workspace,
        ),
        coordinate_reference_budget() as reference_budget,
    ):
        field = prepare_surface_cross_field(
            mesh, source_geometry=source, scope=execution_scope, budget=coordinate_budget
        )
        tangent = np.asarray((1.0, 1.0 / 3.0, 0.0), dtype=np.float64)
        np.testing.assert_allclose(
            field.frames[0, 0],
            tangent / np.linalg.norm(tangent),
            atol=SOURCE_TABULATION_ATOL,
        )
        extraction = extract_surface_quads(
            mesh,
            MeshingLimits(),
            source_geometry=source,
            cross_field=field,
            scope=execution_scope,
            budget=coordinate_budget,
        )
        geometry = extraction.geometry
        assert geometry is not None
        assert extraction.validity.certified_valid_count == 6
        np.testing.assert_array_equal(geometry.coordinates, source.coordinates)
        ancestry = geometry.restriction_source
        assert ancestry is not None and ancestry.source_geometry_id == cell_geometry_id(
            source
        )
        elements, routes, bank = geometry.resolve(extraction.mesh)
        vertices = np.asarray(reference_cell_topology("triangle").vertices)
        references = tuple(
            np.stack(
                (
                    vertices[corner],
                    0.5 * (vertices[corner] + vertices[(corner + 1) % 3]),
                    np.mean(vertices, axis=0),
                    0.5 * (vertices[(corner - 1) % 3] + vertices[corner]),
                )
            )
            for corner in range(3)
        )
        query = np.asarray(((0.2, 0.3), (0.7, 0.8)), dtype=np.float64)
        x, y = query.T
        weights = np.column_stack(
            ((1.0 - x) * (1.0 - y), x * (1.0 - y), x * y, (1.0 - x) * y)
        )
        for index, (composed, route) in enumerate(zip(elements, routes, strict=True)):
            assert isinstance(composed, PolynomialComposedCellGeometryElement)
            local = weights @ references[index]
            global_points = (
                np.column_stack((1.0 - local.sum(axis=1), local)) @ parameter[triangles]
            )
            global_points[..., 1] += 0.25 * global_points[..., 0] ** 2
            actual = (
                np.asarray(composed.tabulate(query)[0])
                @ np.asarray(bank)[np.asarray(route)]
            )
            np.testing.assert_allclose(
                actual, global_points, rtol=0.0, atol=RATIONAL_MIDPOINT_ATOL
            )
            np.testing.assert_array_equal(
                ancestry.parent_cell_ids[index], mesh.blocks[0].global_ids
            )
        exact = coordinate_corner_images(
            elements[0],
            np.asarray(bank)[np.asarray(routes[0])[0]],
            budget=reference_budget,
        )
        assert exact is not None and exact[2] == (Fraction(2, 3), Fraction(4, 9))
        # This curved boundary midpoint is not the carrier's straight chord midpoint.
        midpoint = next(
            row
            for row, support in enumerate(extraction.vertex_supports.tolist())
            if sorted(value for value in support if value >= 0) == [0, 1]
        )
        np.testing.assert_allclose(
            extraction.mesh.coordinates[midpoint], (0.5, 0.0625), atol=0.0, rtol=0.0
        )


class _RationalQuadFixture(NamedTuple):
    controls: np.ndarray
    root: SplineCellGeometryElement
    root_geometry: CellGeometrySpec
    patch: BSplineSurfacePatch
    result: DualExtraction


@pytest.fixture(scope="module")
def rational_spline_quad() -> _RationalQuadFixture:
    prepared = rational_quad_source()
    limits = MeshingLimits()
    with native_meshing_request(limits) as (scope, budget, _):
        result = extract_surface_quads(
            prepared.mesh,
            limits,
            source_geometry=prepared.geometry,
            scope=scope,
            budget=budget,
        )
    return _RationalQuadFixture(
        prepared.controls.reshape((2, 3, 3)),
        prepared.root,
        prepared.root_geometry,
        prepared.patch,
        result,
    )


@pytest.fixture(scope="module")
def restored_rational_spline_quad(
    rational_spline_quad: _RationalQuadFixture, tmp_path_factory: pytest.TempPathFactory
) -> CellGeometrySpec:
    actual_geometry = rational_spline_quad.result.geometry
    if actual_geometry is None:
        raise RuntimeError(
            "rational_spline_quad.geometry must retain a coordinate map; got None."
        )
    register_meshing_source_artifacts()
    limits = ArrayArchiveLimits(
        max_members=4096, max_manifest_nesting=128, max_manifest_bytes=1 << 26
    )
    recipe = model_structure_recipe(actual_geometry)
    arrays = pack_model_array_tree(
        actual_geometry, recipe, prefix="original-source", limits=limits
    )
    path = tmp_path_factory.mktemp("rational-quad") / "rational-source.phx"
    write_array_archive(path, manifest={"recipe": recipe}, arrays=arrays, limits=limits)
    manifest, arrays = read_array_archive(path, limits=limits)
    restored = model_from_logical_array_recipe(
        manifest["recipe"], arrays, prefix="original-source", limits=limits
    )
    if not isinstance(restored, CellGeometrySpec):
        raise TypeError(
            f"restored_rational_spline_quad.geometry must be CellGeometrySpec; got {type(restored).__name__}."
        )
    return restored


def test_rational_spline_quad_archive_preserves_source_identity(
    rational_spline_quad: _RationalQuadFixture,
    restored_rational_spline_quad: CellGeometrySpec,
) -> None:
    restored = restored_rational_spline_quad
    actual_geometry = rational_spline_quad.result.geometry
    controls = rational_spline_quad.controls
    assert actual_geometry is not None
    assert isinstance(restored, CellGeometrySpec)
    assert cell_geometry_id(restored) == cell_geometry_id(actual_geometry)
    np.testing.assert_array_equal(restored.coordinates, controls.reshape(-1, 3))


def test_rational_spline_quad_cold_charts_preserve_source_values(
    rational_spline_quad: _RationalQuadFixture,
    restored_rational_spline_quad: CellGeometrySpec,
) -> None:
    actual_geometry = restored_rational_spline_quad
    controls, patch, result = (
        rational_spline_quad.controls,
        rational_spline_quad.patch,
        rational_spline_quad.result,
    )
    with coordinate_reference_budget() as reference_budget:
        query = np.asarray(((0.2, 0.3), (0.7, 0.8)), dtype=np.float64)
        for composed, route in zip(
            actual_geometry.elements, actual_geometry.geometry_dofs, strict=True
        ):
            assert isinstance(composed, PolynomialComposedCellGeometryElement)
            original = composed.source_element
            assert isinstance(original, RestrictedCellGeometryElement)
            chart = np.asarray(composed.chart_element.tabulate(query)[0]) @ np.asarray(
                composed.chart_coordinates
            )
            parameters = chart @ np.asarray(original.matrix).T + np.asarray(
                original.offset
            )
            expected = np.asarray(patch.evaluate(parameters))
            actual = (
                np.asarray(composed.tabulate(query)[0])
                @ np.asarray(actual_geometry.coordinates)[np.asarray(route)[0]]
            )
            np.testing.assert_allclose(
                actual, expected, rtol=0.0, atol=SOURCE_TABULATION_ATOL
            )
        expressions = coordinate_expressions(
            actual_geometry.elements[0], controls.reshape(-1, 3), budget=reference_budget
        )
        assert expressions is not None
        assert not isinstance(expressions[0], RationalPolynomial)
        assert isinstance(expressions[1], RationalPolynomial) and isinstance(
            expressions[2], RationalPolynomial
        )
        midpoint = next(
            row
            for row, support in enumerate(result.vertex_supports.tolist())
            if sorted(value for value in support if value >= 0) == [1, 2]
        )
        np.testing.assert_allclose(
            result.mesh.coordinates[midpoint],
            (1.0, np.sqrt(0.5), np.sqrt(0.5)),
            rtol=0.0,
            atol=RATIONAL_MIDPOINT_ATOL,
        )


def test_rational_spline_quad_charts_retain_weights_controls_and_roots(
    rational_spline_quad: _RationalQuadFixture,
) -> None:
    controls, root, root_geometry, result = (
        rational_spline_quad.controls,
        rational_spline_quad.root,
        rational_spline_quad.root_geometry,
        rational_spline_quad.result,
    )
    with coordinate_reference_budget() as reference_budget:
        actual_geometry = result.geometry
        assert actual_geometry is not None
        assert result.validity.certified_valid_count == 6
        assert result.validity.invalid_count == result.validity.unresolved_count == 0
        assert source_basis(root, budget=reference_budget) is None
        np.testing.assert_array_equal(
            actual_geometry.coordinates, controls.reshape(-1, 3)
        )
        actual_ancestry = actual_geometry.restriction_source
        assert actual_ancestry is not None
        assert actual_ancestry.source_geometry_id == cell_geometry_id(root_geometry)


def test_nonbinary_tensor_chart_retains_authored_order_under_node_budget() -> None:

    source = coordinate_lagrange_element("hexahedron", 2)
    chart = coordinate_lagrange_element("hexahedron", 1)
    corners = np.asarray(reference_cell_topology("hexahedron").vertices, dtype=np.int64)
    numerators = np.column_stack(
        (
            2 + 3 * corners[:, 2],
            3 + 5 * corners[:, 0],
            4 + 7 * corners[:, 1],
        )
    )
    denominators = np.tile(np.asarray((6, 15, 28), dtype=np.int64), (8, 1))
    with coordinate_reference_budget() as reference_budget:
        composed = PolynomialComposedCellGeometryElement(
            source, chart, numerators, denominators, budget=reference_budget
        )
        controls = np.asarray(source.reference_nodes).copy()
        controls[:, 2] += 0.125 * controls[:, 0] ** 2 * controls[:, 1]
        exact = coordinate_corner_images(composed, controls, budget=reference_budget)
        assert exact is not None
        mesh = CellMesh(
            np.stack(tuple(rounded_point(value) for value in exact)),
            (CellBlock("target", "hexahedron", np.arange(8, dtype=np.int64)[None, :]),),
        )
        geometry = CellGeometrySpec(
            {"target": composed},
            {"target": np.arange(controls.shape[0], dtype=np.int64)[None, :]},
            controls,
        )
        validity = certify_cell_geometry_validity(
            geometry,
            mesh=mesh,
            policy=CellValidityPolicy(maximum_bernstein_nodes=216),
        )
        assert validity.certified_valid_count == 1
        assert validity.invalid_count == validity.unresolved_count == 0
        query = np.asarray(((0.2, 0.3, 0.4), (0.7, 0.6, 0.5)), dtype=np.float64)
        reference = np.column_stack(
            (
                1.0 / 3.0 + query[:, 2] / 2.0,
                1.0 / 5.0 + query[:, 0] / 3.0,
                1.0 / 7.0 + query[:, 1] / 4.0,
            )
        )
        np.testing.assert_allclose(
            np.asarray(composed.tabulate(query)[0]) @ controls,
            np.asarray(source.tabulate(reference)[0]) @ controls,
            rtol=0.0,
            atol=SOURCE_TABULATION_ATOL,
        )
        mixed = corners.copy()
        mixed[:, 0] += corners[:, 1]
        shear = PolynomialComposedCellGeometryElement(
            source, chart, mixed, np.ones((8, 3), dtype=np.int64), budget=reference_budget
        )
        assert composed.degree == 2 and shear.degree == 4


@pytest.mark.parametrize(
    "archive_limits",
    (
        pytest.param(
            ArrayArchiveLimits(
                max_members=4096, max_manifest_nesting=128, max_manifest_bytes=1 << 26
            ),
            id="expanded-archive-limits",
        ),
        pytest.param(ArrayArchiveLimits(), id="default-archive-limits"),
    ),
)
def test_public_spline_quads_keep_source_identity_after_cold_restore(
    tmp_path: Path, archive_limits: ArrayArchiveLimits
) -> None:

    result = generate_source_quads()
    assert {block.cell_kind for block in result.mesh.blocks} == {"quadrilateral"}
    assert result.audit.passed and result.compliance.passed
    certification = result.certification
    assert certification is not None and certification.passed
    fidelity = certification.fidelity
    assert fidelity is not None
    assert fidelity.mesh_to_source_upper == fidelity.source_to_mesh_upper == 0.0
    retained = certification.request
    assert retained is not None
    source = retained.source
    assert isinstance(source, SurfaceNativeRestrictionBoundarySource)
    original = source.support.original
    assert isinstance(original, SurfaceSourceRootAtlas)
    assert original.domain.source_id == "original-quarter-extrusion"
    assert original.domain.source_revision == "original-r1"
    np.testing.assert_array_equal(
        result.geometry.coordinates, original.geometry.coordinates
    )
    limits = archive_limits
    receipt = write_meshing_source_closure(
        tmp_path / "original-spline-quads.phx", result, limits=limits
    )
    restored = read_meshing_source_closure(
        receipt.path, expected_content_id=receipt.content_id, limits=limits
    )
    assert isinstance(restored, CellMeshingResult)
    assert cell_geometry_id(restored.geometry) == cell_geometry_id(result.geometry)
    assert restored.certification is not None and restored.certification.passed
    restored_inputs = restored.certification.request
    assert restored_inputs is not None
    restored_source = restored_inputs.source
    assert isinstance(restored_source, SurfaceNativeRestrictionBoundarySource)
    restored_original = restored_source.support.original
    assert isinstance(restored_original, SurfaceSourceRootAtlas)
    assert restored_original.atlas_id == original.atlas_id
    # Re-evaluate measured restored maps against the immutable source bank.
    query = np.asarray(((0.2, 0.3), (0.7, 0.8)), dtype=np.float64)
    first = result.geometry.resolve(result.mesh)
    second = restored.geometry.resolve(restored.mesh)
    for before, after, before_route, after_route in zip(
        first[0], second[0], first[1], second[1], strict=True
    ):
        before = require_scalar_coordinate_element(before, "Original source regression")
        after = require_scalar_coordinate_element(after, "Restored source regression")
        np.testing.assert_array_equal(
            np.asarray(before.tabulate(query)[0])
            @ np.asarray(first[2])[np.asarray(before_route)[0]],
            np.asarray(after.tabulate(query)[0])
            @ np.asarray(second[2])[np.asarray(after_route)[0]],
        )
    # Exercise the restored coefficient root binding in an measured surface PDE,
    # not merely by checking the quadrilateral family or archive identity.
    policy = phx.linalg.LinearSolvePolicy(
        phx.linalg.DenseLU(),
        tolerance=phx.linalg.TolerancePolicy(
            relative=RESTORE_SOLVE_RTOL, absolute=RESTORE_SOLVE_ATOL
        ),
    )
    before_error = solve_linear_diffusion(
        result,
        2,
        harmonic_direction=(1.0, 0.0, 0.0),
        solve_policy=policy,
        cell_quadrature_order=8,
    )
    after_error = solve_linear_diffusion(
        restored,
        2,
        harmonic_direction=(1.0, 0.0, 0.0),
        solve_policy=policy,
        cell_quadrature_order=8,
    )
    assert (
        before_error <= RESTORE_PDE_ERROR_LIMIT and after_error <= RESTORE_PDE_ERROR_LIMIT
    )
    # Independent finite-volume preparation consumes the same restored curved
    # source. The source extrusion is a Cartesian product with x in [0, 1],
    # so its area-weighted integral of x² is exactly one third of its area;
    # this does not assume that rounded rational weights define a unit circle.
    for publication in (result, restored):
        fv = phx.discretization.UnstructuredFiniteVolumePlan.from_cell_mesh(
            publication.mesh,
            component_names=("u",),
        ).prepare(cell_geometry=publication.geometry)
        assert np.all(np.asarray(fv.cell_volumes) > 0.0)
        points = np.asarray(fv.cell_quadrature_points)
        weights = np.asarray(fv.cell_quadrature_weights)
        valid = np.asarray(fv.cell_quadrature_valid)
        area = float(np.sum(np.where(valid, weights, 0.0)))
        axial_content = float(np.sum(np.where(valid, weights * points[..., 0] ** 2, 0.0)))
        np.testing.assert_allclose(
            axial_content, area / 3.0, rtol=RESTORE_QUADRATURE_RTOL, atol=0.0
        )
        np.testing.assert_allclose(
            np.sum(np.asarray(fv.cell_volumes)),
            area,
            rtol=AREA_CONSERVATION_RTOL,
            atol=0.0,
        )


def test_quad_route_does_not_substitute_for_required_triangle_family() -> None:
    meshing = phx.meshing
    region = phx.geometry.PlanarMeshRegion(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float64),
        ((0, 1, 2, 3),),
        feature_id="required-families",
    )
    source = meshing.NativePlanarSource(region, "original-r1")
    scope = meshing.MeshingScope(
        source.source_id,
        source.source_revision,
        meshing.MeshingEntityKind.GEOMETRY,
        2,
        "square",
        np.asarray((0,), dtype=np.int64),
    )
    request = meshing.SurfaceMeshingSpec(
        meshing.CellMeshingTarget(
            2,
            2,
            meshing.CellFamilyPolicy(
                required=("quadrilateral", "triangle"),
                allow_mixed=True,
            ),
        ),
        scope,
        planar_embedding=phx.geometry.PlanarEmbedding(
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
        ),
        size_controls=(
            meshing.UniformSizeControl(
                scope, 0.8, strength=meshing.SizeControlStrength.SOFT
            ),
        ),
    )
    provider = meshing.NativeMeshingProvider(
        meshing.NativeMeshingOptions("planar_dual_quad")
    )
    with pytest.raises(MeshingFailure):
        provider.plan(
            source, request, coordinate_contract=phx.SpatialCoordinateContract.si()
        )


@pytest.mark.parametrize(
    "feature_edges",
    (np.asarray((0,), dtype=np.int64), [0]),
    ids=("host-array", "integer-sequence"),
)
def test_cross_field_feature_rows_accept_conversion_inputs(
    feature_edges: ConvertibleToArray,
) -> None:
    source = tangent_triangle(1.0)
    limits = MeshingLimits()
    with native_meshing_request(limits) as (scope, budget, _):
        field = prepare_surface_cross_field(
            source,
            feature_edges=feature_edges,
            maximum_steps=2,
            scope=scope,
            budget=budget,
        )
    assert field.source_topology_id == source.topology_id
    assert field.source_geometry_id == source.geometry_id
    assert field.optimization.diagnostics.counts_complete
