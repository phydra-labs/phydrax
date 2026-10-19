#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Test scientific parent frames of planar P1 triangle restrictions."""

from fractions import Fraction

import numpy as np
import pytest
from numpy.typing import NDArray

from phydrax.discretization import (
    CellBlock,
    CellGeometrySpec,
    CellMesh,
    certify_cell_geometry_validity,
    ExactPlcCellGeometrySource,
    PolygonalConnectivity,
    transition_nested_cell_geometry,
)
from phydrax.discretization._cell_geometry import (
    CellGeometryRestrictionSource,
    coordinate_lagrange_element,
    PolynomialComposedCellGeometryElement,
    RestrictedCellGeometryElement,
)
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.discretization.fem._reference import lagrange_element
from phydrax.discretization.fem._reference_types import FiniteElementSpec
from phydrax.geometry import (
    certify_domain_coverage,
    certify_global_embedding,
    GlobalEmbeddingCertificate,
    PiecewiseLinearDomain,
)
from phydrax.meshing._contracts import MeshingLimits
from phydrax.meshing._mixed_adaptation import adapt_mixed_mesh
from phydrax.meshing._quad_generation import extract_surface_quads
from phydrax.meshing._topology_edit import assemble_topology_edit
from tests._support.coordinate_references import (
    coordinate_reference_budget,
    native_meshing_request,
)


_POINTS = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float64)
_ROUTES = np.asarray(((0, 1, 2), (1, 2, 0), (2, 0, 1)), dtype=np.int32)
_PARENT_VERTICES = np.asarray(((37, 41, 43), (41, 43, 37), (43, 37, 41)), dtype=np.int64)
_CELL_IDS = np.asarray((101, 103, 107), dtype=np.int64)
_ZERO, _ONE, _HALF, _THIRD = Fraction(0), Fraction(1), Fraction(1, 2), Fraction(1, 3)
_CHARTS = (
    ((_ZERO, _ZERO), (_HALF, _ZERO), (_THIRD, _THIRD), (_ZERO, _HALF)),
    ((_ONE, _ZERO), (_HALF, _HALF), (_THIRD, _THIRD), (_HALF, _ZERO)),
    ((_ZERO, _ONE), (_ZERO, _HALF), (_THIRD, _THIRD), (_HALF, _HALF)),
)


def _exact_source(points: NDArray[np.float64]) -> ExactPlcCellGeometrySource:
    with coordinate_reference_budget() as budget:
        return ExactPlcCellGeometrySource(
            points,
            np.empty((0, 3), dtype=np.int64),
            np.asarray(((0, 1), (1, 2), (2, 0)), dtype=np.int64),
            np.zeros(points.shape[0], dtype=np.int64),
            np.full(points.shape[0], -1, dtype=np.int64),
            np.zeros((points.shape[0], 2), dtype=np.float64),
            domain_source_id="cyclic-parent-triangle",
            domain_source_revision="exact-corners",
            source_triangle_ids=np.empty(0, dtype=np.int64),
            source_triangle_bounds=np.empty(0, dtype=np.float64),
            source_segment_ids=np.asarray((11, 13, 17), dtype=np.int64),
            source_segment_bounds=np.zeros(3, dtype=np.float64),
            budget=budget,
        )


def _composed(
    source: FiniteElementSpec, corner: int = 0
) -> PolynomialComposedCellGeometryElement:
    points = _CHARTS[corner]
    with coordinate_reference_budget() as budget:
        return PolynomialComposedCellGeometryElement(
            source,
            coordinate_lagrange_element("quadrilateral", 1),
            np.asarray(
                [[value.numerator for value in row] for row in points], dtype=np.int64
            ),
            np.asarray(
                [[value.denominator for value in row] for row in points], dtype=np.int64
            ),
            budget=budget,
        )


def _child_mesh(
    points: NDArray[np.float64],
    routes: NDArray[np.int32],
    *,
    shared_vertices: bool,
) -> CellMesh:
    # Independent affine corner images: the triangle barycentric law determines
    # the carrier, without calling the package's source-expression extraction.
    exact = tuple(tuple(Fraction(float(value)) for value in row) for row in points)
    positions: dict[tuple[Fraction, ...], int] = {}
    corners: list[tuple[Fraction, ...]] = []
    cells: list[list[int]] = []
    for route in routes:
        vertices = []
        for u, v in _CHARTS[0]:
            weights = (1 - u - v, u, v)
            image = tuple(
                sum(
                    (
                        weight * exact[int(index)][axis]
                        for weight, index in zip(weights, route, strict=True)
                    ),
                    Fraction(0),
                )
                for axis in range(2)
            )
            if shared_vertices and image in positions:
                vertex = positions[image]
            else:
                vertex = len(corners)
                corners.append(image)
                positions[image] = vertex
            vertices.append(vertex)
        cells.append(vertices)
    return CellMesh(
        np.asarray(
            [[float(value) for value in row] for row in corners], dtype=np.float64
        ),
        (
            CellBlock(
                "children",
                "quadrilateral",
                np.asarray(cells, dtype=np.int32),
                global_ids=_CELL_IDS,
            ),
        ),
    )


def _case(
    routes: NDArray[np.int32],
    parent_vertices: NDArray[np.int64],
    /,
    *,
    coordinates: NDArray[np.float64] | None = None,
    plc: bool = True,
    shared_vertices: bool = True,
) -> tuple[CellMesh, CellGeometrySpec]:
    points = _POINTS if coordinates is None else coordinates
    source = coordinate_lagrange_element("triangle", 1)
    exact_source = _exact_source(points) if plc else None
    parent_mesh = CellMesh(
        points[:3],
        (
            CellBlock(
                "parent",
                "triangle",
                np.asarray(((0, 1, 2),), dtype=np.int32),
                global_ids=np.asarray((71,), dtype=np.int64),
            ),
        ),
        vertex_global_ids=_PARENT_VERTICES[0],
    )
    parent_geometry = CellGeometrySpec(
        {"parent": source},
        {"parent": np.asarray(((0, 1, 2),), dtype=np.int32)},
        points,
        exact_source=exact_source,
    )
    ancestry = CellGeometryRestrictionSource(
        cell_geometry_id(parent_geometry),
        parent_mesh.topology_id,
        {"children": np.full(3, 71, dtype=np.int64)},
        {"children": parent_vertices},
    )
    geometry = CellGeometrySpec(
        {"children": _composed(source)},
        {"children": routes},
        points,
        restriction_source=ancestry,
        exact_source=exact_source,
    )
    return _child_mesh(points, routes, shared_vertices=shared_vertices), geometry


def _embedding(mesh: CellMesh, geometry: CellGeometrySpec) -> GlobalEmbeddingCertificate:
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    assert validity.all_certified
    return certify_global_embedding(mesh, geometry, validity)


def _assert_identity_refusal(certificate: GlobalEmbeddingCertificate) -> None:
    assert certificate.status == "violated"
    (finding,) = tuple(
        finding
        for finding in certificate.findings
        if finding.check == "restriction_source_identity_mismatch"
    )
    assert finding.status == "violated"
    assert finding.entity_kind == "cell"
    assert finding.entity_ids == (103,)


@pytest.mark.parametrize("first_corner", (0, 1, 2), ids=("corner0", "corner1", "corner2"))
def test_plc_cyclic_parent_frames_jointly_certify_triangle_cover(
    first_corner: int,
) -> None:
    routes = np.roll(_ROUTES, -first_corner, axis=0)
    parent_vertices = np.roll(_PARENT_VERTICES, -first_corner, axis=0)
    mesh, geometry = _case(routes, parent_vertices)
    ancestry = geometry.restriction_source
    assert ancestry is not None
    identity = cell_geometry_id(geometry)
    embedding = _embedding(mesh, geometry)
    domain = PiecewiseLinearDomain(
        _POINTS,
        np.asarray(((0, 1), (1, 2), (2, 0)), dtype=np.int32),
        np.asarray(((0, -1), (0, -1), (0, -1)), dtype=np.int64),
        ("triangle",),
        source_id="independent-triangle-boundary",
    )
    coverage = certify_domain_coverage(
        mesh, geometry, domain, np.zeros(3, dtype=np.int64), embedding=embedding
    )

    assert embedding.status == "certified"
    assert embedding.findings == ()
    assert "mapped_restriction_embedding" in embedding.evaluated_checks
    assert coverage.status == "certified"
    assert coverage.findings == ()
    assert coverage.requested_region_measures == (0.5,)
    assert coverage.achieved_region_measures == (0.5,)
    assert coverage.integration_error_bounds == (0.0,)
    assert coverage.covered_source_facet_count == 3
    assert geometry.restriction_source is ancestry
    assert cell_geometry_id(geometry) == identity
    np.testing.assert_array_equal(
        ancestry.block_parent_cell_ids["children"], (71, 71, 71)
    )
    np.testing.assert_array_equal(
        ancestry.block_parent_vertex_ids["children"], parent_vertices
    )
    np.testing.assert_array_equal(geometry.geometry_dofs[0], routes)


@pytest.mark.parametrize(
    "parent_order",
    (
        pytest.param((41, 43, 47), id="changed-vertex-set"),
        pytest.param((41, 37, 43), id="reflected-order"),
    ),
)
def test_plc_parent_frame_rejects_changed_scientific_corners(
    parent_order: tuple[int, int, int],
) -> None:
    parent_vertices = _PARENT_VERTICES.copy()
    parent_vertices[1] = parent_order
    mesh, geometry = _case(_ROUTES, parent_vertices)
    _assert_identity_refusal(_embedding(mesh, geometry))


def test_plc_parent_frame_rejects_nonpermuted_coefficients() -> None:
    routes = _ROUTES.copy()
    routes[1] = (0, 1, 2)
    mesh, geometry = _case(routes, _PARENT_VERTICES, shared_vertices=False)
    _assert_identity_refusal(_embedding(mesh, geometry))


def test_plc_parent_frame_rejects_one_ulp_coefficient_change() -> None:
    points = np.concatenate((_POINTS, _POINTS[1:2]), axis=0)
    points[3, 0] = np.nextafter(1.0, np.inf)
    routes = _ROUTES.copy()
    routes[1, 0] = 3
    mesh, geometry = _case(
        routes, _PARENT_VERTICES, coordinates=points, shared_vertices=False
    )
    _assert_identity_refusal(_embedding(mesh, geometry))


def test_plc_parent_frame_rejects_foreign_source_basis() -> None:
    _, geometry = _case(_ROUTES, _PARENT_VERTICES)
    foreign = lagrange_element("triangle", 1)
    assert foreign.element_id != coordinate_lagrange_element("triangle", 1).element_id
    composed = _composed(foreign)
    with pytest.raises(ValueError, match=r"require_p1_cardinal_source\.inputs"):
        CellGeometrySpec(
            {"children": composed},
            {"children": _ROUTES},
            _POINTS,
            restriction_source=geometry.restriction_source,
            exact_source=geometry.exact_source,
        )


def test_non_plc_parent_frame_still_rejects_cyclic_order() -> None:
    mesh, geometry = _case(_ROUTES, _PARENT_VERTICES, plc=False)
    _assert_identity_refusal(_embedding(mesh, geometry))


@pytest.mark.parametrize(
    "foreign", (False, True), ids=("canonical-source", "foreign-source")
)
def test_non_plc_common_parent_frame_preserves_source_identity(foreign: bool) -> None:
    carrier, cyclic_geometry = _case(_ROUTES, _PARENT_VERTICES, plc=False)
    origin = cyclic_geometry.restriction_source
    assert origin is not None
    blocks = tuple(
        CellBlock(
            f"corner{corner}",
            "quadrilateral",
            np.asarray(carrier.blocks[0].vertices)[corner : corner + 1],
            global_ids=_CELL_IDS[corner : corner + 1],
        )
        for corner in range(3)
    )
    mesh = CellMesh(carrier.coordinates, blocks)
    canonical = coordinate_lagrange_element("triangle", 1)
    second = lagrange_element("triangle", 1) if foreign else canonical
    ancestry = CellGeometryRestrictionSource(
        origin.source_geometry_id,
        origin.source_topology_id,
        {block.name: np.asarray((71,), dtype=np.int64) for block in blocks},
        {block.name: _PARENT_VERTICES[:1] for block in blocks},
    )
    geometry = CellGeometrySpec(
        {
            block.name: _composed(second if corner == 1 else canonical, corner)
            for corner, block in enumerate(blocks)
        },
        {block.name: np.asarray(((0, 1, 2),), dtype=np.int32) for block in blocks},
        _POINTS,
        restriction_source=ancestry,
    )
    certificate = _embedding(mesh, geometry)
    if foreign:
        _assert_identity_refusal(certificate)
    else:
        assert certificate.status == "certified"
        assert certificate.findings == ()


def _dual_from_parents(
    points: NDArray[np.float64], cells: NDArray[np.int32]
) -> tuple[CellMesh, CellGeometrySpec]:
    source = CellMesh(
        points,
        (
            CellBlock(
                "parents",
                "triangle",
                cells,
                global_ids=np.asarray((71, 73), dtype=np.int64),
            ),
        ),
        vertex_global_ids=np.arange(points.shape[0], dtype=np.int64) + 37,
    )
    geometry = CellGeometrySpec(
        {"parents": coordinate_lagrange_element("triangle", 1)},
        {"parents": cells},
        points,
        exact_source=_exact_source(points),
    )
    limits = MeshingLimits()
    with native_meshing_request(limits) as (scope, budget, _workspace):
        result = extract_surface_quads(
            source, limits, source_geometry=geometry, scope=scope, budget=budget
        )
    assert result.validity.all_certified
    assert result.geometry is not None
    return result.mesh, result.geometry


def test_adjacent_plc_triangle_parents_have_exact_joint_cover() -> None:
    points = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float64
    )
    mesh, geometry = _dual_from_parents(
        points, np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32)
    )
    embedding = _embedding(mesh, geometry)
    domain = PiecewiseLinearDomain(
        points,
        np.asarray(((0, 1), (1, 2), (2, 3), (3, 0)), dtype=np.int32),
        np.asarray(((0, -1),) * 4, dtype=np.int64),
        ("square",),
        source_id="independent-two-parent-square",
    )
    coverage = certify_domain_coverage(
        mesh, geometry, domain, np.zeros(6, dtype=np.int64), embedding=embedding
    )

    assert embedding.status == "certified"
    assert embedding.findings == ()
    assert coverage.status == "certified"
    assert coverage.findings == ()
    assert coverage.achieved_region_measures == (1.0,)
    assert coverage.integration_error_bounds == (0.0,)
    assert coverage.covered_source_facet_count == 4
    ancestry = geometry.restriction_source
    assert ancestry is not None
    np.testing.assert_array_equal(ancestry.parent_cell_ids[0], (71, 71, 71, 73, 73, 73))


def test_plc_triangle_parent_contact_rejects_shared_corner_trace_change() -> None:
    points = np.asarray(
        (
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 0.125),
            (1.0, 1.0),
            (0.0, 1.0),
        ),
        dtype=np.float64,
    )
    mesh, geometry = _dual_from_parents(
        points, np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int32)
    )
    origin = geometry.restriction_source
    assert origin is not None
    name = mesh.blocks[0].name
    parents = np.asarray(origin.block_parent_cell_ids[name])
    corners = np.asarray(origin.block_parent_vertex_ids[name]).copy()
    # Distinct physical corners may not acquire a shared source vertex identity.
    corners[(parents[:, None] == 73) & (corners == 40)] = 37
    ancestry = CellGeometryRestrictionSource(
        origin.source_geometry_id,
        origin.source_topology_id,
        {name: parents},
        {name: corners},
    )
    altered = CellGeometrySpec(
        dict(zip(geometry.block_names, geometry.elements, strict=True)),
        dict(zip(geometry.block_names, geometry.geometry_dofs, strict=True)),
        geometry.coordinates,
        restriction_source=ancestry,
        exact_source=geometry.exact_source,
    )
    certificate = _embedding(mesh, altered)

    assert certificate.status == "violated"
    (finding,) = tuple(
        finding
        for finding in certificate.findings
        if finding.check == "restriction_root_trace_mismatch"
    )
    assert finding.status == "violated"
    assert finding.entity_kind == "cell"
    assert finding.entity_ids == tuple(range(6))


def test_plc_triangle_parent_contact_keeps_root_overlap_unresolved() -> None:
    points = np.concatenate((_POINTS, _POINTS), axis=0)
    mesh, geometry = _dual_from_parents(
        points, np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int32)
    )
    certificate = _embedding(mesh, geometry)

    assert certificate.status == "unresolved"
    (finding,) = tuple(
        finding
        for finding in certificate.findings
        if finding.check == "restriction_root_embedding_premise"
    )
    assert finding.status == "unresolved"
    assert finding.entity_kind == "cell"
    assert finding.entity_ids == tuple(range(6))


def test_shared_plc_parent_chart_traces_reject_sub_ulp_corner_change() -> None:
    carrier, cyclic_geometry = _case(_ROUTES, _PARENT_VERTICES)
    origin = cyclic_geometry.restriction_source
    assert origin is not None
    blocks = tuple(
        CellBlock(
            f"corner{corner}",
            "quadrilateral",
            np.asarray(carrier.blocks[0].vertices)[corner : corner + 1],
            global_ids=_CELL_IDS[corner : corner + 1],
        )
        for corner in range(3)
    )
    mesh = CellMesh(carrier.coordinates, blocks)
    source = coordinate_lagrange_element("triangle", 1)
    elements = {
        block.name: _composed(source, corner) for corner, block in enumerate(blocks)
    }
    corners = list(_CHARTS[1])
    # This source perturbation rounds to the same carrier corner. An
    # endpoint identity proof must still distinguish the source maps.
    changed = _THIRD + Fraction(1, 2**60)
    assert changed != _THIRD
    assert float(changed) == float(_THIRD)
    corners[2] = changed, _THIRD
    with coordinate_reference_budget() as budget:
        elements[blocks[1].name] = PolynomialComposedCellGeometryElement(
            source,
            coordinate_lagrange_element("quadrilateral", 1),
            np.asarray(
                [[value.numerator for value in row] for row in corners], dtype=np.int64
            ),
            np.asarray(
                [[value.denominator for value in row] for row in corners], dtype=np.int64
            ),
            budget=budget,
        )
    ancestry = CellGeometryRestrictionSource(
        origin.source_geometry_id,
        origin.source_topology_id,
        {block.name: np.asarray((71,), dtype=np.int64) for block in blocks},
        {block.name: _PARENT_VERTICES[:1] for block in blocks},
    )
    geometry = CellGeometrySpec(
        elements,
        {block.name: np.asarray(((0, 1, 2),), dtype=np.int32) for block in blocks},
        _POINTS,
        restriction_source=ancestry,
        exact_source=cyclic_geometry.exact_source,
    )
    certificate = _embedding(mesh, geometry)

    assert certificate.status == "violated"
    mismatches = tuple(
        finding
        for finding in certificate.findings
        if finding.check == "mapped_trace_mismatch"
    )
    connectivity = mesh.connectivity
    assert isinstance(connectivity, PolygonalConnectivity)
    cell_vertices = tuple(
        tuple(np.asarray(block.vertices)[0].tolist()) for block in blocks
    )
    affected_edges = {
        tuple(sorted(set(cell_vertices[a]).intersection(cell_vertices[b])))
        for a, b in ((0, 1), (1, 2))
    }
    expected_facets = {
        ("facet", "violated", (int(identifier),))
        for edge, identifier in zip(
            np.asarray(connectivity.edges),
            np.asarray(mesh.entity_set(1).entity_ids),
            strict=True,
        )
        if tuple(sorted(edge.tolist())) in affected_edges
    }
    assert len(expected_facets) == 2
    # Facet identities are definite violations; the two contact proofs also
    # refuse their missing trace evidence, attributed to the affected cell pair.
    expected = expected_facets | {
        ("cell", "unresolved", (101, 103)),
        ("cell", "unresolved", (103, 107)),
    }
    assert len(mismatches) == len(expected)
    assert {
        (finding.entity_kind, finding.status, finding.entity_ids)
        for finding in mismatches
    } == expected


def _refined_cyclic_case(
    routes: NDArray[np.int32],
    parent_vertices: NDArray[np.int64],
    *,
    shared_vertices: bool = True,
) -> tuple[CellMesh, CellGeometrySpec]:
    source, geometry = _case(routes, parent_vertices, shared_vertices=shared_vertices)
    identity = cell_geometry_id(geometry)
    before = np.asarray(geometry.coordinates).copy()
    limits = MeshingLimits()
    refined = adapt_mixed_mesh(
        source,
        refine_cell_ids=_CELL_IDS,
        maximum_cells=limits.maximum_cells,
        maximum_work_units=limits.maximum_work_units,
        maximum_scratch_bytes=limits.maximum_scratch_bytes,
    )
    with coordinate_reference_budget(limits) as budget:
        target, _, _ = assemble_topology_edit(
            source, refined.edit, numeric_version="cyclic-refinement", budget=budget
        )
        transition = transition_nested_cell_geometry(
            source,
            geometry,
            target,
            CellGeometrySpec.affine(target),
            refinement=refined.edit.refinement,
            budget=budget,
        )
    assert transition.evidence.exact
    assert cell_geometry_id(geometry) == identity
    np.testing.assert_array_equal(geometry.coordinates, before)
    return target.with_coordinates(
        transition.vertex_coordinates, numeric_version=target.numeric_version
    ), transition.geometry


def test_refined_plc_cyclic_parent_frames_certify_joint_cover() -> None:
    mesh, geometry = _refined_cyclic_case(_ROUTES, _PARENT_VERTICES)
    assert sum(block.cell_count for block in mesh.blocks) == 12
    for element in geometry.elements:
        assert isinstance(element, RestrictedCellGeometryElement)
        assert isinstance(element.source_element, PolynomialComposedCellGeometryElement)
    embedding = _embedding(mesh, geometry)
    assert embedding.status == "certified"
    assert embedding.findings == ()
    assert "mapped_restriction_embedding" in embedding.evaluated_checks
    domain = PiecewiseLinearDomain(
        _POINTS,
        np.asarray(((0, 1), (1, 2), (2, 0)), dtype=np.int32),
        np.asarray(((0, -1), (0, -1), (0, -1)), dtype=np.int64),
        ("triangle",),
        source_id="refined-independent-triangle-boundary",
    )
    coverage = certify_domain_coverage(
        mesh, geometry, domain, np.zeros(12, dtype=np.int64), embedding=embedding
    )
    assert coverage.status == "certified"
    assert coverage.findings == ()
    assert coverage.requested_region_measures == (0.5,)
    assert coverage.achieved_region_measures == (0.5,)
    assert coverage.integration_error_bounds == (0.0,)
    assert coverage.covered_source_facet_count == 3
    ancestry = geometry.restriction_source
    assert ancestry is not None
    for parents in ancestry.block_parent_cell_ids.values():
        np.testing.assert_array_equal(parents, np.full(parents.shape, 71, dtype=np.int64))
    np.testing.assert_array_equal(geometry.coordinates, _POINTS)


@pytest.mark.parametrize("tampering", ("parent-frame", "coefficients"))
def test_refined_plc_parent_frame_refuses_incompatible_identity(tampering: str) -> None:
    routes = _ROUTES.copy()
    parent_vertices = _PARENT_VERTICES.copy()
    if tampering == "parent-frame":
        parent_vertices[1] = (41, 37, 43)
    else:
        routes[1] = (0, 1, 2)
    mesh, geometry = _refined_cyclic_case(routes, parent_vertices, shared_vertices=False)
    identity = cell_geometry_id(geometry)
    before = np.asarray(geometry.coordinates).copy()
    certificate = _embedding(mesh, geometry)
    assert certificate.status == "violated"
    (finding,) = certificate.findings
    assert finding.check == "restriction_source_identity_mismatch"
    assert finding.status == "violated"
    assert finding.entity_kind == "cell"
    assert len(finding.entity_ids) == 1
    target_ids = {
        int(identifier)
        for block in mesh.blocks
        for identifier in np.asarray(block.global_ids)
    }
    assert set(finding.entity_ids) <= target_ids
    assert cell_geometry_id(geometry) == identity
    np.testing.assert_array_equal(geometry.coordinates, before)
