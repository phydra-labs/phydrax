from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from phydrax.discretization import lagrange_element
from phydrax.discretization._cell_complex import PolyhedralConnectivity
from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.discretization._cell_geometry_validity import certify_cell_geometry_validity
from phydrax.discretization._cell_mesh import CellBlock, CellMesh, PolyhedralBlock
from phydrax.geometry import _domain_certification as certification
from phydrax.geometry._domain_coverage import PiecewiseLinearDomain
from phydrax.geometry._domain_numerics import (
    cell_coordinate_expressions,
    polyhedral_cell_measures,
    polyhedral_vertex_coordinates,
)
from phydrax.geometry._domain_planar_boundary import (
    affine_cell_corners,
    oriented_target_facets,
)
from phydrax.geometry._domain_preparation import (
    DomainCoveragePreparation,
    prepare_domain_coverage,
)
from phydrax.geometry._embedding_state import MeshCertificateFinding
from phydrax.geometry._global_embedding import certify_global_embedding


def _split_square() -> tuple[
    PiecewiseLinearDomain, CellMesh, CellGeometrySpec, np.ndarray
]:
    vertices = np.asarray(
        ((0, 0), (0.5, 0), (1, 0), (1, 1), (0.5, 1), (0, 1)), dtype=np.float64
    )
    domain = PiecewiseLinearDomain(
        vertices,
        np.asarray(((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (1, 4))),
        np.asarray(((0, -1), (1, -1), (1, -1), (1, -1), (0, -1), (0, -1), (0, 1))),
        ("left", "right"),
        source_id="lazy-coverage-parity",
    )
    mesh = CellMesh(
        vertices,
        (
            CellBlock(
                "left",
                "triangle",
                np.asarray(((0, 1, 4), (0, 4, 5))),
                global_ids=(101, 202),
            ),
            CellBlock(
                "right",
                "triangle",
                np.asarray(((1, 2, 3), (1, 3, 4))),
                global_ids=(303, 404),
            ),
        ),
    )
    return domain, mesh, CellGeometrySpec.affine(mesh), np.asarray((0, 0, 1, 1))


def _eager(
    preparation: DomainCoveragePreparation, mesh: CellMesh, geometry: CellGeometrySpec
) -> DomainCoveragePreparation:
    return replace(
        preparation, coordinate_maps=cell_coordinate_expressions(mesh, geometry)
    )


def _finding_fields(
    findings: tuple[MeshCertificateFinding, ...] | list[MeshCertificateFinding],
) -> list[tuple[object, ...]]:
    return [(f.check, f.status, f.entity_ids, f.detail) for f in findings]


def test_multiblock_supplied_subset_matches_eager_and_preserves_order() -> None:
    domain, mesh, geometry, regions = _split_square()
    lazy = prepare_domain_coverage(mesh, geometry, regions, 2)
    eager = _eager(prepare_domain_coverage(mesh, geometry, regions, 2), mesh, geometry)
    supplied = {1: Fraction(1, 4), 2: Fraction(1, 4)}
    lazy_findings: list[MeshCertificateFinding] = []
    eager_findings: list[MeshCertificateFinding] = []
    lazy_measures = certification._target_region_measures(
        mesh, geometry, domain, lazy, lazy_findings, cell_measures=supplied
    )
    eager_measures = certification._target_region_measures(
        mesh, geometry, domain, eager, eager_findings, cell_measures=supplied
    )
    assert lazy_measures == eager_measures == (Fraction(1, 2), Fraction(1, 2))
    assert _finding_fields(lazy_findings) == _finding_fields(eager_findings) == []

    # Negative supplied values on opposite sides of the block boundary must
    # generate findings in cell order, while unsupplied maps still contribute.
    supplied = {1: Fraction(-1, 4), 2: Fraction(-1, 8)}
    lazy_findings = []
    eager_findings = []
    lazy_measures = certification._target_region_measures(
        mesh, geometry, domain, lazy, lazy_findings, cell_measures=supplied
    )
    eager_measures = certification._target_region_measures(
        mesh, geometry, domain, eager, eager_findings, cell_measures=supplied
    )
    assert lazy_measures == eager_measures == (Fraction(0), Fraction(1, 8))
    assert _finding_fields(lazy_findings) == _finding_fields(eager_findings)
    assert [(f.check, f.entity_ids) for f in lazy_findings] == [
        ("target_cell_orientation", (202,)),
        ("target_cell_orientation", (303,)),
    ]


def test_full_coverage_detects_misassigned_cell_against_exact_areas() -> None:
    domain, mesh, geometry, regions = _split_square()
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    embedding = certify_global_embedding(mesh, geometry, validity)
    assert embedding.status == "certified"
    certificate = certification.certify_domain_coverage(
        mesh, geometry, domain, regions, embedding=embedding
    )
    # Each right triangle has base 1/2 and height 1.
    requested = (Fraction(1, 2), Fraction(1, 2))
    assert certificate.status == "certified"
    assert certificate.requested_region_measures == requested
    assert certificate.achieved_region_measures == requested
    assert certificate.findings == ()

    faulty_regions = np.asarray((0, 1, 1, 1))
    faulty = certification.certify_domain_coverage(
        mesh, geometry, domain, faulty_regions, embedding=embedding
    )
    assert faulty.status == "violated"
    assert faulty.requested_region_measures == requested
    assert faulty.achieved_region_measures == (Fraction(1, 4), Fraction(3, 4))
    assert {
        f.entity_ids for f in faulty.findings if f.check == "region_measure"
    } == {(0,), (1,)}


def test_polyhedral_measure_reuses_fixed_cell_maps_with_exact_parity() -> None:
    vertices = np.asarray(
        (
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (2, 0, 0),
            (3, 0, 0),
            (2, 1, 0),
            (2, 0, 1),
        ),
        dtype=np.float64,
    )
    faces = ((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3))
    cells = [
        [np.asarray(tuple(v + offset for v in face)) for face in faces]
        for offset in (0, 4)
    ]
    carrier = CellMesh.from_polyhedra(
        vertices, cells, cell_global_ids=np.asarray((101, 202), dtype=np.int64)
    )
    connectivity = carrier.connectivity
    assert isinstance(connectivity, PolyhedralConnectivity)
    poly_vertices = np.asarray(carrier.blocks[0].vertices)[1:2]
    mesh = CellMesh(
        vertices,
        (
            CellBlock(
                "tetra", "tetrahedron", np.asarray(((0, 1, 2, 3),)), global_ids=(101,)
            ),
            PolyhedralBlock(
                "polyhedron", poly_vertices, global_ids=np.asarray((202,), dtype=np.int64)
            ),
        ),
        vertex_global_ids=carrier.vertex_global_ids,
        polyhedral_connectivity=connectivity,
    )
    geometry = CellGeometrySpec.affine(mesh)
    domain = PiecewiseLinearDomain(
        vertices[:4],
        np.asarray(faces),
        np.asarray(((0, -1),) * 4),
        ("solid",),
        source_id="polyhedral-map-parity",
    )
    lazy = prepare_domain_coverage(mesh, geometry, np.asarray((0, 0)), 1)
    eager = _eager(
        prepare_domain_coverage(mesh, geometry, np.asarray((0, 0)), 1), mesh, geometry
    )
    assert polyhedral_vertex_coordinates(
        mesh, geometry, coordinate_maps=lazy.coordinate_maps
    ) == polyhedral_vertex_coordinates(
        mesh, geometry, coordinate_maps=eager.coordinate_maps
    )
    assert (
        polyhedral_cell_measures(mesh, geometry, coordinate_maps=lazy.coordinate_maps)
        == polyhedral_cell_measures(mesh, geometry, coordinate_maps=eager.coordinate_maps)
        == (None, Fraction(1, 6))
    )
    lazy_findings: list[MeshCertificateFinding] = []
    eager_findings: list[MeshCertificateFinding] = []
    assert (
        certification._target_region_measures(
            mesh, geometry, domain, lazy, lazy_findings, cell_measures={0: Fraction(1, 6)}
        )
        == certification._target_region_measures(
            mesh,
            geometry,
            domain,
            eager,
            eager_findings,
            cell_measures={0: Fraction(1, 6)},
        )
        == (Fraction(1, 3),)
    )
    assert _finding_fields(lazy_findings) == _finding_fields(eager_findings) == []


@pytest.mark.parametrize("degree", (1, 2, 3))
@pytest.mark.parametrize("curvature", (0.0, 0.125))
def test_nodal_boundary_and_carrier_parity_with_eager_maps(
    degree: int, curvature: float
) -> None:
    domain, mesh, _, regions = _split_square()
    element = lagrange_element("triangle", degree)
    nodes = np.asarray(element.reference_nodes, dtype=np.float64)
    vertices = np.asarray(mesh.coordinates, dtype=np.float64)
    coordinates: list[np.ndarray] = []
    routes: dict[str, np.ndarray] = {}
    cursor = 0
    for block in mesh.blocks:
        block_routes = []
        for cell in np.asarray(block.vertices):
            points = vertices[cell]
            values = (
                points[0]
                + nodes[:, :1] * (points[1] - points[0])
                + nodes[:, 1:] * (points[2] - points[0])
            )
            values[:, 1] += curvature * nodes[:, 0] * nodes[:, 1]
            coordinates.extend(values)
            block_routes.append(np.arange(cursor, cursor + len(nodes)))
            cursor += len(nodes)
        routes[block.name] = np.asarray(block_routes, dtype=np.int64)
    geometry = CellGeometrySpec(
        {block.name: element for block in mesh.blocks}, routes, np.asarray(coordinates)
    )
    lazy = prepare_domain_coverage(mesh, geometry, regions, 2)
    eager = _eager(prepare_domain_coverage(mesh, geometry, regions, 2), mesh, geometry)
    actual = oriented_target_facets(
        mesh,
        regions,
        lazy.coordinate_maps,
        geometry=geometry,
        source_domain=domain,
        exact_face_images=True,
    )
    expected = oriented_target_facets(
        mesh,
        regions,
        eager.coordinate_maps,
        geometry=geometry,
        source_domain=domain,
        exact_face_images=True,
    )
    assert actual == expected
    try:
        eager_corners = affine_cell_corners(
            mesh, eager.coordinate_maps, geometry=geometry
        )
    except ValueError:
        with pytest.raises(ValueError):
            affine_cell_corners(mesh, lazy.coordinate_maps, geometry=geometry)
    else:
        assert (
            affine_cell_corners(mesh, lazy.coordinate_maps, geometry=geometry)
            == eager_corners
        )
