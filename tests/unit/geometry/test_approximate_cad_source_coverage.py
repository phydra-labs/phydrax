from collections.abc import Sequence
from fractions import Fraction

import numpy as np

from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.discretization._cell_geometry_validity import (
    cell_geometry_id,
    certify_cell_geometry_validity,
)
from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.geometry._approximate_source_coverage import (
    _area2,
    _cad_chart_partition,
    _surface_point,
    approximate_source_coverage,
    cad_material_sides,
)
from phydrax.geometry._domain_boundary import MeshingDomainBoundarySource
from phydrax.geometry._global_embedding import certify_global_embedding
from phydrax.geometry._meshing_domain import MeshingDomain, PatchCurveUse
from phydrax.geometry._source_boundary_records import (
    rational_chart_tokens,
    SourceBoundaryChartWitness,
)
from phydrax.geometry._source_fidelity import (
    _surface_target_chart_rows,
    _volume_boundary_facet_lookup,
    certify_source_fidelity,
)
from phydrax.geometry.brep import brep_box
from phydrax.units import MILLIMETER


_Point2 = tuple[Fraction, Fraction]
_Point3 = tuple[Fraction, Fraction, Fraction]
_Triangle2 = tuple[_Point2, _Point2, _Point2]
_ReferenceTriangle3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]
_CadFaceRow = tuple[int, _Triangle2, _ReferenceTriangle3, int, int]
_PhysicalTriangle3 = tuple[_Point3, _Point3, _Point3]


def _cross(first: _Point3, second: _Point3, /) -> _Point3:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _dot(first: _Point3, second: _Point3, /) -> Fraction:
    return sum(
        (left * right for left, right in zip(first, second, strict=True)), Fraction(0)
    )


def _cad_tetra_star_inputs() -> tuple[
    MeshingDomain,
    CellMesh,
    CellGeometrySpec,
    MeshingDomainBoundarySource,
    np.ndarray,
    tuple[tuple[tuple[int, int], ...], ...],
    list[_CadFaceRow],
]:
    domain = MeshingDomain.from_brep(
        brep_box(
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 0.001),
            coordinate_contract=SpatialCoordinateContract(MILLIMETER),
        )
    )
    source = MeshingDomainBoundarySource(domain, tuple(range(len(domain.patches))))
    triangle_rows: tuple[tuple[int, _Triangle2, _PhysicalTriangle3], ...] = tuple(
        (
            patch,
            triangle,
            (
                _surface_point(domain.patches[patch].surface, triangle[0]),
                _surface_point(domain.patches[patch].surface, triangle[1]),
                _surface_point(domain.patches[patch].surface, triangle[2]),
            ),
        )
        for patch, triangles in source.boundary_parameter_triangles()
        for triangle in triangles
    )
    physical_vertices = tuple(
        sorted({point for _, _, points in triangle_rows for point in points})
    )
    vertex_row = {point: row for row, point in enumerate(physical_vertices)}
    center: _Point3 = (
        sum((point[0] for point in physical_vertices), Fraction(0))
        / len(physical_vertices),
        sum((point[1] for point in physical_vertices), Fraction(0))
        / len(physical_vertices),
        sum((point[2] for point in physical_vertices), Fraction(0))
        / len(physical_vertices),
    )
    center_row = len(physical_vertices)
    connectivity: list[tuple[int, int, int, int]] = []
    for _, _, points in triangle_rows:
        rows = [vertex_row[point] for point in points]
        edge_first: _Point3 = (
            points[1][0] - points[0][0],
            points[1][1] - points[0][1],
            points[1][2] - points[0][2],
        )
        edge_second: _Point3 = (
            points[2][0] - points[0][0],
            points[2][1] - points[0][1],
            points[2][2] - points[0][2],
        )
        inward: _Point3 = (
            center[0] - points[0][0],
            center[1] - points[0][1],
            center[2] - points[0][2],
        )
        if _dot(_cross(edge_first, edge_second), inward) < 0:
            rows[1], rows[2] = rows[2], rows[1]
        connectivity.append((rows[0], rows[1], rows[2], center_row))
    coordinates = np.asarray(
        [[float(value) for value in point] for point in (*physical_vertices, center)],
        dtype=np.float64,
    )
    cell_ids = tuple(811 + row for row in range(len(connectivity)))
    mesh = CellMesh(
        coordinates,
        (
            CellBlock(
                "tetrahedra",
                "tetrahedron",
                np.asarray(connectivity, dtype=np.int32),
                global_ids=cell_ids,
            ),
        ),
        vertex_global_ids=np.arange(len(physical_vertices) + 1, dtype=np.int64) + 101,
    )
    geometry = CellGeometrySpec.affine(mesh)
    regions = np.zeros((len(connectivity),), dtype=np.int64)
    _, signs_by_patch = cad_material_sides(domain)
    reference = reference_cell_topology("tetrahedron")
    face_rows: list[_CadFaceRow] = []
    for row, (patch, triangle, points) in enumerate(triangle_rows):
        cell_id = cell_ids[row]
        cell_vertices = tuple(int(value) for value in connectivity[row])
        triangle_vertex_rows = {vertex_row[point] for point in points}
        local_faces = tuple(
            local_face
            for local_face, face in enumerate(reference.entities[2])
            if {cell_vertices[int(slot)] for slot in face} == triangle_vertex_rows
        )
        assert len(local_faces) == 1
        local_face = local_faces[0]
        first_reference = reference.vertices[cell_vertices.index(vertex_row[points[0]])]
        second_reference = reference.vertices[cell_vertices.index(vertex_row[points[1]])]
        third_reference = reference.vertices[cell_vertices.index(vertex_row[points[2]])]
        target_reference: _ReferenceTriangle3 = (
            (
                float(first_reference[0]),
                float(first_reference[1]),
                float(first_reference[2]),
            ),
            (
                float(second_reference[0]),
                float(second_reference[1]),
                float(second_reference[2]),
            ),
            (
                float(third_reference[0]),
                float(third_reference[1]),
                float(third_reference[2]),
            ),
        )
        face_rows.append((patch, triangle, target_reference, cell_id, local_face))
    assert len(face_rows) == 12
    return domain, mesh, geometry, source, regions, signs_by_patch, face_rows


def _chart_witness(
    domain: MeshingDomain,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: MeshingDomainBoundarySource,
    face_rows: Sequence[_CadFaceRow],
    patches: Sequence[int],
    /,
) -> SourceBoundaryChartWitness:
    target_faces = _volume_boundary_facet_lookup(mesh)
    entity_ids = np.asarray(
        [target_faces[(row[3], row[4])][0] for row in face_rows], dtype=np.int64
    )

    def create(orientations: np.ndarray, /) -> SourceBoundaryChartWitness:
        return SourceBoundaryChartWitness(
            mesh_id=mesh.mesh_id,
            geometry_id=cell_geometry_id(geometry),
            topology_id=mesh.topology_id,
            source_id=source.source_id,
            source_revision=source.source_revision,
            source_query_id=source.source_scope_id,
            chart_dimension=2,
            reference_dimension=3,
            cell_global_ids=np.asarray([row[3] for row in face_rows], dtype=np.int64),
            entity_global_ids=entity_ids,
            local_entity_indices=np.asarray(
                [row[4] for row in face_rows], dtype=np.int32
            ),
            target_reference_vertices=np.asarray(
                [row[2] for row in face_rows], dtype=np.float64
            ),
            target_reference_rational_tokens=rational_chart_tokens(
                tuple(
                    tuple(Fraction.from_float(float(value)) for value in point)
                    for point in row[2]
                )
                for row in face_rows
            ),
            source_chart_ids=tuple(domain.entity_id(2, patch) for patch in patches),
            source_entity_ids=tuple(domain.entity_id(2, patch) for patch in patches),
            occurrence_paths=tuple(
                domain.source_occurrences[2][patch] for patch in patches
            ),
            source_parameters=np.asarray([row[1] for row in face_rows], dtype=np.float64),
            source_parameter_rational_tokens=rational_chart_tokens(
                tuple(row[1] for row in face_rows)
            ),
            orientations=np.asarray(orientations, dtype=np.int8),
        )

    provisional = create(np.ones((len(face_rows),), dtype=np.int8))
    _, _, target_orientation = _surface_target_chart_rows(
        mesh, geometry, source, provisional
    )
    source_orientation = np.asarray(
        [1 if _area2(row[1]) > 0 else -1 for row in face_rows], dtype=np.int8
    )
    return create(source_orientation * np.asarray(target_orientation, dtype=np.int8))


def test_cad_chart_degree_requires_authored_seam_correspondence() -> None:
    (
        domain,
        mesh,
        geometry,
        source,
        regions,
        signs_by_patch,
        face_rows,
    ) = _cad_tetra_star_inputs()
    patch_rows = [row[0] for row in face_rows]
    valid = _chart_witness(domain, mesh, geometry, source, face_rows, patch_rows)
    source_count, degree_ids = _cad_chart_partition(domain, source, valid, mesh, regions)
    assert source_count == len(face_rows)
    assert len(degree_ids) == len(domain.regions)

    fidelity = certify_source_fidelity(
        mesh, geometry, source, tolerance=0.002, witness=valid
    )
    assert fidelity.status == "certified", fidelity.findings
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    embedding = certify_global_embedding(mesh, geometry, validity)
    assert embedding.status == "certified", embedding.findings

    chart_rows = dict(source.boundary_parameter_triangles())
    chart_vertices = {
        patch: frozenset(point for triangle in triangles for point in triangle)
        for patch, triangles in chart_rows.items()
    }
    chart_areas = {
        patch: sum((abs(_area2(triangle)) for triangle in triangles), Fraction(0))
        for patch, triangles in chart_rows.items()
    }
    physical_chart_points = {
        patch: {
            point: _surface_point(domain.patches[patch].surface, point)
            for point in chart_vertices[patch]
        }
        for patch in range(len(domain.patches))
    }
    curve_rows = {
        patch: {
            use.curve
            for loop in domain.patches[patch].loops
            for use in loop
            if isinstance(use, PatchCurveUse)
        }
        for patch in range(len(domain.patches))
    }
    nonadjacent_equal_charts = tuple(
        (first, second)
        for first in range(len(domain.patches))
        for second in range(first + 1, len(domain.patches))
        if chart_vertices[first] == chart_vertices[second]
        and chart_areas[first] == chart_areas[second]
        and dict(signs_by_patch[first])[0] == dict(signs_by_patch[second])[0]
        and not curve_rows[first].intersection(curve_rows[second])
        and all(
            sum(
                (
                    (first_value - second_value) ** 2
                    for first_value, second_value in zip(
                        physical_chart_points[first][point],
                        physical_chart_points[second][point],
                        strict=True,
                    )
                ),
                Fraction(0),
            )
            <= Fraction(1, 250_000)
            for point in chart_vertices[first]
        )
    )
    assert nonadjacent_equal_charts
    first_patch, second_patch = nonadjacent_equal_charts[0]
    relabeled = [
        second_patch
        if patch == first_patch
        else first_patch
        if patch == second_patch
        else patch
        for patch in patch_rows
    ]
    counterfeit = _chart_witness(domain, mesh, geometry, source, face_rows, relabeled)
    counterfeit_fidelity = certify_source_fidelity(
        mesh, geometry, source, tolerance=0.002, witness=counterfeit
    )
    assert counterfeit_fidelity.status == "certified", counterfeit_fidelity.findings
    attempt = approximate_source_coverage(
        mesh,
        geometry,
        domain,
        regions,
        source=source,
        fidelity=counterfeit_fidelity,
        embedding=embedding,
    )
    assert attempt.bounded is None
    assert attempt.failures and attempt.failures[0].status == "violated"
