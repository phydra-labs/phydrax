from fractions import Fraction

import numpy as np
import pytest

from phydrax.discretization import (
    CellGeometrySpec,
    CellMesh,
    ExactPowerCellGeometrySource,
)
from phydrax.discretization._cell_complex import PolyhedralConnectivity
from phydrax.discretization._cell_geometry_transfer import _certified_cell_measures
from phydrax.geometry._mesh_certificates import (
    MeshCertificateLimits,
    PiecewiseLinearDomain,
)
from phydrax.meshing._contracts import MeshingFailure, MeshingLimits
from phydrax.meshing._hex_cut_preparation import (
    certify_cut_corner_coverage,
    CutCornerLinkError,
    prepare_cut_corner_hexes,
    realize_cut_corner_hexes,
)
from phydrax.meshing._volume_generation import declared_plc_domain, PiecewiseLinearComplex


def _pentagonal_prisms(
    layers: int = 1,
) -> tuple[CellMesh, CellGeometrySpec, PiecewiseLinearDomain]:
    # Independently authored non-orthogonal pentagon, not a fitted source chart.
    polygon = ((0.0, 0.0), (2.0, 0.0), (3.0, 1.0), (1.0, 3.0), (0.0, 2.0))
    points = np.asarray([(x, y, float(z)) for z in range(layers + 1) for x, y in polygon])
    cells, tets = [], []
    for layer in range(layers):
        lower, upper = 5 * layer, 5 * (layer + 1)
        cells.append(
            [
                tuple(lower + row for row in reversed(range(5))),
                tuple(upper + row for row in range(5)),
                *[
                    (
                        lower + row,
                        lower + (row + 1) % 5,
                        upper + (row + 1) % 5,
                        upper + row,
                    )
                    for row in range(5)
                ],
            ]
        )
        for row in range(1, 4):
            a, b, c = lower, lower + row, lower + row + 1
            A, B, C = upper, upper + row, upper + row + 1
            tets.extend(((a, b, c, C), (a, C, b, B), (a, B, C, A)))
    mesh = CellMesh.from_polyhedra(points, cells)
    # One original site has the complete independently authored prism carrier.
    # Vertex witnesses name its original carrier points, never fabricated planes.
    source = ExactPowerCellGeometrySource(
        np.asarray([[1.0, 1.0, 0.5]]),
        np.zeros(1),
        points,
        np.asarray(tets),
        np.arange(points.shape[0] + 1),
        np.zeros(points.shape[0], dtype=np.int64),
        np.column_stack((np.arange(points.shape[0]), np.full((points.shape[0], 3), -1))),
    )
    original_faces = {}
    for region, loops in enumerate(cells):
        for loop in loops:
            original_faces.setdefault(tuple(sorted(loop)), []).append(
                (tuple(loop), region)
            )
    polygons, regions = [], []
    for entries in original_faces.values():
        loop, negative = entries[0]
        positive = entries[1][1] if len(entries) == 2 else -1
        polygons.append(loop)
        regions.append((positive, negative))
    plc = PiecewiseLinearComplex(
        points,
        polygons,
        np.arange(len(polygons)),
        np.asarray(regions),
        tuple(f"original-material-{row}" for row in range(layers)),
    )
    domain = declared_plc_domain(plc, "independent-pentagonal-original")
    return mesh, CellGeometrySpec.power(mesh, source), domain


@pytest.mark.parametrize("layers", (1, 2))
def test_nonpolycube_corner_maps_keep_source_and_exact_volume(layers: int) -> None:
    mesh, geometry, original_domain = _pentagonal_prisms(layers)
    prepared = prepare_cut_corner_hexes(mesh, geometry, MeshingLimits())
    assert prepared.hexes.shape == (10 * layers, 8)
    assert (
        prepared.exact_vertices[: mesh.coordinates.shape[0]]
        == geometry.source_coordinates()
    )
    assert any(
        value.denominator == 5 for point in prepared.exact_vertices for value in point
    )
    construction = realize_cut_corner_hexes(
        prepared, MeshingLimits(), certificate_limits=MeshCertificateLimits()
    )
    coverage, regions = certify_cut_corner_coverage(
        construction,
        original_domain,
        np.arange(layers),
        certificate_limits=MeshCertificateLimits(),
    )
    assert coverage.status == "certified"
    assert np.array_equal(regions, prepared.parent_cells)
    assert construction.validity.all_certified
    assert construction.embedding.status == "certified"
    assert all(block.cell_kind == "hexahedron" for block in construction.mesh.blocks)
    assert construction.geometry.source_coordinates() == prepared.exact_vertices
    volumes, errors, exact = _certified_cell_measures(
        construction.mesh, construction.geometry
    )
    assert exact
    # Independent shoelace sum 2+8+2 gives area 6, through unit layers.
    expected = Fraction(6 * layers)
    measured = sum((Fraction(float(value)) for value in volumes), Fraction(0))
    bound = sum((Fraction(float(value)) for value in errors), Fraction(0))
    assert abs(measured - expected) <= bound
    if layers == 2:
        # The actual original source-cell adjacency is one shared pentagon;
        # its five corner quads use identical source edge/face identities.
        connectivity = mesh.connectivity
        if not isinstance(connectivity, PolyhedralConnectivity):
            raise TypeError("Cut fixture requires polyhedral connectivity.")
        incidence = np.bincount(
            np.asarray(connectivity.cell_face_values), minlength=connectivity.face_count
        )
        shared = int(np.flatnonzero(incidence == 2)[0])
        face_node = connectivity.vertex_count + connectivity.edge_count + shared
        assert np.count_nonzero(prepared.hexes == face_node) == 10
        assert set(prepared.parent_cells.tolist()) == {0, 1}


def test_nonsimple_link_reports_original_scientific_incidence() -> None:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 1.0],
        ]
    )
    cells = [
        [
            np.asarray(face, dtype=np.int64)
            for face in ((0, 3, 2, 1), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4))
        ]
    ]
    mesh = CellMesh.from_polyhedra(
        points,
        cells,
        vertex_global_ids=np.asarray([11, 17, 23, 29, 31]),
    )
    with pytest.raises(CutCornerLinkError) as error:
        prepare_cut_corner_hexes(mesh, CellGeometrySpec.affine(mesh), MeshingLimits())
    witness = error.value.witness
    assert witness.vertex_id == 31
    assert len(witness.edge_ids) == len(witness.face_ids) == 4
    assert len(set(witness.edge_ids)) == len(set(witness.face_ids)) == 4


def test_immutable_original_face_is_not_retagged_as_hex_boundary() -> None:
    mesh, geometry, _ = _pentagonal_prisms()
    face = int(np.asarray(mesh.entity_set(2).entity_ids)[0])
    with pytest.raises(MeshingFailure, match="(?i)immutable source faces"):
        prepare_cut_corner_hexes(
            mesh, geometry, MeshingLimits(), immutable_face_ids=(face,)
        )


def test_cut_capacity_refuses_before_corner_construction() -> None:
    mesh, geometry, _ = _pentagonal_prisms()
    with pytest.raises(MeshingFailure, match="cells"):
        prepare_cut_corner_hexes(mesh, geometry, MeshingLimits(maximum_cells=9))
