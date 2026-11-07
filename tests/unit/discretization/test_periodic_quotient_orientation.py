from __future__ import annotations

from copy import copy
from fractions import Fraction

import numpy as np
import pytest

from phydrax.discretization._cell_complex import PolygonalConnectivity
from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.discretization._exact_plc_geometry import ExactPlcCellGeometrySource
from phydrax.discretization._periodic_cell import PeriodicCell
from phydrax.discretization._periodic_exact import PeriodicIsometryGroup
from phydrax.discretization._periodic_topology import PeriodicMeshTopology


def test_reversed_periodic_edge_copies_keep_quotient_chain_closed() -> None:
    # Two unit squares periodic in x with period two. The seam copy at x = 2 is
    # numbered top-to-bottom, so its lifted edge runs opposite to the x = 0 copy.
    coordinates = np.asarray(
        ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0), (2.0, 1.0), (2.0, 0.0)),
        dtype=np.float64,
    )
    triangles = np.asarray(((0, 2, 3), (0, 3, 1), (2, 5, 4), (2, 4, 3)), dtype=np.int32)
    lifted = CellMesh(
        coordinates,
        (CellBlock("strip", "triangle", triangles),),
        vertex_global_ids=np.arange(6, dtype=np.int64),
    )
    topology = PeriodicMeshTopology(
        lifted,
        PeriodicCell(np.diag((2.0, 1.0)), periodic_axes=(True, False)),
        np.asarray((0, 1, 2, 3, 1, 0), dtype=np.int64),
        np.asarray(((0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0)), dtype=np.int64),
    )

    # The constructor validates the quotient chain; check it independently.
    incidences = topology.quotient.incidences
    for lower, upper in zip(incidences[:-1], incidences[1:], strict=True):
        product = (lower.scipy_boundary() @ upper.scipy_boundary()).toarray()
        np.testing.assert_array_equal(product, np.zeros_like(product))

    connectivity = lifted.connectivity
    assert isinstance(connectivity, PolygonalConnectivity)
    edges = np.asarray(connectivity.edges, dtype=np.int64)
    edge_rows = {(row[0], row[1]): index for index, row in enumerate(edges.tolist())}
    seam = edge_rows[(0, 1)]
    image = edge_rows[(4, 5)]
    begin = int(np.asarray(topology.lifted_offsets)[1])
    orbits = np.asarray(topology.orbit_indices, dtype=np.int64)[begin:]
    orientations = np.asarray(topology.orbit_orientations, dtype=np.int64)[begin:]
    assert orbits[seam] == orbits[image]
    # (0 -> 1) runs up the seam; (4 -> 5) is its translate traversed downward.
    assert orientations[seam] * orientations[image] == -1


@pytest.mark.parametrize("declared_group", (False, True))
def test_embedded_periodic_strip_preserves_oriented_quotient_and_restore(
    declared_group: bool,
) -> None:
    coordinates = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 0.0, 0.5),
            (1.0, 1.0, 0.5),
            (2.0, 1.0, 0.0),
            (2.0, 0.0, 0.0),
        ),
    )
    triangles = np.asarray(((0, 2, 3), (0, 3, 1), (2, 5, 4), (2, 4, 3)))
    lifted = CellMesh(coordinates, (CellBlock("strip", "triangle", triangles),))
    translation = np.eye(4)
    translation[0, 3] = 2.0
    identification = (
        PeriodicIsometryGroup(translation[None])
        if declared_group
        else PeriodicCell(np.asarray(((2.0, 0.0, 0.0),)))
    )
    roots = np.asarray((0, 1, 2, 3, 1, 0))
    shifts = np.asarray(((0,), (0,), (0,), (0,), (1,), (1,)))
    topology = PeriodicMeshTopology(lifted, identification, roots, shifts)
    exact = ExactPlcCellGeometrySource(
        "embedded-strip",
        "revision-1",
        lifted.vertex_global_ids,
        tuple(tuple(Fraction(float(value)) for value in row) for row in coordinates),
        vertex_source_entity_ids=(None,) * len(coordinates),
        vertex_occurrence_paths=((),) * len(coordinates),
    )
    witnessed = PeriodicMeshTopology(
        lifted, identification, roots, shifts, exact_coordinate_source=exact
    )
    assert witnessed.coordinate_source_map_id == exact.source_map_id
    forged_exact = list(exact.exact_coordinates)
    forged_exact[4] = (
        forged_exact[4][0] + Fraction(1, 2**100),
        *forged_exact[4][1:],
    )
    forged_source = ExactPlcCellGeometrySource(
        "embedded-strip",
        "revision-1",
        lifted.vertex_global_ids,
        tuple(forged_exact),
        vertex_source_entity_ids=(None,) * len(coordinates),
        vertex_occurrence_paths=((),) * len(coordinates),
    )
    np.testing.assert_array_equal(forged_source.rounded_vertices, coordinates)
    with pytest.raises(ValueError, match="exact|Exact"):
        PeriodicMeshTopology(
            lifted, identification, roots, shifts, exact_coordinate_source=forged_source
        )
    assert topology.topological_dimension == 2
    assert topology.cell.ambient_dimension == 3
    boundaries = topology.quotient.incidences
    product = boundaries[0].scipy_boundary() @ boundaries[1].scipy_boundary()
    np.testing.assert_array_equal(product.toarray(), np.zeros(product.shape))
    connectivity = lifted.connectivity
    assert isinstance(connectivity, PolygonalConnectivity)
    edges = np.asarray(connectivity.edges)
    rows = {tuple(edge): row for row, edge in enumerate(edges.tolist())}
    offset = int(topology.lifted_offsets[1])
    seam, image = rows[(0, 1)], rows[(4, 5)]
    assert topology.orbit_indices[offset + seam] == topology.orbit_indices[offset + image]
    assert (
        topology.orbit_orientations[offset + seam]
        * topology.orbit_orientations[offset + image]
        == -1
    )
    restored = PeriodicMeshTopology.from_stored_fields(topology, lifted)
    assert restored.periodic_topology_id == topology.periodic_topology_id
    bad = copy(topology)
    object.__setattr__(bad, "orbit_orientations", -topology.orbit_orientations)
    with pytest.raises(ValueError):
        PeriodicMeshTopology.from_stored_fields(bad, lifted)
    object.__setattr__(bad, "periodic_topology_id", bad._content_id())
    with pytest.raises(ValueError, match="Reconstructed periodic quotient"):
        PeriodicMeshTopology.from_stored_fields(bad, lifted)
    wrong = shifts.copy()
    wrong[4, 0] = 0
    with pytest.raises(ValueError):
        PeriodicMeshTopology(lifted, identification, roots, wrong)
    with pytest.raises(ValueError, match="ambient"):
        PeriodicMeshTopology(
            lifted, PeriodicCell(np.asarray(((2.0, 0.0),))), roots, shifts
        )
