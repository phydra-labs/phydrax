from __future__ import annotations

import numpy as np

from phydrax.discretization._cell_complex import PolygonalConnectivity
from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.discretization._periodic_cell import PeriodicCell
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
