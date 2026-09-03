#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import numpy as np

from ._cell_complex import PolygonalConnectivity
from ._cell_mesh import CellMesh
from ._integration_domain import IntegrationDomain


def polygon_integration_domains(
    mesh: CellMesh, /
) -> tuple[IntegrationDomain, IntegrationDomain, IntegrationDomain]:
    """Build canonical cell and facet ownership domains for a polygon mesh."""
    if not isinstance(mesh, CellMesh):
        raise TypeError("mesh must be a CellMesh.")
    if not isinstance(mesh.connectivity, PolygonalConnectivity):
        raise TypeError("Polygon domains require PolygonalConnectivity.")
    connectivity = mesh.connectivity
    cell_edges = np.asarray(connectivity.cell_edges, dtype=np.int32)
    valid = np.asarray(connectivity.cell_edge_valid, dtype=bool)
    edge_count = int(connectivity.edges.shape[0])
    owner = np.full((edge_count,), -1, dtype=np.int32)
    neighbour = np.full((edge_count,), -1, dtype=np.int32)
    owner_local = np.full((edge_count,), -1, dtype=np.int32)
    neighbour_local = np.full((edge_count,), -1, dtype=np.int32)
    for cell in range(cell_edges.shape[0]):
        for local in range(cell_edges.shape[1]):
            if not valid[cell, local]:
                continue
            edge = int(cell_edges[cell, local])
            if owner[edge] < 0:
                owner[edge] = cell
                owner_local[edge] = local
            else:
                neighbour[edge] = cell
                neighbour_local[edge] = local
    exterior = np.flatnonzero(neighbour < 0).astype(np.int32)
    interior = np.flatnonzero(neighbour >= 0).astype(np.int32)
    cell_domain = IntegrationDomain(
        "cell",
        np.arange(connectivity.cell_count, dtype=np.int32),
        mesh.support.support_id,
        mesh.topology.entity_sets[2].entity_set_id,
    )
    exterior_domain = IntegrationDomain(
        "exterior_facet",
        exterior,
        mesh.support.support_id,
        mesh.topology.entity_sets[1].entity_set_id,
        owner_cells=owner[exterior],
        owner_local_entities=owner_local[exterior],
    )
    interior_domain = IntegrationDomain(
        "interior_facet",
        interior,
        mesh.support.support_id,
        mesh.topology.entity_sets[1].entity_set_id,
        owner_cells=owner[interior],
        neighbour_cells=neighbour[interior],
        owner_local_entities=owner_local[interior],
        neighbour_local_entities=neighbour_local[interior],
    )
    return cell_domain, exterior_domain, interior_domain


__all__ = ["polygon_integration_domains"]
