"""Real serial/MPI metric adaptation through the persistent Omega_h worker.

Build native/providers/omega_h, then:

PHYDRAX_OMEGA_H_WORKER=/path/to/phydrax-omega-h-worker \
    python examples/meshing_omega_h.py --ranks 1 --dimension 2
PHYDRAX_OMEGA_H_WORKER=/path/to/phydrax-omega-h-worker \
    python examples/meshing_omega_h.py --ranks 2 --dimension 3

MPI launcher flags are deployment configuration supplied through --launcher
(shell-like splitting, never shell execution). On affected Apple OpenMPI/hwloc
installations only, HWLOC_SYNTHETIC='pack:1 core:10 pu:1' and --launcher
'mpiexec --bind-to none --map-by slot' avoid PRRTE's startup topology crash;
processes still run real MPI.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
from itertools import permutations, product
from typing import Any

import numpy as np

from phydrax import SpatialCoordinateContract
from phydrax.discretization import CellMesh
from phydrax.meshing import (
    certify_cell_mesh,
    MeshingEntityKind,
    MeshingScope,
    MeshMetricField,
    MeshPatch,
    MeshZone,
    MeshZoneRole,
    OmegaHField,
    OmegaHFieldTransfer,
    OmegaHProvider,
)


def simplex_grid(dim: int, n: int) -> CellMesh:
    indices = tuple(product(range(n + 1), repeat=dim))
    lookup = {point: index for index, point in enumerate(indices)}
    points = np.asarray(indices, dtype=np.float64) / n
    cells = []
    for origin in product(range(n), repeat=dim):
        for axes in permutations(range(dim)):
            point = np.asarray(origin)
            row = [lookup[tuple(point)]]
            for axis in axes:
                point = point.copy()
                point[axis] += 1
                row.append(lookup[tuple(point)])
            corners = points[row]
            if np.linalg.det(corners[1:] - corners[0]) < 0:
                row[1], row[2] = row[2], row[1]
            cells.append(row)
    constructor = CellMesh.from_triangles if dim == 2 else CellMesh.from_tetrahedra
    return constructor(
        points,
        np.asarray(cells),
        vertex_global_ids=np.arange(len(points), dtype=np.int64) * 3 + 17,
        cell_global_ids=np.arange(len(cells), dtype=np.int64) * 5 + 31,
    )


def scope(mesh: CellMesh, dimension: int, ids: Any) -> MeshingScope:
    return MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        MeshingEntityKind.MESH,
        dimension,
        mesh.entity_set(dimension).entity_set_id,
        np.asarray(ids, dtype=np.int64),
    )


def cell_table(mesh: CellMesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(global IDs, measures, centroids) of every simplex."""
    dim = mesh.topological_dimension
    corners = np.asarray(mesh.coordinates)[
        np.concatenate([np.asarray(block.vertices) for block in mesh.blocks])
    ]
    ids = np.concatenate([np.asarray(block.global_ids) for block in mesh.blocks])
    measures = np.linalg.det(corners[:, 1:] - corners[:, :1]) / math.factorial(dim)
    return ids, measures, corners.mean(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--dimension", type=int, choices=(2, 3), default=2)
    parser.add_argument("--launcher", default="mpiexec")
    arguments = parser.parse_args()
    dim = arguments.dimension
    n = 8 if dim == 2 else 4
    mesh = simplex_grid(dim, n)
    contract = SpatialCoordinateContract.si()

    # Two region zones and an inlet patch travel as Omega_h class IDs.
    cell_ids, measures, centroids = cell_table(mesh)
    left = MeshZone(
        "left", MeshZoneRole.REGION, scope(mesh, dim, cell_ids[centroids[:, 0] < 0.5])
    )
    right = MeshZone(
        "right", MeshZoneRole.REGION, scope(mesh, dim, cell_ids[centroids[:, 0] > 0.5])
    )
    # ty: ignore[unresolved-attribute]
    facets = np.asarray(mesh.connectivity.edges if dim == 2 else mesh.connectivity.faces)
    on_inlet = np.all(np.asarray(mesh.coordinates)[facets][:, :, 0] == 0.0, axis=1)
    inlet = MeshPatch(
        "inlet",
        scope(mesh, dim - 1, np.asarray(mesh.entity_set(dim - 1).entity_ids)[on_inlet]),
        adjacent_zone_ids=(left.zone_id,),
    )
    source = certify_cell_mesh(mesh, contract, zones=(left, right), patches=(inlet,))

    order = np.argsort(np.asarray(mesh.vertex_global_ids))
    vertex_ids = np.asarray(mesh.vertex_global_ids)[order]
    points = np.asarray(mesh.coordinates)[order]
    matrix = np.eye(dim) * (2 * n) ** 2
    # Nonzero cross term exercises packed symmetric metric ordering in 2D/3D.
    matrix[0, 1] = matrix[1, 0] = 0.15 * matrix[0, 0]
    metric = MeshMetricField(
        scope(mesh, 0, vertex_ids),
        np.broadcast_to(matrix, (len(points), dim, dim)),
        minimum_size=1 / (3 * n),
        maximum_size=1 / n,
        maximum_anisotropy=2,
    )
    cell_order = np.argsort(cell_ids)
    density = 1.0 + np.where(centroids[:, 0] < 0.5, 0.0, 4.0) + centroids[:, 1] ** 2
    fields = (
        OmegaHField(
            "potential",
            OmegaHFieldTransfer.LINEAR,
            scope(mesh, 0, vertex_ids),
            points @ np.arange(1.0, dim + 1.0),
        ),
        OmegaHField(
            "density",
            OmegaHFieldTransfer.CONSERVE,
            scope(mesh, dim, cell_ids[cell_order]),
            density[cell_order],
        ),
    )

    with OmegaHProvider(mpi_launcher=shlex.split(arguments.launcher)) as provider:
        result = provider.execute(
            source, metric, fields=fields, ranks=arguments.ranks, gather=True
        )

    target = result.target
    # ty: ignore[unresolved-attribute]
    ids, target_measures, target_centroids = cell_table(target.mesh)
    if not np.all(target_measures > 0):
        raise RuntimeError("Adaptation inverted a simplex")
    if not np.isclose(target_measures.sum(), 1, atol=1e-12):
        raise RuntimeError("Adaptation changed domain measure")
    if ids.size <= cell_ids.size:
        raise RuntimeError("Requested finer metric did not refine the carrier")
    # ty: ignore[unresolved-attribute]
    if not target.audit.passed or result.lineage_status != "unknown":
        raise RuntimeError("Omega_h audit or lineage evidence failed")
    # ty: ignore[unresolved-attribute]
    zones = {zone.name: zone for zone in target.zones}
    left_measure = float(
        np.sum(target_measures[np.isin(ids, np.asarray(zones["left"].scope.entity_ids))])
    )
    # ty: ignore[unresolved-attribute]
    if not np.isclose(left_measure, 0.5, atol=1e-12) or len(target.patches) != 1:
        raise RuntimeError("Region or patch classification was not preserved")
    potential, transferred = result.fields
    # ty: ignore[unresolved-attribute]
    target_points = np.asarray(target.mesh.coordinates)[
        # ty: ignore[unresolved-attribute]
        np.argsort(np.asarray(target.mesh.vertex_global_ids))
    ]
    if not np.allclose(
        potential.values, target_points @ np.arange(1.0, dim + 1.0), atol=1e-12
    ):
        raise RuntimeError("Linear transfer did not reproduce an affine field")
    mass = float(
        np.sum(
            np.asarray(transferred.values)[
                np.searchsorted(np.asarray(transferred.scope.entity_ids), ids)
            ]
            * target_measures
        )
    )
    source_mass = float(np.sum(density * measures))
    if not math.isclose(mass, source_mass, rel_tol=1e-12):
        raise RuntimeError("Conservative transfer changed the integral")
    owned = [int(np.count_nonzero(part.cell_owned)) for part in result.partitions]
    ghosts = [int(np.count_nonzero(part.cell_ghosts)) for part in result.partitions]
    if sum(owned) != ids.size or not all(value > 0 for value in owned):
        raise RuntimeError("Omega_h ownership evidence is incomplete")
    if arguments.ranks > 1 and not all(value > 0 for value in ghosts):
        raise RuntimeError("No real cross-rank ghost residence")
    # ty: ignore[unresolved-attribute]
    if not np.all(np.linalg.eigvalsh(np.asarray(result.metric.values)) > 0):
        raise RuntimeError("Omega_h returned a non-positive metric")
    print(
        json.dumps(
            {
                "provider": result.provider.version,
                "dimension": dim,
                "ranks": arguments.ranks,
                "source_cells": int(cell_ids.size),
                "target_cells": int(ids.size),
                "owned_cells": owned,
                "ghost_cells": ghosts,
                "domain_measure": float(target_measures.sum()),
                "left_measure": left_measure,
                "mass_before": source_mass,
                "mass_after": mass,
                "minimum_quality": result.evidence.minimum_quality,
                "length_range": [
                    result.evidence.minimum_length,
                    result.evidence.maximum_length,
                ],
                # ty: ignore[unresolved-attribute]
                "audit": target.audit.passed,
                "iterations": result.evidence.iterations,
                "session": result.evidence.session_id[:16],
                "lineage": result.lineage_status,
            }
        )
    )


if __name__ == "__main__":
    main()
