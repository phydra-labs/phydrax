"""Real serial/MPI metric adaptation, requiring the optional native Omega_h bridge.

PHYDRAX_OMEGA_H_EXECUTABLE=/path/to/phydrax_omega_h \
    python examples/meshing_omega_h.py --ranks 1 --dimension 2
PHYDRAX_OMEGA_H_EXECUTABLE=/path/to/phydrax_omega_h \
    python examples/meshing_omega_h.py --ranks 2 --dimension 3

MPI launcher flags can be supplied through --launcher (shell-like splitting,
never shell execution). On affected Apple OpenMPI/hwloc installations only,
HWLOC_SYNTHETIC='pack:1 core:10 pu:1' and --launcher 'mpiexec --bind-to none
--map-by slot' avoid PRRTE's startup topology crash; processes still run real MPI.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
from itertools import permutations, product

import numpy as np

from phydrax import SpatialCoordinateContract
from phydrax.discretization import CellMesh
from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
from phydrax.meshing._sizing import MeshMetricField
from phydrax.meshing.providers._omega_h import OmegaHProvider


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--dimension", type=int, choices=(2, 3), default=2)
    parser.add_argument("--launcher", default="mpiexec")
    arguments = parser.parse_args()
    dim = arguments.dimension
    n = 8 if dim == 2 else 4
    indices = tuple(product(range(n + 1), repeat=dim))
    lookup = {point: index for index, point in enumerate(indices)}
    points = np.asarray(indices, dtype=float) / n
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
    mesh = constructor(
        points,
        np.asarray(cells),
        vertex_global_ids=np.arange(len(points), dtype=np.int64) * 3 + 17,
        cell_global_ids=np.arange(len(cells), dtype=np.int64) * 5 + 31,
    )
    scope = MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        MeshingEntityKind.MESH,
        0,
        mesh.entity_set(0).entity_set_id,
        mesh.vertex_global_ids,
    )
    matrix = np.eye(dim) * (2 * n) ** 2
    # Nonzero cross term exercises packed symmetric metric ordering in 2D/3D.
    matrix[0, 1] = matrix[1, 0] = 0.15 * matrix[0, 0]
    metric = MeshMetricField(
        scope,
        np.broadcast_to(matrix, (len(points), dim, dim)),
        minimum_size=1 / (3 * n),
        maximum_size=1 / n,
        maximum_anisotropy=2,
    )
    result = OmegaHProvider(mpi_launcher=shlex.split(arguments.launcher)).execute(
        mesh, metric, SpatialCoordinateContract.si(), ranks=arguments.ranks
    )
    target = result.target.mesh
    corners = np.asarray(target.coordinates)[np.asarray(target.blocks[0].vertices)]
    measures = np.linalg.det(corners[:, 1:] - corners[:, :1]) / math.factorial(dim)
    assert np.all(measures > 0), "Adaptation inverted a simplex"
    assert np.isclose(measures.sum(), 1, atol=1e-12), "Adaptation changed domain measure"
    assert len(corners) > len(cells), "Requested finer metric did not refine the carrier"
    assert result.target.audit.passed and result.lineage_status == "unknown"
    owned = [
        sum(owner == part.rank for owner in part.cell_owner_ranks)
        for part in result.partitions
    ]
    ghosts = [
        sum(owner != part.rank for owner in part.cell_owner_ranks)
        for part in result.partitions
    ]
    assert sum(owned) == len(corners) and all(value > 0 for value in owned)
    if arguments.ranks > 1:
        assert all(value > 0 for value in ghosts), "No real cross-rank ghost residence"
    assert np.all(np.linalg.eigvalsh(result.metric.values) > 0)
    print(
        json.dumps(
            {
                "provider": result.target.provider.version,
                "dimension": dim,
                "ranks": arguments.ranks,
                "source_cells": len(cells),
                "target_cells": len(corners),
                "owned_cells": owned,
                "ghost_cells": ghosts,
                "domain_measure": float(measures.sum()),
                "audit": result.target.audit.passed,
                "iterations": result.iterations,
                "lineage": result.lineage_status,
            }
        )
    )


if __name__ == "__main__":
    main()
