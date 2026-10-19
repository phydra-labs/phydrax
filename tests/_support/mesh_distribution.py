"""Deterministic mesh, lineage, and state builders for distribution contracts."""

from typing import Any

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.meshing import MeshPart


def cell_part(name: Any = "cells", scale: Any = 1.0, single: Any = False) -> Any:
    points = scale * np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    cells = np.asarray(((0, 1, 3), (1, 2, 3)), dtype=np.int32)
    if single:
        points = points[[0, 1, 3]]
        cells = np.asarray(((0, 1, 2),), dtype=np.int32)
    mesh = phx.discretization.CellMesh.from_triangles(points, cells)
    return MeshPart(
        name, phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    )


def fe_discretization(part: Any) -> Any:
    return phx.discretization.fem.FiniteElementPlan(
        part.carrier.mesh,
        phx.discretization.fem.FiniteElementFieldSpec(
            "u", phx.discretization.fem.discontinuous_element("triangle", 1)
        ),
        coordinate_spec=part.carrier.geometry,
    ).prepare()


def grid_part(name: Any = "grid", *, periodic: Any = True, scale: Any = 1.0) -> Any:
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(8, periodic=periodic),), axis_names=("x",)
    ).prepare(jnp.asarray([[0.0], [scale]]))
    return MeshPart(name, grid, coordinate_contract=phx.SpatialCoordinateContract.si())


def compiled_fv(part: Any) -> Any:
    discretization = phx.discretization.FiniteVolumePlan(part.carrier).prepare()
    system = phx.equations.ScalarConservationSystem(
        1,
        lambda state, axis, args: state,
        lambda left, right, axis, args: jnp.ones(left.shape[:-1]),
        system_id="meshing-distribution-advection",
    )
    problem = phx.equations.ConservationProblemIR(
        "meshing-distribution-advection",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet.periodic(("x",)),
    )
    method = phx.discretization.FiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.RusanovFluxPlan(),
    )
    return phx.equations.compile_conservation_problem(problem, discretization, method)


def quad_mesh(nx: Any, ny: Any, *, refine: Any = (), extend: Any = 0) -> Any:
    """Two triangles per unit quad; ``refine`` quads split into four about their
    center (children of the two original triangles) and ``extend`` columns are
    appended as created cells. Returns the part and each cell's source parent ID."""
    width = nx + extend
    xs, ys = np.meshgrid(np.arange(width + 1.0), np.arange(ny + 1.0), indexing="ij")
    points = [np.stack((xs.ravel(), ys.ravel()), axis=1)]
    vertex = np.arange((width + 1) * (ny + 1)).reshape(width + 1, ny + 1)
    next_vertex, next_id = vertex.size, 2 * nx * ny
    cells, ids, parents = [], [], []
    for i in range(width):
        for j in range(ny):
            a, b, c, d = (
                vertex[i, j],
                vertex[i + 1, j],
                vertex[i + 1, j + 1],
                vertex[i, j + 1],
            )
            quad = i * ny + j
            if i >= nx:
                cells += [(a, b, c), (a, c, d)]
                ids += [next_id, next_id + 1]
                parents += [-1, -1]
                next_id += 2
            elif quad in refine:
                center = next_vertex
                next_vertex += 1
                points.append(np.asarray([[i + 0.5, j + 0.5]]))
                cells += [(a, b, center), (b, c, center), (c, d, center), (d, a, center)]
                ids += [next_id + offset for offset in range(4)]
                parents += [2 * quad, 2 * quad, 2 * quad + 1, 2 * quad + 1]
                next_id += 4
            else:
                cells += [(a, b, c), (a, c, d)]
                ids += [2 * quad, 2 * quad + 1]
                parents += [2 * quad, 2 * quad + 1]
    mesh = phx.discretization.CellMesh.from_triangles(
        np.concatenate(points),
        np.asarray(cells, dtype=np.int32),
        cell_global_ids=np.asarray(ids, dtype=np.int64),
    )
    part = MeshPart(
        "adaptive",
        phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si()),
    )
    by_id = dict(zip(ids, parents, strict=True))
    # ty: ignore[unresolved-attribute]
    native = np.asarray(part.carrier.mesh.blocks[0].global_ids)
    return part, np.asarray([by_id[int(value)] for value in native])


def cell_lineage(source: Any, target: Any, parents: Any) -> Any:
    target_ids = np.asarray(target.carrier.mesh.blocks[0].global_ids)
    routed = parents >= 0
    kinds = np.where(
        parents[routed] == target_ids[routed],
        int(phx.meshing.EntityLineageKind.PRESERVED),
        int(phx.meshing.EntityLineageKind.REFINED_FROM),
    )
    cells = phx.meshing.EntityLineage(
        2,
        source.carrier.mesh.entity_set(2).entity_set_id,
        target.carrier.mesh.entity_set(2).entity_set_id,
        parents[routed],
        target_ids[routed],
        kinds,
        created_target_ids=target_ids[~routed],
    )
    return phx.meshing.MeshLineage(
        source.carrier.mesh.topology_id, target.carrier.mesh.topology_id, (cells,)
    )


def brute_force_halos(part: Any, owner: Any, part_count: Any, width: Any) -> Any:
    block = part.carrier.mesh.blocks[0]
    triangles = np.asarray(block.vertices)
    global_ids = np.asarray(block.global_ids)
    by_edge = {}
    for cell, triangle in enumerate(triangles.tolist()):
        for corner in range(3):
            edge = tuple(sorted((triangle[corner], triangle[(corner + 1) % 3])))
            by_edge.setdefault(edge, []).append(cell)
    neighbors = [set() for _ in triangles]
    for sharing in by_edge.values():
        for first in sharing:
            neighbors[first].update(cell for cell in sharing if cell != first)
    halos = []
    for rank in range(part_count):
        visited = {cell for cell in range(len(triangles)) if owner[cell] == rank}
        frontier = set(visited)
        for _ in range(width):
            frontier = set().union(*(neighbors[cell] for cell in frontier)) - visited
            visited |= frontier
        halos.append(
            sorted(int(global_ids[cell]) for cell in visited if owner[cell] != rank)
        )
    return halos


def cell_state(distribution: Any, values: Any) -> Any:
    return phx.lifecycle.CompositionEntry(
        jnp.asarray(values),
        entry_id="solid/mass",
        role="physical-state",
        owner_id="solid",
        structure_id=distribution.distribution_id,
        revision_id=f"mass-{distribution.distribution_id}",
        semantics_id="solid.cell-mass",
    )


def repartition() -> Any:
    part, parents = quad_mesh(4, 4)
    policy = phx.meshing.MeshPartitionPolicy(phx.meshing.MeshPartitionKind.PROVIDER, 2)
    cells = np.arange(parents.size)
    distribution = phx.meshing.prepare_mesh_distribution(
        part, policy=policy, ownership=cells % 2
    )
    return phx.meshing.prepare_distribution_transition(
        distribution,
        part,
        cell_lineage(part, part, parents),
        policy=policy,
        ownership=cells // (parents.size // 2),
    )
