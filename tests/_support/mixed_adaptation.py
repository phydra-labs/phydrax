"""Curved mixed templates with authored scientific IDs and periodic generators."""

import numpy as np

from phydrax.discretization import (
    CellBlock,
    CellGeometrySpec,
    CellMesh,
    PeriodicCell,
    PeriodicIsometryGroup,
    PeriodicMeshTopology,
)
from phydrax.discretization._cell_geometry import coordinate_lagrange_element
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.meshing._canonical import canonicalize_cell_mesh


def red_shared_face_source(
    mixed: bool, permuted: bool
) -> tuple[CellMesh, CellGeometrySpec]:
    if mixed:
        points = np.concatenate(
            (
                np.asarray(reference_cell_topology("prism").vertices, dtype=np.float64),
                np.asarray(((0.0, 0.0, 2.0),), dtype=np.float64),
            )
        )
        blocks = (
            CellBlock(
                "prism",
                "prism",
                np.arange(6, dtype=np.int32)[None],
                global_ids=np.asarray((11,), dtype=np.int64),
            ),
            CellBlock(
                "cap",
                "tetrahedron",
                np.asarray(((3, 4, 5, 6),), dtype=np.int32),
                global_ids=np.asarray((37,), dtype=np.int64),
            ),
        )
    else:
        points = np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.0, 0.0, -1.0),
            ),
            dtype=np.float64,
        )
        cells = ((1, 2, 0, 3), (2, 1, 0, 4)) if permuted else ((0, 1, 2, 3), (0, 2, 1, 4))
        blocks = (
            CellBlock(
                "tets",
                "tetrahedron",
                np.asarray(cells, dtype=np.int32),
                global_ids=np.asarray((11, 37), dtype=np.int64),
            ),
        )
    original = points.copy()
    points[:, 1] += points[:, 0] * points[:, 2] / 16
    mesh = CellMesh(
        points,
        blocks,
        vertex_global_ids=np.asarray((103, 71, 509, 211, 13, 17, 401), dtype=np.int64)[
            : points.shape[0]
        ],
    )
    mesh = canonicalize_cell_mesh(mesh)
    elements, routes, coefficients = {}, {}, []
    cursor = 0
    for block in mesh.blocks:
        element = coordinate_lagrange_element(block.cell_kind, 2)
        nodes = np.asarray(element.reference_nodes, dtype=np.float64)
        local = []
        for row in np.asarray(block.vertices):
            corners = original[row]
            matrix = (
                (corners[1:] - corners[0])
                if block.cell_kind == "tetrahedron"
                else np.stack(
                    (
                        corners[1] - corners[0],
                        corners[2] - corners[0],
                        corners[3] - corners[0],
                    )
                )
            )
            values = nodes @ matrix + corners[0]
            values[:, 1] += values[:, 0] * values[:, 2] / 16
            local.append(np.arange(cursor, cursor + nodes.shape[0], dtype=np.int32))
            coefficients.append(values)
            cursor += nodes.shape[0]
        elements[block.name] = element
        routes[block.name] = np.stack(local)
    return mesh, CellGeometrySpec(elements, routes, np.concatenate(coefficients))


def periodic_template_source(
    kind: str, rotated: bool
) -> tuple[CellMesh, CellGeometrySpec]:
    points = np.asarray(reference_cell_topology(kind).vertices, dtype=np.float64)
    identification: PeriodicCell | PeriodicIsometryGroup
    if rotated:
        generator = np.eye(4, dtype=np.float64)
        generator[:2, :2] = ((0, -1), (1, 0))
        identification = PeriodicIsometryGroup(generator[None])
        if kind == "pyramid":
            points[4, :2] = 0
        roots = {
            "tetrahedron": (0, 1, 1, 3),
            "prism": (0, 1, 1, 3, 4, 4),
            "hexahedron": (0, 1, 2, 1, 4, 5, 6, 5),
            "pyramid": (0, 1, 2, 1, 4),
        }[kind]
        shifts = np.asarray(
            [int(index != root) for index, root in enumerate(roots)], dtype=np.int32
        )[:, None]
    else:
        identification = PeriodicCell(np.eye(points.shape[1], dtype=np.float64))
        roots = (0,) * points.shape[0]
        shifts = points.astype(np.int32)
    block = CellBlock(
        "periodic",
        kind,
        np.arange(points.shape[0], dtype=np.int32)[None],
        global_ids=np.asarray([17], dtype=np.int64),
    )
    mesh = CellMesh(
        points,
        (block,),
        vertex_global_ids=np.arange(100, 100 + points.shape[0], dtype=np.int64),
    )
    periodic = PeriodicMeshTopology(
        mesh, identification, np.asarray(roots, dtype=np.int32), shifts
    )
    mesh = CellMesh(
        points,
        (block,),
        vertex_global_ids=mesh.vertex_global_ids,
        periodic_topology=periodic,
    )
    element = coordinate_lagrange_element(kind, 3)
    nodes = np.asarray(element.reference_nodes, dtype=np.float64).copy()
    if rotated:
        if kind == "pyramid":
            nodes[:, :2] -= nodes[:, 2, None] / 2
        nodes[:, :2] *= 1 + 0.03 * nodes[:, 2, None] * (1 - nodes[:, 2, None])
    elif kind == "hexahedron":
        nodes[:, 2] += 0.03 * np.prod(nodes * (1 - nodes), axis=1)
    else:
        nodes[:, 1] += 0.03 * np.prod(nodes * (1 - nodes), axis=1)
    geometry = CellGeometrySpec(
        {"periodic": element},
        {"periodic": np.arange(nodes.shape[0], dtype=np.int32)[None]},
        nodes,
    )
    return mesh, geometry
