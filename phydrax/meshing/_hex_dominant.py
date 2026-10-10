#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Hex core with face-conforming pyramid/tetrahedron transition closure.

Selected affine source tetrahedra contribute their genuine four-hex dual.
The complement is coned from each source cell barycenter to one shared face
partition: fully split faces carry three quads (three pyramids), partially
split faces carry a triangular fan, and untouched faces remain triangles.
Marked edge midpoint propagation closes all interfaces without hanging nodes.
No pure-hex request is silently changed to a hybrid request.
"""

from __future__ import annotations

import numpy as np

from .._meshcore import exact_orient3d
from ..discretization import CellBlock, CellMesh
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_geometry_validity import certify_cell_geometry_validity
from ._contracts import CellFamilyPolicy, MeshingLimits
from ._hex_generation import _parent_coordinate_geometry, extract_volume_hexes
from ._quad_generation import (
    _ancestry,
    _budget,
    _entities,
    _failure,
    _family_host_array,
    DualExtraction,
)


def extract_hex_dominant(
    source: CellMesh,
    limits: MeshingLimits,
    policy: CellFamilyPolicy,
    /,
    *,
    hex_core_cells: np.ndarray,
    source_geometry: CellGeometrySpec | None = None,
) -> DualExtraction:
    """Close an explicitly selected source-cell hex core under the family policy.

    ``hex_core_cells`` are stable source *cell IDs*, not local reusable slots.
    This combinatorial closure does not claim feature-aligned core placement or
    continuous fidelity to an unrepresented curved source.
    """
    if not policy.allow_mixed:
        raise _failure(
            "Hex-dominant transitions require an explicitly mixed family policy."
        )
    requested = np.asarray(hex_core_cells)
    if requested.ndim != 1 or requested.dtype.kind not in "iu" or requested.size == 0:
        raise ValueError("hex_core_cells must be a nonempty integer source ID vector.")
    cells = _entities(source, 3)
    source_ids = np.asarray(source.topology.entities(3).entity_ids, dtype=np.int64)
    if np.unique(requested).size != requested.size or np.any(
        ~np.isin(requested, source_ids)
    ):
        raise ValueError("Hex core IDs must uniquely name source tetrahedra.")
    selected = np.isin(source_ids, requested)
    count = cells.shape[0]
    # Every complementary face fan has at most six triangles; four cones per
    # source tetra therefore require at most 24 transition tetrahedra.
    _budget(
        limits,
        0,
        source.coordinates.shape[0],
        0,
        16384 * count + 256 * source.coordinates.shape[0],
        512 * count,
    )
    dual = extract_volume_hexes(source, limits, source_geometry=source_geometry)
    if dual.source_geometry is None:
        raise _failure(
            "Hex transitions require the actual original tetrahedral source coordinate law."
        )
    edges, faces = _entities(source, 1), _entities(source, 2)
    original, edge_count, face_count = (
        source.coordinates.shape[0],
        edges.shape[0],
        faces.shape[0],
    )
    edge_lookup = {tuple(sorted(edge)): row for row, edge in enumerate(edges.tolist())}
    face_lookup = {tuple(sorted(face)): row for row, face in enumerate(faces.tolist())}
    marked_edges: set[int] = set()
    for tetrahedron in cells[selected].tolist():
        for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
            marked_edges.add(edge_lookup[tuple(sorted((tetrahedron[a], tetrahedron[b])))])
    face_pieces: list[tuple[tuple[int, ...], ...]] = []
    for row, face_ in enumerate(faces.tolist()):
        face = tuple(face_)
        local = ((0, 1), (1, 2), (2, 0))
        edge_rows = [edge_lookup[tuple(sorted((face[a], face[b])))] for a, b in local]
        split = [edge in marked_edges for edge in edge_rows]
        center = original + edge_count + row
        if all(split):
            mids = [original + edge for edge in edge_rows]
            face_pieces.append(
                tuple((face[v], mids[v], center, mids[(v + 2) % 3]) for v in range(3))
            )
        elif any(split):
            ring = []
            for v in range(3):
                ring.append(face[v])
                if split[v]:
                    ring.append(original + edge_rows[v])
            face_pieces.append(
                tuple(
                    (ring[v], ring[(v + 1) % len(ring)], center) for v in range(len(ring))
                )
            )
        else:
            face_pieces.append((face,))
    selected_dual = selected[dual.parent_cells]
    hexes = _entities(dual.mesh, 3)[selected_dual]
    pyramids, tetrahedra, pyramid_parents, tetrahedron_parents = [], [], [], []
    for row in np.flatnonzero(~selected).tolist():
        tetrahedron = cells[row]
        center = original + edge_count + face_count + row
        for local in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
            face = face_lookup[tuple(sorted(tetrahedron[list(local)].tolist()))]
            for piece in face_pieces[face]:
                if len(piece) == 4:
                    pyramids.append((*piece, center))
                    pyramid_parents.append(row)
                else:
                    tetrahedra.append((*piece, center))
                    tetrahedron_parents.append(row)
    points = np.asarray(dual.mesh.coordinates, dtype=np.float64)
    dual_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in dual.mesh.blocks]
    )
    blocks = [
        CellBlock("hex_core", "hexahedron", hexes, global_ids=dual_ids[selected_dual])
    ]
    parent_cells = [dual.parent_cells[selected_dual]]
    next_identity = int(np.max(dual_ids)) + 1
    for name, kind, rows, parents in (
        ("pyramid_transition", "pyramid", pyramids, pyramid_parents),
        ("tetrahedron_transition", "tetrahedron", tetrahedra, tetrahedron_parents),
    ):
        if not rows:
            continue
        vertices = np.asarray(rows, dtype=np.int64)
        corners = points[vertices]
        signs = exact_orient3d(
            corners[:, 0], corners[:, 1], corners[:, 2], corners[:, -1]
        )
        if np.any(signs == 0):
            raise _failure("A transition cone is degenerate in represented coordinates.")
        if kind == "pyramid":
            vertices[signs < 0] = vertices[signs < 0][:, (0, 3, 2, 1, 4)]
        else:
            vertices[signs < 0] = vertices[signs < 0][:, (0, 2, 1, 3)]
        blocks.append(
            CellBlock(
                name,
                kind,
                vertices,
                global_ids=np.arange(
                    next_identity, next_identity + vertices.shape[0], dtype=np.int64
                ),
            )
        )
        next_identity += vertices.shape[0]
        parent_cells.append(np.asarray(parents, dtype=np.int64))
    families = {block.cell_kind for block in blocks}
    permitted = set((*policy.required, *policy.preferred, *policy.allowed_transitions))
    if not families <= permitted or not set(policy.required) <= families:
        raise _failure(
            f"Transition closure produces {sorted(families)}; required {policy.required}, permitted {sorted(permitted)}."
        )
    # Compact the template's unneeded centers/midpoints once. Scientific IDs are
    # retained; no coordinate proximity merging is performed.
    used = np.unique(
        np.concatenate(
            [np.asarray(block.vertices, dtype=np.int64).reshape(-1) for block in blocks]
        )
    )
    _budget(
        limits,
        sum(block.cell_count for block in blocks),
        used.size,
        sum(block.vertices.size for block in blocks),
        16384 * count + 256 * source.coordinates.shape[0],
        512 * count,
    )
    compact = _family_host_array((points.shape[0],), np.int64)
    compact.fill(-1)
    compact[used] = np.arange(used.size, dtype=np.int64)
    coordinates = _family_host_array((used.size, 3), np.float64)
    vertex_ids = _family_host_array((used.size,), np.int64)
    np.take(points, used, axis=0, out=coordinates)
    np.take(np.asarray(dual.mesh.vertex_global_ids), used, out=vertex_ids)
    target_blocks = tuple(
        CellBlock(
            block.name,
            block.cell_kind,
            compact[np.asarray(block.vertices, dtype=np.int64)],
            global_ids=block.global_ids,
        )
        for block in blocks
    )
    supports = _family_host_array((used.size, 4), np.int64)
    np.take(dual.vertex_supports, used, axis=0, out=supports)
    raw_parents = _family_host_array(
        (sum(values.size for values in parent_cells),), np.int64
    )
    offset = 0
    for values in parent_cells:
        raw_parents[offset : offset + values.size] = values
        offset += values.size
    mesh, geometry, parents = _parent_coordinate_geometry(
        source,
        dual.source_geometry,
        target_blocks,
        raw_parents,
        coordinates,
        supports,
        vertex_ids,
    )
    for dimension, bound in ((1, limits.maximum_edges), (2, limits.maximum_faces)):
        entities = mesh.topology.entities(dimension).count
        if entities > bound:
            raise _failure(
                f"Hex-dominant closure requires {entities} dimension-{dimension} entities; limit is {bound}.",
                resource=True,
            )
    dimensions, ancestry = _ancestry(source, mesh, supports)
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    if validity.invalid_count or validity.unresolved_count:
        raise _failure("Transition mapped-cell validity is invalid or unresolved.")
    all_vertices = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64).reshape(-1) for block in mesh.blocks]
    )
    valences = np.bincount(all_vertices, minlength=used.size).astype(np.int64)
    # Boundary is inherited combinatorially from the already validated source
    # partition; compaction cannot turn an interior entity into a boundary one.
    boundary = _family_host_array((used.size,), np.bool_)
    np.take(dual.boundary_vertices, used, out=boundary)
    singular = np.flatnonzero((~boundary) & (valences != 8)).astype(np.int64)
    data = mesh.coordinates.nbytes + sum(block.vertices.nbytes for block in mesh.blocks)
    data += (
        supports.nbytes
        + parents.nbytes
        + valences.nbytes
        + boundary.nbytes
        + singular.nbytes
    )
    data += sum(array.nbytes for array in dimensions + ancestry)
    if data > limits.maximum_data_bytes:
        raise _failure(
            f"Hex-dominant publication requires {data} retained array bytes; limit is {limits.maximum_data_bytes}.",
            resource=True,
        )
    return DualExtraction(
        mesh,
        source,
        supports,
        parents,
        dimensions,
        ancestry,
        validity,
        valences,
        boundary,
        singular,
        None,
        512 * count,
        geometry,
        dual.source_geometry,
    )
