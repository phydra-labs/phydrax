#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Hex core with face-conforming pyramid/tetrahedron transition closure.

Selected affine source tetrahedra contribute their four-hex dual.
The complement is coned from each source cell barycenter to one shared face
partition: fully split faces carry three quads (three pyramids), partially
split faces carry a triangular fan, and untouched faces remain triangles.
Marked edge midpoint propagation closes all interfaces without hanging nodes.
"""

from __future__ import annotations

import equinox as eqx
import jax
import numpy as np

import phydrax.discretization._cell_mesh as cell_mesh_owner
import phydrax.discretization._reference_cell_types as reference_cell_types_owner
from phydrax.meshing._host_banks import ancestry, entities, failure, family_host_array

from .._meshcore import exact_orient3d, NativeExecutionScope
from .._strict import StrictModule
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_geometry_validity import certify_cell_geometry_validity
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..typing import checked
from ._contracts import CellFamilyPolicy, MeshingLimits
from ._dual_extraction import DualExtraction
from ._family_execution import check_family_limits, coordinate_phase
from ._hex_generation import extract_volume_hexes, parent_coordinate_geometry


# Host incidence/fan dictionaries and source-coordinate rows use this admitted
# per-tetrahedron/per-vertex envelope; work covers the bounded four-face closure.
_TRANSITION_CELL_SCRATCH_BYTES = 16384
_TRANSITION_VERTEX_SCRATCH_BYTES = 256
_TRANSITION_CELL_WORK_UNITS = 512


class _HexCoreSelection(StrictModule):
    cells: np.ndarray
    selected: np.ndarray
    count: int = eqx.field(static=True)
    scratch: int = eqx.field(static=True)
    work: int = eqx.field(static=True)


class _HexTransitionPieces(StrictModule):
    hexes: np.ndarray
    selected_dual: np.ndarray
    pyramids: tuple[tuple[int, ...], ...]
    tetrahedra: tuple[tuple[int, ...], ...]
    pyramid_parents: tuple[int, ...]
    tetrahedron_parents: tuple[int, ...]


class _HexTransitionBlocks(StrictModule):
    blocks: tuple[reference_cell_types_owner.CellBlock, ...]
    parent_cells: tuple[np.ndarray, ...]
    points: np.ndarray


class _HexTransitionCompaction(StrictModule):
    used: np.ndarray
    coordinates: np.ndarray
    vertex_ids: np.ndarray
    blocks: tuple[reference_cell_types_owner.CellBlock, ...]
    supports: np.ndarray
    parents: np.ndarray


def _admit_hex_core(
    source: cell_mesh_owner.CellMesh,
    limits: MeshingLimits,
    policy: CellFamilyPolicy,
    requested: np.ndarray,
) -> _HexCoreSelection:
    if not policy.allow_mixed:
        raise failure(
            "policy.allow_mixed must permit hex-dominant transitions; got False"
        )
    requested = np.asarray(requested)
    if requested.ndim != 1 or requested.dtype.kind not in "iu" or requested.size == 0:
        raise ValueError(
            f"hex_dominant.hex_core_cells must be a nonempty integer source ID vector; got shape {requested.shape}, dtype {requested.dtype}"
        )
    cells = entities(source, 3)
    source_ids = np.asarray(
        jax.device_get(source.topology.entities(3).entity_ids), dtype=np.int64
    )
    if np.unique(requested).size != requested.size or np.any(
        ~np.isin(requested, source_ids)
    ):
        raise ValueError(
            f"hex_dominant.hex_core_cells must uniquely name source tetrahedra; got duplicate or unknown IDs in {requested.tolist()}"
        )
    selected = np.isin(source_ids, requested)
    count = cells.shape[0]
    scratch = (
        _TRANSITION_CELL_SCRATCH_BYTES * count
        + _TRANSITION_VERTEX_SCRATCH_BYTES * source.coordinates.shape[0]
    )
    work = _TRANSITION_CELL_WORK_UNITS * count
    check_family_limits(limits, 0, source.coordinates.shape[0], 0, scratch, work)
    return _HexCoreSelection(cells, selected, count, scratch, work)


def _prepare_hex_transition_pieces(
    source: cell_mesh_owner.CellMesh,
    dual: DualExtraction,
    selection: _HexCoreSelection,
) -> _HexTransitionPieces:
    edges, faces = entities(source, 1), entities(source, 2)
    original, edge_count, face_count = (
        source.coordinates.shape[0],
        edges.shape[0],
        faces.shape[0],
    )
    edge_lookup = {tuple(sorted(edge)): row for row, edge in enumerate(edges.tolist())}
    face_lookup = {tuple(sorted(face)): row for row, face in enumerate(faces.tolist())}
    marked_edges: set[int] = set()
    for tetrahedron in selection.cells[selection.selected].tolist():
        for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
            marked_edges.add(edge_lookup[tuple(sorted((tetrahedron[a], tetrahedron[b])))])
    face_pieces: list[tuple[tuple[int, ...], ...]] = []
    for row, face_ in enumerate(faces.tolist()):
        face = tuple(face_)
        edge_rows = [
            edge_lookup[tuple(sorted((face[a], face[b])))]
            for a, b in ((0, 1), (1, 2), (2, 0))
        ]
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
    selected_dual = selection.selected[dual.parent_cells]
    hexes = entities(dual.mesh, 3)[selected_dual]
    pyramids, tetrahedra, pyramid_parents, tetrahedron_parents = [], [], [], []
    for row in np.flatnonzero(~selection.selected).tolist():
        tetrahedron = selection.cells[row]
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
    return _HexTransitionPieces(
        hexes,
        selected_dual,
        tuple(pyramids),
        tuple(tetrahedra),
        tuple(pyramid_parents),
        tuple(tetrahedron_parents),
    )


def _prepare_hex_transition_blocks(
    dual: DualExtraction,
    pieces: _HexTransitionPieces,
    policy: CellFamilyPolicy,
) -> _HexTransitionBlocks:
    points = np.asarray(jax.device_get(dual.mesh.coordinates), dtype=np.float64)
    dual_ids = np.concatenate(
        [
            np.asarray(jax.device_get(block.global_ids), dtype=np.int64)
            for block in dual.mesh.blocks
        ]
    )
    blocks = [
        reference_cell_types_owner.CellBlock(
            "hex_core",
            "hexahedron",
            pieces.hexes,
            global_ids=dual_ids[pieces.selected_dual],
        )
    ]
    parent_cells = [dual.parent_cells[pieces.selected_dual]]
    next_identity = int(np.max(dual_ids)) + 1
    for name, kind, rows, parents in (
        ("pyramid_transition", "pyramid", pieces.pyramids, pieces.pyramid_parents),
        (
            "tetrahedron_transition",
            "tetrahedron",
            pieces.tetrahedra,
            pieces.tetrahedron_parents,
        ),
    ):
        if not rows:
            continue
        vertices = np.asarray(rows, dtype=np.int64)
        corners = points[vertices]
        signs = exact_orient3d(
            corners[:, 0], corners[:, 1], corners[:, 2], corners[:, -1]
        )
        if np.any(signs == 0):
            raise failure(
                f"transition.cones must have nonzero represented orientation; got cone signs {signs.tolist()}"
            )
        if kind == "pyramid":
            vertices[signs < 0] = vertices[signs < 0][:, (0, 3, 2, 1, 4)]
        else:
            vertices[signs < 0] = vertices[signs < 0][:, (0, 2, 1, 3)]
        blocks.append(
            reference_cell_types_owner.CellBlock(
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
        raise failure(
            f"transition.families must include {policy.required} and lie in {sorted(permitted)}; got {sorted(families)}"
        )
    return _HexTransitionBlocks(tuple(blocks), tuple(parent_cells), points)


def _compact_hex_transition(
    dual: DualExtraction,
    selection: _HexCoreSelection,
    transition: _HexTransitionBlocks,
    limits: MeshingLimits,
    *,
    scope: NativeExecutionScope,
) -> _HexTransitionCompaction:
    # Compact unused centers/midpoints by incidence while retaining scientific IDs.
    used = np.unique(
        np.concatenate(
            [
                np.asarray(jax.device_get(block.vertices), dtype=np.int64).reshape(-1)
                for block in transition.blocks
            ]
        )
    )
    check_family_limits(
        limits,
        sum(block.cell_count for block in transition.blocks),
        used.size,
        sum(block.vertices.size for block in transition.blocks),
        selection.scratch,
        selection.work,
    )
    compact = family_host_array((transition.points.shape[0],), np.int64, scope=scope)
    compact.fill(-1)
    compact[used] = np.arange(used.size, dtype=np.int64)
    coordinates = family_host_array((used.size, 3), np.float64, scope=scope)
    vertex_ids = family_host_array((used.size,), np.int64, scope=scope)
    np.take(transition.points, used, axis=0, out=coordinates)
    np.take(np.asarray(jax.device_get(dual.mesh.vertex_global_ids)), used, out=vertex_ids)
    target_blocks = tuple(
        reference_cell_types_owner.CellBlock(
            block.name,
            block.cell_kind,
            compact[np.asarray(jax.device_get(block.vertices), dtype=np.int64)],
            global_ids=block.global_ids,
        )
        for block in transition.blocks
    )
    supports = family_host_array((used.size, 4), np.int64, scope=scope)
    np.take(dual.vertex_supports, used, axis=0, out=supports)
    raw_parents = family_host_array(
        (sum(values.size for values in transition.parent_cells),), np.int64, scope=scope
    )
    offset = 0
    for values in transition.parent_cells:
        raw_parents[offset : offset + values.size] = values
        offset += values.size
    return _HexTransitionCompaction(
        used, coordinates, vertex_ids, target_blocks, supports, raw_parents
    )


def _certify_hex_transition(
    source: cell_mesh_owner.CellMesh,
    dual: DualExtraction,
    source_geometry: CellGeometrySpec,
    compact: _HexTransitionCompaction,
    selection: _HexCoreSelection,
    limits: MeshingLimits,
    *,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
) -> DualExtraction:
    mesh, geometry, parents = parent_coordinate_geometry(
        source,
        source_geometry,
        compact.blocks,
        compact.parents,
        compact.coordinates,
        compact.supports,
        compact.vertex_ids,
        scope=scope,
        budget=budget,
    )
    for dimension, bound in ((1, limits.maximum_edges), (2, limits.maximum_faces)):
        local_entities = mesh.topology.entities(dimension).count
        if local_entities > bound:
            raise failure(
                f"transition.entities must respect dimension {dimension} capacity {bound}; got {local_entities}",
                resource=True,
            )
    dimensions, local_ancestry = ancestry(source, mesh, compact.supports)
    with coordinate_phase(scope=scope, budget=budget):
        validity = certify_cell_geometry_validity(geometry, mesh=mesh, budget=budget)
    if validity.invalid_count or validity.unresolved_count:
        raise failure(
            "transition_geometry.validity must certify every mapped cell; got invalid or unresolved cells"
        )
    all_vertices = np.concatenate(
        [
            np.asarray(jax.device_get(block.vertices), dtype=np.int64).reshape(-1)
            for block in mesh.blocks
        ]
    )
    valences = np.bincount(all_vertices, minlength=compact.used.size).astype(np.int64)
    boundary = family_host_array((compact.used.size,), np.bool_, scope=scope)
    np.take(dual.boundary_vertices, compact.used, out=boundary)
    singular = np.flatnonzero((~boundary) & (valences != 8)).astype(np.int64)
    data = mesh.coordinates.nbytes + sum(block.vertices.nbytes for block in mesh.blocks)
    data += (
        compact.supports.nbytes
        + parents.nbytes
        + valences.nbytes
        + boundary.nbytes
        + singular.nbytes
    )
    data += sum(array.nbytes for array in dimensions + local_ancestry)
    if data > limits.maximum_data_bytes:
        raise failure(
            f"transition.data_bytes must respect capacity {limits.maximum_data_bytes}; got {data}",
            resource=True,
        )
    return DualExtraction(
        mesh,
        source,
        compact.supports,
        parents,
        dimensions,
        local_ancestry,
        validity,
        valences,
        boundary,
        singular,
        None,
        selection.work,
        geometry,
        source_geometry,
    )


@checked
def extract_hex_dominant(
    source: cell_mesh_owner.CellMesh,
    limits: MeshingLimits,
    policy: CellFamilyPolicy,
    /,
    *,
    hex_core_cells: np.ndarray,
    source_geometry: CellGeometrySpec | None = None,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
) -> DualExtraction:
    """Close stable source-cell IDs into a hex core with conforming transitions."""
    selection = _admit_hex_core(source, limits, policy, hex_core_cells)
    dual = extract_volume_hexes(
        source, limits, source_geometry=source_geometry, scope=scope, budget=budget
    )
    if dual.source_geometry is None:
        raise failure(
            "dual.source_geometry must retain the tetrahedral coordinate law; got None"
        )
    pieces = _prepare_hex_transition_pieces(source, dual, selection)
    transition = _prepare_hex_transition_blocks(dual, pieces, policy)
    compact = _compact_hex_transition(dual, selection, transition, limits, scope=scope)
    return _certify_hex_transition(
        source,
        dual,
        dual.source_geometry,
        compact,
        selection,
        limits,
        scope=scope,
        budget=budget,
    )
