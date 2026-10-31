#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Owner-local conservative FV overset preparation over a composite local layout."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from itertools import product
from typing import Any, NamedTuple

import jax
import numpy as np

from .._execution_runtime import ExecutionGroup
from ..discretization._cell_mesh import CellBlock, CellMesh
from ..discretization._mapped_locator import prepare_owner_local_mapped_cell_locator
from ..discretization._periodic_cell import PeriodicCell
from ..discretization._view_support import (
    owner_local_union_mesh,
    OwnerLocalCellGeometry,
    OwnerLocalCompositeCellGeometry,
    OwnerLocalRemoteGhostCells,
)
from ..discretization.finite_volume._unstructured import (
    UnstructuredFiniteVolumeDiscretization,
    UnstructuredFiniteVolumePlan,
)
from ..discretization.finite_volume._unstructured_overset import UnstructuredOversetPlan
from ..geometry._supermesh import (
    CommonRefinementCoverage,
    CommonRefinementPolicy,
    CommonRefinementStatus,
    prepare_common_refinement,
)
from ._distribution import (
    _owner_exchange_message_banks,
    _owner_group_allgather,
    _owner_group_fail_with_reasons,
)
from ._overset import _to_source, OversetCellStatus, OversetConnectivity
from ._overset_exchange import _periodic_donor_image_frame, _periodic_donor_images


class OwnerLocalFiniteVolumeOverset(NamedTuple):
    """Composite local layout, its lowered FV geometry, and the overlap plan."""

    composite: OwnerLocalCompositeCellGeometry
    geometry: UnstructuredFiniteVolumeDiscretization
    overset: UnstructuredOversetPlan


def _encode_rows(rows: Sequence[Mapping[str, Any]], /) -> np.ndarray:
    encoded = bytearray()
    for row in rows:
        raw = json.dumps(row, sort_keys=True, separators=(",", ":")).encode("ascii")
        encoded.extend(len(raw).to_bytes(8, byteorder="big", signed=False))
        encoded.extend(raw)
    if not encoded:
        return np.zeros((0, 1), dtype=np.uint8)
    return np.frombuffer(bytes(encoded), dtype=np.uint8).reshape((-1, 1)).copy()


def _decode_rows(bank: np.ndarray, /) -> list[dict[str, Any]]:
    raw = np.asarray(bank, dtype=np.uint8).reshape(-1).tobytes()
    rows: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(raw):
        if cursor + 8 > len(raw):
            raise ValueError("Owner-local FV overset message is truncated.")
        size = int.from_bytes(raw[cursor : cursor + 8], "big", signed=False)
        cursor += 8
        if size == 0 or cursor + size > len(raw):
            raise ValueError("Owner-local FV overset message has an invalid length.")
        row = json.loads(raw[cursor : cursor + size])
        cursor += size
        if not isinstance(row, dict):
            raise TypeError("Owner-local FV overset message rows must be objects.")
        rows.append(row)
    return rows


def _bits(values: np.ndarray, /) -> list[int]:
    return [int(value) for value in np.asarray(values, dtype=np.float64).view(np.uint64)]


def _from_bits(values: Sequence[int], /) -> np.ndarray:
    return np.asarray(values, dtype=np.uint64).view(np.float64)


def _cell_corners(view: OwnerLocalCellGeometry, /) -> dict[int, tuple[str, np.ndarray]]:
    """Lifted corner coordinates of every resident cell, keyed by stable cell ID."""
    mesh = view.reconstruct_local_geometry()[0]
    if mesh is None:
        return {}
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    corners: dict[int, tuple[str, np.ndarray]] = {}
    for block in mesh.blocks:
        if not isinstance(block, CellBlock):
            raise TypeError("Owner-local FV overset requires homogeneous cell blocks.")
        rows = np.asarray(block.vertices, dtype=np.int64)
        for cell_id, row in zip(
            np.asarray(block.global_ids, dtype=np.int64).tolist(), rows, strict=True
        ):
            corners[int(cell_id)] = (block.cell_kind, coordinates[row])
    return corners


def _fringe_submesh(view: OwnerLocalCellGeometry, cell_ids: np.ndarray, /) -> CellMesh:
    """Compact carrier of the selected receptor cells with their own vertex rows."""
    mesh = view.reconstruct_local_geometry()[0]
    if mesh is None:
        raise ValueError("Fringe receptor cells lack their owner-local carrier.")
    return _cell_submesh(mesh, cell_ids)


def _cell_submesh(mesh: CellMesh, cell_ids: np.ndarray, /) -> CellMesh:
    """Compact submesh of the selected cells with their own vertex rows."""
    wanted = set(int(value) for value in cell_ids.tolist())
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    selected_blocks: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    used_rows: set[int] = set()
    for block in mesh.blocks:
        if not isinstance(block, CellBlock):
            raise TypeError("Owner-local FV overset requires homogeneous cell blocks.")
        block_ids = np.asarray(block.global_ids, dtype=np.int64)
        mask = np.asarray([int(value) in wanted for value in block_ids], dtype=np.bool_)
        if not np.any(mask):
            continue
        rows = np.asarray(block.vertices, dtype=np.int64)[mask]
        used_rows.update(int(value) for value in rows.reshape(-1))
        selected_blocks.append((block.name, block.cell_kind, rows, block_ids[mask]))
    order = np.asarray(sorted(used_rows), dtype=np.int64)
    remap = {int(row): index for index, row in enumerate(order.tolist())}
    blocks = tuple(
        CellBlock(
            name,
            kind,
            np.vectorize(remap.__getitem__, otypes=[np.int64])(rows),
            global_ids=ids,
        )
        for name, kind, rows, ids in selected_blocks
    )
    return CellMesh(coordinates[order], blocks, vertex_global_ids=vertex_ids[order])


def _unique_receptor_faces(
    geometry: UnstructuredFiniteVolumeDiscretization,
    route_receptor_cells: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assign each routed receptor cell one distinct physical face of its own."""
    receptor_cells = np.asarray(route_receptor_cells, dtype=np.int32)
    neighbor_cells = np.asarray(geometry.neighbor_cells, dtype=np.int32)
    owner_cells = np.asarray(geometry.owner_cells, dtype=np.int32)
    internal_faces = np.flatnonzero(neighbor_cells >= 0)
    boundary_faces = np.flatnonzero(neighbor_cells < 0)
    faces_by_cell: dict[int, list[int]] = {}
    for face in np.concatenate((internal_faces, boundary_faces)):
        faces_by_cell.setdefault(int(owner_cells[face]), []).append(int(face))
        neighbor = int(neighbor_cells[face])
        if neighbor >= 0:
            faces_by_cell.setdefault(neighbor, []).append(int(face))
    route_face_ids = np.full(receptor_cells.shape, -1, dtype=np.int32)
    route_by_face: dict[int, int] = {}

    def assign(route: int, visited: set[int], /) -> bool:
        for face in faces_by_cell.get(int(receptor_cells[route]), ()):
            if face in visited:
                continue
            visited.add(face)
            previous = route_by_face.get(face)
            if previous is None or assign(previous, visited):
                route_by_face[face] = route
                route_face_ids[route] = face
                return True
        return False

    for route in range(receptor_cells.size):
        if not assign(route, set()):
            raise ValueError(
                "Owner-local FV routed receptor cells exceed their distinct faces."
            )
    face_points = np.asarray(geometry.face_quadrature_points)[route_face_ids]
    area_vectors = np.asarray(geometry.area_vectors)[route_face_ids]
    face_measures = np.asarray(geometry.face_quadrature_weights)[route_face_ids]
    orientation = np.where(owner_cells[route_face_ids] == receptor_cells, 1.0, -1.0)
    normals = np.broadcast_to(
        orientation[:, None, None]
        * (area_vectors / np.asarray(geometry.face_measures)[route_face_ids, None])[
            :, None, :
        ],
        face_points.shape,
    )
    return route_face_ids, face_points, normals, face_measures


def _periodic_lattice(view: OwnerLocalCellGeometry, /) -> PeriodicCell | None:
    """The authored periodic lattice of a periodic owner-local view, if any."""
    from ._periodic import PeriodicPointOrbits
    from .providers._native_periodic import NativePeriodicSource

    source = view.source_value
    if (
        view.periodic_lift_matrix is None
        or not isinstance(source, NativePeriodicSource)
        or not isinstance(source.domain, PeriodicPointOrbits)
    ):
        return None
    if source.domain.cell.cell_id != view.periodic_lift_id:
        raise ValueError("Periodic donor view does not bind its source lattice.")
    return source.domain.cell


def _periodic_image_overlap(
    donor_mesh: CellMesh,
    fringe_mesh: CellMesh,
    lattice: PeriodicCell | None,
    /,
) -> tuple[dict[int, list[tuple[int, float]]], str, set[int]]:
    """Physical overlap of fringe cells with resident donor lifts over all images.

    Each fringe cell is translated by every lattice image that can meet the
    donor carrier's lifted extent; overlaps are summed per receptor cell and
    donor cell, and every fringe cell must be covered exactly once.
    """
    vectors = (
        np.zeros((0, fringe_mesh.ambient_dimension), dtype=np.float64)
        if lattice is None
        else np.asarray(lattice.vectors, dtype=np.float64)
    )
    shifts: list[tuple[int, ...]] = [tuple(0 for _ in range(vectors.shape[0]))]
    if lattice is not None:
        donor_fractional = np.asarray(
            lattice.fractional(np.asarray(donor_mesh.coordinates)), dtype=np.float64
        )
        fringe_fractional = np.asarray(
            lattice.fractional(np.asarray(fringe_mesh.coordinates)), dtype=np.float64
        )
        ranges = [
            range(
                int(np.floor(low - high_f)),
                int(np.ceil(high - low_f)) + 1,
            )
            for low, high, low_f, high_f in zip(
                donor_fractional.min(axis=0),
                donor_fractional.max(axis=0),
                fringe_fractional.min(axis=0),
                fringe_fractional.max(axis=0),
                strict=True,
            )
        ]
        shifts = [tuple(int(value) for value in shift) for shift in product(*ranges)]
    coordinates = np.asarray(fringe_mesh.coordinates, dtype=np.float64)
    # Image copies are a local scratch mesh: sequential identities keep the
    # common-refinement CSR grouped in row order.
    image_coordinates: list[np.ndarray] = []
    image_blocks: list[CellBlock] = []
    receptor_of_copy: dict[int, int] = {}
    base = 0
    next_cell = 0
    for shift in shifts:
        translation = np.asarray(shift, dtype=np.float64) @ vectors
        image_coordinates.append(coordinates + translation)
        for block in fringe_mesh.blocks:
            if not isinstance(block, CellBlock):
                raise TypeError("Owner-local FV overset requires homogeneous blocks.")
            receptor_ids = np.asarray(block.global_ids, dtype=np.int64)
            copy_ids = np.arange(next_cell, next_cell + receptor_ids.size, dtype=np.int64)
            next_cell += receptor_ids.size
            receptor_of_copy.update(
                zip(copy_ids.tolist(), receptor_ids.tolist(), strict=True)
            )
            image_blocks.append(
                CellBlock(
                    f"{block.name}/image-{'_'.join(str(value) for value in shift)}",
                    block.cell_kind,
                    np.asarray(block.vertices, dtype=np.int64) + base,
                    global_ids=copy_ids,
                )
            )
        base += coordinates.shape[0]
    images = CellMesh(
        np.concatenate(image_coordinates, axis=0),
        tuple(image_blocks),
        vertex_global_ids=np.arange(base, dtype=np.int64),
    )
    policy = CommonRefinementPolicy(coverage=CommonRefinementCoverage.PARTIAL)
    refinement = prepare_common_refinement(donor_mesh, images, policy=policy)
    if refinement.status is not CommonRefinementStatus.SUCCESS:
        return {}, refinement.status.name, set()
    source_ids = np.asarray(refinement.source_cell_global_ids, dtype=np.int64)
    target_ids = np.asarray(refinement.target_cell_global_ids, dtype=np.int64)
    target_offsets = np.asarray(refinement.target_offsets, dtype=np.int64)
    source_routes = np.asarray(refinement.source_cells, dtype=np.int64)
    volumes = np.asarray(refinement.volumes, dtype=np.float64)
    target_measures = np.asarray(refinement.target_measures, dtype=np.float64)
    covered: dict[int, float] = {}
    measure_of: dict[int, float] = {}
    by_receptor: dict[int, dict[int, float]] = {}
    for target_row, copy_id in enumerate(target_ids.tolist()):
        receptor_id = receptor_of_copy[int(copy_id)]
        measure_of[receptor_id] = float(target_measures[target_row])
        start, stop = (
            int(value) for value in target_offsets[target_row : target_row + 2]
        )
        for source_row, volume in zip(
            source_routes[start:stop].tolist(), volumes[start:stop].tolist(), strict=True
        ):
            donor_id = int(source_ids[source_row])
            row = by_receptor.setdefault(receptor_id, {})
            row[donor_id] = row.get(donor_id, 0.0) + float(volume)
            covered[receptor_id] = covered.get(receptor_id, 0.0) + float(volume)
    uncovered: set[int] = set()
    for receptor_id, measure in measure_of.items():
        tolerance = policy.coverage_tolerance * measure + 64.0 * np.finfo(
            np.float64
        ).eps * max(measure, 1.0)
        defect = covered.get(receptor_id, 0.0) - measure
        if defect > tolerance:
            return {}, "double coverage", set()
        if -defect > tolerance:
            uncovered.add(receptor_id)
    routes = {
        receptor_id: sorted(row.items()) for receptor_id, row in by_receptor.items()
    }
    return routes, "gap" if uncovered else "covered", uncovered


def _serve_requests(
    donor_view: OwnerLocalCellGeometry,
    donor_status: Mapping[int, int],
    corners: Mapping[int, tuple[str, np.ndarray]],
    requests: Sequence[Mapping[str, Any]],
    rank: int,
    /,
) -> list[dict[str, Any]]:
    """Place requested owned donor cells into each requester's lifted frame."""
    owned = {
        int(identifier)
        for identifier, is_owned in zip(
            np.asarray(donor_view.cell_ids, dtype=np.int64).tolist(),
            np.asarray(donor_view.owned, dtype=np.bool_).tolist(),
            strict=True,
        )
        if is_owned
    }
    graph_ids = np.asarray(donor_view.owned_cell_ids, dtype=np.int64)
    graph_row = {int(value): row for row, value in enumerate(graph_ids.tolist())}
    offsets = np.asarray(donor_view.cell_neighbor_offsets, dtype=np.int64)
    neighbor_ids = np.asarray(donor_view.cell_neighbor_ids, dtype=np.int64)
    neighbor_owners = np.asarray(donor_view.cell_neighbor_owner_ranks, dtype=np.int32)
    replies: list[dict[str, Any]] = []
    for request in requests:
        cell_id = int(request["cell"])
        if cell_id not in owned or cell_id not in corners:
            raise ValueError("A donor-cell request reached a rank that does not own it.")
        kind, cell_corners = corners[cell_id]
        block_name = next(
            block.name
            for block in donor_view.blocks
            if cell_id in set(np.asarray(block.cell_ids, dtype=np.int64).tolist())
        )
        block = next(item for item in donor_view.blocks if item.name == block_name)
        block_ids = np.asarray(block.cell_ids, dtype=np.int64)
        mask = block_ids == cell_id
        frame = _periodic_donor_image_frame(donor_view, block, mask)
        if request["point"] is None:
            # Frontier requests carry no requester point; overlap is evaluated
            # over every lattice image, so the owner's own lift suffices.
            translation = np.zeros((cell_corners.shape[1],), dtype=np.float64)
        else:
            point = _from_bits(request["point"])[None, :]
            images = _periodic_donor_images(frame, point)
            location = prepare_owner_local_mapped_cell_locator(
                donor_view, block_name
            ).locate(images, cell_mask=mask)
            found = np.flatnonzero(
                np.any(np.asarray(location.candidate_cells) >= 0, axis=1)
            )
            if found.size == 0:
                raise ValueError(
                    "A requested donor cell does not contain its requester's point in "
                    "any lattice image."
                )
            translation = images[int(found[0])] - point[0]
        shift = (
            []
            if frame is None
            else [
                int(value)
                for value in np.rint(
                    translation @ np.asarray(frame[0].inverse_vectors, dtype=np.float64)
                ).tolist()
            ]
        )
        neighbors = []
        row = graph_row.get(cell_id)
        if row is not None:
            for neighbor, owner in zip(
                neighbor_ids[offsets[row] : offsets[row + 1]].tolist(),
                neighbor_owners[offsets[row] : offsets[row + 1]].tolist(),
                strict=True,
            ):
                neighbor_corners = corners.get(int(neighbor))
                if neighbor_corners is None:
                    raise ValueError(
                        "A donor owner lacks the resident face neighbor of its cell."
                    )
                neighbors.append(
                    {
                        "cell": int(neighbor),
                        "owner": int(owner),
                        "point": _bits(
                            np.mean(neighbor_corners[1], axis=0) - translation
                        ),
                    }
                )
        replies.append(
            {
                "cell": cell_id,
                "owner": rank,
                "kind": kind,
                "translation": _bits(translation),
                "shift": shift,
                "corners": _bits((cell_corners - translation).reshape(-1)),
                "status": int(donor_status[cell_id]),
                "neighbors": neighbors,
            }
        )
    return replies


def prepare_owner_local_finite_volume_overset(
    connectivity: OversetConnectivity,
    views: Mapping[str, OwnerLocalCellGeometry],
    donor_name: str,
    receptor_name: str,
    execution_group: ExecutionGroup,
    /,
    *,
    component_names: Sequence[str],
    field_name: str = "state",
    axis_name: str = "owner_local_fv_overset",
) -> OwnerLocalFiniteVolumeOverset:
    """Prepare one owner-local conservative FV overset coupling.

    Donor and receptor parts are compiled as one composite local layout. Each
    rank's owned fringe receptor cells must be covered by donor cells resident
    in that layout; missing coverage is closed collectively by fetching the
    native donor winners of the fringe receptor points from their owners, placed
    in this rank's lattice image, and then their face-neighbor rings, until the
    local common refinement covers every fringe cell. Expansion stops with a
    collective refusal only when a round can add no donor cell.
    """
    if not isinstance(connectivity, OversetConnectivity):
        raise TypeError("connectivity must be OversetConnectivity.")
    if not isinstance(execution_group, ExecutionGroup):
        raise TypeError("execution_group must be ExecutionGroup.")
    if set(views) != {donor_name, receptor_name} or donor_name == receptor_name:
        raise ValueError(
            "Owner-local FV overset requires exactly its donor and receptor."
        )
    donor_view = views[donor_name]
    receptor_view = views[receptor_name]
    if not isinstance(donor_view, OwnerLocalCellGeometry) or not isinstance(
        receptor_view, OwnerLocalCellGeometry
    ):
        raise TypeError("Owner-local FV overset parts require OwnerLocalCellGeometry.")
    members = execution_group.spec.process_indices
    rank = members.index(jax.process_index())
    spec_by_name = {spec.part_name: spec for spec in connectivity.specs}
    donor_spec, receptor_spec = spec_by_name[donor_name], spec_by_name[receptor_name]

    def statuses(name: str, /) -> dict[int, int]:
        blanking = connectivity.blanking_of(name)
        return {
            int(identifier): int(status)
            for identifier, status in zip(
                np.asarray(blanking.cell_ids, dtype=np.int64).tolist(),
                np.asarray(blanking.cell_status, dtype=np.int32).tolist(),
                strict=True,
            )
        }

    donor_status = statuses(donor_name)
    receptor_status = statuses(receptor_name)
    receptor_cells = np.asarray(receptor_view.cell_ids, dtype=np.int64)
    receptor_owned = np.asarray(receptor_view.owned, dtype=np.bool_)
    fringe_ids = np.asarray(
        sorted(
            int(identifier)
            for identifier, owned in zip(
                receptor_cells.tolist(), receptor_owned.tolist(), strict=True
            )
            if owned and receptor_status[int(identifier)] == int(OversetCellStatus.FRINGE)
        ),
        dtype=np.int64,
    )
    donor_corners = _cell_corners(donor_view)
    donor_resident = set(np.asarray(donor_view.cell_ids, dtype=np.int64).tolist())
    receptor_mesh = receptor_view.reconstruct_local_geometry()[0]

    # Native winners of fringe receptor points seed the donor halo expansion.
    evidence = connectivity.receptors_of(receptor_name)
    donor_index = connectivity.part_index(donor_name)
    winners = {
        int(vertex): int(cell)
        for vertex, found, part, cell in zip(
            np.asarray(evidence.receptor_ids, dtype=np.int64).tolist(),
            np.asarray(evidence.found, dtype=np.bool_).tolist(),
            np.asarray(evidence.donor_parts, dtype=np.int32).tolist(),
            np.asarray(evidence.donor_cells, dtype=np.int64).tolist(),
            strict=True,
        )
        if found and part == donor_index
    }
    pending: dict[int, tuple[int | None, np.ndarray | None]] = {}
    if fringe_ids.size and receptor_mesh is not None:
        vertex_ids = np.asarray(receptor_mesh.vertex_global_ids, dtype=np.int64)
        coordinates = np.asarray(receptor_mesh.coordinates, dtype=np.float64)
        fringe_set = set(fringe_ids.tolist())
        for block in receptor_mesh.blocks:
            if not isinstance(block, CellBlock):
                raise TypeError("Owner-local FV overset requires homogeneous blocks.")
            block_vertices = np.asarray(block.vertices, dtype=np.int64)
            for cell_id, cell_row in zip(
                np.asarray(block.global_ids, dtype=np.int64).tolist(),
                range(block_vertices.shape[0]),
                strict=True,
            ):
                if cell_id not in fringe_set:
                    continue
                for row in block_vertices[cell_row].tolist():
                    winner = winners.get(int(vertex_ids[row]))
                    if winner is None or winner in donor_resident:
                        continue
                    pending.setdefault(
                        winner,
                        (
                            None,
                            _to_source(
                                coordinates[row][None, :], receptor_spec, donor_spec
                            )[0],
                        ),
                    )

    donor_graph_row = {
        int(identifier): row
        for row, identifier in enumerate(
            np.asarray(donor_view.owned_cell_ids, dtype=np.int64).tolist()
        )
    }
    donor_neighbor_offsets = np.asarray(donor_view.cell_neighbor_offsets, dtype=np.int64)
    donor_neighbor_ids = np.asarray(donor_view.cell_neighbor_ids, dtype=np.int64)
    donor_neighbor_owners = np.asarray(
        donor_view.cell_neighbor_owner_ranks, dtype=np.int32
    )
    donor_owner_of = {
        int(identifier): int(owner)
        for identifier, owner in zip(
            np.asarray(donor_view.cell_ids, dtype=np.int64).tolist(),
            np.asarray(donor_view.cell_owner_ranks, dtype=np.int32).tolist(),
            strict=True,
        )
    }
    ghost_rows: dict[int, dict[str, Any]] = {}
    # Face neighbors of resident halo cells, obtained from their owners.
    probed: dict[int, list[dict[str, Any]]] = {}

    def frontier_requests(
        overlap_routes: Mapping[int, Sequence[tuple[int, float]]],
        uncovered: set[int],
        /,
    ) -> dict[int, tuple[int | None, np.ndarray | None]]:
        """Face neighbors of donor cells meeting uncovered fringe cells."""
        requests: dict[int, tuple[int | None, np.ndarray | None]] = {}
        frontier = sorted(
            {
                donor_id
                for receptor_id in uncovered
                for donor_id, _ in overlap_routes.get(receptor_id, ())
            }
        )
        for donor_id in frontier:
            neighbors: list[tuple[int, int, np.ndarray | None]] = []
            if donor_id in ghost_rows:
                rows = ghost_rows[donor_id]["neighbors"]
            elif donor_id in probed:
                rows = probed[donor_id]
            elif donor_id in donor_graph_row:
                row = donor_graph_row[donor_id]
                rows = []
                neighbors.extend(
                    (int(neighbor), int(owner), None)
                    for neighbor, owner in zip(
                        donor_neighbor_ids[
                            donor_neighbor_offsets[row] : donor_neighbor_offsets[row + 1]
                        ].tolist(),
                        donor_neighbor_owners[
                            donor_neighbor_offsets[row] : donor_neighbor_offsets[row + 1]
                        ].tolist(),
                        strict=True,
                    )
                )
            else:
                # A resident halo cell: its owner reports its face neighbors.
                requests.setdefault(donor_id, (donor_owner_of[donor_id], None))
                continue
            neighbors.extend(
                (int(item["cell"]), int(item["owner"]), _from_bits(item["point"]))
                for item in rows
            )
            for neighbor, owner, point in neighbors:
                if neighbor in donor_resident or neighbor in ghost_rows:
                    continue
                requests.setdefault(neighbor, (owner, point))
        return requests

    expansion_round = 0
    while True:
        ghosts = _ghost_cells(donor_name, ghost_rows)
        overlap_routes: dict[int, list[tuple[int, float]]] = {}
        uncovered: set[int] = set()
        requests: dict[int, tuple[int | None, np.ndarray | None]] = {}
        coverage_status = "covered"
        local_error = None
        try:
            if fringe_ids.size:
                union = owner_local_union_mesh(((donor_name, donor_view),), ghosts)
                # Blanked donor cells are not physical donors.
                live_donors = np.asarray(
                    [
                        identifier
                        for identifier in sorted(donor_resident.union(ghost_rows))
                        if (
                            int(ghost_rows[identifier]["status"])
                            if identifier in ghost_rows
                            else donor_status[identifier]
                        )
                        != int(OversetCellStatus.HOLE)
                    ],
                    dtype=np.int64,
                )
                if union is None or live_donors.size == 0:
                    covered = False
                    coverage_status = "no donor carrier"
                else:
                    overlap_routes, coverage_status, uncovered = _periodic_image_overlap(
                        _cell_submesh(union, live_donors),
                        _fringe_submesh(receptor_view, fringe_ids),
                        _periodic_lattice(donor_view),
                    )
                    covered = coverage_status == "covered"
            else:
                covered = True
            if not covered:
                requests = {
                    **frontier_requests(overlap_routes, uncovered),
                    **pending,
                }
        except Exception as error:
            local_error = f"{type(error).__name__}: {error}"
            covered = False
        _owner_group_fail_with_reasons(
            local_error, execution_group, f"{axis_name}_coverage_{expansion_round}"
        )
        pending = {}
        flags = np.asarray(
            _owner_group_allgather(
                np.asarray((int(covered), len(requests)), dtype=np.int64),
                execution_group,
                f"{axis_name}_round_{expansion_round}",
            ),
            dtype=np.int64,
        ).reshape((len(members), 2))
        if np.all(flags[:, 0] == 1):
            break
        if not np.any(flags[:, 1]):
            _owner_group_fail_with_reasons(
                "Owner-local FV overlap cannot cover its fringe receptors: no owner "
                "holds a further donor cell "
                f"(round {expansion_round}, {fringe_ids.size} fringe cells, "
                f"{len(donor_resident)} resident and {len(ghost_rows)} remote donor "
                f"cells, {len(winners)} native winners, coverage {coverage_status})."
                if not covered
                else None,
                execution_group,
                f"{axis_name}_exhausted_{expansion_round}",
            )
            raise RuntimeError("Owner-local FV donor halo expansion stalled.")
        request_ids = np.asarray(sorted(requests), dtype=np.int64)
        unknown = np.asarray(
            [
                identifier
                for identifier in request_ids.tolist()
                if requests[identifier][0] is None
            ],
            dtype=np.int64,
        )
        resolved = donor_view.resolve_owner_ranks(
            unknown,
            execution_group,
            directory_kind="mesh-cells",
            axis_name=f"{axis_name}_owners_{expansion_round}",
        )
        owner_by_id = {
            int(identifier): int(owner)
            for identifier, owner in zip(
                unknown.tolist(), np.asarray(resolved).tolist(), strict=True
            )
        }
        outgoing: list[list[dict[str, Any]]] = [[] for _ in members]
        for identifier in request_ids.tolist():
            known_owner, point = requests[identifier]
            owner = owner_by_id[identifier] if known_owner is None else known_owner
            if owner == rank:
                raise ValueError("A non-resident donor cell is owned by its requester.")
            outgoing[owner].append(
                {"cell": identifier, "point": None if point is None else _bits(point)}
            )
        incoming = _owner_exchange_message_banks(
            tuple(_encode_rows(rows) for rows in outgoing),
            rank,
            execution_group,
            f"{axis_name}_requests_{expansion_round}",
        )
        local_error = None
        replies_by_requester: list[list[dict[str, Any]]] = [[] for _ in members]
        try:
            for requester, bank in incoming.items():
                replies_by_requester[requester] = _serve_requests(
                    donor_view,
                    donor_status,
                    donor_corners,
                    _decode_rows(bank),
                    rank,
                )
        except Exception as error:
            local_error = f"{type(error).__name__}: {error}"
        _owner_group_fail_with_reasons(
            local_error, execution_group, f"{axis_name}_serve_{expansion_round}"
        )
        returned = _owner_exchange_message_banks(
            tuple(_encode_rows(rows) for rows in replies_by_requester),
            rank,
            execution_group,
            f"{axis_name}_replies_{expansion_round}",
        )
        for bank in returned.values():
            for reply in _decode_rows(bank):
                identifier = int(reply["cell"])
                if identifier in donor_resident:
                    probed[identifier] = list(reply["neighbors"])
                    continue
                previous = ghost_rows.get(identifier)
                if (
                    previous is not None
                    and previous["translation"] != reply["translation"]
                ):
                    raise ValueError(
                        "One donor cell is required under two lattice images."
                    )
                ghost_rows[identifier] = reply
        expansion_round += 1

    ghosts = _ghost_cells(donor_name, ghost_rows)
    composite = OwnerLocalCompositeCellGeometry(
        (donor_view, receptor_view), remote_ghosts=ghosts
    )
    geometry = UnstructuredFiniteVolumePlan.from_owner_local_geometry(
        composite, field_name=field_name, component_names=component_names
    ).prepare()
    row_by_id = {
        int(identifier): row
        for row, identifier in enumerate(
            np.asarray(geometry.cell_global_ids, dtype=np.int64).tolist()
        )
    }
    ghost_status = {
        identifier: int(reply["status"]) for identifier, reply in ghost_rows.items()
    }
    donor_ids = set(donor_resident).union(ghost_rows)
    receptor_ids = set(receptor_cells.tolist())
    count = geometry.cell_count
    donor_fringe = np.zeros((count,), dtype=np.bool_)
    donor_part = np.zeros((count,), dtype=np.bool_)
    receptor_hole = np.zeros((count,), dtype=np.bool_)
    receptor_fringe = np.zeros((count,), dtype=np.bool_)
    for identifier, row in row_by_id.items():
        if identifier in donor_ids:
            status = ghost_status.get(identifier, donor_status.get(identifier))
            if status is None:
                raise ValueError("Native blanking omits a composite donor cell.")
            donor_part[row] = True
            receptor_hole[row] = status == int(OversetCellStatus.HOLE)
            donor_fringe[row] = status == int(OversetCellStatus.FRINGE)
        elif identifier in receptor_ids:
            status = receptor_status[identifier]
            receptor_hole[row] = status == int(OversetCellStatus.HOLE)
            receptor_fringe[row] = status == int(OversetCellStatus.FRINGE)
    route_cells: list[int] = []
    offsets = [0]
    donors: list[int] = []
    measures: list[float] = []
    route_receptor_cells: list[int] = []
    for receptor_id in sorted(overlap_routes):
        receptor_row = row_by_id[receptor_id]
        route_cells.append(receptor_row)
        for donor_id, volume in overlap_routes[receptor_id]:
            donors.append(row_by_id[donor_id])
            measures.append(volume)
            route_receptor_cells.append(receptor_row)
        offsets.append(len(donors))
    face_ids = face_points = face_normals = face_measures = face_cells = None
    route_faces = None
    if route_cells:
        # One distinct certified face per routed receptor cell; every overlap
        # route of that cell supplies its weighted donor trace to that face.
        face_cells = np.asarray(route_cells, dtype=np.int32)
        face_ids, face_points, face_normals, face_measures = _unique_receptor_faces(
            geometry, face_cells
        )
        face_of_cell = {int(cell): face for face, cell in enumerate(route_cells)}
        route_faces = np.asarray(
            [face_of_cell[int(cell)] for cell in route_receptor_cells], dtype=np.int32
        )
    overset = UnstructuredOversetPlan(
        geometry,
        geometry,
        np.asarray(route_cells, dtype=np.int32),
        np.asarray(offsets, dtype=np.int32),
        np.asarray(donors, dtype=np.int32),
        np.asarray(measures, dtype=np.float64),
        native_connectivity_id=connectivity.connectivity_id,
        # Donor and receptor share one compiled layout: every non-hole row is
        # active on both sides, and only donor-part rows are eligible donors.
        donor_active_mask=~receptor_hole,
        donor_hole_mask=receptor_hole,
        donor_fringe_mask=donor_fringe,
        donor_eligible_mask=donor_part & ~receptor_hole,
        receptor_active_mask=~receptor_hole,
        receptor_hole_mask=receptor_hole,
        receptor_fringe_mask=receptor_fringe,
        receptor_face_ids=face_ids,
        receptor_face_points=face_points,
        receptor_face_normals=face_normals,
        receptor_face_measures=face_measures,
        receptor_face_cells=face_cells,
        receptor_route_faces=route_faces,
    )
    return OwnerLocalFiniteVolumeOverset(composite, geometry, overset)


def _ghost_cells(
    part_name: str, rows: Mapping[int, Mapping[str, Any]], /
) -> OwnerLocalRemoteGhostCells | None:
    if not rows:
        return None
    ordered = sorted(rows)
    kinds = {str(rows[identifier]["kind"]) for identifier in ordered}
    if len(kinds) != 1:
        raise ValueError("Remote donor ghost cells must share one cell kind.")
    dimension = len(rows[ordered[0]]["translation"])
    corners = np.stack(
        [
            _from_bits(rows[identifier]["corners"]).reshape((-1, dimension))
            for identifier in ordered
        ]
    )
    return OwnerLocalRemoteGhostCells(
        part_name,
        kinds.pop(),
        np.asarray(ordered, dtype=np.int64),
        np.asarray([int(rows[identifier]["owner"]) for identifier in ordered]),
        np.asarray(
            [list(rows[identifier]["shift"]) for identifier in ordered], dtype=np.int64
        ).reshape((len(ordered), -1)),
        corners,
    )


__all__ = ["OwnerLocalFiniteVolumeOverset", "prepare_owner_local_finite_volume_overset"]
