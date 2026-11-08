from __future__ import annotations

import numpy as np
from jax.typing import ArrayLike

from ..._meshcore import (
    exact_orient3d,
    TetMeshPeriodicEdit,
    TetMeshPeriodicEntityOrbitMap,
    TetMeshPeriodicIdentityMap,
    TetMeshPeriodicMember,
    TetMeshPeriodicOrbit,
    TetMeshPeriodicQuotientCell,
    TetMeshPeriodicState,
)
from ...discretization._cell_mesh import CellMesh
from ...discretization._periodic_exact import exact_periodic_element, identification_id
from ...discretization._periodic_topology import _entity_vertex_rows, PeriodicMeshTopology
from .._periodic import periodic_entity_orbit_map, PeriodicEntityOrbitMap
from .._topology_edit import (
    CellTopologyEdit,
    entity_keys,
    EntityLineageKind,
    key_rows,
    TopologyEditBlock,
)


def _tet_mesh_corner_permutation_bank(
    orbit_map: PeriodicEntityOrbitMap, degree: int, /
) -> np.ndarray:
    """Convert ragged quotient rows only after enforcing the fixed TetMesh ABI."""
    if orbit_map.degree != degree:
        raise ValueError("Periodic TetMesh entity-map dimension is inconsistent.")
    arity = degree + 1
    offsets = np.asarray(orbit_map.corner_permutation_offsets, dtype=np.int64)
    values = np.asarray(orbit_map.corner_permutations, dtype=np.int32)
    count = np.asarray(orbit_map.lifted_entity_ids).size
    if (
        count == 0
        or offsets.shape != (count + 1,)
        or np.any(np.diff(offsets) != arity)
        or offsets[0] != 0
        or values.shape != (count * arity,)
    ):
        raise ValueError(
            "TetMesh periodic maps require one fixed simplex permutation row per lifted entity."
        )
    rows = values.reshape((count, arity))
    if not np.array_equal(
        np.sort(rows, axis=1),
        np.broadcast_to(np.arange(arity, dtype=np.int32), rows.shape),
    ):
        raise ValueError(
            "TetMesh periodic maps require one fixed simplex permutation row per lifted entity."
        )
    return np.ascontiguousarray(rows, dtype=np.int32)


def tet_mesh_periodic_identity_map(mesh: CellMesh, /) -> TetMeshPeriodicIdentityMap:
    """Project a validated periodic tetra mesh onto native TetMesh orbit maps."""
    _require_periodic_tetra_mesh(mesh)
    topology = mesh.periodic_topology
    assert topology is not None
    rank = int(np.asarray(topology.vertex_shifts).shape[1])
    entity_maps = []
    for degree in range(4):
        orbit_map = periodic_entity_orbit_map(mesh, degree)
        entity_maps.append(
            TetMeshPeriodicEntityOrbitMap(
                degree,
                degree + 1,
                rank,
                np.ascontiguousarray(orbit_map.lifted_entity_ids, dtype=np.uint64),
                np.ascontiguousarray(orbit_map.quotient_entity_ids, dtype=np.uint64),
                np.ascontiguousarray(orbit_map.lifted_to_quotient, dtype=np.int32),
                np.ascontiguousarray(orbit_map.orientation_signs, dtype=np.int32),
                np.ascontiguousarray(orbit_map.winding_shifts, dtype=np.int64),
                _tet_mesh_corner_permutation_bank(orbit_map, degree),
                np.ascontiguousarray(
                    orbit_map.representative_lifted_rows, dtype=np.int32
                ),
            )
        )
    return TetMeshPeriodicIdentityMap(
        mesh.mesh_id,
        mesh.topology_id,
        mesh.lifted_topology_id,
        topology.periodic_topology_id,
        mesh.entity_set(3).entity_set_id,
        mesh.entity_set(1).entity_set_id,
        tuple(entity_maps),
    )


def _quotient_member_rows(
    lifted_to_quotient: np.ndarray, quotient_count: int, /
) -> list[np.ndarray]:
    """Ascending lifted rows of each quotient row, one stable sort for all rows."""
    order = np.argsort(lifted_to_quotient, kind="stable")
    bounds = np.searchsorted(
        lifted_to_quotient[order], np.arange(quotient_count + 1, dtype=np.int64)
    )
    return [order[bounds[row] : bounds[row + 1]] for row in range(quotient_count)]


def tet_mesh_periodic_state(
    mesh: CellMesh, regions: ArrayLike | None, /
) -> TetMeshPeriodicState:
    """Build canonical periodic state and exact quotient-cell banks for TetMesh3D.

    ``regions`` follows native block order when supplied. An absent region bank
    is encoded as the TetMesh kernel's unlabeled region 0; it does not create a
    Python material attribute or zone.
    """
    identity_map = tet_mesh_periodic_identity_map(mesh)
    topology = mesh.periodic_topology
    assert topology is not None
    vertex_map = periodic_entity_orbit_map(mesh, 0)
    vertex_lifted_ids = np.asarray(vertex_map.lifted_entity_ids, dtype=np.int64)
    vertex_quotient_ids = np.asarray(vertex_map.quotient_entity_ids, dtype=np.int64)
    vertex_lifted_to_quotient = np.asarray(vertex_map.lifted_to_quotient, dtype=np.int64)
    vertex_windings = np.asarray(vertex_map.winding_shifts, dtype=np.int64)
    vertex_rep_rows = np.asarray(vertex_map.representative_lifted_rows, dtype=np.int64)
    vertex_member_rows = _quotient_member_rows(
        vertex_lifted_to_quotient, vertex_quotient_ids.size
    )
    # Few distinct windings occur; each exact deck isometry is evaluated once.
    isometries: dict[tuple[int, ...], np.ndarray] = {}

    def isometry(winding: np.ndarray, /) -> np.ndarray:
        key = tuple(int(value) for value in winding)
        matrix = isometries.get(key)
        if matrix is None:
            matrix = _representative_to_member(topology, winding)
            isometries[key] = matrix
        return matrix.copy()

    orbit_members = []
    for quotient_row, quotient_id in enumerate(vertex_quotient_ids):
        rows = vertex_member_rows[quotient_row]
        if rows.size == 0:
            raise ValueError("Every periodic vertex quotient ID must have lifted members.")
        representative_row = int(vertex_rep_rows[quotient_row])
        representative_id = int(vertex_lifted_ids[representative_row])
        members = tuple(
            TetMeshPeriodicMember(
                int(vertex_lifted_ids[row]),
                _winding_3d(vertex_windings[row]),
                isometry(vertex_windings[row]),
            )
            for row in rows
        )
        orbit_members.append(
            TetMeshPeriodicOrbit(int(quotient_id), representative_id, members)
        )

    block_cell_ids = np.concatenate(
        tuple(np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks)
    )
    region_by_cell_id = _region_by_cell_id(mesh, block_cell_ids, regions)
    cell_map = periodic_entity_orbit_map(mesh, 3)
    cell_lifted_ids = np.asarray(cell_map.lifted_entity_ids, dtype=np.int64)
    cell_quotient_ids = np.asarray(cell_map.quotient_entity_ids, dtype=np.int64)
    cell_lifted_to_quotient = np.asarray(cell_map.lifted_to_quotient, dtype=np.int64)
    cell_rep_rows = np.asarray(cell_map.representative_lifted_rows, dtype=np.int64)
    cell_rows_by_id = {
        int(identifier): row
        for row, identifier in enumerate(
            np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64)
        )
    }
    # Quotient-cell corners name each vertex orbit by its representative lifted
    # vertex, the key of the native orbit; quotient IDs are separate identities.
    vertex_representative_by_lifted_id = {
        int(identifier): int(
            vertex_lifted_ids[vertex_rep_rows[vertex_lifted_to_quotient[row]]]
        )
        for row, identifier in enumerate(vertex_lifted_ids)
    }
    vertex_winding_by_lifted_id = {
        int(identifier): vertex_windings[row]
        for row, identifier in enumerate(vertex_lifted_ids)
    }
    top_entity_vertices = _entity_vertex_rows(mesh)[3]
    vertex_global_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    staged: list[tuple[int, np.ndarray, int, np.ndarray]] = []
    cell_member_rows = _quotient_member_rows(
        cell_lifted_to_quotient, cell_quotient_ids.size
    )
    cell_corner_offsets = np.asarray(cell_map.corner_permutation_offsets, dtype=np.int64)
    cell_corners = np.asarray(cell_map.corner_permutations, dtype=np.int32)
    cell_orientations = np.asarray(cell_map.orientation_signs)
    for quotient_row, quotient_id in enumerate(cell_quotient_ids):
        members = cell_member_rows[quotient_row]
        if members.size == 0:
            raise ValueError("Every periodic cell quotient ID must have lifted members.")
        representative_row = int(cell_rep_rows[quotient_row])
        representative_id = int(cell_lifted_ids[representative_row])
        cell_row = cell_rows_by_id.get(representative_id)
        if cell_row is None:
            raise ValueError("Periodic representative cell ID is absent from the lifted mesh.")
        lifted_vertex_rows = np.asarray(top_entity_vertices[cell_row], dtype=np.int64)
        corner_permutation = cell_corners[
            int(cell_corner_offsets[representative_row]) : int(
                cell_corner_offsets[representative_row + 1]
            )
        ].astype(np.int64)
        orientation = int(cell_orientations[representative_row])
        if orientation not in (-1, 1):
            raise ValueError("Periodic quotient cell has an invalid corner orientation.")
        if orientation < 0:
            corner_permutation[[0, 1]] = corner_permutation[[1, 0]]
        try:
            region_values = {
                region_by_cell_id[int(cell_lifted_ids[row])] for row in members
            }
        except KeyError as error:
            raise ValueError("Periodic cell references an absent source entity identity.") from error
        if len(region_values) != 1:
            raise ValueError("Periodic material regions must agree over each quotient cell orbit.")
        lifted_cell_ids = np.sort(cell_lifted_ids[members]).astype(np.uint64, copy=False)
        staged.append(
            (
                int(quotient_id),
                lifted_vertex_rows[corner_permutation],
                next(iter(region_values)),
                lifted_cell_ids,
            )
        )
    # The orbit map orients corners relative to the connectivity's entity row,
    # which is the sorted simplex for simplicial connectivity, not the oriented
    # lifted cell. The native kernel requires positive quotient cells, so the
    # exact orientation of the representative lift decides the corner order.
    ordered_rows = np.asarray([rows for _, rows, _, _ in staged], dtype=np.int64).reshape(
        (-1, 4)
    )
    points = np.asarray(mesh.coordinates, dtype=np.float64)[ordered_rows.reshape(-1)]
    corners = points.reshape((-1, 4, 3))
    signs = np.asarray(
        exact_orient3d(corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]),
        dtype=np.int64,
    ).reshape(-1)
    if np.any(signs == 0):
        raise ValueError("Periodic quotient cells must have nonzero exact orientation.")
    negative = signs < 0
    ordered_rows[negative, 0], ordered_rows[negative, 1] = (
        ordered_rows[negative, 1].copy(),
        ordered_rows[negative, 0].copy(),
    )
    quotient_cells = []
    for (quotient_id, _, region, lifted_cell_ids), rows in zip(
        staged, ordered_rows, strict=True
    ):
        lifted_vertex_ids = vertex_global_ids[rows]
        try:
            representative_vertices = np.asarray(
                [
                    vertex_representative_by_lifted_id[int(identifier)]
                    for identifier in lifted_vertex_ids
                ],
                dtype=np.uint64,
            )
            winding = np.stack(
                [
                    _winding_3d(vertex_winding_by_lifted_id[int(identifier)])
                    for identifier in lifted_vertex_ids
                ]
            )
        except KeyError as error:
            raise ValueError("Periodic cell references an absent source entity identity.") from error
        quotient_cells.append(
            TetMeshPeriodicQuotientCell(
                quotient_id,
                quotient_id,
                representative_vertices,
                winding,
                region,
                lifted_cell_ids,
            )
        )
    return TetMeshPeriodicState(
        identity_map,
        tuple(orbit_members),
        tuple(quotient_cells),
    )


def _require_periodic_tetra_mesh(mesh: CellMesh, /) -> None:
    if not isinstance(mesh, CellMesh):
        raise TypeError("mesh must be a CellMesh.")
    if mesh.topological_dimension != 3 or mesh.ambient_dimension != 3:
        raise ValueError("TetMesh periodic maps require a three-dimensional CellMesh.")
    if any(block.cell_kind not in ("tetrahedron", "simplex:3") for block in mesh.blocks):
        raise ValueError("TetMesh periodic maps require tetrahedron or 3-simplex blocks.")
    topology = mesh.periodic_topology
    if topology is None:
        raise ValueError("TetMesh periodic maps require an attached PeriodicMeshTopology.")
    topology.require_lifted(mesh.topology, mesh.lifted_topology_id, mesh.blocks)
    topology.require_coordinates(mesh.coordinates)


def _winding_3d(winding: ArrayLike, /) -> np.ndarray:
    values = np.asarray(winding, dtype=np.int64)
    if values.ndim != 1 or values.size > 3:
        raise ValueError("Periodic winding must be a vector of at most three generators.")
    result = np.zeros((3,), dtype=np.int64)
    result[: values.size] = values
    return result


def _representative_to_member(topology: PeriodicMeshTopology, winding: ArrayLike, /) -> np.ndarray:
    matrix = np.asarray(
        exact_periodic_element(topology.cell, np.asarray(winding, dtype=np.int64)),
        dtype=np.float64,
    )
    if matrix.shape != (4, 4) or np.any(~np.isfinite(matrix)):
        raise ValueError("Periodic vertex isometry must be a finite homogeneous 4x4 matrix.")
    return np.ascontiguousarray(matrix[:3, :4], dtype=np.float64)


def _region_by_cell_id(
    mesh: CellMesh,
    block_cell_ids: np.ndarray,
    cell_regions: ArrayLike | None,
    /,
) -> dict[int, int]:
    if cell_regions is None:
        values = np.zeros((block_cell_ids.size,), dtype=np.int32)
    else:
        values = np.asarray(cell_regions)
        if (
            values.ndim != 1
            or values.dtype.kind not in "iu"
            or values.shape != block_cell_ids.shape
            or (values.size and np.any(values < 0))
            or (values.size and np.any(values > np.iinfo(np.int32).max))
        ):
            raise ValueError("Periodic TetMesh region IDs must be one nonnegative int32 per lifted cell.")
        values = values.astype(np.int32, copy=False)
    cell_ids = np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64)
    if (
        np.unique(block_cell_ids).size != block_cell_ids.size
        or not np.array_equal(np.sort(block_cell_ids), np.sort(cell_ids))
    ):
        raise ValueError("Periodic TetMesh block cell IDs must cover every lifted cell exactly once.")
    return {
        int(identifier): int(region)
        for identifier, region in zip(block_cell_ids, values, strict=True)
    }


def tet_mesh_periodic_edit(
    source: CellMesh,
    target: CellMesh,
    edit: CellTopologyEdit,
    operation_members: tuple[TetMeshPeriodicMember, ...] | None = None,
    /,
    *,
    source_regions: ArrayLike | None = None,
    target_regions: ArrayLike | None = None,
    source_state: TetMeshPeriodicState | None = None,
    target_state: TetMeshPeriodicState | None = None,
) -> TetMeshPeriodicEdit:
    """Build the periodic delta for one staged tetrahedral topology edit.

    ``target`` must be the bound candidate returned by
    :func:`stage_periodic_topology_edit`. Supplied ``operation_members`` are
    checked against exact source winding and target ancestry; omission derives
    that complete bank from the edit relations. A caller that already holds
    ``tet_mesh_periodic_state(source, source_regions)`` or
    ``tet_mesh_periodic_state(target, target_regions)`` passes it as
    ``source_state``/``target_state`` instead of rebuilding it; each must be
    bound to its endpoint's mesh and periodic topology.
    """
    _require_periodic_tetra_mesh(source)
    _require_periodic_tetra_mesh(target)
    if not isinstance(edit, CellTopologyEdit):
        raise TypeError("edit must be a CellTopologyEdit.")
    source_topology = source.periodic_topology
    target_topology = target.periodic_topology
    if source_topology is None or target_topology is None:
        raise ValueError("Periodic edit endpoints require quotient descriptors.")
    if identification_id(source_topology.cell) != identification_id(target_topology.cell):
        raise ValueError("Periodic edit endpoints must retain the same exact identification.")
    _require_edit_target(source, target, edit)

    for mesh, topology, state in (
        (source, source_topology, source_state),
        (target, target_topology, target_state),
    ):
        if state is not None and (
            state.identity_map.source_mesh_id != mesh.mesh_id
            or state.identity_map.periodic_topology_id != topology.periodic_topology_id
        ):
            raise ValueError("A supplied periodic state belongs to another mesh revision.")
    if source_state is None:
        source_state = tet_mesh_periodic_state(source, source_regions)
    target_regions = _target_regions_after_edit(
        source, target, edit, source_regions, target_regions
    )
    if target_state is None:
        target_state = tet_mesh_periodic_state(target, target_regions)
    expected_members = _periodic_operation_member_records(
        source, target, edit, source_state, target_state
    )
    if operation_members is not None:
        if not isinstance(operation_members, tuple) or not all(
            isinstance(member, TetMeshPeriodicMember) for member in operation_members
        ):
            raise TypeError("operation_members must be a tuple of TetMeshPeriodicMember values.")
        actual_by_id = {int(member.vertex_id): member for member in operation_members}
        expected_by_id = {int(member.vertex_id): member for member in expected_members}
        if len(actual_by_id) != len(operation_members) or set(actual_by_id) != set(expected_by_id):
            raise ValueError("Periodic operation members must exactly cover affected orbit members.")
        for identifier, expected in expected_by_id.items():
            actual = actual_by_id[identifier]
            if (
                not np.array_equal(np.asarray(actual.winding), np.asarray(expected.winding))
                or not np.array_equal(
                    np.asarray(actual.representative_to_member),
                    np.asarray(expected.representative_to_member),
                )
            ):
                raise ValueError("Periodic operation member winding/isometry disagrees with source ancestry.")

    source_cell_map = periodic_entity_orbit_map(source, 3)
    target_cell_map = periodic_entity_orbit_map(target, 3)
    source_cell_ids = np.asarray(source_cell_map.lifted_entity_ids, dtype=np.int64)
    target_cell_ids = np.asarray(target_cell_map.lifted_entity_ids, dtype=np.int64)
    source_cell_qids = np.asarray(source_cell_map.quotient_entity_ids, dtype=np.int64)[
        np.asarray(source_cell_map.lifted_to_quotient, dtype=np.int64)
    ]
    target_cell_qids = np.asarray(target_cell_map.quotient_entity_ids, dtype=np.int64)[
        np.asarray(target_cell_map.lifted_to_quotient, dtype=np.int64)
    ]
    source_q_by_cell_id = {
        int(identifier): int(quotient_id)
        for identifier, quotient_id in zip(source_cell_ids, source_cell_qids, strict=True)
    }
    target_q_by_cell_id = {
        int(identifier): int(quotient_id)
        for identifier, quotient_id in zip(target_cell_ids, target_cell_qids, strict=True)
    }
    removed_lifted_cell_ids = set(map(int, source_cell_ids)) - set(map(int, target_cell_ids))
    source_cells_by_qid: dict[int, set[int]] = {}
    for identifier, quotient_id in source_q_by_cell_id.items():
        source_cells_by_qid.setdefault(quotient_id, set()).add(identifier)
    removed_qcell_ids: set[int] = set()
    for quotient_id, member_ids in source_cells_by_qid.items():
        removed = member_ids & removed_lifted_cell_ids
        if removed and removed != member_ids:
            raise ValueError("Periodic topology edits must remove complete lifted quotient-cell orbits.")
        if removed:
            removed_qcell_ids.add(quotient_id)

    relation = edit.relations[3]
    source_keys = np.asarray(relation.source_keys, dtype=np.int64)
    target_keys = np.asarray(relation.target_keys, dtype=np.int64)
    kinds = np.asarray(relation.kinds, dtype=np.int32)
    if (
        source_keys.ndim != 2
        or target_keys.ndim != 2
        or source_keys.shape[1] != 1
        or target_keys.shape != source_keys.shape
        or kinds.shape != (source_keys.shape[0],)
    ):
        raise ValueError("Tetra periodic cell lineage must use one source/target cell ID per row.")
    topology_changed = (kinds != int(EntityLineageKind.PRESERVED)) & (
        kinds != int(EntityLineageKind.RELOCATED)
    )
    changed_source_qcells = {
        source_q_by_cell_id[int(identifier)]
        for identifier in source_keys[topology_changed, 0]
        if int(identifier) in source_q_by_cell_id
    }
    changed_target_qcells = {
        target_q_by_cell_id[int(identifier)]
        for identifier in target_keys[topology_changed, 0]
        if int(identifier) in target_q_by_cell_id
    }
    source_qcell_ids = set(map(int, source_cell_map.quotient_entity_ids))
    target_qcell_ids = set(map(int, target_cell_map.quotient_entity_ids))
    for quotient_id in changed_source_qcells:
        member_ids = source_cells_by_qid[quotient_id]
        if not member_ids <= removed_lifted_cell_ids:
            raise ValueError("Changed periodic source cell orbits must be removed atomically.")
        removed_qcell_ids.add(quotient_id)
    removed_qcell_ids.update(source_qcell_ids - target_qcell_ids)
    for quotient_id in source_qcell_ids - target_qcell_ids:
        if not source_cells_by_qid[quotient_id] <= removed_lifted_cell_ids:
            raise ValueError("A removed periodic quotient-cell ID must lose every lifted member.")
    replacement_qcell_ids = (
        (target_qcell_ids - source_qcell_ids)
        | changed_target_qcells
        | (removed_qcell_ids & target_qcell_ids)
    )
    replacement_quotient_cells = tuple(
        cell
        for cell in target_state.quotient_cells
        if int(cell.cell_id) in replacement_qcell_ids
    )
    if replacement_qcell_ids != {
        int(cell.cell_id) for cell in replacement_quotient_cells
    }:
        raise ValueError("Periodic edit replacement cells must cover every affected target quotient cell.")

    source_orbit_members = {
        int(orbit.orbit_id): {int(member.vertex_id) for member in orbit.members}
        for orbit in source_state.orbits
    }
    target_orbit_members = {
        int(orbit.orbit_id): {int(member.vertex_id) for member in orbit.members}
        for orbit in target_state.orbits
    }
    removed_orbit_ids = {
        orbit_id
        for orbit_id, member_ids in source_orbit_members.items()
        if target_orbit_members.get(orbit_id) != member_ids
    }
    replacement_orbit_ids = {
        orbit_id
        for orbit_id, member_ids in target_orbit_members.items()
        if source_orbit_members.get(orbit_id) != member_ids
    }
    replacement_periodic_orbits = tuple(
        orbit
        for orbit in target_state.orbits
        if int(orbit.orbit_id) in replacement_orbit_ids
    )
    return TetMeshPeriodicEdit(
        expected_members,
        np.asarray(sorted(removed_orbit_ids), dtype=np.uint64),
        replacement_periodic_orbits,
        np.asarray(sorted(removed_qcell_ids), dtype=np.uint64),
        replacement_quotient_cells,
        target_state.identity_map,
    )


def tet_mesh_periodic_operation_members(
    source: CellMesh, target: CellMesh, edit: CellTopologyEdit, /
) -> tuple[TetMeshPeriodicMember, ...]:
    """Return the exact complete native member bank affected by one staged edit."""
    _require_periodic_tetra_mesh(source)
    _require_periodic_tetra_mesh(target)
    if not isinstance(edit, CellTopologyEdit):
        raise TypeError("edit must be a CellTopologyEdit.")
    _require_edit_target(source, target, edit)
    return _periodic_operation_member_records(
        source,
        target,
        edit,
        tet_mesh_periodic_state(source, None),
        tet_mesh_periodic_state(target, None),
    )


def _periodic_operation_member_records(
    source: CellMesh,
    target: CellMesh,
    edit: CellTopologyEdit,
    source_state: TetMeshPeriodicState,
    target_state: TetMeshPeriodicState,
    /,
) -> tuple[TetMeshPeriodicMember, ...]:
    source_map = source_state.identity_map.entity_maps[0]
    source_lifted_ids = np.asarray(source_map.lifted_entity_ids, dtype=np.int64)
    source_qids = np.asarray(source_map.quotient_entity_ids, dtype=np.int64)[
        np.asarray(source_map.lifted_to_quotient, dtype=np.int64)
    ]
    q_by_lifted_id = {
        int(identifier): int(quotient_id)
        for identifier, quotient_id in zip(
            source_lifted_ids, source_qids, strict=True
        )
    }
    source_orbits = {
        int(orbit.orbit_id): tuple(orbit.members) for orbit in source_state.orbits
    }
    target_orbits = {
        int(orbit.orbit_id): tuple(orbit.members) for orbit in target_state.orbits
    }
    touched = {
        q_by_lifted_id[int(identifier)]
        for identifier in _source_vertices_touched_by_edit(source, target, edit)
        if int(identifier) in q_by_lifted_id
    }
    source_members = {
        int(member.vertex_id): member
        for orbit in source_state.orbits
        for member in orbit.members
    }
    target_members = {
        int(member.vertex_id): member
        for orbit in target_state.orbits
        for member in orbit.members
    }
    target_ids = set(target_members)
    source_ids = set(source_members)
    changed_orbits = {
        orbit_id
        for orbit_id, members in source_orbits.items()
        if {int(member.vertex_id) for member in members}
        != {
            int(member.vertex_id)
            for member in target_orbits.get(orbit_id, ())
        }
    }
    touched.update(changed_orbits)
    expected_ids = {
        int(member.vertex_id)
        for orbit_id in touched
        for member in source_orbits[orbit_id]
    }
    expected_ids.update(target_ids - source_ids)
    records = []
    for identifier in sorted(expected_ids):
        member = source_members.get(identifier, target_members.get(identifier))
        if member is None:
            raise ValueError("Periodic edit member ancestry is absent from both endpoint meshes.")
        records.append(member)
    return tuple(records)


def _require_edit_target(
    source: CellMesh, target: CellMesh, edit: CellTopologyEdit, /
) -> None:
    if len(edit.relations) != source.topological_dimension + 1:
        raise ValueError("Periodic tetra edits require one relation bank per entity dimension.")
    if (
        not np.array_equal(np.asarray(edit.coordinates), np.asarray(target.coordinates))
        or not np.array_equal(
            np.asarray(edit.vertex_global_ids, dtype=np.int64),
            np.asarray(target.vertex_global_ids, dtype=np.int64),
        )
    ):
        raise ValueError("Staged periodic target must retain the edit's exact vertex bank.")
    edit_blocks = {
        block.name: block
        for block in edit.blocks
        if isinstance(block, TopologyEditBlock)
    }
    if len(edit_blocks) != len(edit.blocks):
        raise ValueError("Periodic tetra edits require fixed-family simplex blocks.")
    target_blocks = {block.name: block for block in target.blocks}
    if edit_blocks.keys() != target_blocks.keys():
        raise ValueError("Staged periodic target must retain every edit block.")
    for name, edit_block in edit_blocks.items():
        target_block = target_blocks[name]
        if (
            edit_block.cell_kind != target_block.cell_kind
            or not np.array_equal(
                np.asarray(edit_block.cell_ids, dtype=np.int64),
                np.asarray(target_block.global_ids, dtype=np.int64),
            )
            or not np.array_equal(
                np.asarray(edit_block.cells, dtype=np.int32),
                np.asarray(target_block.vertices, dtype=np.int32),
            )
        ):
            raise ValueError("Staged periodic target must retain exact edit cell IDs and connectivity.")


def _source_vertices_touched_by_edit(
    source: CellMesh, target: CellMesh, edit: CellTopologyEdit, /
) -> set[int]:
    if len(edit.relations) != source.topological_dimension + 1:
        raise ValueError("Periodic topology edit requires one relation bank per entity dimension.")
    source_entity_vertices = _entity_vertex_rows(source)
    source_vertex_ids = np.asarray(source.vertex_global_ids, dtype=np.int64)
    affected: set[int] = set()
    for degree, relation in enumerate(edit.relations):
        source_keys = np.asarray(relation.source_keys, dtype=np.int64)
        kinds = np.asarray(relation.kinds, dtype=np.int32)
        if (
            source_keys.ndim != 2
            or kinds.shape != (source_keys.shape[0],)
            or int(relation.dimension) != degree
        ):
            raise ValueError("Periodic topology edit relation bank is malformed.")
        changed_rows = np.flatnonzero(
            kinds != int(EntityLineageKind.PRESERVED)
        )
        if changed_rows.size == 0:
            continue
        selected_keys = source_keys[changed_rows]
        if degree == 0:
            affected.update(int(identifier) for identifier in selected_keys[:, 0])
            continue
        rows = key_rows(entity_keys(source, degree), selected_keys)
        if np.any(rows < 0):
            raise ValueError("Periodic edit lineage references an absent source entity.")
        for entity_row in rows.tolist():
            vertex_rows = np.asarray(
                source_entity_vertices[degree][entity_row], dtype=np.int64
            )
            affected.update(int(identifier) for identifier in source_vertex_ids[vertex_rows])
    target_cell_ids = set(
        map(int, np.asarray(target.entity_set(source.topological_dimension).entity_ids))
    )
    source_cell_ids = np.asarray(
        source.entity_set(source.topological_dimension).entity_ids, dtype=np.int64
    )
    source_cells_by_id = {
        int(identifier): row for row, identifier in enumerate(source_cell_ids)
    }
    for identifier in set(source_cells_by_id) - target_cell_ids:
        rows = np.asarray(
            source_entity_vertices[source.topological_dimension][
                source_cells_by_id[identifier]
            ],
            dtype=np.int64,
        )
        affected.update(int(value) for value in source_vertex_ids[rows])
    target_vertex_ids = set(
        map(int, np.asarray(target.vertex_global_ids, dtype=np.int64))
    )
    affected.update(set(map(int, source_vertex_ids)) - target_vertex_ids)
    return affected


def _target_regions_after_edit(
    source: CellMesh,
    target: CellMesh,
    edit: CellTopologyEdit,
    source_regions: ArrayLike | None,
    target_regions: ArrayLike | None,
    /,
) -> np.ndarray | None:
    target_block_ids = np.concatenate(
        tuple(np.asarray(block.global_ids, dtype=np.int64) for block in target.blocks)
    )
    if target_regions is not None:
        _region_by_cell_id(target, target_block_ids, target_regions)
        return np.asarray(target_regions, dtype=np.int32)
    if source_regions is None:
        return None
    source_block_ids = np.concatenate(
        tuple(np.asarray(block.global_ids, dtype=np.int64) for block in source.blocks)
    )
    source_by_id = _region_by_cell_id(source, source_block_ids, source_regions)
    inherited: dict[int, set[int]] = {}
    for identifier in target_block_ids:
        value = source_by_id.get(int(identifier))
        if value is not None:
            inherited.setdefault(int(identifier), set()).add(value)
    relation = edit.relations[source.topological_dimension]
    source_keys = np.asarray(relation.source_keys, dtype=np.int64).reshape(-1)
    target_keys = np.asarray(relation.target_keys, dtype=np.int64).reshape(-1)
    for source_id, target_id in zip(source_keys, target_keys, strict=True):
        if int(source_id) not in source_by_id:
            raise ValueError("Periodic material ancestry references an absent source cell.")
        inherited.setdefault(int(target_id), set()).add(source_by_id[int(source_id)])
    result = np.empty((target_block_ids.size,), dtype=np.int32)
    for row, identifier in enumerate(target_block_ids):
        values = inherited.get(int(identifier), set())
        if len(values) != 1:
            raise ValueError("Periodic material edits require one exact parent region per target cell.")
        result[row] = next(iter(values))
    return result
__all__ = [
    "tet_mesh_periodic_edit",
    "tet_mesh_periodic_identity_map",
    "tet_mesh_periodic_operation_members",
    "tet_mesh_periodic_state",
]
