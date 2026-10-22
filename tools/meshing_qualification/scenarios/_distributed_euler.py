# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Conservative Euler histories on native affine-planar bisection forests.

Only owned cells contribute physical content. Refinement prolongs cell averages;
coarsening restricts actual accepted child averages with their native dyadic
volume fractions. Publication geometry independently checks every overlap column.
Ghost values are supplied by the canonical stable-ID owner query route.
"""

from collections.abc import Sequence
from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils

from phydrax import units
from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
from phydrax.discretization import AdaptiveSimplexState
from phydrax.discretization._distributed_field import resolve_owner_local_field_values
from phydrax.equations import EulerSystem
from phydrax.meshing import CellMeshingResult, MeshAdaptationResult, MeshAdaptationRoute
from phydrax.meshing._initial_certification import (
    InitialCollectiveMeshEvidence,
    InitialPlanarPremise,
)
from phydrax.meshing._result import require_original_meshing_source
from phydrax.units import UnitDefinition


ENTRY = "euler-conserved"
RAW_ENTRY = "allocated-euler-conserved"
COMPONENTS = ("density", "momentum_x", "momentum_y", "total_energy")
INVENTORY_TOLERANCE = 1.0e-12


@cache
def state_units() -> tuple[UnitDefinition, ...]:
    """SI conserved-density dimensions, per unit out-of-plane depth in 2D."""
    density = units.derived_unit("kg/m^3", ((units.KILOGRAM, 1), (units.METER, -3)))
    momentum = units.derived_unit(
        "kg/(m^2*s)", ((units.KILOGRAM, 1), (units.METER, -2), (units.SECOND, -1))
    )
    return density, momentum, momentum, units.PASCAL


def _require(passed: bool, message: str) -> None:
    rejected = multihost_utils.process_allgather(np.int32(not passed), tiled=False)
    if np.any(np.asarray(rejected)):
        raise ValueError(f"distributed_euler.{message}")


def _sum(values: np.ndarray) -> np.ndarray:
    gathered = np.asarray(multihost_utils.process_allgather(values, tiled=False))
    return gathered.reshape((-1, *values.shape)).sum(axis=0)


def _cell_geometry(carrier: CellMeshingResult) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stable cell IDs, owned row mask and physical areas in entity-bank order."""
    mesh, storage = carrier.mesh, carrier.mesh.storage
    _require(storage is not None, "storage must retain owner-local cell authority")
    assert storage is not None
    _require(
        mesh.topological_dimension == 2
        and mesh.ambient_dimension == 2
        and all(block.cell_kind == "triangle" for block in mesh.blocks),
        "native planar history must use its published two-dimensional triangles",
    )
    ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    block_ids = np.concatenate([np.asarray(block.global_ids) for block in mesh.blocks])
    blocks = np.concatenate([np.asarray(block.vertices) for block in mesh.blocks])
    order = np.argsort(block_ids, kind="stable")
    cells = blocks[order[np.searchsorted(block_ids[order], ids)]]
    triangles = np.asarray(mesh.coordinates)[cells]
    first, second = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    areas = np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]) / 2
    owned_ids = np.asarray(storage.entity_global_ids[2])[np.asarray(storage.entity_owned[2])]
    owned = np.isin(ids, owned_ids)
    _require(bool(np.all(np.isfinite(areas) & (areas > 0))), "cell areas must be positive and finite")
    return ids, owned, areas


def inventory(
    carrier: CellMeshingResult,
    values: jax.Array | np.ndarray,
    *,
    system: EulerSystem,
) -> np.ndarray:
    """Exactly-once owner reduction of all four accepted conserved components."""
    ids, owned, areas = _cell_geometry(carrier)
    data = np.asarray(values)
    _require(data.shape == (ids.size, 4), "state must have four components per cell")
    _require(
        bool(np.all(np.isfinite(data[owned])))
        and bool(np.all(np.asarray(system.admissible(jnp.asarray(data[owned]))))),
        "accepted owner state must be finite and physically admissible",
    )
    return _sum(np.sum(areas[owned, None] * data[owned], axis=0))


def moments(carrier: CellMeshingResult, values: jax.Array | np.ndarray) -> np.ndarray:
    """Placement-independent first and second physical moments of accepted state."""
    ids, owned, areas = _cell_geometry(carrier)
    data = np.asarray(values)
    _require(data.shape == (ids.size, 4), "state must have four components per cell")
    weighted = areas[owned, None] * data[owned]
    return _sum(
        np.stack((np.sum(weighted, axis=0), np.sum(weighted * data[owned], axis=0)))
    )


def synchronize(
    carrier: CellMeshingResult,
    values: jax.Array | np.ndarray,
    devices: Sequence[jax.Device],
    *,
    message_capacity: int,
) -> jax.Array:
    """Fill ghost cells from the actual authoritative accepted owner values."""
    mesh, storage = carrier.mesh, carrier.mesh.storage
    _require(storage is not None, "storage must retain owner-local cell authority")
    assert storage is not None
    ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    stored_ids = np.asarray(storage.entity_global_ids[2], dtype=np.int64)
    order = np.argsort(ids, kind="stable")
    field_rows = order[np.searchsorted(ids[order], stored_ids)]
    stored_order = np.argsort(stored_ids, kind="stable")
    query_rows = stored_order[np.searchsorted(stored_ids[stored_order], ids)]
    (result,) = resolve_owner_local_field_values(
        (mesh,),
        (jnp.asarray(values)[field_rows],),
        (ids,),
        (np.asarray(storage.entity_owner[2])[query_rows],),
        2,
        source_topology_id=mesh.topology_id,
        devices=devices,
        axis_name="parts",
        message_capacity=message_capacity,
    )
    return result


def _local(value: jax.Array) -> np.ndarray:
    _require(len(value.addressable_shards) == 1, "allocator must have one addressable owner row")
    return np.asarray(value.addressable_shards[0].data)[0]


def transfer_history(
    result: MeshAdaptationResult,
    prepared: AdaptiveSimplexState,
    edited: AdaptiveSimplexState,
    values: jax.Array | np.ndarray,
    devices: Sequence[jax.Device],
    *,
    message_capacity: int,
    system: EulerSystem,
) -> tuple[jax.Array, dict[str, object]]:
    """Use the accepted native binary forest, never nodal interpolation or reseeding."""
    _require(result.lineage is not None, "transfer must retain accepted native lineage")
    assert result.lineage is not None
    roots = (
        require_original_meshing_source(result.source),
        require_original_meshing_source(result.target),
    )
    _require(
        result.route is MeshAdaptationRoute.DEVICE_BISECTION
        and all(
            isinstance(root, InitialCollectiveMeshEvidence)
            and isinstance(root.premise, InitialPlanarPremise)
            for root in roots
        ),
        "dyadic physical fractions must bind a native planar initial theorem",
    )
    assert isinstance(roots[0], InitialCollectiveMeshEvidence)
    assert isinstance(roots[1], InitialCollectiveMeshEvidence)
    _require(
        roots[0].evidence_id == roots[1].evidence_id,
        "source and target Euler histories must retain the same native source theorem",
    )
    source_ids, source_owned, source_areas = _cell_geometry(result.source)
    target_ids, target_owned, target_areas = _cell_geometry(result.target)
    old_ids, new_ids = _local(prepared.mesh.cell_ids), _local(edited.mesh.cell_ids)
    old_active, new_active = _local(prepared.mesh.cell_active), _local(edited.mesh.cell_active)
    parents = _local(edited.parents) // 2
    source_slots, target_slots = np.flatnonzero(old_active), np.flatnonzero(new_active)
    data = np.asarray(values)
    _require(
        data.shape == (source_ids.size, 4)
        and old_ids.shape == new_ids.shape
        and np.array_equal(np.sort(old_ids[source_slots]), np.sort(source_ids[source_owned]))
        and np.array_equal(np.sort(new_ids[target_slots]), np.sort(target_ids[target_owned]))
        and bool(np.all((old_ids < 0) | (old_ids == new_ids))),
        "native forest rows must bind exactly the accepted owned source and target cells",
    )
    allocated = np.flatnonzero(new_ids >= 0)
    _require(
        bool(np.all((parents[allocated] == -1) | ((parents[allocated] >= 0) & (parents[allocated] < allocated)))),
        "native ancestry must be an append-only acyclic parent forest",
    )
    source_order, target_order = np.argsort(source_ids), np.argsort(target_ids)
    source_rows = source_order[np.searchsorted(source_ids[source_order], old_ids[source_slots])]
    target_rows = target_order[np.searchsorted(target_ids[target_order], new_ids[target_slots])]
    source_slot_rows = np.full(new_ids.size, -1, dtype=np.int64)
    target_slot_rows = np.full(new_ids.size, -1, dtype=np.int64)
    source_slot_rows[source_slots] = source_rows
    target_slot_rows[target_slots] = target_rows
    nearest_old = np.full(new_ids.size, -1, dtype=np.int64)
    nearest_new = np.full(new_ids.size, -1, dtype=np.int64)
    new_depth = np.zeros(new_ids.size, dtype=np.int64)
    for slot in allocated:
        parent = parents[slot]
        nearest_old[slot] = slot if old_active[slot] else (-1 if parent < 0 else nearest_old[parent])
        nearest_new[slot] = slot if new_active[slot] else (-1 if parent < 0 else nearest_new[parent])
        if not new_active[slot] and parent >= 0:
            new_depth[slot] = new_depth[parent] + 1
    sources: list[int] = []
    targets: list[int] = []
    weights: list[float] = []
    prolonged: set[int] = set()
    for slot in target_slots:
        ancestor = int(nearest_old[slot])
        if ancestor >= 0:
            sources.append(int(source_slot_rows[ancestor]))
            targets.append(int(target_slot_rows[slot]))
            weights.append(1.0)
            prolonged.add(ancestor)
    restricted = [int(slot) for slot in source_slots if int(slot) not in prolonged]
    _require(
        all(nearest_new[slot] >= 0 for slot in restricted),
        "coarsened source cells must reach an accepted target parent",
    )
    for slot in restricted:
        ancestor = int(nearest_new[slot])
        sources.append(int(source_slot_rows[slot]))
        targets.append(int(target_slot_rows[ancestor]))
        weights.append(float(np.ldexp(1.0, -int(new_depth[slot]))))
    source_rows_array = np.asarray(sources, dtype=np.int64)
    target_rows_array = np.asarray(targets, dtype=np.int64)
    fractions = np.asarray(weights, dtype=np.float64)
    row_weights = np.zeros(target_ids.size)
    source_coverage = np.zeros(source_ids.size)
    np.add.at(row_weights, target_rows_array, fractions)
    np.add.at(source_coverage, source_rows_array, target_areas[target_rows_array] * fractions)
    _require(
        bool(np.all(row_weights[target_owned] == 1.0))
        and bool(np.all(np.abs(source_coverage[source_owned] - source_areas[source_owned]) <= INVENTORY_TOLERANCE)),
        "native dyadic transfer must partition every target and conserve every owned source volume",
    )
    target_values = np.zeros((target_ids.size, 4), dtype=data.dtype)
    np.add.at(target_values, target_rows_array, data[source_rows_array] * fractions[:, None])
    before = inventory(result.source, data, system=system)
    after = inventory(result.target, target_values, system=system)
    defect = float(np.max(np.abs(after - before)))
    _require(defect <= INVENTORY_TOLERANCE, "native transfer must conserve all accepted Euler inventories")
    moved = synchronize(result.target, target_values, devices, message_capacity=message_capacity)
    return moved, {
        "route": "accepted-native-binary-forest-conservative-cell-average-transfer",
        "source_result_id": result.source.result_id,
        "target_result_id": result.target.result_id,
        "lineage_id": result.lineage.lineage_id,
        "source_inventory": before.tolist(),
        "target_inventory": after.tolist(),
        "maximum_inventory_defect": defect,
        "local_transfer_id": canonical_fingerprint({
            "source": result.source.result_id,
            "target": result.target.result_id,
            "source_cells": array_tree_fingerprint(source_ids[source_rows_array]),
            "target_cells": array_tree_fingerprint(target_ids[target_rows_array]),
            "fractions": array_tree_fingerprint(fractions),
        }),
    }
