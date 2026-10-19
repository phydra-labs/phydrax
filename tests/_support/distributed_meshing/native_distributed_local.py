# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from phydrax._fingerprint import canonical_fingerprint
from phydrax.discretization._cell_mesh import CellMesh, CellMeshStorage
from phydrax.discretization._distributed_field import (
    DistributedHaloPlan,
    prepare_owner_local_halo_plans,
)
from phydrax.meshing._restart_distribution import logical_restart_routes
from tests._support.distributed_meshing.native_distributed_records import (
    _new_checks,
    _record_array,
)


_CONTRACT_KEYS = (
    "historical_routes_11",
    "historical_routes_17",
    "historical_routes_7",
    "one_owner_halo_36",
    "one_owner_halo_37",
    "one_owner_halo_41",
    "one_owner_halo_42",
)


def historical_routes() -> dict[str, Any]:
    checks: dict[str, Any] = _new_checks(_CONTRACT_KEYS)
    checks.setdefault("historical_routes_7", []).append(bool(jax.device_count() == 1))
    saved_ids = jnp.asarray(
        [[90, 10, -1, -1], [70, 90, -1, -1], [10, 80, 70, -1]], dtype=jnp.int64
    )
    _record_array(
        checks,
        "historical_routes_11",
        logical_restart_routes(saved_ids),
        [[False, True, True], [True, False, True], [True, True, False]],
    )
    changed = saved_ids.at[2, 2].set(-1)
    _record_array(
        checks,
        "historical_routes_17",
        logical_restart_routes(changed),
        [[False, True, True], [True, False, False], [True, False, False]],
    )
    return checks


def one_owner_halo() -> dict[str, Any]:
    checks: dict[str, Any] = _new_checks(_CONTRACT_KEYS)
    points = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    cells = np.asarray([[0, 1, 2]], dtype=np.int32)
    dense = CellMesh.from_triangles(points, cells)
    ids = tuple(
        (
            np.asarray(entities.entity_ids, dtype=np.int64)
            for entities in dense.topology.entity_sets
        )
    )
    counts = tuple((values.size for values in ids))
    owners = tuple((np.zeros(values.shape, dtype=np.int32) for values in ids))
    coverage = canonical_fingerprint(
        {"topology": dense.topology_id, "entity_ids": ids, "owners": owners}
    )
    storage = CellMeshStorage(
        counts,
        ids,
        owners,
        partition_index=0,
        partition_count=1,
        logical_topology_id=dense.topology_id,
        logical_geometry_id=dense.geometry_layout_id,
        evidence_id=coverage,
        logical_arrays=(
            ("cell_vertices", jnp.asarray(cells, dtype=jnp.int64)),
            ("coordinates", jnp.asarray(points, dtype=jnp.float64)),
            ("vertex_global_ids", jnp.asarray(ids[0], dtype=jnp.int64)),
        ),
        local_coordinates=points,
        local_blocks=dense.blocks,
        local_physical_boundary_facets=np.ones(counts[1], dtype=np.bool_),
        local_neighborhood_complete=np.ones(counts[2], dtype=np.bool_),
    )
    mesh = CellMesh(points, dense.blocks, storage=storage)
    (halo,) = prepare_owner_local_halo_plans(
        (mesh,), 2, devices=tuple(jax.devices()), axis_name="parts", message_capacity=1
    )
    _record_array(checks, "one_owner_halo_36", halo.local_global_ids, ids[2])
    _record_array(checks, "one_owner_halo_37", halo.local_owned, [True])
    value = jnp.asarray([2.75], dtype=jnp.float64)

    def exchange(plan: DistributedHaloPlan, field: jax.Array) -> jax.Array:
        return plan.exchange(field, plan.partition_index, axis_name="parts")

    _record_array(checks, "one_owner_halo_41", jax.jit(exchange)(halo, value), value)
    checks.setdefault("one_owner_halo_42", []).append(bool(halo.permutations == ()))
    return checks
