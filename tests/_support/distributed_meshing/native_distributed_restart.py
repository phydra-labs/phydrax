# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding, PartitionSpec

from phydrax._execution_runtime import ExecutionGroup, ExecutionRuntime
from phydrax.discretization._adaptive_simplex import (
    adaptive_simplex_state,
    AdaptiveSimplexLayout,
    AdaptiveSimplexParts,
    AdaptiveSimplexState,
    AdaptiveSimplexStatus,
    coarsen_adaptive_simplex_parts,
    MaskedSimplexMesh,
    refine_adaptive_simplex_parts,
)
from phydrax.lifecycle._distributed_checkpoint import (
    assemble_distributed_checkpoint_from_repository,
    publish_process_checkpoint,
    restore_global_array_from_checkpoint,
)
from phydrax.lifecycle._repository import (
    HPCFilesystemProfile,
    POSIXArtifactRepository,
    POSIXRepositoryPolicy,
)
from phydrax.meshing._restart_distribution import (
    repack_simplex_restart,
    restart_forest_target_owners,
)
from tests._support.distributed_meshing.native_distributed_records import (
    _new_checks,
    _record_array,
)


_CONTRACT_KEYS = (
    "repack_110",
    "repack_148",
    "repack_151",
    "repack_154",
    "repack_158",
    "repack_161",
    "repack_162",
    "repack_165",
    "repack_166",
    "repack_167",
    "repack_168",
    "repack_173",
    "repack_175",
    "repack_176",
    "repack_177",
    "repack_182",
    "repack_189",
    "repack_190",
    "repack_191",
    "repack_192",
    "repack_193",
    "repack_195",
    "repack_197",
    "repack_60",
    "repack_91",
    "repack_refusal_106",
)


def _restart_repository(
    directory: str,
) -> tuple[POSIXArtifactRepository, AdaptiveSimplexLayout]:
    root = Path(directory)
    profile = HPCFilesystemProfile(
        "posix.raw-restart",
        "local-posix",
        atomic_rename_same_filesystem=True,
        file_fsync=True,
        directory_fsync=True,
        advisory_locking=True,
        attempt_private_staging=True,
    )
    repository = POSIXArtifactRepository(
        root / "repository",
        POSIXRepositoryPolicy(
            profile, maximum_chunk_bytes=1024, maximum_metadata_bytes=262144
        ),
    )
    layout = AdaptiveSimplexLayout(
        3,
        3,
        vertex_capacity=32,
        cell_capacity=32,
        protected_edge_capacity=4,
        maximum_closure_iterations=32,
        maximum_coarsening_passes=2,
    )
    return (repository, layout)


def _checkpoint_value(arrays: dict[str, jax.Array], name: str) -> jax.Array:
    return arrays[jax.tree_util.keystr((jax.tree_util.DictKey(name),))]


def _restore_forest(arrays: dict[str, jax.Array], prefix: str) -> AdaptiveSimplexState:
    state_names = (
        "vertex_half_facets",
        "tuples",
        "tags",
        "blocks",
        "generations",
        "parents",
        "children",
        "bisection_vertices",
        "retired",
        "cell_classes",
        "facet_classes",
        "vertex_parents",
        "vertex_levels",
        "vertex_removal",
        "vertex_protected",
        "protected_codes",
        "refine_rejected",
        "coarsen_marked",
        "cursors",
        "clocks",
        "counters",
    )
    mesh = jax.vmap(MaskedSimplexMesh)(
        *[
            _checkpoint_value(arrays, f"{prefix}mesh/{name}")
            for name in (
                "coordinates",
                "vertex_ids",
                "vertex_active",
                "cells",
                "cell_ids",
                "cell_active",
                "facet_neighbors",
            )
        ]
    )
    return jax.vmap(lambda mesh, **leaves: AdaptiveSimplexState(mesh, **leaves))(
        mesh, **{name: _checkpoint_value(arrays, prefix + name) for name in state_names}
    )


def _load_restart(
    directory: str,
) -> tuple[AdaptiveSimplexLayout, ExecutionGroup, dict[str, jax.Array]]:
    repository, layout = _restart_repository(directory)
    group = ExecutionRuntime.current().root_group
    placement = group.named_sharding(PartitionSpec())
    manifest = assemble_distributed_checkpoint_from_repository(
        repository,
        "forest",
        "analysis",
        "revision",
        "execution",
        expected_process_count=2,
    )
    names = {dict(shard.metadata)["array_path"] for shard in manifest.shards}
    arrays = {
        path: restore_global_array_from_checkpoint(
            repository, manifest, path, placement, maximum_host_packet_bytes=1024
        )
        for path in names
    }
    return (layout, group, arrays)


def save_complete_epoch(directory: str) -> dict[str, Any]:
    checks: dict[str, Any] = _new_checks(_CONTRACT_KEYS)
    # `import phydrax` initialized this rank from the runner's PHYDRAX_* variables.
    repository, layout = _restart_repository(directory)
    rank = jax.process_index()
    points = np.asarray(
        [
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rows = np.asarray([[0, 1, 2, 5], [0, 2, 3, 5], [0, 3, 4, 5], [0, 4, 1, 5]], np.int32)
    indices = np.arange(2 * rank, 2 * rank + 2, dtype=np.int32)
    ids = np.unique(rows[indices]).astype(np.int64)
    local = np.searchsorted(ids, rows[indices]).astype(np.int32)
    raw = adaptive_simplex_state(
        layout,
        coordinates=points[ids],
        vertex_ids=ids,
        vertex_active=np.ones(len(ids), np.bool_),
        vertex_parents=np.full((len(ids), 2), -1, np.int32),
        vertex_protected=np.zeros(len(ids), np.bool_),
        cells=local,
        tuples=local,
        tags=np.full(2, 3, np.int32),
        blocks=np.zeros(2, np.int32),
        generations=np.zeros(2, np.int32),
        parents=np.full(2, -1, np.int32),
        children=np.full((2, 2), -1, np.int32),
        bisection_vertices=np.full(2, -1, np.int32),
        cell_ids=indices.astype(np.int64) + 10,
        cell_active=np.ones(2, np.bool_),
        cell_classes=np.full(2, 7, np.int32),
        facet_classes=np.zeros((2, 4), np.int32),
        protected_edges=np.empty((0, 2), np.int32),
        next_vertex_id=6,
        next_cell_id=14,
    )
    parts = AdaptiveSimplexParts(jax.devices(), neighbor_pairs=((0, 1), (1, 0)))
    sharding = NamedSharding(parts.mesh, PartitionSpec(parts.axis_name))
    raw = jax.tree_util.tree_map(
        lambda value: jax.make_array_from_process_local_data(
            sharding, value[None], global_shape=(2, *value.shape)
        ),
        raw,
    )
    initial = raw
    first = refine_adaptive_simplex_parts(layout, parts, raw, raw.mesh.cell_active)
    undo = coarsen_adaptive_simplex_parts(
        layout, parts, first.state, first.state.mesh.cell_active
    )
    second = refine_adaptive_simplex_parts(
        layout, parts, undo.state, undo.state.mesh.cell_active
    )
    raw = second.state
    for report in (first.report, undo.report, second.report):
        checks.setdefault("repack_60", []).append(
            bool(np.all(np.asarray(report.status.addressable_shards[0].data) == 0))
        )
    raw = eqx.tree_at(lambda value: value.vertex_protected, raw, raw.vertex_removal >= 0)
    codes = jnp.full((2, 4), jnp.iinfo(jnp.int64).max, jnp.int64).at[0, 0].set(1)
    raw = eqx.tree_at(lambda value: value.protected_codes, raw, codes)
    co = jnp.where(raw.mesh.cell_ids >= 0, jnp.arange(2, dtype=jnp.int32)[:, None], -1)
    vo = jnp.full(raw.mesh.vertex_ids.shape, 2, jnp.int32)
    for p in range(2):
        matches = jnp.any(
            raw.mesh.vertex_ids[:, :, None] == raw.mesh.vertex_ids[p, None, None, :],
            axis=-1,
        )
        vo = jnp.minimum(vo, jnp.where(matches, p, 2))
    vo = jnp.where(raw.mesh.vertex_ids >= 0, vo, -1).astype(jnp.int32)
    arrays = {
        f"mesh/{name}": getattr(raw.mesh, name)
        for name in (
            "coordinates",
            "vertex_ids",
            "vertex_active",
            "cells",
            "cell_ids",
            "cell_active",
            "facet_neighbors",
        )
    }
    state_names = (
        "vertex_half_facets",
        "tuples",
        "tags",
        "blocks",
        "generations",
        "parents",
        "children",
        "bisection_vertices",
        "retired",
        "cell_classes",
        "facet_classes",
        "vertex_parents",
        "vertex_levels",
        "vertex_removal",
        "vertex_protected",
        "protected_codes",
        "refine_rejected",
        "coarsen_marked",
        "cursors",
        "clocks",
        "counters",
    )
    arrays.update({name: getattr(raw, name) for name in state_names})
    arrays.update(
        {
            f"initial/mesh/{name}": getattr(initial.mesh, name)
            for name in (
                "coordinates",
                "vertex_ids",
                "vertex_active",
                "cells",
                "cell_ids",
                "cell_active",
                "facet_neighbors",
            )
        }
    )
    arrays.update({f"initial/{name}": getattr(initial, name) for name in state_names})
    initial_co = jnp.where(
        initial.mesh.cell_ids >= 0, jnp.arange(2, dtype=jnp.int32)[:, None], -1
    )
    initial_vo = jnp.full(initial.mesh.vertex_ids.shape, 2, jnp.int32)
    for p in range(2):
        matches = jnp.any(
            initial.mesh.vertex_ids[:, :, None]
            == initial.mesh.vertex_ids[p, None, None, :],
            axis=-1,
        )
        initial_vo = jnp.minimum(initial_vo, jnp.where(matches, p, 2))
    arrays.update(
        initial_co=initial_co,
        initial_vo=jnp.where(initial.mesh.vertex_ids >= 0, initial_vo, -1).astype(
            jnp.int32
        ),
    )
    arrays.update(
        co=co,
        vo=vo,
        cell_int=raw.mesh.cell_ids + 2**54,
        cell_bool=raw.retired,
        cell_vector=raw.mesh.cell_ids[:, :, None] * jnp.asarray([2.0, 3.0], jnp.float64),
        vertex_int=raw.mesh.vertex_ids + 2**54,
        vertex_bool=raw.vertex_protected,
        vertex_vector=raw.mesh.coordinates,
    )
    arrays = {name: jax.device_put(value, sharding) for name, value in arrays.items()}
    checks.setdefault("repack_91", []).append(
        bool(not raw.mesh.cell_ids.is_fully_addressable)
    )
    publish_process_checkpoint(
        repository,
        "forest",
        "execution",
        arrays,
        analysis_plan_id="analysis",
        numeric_revision_id="revision",
        writer_id=f"writer-{rank}",
    )
    multihost_utils.sync_global_devices("raw-forest-published")
    assemble_distributed_checkpoint_from_repository(
        repository,
        "forest",
        "analysis",
        "revision",
        "execution",
        expected_process_count=2,
    )
    jax.distributed.shutdown()
    checks["partition"] = rank
    return checks


def restart_positive(directory: str, device_count: str) -> dict[str, Any]:
    checks: dict[str, Any] = _new_checks(_CONTRACT_KEYS)
    count = int(device_count)
    layout, group, arrays = _load_restart(directory)
    raw = _restore_forest(arrays, "")
    co, vo = (_checkpoint_value(arrays, "co"), _checkpoint_value(arrays, "vo"))
    old = tuple((np.asarray(leaf).copy() for leaf in jax.tree_util.tree_leaves(raw)))
    proposal = jnp.broadcast_to(
        (jnp.arange(32, dtype=jnp.int32) % count)[None, :], (2, 32)
    )
    receipt = repack_simplex_restart(
        group,
        layout,
        raw,
        co,
        vo,
        proposal,
        cell_history=(
            _checkpoint_value(arrays, "cell_int"),
            _checkpoint_value(arrays, "cell_bool"),
            _checkpoint_value(arrays, "cell_vector"),
        ),
        vertex_history=(
            _checkpoint_value(arrays, "vertex_int"),
            _checkpoint_value(arrays, "vertex_bool"),
            _checkpoint_value(arrays, "vertex_vector"),
        ),
    )
    assert isinstance(receipt.parts, AdaptiveSimplexParts)
    for expected, leaf in zip(old, jax.tree_util.tree_leaves(raw), strict=True):
        _record_array(checks, "repack_148", expected, leaf)
    _record_array(checks, "repack_154", receipt.status, np.zeros(count, np.int32))
    initial = _restore_forest(arrays, "initial/")
    targets, target_status = restart_forest_target_owners(
        initial.mesh.cell_ids, receipt.states.mesh.cell_ids, receipt.cell_owners
    )
    _record_array(checks, "repack_158", target_status, 0)
    initial_repack = repack_simplex_restart(
        group,
        layout,
        initial,
        _checkpoint_value(arrays, "initial_co"),
        _checkpoint_value(arrays, "initial_vo"),
        targets,
        cell_history=(initial.mesh.facet_neighbors < 0,),
    )
    _record_array(checks, "repack_161", initial_repack.status, np.zeros(count, np.int32))
    checks.setdefault("repack_162", []).append(
        bool(int(jnp.sum(initial_repack.cells_after)) == 4)
    )
    _, missing_status = restart_forest_target_owners(
        initial.mesh.cell_ids.at[0, 0].set(999),
        receipt.states.mesh.cell_ids,
        receipt.cell_owners,
    )
    checks.setdefault("repack_165", []).append(
        bool(int(missing_status) & int(AdaptiveSimplexStatus.INVALID_GEOMETRY))
    )
    checks.setdefault("repack_166", []).append(
        bool(
            receipt.parts.part_count == count
            and receipt.states.mesh.cell_ids.shape == (count, 32)
        )
    )
    checks.setdefault("repack_167", []).append(
        bool(len(receipt.parts.mesh.devices.flat) == count)
    )
    checks.setdefault("repack_168", []).append(
        bool(np.sum(np.asarray(receipt.cells_after)) == 20)
    )
    newids = np.asarray(receipt.states.mesh.cell_ids)
    used = newids >= 0
    locations = np.asarray(receipt.cell_saved_locations)[used]
    for name in (
        "tags",
        "blocks",
        "generations",
        "retired",
        "cell_classes",
        "facet_classes",
        "refine_rejected",
        "coarsen_marked",
    ):
        _record_array(
            checks,
            "repack_173",
            np.asarray(getattr(receipt.states, name))[used],
            np.asarray(getattr(raw, name))[locations[:, 0], locations[:, 1]],
        )
    _record_array(
        checks,
        "repack_175",
        np.asarray(receipt.cell_history[0])[used],
        newids[used] + 2**54,
    )
    _record_array(checks, "repack_176", receipt.cell_history[1], receipt.states.retired)
    _record_array(
        checks,
        "repack_177",
        np.asarray(receipt.cell_history[2])[used],
        newids[used, None] * [2.0, 3.0],
    )
    vids = np.asarray(receipt.states.mesh.vertex_ids)
    valid = vids >= 0
    vlocations = np.asarray(receipt.vertex_saved_locations)[valid]
    for name in ("vertex_levels", "vertex_removal", "vertex_protected"):
        _record_array(
            checks,
            "repack_182",
            np.asarray(getattr(receipt.states, name))[valid],
            np.asarray(getattr(raw, name))[vlocations[:, 0], vlocations[:, 1]],
        )
    constraints = []
    for p in range(count):
        codes = np.asarray(receipt.states.protected_codes[p])
        codes = codes[codes != np.iinfo(np.int64).max]
        constraints.extend(vids[p, np.stack((codes // 32, codes % 32), axis=1)].tolist())
    checks.setdefault("repack_189", []).append(
        bool(constraints and all((edge == [0, 1] for edge in constraints)))
    )
    _record_array(
        checks,
        "repack_190",
        np.asarray(receipt.vertex_history[0])[valid],
        vids[valid] + 2**54,
    )
    _record_array(
        checks, "repack_191", receipt.vertex_history[1], receipt.states.vertex_protected
    )
    _record_array(
        checks, "repack_192", receipt.vertex_history[2], receipt.states.mesh.coordinates
    )
    _record_array(
        checks,
        "repack_193",
        receipt.states.counters.sum(axis=0),
        raw.counters.sum(axis=0),
    )
    continued = coarsen_adaptive_simplex_parts(
        layout, receipt.parts, receipt.states, receipt.states.mesh.cell_active
    )
    _record_array(
        checks, "repack_195", continued.report.status, np.zeros(count, np.int32)
    )
    active = np.asarray(continued.state.mesh.cell_active)
    _record_array(
        checks,
        "repack_197",
        np.sort(np.asarray(continued.state.mesh.cell_ids)[active]),
        [10, 11, 12, 13],
    )
    print(
        "Saved actual2process forest resumed on",
        count,
        "actual devices; typed history and coarsening preserved",
    )
    return checks


def restart_duplicate(directory: str, device_count: str) -> dict[str, Any]:
    checks: dict[str, Any] = _new_checks(_CONTRACT_KEYS)
    count = int(device_count)
    layout, group, arrays = _load_restart(directory)
    raw = _restore_forest(arrays, "")
    co, vo = (_checkpoint_value(arrays, "co"), _checkpoint_value(arrays, "vo"))
    raw = eqx.tree_at(
        lambda x: x.mesh.cell_ids,
        raw,
        raw.mesh.cell_ids.at[1, 0].set(raw.mesh.cell_ids[0, 0]),
    )
    old = tuple((np.asarray(leaf).copy() for leaf in jax.tree_util.tree_leaves(raw)))
    proposal = jnp.broadcast_to(
        (jnp.arange(32, dtype=jnp.int32) % count)[None, :], (2, 32)
    )
    receipt = repack_simplex_restart(
        group,
        layout,
        raw,
        co,
        vo,
        proposal,
        cell_history=(
            _checkpoint_value(arrays, "cell_int"),
            _checkpoint_value(arrays, "cell_bool"),
            _checkpoint_value(arrays, "cell_vector"),
        ),
        vertex_history=(
            _checkpoint_value(arrays, "vertex_int"),
            _checkpoint_value(arrays, "vertex_bool"),
            _checkpoint_value(arrays, "vertex_vector"),
        ),
    )
    for expected, leaf in zip(old, jax.tree_util.tree_leaves(raw), strict=True):
        _record_array(checks, "repack_148", expected, leaf)
    required = AdaptiveSimplexStatus.INVALID_GEOMETRY
    checks.setdefault("repack_151", []).append(
        bool(np.all(np.asarray(receipt.status) & int(required)))
    )
    return checks


def restart_missing(directory: str, device_count: str) -> dict[str, Any]:
    checks: dict[str, Any] = _new_checks(_CONTRACT_KEYS)
    count = int(device_count)
    layout, group, arrays = _load_restart(directory)
    raw = _restore_forest(arrays, "")
    co, vo = (_checkpoint_value(arrays, "co"), _checkpoint_value(arrays, "vo"))
    raw = eqx.tree_at(
        lambda x: x.mesh.vertex_ids, raw, raw.mesh.vertex_ids.at[:, 0].set(-1)
    )
    old = tuple((np.asarray(leaf).copy() for leaf in jax.tree_util.tree_leaves(raw)))
    proposal = jnp.broadcast_to(
        (jnp.arange(32, dtype=jnp.int32) % count)[None, :], (2, 32)
    )
    receipt = repack_simplex_restart(
        group,
        layout,
        raw,
        co,
        vo,
        proposal,
        cell_history=(
            _checkpoint_value(arrays, "cell_int"),
            _checkpoint_value(arrays, "cell_bool"),
            _checkpoint_value(arrays, "cell_vector"),
        ),
        vertex_history=(
            _checkpoint_value(arrays, "vertex_int"),
            _checkpoint_value(arrays, "vertex_bool"),
            _checkpoint_value(arrays, "vertex_vector"),
        ),
    )
    for expected, leaf in zip(old, jax.tree_util.tree_leaves(raw), strict=True):
        _record_array(checks, "repack_148", expected, leaf)
    required = AdaptiveSimplexStatus.INVALID_GEOMETRY
    checks.setdefault("repack_151", []).append(
        bool(np.all(np.asarray(receipt.status) & int(required)))
    )
    return checks


def restart_overflow(directory: str, device_count: str) -> dict[str, Any]:
    checks: dict[str, Any] = _new_checks(_CONTRACT_KEYS)
    count = int(device_count)
    layout, group, arrays = _load_restart(directory)
    raw = _restore_forest(arrays, "")
    co, vo = (_checkpoint_value(arrays, "co"), _checkpoint_value(arrays, "vo"))
    layout = AdaptiveSimplexLayout(
        3,
        3,
        vertex_capacity=32,
        cell_capacity=8,
        protected_edge_capacity=4,
        maximum_closure_iterations=32,
        maximum_coarsening_passes=2,
    )
    old = tuple((np.asarray(leaf).copy() for leaf in jax.tree_util.tree_leaves(raw)))
    proposal = jnp.broadcast_to(
        (jnp.arange(32, dtype=jnp.int32) % count)[None, :], (2, 32)
    )
    receipt = repack_simplex_restart(
        group,
        layout,
        raw,
        co,
        vo,
        proposal,
        cell_history=(
            _checkpoint_value(arrays, "cell_int"),
            _checkpoint_value(arrays, "cell_bool"),
            _checkpoint_value(arrays, "cell_vector"),
        ),
        vertex_history=(
            _checkpoint_value(arrays, "vertex_int"),
            _checkpoint_value(arrays, "vertex_bool"),
            _checkpoint_value(arrays, "vertex_vector"),
        ),
    )
    for expected, leaf in zip(old, jax.tree_util.tree_leaves(raw), strict=True):
        _record_array(checks, "repack_148", expected, leaf)
    required = AdaptiveSimplexStatus.CAPACITY_EXCEEDED
    checks.setdefault("repack_151", []).append(
        bool(np.all(np.asarray(receipt.status) & int(required)))
    )
    return checks


def restart_shared_conflict(directory: str, device_count: str) -> dict[str, Any]:
    checks: dict[str, Any] = _new_checks(_CONTRACT_KEYS)
    count = int(device_count)
    layout, group, arrays = _load_restart(directory)
    raw = _restore_forest(arrays, "")
    co, vo = (_checkpoint_value(arrays, "co"), _checkpoint_value(arrays, "vo"))
    raw = eqx.tree_at(
        lambda x: x.mesh.coordinates, raw, raw.mesh.coordinates.at[1, 0, 0].add(0.25)
    )
    old = tuple((np.asarray(leaf).copy() for leaf in jax.tree_util.tree_leaves(raw)))
    proposal = jnp.broadcast_to(
        (jnp.arange(32, dtype=jnp.int32) % count)[None, :], (2, 32)
    )
    receipt = repack_simplex_restart(
        group,
        layout,
        raw,
        co,
        vo,
        proposal,
        cell_history=(
            _checkpoint_value(arrays, "cell_int"),
            _checkpoint_value(arrays, "cell_bool"),
            _checkpoint_value(arrays, "cell_vector"),
        ),
        vertex_history=(
            _checkpoint_value(arrays, "vertex_int"),
            _checkpoint_value(arrays, "vertex_bool"),
            _checkpoint_value(arrays, "vertex_vector"),
        ),
    )
    for expected, leaf in zip(old, jax.tree_util.tree_leaves(raw), strict=True):
        _record_array(checks, "repack_148", expected, leaf)
    required = AdaptiveSimplexStatus.INVALID_GEOMETRY
    checks.setdefault("repack_151", []).append(
        bool(np.all(np.asarray(receipt.status) & int(required)))
    )
    return checks


def restart_lost_authority(directory: str, device_count: str) -> dict[str, Any]:
    checks: dict[str, Any] = _new_checks(_CONTRACT_KEYS)
    count = int(device_count)
    layout, group, arrays = _load_restart(directory)
    raw = _restore_forest(arrays, "")
    co, vo = (_checkpoint_value(arrays, "co"), _checkpoint_value(arrays, "vo"))
    co = jnp.where(raw.mesh.cell_ids >= 0, jnp.ones_like(co), -1)
    old = tuple((np.asarray(leaf).copy() for leaf in jax.tree_util.tree_leaves(raw)))
    proposal = jnp.broadcast_to(
        (jnp.arange(32, dtype=jnp.int32) % count)[None, :], (2, 32)
    )
    receipt = repack_simplex_restart(
        group,
        layout,
        raw,
        co,
        vo,
        proposal,
        cell_history=(
            _checkpoint_value(arrays, "cell_int"),
            _checkpoint_value(arrays, "cell_bool"),
            _checkpoint_value(arrays, "cell_vector"),
        ),
        vertex_history=(
            _checkpoint_value(arrays, "vertex_int"),
            _checkpoint_value(arrays, "vertex_bool"),
            _checkpoint_value(arrays, "vertex_vector"),
        ),
    )
    for expected, leaf in zip(old, jax.tree_util.tree_leaves(raw), strict=True):
        _record_array(checks, "repack_148", expected, leaf)
    required = AdaptiveSimplexStatus.INVALID_GEOMETRY
    checks.setdefault("repack_151", []).append(
        bool(np.all(np.asarray(receipt.status) & int(required)))
    )
    return checks


def restart_packet_overflow(directory: str) -> dict[str, Any]:
    checks: dict[str, Any] = _new_checks(_CONTRACT_KEYS)
    repository, layout = _restart_repository(directory)
    group = ExecutionRuntime.current().root_group
    placement = group.named_sharding(PartitionSpec())
    manifest = assemble_distributed_checkpoint_from_repository(
        repository,
        "forest",
        "analysis",
        "revision",
        "execution",
        expected_process_count=2,
    )
    with pytest.raises(ValueError) as refusal_106:
        restore_global_array_from_checkpoint(
            repository,
            manifest,
            "['mesh/coordinates']",
            placement,
            maximum_host_packet_bytes=1,
        )
    error = refusal_106.value
    checks.setdefault("repack_110", []).append(
        bool("maximum_host_packet_bytes" in str(error))
    )
    checks["repack_refusal_106"].append(True)
    print("Immutable archive refused before numerical reconstruction")
    return checks
