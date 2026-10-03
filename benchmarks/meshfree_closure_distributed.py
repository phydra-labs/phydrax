# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Q16 distributed meshfree workloads and the forced-device subprocess launcher.

Every workload owns all devices of the current JAX runtime, one owner per
device. Forced host CPU devices
(``XLA_FLAGS=--xla_force_host_platform_device_count=<k>``) prove functional
distributed parity only: neighbor completeness against an independent
``scipy.spatial.cKDTree`` oracle, halo transpose duality, single-device action
equality, migration preserving stable IDs and values, and owner-collective
reductions. Their timings are not GPU or multi-host performance evidence.

Declared floating-point bounds (never tuned): a sum of ``m`` terms computed in
precision ``eps`` is within ``m * eps`` times the sum of term magnitudes of the
exact value (Higham's gamma_m to first order). Two computations of one sum
differ by at most twice that; a safety factor of two gives ``4 * m * eps``.
Neighbor identity is compared only on rows whose oracle k-th/(k+1)-th squared
distance gap exceeds the certification margin ``16 * d * eps`` (squared
distances of unit-cube points carry at most a few ``d * eps`` rounding);
tied rows are counted and reported.
"""

from __future__ import annotations

import json
import math
import os
import resource
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax import Array

from benchmarks._runtime import logical_array_bytes
from benchmarks.meshfree_scaling import (
    cloud_points,
    declare_reservation,
    MeshfreeConfig,
    PhaseRecorder,
    PROJECT_ROOT,
)
from phydrax.discretization.meshfree import (
    distributed_inner,
    distributed_norm,
    distributed_sum,
    DistributedMeshfreeOperator,
    LocalStencilPolicy,
    MeshfreeFunctional,
    MeshfreeNeighborhoodPlan,
    MeshfreeOperator,
    MeshfreePrecisionPolicy,
    MeshfreePrecisionRole,
    prepare_local_stencils,
)
from phydrax.discretization.spatial import (
    DistributedHaloPlan,
    DistributedMigrationResult,
    DistributedNeighborQueryPlan,
    DistributedNeighborResult,
    DistributedOwnershipPlan,
    DistributedPointLayout,
    DistributedRadiusQueryPlan,
    DistributedRelationStatus,
    MortonAddressPlan,
)
from phydrax.execution import ExecutionRuntime
from phydrax.linalg import ArraySpace, DiagonalPairing
from phydrax.sparse import EdgeRelation, RowRelation, SparseCoordinateOperator
from phydrax.typing import parse


DistributedPrecisionProfile: TypeAlias = Literal["uniform", "mixed"]
DistributedBoundary: TypeAlias = Literal["open-box", "periodic-all-axes"]
ReferenceMode: TypeAlias = Literal["apply", "transpose", "adjoint"]
type Reference = Callable[[np.ndarray, ReferenceMode], np.ndarray]

# Expected radius-row population and declared row width: four times the mean
# count keeps accepted quasi-uniform rows far below overflow.
_RADIUS_MEAN_NEIGHBORS = 12
_RADIUS_ROW_WIDTH = 4 * _RADIUS_MEAN_NEIGHBORS
# Declared owner-slot slack over the larger of the sector and slab partitions.
_LOCAL_CAPACITY_SLACK = 1.25
_MORTON_BITS = 10
_FORCED_SCOPE = (
    "forced host CPU devices: functional distributed parity only, not GPU or "
    "multi-host performance evidence"
)


# ---------------------------------------------------------------------------
# Forced multi-device subprocess launcher (shared by Q16 and Q17 changed ownership)


def run_forced_device_workloads(
    workloads: tuple[str, ...],
    config: MeshfreeConfig,
    /,
    *,
    devices: int,
    output: Path,
) -> dict[str, Any]:
    """Run closure workloads in a fresh process with ``devices`` forced CPU devices.

    The device count must be fixed before JAX initializes, so workloads run
    through ``python -m benchmarks.meshfree_closure``; the written record is
    returned with the subprocess wall time and child CPU time.
    """
    if type(devices) is not int or devices < 1:
        raise ValueError("devices must be a positive forced CPU device count.")
    if not workloads:
        raise ValueError("Select at least one workload.")
    if config.working_set_bytes % 1024**2 or config.resource_bytes % 1024**2:
        raise ValueError("Subprocess budgets must be whole MiB; none is rounded.")
    command = [
        sys.executable,
        "-m",
        "benchmarks.meshfree_closure",
        "--workloads",
        *workloads,
        "--sizes",
        *(str(size) for size in config.sizes),
        "--seeds",
        *(str(seed) for seed in config.seeds),
        "--dimension",
        str(config.dimension),
        "--repeats",
        str(config.repeats),
        "--neighbors",
        str(config.neighbors),
        "--degree",
        str(config.degree),
        "--chunk-rows",
        str(config.chunk_rows),
        "--working-set-mib",
        str(config.working_set_bytes // 1024**2),
        "--resource-mib",
        str(config.resource_bytes // 1024**2),
        "--max-points",
        str(config.max_points),
        "--steps",
        str(config.steps),
        "--precision",
        config.precision,
        *(() if config.candidates is None else ("--candidates", str(config.candidates))),
        "--output",
        str(output),
    ]
    flags = f"--xla_force_host_platform_device_count={devices}"
    environment = {
        **os.environ,
        "XLA_FLAGS": flags,
        "JAX_PLATFORMS": "cpu",
        "JAX_ENABLE_X64": "1",
    }
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    wall = time.perf_counter() - started
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Forced-device workload process exited {completed.returncode}:\n"
            + completed.stderr[-6000:]
        )
    record = json.loads(Path(output).read_text(encoding="utf-8"))
    record["subprocess"] = {
        "command": command,
        "devices": devices,
        "xla_flags": flags,
        "wall_seconds": wall,
        "cpu_seconds": (after.ru_utime - before.ru_utime)
        + (after.ru_stime - before.ru_stime),
        "returncode": completed.returncode,
        "scope": _FORCED_SCOPE,
    }
    return record


# ---------------------------------------------------------------------------
# Shared preparation and independent oracles


@dataclass(frozen=True, slots=True)
class _Setup:
    """Host-prepared cloud, uneven sector ownership and its owner-blocked layout."""

    points: np.ndarray
    ids: np.ndarray
    sectors: np.ndarray
    owners: int
    local: int
    layout: DistributedPointLayout
    plan: DistributedOwnershipPlan
    dtype: np.dtype
    periodic: bool


def _sector_owners(points: np.ndarray, owners: int, /) -> np.ndarray:
    """Uneven angular sectors (widths proportional to 1..R) about the domain center.

    Every sector meets the center, so kNN shells there cross at least three
    owners whenever R >= 3, and owner loads differ by up to a factor R.
    """
    angle = np.arctan2(points[:, 1] - 0.5, points[:, 0] - 0.5)
    widths = np.arange(1, owners + 1, dtype=np.float64)
    edges = -np.pi + 2.0 * np.pi * np.cumsum(widths)[:-1] / widths.sum()
    return np.digitize(angle, edges).astype(np.int32)


def _slab_owners(points: np.ndarray, owners: int, /) -> np.ndarray:
    """Balanced x-slab repartition used as the migration destination."""
    return np.minimum(np.floor(points[:, 0] * owners), owners - 1).astype(np.int32)


def _relative(first: np.ndarray, second: np.ndarray, periodic: bool, /) -> np.ndarray:
    difference = first - second
    return difference - np.round(difference) if periodic else difference


def _knn_oracle(
    points: np.ndarray, k: int, periodic: bool, /
) -> tuple[np.ndarray, np.ndarray]:
    """Independent float64 cKDTree kNN with k + 1 columns and exact squared distances."""
    from scipy.spatial import cKDTree

    tree = (
        cKDTree(points, boxsize=np.ones(points.shape[1], dtype=np.float64))
        if periodic
        else cKDTree(points)
    )
    _, index = tree.query(points, k=k + 1)
    index = np.asarray(index, dtype=np.intp)
    if index.shape != (points.shape[0], k + 1):
        raise ValueError(
            "Independent kNN oracle requires one complete k + 1 row per point."
        )
    squared = np.sum(_relative(points[:, None, :], points[index], periodic) ** 2, axis=-1)
    order = np.argsort(squared, axis=1, kind="stable")
    return np.take_along_axis(index, order, 1), np.take_along_axis(squared, order, 1)


def _setup(
    capacity: int,
    seed: int,
    config: MeshfreeConfig,
    dtype: np.dtype,
    boundary: DistributedBoundary,
    /,
) -> _Setup:
    selected = parse(boundary, DistributedBoundary, "boundary")
    if config.dimension < 2:
        raise ValueError("Distributed sector ownership needs dimension 2 or 3.")
    periodic = selected == "periodic-all-axes"
    group = ExecutionRuntime.current().child_groups(1)[0]
    owners = len(group.devices)
    points = cloud_points(capacity, config.dimension, seed).astype(dtype)
    sectors = _sector_owners(points.astype(np.float64), owners)
    slabs = _slab_owners(points.astype(np.float64), owners)
    loads = max(
        int(np.bincount(sectors, minlength=owners).max()),
        int(np.bincount(slabs, minlength=owners).max()),
    )
    local = math.ceil(_LOCAL_CAPACITY_SLACK * loads)
    address = MortonAddressPlan(
        (0.0,) * config.dimension,
        (1.0,) * config.dimension,
        _MORTON_BITS,
        periodic_axes=(periodic,) * config.dimension,
    )
    plan = DistributedOwnershipPlan(address, group, local)
    ids = np.arange(capacity, dtype=np.int64) * 7 + 11
    layout = DistributedPointLayout.from_global(plan, points, sectors, stable_ids=ids)
    return _Setup(
        points, ids, sectors, owners, local, layout, plan, np.dtype(dtype), periodic
    )


def _reservation(setup: _Setup, k: int, config: MeshfreeConfig, /) -> int:
    """Planning reservation (not a measured bound): relations, candidates, halos, routes."""
    slots = setup.owners * setup.local
    relation_bytes = 29 * slots * (k + _RADIUS_ROW_WIDTH)
    halo_bytes = 8 * setup.owners * setup.owners * setup.local * (config.dimension + 4)
    route_bytes = 24 * setup.points.shape[0] * k
    return declare_reservation(
        5 * relation_bytes + halo_bytes + route_bytes,
        config,
        scope="distributed relation planning",
    )


def _margin(dimension: int, dtype: np.dtype, /) -> float:
    return 16.0 * dimension * float(np.finfo(dtype).eps)


def _knn_metrics(
    result: DistributedNeighborResult,
    layout: DistributedPointLayout,
    ids: np.ndarray,
    oracle: tuple[np.ndarray, np.ndarray],
    margin: float,
    k: int,
    /,
) -> dict[str, Any]:
    index, squared = oracle
    found = np.asarray(layout.collect(result.source_stable_ids))
    distance = np.asarray(layout.collect(result.distance_squared), dtype=np.float64)
    status_codes = np.asarray(layout.collect(result.status))
    complete = status_codes == DistributedRelationStatus.COMPLETE
    certified = (squared[:, k] - squared[:, k - 1]) > margin
    expected_ids = ids[index[:, :k]]
    matched = np.all(np.sort(found, axis=1) == np.sort(expected_ids, axis=1), axis=1)
    gated = complete & certified
    found_order = np.argsort(found, axis=1)
    expected_order = np.argsort(expected_ids, axis=1)
    distance_error = np.abs(
        np.take_along_axis(distance, found_order, 1)
        - np.take_along_axis(squared[:, :k], expected_order, 1)
    )
    owners = np.asarray(layout.collect(result.source_owners))
    entries = np.asarray(layout.collect(result.valid))
    refusals = {
        f"status_{status.name.lower()}_rows": int(
            np.count_nonzero(status_codes == status)
        )
        for status in DistributedRelationStatus
        if status != DistributedRelationStatus.COMPLETE and np.any(status_codes == status)
    }
    return {
        **refusals,
        "incomplete_rows": int(np.count_nonzero(~complete)),
        "certified_rows": int(np.count_nonzero(certified)),
        "near_tie_rows": int(np.count_nonzero(~certified)),
        "mismatched_certified_rows": int(np.count_nonzero(gated & ~matched)),
        "distance_error": float(np.max(distance_error[gated & matched], initial=0.0)),
        "max_row_owners": max(
            len(set(row[used].tolist()))
            for row, used in zip(owners, entries, strict=True)
        ),
        "communicated_targets": int(result.evidence.communicated_targets),
        # Shell contact ships a target only to owners that may hold a neighbor;
        # all-to-all replication would ship N * (R - 1) targets.
        "targets_not_replicated": layout.plan.owner_count == 1
        or int(result.evidence.communicated_targets)
        < layout.logical_count * (layout.plan.owner_count - 1),
        "maximum_halo_load": int(result.evidence.maximum_halo_load),
        "maximum_required_owners": int(result.evidence.maximum_required_owners),
        "query_successful": bool(result.evidence.successful),
        "required_candidates": int(result.evidence.required_candidates),
        "candidate_capacity": int(result.evidence.candidate_capacity),
    }


def _pair_metrics(
    rows: DistributedNeighborResult, setup: _Setup, radius: float, margin: float, /
) -> dict[str, Any]:
    """Pair-once radius edges against an independent cKDTree pair enumeration."""
    from scipy.spatial import cKDTree

    valid = np.asarray(rows.valid)
    sources = np.asarray(rows.source_stable_ids)[valid]
    targets = np.repeat(np.asarray(setup.layout.stable_ids), valid.sum(axis=1))
    pairs = list(zip(targets.tolist(), sources.tolist(), strict=True))
    found = set(pairs)
    points = setup.points.astype(np.float64)
    tree = (
        cKDTree(points, boxsize=np.ones(points.shape[1], dtype=np.float64))
        if setup.periodic
        else cKDTree(points)
    )
    reach = math.sqrt(radius * radius + margin)
    candidates = np.asarray(sorted(tree.query_pairs(reach)), dtype=np.int64).reshape(
        (-1, 2)
    )
    squared = np.sum(
        _relative(points[candidates[:, 0]], points[candidates[:, 1]], setup.periodic)
        ** 2,
        axis=-1,
    )
    first, second = setup.ids[candidates[:, 0]], setup.ids[candidates[:, 1]]
    keyed = list(
        zip(
            np.minimum(first, second).tolist(),
            np.maximum(first, second).tolist(),
            strict=True,
        )
    )
    certain = {
        pair
        for pair, value in zip(keyed, squared.tolist(), strict=True)
        if value < radius**2 - margin
    }
    status = np.asarray(setup.layout.collect(rows.status))
    return {
        "pair_missing": len(certain - found),
        "pair_extra": len(found - set(keyed)),
        "pair_duplicates": len(pairs) - len(found),
        "pair_near_boundary": int(
            np.count_nonzero(np.abs(squared - radius**2) <= margin)
        ),
        "pair_count": len(found),
        "pair_incomplete_rows": int(
            np.count_nonzero(status != DistributedRelationStatus.COMPLETE)
        ),
    }


def _relation_phase(
    recorder: PhaseRecorder, setup: _Setup, k: int, config: MeshfreeConfig, /
) -> tuple[
    dict[str, Any],
    DistributedNeighborQueryPlan,
    DistributedNeighborResult,
    tuple[np.ndarray, np.ndarray],
]:
    margin = _margin(config.dimension, setup.dtype)
    knn = DistributedNeighborQueryPlan(
        setup.plan,
        setup.plan,
        k,
        maximum_remote_owners=max(setup.owners - 1, 1),
        halo_capacity=setup.local,
        maximum_candidates=config.candidates,
    )
    rows = recorder.run_repeated(
        "search",
        lambda: knn.query(setup.layout, setup.layout),
        repeats=config.repeats,
        scope="distributed-knn",
    )
    oracle = _knn_oracle(setup.points.astype(np.float64), k, setup.periodic)
    metrics = _knn_metrics(rows, setup.layout, setup.ids, oracle, margin, k)
    volume = math.pi if config.dimension == 2 else 4.0 * math.pi / 3.0
    radius = (_RADIUS_MEAN_NEIGHBORS / (setup.points.shape[0] * volume)) ** (
        1.0 / config.dimension
    )
    radius_plan = DistributedRadiusQueryPlan(
        setup.plan,
        setup.plan,
        radius,
        _RADIUS_ROW_WIDTH,
        maximum_remote_owners=max(setup.owners - 1, 1),
        halo_capacity=setup.local,
        maximum_candidates=config.candidates,
    )
    pairs = recorder.run(
        "search",
        lambda: radius_plan.query(
            setup.layout, setup.layout, exclude_self=True, pair_once=True
        ),
        scope="distributed-radius-pair-once",
    )
    metrics.update(_pair_metrics(pairs, setup, radius, margin))
    metrics["radius"] = radius
    metrics["certification_margin"] = margin
    return metrics, knn, rows, oracle


def _halo_metrics(
    recorder: PhaseRecorder,
    rows: DistributedNeighborResult,
    setup: _Setup,
    rng: np.random.Generator,
    /,
) -> dict[str, Any]:
    """Forward gather exactness, transpose duality and exactly-once arrival."""
    halo = recorder.run(
        "assembly",
        lambda: DistributedHaloPlan(
            setup.plan,
            rows.source_owners.reshape((-1,)),
            rows.source_slots.reshape((-1,)),
            rows.valid.reshape((-1,)),
            halo_capacity=setup.local,
        ),
        scope="halo-plan",
    )
    owners, local, columns = setup.owners, setup.local, halo.column_count
    values = rng.normal(size=(owners * local,)).astype(setup.dtype)
    cotangent = rng.normal(size=(owners * columns,)).astype(setup.dtype)
    gathered = recorder.run_repeated(
        "communication-migration",
        lambda: halo.gather(values),
        repeats=2,
        scope="halo-gather",
    )
    transposed = recorder.run_repeated(
        "communication-migration",
        lambda: halo.transpose(cotangent),
        repeats=2,
        scope="halo-transpose",
    )
    left = np.asarray(gathered, dtype=np.float64) * cotangent.astype(np.float64)
    right = values.astype(np.float64) * np.asarray(transposed, dtype=np.float64)
    route_columns = np.asarray(halo.route_columns).reshape((owners, -1))
    route_valid = np.asarray(halo.route_valid).reshape((owners, -1))
    source_rows = (
        np.asarray(rows.source_owners) * local + np.asarray(rows.source_slots)
    ).reshape((owners, -1))
    gathered_host = np.asarray(gathered).reshape((owners, columns))
    marks = np.zeros((owners, columns), dtype=setup.dtype)
    expected = np.zeros((owners * local,), dtype=np.float64)
    gather_errors = 0
    for owner in range(owners):
        selected = route_valid[owner]
        marks[owner, route_columns[owner, selected]] = 1.0
        for source in set(source_rows[owner, selected].tolist()):
            expected[source] += 1.0
        gather_errors += int(
            np.count_nonzero(
                gathered_host[owner, route_columns[owner, selected]]
                != values[source_rows[owner, selected]]
            )
        )
    received = np.asarray(halo.transpose(marks.reshape(-1)), dtype=np.float64)
    return {
        "halo_successful": bool(halo.evidence.successful),
        "halo_columns": int(halo.evidence.halo_columns),
        "halo_gather_errors": gather_errors,
        "halo_duality_error": abs(math.fsum(left.tolist()) - math.fsum(right.tolist()))
        / math.fsum(np.abs(left).tolist()),
        "halo_exactly_once_errors": int(np.count_nonzero(received != expected)),
    }


def _migration(
    recorder: PhaseRecorder, setup: _Setup, payload: Array, /
) -> tuple[dict[str, Any], DistributedPointLayout]:
    """Atomic repartition from uneven sectors to balanced slabs carrying a field."""
    layout = setup.layout
    blocked = np.asarray(layout.points, dtype=np.float64)
    active = np.asarray(layout.active)
    destination = np.where(active, _slab_owners(blocked, setup.owners), 0).astype(
        np.int32
    )
    moved = recorder.run(
        "communication-migration",
        lambda: layout.migrate(
            destination,
            packet_capacity=setup.local,
            payload={"field": payload, "ids": layout.stable_ids},
        ),
        scope="migrate-sectors-to-slabs",
    )
    new = moved.layout
    new_active = np.asarray(new.active)
    new_ids = np.asarray(new.stable_ids)[new_active]
    intended = dict(
        zip(
            np.asarray(layout.stable_ids)[active].tolist(),
            destination[active].tolist(),
            strict=True,
        )
    )
    slot_owner = (np.arange(new_active.size) // setup.local)[new_active]
    owner_errors = sum(
        intended[identifier] != owner
        for identifier, owner in zip(new_ids.tolist(), slot_owner.tolist(), strict=True)
    )
    field_error = np.abs(
        np.asarray(new.collect(moved.payload["field"]), dtype=np.float64)
        - np.asarray(layout.collect(payload), dtype=np.float64)
    )
    point_error = np.abs(
        np.asarray(new.collect(new.points), dtype=np.float64)
        - setup.points.astype(np.float64)
    )
    return {
        "migration_committed": bool(moved.evidence.committed),
        "migration_epoch_advanced": bool(
            np.all(np.asarray(new.owner_epochs) == np.asarray(layout.owner_epochs) + 1)
        ),
        "migration_ids_preserved": sorted(new_ids.tolist()) == sorted(setup.ids.tolist()),
        "migration_owner_errors": int(owner_errors),
        "migration_payload_error": float(np.max(field_error)),
        "migration_id_payload_errors": int(
            np.count_nonzero(np.asarray(moved.payload["ids"])[new_active] != new_ids)
        ),
        "migration_points_error": float(np.max(point_error)),
        "migration_maximum_packet": int(moved.evidence.maximum_packet),
        "migration_maximum_received": int(moved.evidence.maximum_received),
    }, new


def _requery(
    recorder: PhaseRecorder,
    knn: DistributedNeighborQueryPlan,
    moved: DistributedPointLayout,
    setup: _Setup,
    oracle: tuple[np.ndarray, np.ndarray],
    k: int,
    config: MeshfreeConfig,
    /,
) -> tuple[dict[str, Any], DistributedNeighborResult]:
    requery = recorder.run(
        "search", lambda: knn.query(moved, moved), scope="knn-after-migration"
    )
    after = _knn_metrics(
        requery, moved, setup.ids, oracle, _margin(config.dimension, setup.dtype), k
    )
    return {
        "requery_mismatched_certified_rows": after["mismatched_certified_rows"],
        "requery_incomplete_rows": after["incomplete_rows"],
    }, requery


def _row_relative(error: np.ndarray, scale: np.ndarray, /) -> float:
    """max_i |error_i| / scale_i over rows of the |A||x| magnitude scale.

    A zero-scale row must be exact; any error there is divided by the smallest
    normal float64 (clamped to the largest finite value) so it fails every
    declared bound while remaining JSON-representable.
    """
    floor = np.finfo(np.float64)
    ratio = np.abs(error) / np.maximum(scale, floor.tiny)
    return float(np.minimum(np.max(ratio, initial=0.0), floor.max))


def _host_matrix(operator: SparseCoordinateOperator, /) -> sp.csr_matrix:
    """Host float64 CSR of the stored (signed) scalar coefficients."""
    relation = operator.relation
    edge = relation.as_edge_relation() if isinstance(relation, RowRelation) else relation
    if not isinstance(edge, EdgeRelation) or operator.block_shape is not None:
        raise TypeError("Q16 host references need a scalar row or edge relation.")
    valid = np.asarray(edge.valid).reshape((-1,))
    data = np.asarray(operator.coefficients, dtype=np.float64).reshape((-1,))[valid]
    targets = np.asarray(edge.target_indices).reshape((-1,))[valid]
    sources = np.asarray(edge.source_indices).reshape((-1,))[valid]
    return sp.csr_matrix(
        (data, (targets, sources)), shape=(edge.target_size, edge.source_size)
    )


def _action_metrics(
    recorder: PhaseRecorder,
    distributed: DistributedMeshfreeOperator,
    layout: DistributedPointLayout,
    reference: Reference,
    signed: sp.csr_matrix,
    weights: tuple[np.ndarray, np.ndarray],
    compute: np.dtype,
    config: MeshfreeConfig,
    rng: np.random.Generator,
    /,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, np.ndarray]:
    """Distributed apply/transpose/adjoint/JVP/VJP/reductions against single-device actions.

    ``signed`` is the host float64 CSR of A; |A| sets every rounding scale.
    Returns metrics, compiler evidence, the primal input and the collected output.
    """
    count = layout.logical_count
    x = rng.normal(size=(count,)).astype(compute)
    c = rng.normal(size=(count,)).astype(compute)
    source_weights, target_weights = weights
    blocked_x = layout.distribute(x)
    blocked_c = layout.distribute(c)
    applied, compiled = recorder.compiled_action(
        distributed.apply,
        blocked_x,
        budget_bytes=config.resource_bytes,
        repeats=config.repeats,
    )
    applied_host = np.asarray(layout.collect(applied), dtype=np.float64)
    transposed = recorder.run_repeated(
        "warm",
        lambda: distributed.transpose_apply(blocked_c),
        repeats=config.repeats,
        scope="transpose",
    )
    adjoint = recorder.run_repeated(
        "warm",
        lambda: distributed.adjoint_apply(blocked_c),
        repeats=config.repeats,
        scope="adjoint",
    )
    transposed_host = np.asarray(layout.collect(transposed), dtype=np.float64)
    adjoint_host = np.asarray(layout.collect(adjoint), dtype=np.float64)
    x64, c64 = x.astype(np.float64), c.astype(np.float64)
    absolute = abs(signed)
    forward_scale = absolute @ np.abs(x64)
    reverse_scale = absolute.T @ np.abs(c64)
    adjoint_scale = (absolute.T @ (target_weights * np.abs(c64))) / source_weights
    tangent = jax.jvp(distributed.apply, (blocked_x,), (blocked_c,))[1]
    cotangent = jax.vjp(distributed.apply, blocked_x)[1](blocked_c)[0]
    duality_left = applied_host * c64
    duality_right = x64 * transposed_host
    target_blocked = layout.distribute(target_weights.astype(compute))
    weighted_left = float(
        distributed_inner(layout, applied, blocked_c, weights=target_blocked)
    )
    weighted_right = float(
        distributed_inner(
            layout,
            blocked_x,
            adjoint,
            weights=layout.distribute(source_weights.astype(compute)),
        )
    )
    ones = layout.distribute(np.ones((count,), dtype=compute))
    column_total = recorder.run(
        "communication-migration",
        lambda: distributed_sum(layout, distributed.transpose_apply(ones)),
        scope="global-reduction",
    )
    reduced = float(distributed_sum(layout, blocked_x, weights=target_blocked))
    norm = float(distributed_norm(layout, blocked_x, weights=target_blocked))
    terms = target_weights * x64
    metrics = {
        "apply_parity_error": _row_relative(
            applied_host - reference(x, "apply"), forward_scale
        ),
        "transpose_parity_error": _row_relative(
            transposed_host - reference(c, "transpose"), reverse_scale
        ),
        "adjoint_parity_error": _row_relative(
            adjoint_host - reference(c, "adjoint"), adjoint_scale
        ),
        "duality_error": abs(
            math.fsum(duality_left.tolist()) - math.fsum(duality_right.tolist())
        )
        / math.fsum((np.abs(c64) * forward_scale).tolist()),
        "weighted_duality_error": abs(weighted_left - weighted_right)
        / math.fsum((target_weights * np.abs(c64) * forward_scale).tolist()),
        "jvp_parity_error": _row_relative(
            np.asarray(layout.collect(tangent), dtype=np.float64)
            - np.asarray(layout.collect(distributed.apply(blocked_c)), dtype=np.float64),
            absolute @ np.abs(c64),
        ),
        "vjp_parity_error": _row_relative(
            np.asarray(layout.collect(cotangent), dtype=np.float64) - transposed_host,
            reverse_scale,
        ),
        # Laplacian rows sum to zero: the column total is pure cancellation,
        # measured relative to the cancelled magnitudes.
        "conservation_error": abs(float(column_total) - math.fsum(signed.data.tolist()))
        / math.fsum(absolute.data.tolist()),
        "reduction_sum_error": abs(reduced - math.fsum(terms.tolist()))
        / math.fsum(np.abs(terms).tolist()),
        "reduction_norm_error": abs(norm - math.sqrt(math.fsum((terms * x64).tolist())))
        / norm,
        "maximum_row_entries": int(np.max(np.diff(absolute.indptr))),
        "maximum_column_entries": int(np.max(np.diff(absolute.tocsc().indptr))),
        "halo_routes": int(distributed.evidence.routes),
    }
    return metrics, compiled, x, applied_host


def _phase_reasons(recorder: PhaseRecorder, /, *, fit: bool) -> None:
    for phase, reason in (
        ("geometry", "No geometry stage: unit-cube coordinates are the authority"),
        (
            "rank-certificate",
            "Local fit rank evidence is fused into prepare_local_stencils",
        ),
        ("conic", "No conic subproblem"),
        ("ordering-fill", "No sparse factorization"),
        ("hierarchy", "No multilevel hierarchy"),
        ("transfer", "Migration carries fields exactly; no field transfer operator"),
        ("solve", "Distributed actions only; no linear solve"),
        ("numeric-refresh", "Coefficients are rebound after migration (assembly phase)"),
        (
            "epoch-commit",
            "The migration commit is atomic inside DistributedPointLayout.migrate "
            "(measured as communication-migration)",
        ),
        ("restart", "Restart under changed ownership is campaign Q17"),
        ("output", "No output stage"),
        ("end-to-end", "Phases are measured separately"),
    ):
        recorder.unavailable(phase, reason)
    if not fit:
        recorder.unavailable("local-fit", "Neighbor-row coefficients need no local fit")


def _policy(
    profile: DistributedPrecisionProfile, config: MeshfreeConfig, /
) -> MeshfreePrecisionPolicy:
    selected = parse(profile, DistributedPrecisionProfile, "profile")
    match selected:
        case "uniform":
            return MeshfreePrecisionPolicy(geometry_dtype=config.precision)
        case "mixed":
            if config.precision != "float64":
                raise ValueError("The mixed profile needs a float64 (x64) runtime.")
            # float64 geometry/fit/accumulation/residual/certification and
            # halo communication; float32 stored coefficients, fields and output.
            return MeshfreePrecisionPolicy(
                geometry_dtype="float64",
                coefficient_dtype="float32",
                fit_dtype="float64",
                compute_dtype="float32",
                accumulation_dtype="float64",
                communication_dtype="float64",
            )
        case _:
            raise ValueError(f"Unknown precision profile {selected!r}.")


def _record(
    name: str,
    capacity: int,
    seed: int,
    config: MeshfreeConfig,
    setup: _Setup,
    recorder: PhaseRecorder,
    metrics: dict[str, Any],
    /,
) -> dict[str, Any]:
    devices = setup.plan.execution_group.devices
    metrics["owner_count"] = setup.owners
    metrics["process_count"] = jax.process_count()
    return {
        "workload": name,
        "capacity": setup.points.shape[0],
        "requested_capacity": capacity,
        "seed": seed,
        "dimension": config.dimension,
        "status": "measured",
        "local_capacity": setup.local,
        "owner_loads": np.bincount(setup.sectors, minlength=setup.owners).tolist(),
        "devices": {
            "platforms": sorted({device.platform for device in devices}),
            "device_count": len(devices),
            "process_count": jax.process_count(),
            "owner_processes": list(setup.plan.owner_processes),
            "forced_host_devices": "xla_force_host_platform_device_count"
            in os.environ.get("XLA_FLAGS", ""),
        },
        "phases": recorder.record(),
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Accepted workloads


def _gmls_workload(
    name: str,
    profile: DistributedPrecisionProfile,
    capacity: int,
    seed: int,
    config: MeshfreeConfig,
    /,
) -> dict[str, Any]:
    """Open-box GMLS Laplacian over uneven sector owners, migrated to slabs and rebound."""
    policy = _policy(profile, config)
    geometry = np.dtype(policy.dtype("geometry"))
    compute = np.dtype(policy.dtype("compute"))
    output = np.dtype(policy.dtype("output"))
    k = config.neighbors
    reservation = config.check_capacity(capacity)
    recorder = PhaseRecorder()
    setup = _setup(capacity, seed, config, geometry, "open-box")
    reservation += _reservation(setup, k, config)
    metrics, knn, rows, oracle = _relation_phase(recorder, setup, k, config)
    rng = np.random.default_rng(seed + 101)
    metrics.update(_halo_metrics(recorder, rows, setup, rng))
    cloud = jnp.asarray(setup.points)
    # The single-device reference fit publishes weights in the declared fit
    # role; the distributed operator stores them in the coefficient role.
    fit_policy = MeshfreePrecisionPolicy(
        geometry_dtype=geometry,
        fit_dtype=policy.fit_dtype,
        coefficient_dtype=policy.fit_dtype,
    )
    neighborhood = recorder.run(
        "search",
        lambda: MeshfreeNeighborhoodPlan(
            cloud,
            k,
            maximum_candidates=config.candidates,
            target_chunk_size=config.chunk_rows,
            precision=fit_policy,
        ).prepare(),
        scope="single-device-reference-knn",
    )
    laplacian = MeshfreeFunctional(
        tuple(
            tuple(2 if axis == d else 0 for axis in range(config.dimension))
            for d in range(config.dimension)
        ),
        np.ones((config.dimension,), dtype=geometry),
        name="laplacian",
    )
    stencils = recorder.run(
        "local-fit",
        lambda: prepare_local_stencils(
            neighborhood,
            cloud,
            cloud,
            (laplacian,),
            LocalStencilPolicy(
                polynomial_degree=config.degree, chunk_rows=config.chunk_rows
            ),
        ),
        scope="single-device-reference-gmls",
    )
    fitted = MeshfreeOperator(stencils).operator
    count = setup.points.shape[0]
    source_weights = rng.uniform(0.5, 2.0, count)
    target_weights = rng.uniform(0.5, 2.0, count)
    reference = SparseCoordinateOperator(
        fitted.relation,
        policy.cast("coefficient", fitted.coefficients),
        source=ArraySpace(
            (count,),
            dtype=compute,
            pairing=DiagonalPairing(jnp.asarray(source_weights, dtype=compute)),
        ),
        target=ArraySpace(
            (count,),
            dtype=output,
            pairing=DiagonalPairing(jnp.asarray(target_weights, dtype=output)),
        ),
        accumulation_dtype=policy.dtype("accumulation"),
        operator_id="q16-gmls-laplacian",
    )
    distributed = recorder.run(
        "assembly",
        lambda: DistributedMeshfreeOperator.bind(
            reference, setup.layout, setup.layout, halo_capacity=setup.local
        ),
        scope="distributed-bind",
    )

    def single_device(values: np.ndarray, mode: ReferenceMode, /) -> np.ndarray:
        match mode:
            case "apply":
                result = reference.mv(jnp.asarray(values))
            case "transpose":
                result = reference.transpose_mv(jnp.asarray(values))
            case "adjoint":
                result = reference.adjoint_mv(jnp.asarray(values))
            case _:
                raise ValueError(f"Unknown reference mode {mode!r}.")
        return np.asarray(result, dtype=np.float64)

    signed = _host_matrix(reference)
    action, compiled, x, applied = _action_metrics(
        recorder,
        distributed,
        setup.layout,
        single_device,
        signed,
        (source_weights, target_weights),
        compute,
        config,
        rng,
    )
    metrics.update(action)
    absolute = abs(signed)
    if profile == "mixed":
        # Accuracy against the float64 fit: float32 coefficient storage and
        # output each round once (2 eps32 of |A||x|); float64 accumulation adds
        # no k-dependent term. Reported for the declared 4 * eps32 gate.
        exact = _host_matrix(fitted)
        metrics["mixed_accuracy_error"] = _row_relative(
            applied - exact @ x.astype(np.float64),
            absolute @ np.abs(x.astype(np.float64)),
        )
    # Degree >= 2 GMLS reproduces sum(x_i^2) exactly; reported, not a Q16 gate.
    quadratic = setup.layout.distribute(
        np.sum(setup.points.astype(np.float64) ** 2, axis=1).astype(compute)
    )
    metrics["polynomial_laplacian_defect"] = float(
        np.max(
            np.abs(
                np.asarray(
                    setup.layout.collect(distributed.apply(quadratic)), dtype=np.float64
                )
                - 2.0 * config.dimension
            )
        )
    )
    metrics["refused_stencil_rows"] = int(stencils.report.refused_rows)
    observed: dict[MeshfreePrecisionRole, str] = {
        "geometry": np.dtype(setup.layout.points.dtype).name,
        "fit": dict(stencils.report.precision)["fit"],
        "coefficient": np.dtype(distributed.coefficients.dtype).name,
        "compute": np.dtype(quadratic.dtype).name,
        "accumulation": np.dtype(distributed.accumulation_dtype).name,
        # The bound operator exchanges halo values in its accumulation dtype.
        "communication": np.dtype(distributed.accumulation_dtype).name,
        "certification": np.dtype(rows.distance_squared.dtype).name,
        "output": np.dtype(distributed.output_dtype).name,
    }
    metrics["precision_roles_match"] = all(
        policy.dtype(parse(role, MeshfreePrecisionRole, "precision role")) == dtype
        for role, dtype in observed.items()
    )
    payload = setup.layout.distribute(rng.normal(size=(count,)).astype(compute))
    migration, moved = _migration(recorder, setup, payload)
    metrics.update(migration)
    requery, _ = _requery(recorder, knn, moved, setup, oracle, k, config)
    metrics.update(requery)
    rebound = recorder.run(
        "assembly",
        lambda: DistributedMeshfreeOperator.bind(
            reference, moved, moved, halo_capacity=setup.local
        ),
        scope="rebind-after-migration",
    )
    replay = np.asarray(
        moved.collect(rebound.apply(moved.distribute(x))), dtype=np.float64
    )
    metrics["rebind_parity_error"] = _row_relative(
        replay - applied, absolute @ np.abs(x.astype(np.float64))
    )
    _phase_reasons(recorder, fit=True)
    record = _record(name, capacity, seed, config, setup, recorder, metrics)
    record.update(
        {
            "compiler": compiled,
            "retained_bytes": logical_array_bytes(
                (setup.layout, distributed, rows, stencils, moved, rebound)
            ),
            "reserved_working_set_bytes": reservation,
            "precision_policy": {
                "profile": profile,
                "policy_id": policy.policy_id,
                "roles": dict(policy.roles),
                "observed": observed,
            },
            "oracle_provenance": "scipy.spatial.cKDTree float64 kNN and pair enumeration; "
            "single-device SparseCoordinateOperator mv/transpose_mv/adjoint_mv; "
            "math.fsum host sums; " + _FORCED_SCOPE,
            "consumer": "phydrax.discretization.meshfree.DistributedMeshfreeOperator.bind",
        }
    )
    return record


def measure_distributed_gmls_open(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """Uniform-precision (config.precision) open-box distributed GMLS Laplacian."""
    return _gmls_workload("distributed-gmls-open", "uniform", capacity, seed, config)


def measure_distributed_gmls_open_mixed(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """Mixed float32-storage/float64-accumulation open-box distributed GMLS Laplacian."""
    return _gmls_workload("distributed-gmls-open-mixed", "mixed", capacity, seed, config)


def measure_distributed_smoother_periodic(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """All-axes periodic kNN rows bound without a host step and checked on host CSR."""
    policy = _policy("uniform", config)
    dtype = np.dtype(policy.dtype("geometry"))
    k = config.neighbors
    recorder = PhaseRecorder()
    setup = _setup(capacity, seed, config, dtype, "periodic-all-axes")
    reservation = _reservation(setup, k, config)
    metrics, knn, rows, oracle = _relation_phase(recorder, setup, k, config)
    rng = np.random.default_rng(seed + 211)
    metrics.update(_halo_metrics(recorder, rows, setup, rng))
    # Inverse-quadratic smoother on the kNN scale h^2 ~ N^(-2/d).
    inverse_scale = float(capacity) ** (2.0 / config.dimension)

    def coefficients(result: DistributedNeighborResult, /) -> Array:
        values = 1.0 / (1.0 + inverse_scale * result.distance_squared)
        return policy.cast("coefficient", jnp.where(result.valid, values, 0.0))

    def bind(
        result: DistributedNeighborResult, layout: DistributedPointLayout, /
    ) -> DistributedMeshfreeOperator:
        return DistributedMeshfreeOperator.from_neighbor_rows(
            result,
            coefficients(result),
            layout,
            layout,
            halo_capacity=setup.local,
            operator_id="q16-periodic-inverse-quadratic-smoother",
        )

    distributed = recorder.run(
        "assembly", lambda: bind(rows, setup.layout), scope="bind-neighbor-rows"
    )
    # Independent host CSR from the collected rows: logical source = (id - 11) / 7.
    found = np.asarray(setup.layout.collect(rows.source_stable_ids))
    valid = np.asarray(setup.layout.collect(rows.valid))
    squared = np.asarray(setup.layout.collect(rows.distance_squared), dtype=np.float64)
    count = setup.points.shape[0]
    signed = sp.csr_matrix(
        (
            (1.0 / (1.0 + inverse_scale * squared))[valid],
            ((found - 11) // 7)[valid],
            np.concatenate(([0], np.cumsum(valid.sum(axis=1)))),
        ),
        shape=(count, count),
    )

    def host_reference(values: np.ndarray, mode: ReferenceMode, /) -> np.ndarray:
        vector = values.astype(np.float64)
        match mode:
            case "apply":
                return signed @ vector
            case "transpose" | "adjoint":
                # Neighbor-row operators carry the Euclidean pairing.
                return signed.T @ vector
            case _:
                raise ValueError(f"Unknown reference mode {mode!r}.")

    ones = np.ones((count,), dtype=np.float64)
    action, compiled, x, applied = _action_metrics(
        recorder,
        distributed,
        setup.layout,
        host_reference,
        signed,
        (ones, ones),
        dtype,
        config,
        rng,
    )
    metrics.update(action)
    payload = setup.layout.distribute(rng.normal(size=(count,)).astype(dtype))
    migration, moved = _migration(recorder, setup, payload)
    metrics.update(migration)
    requery_metrics, requery = _requery(recorder, knn, moved, setup, oracle, k, config)
    metrics.update(requery_metrics)
    rebound = recorder.run(
        "assembly", lambda: bind(requery, moved), scope="rebind-after-migration"
    )
    replay = np.asarray(
        moved.collect(rebound.apply(moved.distribute(x))), dtype=np.float64
    )
    # Only rows COMPLETE in both bindings carry routes in both; refused rows
    # are counted by the knn-status/requery-status gates instead.
    both = (
        np.asarray(setup.layout.collect(rows.status))
        == DistributedRelationStatus.COMPLETE
    ) & (np.asarray(moved.collect(requery.status)) == DistributedRelationStatus.COMPLETE)
    metrics["rebind_compared_rows"] = int(np.count_nonzero(both))
    metrics["rebind_parity_error"] = _row_relative(
        (replay - applied)[both], (abs(signed) @ np.abs(x.astype(np.float64)))[both]
    )
    observed: dict[MeshfreePrecisionRole, str] = {
        "geometry": np.dtype(setup.layout.points.dtype).name,
        "coefficient": np.dtype(distributed.coefficients.dtype).name,
        "accumulation": np.dtype(distributed.accumulation_dtype).name,
        "certification": np.dtype(rows.distance_squared.dtype).name,
        "output": np.dtype(distributed.output_dtype).name,
    }
    metrics["precision_roles_match"] = all(
        policy.dtype(parse(role, MeshfreePrecisionRole, "precision role")) == value
        for role, value in observed.items()
    )
    _phase_reasons(recorder, fit=False)
    record = _record(
        "distributed-smoother-periodic", capacity, seed, config, setup, recorder, metrics
    )
    record.update(
        {
            "compiler": compiled,
            "retained_bytes": logical_array_bytes(
                (setup.layout, distributed, rows, moved, rebound)
            ),
            "reserved_working_set_bytes": reservation,
            "precision_policy": {
                "profile": "uniform",
                "policy_id": policy.policy_id,
                "roles": dict(policy.roles),
                "observed": observed,
            },
            "oracle_provenance": "scipy.spatial.cKDTree(boxsize=1) float64 minimum-image kNN "
            "and pairs; host scipy CSR of the collected rows; math.fsum host sums; "
            + _FORCED_SCOPE,
            "consumer": "phydrax.discretization.meshfree.DistributedMeshfreeOperator.from_neighbor_rows",
        }
    )
    return record


# ---------------------------------------------------------------------------
# Expected-refusal workload


def _refused(operation: Callable[[], object], phrase: str, /) -> bool:
    """Whether ``operation`` raises the documented ``ValueError`` naming ``phrase``.

    Any other exception, or a ValueError for a different reason, propagates.
    """
    try:
        operation()
    except ValueError as error:
        if phrase not in str(error):
            raise
        return True
    return False


def _status_refusal(
    result: DistributedNeighborResult,
    layout: DistributedPointLayout,
    status: DistributedRelationStatus,
    /,
) -> tuple[bool, bool, np.ndarray]:
    """(declared status present and query unsuccessful, refused rows carry no routes, mask)."""
    codes = np.asarray(layout.collect(result.status))
    refused = codes == status
    valid = np.asarray(layout.collect(result.valid))
    return (
        bool(refused.any()) and not bool(result.evidence.successful),
        not bool(valid[refused].any()),
        refused,
    )


def _relation_refusals(
    recorder: PhaseRecorder, setup: _Setup, k: int, config: MeshfreeConfig, /
) -> tuple[dict[str, Any], DistributedNeighborResult]:
    margin = _margin(config.dimension, setup.dtype)
    oracle = _knn_oracle(setup.points.astype(np.float64), k, setup.periodic)
    complete = recorder.run(
        "search",
        lambda: DistributedNeighborQueryPlan(
            setup.plan, setup.plan, k, halo_capacity=setup.local
        ).query(setup.layout, setup.layout),
        scope="complete-halo",
    )
    load = int(complete.evidence.maximum_halo_load)
    tight = recorder.run(
        "search",
        lambda: DistributedNeighborQueryPlan(
            setup.plan, setup.plan, k, halo_capacity=load
        ).query(setup.layout, setup.layout),
        scope="tight-halo",
    )
    overflow = recorder.run(
        "search",
        lambda: DistributedNeighborQueryPlan(
            setup.plan, setup.plan, k, halo_capacity=max(load - 1, 1)
        ).query(setup.layout, setup.layout),
        scope="halo-overflow",
    )
    refused, invalid, mask = _status_refusal(
        overflow, setup.layout, DistributedRelationStatus.HALO_OVERFLOW
    )
    survivors = _knn_metrics(overflow, setup.layout, setup.ids, oracle, margin, k)
    owners = recorder.run(
        "search",
        lambda: DistributedNeighborQueryPlan(
            setup.plan, setup.plan, k, maximum_remote_owners=1, halo_capacity=setup.local
        ).query(setup.layout, setup.layout),
        scope="owner-overflow",
    )
    owner_refused, owner_invalid, _ = _status_refusal(
        owners, setup.layout, DistributedRelationStatus.OWNER_OVERFLOW
    )
    volume = math.pi if config.dimension == 2 else 4.0 * math.pi / 3.0
    radius = (_RADIUS_MEAN_NEIGHBORS / (setup.points.shape[0] * volume)) ** (
        1.0 / config.dimension
    )
    narrow = recorder.run(
        "search",
        lambda: DistributedRadiusQueryPlan(
            setup.plan, setup.plan, radius, 2, halo_capacity=setup.local
        ).query(setup.layout, setup.layout),
        scope="row-overflow",
    )
    row_refused, row_invalid, _ = _status_refusal(
        narrow, setup.layout, DistributedRelationStatus.ROW_OVERFLOW
    )
    stale = eqx.tree_at(
        lambda layout: layout.owner_epochs,
        setup.layout,
        setup.layout.owner_epochs.at[setup.owners - 1].add(-1),
    )
    missing = recorder.run(
        "search",
        lambda: DistributedNeighborQueryPlan(stale.plan, stale.plan, k).query(
            stale, stale
        ),
        scope="stale-owner-epoch",
    )
    missing_codes = np.asarray(stale.collect(missing.status))
    return {
        "maximum_halo_load": load,
        "halo_tight_successful": bool(tight.evidence.successful),
        "halo_overflow_refused": refused,
        "halo_overflow_rows": int(np.count_nonzero(mask)),
        "halo_overflow_refused_rows_invalid": invalid,
        "halo_overflow_surviving_mismatches": survivors["mismatched_certified_rows"],
        "owner_overflow_refused": owner_refused
        and int(owners.evidence.maximum_required_owners) > 1,
        "owner_overflow_refused_rows_invalid": owner_invalid,
        "row_overflow_refused": row_refused,
        "row_overflow_refused_rows_invalid": row_invalid,
        "missing_owner_refused": bool(
            np.all(missing_codes == DistributedRelationStatus.MISSING_OWNER)
        )
        and not bool(missing.evidence.owners_current)
        and not bool(np.asarray(missing.valid).any()),
    }, complete


def measure_distributed_refusal(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """Declared distributed capacity, ownership and precision refusals."""
    dtype = np.dtype(config.precision)
    k = config.neighbors
    recorder = PhaseRecorder()
    setup = _setup(capacity, seed, config, dtype, "open-box")
    reservation = _reservation(setup, k, config)
    metrics, complete = _relation_refusals(recorder, setup, k, config)
    # Local capacity at ingress: the exact sector load is admitted, one less refused.
    load = int(np.bincount(setup.sectors, minlength=setup.owners).max())
    address = setup.plan.address_plan
    group = setup.plan.execution_group
    admitted = DistributedPointLayout.from_global(
        DistributedOwnershipPlan(address, group, load), setup.points, setup.sectors
    )
    metrics["local_capacity_tight_admitted"] = admitted.logical_count == capacity
    metrics["local_capacity_refused"] = load > 1 and _refused(
        lambda: DistributedPointLayout.from_global(
            DistributedOwnershipPlan(address, group, load - 1),
            setup.points,
            setup.sectors,
        ),
        "exceeds local_capacity",
    )
    layout = setup.layout
    before = {
        name: np.asarray(value)
        for name, value in (
            ("points", layout.points),
            ("ids", layout.stable_ids),
            ("active", layout.active),
            ("epochs", layout.owner_epochs),
        )
    }
    field = layout.distribute(np.arange(capacity, dtype=dtype))

    def unchanged(result: DistributedMigrationResult, /) -> bool:
        moved = result.layout
        return all(
            np.array_equal(np.asarray(value), before[name])
            for name, value in (
                ("points", moved.points),
                ("ids", moved.stable_ids),
                ("active", moved.active),
                ("epochs", moved.owner_epochs),
            )
        ) and np.array_equal(np.asarray(result.payload), np.asarray(field))

    crowded = recorder.run(
        "communication-migration",
        lambda: layout.migrate(
            np.zeros((setup.plan.total_capacity,), dtype=np.int32),
            packet_capacity=setup.local,
            payload=field,
        ),
        scope="receive-overflow",
    )
    metrics["migration_receive_overflow_refused"] = (
        not bool(crowded.evidence.committed)
        and int(crowded.evidence.maximum_received) > setup.local
    )
    metrics["migration_receive_overflow_rolled_back"] = unchanged(crowded)
    blocked = np.asarray(layout.points, dtype=np.float64)
    slabs = np.where(np.asarray(layout.active), _slab_owners(blocked, setup.owners), 0)
    squeezed = recorder.run(
        "communication-migration",
        lambda: layout.migrate(slabs.astype(np.int32), packet_capacity=1, payload=field),
        scope="packet-overflow",
    )
    metrics["migration_packet_overflow_refused"] = (
        not bool(squeezed.evidence.committed)
        and int(squeezed.evidence.maximum_packet) > 1
    )
    metrics["migration_packet_overflow_rolled_back"] = unchanged(squeezed)
    metrics["incomplete_halo_bind_refused"] = _refused(
        lambda: DistributedMeshfreeOperator.from_neighbor_rows(
            complete,
            jnp.where(complete.valid, 1.0, 0.0).astype(dtype),
            layout,
            layout,
            halo_capacity=1,
            operator_id="q16-incomplete-halo",
        ),
        "halo is incomplete",
    )
    metrics["unsupported_float16_refused"] = _refused(
        lambda: MeshfreePrecisionPolicy(geometry_dtype=np.float16), "geometry_dtype"
    )
    metrics["unsupported_bfloat16_refused"] = _refused(
        lambda: MeshfreePrecisionPolicy(geometry_dtype=jnp.bfloat16), "geometry_dtype"
    )
    # The runtime keeps x64 (64-bit Morton codes and stable IDs); data precision
    # narrowing is refused by the precision policy itself.
    metrics["certification_narrowing_refused"] = _refused(
        lambda: MeshfreePrecisionPolicy(
            geometry_dtype="float64", certification_dtype="float32"
        ),
        "Certification precision cannot be narrower",
    )
    metrics["certification_downcast_refused"] = _refused(
        lambda: MeshfreePrecisionPolicy(geometry_dtype="float32").cast(
            "certification", np.zeros((3,), dtype=np.float64)
        ),
        "Refusing to downcast",
    )
    for phase, reason in (
        ("local-fit", "Refusal workload binds no local fit"),
        ("assembly", "The incomplete-halo binding is refused before assembly"),
    ):
        recorder.unavailable(phase, reason)
    _phase_reasons(recorder, fit=True)
    record = _record(
        "distributed-refusal", capacity, seed, config, setup, recorder, metrics
    )
    record.update(
        {
            "retained_bytes": logical_array_bytes((setup.layout, complete)),
            "reserved_working_set_bytes": reservation,
            "oracle_provenance": "documented DistributedRelationStatus codes, "
            "DistributedMigrationEvidence rollback, documented ValueError refusals; "
            "cKDTree survivors oracle; " + _FORCED_SCOPE,
            "consumer": "phydrax.discretization.spatial.Distributed* and "
            "phydrax.discretization.meshfree.MeshfreePrecisionPolicy",
        }
    )
    return record


DISTRIBUTED_WORKLOADS: dict[str, Callable[[int, int, MeshfreeConfig], dict[str, Any]]] = {
    "distributed-gmls-open": measure_distributed_gmls_open,
    "distributed-gmls-open-mixed": measure_distributed_gmls_open_mixed,
    "distributed-smoother-periodic": measure_distributed_smoother_periodic,
    "distributed-refusal": measure_distributed_refusal,
}
