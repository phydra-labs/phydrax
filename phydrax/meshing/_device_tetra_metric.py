#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Compiled device candidate evaluation for 3D tetrahedral metric adaptation.

The device evaluates the full tensor metric on every live edge. Its deterministically
ordered, conflict-free split batches cross an owning-host exact-predicate barrier
before the native tetra owner mutates. Only that split-batch execution is
route-specific: the phase schedule, native owner, and stable target-ID numbering are
the shared ``_tetra_metric._execute_native_state``, so the device route realizes the
native ``execute_tetra_metric_adaptation`` target exactly. All accepted edits remain
provisional in a source-local native owner until the complete adaptation is
assembled and accepted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import final, NamedTuple, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._fingerprint import canonical_fingerprint
from .._geometry_predicates import orient3d, PredicateMode, PredicateSign
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..typing import checked
from ._metric import metric_edge_lengths
from ._tetra_metric import (
    _current_edge_controls,
    _disjoint_candidates,
    _execute_native_state,
    _implicit_metric_source,
    _metric_midpoint,
    _NativeState,
    _periodic_edge_orbit_groups,
    _prepare_native_state,
    _topology,
    _try_split_batch,
    TetraMetricEvidence,
    TetraMetricStatus,
)


if TYPE_CHECKING:
    from ._adaptation import _RouteOutcome, PreparedMeshAdaptation


_UPPER = np.sqrt(2.0)
_TETRA_EDGE_CORNERS = np.asarray(tuple(combinations(range(4), 2)), dtype=np.int32)
_TETRA_EDGE_CORNERS.setflags(write=False)


@final
class DeviceTetraMetricEvidence(StrictModule, NonTrainableState):
    """Host-visible device-candidate and exact-batch evidence."""

    metric_evidence: TetraMetricEvidence | None
    status: TetraMetricStatus = eqx.field(static=True)
    evaluated_edges: int = eqx.field(static=True)
    long_edge_candidates: int = eqx.field(static=True)
    independent_split_candidates: int = eqx.field(static=True)
    candidate_edge_keys: tuple[tuple[int, int], ...] = eqx.field(static=True)
    exact_predicates_resolved: int = eqx.field(static=True)
    host_barriers: int = eqx.field(static=True)
    committed: bool = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        metric_evidence: TetraMetricEvidence | None,
        status: TetraMetricStatus,
        evaluated_edges: int,
        long_edge_candidates: int,
        independent_split_candidates: int,
        candidate_edge_keys: tuple[tuple[int, int], ...],
        exact_predicates_resolved: int,
        host_barriers: int,
        committed: bool,
        /,
    ) -> None:
        if metric_evidence is not None and not isinstance(
            metric_evidence, TetraMetricEvidence
        ):
            raise TypeError("metric_evidence must be TetraMetricEvidence or None.")
        if not isinstance(status, TetraMetricStatus):
            raise TypeError("status must be TetraMetricStatus.")
        counts = (
            evaluated_edges,
            long_edge_candidates,
            independent_split_candidates,
            exact_predicates_resolved,
            host_barriers,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counts
        ):
            raise ValueError("Device tetra metric counts must be nonnegative integers.")
        if not isinstance(committed, bool):
            raise TypeError("committed must be bool.")
        if metric_evidence is not None and status is not metric_evidence.status:
            raise ValueError("Device and native tetra metric statuses must agree.")
        if metric_evidence is None and committed:
            raise ValueError("A rejected device candidate batch cannot be committed.")
        if len(candidate_edge_keys) != independent_split_candidates:
            raise ValueError(
                "Candidate edge keys must align with selected split candidates."
            )
        if any(
            len(key) != 2
            or any(
                isinstance(identifier, bool) or not isinstance(identifier, int)
                for identifier in key
            )
            or key[0] >= key[1]
            for key in candidate_edge_keys
        ):
            raise ValueError(
                "Candidate edge keys must be canonical increasing integer pairs."
            )
        self.metric_evidence = metric_evidence
        self.status = status
        self.evaluated_edges = evaluated_edges
        self.long_edge_candidates = long_edge_candidates
        self.independent_split_candidates = independent_split_candidates
        self.candidate_edge_keys = candidate_edge_keys
        self.exact_predicates_resolved = exact_predicates_resolved
        self.host_barriers = host_barriers
        self.committed = committed
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "device-tetra-metric-evidence",
                "metric_evidence": (
                    None if metric_evidence is None else metric_evidence.evidence_id
                ),
                "status": status.value,
                "counts": counts,
                "candidate_edge_keys": candidate_edge_keys,
                "committed": committed,
            }
        )

    @property
    def converged(self) -> bool:
        """Whether the native metric criterion was fully realized."""
        return self.status is TetraMetricStatus.COMPLETE


class _DeviceSplitPlan(NamedTuple):
    groups: tuple[tuple[tuple[int, int], ...], ...]
    edge_keys: tuple[tuple[int, int], ...]
    edge_rows: np.ndarray
    points: np.ndarray
    metric: np.ndarray
    cells: np.ndarray
    cavities: tuple[tuple[int, ...], ...]
    lengths: np.ndarray
    long_edge_candidates: int


def _compiled_edge_lengths(points: Array, metric: Array, edges: Array, /) -> Array:
    return metric_edge_lengths(metric, points, edges)


_compiled_candidates = eqx.filter_jit(_compiled_edge_lengths)


def _device_split_plan(
    state: _NativeState, blocked: set[tuple[int, int]], /
) -> _DeviceSplitPlan:
    """Evaluate all live edges on device and select canonical independent cavities."""
    snapshot = state.snapshot
    topology = _topology(snapshot)
    edge_keys = tuple(sorted(topology.edge_cells))
    points = np.asarray(snapshot.points, dtype=np.float64)
    vertex_ids = np.asarray(snapshot.vertex_ids, dtype=np.int64)
    metric = np.asarray(
        [state.metric_by_vertex[int(identifier)] for identifier in vertex_ids],
        dtype=np.float64,
    )
    edge_rows = np.asarray(
        [
            [topology.vertex_row[first], topology.vertex_row[second]]
            for first, second in edge_keys
        ],
        dtype=np.int32,
    ).reshape((-1, 2))
    lengths_device = _compiled_candidates(
        jnp.asarray(points, dtype=jnp.float64),
        jnp.asarray(metric, dtype=jnp.float64),
        jnp.asarray(edge_rows, dtype=jnp.int32),
    )
    lengths = np.asarray(jax.device_get(lengths_device), dtype=np.float64)
    _, protected = _current_edge_controls(state, topology)
    cell_ids = np.asarray(snapshot.cell_ids, dtype=np.int64)
    eligible: dict[tuple[int, int], tuple[float, frozenset[int]]] = {}
    for edge, length in zip(edge_keys, lengths.tolist(), strict=True):
        if edge in blocked or protected[edge] or length <= _UPPER:
            continue
        eligible[edge] = (
            float(length),
            frozenset(int(cell_ids[row]) for row in topology.edge_cells[edge]),
        )
    candidates = []
    for orbit in _periodic_edge_orbit_groups(state):
        if not orbit or any(edge not in eligible for edge in orbit):
            continue
        cavities = [eligible[edge][1] for edge in orbit]
        cavity = frozenset().union(*cavities)
        if sum(map(len, cavities)) != len(cavity):
            continue
        representative = min(
            tuple(sorted(state.vertex_target_ids[vertex] for vertex in edge))
            for edge in orbit
        )
        priority = (-min(eligible[edge][0] for edge in orbit), representative)
        candidates.append((priority, orbit, cavity))
    groups = tuple(
        tuple(tuple(edge) for edge in group) for group in _disjoint_candidates(candidates)
    )
    cavities_by_edge = tuple(topology.edge_cells[edge] for edge in edge_keys)
    return _DeviceSplitPlan(
        groups,
        edge_keys,
        edge_rows,
        points,
        metric,
        np.asarray(snapshot.tetrahedra, dtype=np.int64),
        cavities_by_edge,
        lengths,
        len(eligible),
    )


def _resolve_exact_split_batch(
    points: np.ndarray,
    metric: np.ndarray,
    cells: np.ndarray,
    edge_rows: np.ndarray,
    cavities: tuple[tuple[int, ...], ...],
    selected: tuple[int, ...],
    /,
) -> tuple[bool, int]:
    """Resolve every proposed child orientation exactly in one host predicate call."""
    child_corners: list[np.ndarray] = []
    for edge_row in selected:
        first, second = edge_rows[edge_row]
        _, midpoint = _metric_midpoint(
            points[first], points[second], metric[first], metric[second]
        )
        for cell_row in cavities[edge_row]:
            parent = cells[cell_row]
            first_local = int(np.flatnonzero(parent == first)[0])
            second_local = int(np.flatnonzero(parent == second)[0])
            corners = points[parent]
            first_child = corners.copy()
            second_child = corners.copy()
            first_child[second_local] = midpoint
            second_child[first_local] = midpoint
            child_corners.extend((first_child, second_child))
    if not child_corners:
        return True, 0
    children = np.asarray(child_corners, dtype=np.float64)
    result = orient3d(
        children[:, 0],
        children[:, 1],
        children[:, 2],
        children[:, 3],
        mode=PredicateMode.EXACT,
    )
    signs = np.asarray(result.signs, dtype=np.int8)
    certain = np.asarray(result.certain, dtype=np.bool_)
    valid = np.all(certain) and np.all(signs != int(PredicateSign.ZERO))
    return bool(valid), children.shape[0]


class _RejectedDeviceBatch(Exception):
    """The exact host barrier refused a device-selected split batch."""


@dataclass(slots=True)
class _DeviceSplitBatches:
    """Route-specific split execution of the shared schedule, with its evidence.

    Each call mirrors ``_split_phase(..., one_batch=True)``: it applies at most one
    disjoint split batch, re-planning with refused edges blocked until a batch
    applies or no candidate remains.
    """

    evaluated: bool = False
    evaluated_edges: int = 0
    long_edge_candidates: int = 0
    candidate_edge_keys: tuple[tuple[int, int], ...] = field(default=())
    exact_predicates_resolved: int = 0
    host_barriers: int = 0

    def __call__(self, state: _NativeState, counters: dict[str, int], /) -> int:
        blocked: set[tuple[int, int]] = set()
        while True:
            plan = _device_split_plan(state, blocked)
            self.host_barriers += 1
            if not self.evaluated:
                self.evaluated = True
                self.evaluated_edges = len(plan.edge_keys)
                self.long_edge_candidates = plan.long_edge_candidates
                self.candidate_edge_keys = tuple(
                    edge for group in plan.groups for edge in group
                )
            if not plan.groups:
                return 0
            row_by_edge = {edge: row for row, edge in enumerate(plan.edge_keys)}
            selected = tuple(row_by_edge[edge] for group in plan.groups for edge in group)
            exact_valid, resolved = _resolve_exact_split_batch(
                plan.points,
                plan.metric,
                plan.cells,
                plan.edge_rows,
                plan.cavities,
                selected,
            )
            self.exact_predicates_resolved += resolved
            if not exact_valid:
                raise _RejectedDeviceBatch
            changed = _try_split_batch(state, list(plan.groups), counters, blocked)
            if changed:
                return changed

    def evidence(
        self,
        metric_evidence: TetraMetricEvidence | None,
        status: TetraMetricStatus,
        committed: bool,
        /,
    ) -> DeviceTetraMetricEvidence:
        return DeviceTetraMetricEvidence(
            metric_evidence,
            status,
            self.evaluated_edges,
            self.long_edge_candidates,
            len(self.candidate_edge_keys),
            self.candidate_edge_keys,
            self.exact_predicates_resolved,
            self.host_barriers,
            committed,
        )


def _execute_device_tetra_metric_route(
    prepared: PreparedMeshAdaptation, /
) -> _RouteOutcome:
    """Run compiled edge evaluation, host predicate batches, and native tetra edits."""
    from ._adaptation import (
        _finalize_native,
        _RouteOutcome,
        _target_metric,
        _unchanged,
        MeshAdaptationStatus,
        MeshTransitionKind,
        MetricMeshAdaptation,
        PreparedMeshAdaptation,
    )

    if not isinstance(prepared, PreparedMeshAdaptation):
        raise TypeError("prepared must be PreparedMeshAdaptation.")
    if not isinstance(prepared.request, MetricMeshAdaptation):
        raise TypeError("DEVICE_METRIC_3D requires MetricMeshAdaptation.")
    source = prepared.source
    constraints = prepared.constraints
    state = _prepare_native_state(
        source.mesh,
        source.geometry,
        source.coordinate_contract,
        np.asarray(constraints.metric_values, dtype=np.float64),
        cell_classes=constraints.cell_classes,
        facet_classes=constraints.facet_classes,
        edge_classes=constraints.edge_classes,
        protected_edges=constraints.protected_edge_mask,
        fixed_vertices=constraints.fixed_vertex_mask,
        implicit_source=_implicit_metric_source(source),
    )
    batches = _DeviceSplitBatches()
    try:
        outcome = _execute_native_state(
            state,
            PredicateMode.EXACT,
            topology_operations=True,
            relocation=prepared.policy.relocation,
            split_batch=batches,
        )
    except _RejectedDeviceBatch:
        return _RouteOutcome(
            MeshAdaptationStatus.NUMERICAL_STOP,
            source,
            None,
            None,
            None,
            None,
            None,
            batches.evidence(None, TetraMetricStatus.NUMERICAL_STOP, False),
            None,
        )
    metric_evidence = outcome.evidence
    status = metric_evidence.status
    adaptation_status = {
        TetraMetricStatus.COMPLETE: MeshAdaptationStatus.COMPLETE,
        TetraMetricStatus.STALLED: MeshAdaptationStatus.STALLED,
        TetraMetricStatus.NUMERICAL_STOP: MeshAdaptationStatus.NUMERICAL_STOP,
    }[status]
    applied = (
        metric_evidence.splits
        + metric_evidence.collapses
        + metric_evidence.flips23
        + metric_evidence.flips32
        + metric_evidence.ring_reconnections
        + metric_evidence.relocations
    )
    evidence = batches.evidence(metric_evidence, status, applied > 0)
    if applied == 0:
        return _unchanged(prepared, evidence, None, status=adaptation_status)
    accepted = _finalize_native(
        prepared,
        outcome.edit,
        MeshTransitionKind.REMESH,
        conservative=False,
        exact_source_vertices=outcome.exact_source_vertices,
    )
    return _RouteOutcome(
        adaptation_status,
        accepted.target,
        accepted.transition,
        accepted.lineage,
        accepted.stencil,
        accepted.transfer,
        _target_metric(prepared.request.metric, accepted.target.mesh, outcome.metric),
        evidence,
        None,
    )


__all__ = ["DeviceTetraMetricEvidence"]
