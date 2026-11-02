#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Compiled planar metric adaptation (``DEVICE_METRIC_2D``).

The compiled route performs true SPD-metric edge splits, admissible collapses,
quality-improving flips, and relocation with filtered device predicates and
deterministic cavity-independent batches. It continues passes until the unit
mesh is reached, a full pass makes no change, or a real geometry/status failure
occurs. Static slot and vertex-star workspaces grow at host barriers and retry
the unchanged operation; they never truncate scientific candidates.

`prepare_device_metric_adaptation` binds a certified planar source, its metric,
protected strata, organization, and lineage workspace. Target construction,
accepted geometry, lineage, transfer, and certification remain with the
canonical host acceptance owner.

`commit_device_metric_adaptation` performs one device-to-host transfer and
assembles through the canonical host acceptance path.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from fractions import Fraction
from typing import Any, final, NamedTuple, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._geometry_predicates import (
    orient2d,
    PredicateMode,
    PredicateSign,
    resolve_host_predicate_mode,
)
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._adaptive_simplex import (
    _count,
    _FAILURES,
    _INDEX_LIMIT,
    _select,
    AdaptiveSimplexStatus,
    masked_simplex_facet_neighbors,
    masked_simplex_signature,
    MaskedSimplexMesh,
)
from ..discretization._periodic_exact import (
    identification_orders,
    reduce_periodic_exponents,
)
from ..linalg import hermitian_exp, hermitian_log
from ..typing import checked
from ._device_status import DeviceEpoch, require_committed_status
from ._lineage import MeshTransitionKind
from ._local_metric import (
    _assemble,
    _boolean_array,
    _codes,
    _COLLAPSE_LIMIT,
    _COLLAPSED,
    _CORNER,
    _CURVE,
    _DEVIATION_QUANTUM,
    _distinct,
    _IMPROVEMENT,
    _initial_state,
    _integer_array,
    _INTERIOR,
    _lookup,
    _LOWER,
    _measured,
    _OPTIMAL_BLOCK,
    _OPTIMAL_LATTICE,
    _periodic_state_context,
    _REFINED,
    _RELOCATED,
    _Source,
    _State,
    _SWAPPED,
    _topology as _host_topology,
    _UPPER,
    _validate_classification,
    _validate_mesh,
)
from ._metric import _constant_preserving_metric_mean, metric_edge_lengths


if TYPE_CHECKING:
    from ._adaptation import (
        _RouteOutcome,
        MeshAdaptationPolicy,
        MeshAdaptationResult,
        MetricMeshAdaptation,
        PreparedMeshAdaptation,
        RelocationMeshAdaptation,
    )
    from ._result import CellMeshingResult


# Edge-lineage records per (vertex + cell) slot: a planar triangulation has at
# most V + C edges; merged collapse records rarely exceed that bound.
_RECORDS_PER_SLOT = 2
# (cell, source cell) ancestry pairs per cell slot.
_PAIRS_PER_CELL = 8
# Collapse round modes, also the escape attempt of a pass: regular passes, then
# the stall-escape attempts in order (see ``_adapted``).
_REGULAR, _OPTIMAL, _ENDPOINT = 0, 1, 2

# Operation statuses, worst last; the minimum over the two collapse directions
# of one edge is its tallied outcome.
_OK, _UNCERTAIN, _CAVITY, _INVALID, _INADMISSIBLE = 0, 1, 2, 3, 4
_CODE_SENTINEL = np.iinfo(np.int64).max
_RANK_SENTINEL = np.iinfo(np.int32).max
_NEXT = np.asarray((1, 2, 0), dtype=np.int32)
_PREVIOUS = np.asarray((2, 0, 1), dtype=np.int32)

# cursors: allocated slots, live records/pairs, and next lifted/quotient IDs.
(
    _VERTICES,
    _CELLS,
    _RECORDS,
    _PAIRS,
    _NEXT_VERTEX,
    _NEXT_CELL,
    _NEXT_QUOTIENT_VERTEX,
) = range(7)
_TOPOLOGY, _RELOCATION = range(2)
# counters: totals since preparation.
(
    _PASSES,
    _SPLITS,
    _COLLAPSES,
    _FLIPS,
    _RELOCATIONS,
    _REJECTED_UNCERTAIN,
    _REJECTED_INVALID,
    _REJECTED_CAVITY,
) = range(8)
# flags: union of applied status flags, then convergence and stall of the last call.
_STATUS, _CONVERGED, _STALLED = range(3)


# ------------------------------------------------------------------ public records


@final
class DeviceMetricLayout(StrictModule, NonTrainableState):
    """Static workspace identity, including the exact current orbit width.

    Slot and orbit workspaces are host-prepared from represented topology and
    grow only when an actual finite workset requires more storage.
    """

    vertex_capacity: int = eqx.field(static=True)
    cell_capacity: int = eqx.field(static=True)
    record_capacity: int = eqx.field(static=True)
    ancestry_capacity: int = eqx.field(static=True)
    ball_width: int = eqx.field(static=True)
    periodic_orbit_width: int = eqx.field(static=True)
    mesh_signature_id: str = eqx.field(static=True)
    signature_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        vertex_capacity: int,
        cell_capacity: int,
        ball_width: int,
        periodic_orbit_width: int = 1,
    ) -> None:
        vertices = _count(vertex_capacity, "vertex_capacity")
        cells = _count(cell_capacity, "cell_capacity")
        records = _RECORDS_PER_SLOT * (vertices + cells)
        pairs = _PAIRS_PER_CELL * cells
        if max(2 * vertices, 6 * cells, 2 * records, 2 * pairs) > _INDEX_LIMIT:
            raise ValueError("Device metric capacities must address int32 slots.")
        width = _count(ball_width, "ball_width")
        if width > cells:
            raise ValueError("ball_width cannot exceed the cell capacity.")
        orbit_width = _count(periodic_orbit_width, "periodic_orbit_width")
        mesh_signature = masked_simplex_signature(
            "triangle", 2, vertices, cells, np.dtype(np.float64)
        )
        self.vertex_capacity = vertices
        self.cell_capacity = cells
        self.record_capacity = records
        self.ancestry_capacity = pairs
        self.ball_width = width
        self.periodic_orbit_width = orbit_width
        self.mesh_signature_id = mesh_signature
        self.signature_id = canonical_fingerprint(
            {
                "kind": "device-metric-layout",
                "mesh": mesh_signature,
                "record_capacity": records,
                "ancestry_capacity": pairs,
                "ball_width": width,
                "periodic_orbit_width": orbit_width,
            }
        )


@final
class DeviceMetricState(StrictModule, NonTrainableState):
    """Device epoch arrays plus exact source-bound periodic orbit worksets.

    ``periodic_vertex_translation`` holds, per vertex slot, the deck translation
    from its orbit representative and ``periodic_edge_translation``, per half
    edge, the deck translation from its orbit's representative half; the
    matching ``*_deck`` masks mark slots whose deck element is a pure translation
    with representable components, so ``representative + translation`` rounds to
    the exact deck image. ``periodic_vertex_key`` and ``periodic_vertex_winding``
    are the producer-owned orbit identity (one key per orbit, absolute deck
    exponents per member) published with every edit; splits give each new orbit
    the key of its representative birth and the windings of the split edge
    members (``periodic_edge_winding``).
    """

    mesh: MaskedSimplexMesh
    layout_signature_id: str = eqx.field(static=True)
    ball_width: int = eqx.field(static=True)
    periodic_enabled: bool = eqx.field(static=True)
    periodic_mesh_id: str = eqx.field(static=True)
    periodic_source_mesh_id: str = eqx.field(static=True)
    periodic_source_topology_id: str = eqx.field(static=True)
    periodic_source_lifted_topology_id: str = eqx.field(static=True)
    periodic_source_topology_id_binding: str = eqx.field(static=True)
    metric: Array
    source_rows: Array
    vertex_fixed: Array
    vertex_moved: Array
    collapse_to: Array
    cell_regions: Array
    cell_blocks: Array
    cell_ranks: Array
    edge_classes: Array
    edge_protected: Array
    record_keys: Array
    record_sources: Array
    record_ranks: Array
    ancestry: Array
    periodic_vertex_orbits: Array
    periodic_vertex_representative: Array
    periodic_vertex_members: Array
    periodic_edge_orbits: Array
    periodic_edge_representative: Array
    periodic_edge_start: Array
    periodic_edge_end: Array
    periodic_edge_members: Array
    periodic_cell_orbits: Array
    periodic_vertex_translation: Array
    periodic_vertex_deck: Array
    periodic_edge_translation: Array
    periodic_edge_deck: Array
    periodic_vertex_key: Array
    periodic_vertex_winding: Array
    periodic_edge_winding: Array
    cursors: Array
    controls: Array
    counters: Array
    flags: Array

    @checked
    def __init__(
        self,
        mesh: MaskedSimplexMesh,
        /,
        *,
        layout_signature_id: str,
        ball_width: int,
        metric: ArrayLike,
        source_rows: ArrayLike,
        vertex_fixed: ArrayLike,
        vertex_moved: ArrayLike,
        collapse_to: ArrayLike,
        cell_regions: ArrayLike,
        cell_blocks: ArrayLike,
        cell_ranks: ArrayLike,
        edge_classes: ArrayLike,
        edge_protected: ArrayLike,
        record_keys: ArrayLike,
        record_sources: ArrayLike,
        record_ranks: ArrayLike,
        ancestry: ArrayLike,
        periodic_vertex_orbits: ArrayLike,
        periodic_vertex_representative: ArrayLike,
        periodic_vertex_members: ArrayLike,
        periodic_edge_orbits: ArrayLike,
        periodic_edge_representative: ArrayLike,
        periodic_edge_start: ArrayLike,
        periodic_edge_end: ArrayLike,
        periodic_edge_members: ArrayLike,
        periodic_cell_orbits: ArrayLike,
        periodic_vertex_translation: ArrayLike,
        periodic_vertex_deck: ArrayLike,
        periodic_edge_translation: ArrayLike,
        periodic_edge_deck: ArrayLike,
        periodic_vertex_key: ArrayLike,
        periodic_vertex_winding: ArrayLike,
        periodic_edge_winding: ArrayLike,
        cursors: ArrayLike,
        controls: ArrayLike,
        counters: ArrayLike,
        flags: ArrayLike,
        periodic_enabled: bool = False,
        periodic_mesh_id: str = "",
        periodic_source_mesh_id: str = "",
        periodic_source_topology_id: str = "",
        periodic_source_lifted_topology_id: str = "",
        periodic_source_topology_id_binding: str = "",
    ) -> None:
        if mesh.cell_kind != "triangle" or mesh.ambient_dimension != 2:
            raise ValueError("Device metric adaptation holds planar triangle meshes.")
        if not isinstance(layout_signature_id, str) or not layout_signature_id:
            raise ValueError("layout_signature_id must be nonempty.")
        if not isinstance(periodic_enabled, bool):
            raise TypeError("periodic_enabled must be bool.")
        identity = (
            periodic_source_mesh_id,
            periodic_source_topology_id,
            periodic_source_lifted_topology_id,
            periodic_source_topology_id_binding,
        )
        if any(not isinstance(value, str) for value in identity):
            raise TypeError("Periodic source bindings must be strings.")
        if periodic_enabled != all(identity):
            raise ValueError(
                "Periodic device source identity is incomplete or unexpected."
            )
        width = _count(ball_width, "ball_width")
        vertices, cells = mesh.vertex_capacity, mesh.cell_capacity
        if width > cells:
            raise ValueError("ball_width cannot exceed the cell capacity.")
        orbit_width = _count(
            int(jnp.asarray(periodic_vertex_members).shape[1]),
            "periodic_orbit_width",
        )
        rank = int(jnp.asarray(periodic_vertex_winding).shape[1])
        records = jnp.asarray(record_keys).shape[0]
        pairs = jnp.asarray(ancestry).shape[0]
        specification = {
            "metric": (metric, (vertices, 2, 2), mesh.coordinates.dtype),
            "source_rows": (source_rows, (vertices,), jnp.int32),
            "vertex_fixed": (vertex_fixed, (vertices,), jnp.bool_),
            "vertex_moved": (vertex_moved, (vertices,), jnp.bool_),
            "collapse_to": (collapse_to, (vertices,), jnp.int32),
            "cell_regions": (cell_regions, (cells,), jnp.int32),
            "cell_blocks": (cell_blocks, (cells,), jnp.int32),
            "cell_ranks": (cell_ranks, (cells,), jnp.int32),
            "edge_classes": (edge_classes, (cells, 3), jnp.int32),
            "edge_protected": (edge_protected, (cells, 3), jnp.bool_),
            "record_keys": (record_keys, (records, 2), jnp.int32),
            "record_sources": (record_sources, (records,), jnp.int32),
            "record_ranks": (record_ranks, (records,), jnp.int32),
            "ancestry": (ancestry, (pairs, 2), jnp.int32),
            "periodic_vertex_orbits": (
                periodic_vertex_orbits,
                (vertices,),
                jnp.int64,
            ),
            "periodic_vertex_representative": (
                periodic_vertex_representative,
                (vertices,),
                jnp.int32,
            ),
            "periodic_vertex_members": (
                periodic_vertex_members,
                (vertices, orbit_width),
                jnp.int32,
            ),
            "periodic_edge_orbits": (
                periodic_edge_orbits,
                (3 * cells,),
                jnp.int64,
            ),
            "periodic_edge_representative": (
                periodic_edge_representative,
                (3 * cells,),
                jnp.int32,
            ),
            "periodic_edge_start": (
                periodic_edge_start,
                (3 * cells,),
                jnp.int32,
            ),
            "periodic_edge_end": (
                periodic_edge_end,
                (3 * cells,),
                jnp.int32,
            ),
            "periodic_edge_members": (
                periodic_edge_members,
                (3 * cells, orbit_width),
                jnp.int32,
            ),
            "periodic_cell_orbits": (
                periodic_cell_orbits,
                (cells,),
                jnp.int64,
            ),
            "periodic_vertex_translation": (
                periodic_vertex_translation,
                (vertices, 2),
                mesh.coordinates.dtype,
            ),
            "periodic_vertex_deck": (periodic_vertex_deck, (vertices,), jnp.bool_),
            "periodic_edge_translation": (
                periodic_edge_translation,
                (3 * cells, 2),
                mesh.coordinates.dtype,
            ),
            "periodic_edge_deck": (periodic_edge_deck, (3 * cells,), jnp.bool_),
            "periodic_vertex_key": (periodic_vertex_key, (vertices, 2), jnp.int64),
            "periodic_vertex_winding": (
                periodic_vertex_winding,
                (vertices, rank),
                jnp.int64,
            ),
            "periodic_edge_winding": (
                periodic_edge_winding,
                (3 * cells, rank),
                jnp.int64,
            ),
            "cursors": (cursors, (7,), jnp.int64),
            "controls": (controls, (2,), jnp.int32),
            "counters": (counters, (8,), jnp.int64),
            "flags": (flags, (3,), jnp.int32),
        }
        arrays = {}
        for name, (value, shape, dtype) in specification.items():
            array = jnp.asarray(value)
            if array.shape != shape or array.dtype != dtype:
                raise TypeError(f"{name} must be {jnp.dtype(dtype).name} {shape}.")
            arrays[name] = array
        if records < 1 or pairs < 1:
            raise ValueError("Record and ancestry capacities must be positive.")
        self.mesh = mesh
        self.layout_signature_id = layout_signature_id
        self.ball_width = width
        self.periodic_enabled = periodic_enabled
        self.periodic_mesh_id = periodic_mesh_id
        self.periodic_source_mesh_id = periodic_source_mesh_id
        self.periodic_source_topology_id = periodic_source_topology_id
        self.periodic_source_lifted_topology_id = periodic_source_lifted_topology_id
        self.periodic_source_topology_id_binding = periodic_source_topology_id_binding
        for name, array in arrays.items():
            setattr(self, name, array)

    @property
    def status_flags(self) -> Array:
        """Cumulative terminal status flags carried by this device epoch."""
        return self.flags[..., _STATUS]


@final
class DeviceMetricReport(StrictModule):
    """Traceable evidence of one `adapt_device_metric` call.

    ``status`` distinguishes actual capacity overflow, invalid geometry, and
    predicates requiring host resolution. The host wrapper grows/reprepares
    static slots or cavity workspace and retries the same operation before
    returning. A terminal mathematical or numerical failure rolls back the
    candidate arrays; a state already carrying such a flag is refused.
    Operation and rejection counts cover the returned compiled call, and
    ``rejected_cavity`` is zero after successful workspace growth.
    Measurements cover all unprotected edges, or every edge when all are
    protected, and the active cells of the returned state.
    """

    status: Array
    passes: Array
    splits: Array
    collapses: Array
    flips: Array
    relocations: Array
    rejected_uncertain: Array
    rejected_invalid: Array
    rejected_cavity: Array
    converged: Array
    stalled: Array
    measured_edges: Array
    out_of_range_edges: Array
    minimum_metric_length: Array
    maximum_metric_length: Array
    unit_fraction: Array
    minimum_metric_quality: Array
    uncertain_cells: Array
    invalid_cells: Array

    @property
    def failed(self) -> Array:
        return (self.status & _FAILURES) != 0


@final
class DeviceMetricUpdate(StrictModule):
    """State and report of one compiled device metric adaptation call."""

    state: DeviceMetricState
    report: DeviceMetricReport
    layout: DeviceMetricLayout = eqx.field(static=True)


@final
class DeviceMetricEvidence(StrictModule, NonTrainableState):
    """Committed evidence of one device metric adaptation epoch.

    Operation and rejection counts are totals since preparation; ``status`` is
    the union of the applied `AdaptiveSimplexStatus` flags. Length and quality
    measurements cover the unprotected edges of the committed mesh (all edges
    when every edge is protected); ``converged`` certifies the unit-mesh
    criterion on them and ``stalled`` a last pass that applied nothing.
    """

    passes: int = eqx.field(static=True)
    splits: int = eqx.field(static=True)
    collapses: int = eqx.field(static=True)
    flips: int = eqx.field(static=True)
    relocations: int = eqx.field(static=True)
    rejected_uncertain: int = eqx.field(static=True)
    rejected_invalid: int = eqx.field(static=True)
    rejected_cavity: int = eqx.field(static=True)
    converged: bool = eqx.field(static=True)
    stalled: bool = eqx.field(static=True)
    measured_edges: int = eqx.field(static=True)
    out_of_range_edges: int = eqx.field(static=True)
    minimum_metric_length: float = eqx.field(static=True)
    maximum_metric_length: float = eqx.field(static=True)
    unit_fraction: float = eqx.field(static=True)
    minimum_metric_quality: float = eqx.field(static=True)
    topology_operations: bool = eqx.field(static=True)
    relocation: bool = eqx.field(static=True)
    status: AdaptiveSimplexStatus = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        passes: int,
        splits: int,
        collapses: int,
        flips: int,
        relocations: int,
        rejected_uncertain: int,
        rejected_invalid: int,
        rejected_cavity: int,
        converged: bool,
        stalled: bool,
        measured_edges: int,
        out_of_range_edges: int,
        minimum_metric_length: float,
        maximum_metric_length: float,
        unit_fraction: float,
        minimum_metric_quality: float,
        topology_operations: bool,
        relocation: bool,
        status: AdaptiveSimplexStatus,
        layout_id: str,
    ) -> None:
        counts = {
            "passes": passes,
            "splits": splits,
            "collapses": collapses,
            "flips": flips,
            "relocations": relocations,
            "rejected_uncertain": rejected_uncertain,
            "rejected_invalid": rejected_invalid,
            "rejected_cavity": rejected_cavity,
            "measured_edges": measured_edges,
            "out_of_range_edges": out_of_range_edges,
        }
        flags = {
            "converged": converged,
            "stalled": stalled,
            "topology_operations": topology_operations,
            "relocation": relocation,
        }
        measures = {
            "minimum_metric_length": minimum_metric_length,
            "maximum_metric_length": maximum_metric_length,
            "unit_fraction": unit_fraction,
            "minimum_metric_quality": minimum_metric_quality,
        }
        for name, value in counts.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        for name, value in flags.items():
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be bool.")
        for name, value in measures.items():
            if not isinstance(value, float) or not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite non-negative float.")
        if unit_fraction > 1.0:
            raise ValueError("unit_fraction cannot exceed one.")
        if converged and stalled:
            raise ValueError("A converged adaptation cannot be stalled.")
        if not isinstance(status, AdaptiveSimplexStatus):
            raise TypeError("status must be AdaptiveSimplexStatus.")
        if status & _FAILURES:
            raise ValueError("Committed evidence cannot carry a refused call.")
        if not isinstance(layout_id, str):
            raise TypeError("layout_id must be a str.")
        self.passes = passes
        self.splits = splits
        self.collapses = collapses
        self.flips = flips
        self.relocations = relocations
        self.rejected_uncertain = rejected_uncertain
        self.rejected_invalid = rejected_invalid
        self.rejected_cavity = rejected_cavity
        self.converged = converged
        self.stalled = stalled
        self.measured_edges = measured_edges
        self.out_of_range_edges = out_of_range_edges
        self.minimum_metric_length = minimum_metric_length
        self.maximum_metric_length = maximum_metric_length
        self.unit_fraction = unit_fraction
        self.minimum_metric_quality = minimum_metric_quality
        self.topology_operations = topology_operations
        self.relocation = relocation
        self.status = status
        self.layout_id = layout_id
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "device-metric-evidence",
                **counts,
                **flags,
                **measures,
                "status": int(status),
            }
        )


class _Anchor(NamedTuple):
    """Host source description shared by preparation and commit."""

    source: _Source
    first_cell_id: int


@final
class PreparedDeviceMetricAdaptation(StrictModule, NonTrainableState):
    """Prepared source context for one device metric workspace.

    The returned update carries the new layout and state. Periodic callers
    also pass this source context to ``adapt_device_metric`` so host topology
    barriers can rebuild exact quotient maps; all numerical edits remain in
    the compiled device kernels.
    """

    adaptation: PreparedMeshAdaptation
    layout: DeviceMetricLayout
    state: DeviceMetricState
    anchor: _Anchor
    periodic_mesh_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        adaptation: PreparedMeshAdaptation,
        layout: DeviceMetricLayout,
        state: DeviceMetricState,
        anchor: _Anchor,
        /,
        *,
        periodic_mesh_id: str | None = None,
    ) -> None:
        self.adaptation = adaptation
        self.layout = layout
        self.state = state
        self.anchor = anchor
        bound_mesh_id = (
            state.periodic_mesh_id if periodic_mesh_id is None else periodic_mesh_id
        )
        if bound_mesh_id != state.periodic_mesh_id:
            raise ValueError("Prepared periodic mesh identity disagrees with its state.")
        if not isinstance(bound_mesh_id, str):
            raise TypeError("periodic_mesh_id must be a string.")
        self.periodic_mesh_id = bound_mesh_id
        source_mesh = adaptation.source.mesh
        source_identity = (
            (
                source_mesh.mesh_id,
                source_mesh.topology_id,
                source_mesh.lifted_topology_id,
                source_mesh.periodic_topology.periodic_topology_id,
            )
            if source_mesh.periodic_topology is not None
            else ("", "", "", "")
        )
        if (
            state.periodic_enabled != (source_mesh.periodic_topology is not None)
            or _periodic_source_identity(state) != source_identity
        ):
            raise ValueError("Prepared adaptation and device source identities differ.")
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-device-metric-adaptation",
                "adaptation": adaptation.prepared_id,
                "layout": layout.signature_id,
                "periodic_mesh": bound_mesh_id,
            }
        )


# ------------------------------------------------------------------ kernel records


class _Work(NamedTuple):
    """Array view of one state inside the compiled kernel.

    ``periodic_enabled`` is a Python bool carried as static pytree data (see the
    registration below), so non-periodic executables never trace the periodic
    orbit branches.
    """

    coordinates: Array
    vertex_ids: Array
    vertex_active: Array
    cells: Array
    cell_ids: Array
    cell_active: Array
    periodic_enabled: bool
    metric: Array
    source_rows: Array
    vertex_fixed: Array
    vertex_moved: Array
    collapse_to: Array
    cell_regions: Array
    cell_blocks: Array
    cell_ranks: Array
    edge_classes: Array
    edge_protected: Array
    record_keys: Array
    record_sources: Array
    record_ranks: Array
    ancestry: Array
    periodic_vertex_orbits: Array
    periodic_vertex_representative: Array
    periodic_vertex_members: Array
    periodic_edge_orbits: Array
    periodic_edge_representative: Array
    periodic_edge_start: Array
    periodic_edge_end: Array
    periodic_edge_members: Array
    periodic_cell_orbits: Array
    periodic_vertex_translation: Array
    periodic_vertex_deck: Array
    periodic_edge_translation: Array
    periodic_edge_deck: Array
    periodic_vertex_key: Array
    periodic_vertex_winding: Array
    periodic_edge_winding: Array
    cursors: Array
    controls: Array
    counters: Array
    flags: Array


_WORK_PERIODIC = _Work._fields.index("periodic_enabled")


def _flatten_work(work: _Work, /) -> tuple[tuple[Any, ...], bool]:
    values = tuple(work)
    return values[:_WORK_PERIODIC] + values[_WORK_PERIODIC + 1 :], work.periodic_enabled


def _unflatten_work(periodic_enabled: bool, children: Any, /) -> _Work:
    values = tuple(children)
    return _Work._make(
        (*values[:_WORK_PERIODIC], periodic_enabled, *values[_WORK_PERIODIC:])
    )


jax.tree_util.register_pytree_node(_Work, _flatten_work, _unflatten_work)


def _work(state: DeviceMetricState, /) -> _Work:
    mesh = state.mesh
    return _Work(
        mesh.coordinates,
        mesh.vertex_ids,
        mesh.vertex_active,
        mesh.cells,
        mesh.cell_ids,
        mesh.cell_active,
        bool(state.periodic_enabled),
        state.metric,
        state.source_rows,
        state.vertex_fixed,
        state.vertex_moved,
        state.collapse_to,
        state.cell_regions,
        state.cell_blocks,
        state.cell_ranks,
        state.edge_classes,
        state.edge_protected,
        state.record_keys,
        state.record_sources,
        state.record_ranks,
        state.ancestry,
        state.periodic_vertex_orbits,
        state.periodic_vertex_representative,
        state.periodic_vertex_members,
        state.periodic_edge_orbits,
        state.periodic_edge_representative,
        state.periodic_edge_start,
        state.periodic_edge_end,
        state.periodic_edge_members,
        state.periodic_cell_orbits,
        state.periodic_vertex_translation,
        state.periodic_vertex_deck,
        state.periodic_edge_translation,
        state.periodic_edge_deck,
        state.periodic_vertex_key,
        state.periodic_vertex_winding,
        state.periodic_edge_winding,
        state.cursors,
        state.controls,
        state.counters,
        state.flags,
    )


def _state(
    work: _Work,
    layout: DeviceMetricLayout,
    periodic_enabled: bool,
    periodic_mesh_id: str = "",
    periodic_source_identity: tuple[str, str, str, str] = ("", "", "", ""),
    /,
) -> DeviceMetricState:
    """Rebuild public state; half-facet adjacency follows current cells."""
    mesh = MaskedSimplexMesh(
        work.coordinates,
        work.vertex_ids,
        work.vertex_active,
        work.cells,
        work.cell_ids,
        work.cell_active,
        masked_simplex_facet_neighbors(work.cells, work.cell_active),
    )
    return DeviceMetricState(
        mesh,
        layout_signature_id=layout.signature_id,
        ball_width=layout.ball_width,
        periodic_enabled=periodic_enabled,
        periodic_mesh_id=periodic_mesh_id,
        periodic_source_mesh_id=periodic_source_identity[0],
        periodic_source_topology_id=periodic_source_identity[1],
        periodic_source_lifted_topology_id=periodic_source_identity[2],
        periodic_source_topology_id_binding=periodic_source_identity[3],
        metric=work.metric,
        source_rows=work.source_rows,
        vertex_fixed=work.vertex_fixed,
        vertex_moved=work.vertex_moved,
        collapse_to=work.collapse_to,
        cell_regions=work.cell_regions,
        cell_blocks=work.cell_blocks,
        cell_ranks=work.cell_ranks,
        edge_classes=work.edge_classes,
        edge_protected=work.edge_protected,
        record_keys=work.record_keys,
        record_sources=work.record_sources,
        record_ranks=work.record_ranks,
        ancestry=work.ancestry,
        periodic_vertex_orbits=work.periodic_vertex_orbits,
        periodic_vertex_representative=work.periodic_vertex_representative,
        periodic_vertex_members=work.periodic_vertex_members,
        periodic_edge_orbits=work.periodic_edge_orbits,
        periodic_edge_representative=work.periodic_edge_representative,
        periodic_edge_start=work.periodic_edge_start,
        periodic_edge_end=work.periodic_edge_end,
        periodic_edge_members=work.periodic_edge_members,
        periodic_cell_orbits=work.periodic_cell_orbits,
        periodic_vertex_translation=work.periodic_vertex_translation,
        periodic_vertex_deck=work.periodic_vertex_deck,
        periodic_edge_translation=work.periodic_edge_translation,
        periodic_edge_deck=work.periodic_edge_deck,
        periodic_vertex_key=work.periodic_vertex_key,
        periodic_vertex_winding=work.periodic_vertex_winding,
        periodic_edge_winding=work.periodic_edge_winding,
        cursors=work.cursors,
        controls=work.controls,
        counters=work.counters,
        flags=work.flags,
    )


def _periodic_source_identity(state: DeviceMetricState, /) -> tuple[str, str, str, str]:
    return (
        state.periodic_source_mesh_id,
        state.periodic_source_topology_id,
        state.periodic_source_lifted_topology_id,
        state.periodic_source_topology_id_binding,
    )


class _Topology(NamedTuple):
    """Derived adjacency and classification of one round (half-edge indexed).

    Half-edge ``3 c + i`` is the edge of cell slot ``c`` opposite local vertex
    ``i``; ``canonical`` selects one half per active edge.
    """

    neighbors: Array
    canonical: Array
    ends: Array
    lengths: Array
    classes: Array
    protected: Array
    quality: Array
    codes: Array
    code_halves: Array
    kind: Array
    curve_class: Array
    curve_neighbors: Array
    straight: Array
    uniform: Array
    valence: Array
    ball: Array
    link: Array
    link_half: Array


class _Children(NamedTuple):
    """Cells created by one round; ``parents[:, 0]`` is the primary parent."""

    mask: Array
    rows: Array
    parents: Array
    rank: int
    classes: Array
    protected: Array


# ------------------------------------------------------------------ primitives


def _take(values: Array, index: Array, /) -> Array:
    return jnp.take_along_axis(values, index[..., None], axis=-1)[..., 0]


def _sorted_pair(first: Array, second: Array, /) -> Array:
    return jnp.stack((jnp.minimum(first, second), jnp.maximum(first, second)), axis=-1)


def _edge_code(first: Array, second: Array, vertex_capacity: int, /) -> Array:
    low = jnp.minimum(first, second).astype(jnp.int64)
    return low * vertex_capacity + jnp.maximum(first, second).astype(jnp.int64)


def _half_ends(cells: Array, /) -> Array:
    """Ascending vertex slots of the edge opposite every local vertex."""
    return _sorted_pair(
        cells[:, _NEXT].reshape((-1,)), cells[:, _PREVIOUS].reshape((-1,))
    )


def _lengths(work: _Work, ends: Array, /) -> Array:
    """Riemannian lengths through the owner; ascending ends make halves identical."""
    return metric_edge_lengths(work.metric, work.coordinates, ends)


def _log_euclidean(tensors: Array, weights: Array, /) -> Array:
    """``exp(sum_k w_k log M_k)`` of SPD samples ``(..., k, 2, 2)``.

    The formula of `interpolate_mesh_metric`; its SPD/convexity guard is a host
    callback (`eqx.error_if`) that compiled passes must not embed. Samples are
    metric tensors of the state and weights are convex by construction.
    """
    return _constant_preserving_metric_mean(tensors, weights)


def _quality(lengths: Array, /) -> Array:
    """Metric shape quality ``4 sqrt(3) area / sum(l^2)`` from edge lengths (Heron)."""
    a, b, c = lengths[..., 0], lengths[..., 1], lengths[..., 2]
    product = (a + b + c) * (b + c - a) * (a + c - b) * (a + b - c)
    return math.sqrt(3.0) * jnp.sqrt(jnp.maximum(product, 0.0)) / (a * a + b * b + c * c)


def _barycentric(triangles: Array, points: Array, /) -> Array:
    """Floating barycentric weights of ``points`` in ``(..., 3, 2)`` triangles."""

    def cross(origin: Any, first: Any, second: Any) -> Any:
        u = first - origin
        v = second - origin
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

    a, b, c = triangles[..., 0, :], triangles[..., 1, :], triangles[..., 2, :]
    weights = jnp.stack(
        (cross(points, b, c), cross(points, c, a), cross(points, a, b)), axis=-1
    )
    return weights / cross(a, b, c)[..., None]


def _metric_midpoint(metric: Array, points: Array, first: Array, second: Array, /) -> Any:
    """Parameter of the metric-length midpoint under geometric size variation."""
    delta = points[second] - points[first]

    def length(rows: Any) -> Any:
        tensor = metric[rows]
        return jnp.sqrt(
            delta[:, 0] * delta[:, 0] * tensor[:, 0, 0]
            + 2.0 * delta[:, 0] * delta[:, 1] * tensor[:, 0, 1]
            + delta[:, 1] * delta[:, 1] * tensor[:, 1, 1]
        )

    ratio = length(second) / length(first)
    nearly_equal = jnp.abs(ratio - 1.0) <= 1.0e-6
    safe = jnp.where(nearly_equal, 2.0, ratio)
    return jnp.where(nearly_equal, 0.5, jnp.log(0.5 * (1.0 + safe)) / jnp.log(safe))


def _orientation(a: Array, b: Array, c: Array, /) -> Array:
    """``_OK`` for certified POSITIVE, ``_UNCERTAIN``, else ``_INVALID``."""
    result = orient2d(a, b, c, mode=PredicateMode.FILTERED_DEVICE)
    positive = result.signs == int(PredicateSign.POSITIVE)
    return jnp.where(
        ~result.certain, _UNCERTAIN, jnp.where(positive, _OK, _INVALID)
    ).astype(jnp.int32)


def _collinear(a: Array, b: Array, c: Array, /) -> Array:
    """``_OK`` for certified ZERO orientation, ``_UNCERTAIN``, else ``_INVALID``."""
    result = orient2d(a, b, c, mode=PredicateMode.FILTERED_DEVICE)
    zero = result.signs == int(PredicateSign.ZERO)
    return jnp.where(~result.certain, _UNCERTAIN, jnp.where(zero, _OK, _INVALID)).astype(
        jnp.int32
    )


def _ranks(valid: Array, *keys: Array) -> Array:
    """Unique priority ranks: valid operations first, then lexicographic keys."""
    count = valid.shape[0]
    index = jnp.arange(count, dtype=jnp.int32)
    ordered = jax.lax.sort(
        (jnp.where(valid, 0, 1).astype(jnp.int32), *keys, index),
        num_keys=len(keys) + 1,
        is_stable=True,
    )
    return jnp.zeros((count,), dtype=jnp.int32).at[ordered[-1]].set(index)


def _independent(rank: Array, valid: Array, items: Array, cell_capacity: int, /) -> Any:
    """Greedy maximal set of valid operations with pairwise disjoint cavities."""
    present = items >= 0
    gather = jnp.where(present, items, 0)
    own = jnp.broadcast_to(rank[:, None], items.shape)

    def undecided_remain(carry: Any) -> Any:
        undecided, _ = carry
        return jnp.any(undecided)

    def select_round(carry: Any) -> Any:
        undecided, selected = carry
        live = undecided[:, None] & present
        best = (
            jnp.full((cell_capacity + 1,), _RANK_SENTINEL, dtype=jnp.int32)
            .at[jnp.where(live, items, cell_capacity)]
            .min(own)
        )
        losing = jnp.any(live & (best[gather] != own), axis=1)
        winners = undecided & ~losing
        taken = (
            jnp.zeros((cell_capacity + 1,), dtype=jnp.bool_)
            .at[jnp.where(winners[:, None] & present, items, cell_capacity)]
            .set(True)
        )
        blocked = jnp.any(present & taken[gather], axis=1)
        return undecided & ~winners & ~blocked, selected | winners

    _, selected = jax.lax.while_loop(
        undecided_remain,
        select_round,
        (valid, jnp.zeros(valid.shape, dtype=jnp.bool_)),
    )
    return selected


def _distinct_count(values: Array, present: Array, /) -> Array:
    """Number of distinct present values in each row."""
    width = values.shape[1]
    earlier = jnp.arange(width)[None, :] < jnp.arange(width)[:, None]
    repeated = jnp.any(
        (values[:, :, None] == values[:, None, :])
        & present[:, None, :]
        & earlier[None, :, :],
        axis=2,
    )
    return jnp.sum(present & ~repeated, axis=1, dtype=jnp.int32)


def _periodic_edge_batch(
    work: _Work,
    topology: _Topology,
    valid: Array,
    items: Array,
    rank: Array,
    layout: DeviceMetricLayout,
    guard: Array | None = None,
    /,
) -> tuple[Array, Array, Array]:
    """Select complete edge orbits with disjoint lifted cavities.

    ``guard`` (per representative half) adds cells a group's decision read
    to its independence footprint without counting as member overlap.
    """
    half_count = 3 * layout.cell_capacity
    halves = jnp.arange(half_count, dtype=jnp.int32)
    representative = work.periodic_edge_representative
    group_rep = topology.canonical & (representative == halves)
    members = work.periodic_edge_members
    present = members >= 0
    member_valid = valid[jnp.maximum(members, 0)]
    group_valid = (
        group_rep & jnp.any(present, axis=1) & jnp.all(~present | member_valid, axis=1)
    )
    member_items = items[jnp.maximum(members, 0)]
    member_columns = jnp.broadcast_to(
        jnp.arange(members.shape[1], dtype=jnp.int32)[None, :], members.shape
    )
    group_items = jnp.where(present[..., None], member_items, -1).reshape(
        (half_count, -1)
    )
    group_members = jnp.broadcast_to(member_columns[..., None], member_items.shape)
    group_members = jnp.where(present[..., None], group_members, -1).reshape(
        (half_count, -1)
    )
    safe_items = jnp.where(group_items >= 0, group_items, layout.cell_capacity)
    order = jnp.argsort(safe_items, axis=1)
    sorted_items = jnp.take_along_axis(safe_items, order, axis=1)
    sorted_members = jnp.take_along_axis(group_members, order, axis=1)
    overlap = group_valid & jnp.any(
        (sorted_items[:, 1:] == sorted_items[:, :-1])
        & (sorted_items[:, :-1] < layout.cell_capacity)
        & (sorted_members[:, 1:] != sorted_members[:, :-1]),
        axis=1,
    )
    footprint = group_items if guard is None else jnp.concatenate((group_items, guard), 1)
    selected_representatives = _independent(
        rank, group_valid & ~overlap, footprint, layout.cell_capacity
    )
    selected_members = (
        (representative >= 0)
        & topology.canonical
        & selected_representatives[jnp.maximum(representative, 0)]
    )
    return selected_representatives, selected_members, group_valid & overlap


def _tally(counters: Array, status: Any, mask: Array, /) -> tuple[Array, Array]:
    """Add the rejections among ``mask``; uncertain ones need host resolution."""

    def count(condition: Any) -> Any:
        return jnp.sum(mask & condition, dtype=jnp.int64)

    uncertain = count(status == _UNCERTAIN)
    counters = (
        counters.at[_REJECTED_UNCERTAIN]
        .add(uncertain)
        .at[_REJECTED_CAVITY]
        .add(count(status == _CAVITY))
        .at[_REJECTED_INVALID]
        .add(count(status >= _INVALID))
    )
    bits = jnp.where(
        uncertain > 0, int(AdaptiveSimplexStatus.NEEDS_HOST_RESOLUTION), 0
    ).astype(jnp.int32)
    return counters, bits


def _overflow(condition: Array, /) -> Array:
    return jnp.where(condition, int(AdaptiveSimplexStatus.CAPACITY_EXCEEDED), 0).astype(
        jnp.int32
    )


def _edge_half(
    topology: _Topology, first: Array, second: Array, vertex_capacity: int
) -> Any:
    """Canonical half-edge of each queried vertex pair, or -1 when not an edge."""
    code = _edge_code(first, second, vertex_capacity)
    position = jnp.minimum(
        jnp.searchsorted(topology.codes, code), topology.codes.shape[0] - 1
    )
    return jnp.where(topology.codes[position] == code, topology.code_halves[position], -1)


# ------------------------------------------------------------------ topology


def _vertex_classes(
    work: _Work,
    canonical: Any,
    ends: Any,
    classes: Any,
    protected: Any,
    vertex_capacity: Any,
) -> Any:
    """Host vertex kinds: curve vertices carry exactly two edges of one class."""
    low, high = ends[:, 0], ends[:, 1]
    classified = canonical & (classes > 0)
    guarded_edge = canonical & protected

    def at(mask: Any, index: Any) -> Any:
        return jnp.where(mask, index, vertex_capacity)

    shape = (vertex_capacity,)
    incident = (
        jnp.zeros(shape, dtype=jnp.int32)
        .at[at(classified, low)]
        .add(1, mode="drop")
        .at[at(classified, high)]
        .add(1, mode="drop")
    )
    label_low = (
        jnp.full(shape, _RANK_SENTINEL, dtype=jnp.int32)
        .at[at(classified, low)]
        .min(classes, mode="drop")
        .at[at(classified, high)]
        .min(classes, mode="drop")
    )
    label_high = (
        jnp.full(shape, -1, dtype=jnp.int32)
        .at[at(classified, low)]
        .max(classes, mode="drop")
        .at[at(classified, high)]
        .max(classes, mode="drop")
    )
    guarded = (
        work.vertex_fixed.at[at(guarded_edge, low)]
        .set(True, mode="drop")
        .at[at(guarded_edge, high)]
        .set(True, mode="drop")
    )
    curve = (incident == 2) & (label_low == label_high) & ~guarded
    kind = jnp.where(
        curve, _CURVE, jnp.where(guarded | (incident > 0), _CORNER, _INTERIOR)
    ).astype(jnp.int32)
    first = (
        jnp.full(shape, vertex_capacity, dtype=jnp.int32)
        .at[at(classified, low)]
        .min(high, mode="drop")
        .at[at(classified, high)]
        .min(low, mode="drop")
    )
    second = (
        jnp.full(shape, -1, dtype=jnp.int32)
        .at[at(classified, low)]
        .max(high, mode="drop")
        .at[at(classified, high)]
        .max(low, mode="drop")
    )
    neighbors = jnp.where(curve[:, None], jnp.stack((first, second), axis=1), -1)
    points = work.coordinates
    straight = jnp.where(
        curve,
        _collinear(
            points[jnp.maximum(neighbors[:, 0], 0)],
            points,
            points[jnp.maximum(neighbors[:, 1], 0)],
        ),
        _INVALID,
    )
    return kind, jnp.where(curve, label_low, 0), neighbors, straight


def _topology(work: _Work, layout: DeviceMetricLayout, /) -> _Topology:
    vertex_capacity = layout.vertex_capacity
    cell_capacity = layout.cell_capacity
    width = layout.ball_width
    halves_count = 3 * cell_capacity
    cells = work.cells
    active = work.cell_active
    neighbors = masked_simplex_facet_neighbors(cells, active)
    other = neighbors.reshape((-1,))
    halves = jnp.arange(halves_count, dtype=jnp.int32)
    half_active = jnp.repeat(active, 3)
    canonical = half_active & ((other < 0) | (halves < other))
    ends = _half_ends(cells)
    lengths = _lengths(work, ends)
    classes = work.edge_classes.reshape((-1,))
    protected = work.edge_protected.reshape((-1,))
    cell_lengths = jnp.where(active[:, None], lengths.reshape((cell_capacity, 3)), 1.0)
    quality = jnp.where(active, _quality(cell_lengths), jnp.inf)
    codes, code_halves = jax.lax.sort(
        (
            jnp.where(
                canonical,
                _edge_code(ends[:, 0], ends[:, 1], vertex_capacity),
                _CODE_SENTINEL,
            ),
            halves,
        ),
        num_keys=1,
    )
    kind, curve_class, curve_neighbors, straight = _vertex_classes(
        work, canonical, ends, classes, protected, vertex_capacity
    )
    corner_vertex = jnp.where(half_active, cells.reshape((-1,)), vertex_capacity)
    corner_region = jnp.repeat(work.cell_regions, 3)
    region_low = (
        jnp.full((vertex_capacity,), _RANK_SENTINEL, dtype=jnp.int32)
        .at[corner_vertex]
        .min(corner_region, mode="drop")
    )
    region_high = (
        jnp.full((vertex_capacity,), -1, dtype=jnp.int32)
        .at[corner_vertex]
        .max(corner_region, mode="drop")
    )
    # Vertex balls: corners sorted by (vertex, corner) list incident cells in
    # ascending slot order.
    ordered_vertex, ordered_corner = jax.lax.sort((corner_vertex, halves), num_keys=2)
    vertices = jnp.arange(vertex_capacity, dtype=jnp.int32)
    starts = jnp.searchsorted(ordered_vertex, vertices, side="left")
    valence = jnp.searchsorted(ordered_vertex, vertices, side="right") - starts
    column = jnp.arange(width, dtype=jnp.int32)
    inside = column[None, :] < valence[:, None]
    corner = ordered_corner[
        jnp.minimum(starts[:, None] + column[None, :], halves_count - 1)
    ]
    ball = jnp.where(inside, corner // 3, -1)
    local = jnp.where(inside, corner % 3, 0)
    safe_ball = jnp.maximum(ball, 0)
    rows = cells[safe_ball]
    # Every neighbor once: the successor in each ball cell, plus the predecessor
    # across a boundary edge (manifold fans).
    following_half = 3 * safe_ball + (local + 2) % 3
    preceding_half = 3 * safe_ball + (local + 1) % 3
    link = jnp.concatenate(
        (
            jnp.where(inside, _take(rows, (local + 1) % 3), -1),
            jnp.where(
                inside & (other[preceding_half] < 0), _take(rows, (local + 2) % 3), -1
            ),
        ),
        axis=1,
    )
    return _Topology(
        neighbors=neighbors,
        canonical=canonical,
        ends=ends,
        lengths=lengths,
        classes=classes,
        protected=protected,
        quality=quality,
        codes=codes,
        code_halves=code_halves,
        kind=kind,
        curve_class=curve_class,
        curve_neighbors=curve_neighbors,
        straight=straight,
        uniform=region_low == region_high,
        valence=valence,
        ball=ball,
        link=link,
        link_half=jnp.concatenate((following_half, preceding_half), axis=1),
    )


# ------------------------------------------------------------------ state updates


def _ancestry(
    work: _Work, removed: Array, lineage: Array, layout: DeviceMetricLayout
) -> Any:
    """Children inherit the source cells of all listed parents; removed cells drop.

    ``lineage[c]`` lists the parents of new cell slot ``c`` (``-1`` padded). The
    parents' pairs are gathered by prefix sums over the sorted pair table, then
    merged with the surviving pairs, sorted, and deduplicated.
    """
    cell_capacity = layout.cell_capacity
    capacity = layout.ancestry_capacity
    pair_cells, pair_sources = work.ancestry[:, 0], work.ancestry[:, 1]
    slots = jnp.arange(cell_capacity, dtype=jnp.int32)
    starts = jnp.searchsorted(pair_cells, slots, side="left")
    stops = jnp.searchsorted(pair_cells, slots, side="right")
    parent = lineage.reshape((-1,))
    child = jnp.repeat(slots, 3)
    safe = jnp.maximum(parent, 0)
    counts = jnp.where(parent >= 0, stops[safe] - starts[safe], 0)
    ends = jnp.cumsum(counts)
    total = ends[-1]
    position = jnp.arange(capacity, dtype=jnp.int32)
    relation = jnp.minimum(
        jnp.searchsorted(ends, position, side="right"), parent.shape[0] - 1
    )
    offset = position - (ends[relation] - counts[relation])
    fresh = position < total
    source_row = jnp.clip(starts[safe[relation]] + offset, 0, capacity - 1)
    kept = (pair_cells < cell_capacity) & ~removed[
        jnp.minimum(pair_cells, cell_capacity - 1)
    ]
    ordered_cells, ordered_sources = jax.lax.sort(
        (
            jnp.concatenate(
                (
                    jnp.where(kept, pair_cells, cell_capacity),
                    jnp.where(fresh, child[relation], cell_capacity),
                )
            ),
            jnp.concatenate(
                (
                    jnp.where(kept, pair_sources, 0),
                    jnp.where(fresh, pair_sources[source_row], 0),
                )
            ),
        ),
        num_keys=2,
    )
    first = jnp.ones((1,), dtype=jnp.bool_)
    distinct = (ordered_cells < cell_capacity) & jnp.concatenate(
        (
            first,
            (ordered_cells[1:] != ordered_cells[:-1])
            | (ordered_sources[1:] != ordered_sources[:-1]),
        )
    )
    count = jnp.sum(distinct, dtype=jnp.int32)
    target = jnp.where(distinct, jnp.cumsum(distinct, dtype=jnp.int32) - 1, capacity)
    padding = jnp.stack(
        (
            jnp.full((capacity,), cell_capacity, dtype=jnp.int32),
            jnp.zeros((capacity,), dtype=jnp.int32),
        ),
        axis=1,
    )
    pairs = padding.at[target].set(
        jnp.stack((ordered_cells, ordered_sources), axis=1), mode="drop"
    )
    return pairs, count, (total > capacity) | (count > capacity)


def _replace_cells(
    work: _Work, removed: Array, children: _Children, layout: DeviceMetricLayout, /
) -> tuple[_Work, Array]:
    """Deactivate replaced cells and write the children into prefix-sum slots.

    A child takes the region and block of its primary parent, the strongest
    lineage rank of the operation and all its parents, and their ancestry.
    """
    cell_capacity = layout.cell_capacity
    mask = children.mask
    count = jnp.sum(mask, dtype=jnp.int32)
    order = jnp.cumsum(mask, dtype=jnp.int32) - 1
    cursors = work.cursors
    slot = jnp.where(mask, cursors[_CELLS].astype(jnp.int32) + order, cell_capacity)
    parents = children.parents
    safe = jnp.maximum(parents, 0)
    inherited = jnp.max(jnp.where(parents >= 0, work.cell_ranks[safe], 0), axis=1)
    primary = safe[:, 0]
    lineage = (
        jnp.full((cell_capacity, 3), -1, dtype=jnp.int32)
        .at[slot]
        .set(parents, mode="drop")
    )
    pairs, pair_count, pair_overflow = _ancestry(work, removed, lineage, layout)
    work = work._replace(
        cells=work.cells.at[slot].set(children.rows, mode="drop"),
        cell_ids=work.cell_ids.at[slot].set(cursors[_NEXT_CELL] + order, mode="drop"),
        cell_active=(work.cell_active & ~removed).at[slot].set(True, mode="drop"),
        cell_regions=work.cell_regions.at[slot].set(
            work.cell_regions[primary], mode="drop"
        ),
        cell_blocks=work.cell_blocks.at[slot].set(work.cell_blocks[primary], mode="drop"),
        cell_ranks=work.cell_ranks.at[slot].set(
            jnp.maximum(inherited, children.rank), mode="drop"
        ),
        edge_classes=work.edge_classes.at[slot].set(children.classes, mode="drop"),
        edge_protected=work.edge_protected.at[slot].set(children.protected, mode="drop"),
        ancestry=pairs,
        cursors=cursors.at[_CELLS]
        .add(count)
        .at[_NEXT_CELL]
        .add(count)
        .at[_PAIRS]
        .set(pair_count),
    )
    return work, (cursors[_CELLS] + count > cell_capacity) | pair_overflow


def _store_records(
    work: _Work,
    keep: Array,
    keys: Array,
    sources: Array,
    ranks: Array,
    layout: DeviceMetricLayout,
    /,
    appended: tuple[Array, Array, Array, Array] | None = None,
) -> tuple[_Work, Array]:
    """Compact the kept records (then the appended ones) in order by prefix sums."""
    capacity = layout.record_capacity
    if appended is not None:
        extra_keep, extra_keys, extra_sources, extra_ranks = appended
        keep = jnp.concatenate((keep, extra_keep))
        keys = jnp.concatenate((keys, extra_keys))
        sources = jnp.concatenate((sources, extra_sources))
        ranks = jnp.concatenate((ranks, extra_ranks))
    count = jnp.sum(keep, dtype=jnp.int32)
    target = jnp.where(keep, jnp.cumsum(keep, dtype=jnp.int32) - 1, capacity)
    work = work._replace(
        record_keys=jnp.zeros((capacity, 2), dtype=jnp.int32)
        .at[target]
        .set(keys, mode="drop"),
        record_sources=jnp.full((capacity,), -1, dtype=jnp.int32)
        .at[target]
        .set(sources, mode="drop"),
        record_ranks=jnp.zeros((capacity,), dtype=jnp.int32)
        .at[target]
        .set(ranks, mode="drop"),
        cursors=work.cursors.at[_RECORDS].set(count),
    )
    return work, count > capacity


def _compact_cells(work: _Work, layout: DeviceMetricLayout, /) -> _Work:
    """Pack the active cell slots in order; cell IDs are never reused.

    Collapses and flips replace cells, so without compaction the slot demand
    would be every cell ever created. Order preservation keeps slot order equal
    to (provisional) ID order and the ancestry pairs sorted.
    """
    capacity = layout.cell_capacity
    active = work.cell_active
    target = jnp.where(active, jnp.cumsum(active, dtype=jnp.int32) - 1, capacity)

    def packed(values: Any, fill: Any) -> Any:
        return jnp.full_like(values, fill).at[target].set(values, mode="drop")

    pair_cells = work.ancestry[:, 0]
    moved = jnp.where(
        pair_cells < capacity, target[jnp.minimum(pair_cells, capacity - 1)], capacity
    )
    half = jnp.arange(3 * capacity, dtype=jnp.int32)
    half_cell, half_local = half // 3, half % 3
    half_target = jnp.where(
        active[half_cell],
        3 * target[half_cell] + half_local,
        3 * capacity,
    )

    def packed_halves(values: Any, fill: Any) -> Any:
        return jnp.full_like(values, fill).at[half_target].set(values, mode="drop")

    old_representative = work.periodic_edge_representative
    representative = jnp.where(
        old_representative >= 0,
        half_target[jnp.maximum(old_representative, 0)],
        -1,
    )
    old_members = work.periodic_edge_members
    members = jnp.where(
        old_members >= 0,
        half_target[jnp.maximum(old_members, 0)],
        -1,
    )
    packed_members = (
        jnp.full_like(old_members, -1).at[half_target].set(members, mode="drop")
    )
    return work._replace(
        cells=packed(work.cells, 0),
        cell_ids=packed(work.cell_ids, -1),
        cell_active=packed(active, False),
        cell_regions=packed(work.cell_regions, 0),
        cell_blocks=packed(work.cell_blocks, 0),
        cell_ranks=packed(work.cell_ranks, 0),
        edge_classes=packed(work.edge_classes, 0),
        edge_protected=packed(work.edge_protected, False),
        ancestry=work.ancestry.at[:, 0].set(moved),
        periodic_edge_orbits=packed_halves(work.periodic_edge_orbits, -1),
        periodic_edge_representative=packed_halves(representative, -1),
        periodic_edge_start=packed_halves(work.periodic_edge_start, -1),
        periodic_edge_end=packed_halves(work.periodic_edge_end, -1),
        periodic_edge_members=packed_members,
        periodic_cell_orbits=packed(work.periodic_cell_orbits, -1),
        periodic_edge_translation=packed_halves(work.periodic_edge_translation, 0.0),
        periodic_edge_deck=packed_halves(work.periodic_edge_deck, False),
        periodic_edge_winding=packed_halves(work.periodic_edge_winding, 0),
        cursors=work.cursors.at[_CELLS].set(jnp.sum(active, dtype=jnp.int64)),
    )


def _live_records(work: _Work, /) -> Array:
    return jnp.arange(work.record_keys.shape[0]) < work.cursors[_RECORDS]


# ------------------------------------------------------------------ split


class _Sides(NamedTuple):
    """The (at most two) cells of every half-edge's edge: ``(u, v, apex)`` CCW."""

    cell: Array
    local: Array
    valid: Array
    u: Array
    v: Array
    apex: Array


def _split_sides(work: _Work, topology: _Topology, /) -> _Sides:
    other = topology.neighbors.reshape((-1,))
    halves = jnp.arange(other.shape[0], dtype=jnp.int32)
    half = jnp.stack((halves, jnp.maximum(other, 0)), axis=1)
    local = half % 3
    rows = work.cells[half // 3]
    return _Sides(
        cell=half // 3,
        local=local,
        valid=jnp.stack((jnp.ones(halves.shape, dtype=jnp.bool_), other >= 0), 1),
        u=_take(rows, (local + 1) % 3),
        v=_take(rows, (local + 2) % 3),
        apex=_take(rows, local),
    )


def _split_children(
    work: _Work, sides: _Sides, selected: Array, vertex_slot: Array, capacity: int, /
) -> tuple[_Children, Array]:
    """Children (u, m, apex) and (m, v, apex) of every cavity cell.

    The halves of the split edge keep its class; the edges to the apex are new.
    """
    local = sides.local

    def child_features(values: Any) -> Any:
        values = values[sides.cell]
        split = _take(values, local)
        following = _take(values, (local + 1) % 3)
        preceding = _take(values, (local + 2) % 3)
        fresh = jnp.zeros_like(split)
        return jnp.stack(
            (
                jnp.stack((fresh, preceding, split), axis=-1),
                jnp.stack((following, fresh, split), axis=-1),
            ),
            axis=2,
        ).reshape((-1, 3))

    middle = jnp.broadcast_to(vertex_slot[:, None], sides.u.shape)
    split_cells = selected[:, None] & sides.valid
    none = jnp.full(sides.cell.shape, -1, dtype=jnp.int32)
    parents = jnp.stack((sides.cell, none, none), axis=-1).reshape((-1, 3))
    children = _Children(
        mask=jnp.repeat(split_cells.reshape((-1,)), 2),
        rows=jnp.stack(
            (
                jnp.stack((sides.u, middle, sides.apex), axis=-1),
                jnp.stack((middle, sides.v, sides.apex), axis=-1),
            ),
            axis=2,
        ).reshape((-1, 3)),
        parents=jnp.repeat(parents, 2, axis=0),
        rank=_REFINED,
        classes=child_features(work.edge_classes),
        protected=child_features(work.edge_protected),
    )
    removed = (
        jnp.zeros((capacity + 1,), dtype=jnp.bool_)
        .at[jnp.where(split_cells, sides.cell, capacity)]
        .set(True)[:capacity]
    )
    return children, removed


def _split_round_nonperiodic(
    work: _Work, done: Array, layout: DeviceMetricLayout, /
) -> Any:
    work = _compact_cells(work, layout)
    topology = _topology(work, layout)
    vertex_capacity = layout.vertex_capacity
    candidate = topology.canonical & (topology.lengths > _UPPER) & ~topology.protected
    first, second = topology.ends[:, 0], topology.ends[:, 1]
    points = work.coordinates
    t = jnp.where(candidate, _metric_midpoint(work.metric, points, first, second), 0.5)
    middle = points[first] + t[:, None] * (points[second] - points[first])
    sides = _split_sides(work, topology)
    point = middle[:, None, :]
    checks = jnp.maximum(
        _orientation(points[sides.u], point, points[sides.apex]),
        _orientation(point, points[sides.v], points[sides.apex]),
    )
    status = jnp.max(jnp.where(sides.valid, checks, _OK), axis=1)
    counters, bits = _tally(work.counters, status, candidate)
    valid = candidate & (status == _OK)
    ids = work.vertex_ids
    rank = _ranks(valid, -topology.lengths, ids[first], ids[second])
    selected = _independent(
        rank, valid, jnp.where(sides.valid, sides.cell, -1), layout.cell_capacity
    )
    count = jnp.sum(selected, dtype=jnp.int32)
    order = jnp.cumsum(selected, dtype=jnp.int32) - 1
    cursors = work.cursors
    vertex_slot = jnp.where(
        selected, cursors[_VERTICES].astype(jnp.int32) + order, vertex_capacity
    )
    tensors = _log_euclidean(
        jnp.stack((work.metric[first], work.metric[second]), axis=1),
        jnp.stack((1.0 - t, t), axis=1),
    )
    vertex_overflow = cursors[_VERTICES] + count > vertex_capacity
    work = work._replace(
        coordinates=work.coordinates.at[vertex_slot].set(middle, mode="drop"),
        metric=work.metric.at[vertex_slot].set(tensors, mode="drop"),
        vertex_ids=work.vertex_ids.at[vertex_slot].set(
            cursors[_NEXT_VERTEX] + order, mode="drop"
        ),
        vertex_active=work.vertex_active.at[vertex_slot].set(True, mode="drop"),
        cursors=cursors.at[_VERTICES].add(count).at[_NEXT_VERTEX].add(count),
        counters=counters.at[_SPLITS].add(count),
    )
    children, removed = _split_children(
        work, sides, selected, vertex_slot, layout.cell_capacity
    )
    work, cell_overflow = _replace_cells(work, removed, children, layout)
    # Records on a split edge become both halves.
    keys = work.record_keys
    half = _edge_half(topology, keys[:, 0], keys[:, 1], vertex_capacity)
    hit = _live_records(work) & (half >= 0) & selected[jnp.maximum(half, 0)]
    midpoint = vertex_slot[jnp.maximum(half, 0)]
    ranks = jnp.where(hit, jnp.maximum(work.record_ranks, _REFINED), work.record_ranks)
    work, record_overflow = _store_records(
        work,
        _live_records(work),
        jnp.where(hit[:, None], _sorted_pair(keys[:, 0], midpoint), keys),
        work.record_sources,
        ranks,
        layout,
        appended=(hit, _sorted_pair(keys[:, 1], midpoint), work.record_sources, ranks),
    )
    bits = bits | _overflow(vertex_overflow | cell_overflow | record_overflow)
    return work, done, bits, count


def _split_round_periodic(work: _Work, done: Array, layout: DeviceMetricLayout, /) -> Any:
    """Split complete lifted edge orbits in one atomic device batch."""
    work = _compact_cells(work, layout)
    topology = _topology(work, layout)
    vertex_capacity = layout.vertex_capacity
    halves = jnp.arange(3 * layout.cell_capacity, dtype=jnp.int32)
    representative = work.periodic_edge_representative
    representative_mask = (
        topology.canonical & (representative == halves) & (representative >= 0)
    )
    candidate = topology.canonical & (topology.lengths > _UPPER) & ~topology.protected
    first = jnp.maximum(work.periodic_edge_start, 0)
    second = jnp.maximum(work.periodic_edge_end, 0)
    points = work.coordinates
    raw_t = _metric_midpoint(work.metric, points, first, second)
    t = jnp.where(
        candidate,
        raw_t[jnp.maximum(representative, 0)],
        0.5,
    )
    # Every member midpoint is the exact deck image of its representative's.
    anchored = points[first] + t[:, None] * (points[second] - points[first])
    middle = jnp.where(
        (representative >= 0)[:, None],
        anchored[jnp.maximum(representative, 0)] + work.periodic_edge_translation,
        anchored,
    )
    sides = _split_sides(work, topology)
    point = middle[:, None, :]
    checks = jnp.maximum(
        _orientation(points[sides.u], point, points[sides.apex]),
        _orientation(point, points[sides.v], points[sides.apex]),
    )
    status = jnp.max(jnp.where(sides.valid, checks, _OK), axis=1)
    counters, bits = _tally(work.counters, status, candidate)
    deck = work.periodic_edge_deck
    valid = candidate & (status == _OK) & deck
    rank = _ranks(
        representative_mask & valid,
        -topology.lengths,
        work.periodic_edge_orbits,
    )
    items = jnp.where(sides.valid, sides.cell, -1)
    selected_representatives, selected, overlap = _periodic_edge_batch(
        work, topology, valid, items, rank, layout
    )
    unresolved = jnp.any(overlap) | jnp.any(candidate & ~deck)
    bits = bits | jnp.where(
        unresolved,
        int(AdaptiveSimplexStatus.NEEDS_HOST_RESOLUTION),
        0,
    ).astype(jnp.int32)
    count = jnp.sum(selected, dtype=jnp.int32)
    group_count = jnp.sum(selected_representatives, dtype=jnp.int32)
    order = jnp.cumsum(selected, dtype=jnp.int32) - 1
    group_order = jnp.cumsum(selected_representatives, dtype=jnp.int32) - 1
    cursors = work.cursors
    vertex_slot = jnp.where(
        selected, cursors[_VERTICES].astype(jnp.int32) + order, vertex_capacity
    )
    tensors = _log_euclidean(
        jnp.stack((work.metric[first], work.metric[second]), axis=1),
        jnp.stack((1.0 - t, t), axis=1),
    )
    # Every member takes its representative's tensor (translations act trivially).
    tensors = jnp.where(
        (representative >= 0)[:, None, None],
        tensors[jnp.maximum(representative, 0)],
        tensors,
    )
    # Producer-owned orbit identity: kind 1 names the representative birth, and
    # each member carries its split edge member's absolute deck exponents.
    birth_keys = jnp.stack(
        (
            jnp.ones_like(order, dtype=jnp.int64),
            cursors[_NEXT_VERTEX] + order[jnp.maximum(representative, 0)],
        ),
        axis=1,
    )
    vertex_overflow = cursors[_VERTICES] + count > vertex_capacity
    work = work._replace(
        coordinates=work.coordinates.at[vertex_slot].set(middle, mode="drop"),
        metric=work.metric.at[vertex_slot].set(tensors, mode="drop"),
        vertex_ids=work.vertex_ids.at[vertex_slot].set(
            cursors[_NEXT_VERTEX] + order, mode="drop"
        ),
        vertex_active=work.vertex_active.at[vertex_slot].set(True, mode="drop"),
        periodic_vertex_key=work.periodic_vertex_key.at[vertex_slot].set(
            birth_keys, mode="drop"
        ),
        periodic_vertex_winding=work.periodic_vertex_winding.at[vertex_slot].set(
            work.periodic_edge_winding, mode="drop"
        ),
        cursors=cursors.at[_VERTICES]
        .add(count)
        .at[_NEXT_VERTEX]
        .add(count)
        .at[_NEXT_QUOTIENT_VERTEX]
        .add(group_count),
        counters=counters.at[_SPLITS].add(count),
    )
    children, removed = _split_children(
        work, sides, selected, vertex_slot, layout.cell_capacity
    )
    work, cell_overflow = _replace_cells(work, removed, children, layout)

    group_rep = jnp.maximum(representative, 0)
    member_edges = work.periodic_edge_members[group_rep]
    member_live = selected_representatives[:, None] & (member_edges >= 0)
    member_slots = vertex_slot[jnp.maximum(member_edges, 0)]
    representative_slots = vertex_slot[group_rep]
    quotient_order = group_order[group_rep]
    quotient_ids = cursors[_NEXT_QUOTIENT_VERTEX] + quotient_order
    vertex_orbits = work.periodic_vertex_orbits.at[
        jnp.where(member_live, member_slots, vertex_capacity)
    ].set(jnp.broadcast_to(quotient_ids[:, None], member_slots.shape), mode="drop")
    vertex_representatives = work.periodic_vertex_representative.at[
        jnp.where(member_live, member_slots, vertex_capacity)
    ].set(
        jnp.broadcast_to(representative_slots[:, None], member_slots.shape), mode="drop"
    )
    vertex_members = work.periodic_vertex_members.at[
        jnp.where(selected_representatives, representative_slots, vertex_capacity)
    ].set(jnp.where(member_edges >= 0, member_slots, -1), mode="drop")
    work = work._replace(
        periodic_vertex_orbits=vertex_orbits,
        periodic_vertex_representative=vertex_representatives,
        periodic_vertex_members=vertex_members,
    )

    keys = work.record_keys
    half = _edge_half(topology, keys[:, 0], keys[:, 1], vertex_capacity)
    hit = _live_records(work) & (half >= 0) & selected[jnp.maximum(half, 0)]
    midpoint_slot = vertex_slot[jnp.maximum(half, 0)]
    ranks = jnp.where(hit, jnp.maximum(work.record_ranks, _REFINED), work.record_ranks)
    work, record_overflow = _store_records(
        work,
        _live_records(work),
        jnp.where(hit[:, None], _sorted_pair(keys[:, 0], midpoint_slot), keys),
        work.record_sources,
        ranks,
        layout,
        appended=(
            hit,
            _sorted_pair(keys[:, 1], midpoint_slot),
            work.record_sources,
            ranks,
        ),
    )
    bits = bits | _overflow(vertex_overflow | cell_overflow | record_overflow)
    return work, done, bits, count


def _split_round(work: _Work, done: Array, layout: DeviceMetricLayout, /) -> Any:
    def periodic(operands: Any) -> Any:
        return _split_round_periodic(operands[0], operands[1], layout)

    def ordinary(operands: Any) -> Any:
        return _split_round_nonperiodic(operands[0], operands[1], layout)

    if work.periodic_enabled:
        return periodic((work, done))
    return ordinary((work, done))


# ------------------------------------------------------------------ collapse


class _CollapseCandidates(NamedTuple):
    """Both directions of every half-edge: ``[removed = low | removed = high]``."""

    edge: Array
    removed: Array
    kept: Array
    ball: Array
    holds_kept: Array
    moved: Array
    longest: Array
    status: Array
    # Classification of the removed end (straight curve vertices certified),
    # the link condition, and a ball wider than the layout.
    admissible: Array
    linked: Array
    cavity: Array


def _collapse_candidates(
    work: _Work, topology: _Topology, layout: DeviceMetricLayout, escape: Any, /
) -> _CollapseCandidates:
    """Classification, link condition, moved-ball orientation, and new lengths.

    Length admission as on the host route: a regular collapse that removes or
    keeps a birth must keep every new edge at or below the split threshold,
    other collapses up to ``_COLLAPSE_LIMIT``; the stall escape (``_adapted``)
    admits every collapse up to ``_COLLAPSE_LIMIT``.
    """
    vertex_capacity = layout.vertex_capacity
    width = layout.ball_width
    halves = jnp.arange(3 * layout.cell_capacity, dtype=jnp.int32)
    edge = jnp.concatenate((halves, halves))
    removed = jnp.concatenate((topology.ends[:, 0], topology.ends[:, 1]))
    kept = jnp.concatenate((topology.ends[:, 1], topology.ends[:, 0]))
    # Classification: interior vertices along unclassified edges inside one
    # region; straight curve vertices along an edge of their own curve class.
    label = topology.classes[edge]
    kind = topology.kind[removed]
    interior = (kind == _INTERIOR) & (label == 0) & topology.uniform[removed]
    curve = (kind == _CURVE) & (label > 0) & (label == topology.curve_class[removed])
    straight = jnp.where(curve, topology.straight[removed], _OK)
    cavity = (topology.valence[removed] > width) | (topology.valence[kept] > width)
    ball = topology.ball[removed]
    present = ball >= 0
    rows = work.cells[jnp.maximum(ball, 0)]
    holds_kept = jnp.any(rows == kept[:, None, None], axis=2)
    modified = present & ~holds_kept
    moved = jnp.where(rows == removed[:, None, None], kept[:, None, None], rows)
    points = work.coordinates
    orientation = jnp.max(
        jnp.where(
            modified,
            _orientation(
                points[moved[..., 0]], points[moved[..., 1]], points[moved[..., 2]]
            ),
            _OK,
        ),
        axis=1,
    )
    # Link condition: the common neighbors are exactly the apexes of the edge;
    # every other neighbor of the removed vertex forms a new edge.
    link = topology.link[removed]
    listed = (link >= 0) & (link != kept[:, None])
    joined = (
        _edge_half(
            topology,
            jnp.broadcast_to(kept[:, None], link.shape),
            jnp.maximum(link, 0),
            vertex_capacity,
        )
        >= 0
    )
    interior_edge = topology.neighbors.reshape((-1,))[edge] >= 0
    linked = jnp.sum(listed & joined, axis=1) == 1 + interior_edge
    fresh = listed & ~joined
    created = _lengths(
        work,
        _sorted_pair(
            jnp.broadcast_to(kept[:, None], link.shape), jnp.maximum(link, 0)
        ).reshape((-1, 2)),
    ).reshape(link.shape)
    longest = jnp.max(jnp.where(fresh, created, 0.0), axis=1)
    births = (work.source_rows[removed] < 0) | (work.source_rows[kept] < 0)
    limit = jnp.where(escape, _COLLAPSE_LIMIT, jnp.where(births, _UPPER, _COLLAPSE_LIMIT))
    geometric = jnp.maximum(
        jnp.maximum(orientation, straight),
        jnp.where(linked & (longest <= limit), _OK, _INVALID),
    )
    status = jnp.where(
        interior | curve, jnp.where(cavity, _CAVITY, geometric), _INADMISSIBLE
    )
    return _CollapseCandidates(
        edge,
        removed,
        kept,
        ball,
        holds_kept,
        moved,
        longest,
        status,
        interior | (curve & (straight == _OK)),
        linked,
        cavity,
    )


def _collapse_children(
    work: _Work,
    topology: _Topology,
    found: _CollapseCandidates,
    selected: Array,
    layout: DeviceMetricLayout,
    /,
) -> tuple[_Children, Array]:
    """Survivors of each removed ball move onto the kept vertex as new cells.

    Each survivor absorbs the dropped edge cells of its own region; a survivor
    edge (removed, apex) merges with the dropped cell's (kept, apex) edge and
    keeps the stronger classification.
    """
    width = layout.ball_width
    cell_capacity = layout.cell_capacity
    ball = found.ball
    present = ball >= 0
    lost = selected[:, None] & present & found.holds_kept
    lost_order = jnp.sort(
        jnp.where(lost, jnp.arange(width, dtype=jnp.int32), width), axis=1
    )[:, :2]
    lost_cells = jnp.where(
        lost_order < width,
        jnp.take_along_axis(ball, jnp.minimum(lost_order, width - 1), axis=1),
        -1,
    )
    regions = work.cell_regions
    safe_ball = jnp.maximum(ball, 0)
    absorbed = jnp.where(
        (lost_cells[:, None, :] >= 0)
        & (
            regions[jnp.maximum(lost_cells, 0)][:, None, :]
            == regions[safe_ball][:, :, None]
        ),
        lost_cells[:, None, :],
        -1,
    )
    neighbor = topology.neighbors[safe_ball]
    neighbor_cell = jnp.where(neighbor >= 0, neighbor // 3, -1)
    merging = (neighbor_cell >= 0) & jnp.any(
        (neighbor_cell[..., None] == lost_cells[:, None, None, :])
        & (lost_cells[:, None, None, :] >= 0),
        axis=-1,
    )
    safe_neighbor = jnp.maximum(neighbor_cell, 0)
    position = jnp.argmax(
        work.cells[safe_neighbor] == found.removed[:, None, None, None], axis=-1
    ).astype(jnp.int32)
    own_classes = work.edge_classes[safe_ball]
    own_protected = work.edge_protected[safe_ball]
    children = _Children(
        mask=(selected[:, None] & present & ~found.holds_kept).reshape((-1,)),
        rows=found.moved.reshape((-1, 3)),
        parents=jnp.concatenate((ball[..., None], absorbed), axis=2).reshape((-1, 3)),
        rank=_COLLAPSED,
        classes=jnp.where(
            merging,
            jnp.maximum(own_classes, work.edge_classes[safe_neighbor, position]),
            own_classes,
        ).reshape((-1, 3)),
        protected=(
            own_protected | (merging & work.edge_protected[safe_neighbor, position])
        ).reshape((-1, 3)),
    )
    replaced = (
        jnp.zeros((cell_capacity + 1,), dtype=jnp.bool_)
        .at[jnp.where(selected[:, None] & present, ball, cell_capacity)]
        .set(True)[:cell_capacity]
    )
    return children, replaced


def _apply_collapses(
    work: _Work,
    topology: _Topology,
    found: _CollapseCandidates,
    selected: Array,
    layout: DeviceMetricLayout,
    /,
) -> tuple[_Work, Array]:
    """Collapse each selected direction's removed vertex onto its kept vertex."""
    vertex_capacity = layout.vertex_capacity
    children, replaced = _collapse_children(work, topology, found, selected, layout)
    work, cell_overflow = _replace_cells(work, replaced, children, layout)
    target = jnp.where(selected, found.removed, vertex_capacity)
    work = work._replace(
        vertex_active=work.vertex_active.at[target].set(False, mode="drop"),
        collapse_to=work.collapse_to.at[target].set(found.kept, mode="drop"),
    )
    # Records follow the removed vertex; the collapsed edge's records vanish.
    remap = (
        jnp.arange(vertex_capacity, dtype=jnp.int32)
        .at[target]
        .set(found.kept, mode="drop")
    )
    gone = (
        jnp.zeros((vertex_capacity,), dtype=jnp.bool_).at[target].set(True, mode="drop")
    )
    keys = work.record_keys
    remapped = jnp.sort(remap[keys], axis=1)
    touched = jnp.any(gone[keys], axis=1)
    work, record_overflow = _store_records(
        work,
        _live_records(work) & (remapped[:, 0] != remapped[:, 1]),
        remapped,
        work.record_sources,
        jnp.where(touched, jnp.maximum(work.record_ranks, _COLLAPSED), work.record_ranks),
        layout,
    )
    return work, _overflow(cell_overflow | record_overflow)


def _collapse_round_nonperiodic(
    work: _Work, done: Array, layout: DeviceMetricLayout, escape: Any, /
) -> Any:
    work = _compact_cells(work, layout)
    topology = _topology(work, layout)
    halves_count = 3 * layout.cell_capacity
    short = topology.canonical & (topology.lengths < _LOWER) & ~topology.protected
    found = _collapse_candidates(work, topology, layout, escape)
    counters, bits = _tally(
        work.counters,
        jnp.minimum(found.status[:halves_count], found.status[halves_count:]),
        short,
    )
    valid = jnp.concatenate((short, short)) & (found.status == _OK)
    removed, kept = found.removed, found.kept
    ids = work.vertex_ids
    rank = _ranks(
        valid, topology.lengths[found.edge], found.longest, ids[removed], ids[kept]
    )
    selected = _independent(
        rank,
        valid,
        jnp.concatenate((found.ball, topology.ball[kept]), axis=1),
        layout.cell_capacity,
    )
    count = jnp.sum(selected, dtype=jnp.int32)
    work = work._replace(counters=counters.at[_COLLAPSES].add(count))
    work, overflow = _apply_collapses(work, topology, found, selected, layout)
    return work, done, bits | overflow, count


class _QuotientLink(NamedTuple):
    """Quotient link condition of every collapse direction (periodic rounds).

    ``exact``: every vertex involved is a pure-translation deck image, so a
    lifted edge ``(a, b)`` is the quotient edge ``(orbit a, orbit b, t_b - t_a)``.
    ``conflict``: the collapse would make a quotient loop or duplicate a
    quotient edge some other copy of the kept orbit already has. ``guard``: the
    balls of every kept copy, whose links the decision read.
    """

    exact: Array
    conflict: Array
    guard: Array


def _quotient_link(
    work: _Work,
    topology: _Topology,
    found: _CollapseCandidates,
    layout: DeviceMetricLayout,
    /,
) -> _QuotientLink:
    """Decide the link condition on the quotient, not only on the lifted cover.

    Moving the removed vertex onto lifted kept copy ``k`` turns each fresh link
    vertex ``x`` into the quotient edge ``(orbit k, orbit x, t_x - t_k)``. It
    duplicates an existing edge exactly when some kept copy ``p`` has a lifted
    neighbor ``z`` of orbit ``x`` with ``t_z - t_p = t_x - t_k`` (every quotient
    edge has a lifted member). Rounded differences of exact translations agree
    whenever the exact differences do, so a false match only refuses.
    """
    vertex_capacity = layout.vertex_capacity
    removed, kept = found.removed, found.kept
    orbit = work.periodic_vertex_orbits
    translation = work.periodic_vertex_translation
    deck = work.periodic_vertex_deck

    def copies(vertex: Array, /) -> Array:
        return work.periodic_vertex_members[
            jnp.maximum(work.periodic_vertex_representative[vertex], 0)
        ]

    link = topology.link[removed]
    listed = (link >= 0) & (link != kept[:, None])
    neighbor = jnp.maximum(link, 0)
    kept_rows = jnp.broadcast_to(kept[:, None], link.shape)
    fresh = listed & (_edge_half(topology, kept_rows, neighbor, vertex_capacity) < 0)
    loop = jnp.any(
        listed
        & (
            (orbit[neighbor] == orbit[kept][:, None])
            | (orbit[neighbor] == orbit[removed][:, None])
        ),
        axis=1,
    )
    kept_copies = copies(kept)
    kept_present = kept_copies >= 0
    safe_kept = jnp.maximum(kept_copies, 0)
    neighbor_copies = copies(neighbor)
    neighbor_present = neighbor_copies >= 0
    safe_neighbor = jnp.maximum(neighbor_copies, 0)
    # Axes: [direction, link vertex, kept copy, link vertex copy].
    offset = translation[neighbor] - translation[kept][:, None, :]
    relative = (
        translation[safe_neighbor][:, :, None, :, :]
        - translation[safe_kept][:, None, :, None, :]
    )
    matched = (
        fresh[:, :, None, None]
        & kept_present[:, None, :, None]
        & neighbor_present[:, :, None, :]
        & jnp.all(relative == offset[:, :, None, None, :], axis=-1)
    )
    adjacent = (
        _edge_half(
            topology,
            jnp.broadcast_to(safe_kept[:, None, :, None], matched.shape),
            jnp.broadcast_to(safe_neighbor[:, :, None, :], matched.shape),
            vertex_capacity,
        )
        >= 0
    )
    duplicate = jnp.any(matched & adjacent, axis=(1, 2, 3))
    exact = (
        jnp.all(~kept_present | deck[safe_kept], axis=1)
        & jnp.all(~listed | deck[neighbor], axis=1)
        & jnp.all(
            ~(listed[..., None] & neighbor_present) | deck[safe_neighbor], axis=(1, 2)
        )
    )
    guard = jnp.where(kept_present[..., None], topology.ball[safe_kept], -1).reshape(
        (kept.shape[0], -1)
    )
    return _QuotientLink(exact, loop | duplicate, guard)


def _collapse_round_periodic(
    work: _Work, done: Array, layout: DeviceMetricLayout, escape: Any, /
) -> Any:
    """Collapse only when every lifted member of one quotient edge is valid."""
    work = _compact_cells(work, layout)
    topology = _topology(work, layout)
    edge_count = 3 * layout.cell_capacity
    halves = jnp.arange(edge_count, dtype=jnp.int32)
    found = _collapse_candidates(work, topology, layout, escape)
    quotient = _quotient_link(work, topology, found, layout)
    short = topology.canonical & (topology.lengths < _LOWER) & ~topology.protected
    counters, bits = _tally(
        work.counters,
        jnp.minimum(found.status[:edge_count], found.status[edge_count:]),
        short,
    )
    start_is_first = topology.ends[:, 0] == work.periodic_edge_start
    start_direction = jnp.where(start_is_first, halves, halves + edge_count)
    end_direction = jnp.where(start_is_first, halves + edge_count, halves)
    start_valid = short & (found.status[start_direction] == _OK)
    end_valid = short & (found.status[end_direction] == _OK)
    representative = work.periodic_edge_representative
    group_representative = (
        topology.canonical & (representative == halves) & (representative >= 0)
    )
    members = work.periodic_edge_members
    present = members >= 0
    safe_members = jnp.maximum(members, 0)
    vertex_orbit_size = jnp.sum(
        work.periodic_vertex_members[jnp.maximum(work.periodic_vertex_representative, 0)]
        >= 0,
        axis=1,
        dtype=jnp.int32,
    )

    def covers_vertex_orbits(direction: Array, /) -> Array:
        # The edge orbit's lifted members must remove every lifted copy of the
        # removed vertex, otherwise the collapse is a cover collapse that only
        # the host route performs. Landing on only some copies of the kept
        # vertex needs the exact quotient link condition; every exactly
        # decided collapse must pass it.
        removed = found.removed[direction][safe_members]
        kept = found.kept[direction][safe_members]
        exact = jnp.all(~present | quotient.exact[direction][safe_members], axis=1)
        clear = jnp.all(~present | ~quotient.conflict[direction][safe_members], axis=1)
        kept_cover = _distinct_count(kept, present) == vertex_orbit_size[kept[:, 0]]
        return (
            (_distinct_count(removed, present) == vertex_orbit_size[removed[:, 0]])
            & (kept_cover | exact)
            & (clear | ~exact)
        )

    start_members = (
        group_representative
        & jnp.any(present, axis=1)
        & jnp.all(~present | start_valid[safe_members], axis=1)
    )
    end_members = (
        group_representative
        & jnp.any(present, axis=1)
        & jnp.all(~present | end_valid[safe_members], axis=1)
    )
    start_vertex = work.periodic_edge_start
    end_vertex = work.periodic_edge_end
    start_orbit = work.periodic_vertex_orbits[jnp.maximum(start_vertex, 0)]
    end_orbit = work.periodic_vertex_orbits[jnp.maximum(end_vertex, 0)]
    # An edge joining a vertex to its own deck image is a quotient loop; it
    # cannot be collapsed without changing the quotient topology.
    loop = start_orbit == end_orbit
    start_all = start_members & ~loop & covers_vertex_orbits(start_direction)
    end_all = end_members & ~loop & covers_vertex_orbits(end_direction)
    cover_only = (start_members | end_members) & ~loop & ~start_all & ~end_all
    start_longest = jnp.max(
        jnp.where(present, found.longest[start_direction[safe_members]], 0.0),
        axis=1,
    )
    end_longest = jnp.max(
        jnp.where(present, found.longest[end_direction[safe_members]], 0.0),
        axis=1,
    )
    stable_start = (start_orbit < end_orbit) | (
        (start_orbit == end_orbit)
        & (
            work.vertex_ids[jnp.maximum(start_vertex, 0)]
            <= work.vertex_ids[jnp.maximum(end_vertex, 0)]
        )
    )
    choose_start = start_all & (
        ~end_all
        | (start_longest < end_longest)
        | ((start_longest == end_longest) & stable_start)
    )
    group_valid = start_all | end_all
    choose_member_start = choose_start[jnp.maximum(representative, 0)]
    chosen_direction = jnp.where(choose_member_start, start_direction, end_direction)
    chosen_valid = (
        jnp.where(choose_member_start, start_valid, end_valid)
        & group_valid[jnp.maximum(representative, 0)]
    )
    start_items = jnp.concatenate(
        (
            found.ball[start_direction],
            topology.ball[found.kept[start_direction]],
        ),
        axis=1,
    )
    end_items = jnp.concatenate(
        (
            found.ball[end_direction],
            topology.ball[found.kept[end_direction]],
        ),
        axis=1,
    )
    items = jnp.where(choose_member_start[:, None], start_items, end_items)
    chosen_longest = jnp.where(choose_start, start_longest, end_longest)
    rank = _ranks(
        group_representative & group_valid,
        topology.lengths,
        chosen_longest,
        start_orbit,
        end_orbit,
        work.periodic_edge_orbits,
    )
    selected_representatives, selected_edges, overlap = _periodic_edge_batch(
        work,
        topology,
        chosen_valid,
        items,
        rank,
        layout,
        quotient.guard[chosen_direction],
    )
    bits = bits | jnp.where(
        jnp.any(overlap) | jnp.any(cover_only),
        int(AdaptiveSimplexStatus.NEEDS_HOST_RESOLUTION),
        0,
    ).astype(jnp.int32)
    selected = (
        jnp.zeros((2 * edge_count + 1,), dtype=jnp.bool_)
        .at[jnp.where(selected_edges, chosen_direction, 2 * edge_count)]
        .set(True, mode="drop")[: 2 * edge_count]
    )
    count = jnp.sum(selected, dtype=jnp.int32)
    work = work._replace(counters=counters.at[_COLLAPSES].add(count))
    work, overflow = _apply_collapses(work, topology, found, selected, layout)
    return work, done, bits | overflow, count


def _optimal_constant_remainder(
    work: _Work, topology: _Topology, kept: Array, removed: Array, /
) -> tuple[Array, Array, Array, Array, Array]:
    """Search an interior edge's constant-metric star at floating resolution.

    Interval operations are rounded outwards; uncertain predicate and terminal
    boxes stay unresolved, rather than being treated as exclusions.  The stack
    bounds the depth of binary subdivision of the parameter format, including
    subnormals; it is not a work or scientific depth limit.
    """
    points = work.coordinates
    dtype = points.dtype
    info = np.finfo(dtype)
    capacity = int(info.nmant - info.minexp + 4)
    origin, destination = points[kept], points[removed]
    ring = jnp.concatenate((topology.link[kept], topology.link[removed]))
    present = (ring >= 0) & (ring != kept) & (ring != removed)
    neighbor = jnp.maximum(ring, 0)
    cells = jnp.concatenate((topology.ball[kept], topology.ball[removed]))
    corners = work.cells[jnp.maximum(cells, 0)]
    hit = (corners == kept) | (corners == removed)
    star = (cells >= 0) & ~(
        jnp.any(corners == kept, axis=1) & jnp.any(corners == removed, axis=1)
    )
    tensor = work.metric[kept]
    constant = jnp.all(
        ~present[:, None, None] | (work.metric[neighbor] == tensor)
    ) & jnp.all(work.metric[removed] == tensor)

    def down(value: Array) -> Array:
        return jnp.nextafter(value, jnp.asarray(-jnp.inf, dtype=dtype))

    def up(value: Array) -> Array:
        return jnp.nextafter(value, jnp.asarray(jnp.inf, dtype=dtype))

    def multiply(al: Array, ah: Array, bl: Array, bh: Array) -> tuple[Array, Array]:
        values = jnp.stack((al * bl, al * bh, ah * bl, ah * bh))
        return down(jnp.min(values, axis=0)), up(jnp.max(values, axis=0))

    def point_box(lo: Array, hi: Array) -> tuple[Array, Array]:
        # Enclose executed subtraction, multiplication and addition, including
        # a fused multiply-add implementation.
        difference = destination - origin
        dl, dh = down(difference), up(difference)
        pl, ph = multiply(lo, hi, dl, dh)
        return down(origin + pl), up(origin + ph)

    def bounds(lo: Array, hi: Array) -> tuple[Array, Array]:
        point_low, point_high = point_box(lo, hi)
        delta_low = down(points[neighbor] - point_high)
        delta_high = up(points[neighbor] - point_low)
        lower = jnp.zeros(ring.shape, dtype=dtype)
        upper = jnp.zeros(ring.shape, dtype=dtype)
        for i in range(2):
            for j in range(2):
                a, b = multiply(
                    delta_low[:, i], delta_high[:, i], delta_low[:, j], delta_high[:, j]
                )
                a, b = multiply(a, b, tensor[i, j], tensor[i, j])
                lower, upper = down(lower + a), up(upper + b)
        return lower, upper

    def geometry_excluded(lo: Array, hi: Array) -> Array:
        point_low, point_high = point_box(lo, hi)
        low = jnp.where(hit[..., None], point_low, points[corners])
        high = jnp.where(hit[..., None], point_high, points[corners])
        u_low, u_high = down(low[:, 1] - high[:, 0]), up(high[:, 1] - low[:, 0])
        v_low, v_high = down(low[:, 2] - high[:, 0]), up(high[:, 2] - low[:, 0])
        _, positive = multiply(u_low[:, 0], u_high[:, 0], v_low[:, 1], v_high[:, 1])
        negative, _ = multiply(u_low[:, 1], u_high[:, 1], v_low[:, 0], v_high[:, 0])
        return jnp.any(star & (up(positive - negative) <= 0.0))

    def body(carry: Any) -> Any:
        size, lows, highs, found, point, score, uncertain, overflow = carry
        index = size - 1
        lo, hi = lows[index], highs[index]
        size = index
        low, _ = bounds(lo, hi)
        excluded = jnp.any(present & (low > 2.0)) | geometry_excluded(lo, hi)
        middle = lo + (hi - lo) * 0.5
        terminal = (middle == lo) | (middle == hi)
        proposal = origin + middle * (destination - origin)
        _, upper = bounds(middle, middle)
        moved = jnp.where(hit[..., None], proposal, points[corners])
        orientation = _orientation(moved[:, 0], moved[:, 1], moved[:, 2])
        geometry = jnp.max(jnp.where(star, orientation, _OK))
        pairs = jnp.stack((neighbor, jnp.full_like(neighbor, points.shape[0])), axis=1)
        measured = metric_edge_lengths(
            jnp.concatenate((work.metric, tensor[None])),
            jnp.concatenate((points, proposal[None])),
            pairs,
        )
        candidate_score = jnp.max(jnp.where(present, measured, 0.0))
        admitted = (
            ~excluded
            & ~terminal
            & (geometry == _OK)
            & jnp.all(~present | (upper <= 2.0))
            & jnp.all(jnp.isfinite(upper))
            & jnp.isfinite(candidate_score)
            & (candidate_score <= _UPPER)
        )
        subdivide = ~excluded & ~terminal & ~admitted
        fits = size + 2 <= capacity
        low_slot = jnp.minimum(size, capacity - 1)
        high_slot = jnp.minimum(size + 1, capacity - 1)
        lows = lows.at[low_slot].set(jnp.where(subdivide & fits, middle, lows[low_slot]))
        highs = highs.at[low_slot].set(jnp.where(subdivide & fits, hi, highs[low_slot]))
        lows = lows.at[high_slot].set(jnp.where(subdivide & fits, lo, lows[high_slot]))
        highs = highs.at[high_slot].set(
            jnp.where(subdivide & fits, middle, highs[high_slot])
        )
        return (
            size + jnp.where(subdivide & fits, 2, 0),
            lows,
            highs,
            found | admitted,
            jnp.where(admitted, proposal, point),
            jnp.where(admitted, candidate_score, score),
            uncertain | (~excluded & terminal),
            overflow | (subdivide & ~fits),
        )

    initial = (
        jnp.int32(1),
        jnp.zeros((capacity,), dtype=dtype),
        jnp.zeros((capacity,), dtype=dtype).at[0].set(1.0),
        jnp.bool_(False),
        jnp.zeros((2,), dtype=dtype),
        jnp.asarray(jnp.inf, dtype=dtype),
        ~constant,
        jnp.bool_(False),
    )
    result = jax.lax.while_loop(
        lambda state: (state[0] > 0) & ~state[3] & ~state[6] & ~state[7],
        body,
        initial,
    )
    return result[3], result[4], result[5], result[6], result[7]


def _optimal_collapse_round(
    work: _Work, done: Array, layout: DeviceMetricLayout, /
) -> Any:
    """Optimal-point collapses, the host route's ``_optimal_collapse_search``.

    A short admissible edge first tries the existing finite seed lattice on
    its permitted edge or curve segments.  A missed nonperiodic interior edge
    with an identical current star tensor is searched by outward interval
    subdivision down to floating parameter resolution. Its actual binary64
    point is authoritative, and commit reconstructs its exact source stencil.
    A box is discarded only by a proved metric-band exclusion; otherwise a
    filtered positive star and a certified band admit a point, or numerical
    resolution returns ``NEEDS_HOST_RESOLUTION``. Nonconstant, curve, and
    periodic remainders remain unresolved. Seed proposals are ranked by their
    longest star edge, with the first segment and lowest lattice index tying.
    The kept end is a source vertex rather than a birth where possible, then
    the lower vertex ID. Eligible seeds are searched in bounded-memory blocks.
    """
    work = _compact_cells(work, layout)
    topology = _topology(work, layout)
    vertex_capacity = layout.vertex_capacity
    edge_count = 3 * layout.cell_capacity
    halves = jnp.arange(edge_count, dtype=jnp.int32)
    points = work.coordinates
    found = _collapse_candidates(work, topology, layout, True)
    short = topology.canonical & (topology.lengths < _LOWER) & ~topology.protected
    label = topology.classes

    def movable(direction: Array, /) -> tuple[Array, Array]:
        kept = found.kept[direction]
        kind = topology.kind[kept]
        curve = (kind == _CURVE) & (topology.straight[kept] == _OK)
        along_edge = ((kind == _INTERIOR) & (label == 0) & topology.uniform[kept]) | (
            curve & (label > 0) & (topology.curve_class[kept] == label)
        )
        along_curve = (
            curve & (label == 0) & (topology.kind[found.removed[direction]] == _INTERIOR)
        )
        valid = found.admissible[direction] & found.linked[direction]
        valid = valid & ~found.cavity[direction]
        return valid & (along_edge | along_curve), along_curve & ~along_edge

    forward_ok, forward_slide = movable(halves)
    backward_ok, backward_slide = movable(halves + edge_count)
    low, high = topology.ends[:, 0], topology.ends[:, 1]
    births = work.source_rows < 0
    ids = work.vertex_ids
    keep_high = (births[high] < births[low]) | (
        (births[high] == births[low]) & (ids[high] < ids[low])
    )
    forward = forward_ok & (~backward_ok | keep_high)
    direction = jnp.where(forward, halves, halves + edge_count)
    sliding = jnp.where(forward, forward_slide, backward_slide)
    removed = found.removed[direction]
    kept = found.kept[direction]
    eligible = short & (forward_ok | backward_ok)
    periodic_unresolved = jnp.bool_(False)
    if work.periodic_enabled:
        representative = work.periodic_vertex_representative
        size = jnp.where(
            representative >= 0,
            jnp.sum(
                work.periodic_vertex_members[jnp.maximum(representative, 0)] >= 0,
                axis=1,
                dtype=jnp.int32,
            ),
            1,
        )
        eligible = (
            eligible
            & (size[removed] == 1)
            & (size[kept] == 1)
            & work.periodic_vertex_deck[removed]
            & work.periodic_vertex_deck[kept]
            & (topology.neighbors.reshape((-1,)) >= 0)
        )
        periodic_unresolved = jnp.any(short & (forward_ok | backward_ok) & ~eligible)
    steps = _OPTIMAL_LATTICE - 1
    lanes = 2 * steps
    lattice = jnp.arange(1, _OPTIMAL_LATTICE, dtype=points.dtype) / _OPTIMAL_LATTICE
    block = min(_OPTIMAL_BLOCK, edge_count)
    order = jnp.argsort(~eligible, stable=True).astype(jnp.int32)
    total = jnp.sum(eligible, dtype=jnp.int32)
    lane = jnp.arange(lanes, dtype=jnp.int32)
    merged = vertex_capacity + jnp.arange(block, dtype=jnp.int32)[:, None] * lanes + lane

    def search(carry: Any) -> Any:
        start, longest_all, point_all, metric_all = carry
        rows = jax.lax.dynamic_slice(
            order, (jnp.minimum(start, edge_count - block),), (block,)
        )
        live = eligible[rows]
        kept_rows, removed_rows = kept[rows], removed[rows]
        slide = sliding[rows]
        kept_neighbors = jnp.maximum(topology.curve_neighbors[kept_rows], 0)
        # Segment targets: the removed end (second segment absent), or the
        # kept curve vertex's two curve neighbors.
        targets = jnp.stack(
            (
                jnp.where(slide, kept_neighbors[:, 0], removed_rows),
                jnp.where(slide, kept_neighbors[:, 1], kept_rows),
            ),
            axis=1,
        )
        absent = jnp.repeat(
            jnp.stack((jnp.zeros_like(slide), ~slide), axis=1), steps, axis=1
        )
        origin = points[kept_rows]
        candidate = (
            origin[:, None, None, :]
            + lattice[None, None, :, None]
            * (points[targets] - origin[:, None, :])[:, :, None, :]
        ).reshape((block, lanes, 2))
        weight = lattice[None, None, :, None, None]
        proposal = hermitian_exp(
            (1.0 - weight) * hermitian_log(work.metric[kept_rows]).value[:, None, None]
            + weight * hermitian_log(work.metric[targets]).value[:, :, None]
        ).value
        identical = jnp.all(
            work.metric[kept_rows, None] == work.metric[targets], axis=(-2, -1)
        )
        proposal = jnp.where(
            identical[:, :, None, None, None],
            work.metric[kept_rows, None, None],
            proposal,
        ).reshape((block, lanes, 2, 2))
        extended_points = jnp.concatenate((points, candidate.reshape((-1, 2))))
        extended_metric = jnp.concatenate((work.metric, proposal.reshape((-1, 2, 2))))
        # Star edges: every neighbor of either end but the ends themselves.
        ring = jnp.concatenate(
            (topology.link[kept_rows], topology.link[removed_rows]), axis=1
        )
        ring_present = (
            (ring >= 0) & (ring != kept_rows[:, None]) & (ring != removed_rows[:, None])
        )
        pairs = jnp.stack(
            jnp.broadcast_arrays(jnp.maximum(ring, 0)[:, None, :], merged[:, :, None]),
            axis=-1,
        )
        lengths = metric_edge_lengths(
            extended_metric, extended_points, pairs.reshape((-1, 2))
        ).reshape(pairs.shape[:-1])
        longest = jnp.max(jnp.where(ring_present[:, None, :], lengths, 0.0), axis=2)
        # Star cells: every ball cell of either end but the edge cells.
        cells = jnp.concatenate(
            (topology.ball[kept_rows], topology.ball[removed_rows]), axis=1
        )
        corners = work.cells[jnp.maximum(cells, 0)]
        hit_kept = corners == kept_rows[:, None, None]
        hit_removed = corners == removed_rows[:, None, None]
        star = (cells >= 0) & ~(jnp.any(hit_kept, axis=2) & jnp.any(hit_removed, axis=2))
        moved = jnp.where(
            (hit_kept | hit_removed)[:, None], merged[:, :, None, None], corners[:, None]
        )
        orientation = _orientation(
            extended_points[moved[..., 0]],
            extended_points[moved[..., 1]],
            extended_points[moved[..., 2]],
        )
        status = jnp.max(jnp.where(star[:, None], orientation, _OK), axis=2)
        removed_neighbors = topology.curve_neighbors[removed_rows]
        # The merged point's curve neighbors: the outer curve neighbor of each
        # end along a curve edge, the kept vertex's own when sliding.
        outer_kept = jnp.where(
            slide | (kept_neighbors[:, 0] != removed_rows),
            kept_neighbors[:, 0],
            kept_neighbors[:, 1],
        )
        outer_removed = jnp.where(
            slide,
            kept_neighbors[:, 1],
            jnp.where(
                removed_neighbors[:, 0] == kept_rows,
                removed_neighbors[:, 1],
                removed_neighbors[:, 0],
            ),
        )
        collinear = _collinear(
            jnp.broadcast_to(
                points[jnp.maximum(outer_kept, 0)][:, None], candidate.shape
            ),
            candidate,
            jnp.broadcast_to(
                points[jnp.maximum(outer_removed, 0)][:, None], candidate.shape
            ),
        )
        status = jnp.where(
            (topology.kind[kept_rows] == _CURVE)[:, None],
            jnp.maximum(status, collinear),
            status,
        )
        status = jnp.where(absent, _INVALID, status)
        finite = jnp.all(~ring_present[:, None, :] | jnp.isfinite(lengths), axis=2)
        score = jnp.where((status == _OK) & finite, longest, jnp.inf)
        best = jnp.argmin(score, axis=1)
        chosen = jnp.arange(block)
        target = jnp.where(live, rows, edge_count)
        return (
            start + block,
            longest_all.at[target].set(score[chosen, best], mode="drop"),
            point_all.at[target].set(candidate[chosen, best], mode="drop"),
            metric_all.at[target].set(proposal[chosen, best], mode="drop"),
        )

    _, longest, point, metric = jax.lax.while_loop(
        lambda carry: carry[0] < total,
        search,
        (
            jnp.int32(0),
            jnp.full((edge_count,), jnp.inf, dtype=points.dtype),
            jnp.zeros((edge_count, 2), dtype=points.dtype),
            jnp.zeros((edge_count, 2, 2), dtype=points.dtype),
        ),
    )

    def remainder(carry: Any) -> Any:
        index, longest_all, point_all, metric_all, unresolved, resource = carry
        row = order[index]
        missing = ~jnp.isfinite(longest_all[row]) | (longest_all[row] > _UPPER)
        supported = (
            (topology.kind[kept[row]] == _INTERIOR)
            & ~sliding[row]
            & jnp.bool_(not work.periodic_enabled)
        )

        def attempt(_: Any) -> Any:
            return _optimal_constant_remainder(work, topology, kept[row], removed[row])

        result = jax.lax.cond(
            missing & supported,
            attempt,
            lambda _: (
                jnp.bool_(False),
                jnp.zeros((2,), dtype=points.dtype),
                jnp.asarray(jnp.inf, dtype=points.dtype),
                missing & ~supported,
                jnp.bool_(False),
            ),
            operand=None,
        )
        found_point, proposal, score, uncertain, overflow = result
        return (
            index + 1,
            longest_all.at[row].set(jnp.where(found_point, score, longest_all[row])),
            point_all.at[row].set(jnp.where(found_point, proposal, point_all[row])),
            metric_all.at[row].set(
                jnp.where(found_point, work.metric[kept[row]], metric_all[row])
            ),
            unresolved | uncertain,
            resource | overflow,
        )

    _, longest, point, metric, unresolved, resource = jax.lax.while_loop(
        lambda carry: carry[0] < total,
        remainder,
        (jnp.int32(0), longest, point, metric, periodic_unresolved, jnp.bool_(False)),
    )
    admitted = eligible & jnp.isfinite(longest) & (longest <= _UPPER)
    rank = _ranks(admitted, topology.lengths, longest, ids[removed], ids[kept])
    selected = _independent(
        rank,
        admitted,
        jnp.concatenate((topology.ball[removed], topology.ball[kept]), axis=1),
        layout.cell_capacity,
    )
    count = jnp.sum(selected, dtype=jnp.int32)
    target = jnp.where(selected, kept, vertex_capacity)
    moved = (
        jnp.zeros((vertex_capacity,), dtype=jnp.bool_).at[target].set(True, mode="drop")
    )
    touched_cells = jnp.any(moved[work.cells], axis=1)
    touched_records = _live_records(work) & jnp.any(moved[work.record_keys], axis=1)
    work = work._replace(
        coordinates=points.at[target].set(point, mode="drop"),
        metric=work.metric.at[target].set(metric, mode="drop"),
        vertex_moved=work.vertex_moved | moved,
        cell_ranks=jnp.where(
            touched_cells, jnp.maximum(work.cell_ranks, _RELOCATED), work.cell_ranks
        ),
        record_ranks=jnp.where(
            touched_records,
            jnp.maximum(work.record_ranks, _RELOCATED),
            work.record_ranks,
        ),
        counters=work.counters.at[_COLLAPSES].add(count),
    )
    directions = (
        jnp.zeros((2 * edge_count + 1,), dtype=jnp.bool_)
        .at[jnp.where(selected, direction, 2 * edge_count)]
        .set(True, mode="drop")[: 2 * edge_count]
    )
    work, overflow = _apply_collapses(work, topology, found, directions, layout)
    bits = jnp.where(
        unresolved & (count == 0),
        int(AdaptiveSimplexStatus.NEEDS_HOST_RESOLUTION),
        0,
    ).astype(jnp.int32)
    return work, done, overflow | bits | _overflow(resource), count


def _collapse_round(
    work: _Work, done: Array, layout: DeviceMetricLayout, mode: Any = _REGULAR, /
) -> Any:
    """One collapse round of ``mode`` (traced or static): ``_REGULAR`` endpoint
    collapses, the ``_OPTIMAL`` stall-escape attempt (``_optimal_collapse_round``),
    or the ``_ENDPOINT`` stall-escape attempt (every endpoint collapse up to
    ``_COLLAPSE_LIMIT``, births included)."""

    def endpoint(escape: bool, /) -> Any:
        def run(operand: Any) -> Any:
            work, done = operand
            if work.periodic_enabled:
                return _collapse_round_periodic(work, done, layout, escape)
            return _collapse_round_nonperiodic(work, done, layout, escape)

        return run

    def optimal(operand: tuple[_Work, Array]) -> Any:
        return _optimal_collapse_round(operand[0], operand[1], layout)

    branches = (endpoint(False), optimal, endpoint(True))
    if isinstance(mode, int):
        return branches[mode]((work, done))
    return jax.lax.switch(mode, branches, (work, done))


# ------------------------------------------------------------------ flip


def _flip_children(
    work: _Work,
    halves: Array,
    other: Array,
    triangles: Array,
    selected: Array,
    capacity: int,
    /,
) -> tuple[_Children, Array]:
    """Children (u, x1, x0) and (v, x0, x1); the new diagonal is unclassified."""
    first_cell, first_local = halves // 3, halves % 3
    second_cell, second_local = other // 3, other % 3

    def child_features(values: Any) -> Any:
        first = values[first_cell]
        second = values[second_cell]
        fresh = jnp.zeros_like(first[:, 0])
        return jnp.stack(
            (
                jnp.stack(
                    (
                        fresh,
                        _take(first, (first_local + 2) % 3),
                        _take(second, (second_local + 1) % 3),
                    ),
                    axis=-1,
                ),
                jnp.stack(
                    (
                        fresh,
                        _take(second, (second_local + 2) % 3),
                        _take(first, (first_local + 1) % 3),
                    ),
                    axis=-1,
                ),
            ),
            axis=1,
        ).reshape((-1, 3))

    none = jnp.full(first_cell.shape, -1, dtype=jnp.int32)
    children = _Children(
        mask=jnp.repeat(selected, 2),
        rows=triangles.reshape((-1, 3)),
        parents=jnp.stack(
            (
                jnp.stack((first_cell, second_cell, none), axis=-1),
                jnp.stack((second_cell, first_cell, none), axis=-1),
            ),
            axis=1,
        ).reshape((-1, 3)),
        rank=_SWAPPED,
        classes=child_features(work.edge_classes),
        protected=child_features(work.edge_protected),
    )
    replaced = (
        jnp.zeros((capacity + 1,), dtype=jnp.bool_)
        .at[jnp.where(selected, first_cell, capacity)]
        .set(True)
        .at[jnp.where(selected, second_cell, capacity)]
        .set(True)[:capacity]
    )
    return children, replaced


def _flip_round_nonperiodic(
    work: _Work, done: Array, layout: DeviceMetricLayout, /
) -> Any:
    work = _compact_cells(work, layout)
    topology = _topology(work, layout)
    vertex_capacity = layout.vertex_capacity
    cell_capacity = layout.cell_capacity
    halves = jnp.arange(3 * cell_capacity, dtype=jnp.int32)
    other = topology.neighbors.reshape((-1,))
    safe_other = jnp.maximum(other, 0)
    first_cell, first_local = halves // 3, halves % 3
    second_cell, second_local = safe_other // 3, safe_other % 3
    regions = work.cell_regions
    candidate = (
        topology.canonical
        & (other >= 0)
        & (topology.classes == 0)
        & ~topology.protected
        & (regions[first_cell] == regions[second_cell])
    )
    first_rows = work.cells[first_cell]
    u = _take(first_rows, (first_local + 1) % 3)
    v = _take(first_rows, (first_local + 2) % 3)
    x0 = _take(first_rows, first_local)
    x1 = _take(work.cells[second_cell], second_local)
    triangles = jnp.stack(
        (jnp.stack((u, x1, x0), axis=-1), jnp.stack((v, x0, x1), axis=-1)), axis=1
    )
    lengths = _lengths(
        work,
        _sorted_pair(triangles[..., _NEXT], triangles[..., _PREVIOUS]).reshape((-1, 2)),
    ).reshape(triangles.shape)
    quality = _quality(lengths)
    after = jnp.minimum(quality[:, 0], quality[:, 1])
    before = jnp.minimum(topology.quality[first_cell], topology.quality[second_cell])
    points = work.coordinates
    status = jnp.maximum(
        _orientation(points[u], points[x1], points[x0]),
        _orientation(points[v], points[x0], points[x1]),
    )
    duplicate = _edge_half(topology, x0, x1, vertex_capacity) >= 0
    status = jnp.where(duplicate, _INVALID, status)
    diagonal_low = jnp.minimum(lengths[:, 0, 1], lengths[:, 1, 1])
    diagonal_high = jnp.maximum(lengths[:, 0, 1], lengths[:, 1, 1])
    metric_admissible = (diagonal_low >= _LOWER) & (diagonal_high <= _UPPER)
    # Exclude a flip that the next metric pass would immediately refine/coarsen.
    improving = candidate & (after > before + _IMPROVEMENT) & metric_admissible
    counters, bits = _tally(work.counters, status, improving)
    valid = improving & (status == _OK)
    ids = work.vertex_ids
    rank = _ranks(
        valid, before - after, ids[topology.ends[:, 0]], ids[topology.ends[:, 1]]
    )
    selected = _independent(
        rank, valid, jnp.stack((first_cell, second_cell), axis=1), cell_capacity
    )
    count = jnp.sum(selected, dtype=jnp.int32)
    work = work._replace(counters=counters.at[_FLIPS].add(count))
    children, replaced = _flip_children(
        work, halves, safe_other, triangles, selected, cell_capacity
    )
    work, cell_overflow = _replace_cells(work, replaced, children, layout)
    keys = work.record_keys
    half = _edge_half(topology, keys[:, 0], keys[:, 1], vertex_capacity)
    flipped = (half >= 0) & selected[jnp.maximum(half, 0)]
    work, record_overflow = _store_records(
        work,
        _live_records(work) & ~flipped,
        keys,
        work.record_sources,
        work.record_ranks,
        layout,
    )
    bits = bits | _overflow(cell_overflow | record_overflow)
    return work, done, bits, count


def _flip_round_periodic(work: _Work, done: Array, layout: DeviceMetricLayout, /) -> Any:
    """Flip a quotient edge only after validating every lifted cavity."""
    work = _compact_cells(work, layout)
    topology = _topology(work, layout)
    vertex_capacity = layout.vertex_capacity
    cell_capacity = layout.cell_capacity
    halves = jnp.arange(3 * cell_capacity, dtype=jnp.int32)
    other = topology.neighbors.reshape((-1,))
    safe_other = jnp.maximum(other, 0)
    first_cell, first_local = halves // 3, halves % 3
    second_cell, second_local = safe_other // 3, safe_other % 3
    regions = work.cell_regions
    candidate = (
        topology.canonical
        & (other >= 0)
        & (topology.classes == 0)
        & ~topology.protected
        & (regions[first_cell] == regions[second_cell])
    )
    first_rows = work.cells[first_cell]
    u = _take(first_rows, (first_local + 1) % 3)
    v = _take(first_rows, (first_local + 2) % 3)
    x0 = _take(first_rows, first_local)
    x1 = _take(work.cells[second_cell], second_local)
    triangles = jnp.stack(
        (jnp.stack((u, x1, x0), axis=-1), jnp.stack((v, x0, x1), axis=-1)),
        axis=1,
    )
    lengths = _lengths(
        work,
        _sorted_pair(triangles[..., _NEXT], triangles[..., _PREVIOUS]).reshape((-1, 2)),
    ).reshape(triangles.shape)
    quality = _quality(lengths)
    after = jnp.minimum(quality[:, 0], quality[:, 1])
    before = jnp.minimum(topology.quality[first_cell], topology.quality[second_cell])
    points = work.coordinates
    status = jnp.maximum(
        _orientation(points[u], points[x1], points[x0]),
        _orientation(points[v], points[x0], points[x1]),
    )
    duplicate = _edge_half(topology, x0, x1, vertex_capacity) >= 0
    status = jnp.where(duplicate, _INVALID, status)
    diagonal_low = jnp.minimum(lengths[:, 0, 1], lengths[:, 1, 1])
    diagonal_high = jnp.maximum(lengths[:, 0, 1], lengths[:, 1, 1])
    metric_admissible = (diagonal_low >= _LOWER) & (diagonal_high <= _UPPER)
    improving = candidate & (after > before + _IMPROVEMENT) & metric_admissible
    counters, bits = _tally(work.counters, status, improving)
    valid = improving & (status == _OK)
    representative = work.periodic_edge_representative
    group_representative = (
        topology.canonical & (representative == halves) & (representative >= 0)
    )
    rank = _ranks(
        group_representative & valid,
        before - after,
        work.periodic_edge_orbits,
        work.vertex_ids[topology.ends[:, 0]],
        work.vertex_ids[topology.ends[:, 1]],
    )
    items = jnp.stack((first_cell, second_cell), axis=1)
    selected_representatives, selected, overlap = _periodic_edge_batch(
        work, topology, valid, items, rank, layout
    )
    bits = bits | jnp.where(
        jnp.any(overlap),
        int(AdaptiveSimplexStatus.NEEDS_HOST_RESOLUTION),
        0,
    ).astype(jnp.int32)
    count = jnp.sum(selected, dtype=jnp.int32)
    work = work._replace(counters=counters.at[_FLIPS].add(count))
    children, replaced = _flip_children(
        work, halves, safe_other, triangles, selected, cell_capacity
    )
    work, cell_overflow = _replace_cells(work, replaced, children, layout)
    keys = work.record_keys
    half = _edge_half(topology, keys[:, 0], keys[:, 1], vertex_capacity)
    flipped = (half >= 0) & selected[jnp.maximum(half, 0)]
    work, record_overflow = _store_records(
        work,
        _live_records(work) & ~flipped,
        keys,
        work.record_sources,
        work.record_ranks,
        layout,
    )
    bits = bits | _overflow(cell_overflow | record_overflow)
    return work, done, bits, count


def _flip_round(work: _Work, done: Array, layout: DeviceMetricLayout, /) -> Any:
    def periodic(operands: Any) -> Any:
        return _flip_round_periodic(operands[0], operands[1], layout)

    def ordinary(operands: Any) -> Any:
        return _flip_round_nonperiodic(operands[0], operands[1], layout)

    if work.periodic_enabled:
        return periodic((work, done))
    return ordinary((work, done))


# ------------------------------------------------------------------ relocation


class _VertexOrbits(NamedTuple):
    """Lifted members of each vertex's periodic orbit (a vertex outside one is its own).

    ``anchor`` is the orbit representative whose position every member's deck
    translation is measured from.
    """

    members: Array
    present: Array
    anchor: Array


def _vertex_orbits(work: _Work, /) -> _VertexOrbits:
    representative = work.periodic_vertex_representative
    slots = jnp.arange(representative.shape[0], dtype=jnp.int32)
    grouped = work.periodic_enabled & (representative >= 0)
    own = jnp.full_like(work.periodic_vertex_members, -1).at[:, 0].set(slots)
    members = jnp.where(
        grouped[:, None],
        work.periodic_vertex_members[jnp.maximum(representative, 0)],
        own,
    )
    return _VertexOrbits(members, members >= 0, jnp.where(grouped, representative, slots))


def _orbit_values(orbits: _VertexOrbits, values: Array, fill: Any, /) -> Array:
    """``values`` of every orbit member, ``fill`` in absent member columns."""
    gathered = values[jnp.maximum(orbits.members, 0)]
    present = orbits.present.reshape(orbits.present.shape + (1,) * (values.ndim - 1))
    return jnp.where(present, gathered, fill)


def _relocation_targets(
    work: _Work, topology: _Topology, orbits: _VertexOrbits, /
) -> Any:
    """Spring displacement toward unit metric lengths and each vertex's badness.

    Interior vertices use every neighbor; curve vertices only their two curve
    neighbors, so the displacement stays on the straight curve line. A periodic
    orbit sums the translation-invariant pulls of all lifted members (its
    quotient star), so every member moves by one common displacement.
    """
    points = work.coordinates
    link = topology.link
    listed = link >= 0
    length = jnp.where(listed, topology.lengths[topology.link_half], 1.0)
    badness = jnp.max(jnp.where(listed, jnp.maximum(length, 1.0 / length), 0.0), axis=1)
    curve = topology.kind == _CURVE
    along = (link == topology.curve_neighbors[:, :1]) | (
        link == topology.curve_neighbors[:, 1:]
    )
    use = listed & (~curve[:, None] | along)
    pull = jnp.where(
        use[..., None],
        (1.0 - 1.0 / length)[..., None]
        * (points[jnp.maximum(link, 0)] - points[:, None, :]),
        0.0,
    )
    pulls = jnp.sum(_orbit_values(orbits, jnp.sum(pull, axis=1), 0.0), axis=1)
    uses = jnp.sum(_orbit_values(orbits, jnp.sum(use, axis=1), 0), axis=1)
    displacement = pulls / jnp.maximum(uses, 1)[:, None]
    return displacement, jnp.max(_orbit_values(orbits, badness, 0.0), axis=1)


def _periodic_vertex_selection(
    work: _Work,
    topology: _Topology,
    active: Array,
    success: Array,
    best_scale: Array,
    badness: Array,
    layout: DeviceMetricLayout,
    /,
) -> tuple[Array, Array]:
    """Select complete vertex orbits with one common representable step."""
    vertices = layout.vertex_capacity
    slots = jnp.arange(vertices, dtype=jnp.int32)
    representative = work.periodic_vertex_representative
    group_representative = (
        work.vertex_active & (representative == slots) & (representative >= 0)
    )
    members = work.periodic_vertex_members
    present = members >= 0
    safe_members = jnp.maximum(members, 0)
    member_success = success[safe_members]
    member_active = active[safe_members]
    attempted = group_representative & jnp.any(present & member_active, axis=1)
    all_success = (
        group_representative
        & jnp.any(present, axis=1)
        & jnp.all(~present | member_success, axis=1)
    )
    member_scale = best_scale[safe_members]
    scale_agrees = jnp.all(~present | (member_scale == best_scale[:, None]), axis=1)
    partial = attempted & jnp.any(present & member_success, axis=1) & ~all_success
    scale_mismatch = all_success & ~scale_agrees
    balls = topology.ball[safe_members]
    cavity_present = present[..., None] & (balls >= 0)
    cavity = jnp.where(cavity_present, balls, -1).reshape((vertices, -1))
    ordered = jnp.sort(jnp.where(cavity >= 0, cavity, layout.cell_capacity), axis=1)
    overlap = jnp.any(
        (ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, :-1] < layout.cell_capacity),
        axis=1,
    )
    eligible = all_success & scale_agrees & ~overlap
    rank = _ranks(eligible, -badness, work.periodic_vertex_orbits)
    selected_representatives = _independent(rank, eligible, cavity, layout.cell_capacity)
    selected = (
        (representative >= 0)
        & work.vertex_active
        & selected_representatives[jnp.maximum(representative, 0)]
    )
    unresolved = jnp.any(partial | scale_mismatch | (attempted & overlap))
    bits = jnp.where(
        unresolved,
        int(AdaptiveSimplexStatus.NEEDS_HOST_RESOLUTION),
        0,
    ).astype(jnp.int32)
    return selected, bits


def _relocation_round(work: _Work, done: Array, layout: DeviceMetricLayout, /) -> Any:
    """Move vertices (whole periodic orbits) toward their spring targets.

    A periodic orbit is decided on its quotient star: the representative moves
    by the orbit displacement, every member lands on the exact deck image of
    that position and takes one metric proposal, and the move is accepted when
    the minimum quality over all member balls improves with every member cell
    certified positive. Orbits whose deck element is not a representable
    translation are left to the host route.
    """
    topology = _topology(work, layout)
    vertex_capacity = layout.vertex_capacity
    cell_capacity = layout.cell_capacity
    points = work.coordinates
    orbits = _vertex_orbits(work)

    def every(values: Array, /) -> Array:
        return jnp.all(_orbit_values(orbits, values, True), axis=1)

    def some(values: Array, /) -> Array:
        return jnp.any(_orbit_values(orbits, values, False), axis=1)

    kind = topology.kind
    curve = kind == _CURVE
    movable = every(
        (topology.valence > 0)
        & ~done
        & ((kind == _INTERIOR) | (curve & (topology.straight != _INVALID)))
    )
    cavity = some(topology.valence > layout.ball_width)
    displacement, badness = _relocation_targets(work, topology, orbits)
    candidate = movable & ~cavity & jnp.any(displacement != 0.0, axis=1)
    deck = every(work.periodic_vertex_deck) | (not work.periodic_enabled)
    active = candidate & deck
    ball = topology.ball
    present = ball >= 0
    rows = work.cells[jnp.maximum(ball, 0)]
    before = jnp.min(
        _orbit_values(
            orbits,
            jnp.min(
                jnp.where(present, topology.quality[jnp.maximum(ball, 0)], jnp.inf),
                axis=1,
            ),
            jnp.inf,
        ),
        axis=1,
    )

    def targets(scale: Array, /) -> Array:
        step = points + scale * displacement
        if work.periodic_enabled:
            return step[orbits.anchor] + work.periodic_vertex_translation
        return step

    initial_point = points
    initial_metric = work.metric
    initial_scale = jnp.asarray(1.0, dtype=points.dtype)
    initial_success = jnp.zeros((vertex_capacity,), dtype=jnp.bool_)
    initial_invalid = jnp.zeros((vertex_capacity,), dtype=jnp.bool_)
    initial_uncertain = jnp.zeros((vertex_capacity,), dtype=jnp.bool_)
    initial_pending = active

    def continue_search(carry: Any) -> Array:
        _, pending, _, _, _, _, _, _ = carry
        return jnp.any(pending)

    def search_step(carry: Any) -> Any:
        (
            scale,
            pending,
            success,
            best_point,
            best_metric,
            best_scale,
            had_invalid,
            had_uncertain,
        ) = carry
        target = targets(scale)
        weights = _barycentric(points[rows], target[:, None, :])
        score = jnp.where(present, jnp.min(weights, axis=-1), -jnp.inf)
        best_cell = jnp.argmax(score, axis=1)
        cell_index = jnp.broadcast_to(best_cell[:, None, None], (vertex_capacity, 1, 3))
        host_rows = jnp.take_along_axis(rows, cell_index, axis=1)[:, 0]
        host_weights = jnp.take_along_axis(weights, cell_index, axis=1)[:, 0]
        clipped = jnp.maximum(host_weights, 0.0)
        normalized = jnp.where(
            active[:, None],
            clipped / jnp.sum(clipped, axis=-1, keepdims=True),
            1.0 / 3.0,
        )
        own_proposal = _log_euclidean(work.metric[host_rows], normalized)
        # One metric per orbit: the member whose ball best contains its target.
        chosen = jnp.argmax(
            _orbit_values(orbits, jnp.max(score, axis=1), -jnp.inf), axis=1
        )
        proposal = own_proposal[
            jnp.take_along_axis(jnp.maximum(orbits.members, 0), chosen[:, None], axis=1)[
                :, 0
            ]
        ]
        image = vertex_capacity + jnp.arange(vertex_capacity, dtype=jnp.int32)
        extended_points = jnp.concatenate((points, target), axis=0)
        extended_metric = jnp.concatenate((work.metric, proposal), axis=0)
        vertex = jnp.arange(vertex_capacity, dtype=jnp.int32)[:, None, None]
        moved_rows = jnp.where(rows == vertex, image[:, None, None], rows)
        orientation = _orientation(
            extended_points[moved_rows[..., 0]],
            extended_points[moved_rows[..., 1]],
            extended_points[moved_rows[..., 2]],
        )
        status = jnp.max(jnp.where(present, orientation, _OK), axis=1)
        neighbors = jnp.maximum(topology.curve_neighbors, 0)
        status = jnp.where(
            curve,
            jnp.maximum(
                status,
                _collinear(points[neighbors[:, 0]], target, points[neighbors[:, 1]]),
            ),
            status,
        )
        lengths = metric_edge_lengths(
            extended_metric,
            extended_points,
            _sorted_pair(moved_rows[..., _NEXT], moved_rows[..., _PREVIOUS]).reshape(
                (-1, 2)
            ),
        ).reshape(moved_rows.shape)
        after = jnp.min(
            jnp.where(
                present,
                _quality(jnp.where(present[..., None], lengths, 1.0)),
                jnp.inf,
            ),
            axis=1,
        )
        after = jnp.min(_orbit_values(orbits, after, jnp.inf), axis=1)
        status = jnp.max(_orbit_values(orbits, status, _OK), axis=1)
        improving = pending & (after > before + _IMPROVEMENT)
        accepted = improving & (status == _OK)
        best_point = jnp.where(accepted[:, None], target, best_point)
        best_metric = jnp.where(accepted[:, None, None], proposal, best_metric)
        best_scale = jnp.where(accepted, scale, best_scale)
        had_invalid = had_invalid | (improving & (status >= _INVALID))
        had_uncertain = had_uncertain | (improving & (status == _UNCERTAIN))
        pending = pending & ~accepted
        next_scale = scale * jnp.asarray(0.5, dtype=points.dtype)
        pending = pending & some(jnp.any(targets(next_scale) != points, axis=1))
        return (
            next_scale,
            pending,
            success | accepted,
            best_point,
            best_metric,
            best_scale,
            had_invalid,
            had_uncertain,
        )

    (
        _,
        _,
        success,
        best_point,
        best_metric,
        best_scale,
        had_invalid,
        had_uncertain,
    ) = jax.lax.while_loop(
        continue_search,
        search_step,
        (
            initial_scale,
            initial_pending,
            initial_success,
            initial_point,
            initial_metric,
            jnp.zeros((vertex_capacity,), dtype=points.dtype),
            initial_invalid,
            initial_uncertain,
        ),
    )
    rejected = jnp.where(
        had_invalid,
        _INVALID,
        jnp.where(had_uncertain, _UNCERTAIN, _OK),
    )
    counters, bits = _tally(work.counters, rejected, active & ~success)
    bits = bits | jnp.where(
        jnp.any(candidate & ~deck),
        int(AdaptiveSimplexStatus.NEEDS_HOST_RESOLUTION),
        0,
    ).astype(jnp.int32)
    counters, _ = _tally(counters, _CAVITY, movable & cavity)
    rank = _ranks(success, -badness, work.vertex_ids)
    if work.periodic_enabled:
        selected, periodic_bits = _periodic_vertex_selection(
            work, topology, active, success, best_scale, badness, layout
        )
        bits = bits | periodic_bits
    else:
        selected = _independent(rank, success, ball, cell_capacity)
    count = jnp.sum(selected, dtype=jnp.int32)
    touched_cells = jnp.any(selected[work.cells], axis=1)
    touched_records = _live_records(work) & jnp.any(selected[work.record_keys], axis=1)
    work = work._replace(
        coordinates=jnp.where(selected[:, None], best_point, points),
        metric=jnp.where(selected[:, None, None], best_metric, work.metric),
        vertex_moved=work.vertex_moved | selected,
        cell_ranks=jnp.where(
            touched_cells, jnp.maximum(work.cell_ranks, _RELOCATED), work.cell_ranks
        ),
        record_ranks=jnp.where(
            touched_records,
            jnp.maximum(work.record_ranks, _RELOCATED),
            work.record_ranks,
        ),
        counters=counters.at[_RELOCATIONS].add(count),
    )
    return work, done | selected, bits, count


# ------------------------------------------------------------------ passes


def _measured_lengths(work: _Work, /) -> tuple[Array, Array]:
    """Share measured-edge semantics across numerical admission and band checks."""
    lengths = _lengths(work, _half_ends(work.cells))
    active = jnp.repeat(work.cell_active, 3)
    unprotected = active & ~work.edge_protected.reshape((-1,))
    measured = jnp.where(jnp.any(unprotected), unprotected, active)
    return lengths, measured


def _metric_failure(work: _Work, /) -> Array:
    lengths, measured = _measured_lengths(work)
    invalid = jnp.any(measured & (~jnp.isfinite(lengths) | (lengths <= 0.0)))
    return jnp.where(invalid, int(AdaptiveSimplexStatus.NEEDS_HOST_RESOLUTION), 0).astype(
        jnp.int32
    )


def _unit(work: _Work, /) -> Array:
    """Every measured (unprotected, else every) active edge in the unit range."""
    lengths, measured = _measured_lengths(work)
    inside = (lengths >= _LOWER) & (lengths <= _UPPER)
    return jnp.all(~measured | inside)


def _stall_potential(work: _Work, /) -> Array:
    """Well-founded stall measure ``(out-of-band, quantized band excess)``.

    Counted over the measured half-edges of ``_unit``; both entries are natural
    numbers, so their lexicographic order admits no infinite strictly
    decreasing sequence (see ``_adapted``).
    """
    lengths, measured = _measured_lengths(work)
    outside = measured & ((lengths < _LOWER) | (lengths > _UPPER))
    excess = jnp.maximum(
        jnp.abs(jnp.log(jnp.where(measured, lengths, 1.0))) - np.log(_UPPER), 0.0
    )
    return jnp.stack(
        (
            jnp.sum(outside, dtype=jnp.int64),
            jnp.sum(jnp.floor(excess / _DEVIATION_QUANTUM).astype(jnp.int64)),
        )
    )


def _potential_below(potential: Array, bound: Array, /) -> Array:
    """Strict lexicographic order of two stall potentials."""
    return (potential[0] < bound[0]) | (
        (potential[0] == bound[0]) & (potential[1] < bound[1])
    )


def _phase(
    step: Any,
    work: _Work,
    done: Array,
    bits: Array,
    layout: Any,
    enabled: Any = True,
    /,
) -> Any:
    """Exhaust a finite phase batch until no operation applies or status stops it.

    A disabled (traced ``enabled``) phase runs no round.
    """

    def proceed(carry: Any) -> Any:
        _, _, bits, count, _, numerical_failure = carry
        return (count > 0) & ((bits & _FAILURES) == 0) & ~numerical_failure

    def sub_round(carry: Any) -> Any:
        work, done, bits, _, total, _ = carry
        work, done, round_bits, count = step(work, done, layout)
        numerical_status = _metric_failure(work)
        return (
            work,
            done,
            bits | round_bits | numerical_status,
            count,
            total + count,
            numerical_status != 0,
        )

    work, done, bits, _, total, numerical_failure = jax.lax.while_loop(
        proceed,
        sub_round,
        (
            work,
            done,
            bits,
            jnp.where(enabled, 1, 0).astype(jnp.int32),
            jnp.int32(0),
            jnp.bool_(False),
        ),
    )
    return work, done, bits, total, numerical_failure


def _pass(
    work: _Work, bits: Array, layout: DeviceMetricLayout, escaping: Array, /
) -> Any:
    """Split, collapse, and flip phases, then relocation (each vertex once).

    An escaping pass (``escaping``, a traced collapse mode, ``_adapted``) runs
    only the collapse phase of that stall-escape attempt; the collapse rounds
    are compiled once with the traced mode.
    """
    done = jnp.zeros((layout.vertex_capacity,), dtype=jnp.bool_)
    regular = escaping == _REGULAR

    def skipped(operand: Any) -> Any:
        work, bits = operand
        return work, bits, jnp.int32(0), jnp.bool_(False)

    def collapse_round(work: _Work, done: Array, layout: DeviceMetricLayout, /) -> Any:
        return _collapse_round(work, done, layout, escaping)

    def topological(operand: Any) -> Any:
        work, bits = operand
        total = jnp.int32(0)
        numerical_failure = jnp.bool_(False)
        for step, enabled in (
            (_split_round, regular),
            (collapse_round, True),
            (_flip_round, regular),
        ):
            work, _, bits, count, phase_failure = _phase(
                step,
                work,
                done,
                bits,
                layout,
                enabled & ~numerical_failure,
            )
            total = total + count
            numerical_failure |= phase_failure
        return work, bits, total, numerical_failure

    def relocating(operand: Any) -> Any:
        work, bits = operand
        work, _, bits, count, numerical_failure = _phase(
            _relocation_round,
            work,
            done,
            bits,
            layout,
        )
        return work, bits, count, numerical_failure

    work, bits, changed, numerical_failure = jax.lax.cond(
        work.controls[_TOPOLOGY] != 0, topological, skipped, (work, bits)
    )
    work, bits, moved, relocation_failure = jax.lax.cond(
        (work.controls[_RELOCATION] != 0) & regular & ~numerical_failure,
        relocating,
        skipped,
        (work, bits),
    )
    work = work._replace(counters=work.counters.at[_PASSES].add(jnp.where(regular, 1, 0)))
    return work, bits, changed + moved, numerical_failure | relocation_failure


def _geometry(work: _Work, /) -> tuple[Array, Array]:
    """FILTERED_DEVICE orientation of the active cells: uncertain and invalid counts."""
    corners = work.coordinates[work.cells]
    result = orient2d(
        corners[:, 0], corners[:, 1], corners[:, 2], mode=PredicateMode.FILTERED_DEVICE
    )
    active = work.cell_active
    positive = result.signs == int(PredicateSign.POSITIVE)
    return (
        jnp.sum(active & ~result.certain, dtype=jnp.int32),
        jnp.sum(active & result.certain & ~positive, dtype=jnp.int32),
    )


def _report(
    source: _Work,
    result: _Work,
    status: Array,
    stalled: Array,
    geometry: tuple[Array, Array],
    layout: DeviceMetricLayout,
    /,
) -> DeviceMetricReport:
    cells = result.cells
    active = result.cell_active
    other = masked_simplex_facet_neighbors(cells, active).reshape((-1,))
    halves = jnp.arange(other.shape[0], dtype=jnp.int32)
    canonical = jnp.repeat(active, 3) & ((other < 0) | (halves < other))
    lengths = _lengths(result, _half_ends(cells))
    unprotected = canonical & ~result.edge_protected.reshape((-1,))
    measured = jnp.where(jnp.any(unprotected), unprotected, canonical)
    inside = measured & (lengths >= _LOWER) & (lengths <= _UPPER)
    count = jnp.sum(measured, dtype=jnp.int32)
    within = jnp.sum(inside, dtype=jnp.int32)
    cell_lengths = jnp.where(
        active[:, None], lengths.reshape((layout.cell_capacity, 3)), 1.0
    )
    delta = (result.counters - source.counters).astype(jnp.int32)
    return DeviceMetricReport(
        status=status,
        passes=delta[_PASSES],
        splits=delta[_SPLITS],
        collapses=delta[_COLLAPSES],
        flips=delta[_FLIPS],
        relocations=delta[_RELOCATIONS],
        rejected_uncertain=delta[_REJECTED_UNCERTAIN],
        rejected_invalid=delta[_REJECTED_INVALID],
        rejected_cavity=delta[_REJECTED_CAVITY],
        converged=within == count,
        stalled=stalled,
        measured_edges=count,
        out_of_range_edges=count - within,
        minimum_metric_length=jnp.min(jnp.where(measured, lengths, jnp.inf)),
        maximum_metric_length=jnp.max(jnp.where(measured, lengths, 0.0)),
        unit_fraction=within.astype(jnp.float64) / jnp.maximum(count, 1),
        minimum_metric_quality=jnp.min(
            jnp.where(active, _quality(cell_lengths), jnp.inf)
        ),
        uncertain_cells=geometry[0],
        invalid_cells=geometry[1],
    )


class _DeviceMetricRound(NamedTuple):
    work: _Work
    report: DeviceMetricReport


def _periodic_round(
    layout: DeviceMetricLayout, source: _Work, operation: Array, /
) -> _DeviceMetricRound:
    """Execute one source-bound orbit transaction without host synchronization.

    Operations: split, collapse, flip, relocate, and the stall-escape attempts
    (4) endpoint collapses up to ``_COLLAPSE_LIMIT`` and (5) optimal-point
    collapses, which share the collapse branch through its traced mode.
    """
    done = jnp.zeros((layout.vertex_capacity,), dtype=jnp.bool_)
    initial_failure = _metric_failure(source)

    def split(operand: Any) -> Any:
        work, done = operand
        work, _, bits, count = _split_round(work, done, layout)
        return work, bits, count

    def collapse(operand: Any) -> Any:
        work, done = operand
        mode = jnp.where(
            operation == 4, _ENDPOINT, jnp.where(operation == 5, _OPTIMAL, _REGULAR)
        )
        work, _, bits, count = _collapse_round(work, done, layout, mode)
        return work, bits, count

    def flip(operand: Any) -> Any:
        work, done = operand
        work, _, bits, count = _flip_round(work, done, layout)
        return work, bits, count

    def relocate(operand: Any) -> Any:
        work, done = operand
        work, _, bits, count, _ = _phase(
            _relocation_round,
            work,
            done,
            jnp.int32(0),
            layout,
        )
        return work, bits, count

    work, bits, _ = jax.lax.cond(
        initial_failure == 0,
        lambda: jax.lax.switch(
            jnp.where(operation >= 4, 1, operation),
            (split, collapse, flip, relocate),
            (source, done),
        ),
        lambda: (source, initial_failure, jnp.int32(0)),
    )
    uncertain, invalid = _geometry(work)
    numerical_status = _metric_failure(work)
    bits = (
        bits
        | numerical_status
        | jnp.where(invalid > 0, int(AdaptiveSimplexStatus.INVALID_GEOMETRY), 0)
        | jnp.where(uncertain > 0, int(AdaptiveSimplexStatus.NEEDS_HOST_RESOLUTION), 0)
    ).astype(jnp.int32)
    failed = ((bits & _FAILURES) != 0) | (numerical_status != 0)
    capacity = int(AdaptiveSimplexStatus.CAPACITY_EXCEEDED)
    rollback = failed | ((bits & capacity) != 0)
    result = _select(rollback, source, work)
    result = result._replace(
        flags=result.flags.at[_STATUS]
        .set(source.flags[_STATUS] | bits)
        .at[_CONVERGED]
        .set(jnp.int32(0))
        .at[_STALLED]
        .set(jnp.int32(0))
    )
    geometry = jax.lax.cond(
        rollback,
        lambda: _geometry(source),
        lambda: (uncertain, invalid),
    )
    report = _report(source, result, bits, jnp.bool_(False), geometry, layout)
    cavity_growth = (
        work.counters[_REJECTED_CAVITY] - source.counters[_REJECTED_CAVITY]
    ).astype(jnp.int32)
    report = eqx.tree_at(
        lambda value: value.rejected_cavity,
        report,
        jnp.where((bits & capacity) != 0, cavity_growth, report.rejected_cavity),
    )
    return _DeviceMetricRound(result, report)


_compiled_periodic_round = eqx.filter_jit(_periodic_round)


def _adapt(layout: DeviceMetricLayout, state: DeviceMetricState, /) -> DeviceMetricUpdate:
    source = _work(state)
    return jax.lax.cond(
        (source.flags[_STATUS] & _FAILURES) != 0,
        lambda: _refused_call(layout, state, source),
        lambda: _adapted(layout, source),
    )


def _refused_call(
    layout: DeviceMetricLayout, state: DeviceMetricState, source: _Work, /
) -> DeviceMetricUpdate:
    """A call on an epoch that already failed: nothing runs, the state is kept."""

    status = (source.flags[_STATUS] & _FAILURES).astype(jnp.int32)
    report = _report(source, source, status, jnp.bool_(False), _geometry(source), layout)
    return DeviceMetricUpdate(state, report, layout)


def _adapted(layout: DeviceMetricLayout, source: _Work, /) -> DeviceMetricUpdate:
    """Run complete metric phases, rolling back only on actual terminal status.

    Stall escape (the host route's, ``execute_local_metric_adaptation``): at a
    pass that applies nothing outside the unit criterion, an escape phase runs
    optimal-point collapses (``_optimal_collapse_round``: the merged vertex at
    the edge or curve point minimizing the longest star edge, admitted at most
    sqrt(2)) and, once none applies, every endpoint collapse up to
    ``_COLLAPSE_LIMIT`` (births included). It stands when the run reaches the
    unit criterion or a stall of strictly lower ``_stall_potential``;
    otherwise the run returns the escaped stall's state and status bits as a
    stall. Termination: the measure is ``_stall_potential``, the pair
    (out-of-band measured edges, quantized band excess) of natural numbers;
    the escaped stalls' potentials strictly decrease in its well-founded
    lexicographic order, each escape phase removes a vertex per collapse, and
    between escapes the regular passes are unchanged.
    """

    def proceed(carry: Any) -> Any:
        work, bits, stalled = carry[:3]
        numerical_failure = carry[-1]
        return ~stalled & ~numerical_failure & ((bits & _FAILURES) == 0) & ~_unit(work)

    def adaptation_pass(carry: Any) -> Any:
        work, bits, _, escaping, escaped, bound, anchor, anchor_bits, attempt, _ = carry
        work, bits, applied, numerical_failure = _pass(work, bits, layout, escaping)
        quiet = applied == 0
        healthy = ((bits & _FAILURES) == 0) & ~numerical_failure
        potential = jax.lax.cond(healthy, lambda: _stall_potential(work), lambda: bound)
        stall = (
            quiet
            & (escaping == _REGULAR)
            & healthy
            & (work.controls[_TOPOLOGY] != 0)
            & ~_unit(work)
        )
        admitted = ~escaped | _potential_below(potential, bound)
        start = stall & admitted
        # A stall that does not lower the last escaped stall's potential
        # returns to that stall for its next escape attempt, or stalls there
        # when none is left; an optimal-point attempt that applies nothing
        # passes to the endpoint attempt, and an endpoint attempt that applies
        # nothing stalls.
        restore = stall & ~admitted
        retry = restore & (attempt == _OPTIMAL)
        advance = quiet & (escaping == _OPTIMAL) & healthy
        following = retry | advance
        return (
            _select(restore, anchor, work),
            jnp.where(restore, anchor_bits, bits),
            quiet & ~start & ~following,
            jnp.where(start, _OPTIMAL, jnp.where(following, _ENDPOINT, _REGULAR)).astype(
                jnp.int32
            ),
            escaped | start,
            jnp.where(start, potential, bound),
            _select(start, work, anchor),
            jnp.where(start, bits, anchor_bits),
            jnp.where(start, _OPTIMAL, jnp.where(following, _ENDPOINT, attempt)),
            numerical_failure,
        )

    # Host predicate resolution can accompany valid progress; invalid metric
    # arithmetic instead stops this numerical epoch and rolls it back.
    initial_numerical_status = _metric_failure(source)
    work, bits, stalled, *_, numerical_failure = jax.lax.while_loop(
        proceed,
        adaptation_pass,
        (
            source,
            initial_numerical_status,
            jnp.bool_(False),
            jnp.int32(_REGULAR),
            jnp.bool_(False),
            jnp.zeros((2,), dtype=jnp.int64),
            source,
            jnp.int32(0),
            jnp.int32(_REGULAR),
            initial_numerical_status != 0,
        ),
    )
    converged = _unit(work)
    stalled = stalled & ~converged
    uncertain, invalid = _geometry(work)
    bits = (
        bits
        | jnp.where(invalid > 0, int(AdaptiveSimplexStatus.INVALID_GEOMETRY), 0)
        | jnp.where(uncertain > 0, int(AdaptiveSimplexStatus.NEEDS_HOST_RESOLUTION), 0)
    ).astype(jnp.int32)
    failed = ((bits & _FAILURES) != 0) | numerical_failure
    capacity = int(AdaptiveSimplexStatus.CAPACITY_EXCEEDED)
    rollback = failed | ((bits & capacity) != 0)
    work = work._replace(
        flags=work.flags.at[_STATUS]
        .set(work.flags[_STATUS] | bits)
        .at[_CONVERGED]
        .set(converged.astype(jnp.int32))
        .at[_STALLED]
        .set(stalled.astype(jnp.int32))
    )
    result = _select(rollback, source, work)
    result = result._replace(
        flags=result.flags.at[_STATUS]
        .set(work.flags[_STATUS])
        .at[_CONVERGED]
        .set(jnp.where(rollback, jnp.int32(0), work.flags[_CONVERGED]))
        .at[_STALLED]
        .set(jnp.where(rollback, jnp.int32(0), work.flags[_STALLED]))
    )
    report = _report(
        source,
        result,
        bits,
        stalled & ~rollback,
        _select(rollback, _geometry(source), (uncertain, invalid)),
        layout,
    )
    cavity_growth = (
        work.counters[_REJECTED_CAVITY] - source.counters[_REJECTED_CAVITY]
    ).astype(jnp.int32)
    report = eqx.tree_at(
        lambda value: value.rejected_cavity,
        report,
        jnp.where((bits & capacity) != 0, cavity_growth, report.rejected_cavity),
    )
    return DeviceMetricUpdate(_state(result, layout, False, ""), report, layout)


_compiled_adapt = eqx.filter_jit(_adapt)


def _check_state(layout: DeviceMetricLayout, state: DeviceMetricState, /) -> None:
    if not isinstance(layout, DeviceMetricLayout):
        raise TypeError("layout must be DeviceMetricLayout.")
    if not isinstance(state, DeviceMetricState):
        raise TypeError("state must be DeviceMetricState.")
    if (
        state.mesh.signature_id != layout.mesh_signature_id
        or state.layout_signature_id != layout.signature_id
        or state.ball_width != layout.ball_width
        or state.record_keys.shape[0] != layout.record_capacity
        or state.periodic_vertex_members.shape[1] != layout.periodic_orbit_width
        or state.periodic_edge_members.shape[1] != layout.periodic_orbit_width
        or state.ancestry.shape[0] != layout.ancestry_capacity
    ):
        raise ValueError("The state does not belong to this device metric layout.")


def _grow_metric_layout(
    layout: DeviceMetricLayout,
    state: DeviceMetricState,
    *,
    grow_slots: bool,
    grow_ball: bool,
) -> DeviceMetricLayout:
    vertices = layout.vertex_capacity
    cells = layout.cell_capacity
    if grow_slots:
        vertices = max(vertices + 1, 2 * vertices)
        cells = max(cells + 1, 2 * cells)
    width = layout.ball_width
    if grow_ball:
        work = _work(state)
        active = np.asarray(jax.device_get(work.cell_active))
        rows = np.asarray(jax.device_get(work.cells))[active]
        valence = np.bincount(rows.reshape((-1,)), minlength=layout.vertex_capacity)
        required = int(np.max(valence, initial=0))
        width = max(width + 1, required)
        if grow_slots:
            width = max(width, min(cells, 2 * layout.ball_width))
    return DeviceMetricLayout(
        vertex_capacity=vertices,
        cell_capacity=cells,
        ball_width=min(width, cells),
        periodic_orbit_width=layout.periodic_orbit_width,
    )


def _resize_device_state(
    state: DeviceMetricState,
    layout: DeviceMetricLayout,
    /,
    *,
    clear_capacity: bool,
    clear_cavity: bool = False,
) -> DeviceMetricState:
    work = _work(state)
    vertices, cells = layout.vertex_capacity, layout.cell_capacity

    def padded(values: Array, capacity: int, fill: Any, /) -> Array:
        if values.shape[0] == capacity:
            return values
        current = np.asarray(jax.device_get(values))
        result = np.full((capacity, *current.shape[1:]), fill, dtype=current.dtype)
        result[: current.shape[0]] = current
        return jnp.asarray(result)

    def padded_members(values: Array, rows: int, width: int, /) -> Array:
        if values.shape == (rows, width):
            return values
        current = np.asarray(jax.device_get(values))
        result = np.full((rows, width), -1, dtype=current.dtype)
        result[: min(rows, current.shape[0]), : min(width, current.shape[1])] = current[
            :rows, :width
        ]
        return jnp.asarray(result)

    metric = work.metric
    if metric.shape[0] != vertices:
        current = np.asarray(jax.device_get(metric))
        result = np.broadcast_to(np.eye(2), (vertices, 2, 2)).copy()
        result[: current.shape[0]] = current
        metric = jnp.asarray(result)

    ancestry = work.ancestry
    if ancestry.shape[0] != layout.ancestry_capacity:
        pairs = int(np.asarray(jax.device_get(work.cursors))[_PAIRS])
        result = np.zeros((layout.ancestry_capacity, 2), dtype=np.int32)
        result[:, 0] = cells
        result[:pairs] = np.asarray(jax.device_get(ancestry))[:pairs]
        ancestry = jnp.asarray(result)

    resized = _Work(
        coordinates=padded(work.coordinates, vertices, 0.0),
        vertex_ids=padded(work.vertex_ids, vertices, -1),
        vertex_active=padded(work.vertex_active, vertices, False),
        cells=padded(work.cells, cells, 0),
        cell_ids=padded(work.cell_ids, cells, -1),
        cell_active=padded(work.cell_active, cells, False),
        periodic_enabled=work.periodic_enabled,
        metric=metric,
        source_rows=padded(work.source_rows, vertices, -1),
        vertex_fixed=padded(work.vertex_fixed, vertices, False),
        vertex_moved=padded(work.vertex_moved, vertices, False),
        collapse_to=padded(work.collapse_to, vertices, -1),
        cell_regions=padded(work.cell_regions, cells, 0),
        cell_blocks=padded(work.cell_blocks, cells, 0),
        cell_ranks=padded(work.cell_ranks, cells, 0),
        edge_classes=padded(work.edge_classes, cells, 0),
        edge_protected=padded(work.edge_protected, cells, False),
        record_keys=padded(work.record_keys, layout.record_capacity, 0),
        record_sources=padded(work.record_sources, layout.record_capacity, -1),
        record_ranks=padded(work.record_ranks, layout.record_capacity, 0),
        ancestry=ancestry,
        periodic_vertex_orbits=padded(work.periodic_vertex_orbits, vertices, -1),
        periodic_vertex_representative=padded(
            work.periodic_vertex_representative, vertices, -1
        ),
        periodic_vertex_members=padded_members(
            work.periodic_vertex_members, vertices, layout.periodic_orbit_width
        ),
        periodic_edge_orbits=padded(work.periodic_edge_orbits, 3 * cells, -1),
        periodic_edge_representative=padded(
            work.periodic_edge_representative, 3 * cells, -1
        ),
        periodic_edge_start=padded(work.periodic_edge_start, 3 * cells, -1),
        periodic_edge_end=padded(work.periodic_edge_end, 3 * cells, -1),
        periodic_edge_members=padded_members(
            work.periodic_edge_members,
            3 * cells,
            layout.periodic_orbit_width,
        ),
        periodic_cell_orbits=padded(work.periodic_cell_orbits, cells, -1),
        periodic_vertex_translation=padded(
            work.periodic_vertex_translation, vertices, 0.0
        ),
        periodic_vertex_deck=padded(work.periodic_vertex_deck, vertices, False),
        periodic_edge_translation=padded(work.periodic_edge_translation, 3 * cells, 0.0),
        periodic_edge_deck=padded(work.periodic_edge_deck, 3 * cells, False),
        periodic_vertex_key=padded(work.periodic_vertex_key, vertices, 0),
        periodic_vertex_winding=padded(work.periodic_vertex_winding, vertices, 0),
        periodic_edge_winding=padded(work.periodic_edge_winding, 3 * cells, 0),
        cursors=work.cursors,
        controls=work.controls,
        counters=work.counters,
        flags=work.flags,
    )
    if clear_capacity:
        flags = resized.flags.at[_STATUS].set(
            resized.flags[_STATUS]
            & jnp.asarray(~int(AdaptiveSimplexStatus.CAPACITY_EXCEEDED), dtype=jnp.int32)
        )
        resized = resized._replace(flags=flags)
    if clear_cavity:
        resized = resized._replace(
            counters=resized.counters.at[_REJECTED_CAVITY].set(
                jnp.zeros_like(resized.counters[_REJECTED_CAVITY])
            )
        )
    return _state(
        resized,
        layout,
        state.periodic_enabled,
        state.periodic_mesh_id,
        _periodic_source_identity(state),
    )


def _layout_from_state(state: DeviceMetricState, /) -> DeviceMetricLayout:
    if not isinstance(state, DeviceMetricState):
        raise TypeError("state must be DeviceMetricState.")
    layout = DeviceMetricLayout(
        vertex_capacity=state.mesh.vertex_capacity,
        cell_capacity=state.mesh.cell_capacity,
        ball_width=state.ball_width,
        periodic_orbit_width=state.periodic_vertex_members.shape[1],
    )
    if layout.signature_id != state.layout_signature_id:
        raise ValueError("The device metric state carries an invalid layout binding.")
    return layout


def _periodic_device_adaptation(
    prepared: PreparedDeviceMetricAdaptation,
    layout: DeviceMetricLayout,
    state: DeviceMetricState,
    /,
) -> DeviceMetricUpdate:
    if not state.periodic_enabled:
        raise ValueError("Periodic device execution requires periodic device state.")
    source_mesh = prepared.adaptation.source.mesh
    periodic_topology = source_mesh.periodic_topology
    if periodic_topology is None:
        raise ValueError("Periodic device state is bound to a nonperiodic source.")
    expected_identity = (
        source_mesh.mesh_id,
        source_mesh.topology_id,
        source_mesh.lifted_topology_id,
        periodic_topology.periodic_topology_id,
    )
    if _periodic_source_identity(state) != expected_identity:
        raise ValueError("Prepared source and periodic device state identities differ.")
    if int(np.asarray(jax.device_get(state.status_flags))) & _FAILURES:
        return _compiled_adapt(layout, state)

    initial_layout = layout
    initial_state = state
    initial_work = _work(state)
    current_layout = layout
    current_state = state
    current_prepared = PreparedDeviceMetricAdaptation(
        prepared.adaptation,
        layout,
        state,
        prepared.anchor,
        periodic_mesh_id=state.periodic_mesh_id,
    )
    mode = _host_mode()
    capacity_status = int(AdaptiveSimplexStatus.CAPACITY_EXCEEDED)

    def rollback(status: int, /) -> DeviceMetricUpdate:
        flags = initial_work.flags.at[_STATUS].set(
            initial_work.flags[_STATUS] | jnp.asarray(status, dtype=jnp.int32)
        )
        failed = initial_work._replace(
            flags=flags.at[_CONVERGED].set(jnp.int32(0)).at[_STALLED].set(jnp.int32(0))
        )
        report = _report(
            initial_work,
            failed,
            jnp.asarray(status, dtype=jnp.int32),
            jnp.bool_(False),
            _geometry(failed),
            initial_layout,
        )
        result = _state(
            failed,
            initial_layout,
            True,
            initial_state.periodic_mesh_id,
            _periodic_source_identity(initial_state),
        )
        return DeviceMetricUpdate(result, report, initial_layout)

    def ensure_ball_workspace() -> None:
        nonlocal current_layout, current_state, current_prepared
        work = _work(current_state)
        active = np.asarray(jax.device_get(work.cell_active), dtype=np.bool_)
        cells = np.asarray(jax.device_get(work.cells), dtype=np.int64)[active]
        valence = np.bincount(
            cells.reshape((-1,)), minlength=current_layout.vertex_capacity
        )
        required = max(1, int(np.max(valence, initial=0)))
        if required <= current_layout.ball_width:
            return
        current_layout = DeviceMetricLayout(
            vertex_capacity=current_layout.vertex_capacity,
            cell_capacity=current_layout.cell_capacity,
            ball_width=required,
            periodic_orbit_width=current_layout.periodic_orbit_width,
        )
        current_state = _resize_device_state(
            current_state,
            current_layout,
            clear_capacity=False,
            clear_cavity=True,
        )
        current_prepared = PreparedDeviceMetricAdaptation(
            current_prepared.adaptation,
            current_layout,
            current_state,
            current_prepared.anchor,
        )

    def run_phase(phase: int, /) -> tuple[int, int]:
        nonlocal current_layout, current_state, current_prepared
        phase_applied = 0
        while True:
            update = _compiled_periodic_round(
                current_layout,
                _work(current_state),
                jnp.asarray(phase, dtype=jnp.int32),
            )
            report = update.report
            status, rejected_cavity, applied = jax.device_get(
                (
                    report.status,
                    report.rejected_cavity,
                    report.splits + report.collapses + report.flips + report.relocations,
                )
            )
            status = int(np.asarray(status))
            rejected_cavity = int(np.asarray(rejected_cavity))
            applied = int(np.asarray(applied))
            grow_slots = bool(status & capacity_status)
            other_failure = status & _FAILURES & ~capacity_status
            if other_failure:
                return phase_applied, other_failure
            if grow_slots:
                failed_epoch = _state(
                    update.work,
                    current_layout,
                    True,
                    current_state.periodic_mesh_id,
                    _periodic_source_identity(current_state),
                )
                try:
                    grown = _grow_metric_layout(
                        current_layout,
                        failed_epoch,
                        grow_slots=True,
                        grow_ball=rejected_cavity > 0,
                    )
                except ValueError:
                    return phase_applied, capacity_status
                if grown.signature_id == current_layout.signature_id:
                    return phase_applied, capacity_status
                current_layout = grown
                current_state = _resize_device_state(
                    failed_epoch,
                    grown,
                    clear_capacity=True,
                    clear_cavity=rejected_cavity > 0,
                )
                current_prepared = PreparedDeviceMetricAdaptation(
                    current_prepared.adaptation,
                    grown,
                    current_state,
                    current_prepared.anchor,
                )
                continue

            candidate = _state(
                update.work,
                current_layout,
                True,
                current_state.periodic_mesh_id,
                _periodic_source_identity(current_state),
            )
            if phase == 3 and rejected_cavity:
                return phase_applied, int(AdaptiveSimplexStatus.NEEDS_HOST_RESOLUTION)
            if applied:
                current_state = candidate
                current_layout, current_state = _refresh_periodic_device_state(
                    current_prepared, current_layout, current_state, mode=mode
                )
                current_prepared = PreparedDeviceMetricAdaptation(
                    current_prepared.adaptation,
                    current_layout,
                    current_state,
                    current_prepared.anchor,
                )
                phase_applied += applied
            else:
                current_state = candidate

            if rejected_cavity:
                try:
                    grown = _grow_metric_layout(
                        current_layout,
                        current_state,
                        grow_slots=False,
                        grow_ball=True,
                    )
                except ValueError:
                    return phase_applied, capacity_status
                if grown.signature_id == current_layout.signature_id:
                    return phase_applied, capacity_status
                current_layout = grown
                current_state = _resize_device_state(
                    current_state, grown, clear_capacity=False, clear_cavity=True
                )
                current_prepared = PreparedDeviceMetricAdaptation(
                    current_prepared.adaptation,
                    grown,
                    current_state,
                    current_prepared.anchor,
                )
                continue
            if phase == 3 or not applied:
                return phase_applied, 0

    controls = np.asarray(jax.device_get(current_state.controls), dtype=np.int32)
    stalled = False
    # The last escaped stall: its layout, state, preparation, potential, and the
    # index of the escape attempt that ran from it.
    escaped_from: (
        tuple[
            DeviceMetricLayout,
            DeviceMetricState,
            PreparedDeviceMetricAdaptation,
            tuple[int, ...],
            int,
        ]
        | None
    ) = None
    # Stall-escape attempts in order (``_periodic_round`` operations): optimal-
    # point collapses, then endpoint collapses up to ``_COLLAPSE_LIMIT``.
    escape_operations = (5, 4)
    while True:
        if bool(np.asarray(jax.device_get(_unit(_work(current_state))))):
            break
        work = _work(current_state)
        work = work._replace(counters=work.counters.at[_PASSES].add(jnp.int64(1)))
        current_state = _state(
            work,
            current_layout,
            True,
            current_state.periodic_mesh_id,
            _periodic_source_identity(current_state),
        )
        pass_applied = 0
        phases = (tuple(range(3)) if controls[_TOPOLOGY] else ()) + (
            (3,) if controls[_RELOCATION] else ()
        )
        for phase in phases:
            if phase == 3:
                ensure_ball_workspace()
            applied, failure = run_phase(phase)
            if failure:
                return rollback(failure)
            pass_applied += applied
        if bool(np.asarray(jax.device_get(_unit(_work(current_state))))):
            break
        if pass_applied != 0:
            continue
        stalled = True
        if not controls[_TOPOLOGY]:
            break
        # Stall escape and its well-founded potential, as in ``_adapted``.
        potential = tuple(
            int(value)
            for value in np.asarray(
                jax.device_get(_stall_potential(_work(current_state)))
            ).tolist()
        )
        first = 0
        if escaped_from is not None and potential >= escaped_from[3]:
            # Return to the escaped stall and try its next attempt.
            current_layout, current_state, current_prepared, potential, attempt = (
                escaped_from
            )
            first = attempt + 1
        stall = (current_layout, current_state, current_prepared, potential)
        applied = 0
        attempt = first
        for attempt in range(first, len(escape_operations)):
            applied, failure = run_phase(escape_operations[attempt])
            if failure:
                return rollback(failure)
            if applied:
                break
        if applied == 0:
            break
        stalled = False
        escaped_from = (*stall, attempt)
    converged = bool(np.asarray(jax.device_get(_unit(_work(current_state)))))
    stalled = stalled and not converged
    result_work = _work(current_state)
    flags = (
        result_work.flags.at[_STATUS]
        .set(initial_work.flags[_STATUS])
        .at[_CONVERGED]
        .set(jnp.asarray(converged, dtype=jnp.int32))
        .at[_STALLED]
        .set(jnp.asarray(stalled, dtype=jnp.int32))
    )
    result_work = result_work._replace(flags=flags)
    result = _state(
        result_work,
        current_layout,
        True,
        current_state.periodic_mesh_id,
        _periodic_source_identity(current_state),
    )
    report = _report(
        initial_work,
        result_work,
        jnp.int32(0),
        jnp.asarray(stalled),
        _geometry(result_work),
        current_layout,
    )
    return DeviceMetricUpdate(result, report, current_layout)


def adapt_device_metric(
    layout: DeviceMetricLayout,
    state: DeviceMetricState,
    /,
    *,
    prepared: PreparedDeviceMetricAdaptation | None = None,
) -> DeviceMetricUpdate:
    """Run metric epochs to convergence or a true stall.

    Periodic states require their prepared source context so every compiled
    operation batch can be rebound through the exact source-bound quotient.
    Device numeric edits remain compiled; host barriers rebuild topology maps
    only and never execute a numerical fallback.
    """
    _check_state(layout, state)
    if state.periodic_enabled:
        if not isinstance(prepared, PreparedDeviceMetricAdaptation):
            raise ValueError(
                "Periodic device adaptation requires its prepared source context."
            )
        return _periodic_device_adaptation(prepared, layout, state)
    if prepared is not None and not isinstance(prepared, PreparedDeviceMetricAdaptation):
        raise TypeError("prepared must be PreparedDeviceMetricAdaptation or None.")
    initial_status = int(np.asarray(jax.device_get(state.status_flags)))
    if initial_status & _FAILURES:
        return _compiled_adapt(layout, state)
    current_layout, current_state = layout, state
    capacity_status = int(AdaptiveSimplexStatus.CAPACITY_EXCEEDED)
    while True:
        update = _compiled_adapt(current_layout, current_state)
        status, rejected_cavity = jax.device_get(
            (update.report.status, update.report.rejected_cavity)
        )
        status = int(np.asarray(status))
        rejected_cavity = int(np.asarray(rejected_cavity))
        grow_slots = bool(status & capacity_status)
        grow_ball = rejected_cavity > 0
        other_failure = status & _FAILURES & ~capacity_status
        if other_failure or not (grow_slots or grow_ball):
            return update
        try:
            grown = _grow_metric_layout(
                current_layout,
                update.state,
                grow_slots=grow_slots,
                grow_ball=grow_ball,
            )
        except ValueError:
            return update
        if grown.signature_id == current_layout.signature_id:
            return update
        current_layout = grown
        current_state = _resize_device_state(
            update.state,
            grown,
            clear_capacity=grow_slots,
            clear_cavity=grow_ball,
        )


# ------------------------------------------------------------------ preparation


def _host_mode() -> PredicateMode:
    """Host predicates of preparation (source orientation) and commit (location)."""
    return resolve_host_predicate_mode(PredicateMode.EXACT)


def _edge_features(topology: Any, host_cells: np.ndarray, /) -> Any:
    """Feature class and protection of the edge opposite every local vertex."""
    first = host_cells[:, _NEXT].reshape((-1,))
    second = host_cells[:, _PREVIOUS].reshape((-1,))
    rows = _lookup(topology.codes, _codes(np.sort(np.stack((first, second), 1), 1)))
    shape = host_cells.shape
    return (
        topology.edge_class[rows].reshape(shape),
        topology.edge_protected[rows].reshape(shape),
    )


class _PeriodicDeviceTables(NamedTuple):
    vertex_orbits: np.ndarray
    vertex_representative: np.ndarray
    vertex_members: np.ndarray
    edge_orbits: np.ndarray
    edge_representative: np.ndarray
    edge_start: np.ndarray
    edge_end: np.ndarray
    edge_members: np.ndarray
    cell_orbits: np.ndarray
    orbit_width: int
    target_mesh_id: str
    next_quotient_vertex: int
    vertex_translation: np.ndarray
    vertex_deck: np.ndarray
    edge_translation: np.ndarray
    edge_deck: np.ndarray
    vertex_key: np.ndarray
    vertex_winding: np.ndarray
    edge_winding: np.ndarray


def _deck_offset(
    isometry: tuple[tuple[Fraction, ...], ...], /
) -> tuple[Fraction, ...] | None:
    """Translation of an exact homogeneous deck element, or None if it is not one."""
    dimension = len(isometry) - 1
    linear = all(
        isometry[row][column] == (1 if row == column else 0)
        for row in range(dimension)
        for column in range(dimension)
    )
    return tuple(isometry[row][dimension] for row in range(dimension)) if linear else None


def _deck_translation(
    member: tuple[Fraction, ...] | None,
    anchor: tuple[Fraction, ...] | None,
    /,
) -> tuple[Fraction, ...] | None:
    """Exact translation from ``anchor`` to ``member`` when binary64 represents it."""
    if member is None or anchor is None:
        return None
    difference = tuple(
        first - second for first, second in zip(member, anchor, strict=True)
    )
    if any(Fraction(float(value)) != value for value in difference):
        return None
    return difference


def _periodic_device_tables(
    mesh: Any,
    source: _Source,
    host: _State,
    topology: Any,
    device: DeviceMetricState,
    mode: PredicateMode,
    /,
) -> _PeriodicDeviceTables:
    """Bind exact current quotient orbits to device slots between compiled batches."""
    context = _periodic_state_context(mesh, source, host, topology, mode)
    device_mesh = device.mesh
    vertex_capacity = device_mesh.vertex_capacity
    cell_capacity = device_mesh.cell_capacity
    half_capacity = 3 * cell_capacity
    vertex_ids = np.asarray(device_mesh.vertex_ids, dtype=np.int64)
    vertex_active = np.asarray(device_mesh.vertex_active, dtype=np.bool_)
    cell_active = np.asarray(device_mesh.cell_active, dtype=np.bool_)
    cells = np.asarray(device_mesh.cells, dtype=np.int64)
    neighbors = np.asarray(device_mesh.facet_neighbors, dtype=np.int64)
    active_vertices = np.flatnonzero(vertex_active)
    active_cells = np.flatnonzero(cell_active)
    host_vertex_ids = np.asarray(host.vertex_ids, dtype=np.int64)
    host_vertex_rows = {
        int(identifier): row for row, identifier in enumerate(host_vertex_ids.tolist())
    }
    if context is None:
        host_vertex_orbits = host_vertex_ids.copy()
        edge_keys = np.sort(
            host_vertex_ids[np.asarray(topology.edges, dtype=np.int64)], axis=1
        )
        host_edge_orbits = np.arange(edge_keys.shape[0], dtype=np.int64)
        edge_start_ids, edge_end_ids = edge_keys[:, 0], edge_keys[:, 1]
        host_cell_orbits = np.arange(host.cells.shape[0], dtype=np.int64)
        rank = int(device.periodic_vertex_winding.shape[1])
        host_vertex_windings = np.zeros((host_vertex_ids.size, rank), dtype=np.int64)
        host_edge_windings = np.zeros((edge_keys.shape[0], rank), dtype=np.int64)
    else:
        vertex_orbits = np.asarray(context.vertex_orbits, dtype=np.int64)
        host_vertex_orbits = np.full(host.vertex_ids.shape, -1, dtype=np.int64)
        live = vertex_orbits >= 0
        host_vertex_orbits[live] = np.asarray(
            context.vertex_quotient_ids, dtype=np.int64
        )[vertex_orbits[live]]
        edge_keys = np.sort(
            host_vertex_ids[np.asarray(topology.edges, dtype=np.int64)], axis=1
        )
        edge_rows = np.asarray(context.edge_orbits, dtype=np.int64)
        host_edge_orbits = np.asarray(context.edge_quotient_ids, dtype=np.int64)[
            edge_rows
        ]
        canonical_rows = np.asarray(context.edge_canonical_vertices, dtype=np.int64)
        edge_start_ids = host_vertex_ids[canonical_rows[:, 0]]
        edge_end_ids = host_vertex_ids[canonical_rows[:, 1]]
        host_cell_orbits = np.asarray(context.cell_quotient_ids, dtype=np.int64)[
            np.asarray(context.cell_orbits, dtype=np.int64)
        ]
        orders = identification_orders(mesh.periodic_topology.cell)
        host_vertex_windings = reduce_periodic_exponents(
            np.asarray(context.vertex_windings, dtype=np.int64), orders
        )
        host_edge_windings = reduce_periodic_exponents(
            np.asarray(context.edge_member_windings, dtype=np.int64), orders
        )
    edge_data: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for row, (key, orbit, start, end) in enumerate(
        zip(
            edge_keys.tolist(),
            host_edge_orbits.tolist(),
            edge_start_ids.tolist(),
            edge_end_ids.tolist(),
            strict=True,
        )
    ):
        normalized = (int(key[0]), int(key[1]))
        edge_data[normalized] = (int(orbit), int(start), int(end), row)
    device_slot_by_id = {
        int(identifier): slot for slot, identifier in enumerate(vertex_ids.tolist())
    }
    vertex_orbits = np.full((vertex_capacity,), -1, dtype=np.int64)
    # One producer-owned key per current orbit (kind 0: an existing orbit).
    vertex_key = np.zeros((vertex_capacity, 2), dtype=np.int64)
    vertex_winding = np.zeros(
        (vertex_capacity, host_vertex_windings.shape[1]), dtype=np.int64
    )
    for slot in active_vertices.tolist():
        row = host_vertex_rows.get(int(vertex_ids[slot]))
        if row is None or host_vertex_orbits[row] < 0:
            raise ValueError("A live device vertex lost its exact periodic source orbit.")
        vertex_orbits[slot] = host_vertex_orbits[row]
        vertex_key[slot] = (0, host_vertex_orbits[row])
        vertex_winding[slot] = host_vertex_windings[row]
    # A retired slot keeps a key of its own so it never reads as an orbit member.
    retired = np.flatnonzero((vertex_ids >= 0) & ~vertex_active)
    vertex_key[retired, 0] = 2
    vertex_key[retired, 1] = vertex_ids[retired]
    vertex_groups: dict[int, list[int]] = defaultdict(list)
    for slot in active_vertices.tolist():
        vertex_groups[int(vertex_orbits[slot])].append(slot)
    vertex_representative = np.full((vertex_capacity,), -1, dtype=np.int32)
    vertex_members = np.full(
        (vertex_capacity, max(1, max(map(len, vertex_groups.values()), default=0))),
        -1,
        dtype=np.int32,
    )
    for members in vertex_groups.values():
        members.sort(key=lambda slot: (int(vertex_ids[slot]), slot))
        representative = members[0]
        vertex_representative[members] = representative
        vertex_members[representative, : len(members)] = members

    edge_orbits = np.full((half_capacity,), -1, dtype=np.int64)
    edge_representative = np.full((half_capacity,), -1, dtype=np.int32)
    edge_start = np.full((half_capacity,), -1, dtype=np.int32)
    edge_end = np.full((half_capacity,), -1, dtype=np.int32)
    edge_winding = np.zeros((half_capacity, host_edge_windings.shape[1]), dtype=np.int64)
    canonical = np.zeros((half_capacity,), dtype=np.bool_)
    for cell in active_cells.tolist():
        for local in range(3):
            half = 3 * cell + local
            other = int(neighbors[cell, local])
            canonical[half] = other < 0 or half < other
            first = int(cells[cell, _NEXT[local]])
            second = int(cells[cell, _PREVIOUS[local]])
            key = tuple(sorted((int(vertex_ids[first]), int(vertex_ids[second]))))
            data = edge_data.get(key)
            if data is None:
                raise ValueError(
                    "A live device edge lost its exact periodic source orbit."
                )
            orbit, start_id, end_id, edge_row = data
            edge_winding[half] = host_edge_windings[edge_row]
            start_slot = device_slot_by_id.get(start_id)
            end_slot = device_slot_by_id.get(end_id)
            if start_slot is None or end_slot is None:
                raise ValueError(
                    "A periodic edge representative names an absent device vertex."
                )
            edge_orbits[half] = orbit
            edge_start[half] = start_slot
            edge_end[half] = end_slot
    edge_groups: dict[int, list[int]] = defaultdict(list)
    for half in np.flatnonzero(canonical).tolist():
        edge_groups[int(edge_orbits[half])].append(half)
    edge_members = np.full(
        (half_capacity, max(1, max(map(len, edge_groups.values()), default=0))),
        -1,
        dtype=np.int32,
    )
    for members in edge_groups.values():
        members.sort()
        representative = members[0]
        edge_representative[members] = representative
        edge_members[representative, : len(members)] = members
    for half in np.flatnonzero(edge_orbits >= 0).tolist():
        representative = edge_representative[half]
        if representative < 0:
            neighbor = int(neighbors[half // 3, half % 3])
            canonical_half = half if canonical[half] else neighbor
            representative = edge_representative[canonical_half]
            edge_representative[half] = representative

    # Deck translations relative to each orbit representative: device kernels
    # place every lifted member at ``representative + translation``, which
    # rounds to the exact deck image exactly when the translation is a float.
    vertex_translation = np.zeros((vertex_capacity, 2), dtype=np.float64)
    vertex_deck = np.zeros((vertex_capacity,), dtype=np.bool_)
    edge_translation = np.zeros((half_capacity, 2), dtype=np.float64)
    edge_deck = np.zeros((half_capacity,), dtype=np.bool_)
    if context is None:
        vertex_deck[active_vertices] = True
        edge_deck[edge_orbits >= 0] = True
    else:
        offsets = {
            slot: _deck_offset(
                context.vertex_exact_isometries[host_vertex_rows[int(vertex_ids[slot])]]
            )
            for slot in active_vertices.tolist()
        }
        for slot in active_vertices.tolist():
            translation = _deck_translation(
                offsets[slot], offsets[int(vertex_representative[slot])]
            )
            if translation is not None:
                vertex_translation[slot] = np.asarray(translation, dtype=np.float64)
                vertex_deck[slot] = True
        for half in np.flatnonzero(edge_orbits >= 0).tolist():
            representative = int(edge_representative[half])
            start = _deck_translation(
                offsets[int(edge_start[half])],
                offsets[int(edge_start[representative])],
            )
            end = _deck_translation(
                offsets[int(edge_end[half])],
                offsets[int(edge_end[representative])],
            )
            if start is not None and start == end:
                edge_translation[half] = np.asarray(start, dtype=np.float64)
                edge_deck[half] = True

    cell_orbits = np.full((cell_capacity,), -1, dtype=np.int64)
    if active_cells.size != host_cell_orbits.size:
        raise ValueError(
            "Device cells do not match the source-bound quotient cell table."
        )
    cell_orbits[active_cells] = host_cell_orbits
    orbit_width = max(
        1,
        vertex_members.shape[1],
        edge_members.shape[1],
    )
    if context is None:
        target_mesh_id = ""
        next_quotient_vertex = -1
    else:
        target_mesh_id = context.target_mesh_id
        next_quotient_vertex = int(context.vertex_allocator_next_id)
    return _PeriodicDeviceTables(
        vertex_orbits,
        vertex_representative,
        vertex_members,
        edge_orbits,
        edge_representative,
        edge_start,
        edge_end,
        edge_members,
        cell_orbits,
        orbit_width,
        target_mesh_id,
        next_quotient_vertex,
        vertex_translation,
        vertex_deck,
        edge_translation,
        edge_deck,
        vertex_key,
        vertex_winding,
        edge_winding,
    )


def _install_periodic_tables(
    layout: DeviceMetricLayout,
    state: DeviceMetricState,
    tables: _PeriodicDeviceTables,
    /,
) -> tuple[DeviceMetricLayout, DeviceMetricState]:
    if tables.orbit_width > layout.periodic_orbit_width:
        layout = DeviceMetricLayout(
            vertex_capacity=layout.vertex_capacity,
            cell_capacity=layout.cell_capacity,
            ball_width=layout.ball_width,
            periodic_orbit_width=tables.orbit_width,
        )
        state = _resize_device_state(state, layout, clear_capacity=False)
    width = layout.periodic_orbit_width

    def members(values: np.ndarray, rows: int, /) -> Array:
        source = np.asarray(values, dtype=np.int32)
        result = np.full((rows, width), -1, dtype=np.int32)
        result[: source.shape[0], : source.shape[1]] = source
        return jnp.asarray(result)

    work = _work(state)._replace(
        periodic_vertex_orbits=jnp.asarray(tables.vertex_orbits, dtype=jnp.int64),
        periodic_vertex_representative=jnp.asarray(
            tables.vertex_representative, dtype=jnp.int32
        ),
        periodic_vertex_members=members(tables.vertex_members, layout.vertex_capacity),
        periodic_edge_orbits=jnp.asarray(tables.edge_orbits, dtype=jnp.int64),
        periodic_edge_representative=jnp.asarray(
            tables.edge_representative, dtype=jnp.int32
        ),
        periodic_edge_start=jnp.asarray(tables.edge_start, dtype=jnp.int32),
        periodic_edge_end=jnp.asarray(tables.edge_end, dtype=jnp.int32),
        periodic_edge_members=members(tables.edge_members, 3 * layout.cell_capacity),
        periodic_cell_orbits=jnp.asarray(tables.cell_orbits, dtype=jnp.int64),
        periodic_vertex_translation=jnp.asarray(
            tables.vertex_translation, dtype=state.mesh.coordinates.dtype
        ),
        periodic_vertex_deck=jnp.asarray(tables.vertex_deck, dtype=jnp.bool_),
        periodic_edge_translation=jnp.asarray(
            tables.edge_translation, dtype=state.mesh.coordinates.dtype
        ),
        periodic_edge_deck=jnp.asarray(tables.edge_deck, dtype=jnp.bool_),
        periodic_vertex_key=jnp.asarray(tables.vertex_key, dtype=jnp.int64),
        periodic_vertex_winding=jnp.asarray(tables.vertex_winding, dtype=jnp.int64),
        periodic_edge_winding=jnp.asarray(tables.edge_winding, dtype=jnp.int64),
    )
    # The refreshed state is host-fetched; its cursors may be a NumPy array.
    cursors = jnp.asarray(work.cursors)
    if tables.next_quotient_vertex >= 0:
        cursors = cursors.at[_NEXT_QUOTIENT_VERTEX].set(tables.next_quotient_vertex)
    work = work._replace(cursors=cursors)
    return layout, _state(
        work,
        layout,
        state.periodic_enabled,
        tables.target_mesh_id,
        _periodic_source_identity(state),
    )


def _refresh_periodic_device_state(
    prepared: PreparedDeviceMetricAdaptation,
    layout: DeviceMetricLayout,
    state: DeviceMetricState,
    /,
    *,
    mode: PredicateMode,
) -> tuple[DeviceMetricLayout, DeviceMetricState]:
    if prepared.periodic_mesh_id != state.periodic_mesh_id:
        raise ValueError("Device periodic state is bound to a different target mesh.")
    host = jax.device_get(state)
    adapted = _host_state(prepared, host)
    topology = _host_topology(adapted, mode)
    tables = _periodic_device_tables(
        prepared.adaptation.source.mesh,
        prepared.anchor.source,
        adapted,
        topology,
        host,
        mode,
    )
    return _install_periodic_tables(layout, host, tables)


def _device_state(
    layout: DeviceMetricLayout,
    state: _State,
    topology: Any,
    controls: np.ndarray,
    /,
    *,
    periodic_enabled: bool,
    next_quotient_vertex: int,
    periodic_mesh_id: str,
    periodic_source_identity: tuple[str, str, str, str],
    periodic_rank: int,
) -> DeviceMetricState:
    """Pad source state and initialize exact periodic-workspace rows."""
    vertex_capacity = layout.vertex_capacity
    cell_capacity = layout.cell_capacity
    vertex_order = np.argsort(state.vertex_ids, kind="stable")
    cell_order = np.argsort(state.cell_ids, kind="stable")
    slot_of = np.empty(vertex_order.shape, dtype=np.int64)
    slot_of[vertex_order] = np.arange(vertex_order.size, dtype=np.int64)
    host_cells = state.cells[cell_order]
    vertex_count, cell_count = vertex_order.size, cell_order.size
    record_count = state.lineage_keys.shape[0]
    if record_count > layout.record_capacity or cell_count > layout.ancestry_capacity:
        raise ValueError("The prepared mesh exceeds its lineage capacities.")
    classes, protected = _edge_features(topology, host_cells)

    def padded(values: Any, capacity: Any, fill: Any, dtype: Any) -> Any:
        array = np.asarray(values, dtype=dtype)
        result = np.full((capacity, *array.shape[1:]), fill, dtype=dtype)
        result[: array.shape[0]] = array
        return jnp.asarray(result)

    metric = np.broadcast_to(np.eye(2), (vertex_capacity, 2, 2)).copy()
    metric[:vertex_count] = state.metric[vertex_order]
    ancestry = np.zeros((layout.ancestry_capacity, 2), dtype=np.int32)
    ancestry[:, 0] = cell_capacity
    ancestry[:cell_count, 0] = np.arange(cell_count)
    ancestry[:cell_count, 1] = cell_order
    records = layout.record_capacity
    orbit_width = layout.periodic_orbit_width
    work = _Work(
        coordinates=padded(state.points[vertex_order], vertex_capacity, 0.0, np.float64),
        vertex_ids=padded(state.vertex_ids[vertex_order], vertex_capacity, -1, np.int64),
        vertex_active=padded(np.ones(vertex_count), vertex_capacity, False, np.bool_),
        cells=padded(slot_of[host_cells], cell_capacity, 0, np.int32),
        cell_ids=padded(state.cell_ids[cell_order], cell_capacity, -1, np.int64),
        cell_active=padded(np.ones(cell_count), cell_capacity, False, np.bool_),
        periodic_enabled=bool(periodic_enabled),
        metric=jnp.asarray(metric),
        source_rows=padded(vertex_order, vertex_capacity, -1, np.int32),
        vertex_fixed=padded(state.fixed[vertex_order], vertex_capacity, False, np.bool_),
        vertex_moved=jnp.zeros((vertex_capacity,), dtype=jnp.bool_),
        collapse_to=jnp.full((vertex_capacity,), -1, dtype=jnp.int32),
        cell_regions=padded(state.cell_region[cell_order], cell_capacity, 0, np.int32),
        cell_blocks=padded(state.cell_block[cell_order], cell_capacity, 0, np.int32),
        cell_ranks=jnp.zeros((cell_capacity,), dtype=jnp.int32),
        edge_classes=padded(classes, cell_capacity, 0, np.int32),
        edge_protected=padded(protected, cell_capacity, False, np.bool_),
        record_keys=padded(
            np.sort(slot_of[state.lineage_keys], axis=1), records, 0, np.int32
        ),
        record_sources=padded(state.lineage_sources, records, -1, np.int32),
        record_ranks=jnp.zeros((records,), dtype=jnp.int32),
        ancestry=jnp.asarray(ancestry),
        periodic_vertex_orbits=jnp.full((vertex_capacity,), -1, dtype=jnp.int64),
        periodic_vertex_representative=jnp.full((vertex_capacity,), -1, dtype=jnp.int32),
        periodic_vertex_members=jnp.full(
            (vertex_capacity, orbit_width), -1, dtype=jnp.int32
        ),
        periodic_edge_orbits=jnp.full((3 * cell_capacity,), -1, dtype=jnp.int64),
        periodic_edge_representative=jnp.full((3 * cell_capacity,), -1, dtype=jnp.int32),
        periodic_edge_start=jnp.full((3 * cell_capacity,), -1, dtype=jnp.int32),
        periodic_edge_end=jnp.full((3 * cell_capacity,), -1, dtype=jnp.int32),
        periodic_edge_members=jnp.full(
            (3 * cell_capacity, orbit_width), -1, dtype=jnp.int32
        ),
        periodic_cell_orbits=jnp.full((cell_capacity,), -1, dtype=jnp.int64),
        periodic_vertex_translation=jnp.zeros((vertex_capacity, 2), dtype=jnp.float64),
        periodic_vertex_deck=jnp.zeros((vertex_capacity,), dtype=jnp.bool_),
        periodic_edge_translation=jnp.zeros((3 * cell_capacity, 2), dtype=jnp.float64),
        periodic_edge_deck=jnp.zeros((3 * cell_capacity,), dtype=jnp.bool_),
        periodic_vertex_key=(
            jnp.zeros((vertex_capacity, 2), dtype=jnp.int64)
            if state.orbit_keys is None
            else padded(state.orbit_keys[vertex_order], vertex_capacity, 0, np.int64)
        ),
        periodic_vertex_winding=(
            jnp.zeros((vertex_capacity, periodic_rank), dtype=jnp.int64)
            if state.orbit_windings is None
            else padded(state.orbit_windings[vertex_order], vertex_capacity, 0, np.int64)
        ),
        periodic_edge_winding=jnp.zeros(
            (3 * cell_capacity, periodic_rank), dtype=jnp.int64
        ),
        cursors=jnp.asarray(
            np.asarray(
                (
                    vertex_count,
                    cell_count,
                    record_count,
                    cell_count,
                    state.next_vertex_id,
                    int(np.max(state.cell_ids)) + 1,
                    next_quotient_vertex,
                ),
                dtype=np.int64,
            )
        ),
        controls=jnp.asarray(controls),
        counters=jnp.zeros((8,), dtype=jnp.int64),
        flags=jnp.zeros((3,), dtype=jnp.int32),
    )
    return _state(
        work,
        layout,
        periodic_enabled,
        periodic_mesh_id,
        periodic_source_identity,
    )


def _periodic_orbit_width(mesh: Any, /) -> int:
    if mesh.periodic_topology is None:
        return 1
    from ._periodic import periodic_entity_orbit_map

    width = 1
    for degree in (0, 1):
        orbit_map = periodic_entity_orbit_map(mesh, degree)
        orbit_map.require_source(mesh)
        counts = np.bincount(
            np.asarray(orbit_map.lifted_to_quotient, dtype=np.int64),
            minlength=orbit_map.quotient_entity_ids.shape[0],
        )
        width = max(width, int(np.max(counts, initial=0)))
    return width


def _prepared_metric(adaptation: PreparedMeshAdaptation, /) -> Any:
    # Lazy: the adaptation transaction imports this module's evidence type.
    from ._adaptation import MetricMeshAdaptation

    policy = adaptation.policy
    constraints = adaptation.constraints
    mesh = adaptation.source.mesh
    mode = _host_mode()
    values = _validate_mesh(mesh, constraints.metric_values)
    vertex_count = mesh.coordinates.shape[0]
    edge_count = mesh.entity_set(1).entity_ids.shape[0]
    cell_count = mesh.entity_set(2).entity_ids.shape[0]
    state, source = _initial_state(
        mesh,
        values,
        _integer_array(constraints.cell_classes, cell_count, "cell_classes"),
        _integer_array(constraints.edge_classes, edge_count, "edge_classes"),
        _boolean_array(constraints.protected_edge_mask, edge_count, "protected_edges"),
        _boolean_array(constraints.fixed_vertex_mask, vertex_count, "fixed_vertices"),
        mode,
    )
    topology = _host_topology(state, mode)
    _validate_classification(
        state, topology, _periodic_state_context(mesh, source, state, topology, mode)
    )
    # ty: ignore[unresolved-attribute]
    vertex_capacity, cell_capacity = policy.device_policy.capacities(
        vertex_count, cell_count
    )
    ball_width = max(1, int(np.max(np.diff(topology.ball_offsets), initial=0)))
    periodic_enabled = mesh.periodic_topology is not None
    periodic_source_identity = (
        (
            mesh.mesh_id,
            mesh.topology_id,
            mesh.lifted_topology_id,
            mesh.periodic_topology.periodic_topology_id,
        )
        if periodic_enabled
        else ("", "", "", "")
    )
    orbit_width = _periodic_orbit_width(mesh)
    if periodic_enabled:
        next_quotient_vertex = int(mesh.periodic_topology.allocator_next_ids[0])
        if next_quotient_vertex < 0:
            quotient_vertex_ids = np.asarray(
                mesh.periodic_topology.quotient.entity_sets[0].entity_ids,
                dtype=np.int64,
            )
            next_quotient_vertex = int(np.max(quotient_vertex_ids)) + 1
    else:
        next_quotient_vertex = state.next_vertex_id
    layout = DeviceMetricLayout(
        vertex_capacity=vertex_capacity,
        cell_capacity=cell_capacity,
        ball_width=ball_width,
        periodic_orbit_width=orbit_width,
    )
    controls = np.asarray(
        (
            isinstance(adaptation.request, MetricMeshAdaptation),
            policy.relocation,
        ),
        dtype=np.int32,
    )
    device = _device_state(
        layout,
        state,
        topology,
        controls,
        periodic_enabled=periodic_enabled,
        next_quotient_vertex=next_quotient_vertex,
        periodic_mesh_id=mesh.mesh_id if periodic_enabled else "",
        periodic_source_identity=periodic_source_identity,
        periodic_rank=(
            len(identification_orders(mesh.periodic_topology.cell))
            if periodic_enabled
            else 0
        ),
    )
    anchor = _Anchor(source, int(np.max(state.cell_ids)) + 1)
    prepared = PreparedDeviceMetricAdaptation(adaptation, layout, device, anchor)
    layout, device = _refresh_periodic_device_state(prepared, layout, device, mode=mode)
    prepared = PreparedDeviceMetricAdaptation(adaptation, layout, device, anchor)
    return prepared


def prepare_device_metric_adaptation(
    source: CellMeshingResult,
    request: MetricMeshAdaptation | RelocationMeshAdaptation,
    /,
    *,
    policy: MeshAdaptationPolicy,
) -> PreparedDeviceMetricAdaptation:
    """Validate, classify, and pad one certified planar source into a device state.

    ``policy.route`` must be DEVICE_METRIC_2D; ``policy.device_policy`` supplies
    the initial compile bucket, which grows and retries on actual slot overflow.
    Protection, organization classes, and the SPD metric are resolved as for the
    host route; metric requests enable topology phases and ``policy.relocation``
    enables relocation.
    """

    from ._adaptation import (
        MeshAdaptationPolicy,
        MeshAdaptationRoute,
        prepare_mesh_adaptation,
    )

    if not isinstance(policy, MeshAdaptationPolicy):
        raise TypeError("policy must be MeshAdaptationPolicy.")
    if policy.route is not MeshAdaptationRoute.DEVICE_METRIC_2D:
        raise ValueError("Device metric adaptation requires the DEVICE_METRIC_2D route.")
    return _prepared_metric(prepare_mesh_adaptation(source, request, policy=policy))


# ------------------------------------------------------------------ commit


def _host_state(
    prepared: PreparedDeviceMetricAdaptation, host: DeviceMetricState, /
) -> Any:
    """The host working state of the device slots (vertex rows are vertex slots)."""
    mesh = host.mesh
    cursors = np.asarray(host.cursors)
    count = int(cursors[_VERTICES])
    active = np.flatnonzero(np.asarray(mesh.cell_active))
    row_of = np.full((prepared.layout.cell_capacity,), -1, dtype=np.int64)
    row_of[active] = np.arange(active.size, dtype=np.int64)
    cells = np.asarray(mesh.cells, dtype=np.int64)[active]
    classes = np.asarray(host.edge_classes, dtype=np.int64)[active].reshape((-1,))
    protected = np.asarray(host.edge_protected)[active].reshape((-1,))
    halves = np.sort(
        np.stack((cells[:, _NEXT], cells[:, _PREVIOUS]), axis=2).reshape((-1, 2)), axis=1
    )
    flagged = np.flatnonzero((classes > 0) | protected)
    features = flagged[_distinct(_codes(halves[flagged]))]
    records = int(cursors[_RECORDS])
    keys = np.sort(np.asarray(host.record_keys, dtype=np.int64)[:records], axis=1)
    sources = np.asarray(host.record_sources, dtype=np.int64)[:records]
    ranks = np.asarray(host.record_ranks, dtype=np.int64)[:records]
    # Identical records (same key, source edge, and rank) stem from one record
    # and share every later operation; the host deduplicates them eagerly.
    distinct = _distinct(_codes(keys), sources, -ranks)
    pairs = np.asarray(host.ancestry, dtype=np.int64)[: int(cursors[_PAIRS])]
    vertex_source_rows = np.asarray(host.source_rows, dtype=np.int64)[:count]
    vertex_moved = np.asarray(host.vertex_moved, dtype=np.bool_)[:count]
    source_stencil_sources = np.full((count, 3), -1, dtype=np.int64)
    source_stencil_weights = np.zeros((count, 3), dtype=np.float64)
    source_stencil_valid = np.zeros((count, 3), dtype=np.bool_)
    preserved = (vertex_source_rows >= 0) & ~vertex_moved
    source_stencil_sources[preserved, 0] = np.asarray(
        prepared.anchor.source.vertex_ids, dtype=np.int64
    )[vertex_source_rows[preserved]]
    source_stencil_weights[preserved, 0] = 1.0
    source_stencil_valid[preserved, 0] = True
    source_ids = np.asarray(prepared.anchor.source.vertex_ids, dtype=np.int64)
    source_stencil_exact: list[tuple[tuple[int, Fraction], ...]] = [()] * count
    for row in np.flatnonzero(preserved):
        source_stencil_exact[int(row)] = (
            (int(source_ids[vertex_source_rows[row]]), Fraction(1)),
        )
    cell_ids = np.asarray(mesh.cell_ids, dtype=np.int64)[active]
    return _State(
        points=np.asarray(mesh.coordinates, dtype=np.float64)[:count],
        metric=np.asarray(host.metric, dtype=np.float64)[:count],
        vertex_ids=np.asarray(mesh.vertex_ids, dtype=np.int64)[:count],
        source_stencil_sources=source_stencil_sources,
        source_stencil_weights=source_stencil_weights,
        source_stencil_valid=source_stencil_valid,
        source_stencil_exact=tuple(source_stencil_exact),
        source_rows=vertex_source_rows,
        fixed=np.asarray(host.vertex_fixed)[:count],
        moved=vertex_moved,
        alive=np.asarray(mesh.vertex_active)[:count],
        collapse_to=np.asarray(host.collapse_to, dtype=np.int64)[:count],
        cells=cells,
        cell_region=np.asarray(host.cell_regions, dtype=np.int64)[active],
        cell_block=np.asarray(host.cell_blocks, dtype=np.int64)[active],
        cell_ids=np.where(cell_ids < prepared.anchor.first_cell_id, cell_ids, -1),
        cell_rank=np.asarray(host.cell_ranks, dtype=np.int64)[active],
        ancestry_cells=row_of[pairs[:, 0]],
        ancestry_sources=pairs[:, 1],
        feature_keys=halves[features],
        feature_class=classes[features],
        feature_protected=protected[features],
        lineage_keys=keys[distinct],
        lineage_sources=sources[distinct],
        lineage_ranks=ranks[distinct],
        next_vertex_id=int(cursors[_NEXT_VERTEX]),
        orbit_keys=(
            np.asarray(host.periodic_vertex_key, dtype=np.int64)[:count]
            if host.periodic_enabled
            else None
        ),
        orbit_windings=(
            np.asarray(host.periodic_vertex_winding, dtype=np.int64)[:count]
            if host.periodic_enabled
            else None
        ),
    )


def _evidence(
    prepared: PreparedDeviceMetricAdaptation, host: DeviceMetricState, topology: Any, /
) -> DeviceMetricEvidence:
    counters = np.asarray(host.counters)
    flags = np.asarray(host.flags)
    controls = np.asarray(host.controls)
    lengths = _measured(topology)
    inside = (lengths >= _LOWER) & (lengths <= _UPPER)
    converged = bool(np.all(inside))
    return DeviceMetricEvidence(
        passes=int(counters[_PASSES]),
        splits=int(counters[_SPLITS]),
        collapses=int(counters[_COLLAPSES]),
        flips=int(counters[_FLIPS]),
        relocations=int(counters[_RELOCATIONS]),
        rejected_uncertain=int(counters[_REJECTED_UNCERTAIN]),
        rejected_invalid=int(counters[_REJECTED_INVALID]),
        rejected_cavity=int(counters[_REJECTED_CAVITY]),
        converged=converged,
        stalled=bool(flags[_STALLED]) and not converged,
        measured_edges=lengths.size,
        out_of_range_edges=int(np.count_nonzero(~inside)),
        minimum_metric_length=float(np.min(lengths)),
        maximum_metric_length=float(np.max(lengths)),
        unit_fraction=float(np.count_nonzero(inside) / lengths.size),
        minimum_metric_quality=float(np.min(topology.quality)),
        topology_operations=bool(controls[_TOPOLOGY]),
        relocation=bool(controls[_RELOCATION]),
        status=AdaptiveSimplexStatus(int(flags[_STATUS])),
        layout_id=prepared.layout.signature_id,
    )


def _outcome(
    prepared: PreparedDeviceMetricAdaptation, host: DeviceMetricState, /
) -> _RouteOutcome:
    # Lazy: the adaptation transaction imports this module's evidence type.
    from ._adaptation import (
        _finalize_native,
        _metric_status,
        _RouteOutcome,
        _target_metric,
        _unchanged,
    )

    adaptation = prepared.adaptation
    mesh = host.mesh
    require_committed_status(
        int(np.asarray(host.status_flags)),
        np.asarray(mesh.coordinates, dtype=np.float64),
        np.asarray(mesh.cells, dtype=np.int64)[np.asarray(mesh.cell_active)],
        DeviceEpoch.METRIC,
    )
    mode = _host_mode()
    state = _host_state(prepared, host)
    evidence = _evidence(prepared, host, _host_topology(state, mode))
    applied = evidence.splits + evidence.collapses + evidence.flips + evidence.relocations
    status = _metric_status(evidence, evidence.topology_operations)
    if applied == 0:
        # Preserve the exact source while exposing an unmet criterion as STALLED
        # rather than the converged UNCHANGED status.
        return _unchanged(adaptation, evidence, None, status=status)
    edit, metric = _assemble(state, prepared.anchor.source, mode)
    native = _finalize_native(
        adaptation, edit, MeshTransitionKind.REMESH, conservative=False
    )
    return _RouteOutcome(
        status,
        native.target,
        native.transition,
        native.lineage,
        native.stencil,
        native.transfer,
        # ty: ignore[unresolved-attribute]
        _target_metric(adaptation.request.metric, native.target.mesh, metric),
        evidence,
        None,
    )


def commit_device_metric_adaptation(
    prepared: PreparedDeviceMetricAdaptation, state: DeviceMetricState, /
) -> MeshAdaptationResult:
    """Commit one device epoch through the common canonical edit boundary."""

    from ._adaptation import _adaptation_result

    if not isinstance(prepared, PreparedDeviceMetricAdaptation):
        raise TypeError("prepared must be PreparedDeviceMetricAdaptation.")
    layout = _layout_from_state(state)
    _check_state(layout, state)
    committed = PreparedDeviceMetricAdaptation(
        prepared.adaptation, layout, state, prepared.anchor
    )
    started = time.monotonic()
    host = jax.device_get(state)
    return _adaptation_result(prepared.adaptation, _outcome(committed, host), started)


def _execute_device_metric_route(prepared: PreparedMeshAdaptation, /) -> _RouteOutcome:
    """Run compiled metric epochs, growing static workspaces before retry."""

    metric = _prepared_metric(prepared)
    update = adapt_device_metric(metric.layout, metric.state, prepared=metric)
    committed = PreparedDeviceMetricAdaptation(
        metric.adaptation,
        update.layout,
        update.state,
        metric.anchor,
        periodic_mesh_id=update.state.periodic_mesh_id,
    )
    return _outcome(committed, jax.device_get(update.state))


__all__ = [
    "DeviceMetricEvidence",
    "DeviceMetricLayout",
    "DeviceMetricReport",
    "DeviceMetricState",
    "DeviceMetricUpdate",
    "PreparedDeviceMetricAdaptation",
    "adapt_device_metric",
    "commit_device_metric_adaptation",
    "prepare_device_metric_adaptation",
]
