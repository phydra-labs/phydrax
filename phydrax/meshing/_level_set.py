#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Insert P1 zero interfaces into existing meshes with conservative W08 transitions.

Native roots split the declared piecewise-affine vertex field through the unrounded
subdivision of every crossing source edge at its exact source-owned P1 root
``S = (|v_b| A + |v_a| B) / (|v_a| + |v_b|)`` over the exact source positions
``A`` and ``B``. Its exact homogeneous weights ``(|v_b|, |v_a|)`` are the
scientific vertex identity; published coordinates are correctly rounded
carriers with certified outward deviations from ``S``. When the source carrier
is not itself the exact source (for example a non-dyadic root of an earlier
adaptation), native execution publishes topology only and the carriers are
rounded from the exact source positions, then certified with the native
exact predicates before any edit is formed. Admitted sources are affine tetrahedral
maps: binary64 P1 carriers, exact P1 coefficient actions, restricted parents and
composed rational charts (including earlier zero insertions and uniform
bisection), and declared exact rational or algebraic sources; PLC certificate
carriers are refused. Rational-chart children publish one tetrahedral block per distinct
coefficient action or nested reference action.
Material assignments inherit cell lineage independently
of level-set side identities. Moving interfaces bind the target numerical
revision even when coordinates, connectivity, and content identity are unchanged.
Tangential zero faces remain interface patches without distinct-zone adjacency
when both incident cells belong to the same side.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, ExitStack, nullcontext
from itertools import combinations
from operator import index
from typing import final, NamedTuple, TYPE_CHECKING

import equinox as eqx
import jax
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax.discretization._cell_mesh as cell_mesh_owner
import phydrax.discretization._reference_cell_types as reference_cell_types_owner
import phydrax.meshing._periodic as periodic_owner

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._meshcore import (
    exact_orient3d,
    level_set_split_3d,
    LevelSetCounters,
    LevelSetOutputMode,
    LevelSetSplitRequest,
    meshcore_exact_domain,
    MeshcoreError,
    MeshcoreStatus,
    NativeBatchTicket,
    NativeExecutionScope,
)
from .._resource_ticket import HostBatchTicket
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._cell_complex import PolyhedralConnectivity, TetrahedralConnectivity
from ..discretization._cell_geometry_transfer import (
    affine_geometry_is_carrier,
    affine_vertex_positions,
    exact_root_points,
    is_affine_cell_geometry,
    vertex_positions_are_carrier,
)
from ..discretization._cell_geometry_transition_types import NestedReferenceWitnesses
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..discretization._coordinate_enclosure import ExactScalar
from ..typing import checked
from ._entity_keys import entity_keys
from ._lineage_records import (
    EntityLineageKind,
    MeshTransitionKind,
    rounded_homogeneous_barycentrics,
)
from ._organization_records import MeshLabel, MeshPatch, MeshZone, MeshZoneRole
from ._scope import MeshingEntityKind, MeshingScope, resolve_mesh_scope
from ._topology_edit import (
    CellTopologyEdit,
    EntityRelations,
    exact_nested_reference_vertices,
    TopologyEditBlock,
)


if TYPE_CHECKING:
    from . import _adaptation as adaptation_owner
    from ._adaptation import PreparedMeshAdaptation


@contextmanager
def _level_set_coordinate_budget(
    prepared: PreparedMeshAdaptation,
    /,
    *,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
) -> Iterator[CoordinateEnclosureBudget]:
    """Reserve a bounded host phase before native topology execution."""
    limits = prepared.policy.limits
    remaining = scope.remaining()
    ticket = scope.batch(
        min(limits.maximum_work_units, remaining.remaining_work_units),
        queries_upper=min(
            limits.maximum_geometry_queries, remaining.remaining_geometry_queries
        ),
        bytes_upper=min(limits.maximum_scratch_bytes, remaining.remaining_managed_bytes),
    )
    with budget.bind_ticket(ticket):
        yield budget


@final
class LevelSetMeshAdaptation(StrictModule, NonTrainableState):
    """Insert the zero set of one mesh-bound scalar P1 field.

    ``values`` follow ``scope.entity_ids``, not incidental mesh row order. Samples
    equal to zero remain existing vertices. ``previous`` binds the last accepted
    interface to this source; changing its topology requires explicit consent.
    """

    scope: MeshingScope
    values: Array
    inside_region_id: str = eqx.field(static=True)
    outside_region_id: str = eqx.field(static=True)
    interface_id: str = eqx.field(static=True)
    previous: LevelSetEvidence | None
    accept_topology_change: bool = eqx.field(static=True)
    request_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        scope: MeshingScope,
        values: ArrayLike,
        /,
        *,
        inside_region_id: str = "level-set-inside",
        outside_region_id: str = "level-set-outside",
        interface_id: str = "level-set-interface",
        previous: LevelSetEvidence | None = None,
        accept_topology_change: bool = False,
    ) -> None:
        if scope.entity_kind is not MeshingEntityKind.MESH or scope.entity_dimension != 0:
            raise ValueError(
                "LevelSetMeshAdaptation.scope must be bound to mesh vertices; got a non-mesh or non-vertex scope."
            )
        scalar = np.asarray(values, dtype=np.float64)
        if scalar.shape != scope.entity_ids.shape or not np.all(np.isfinite(scalar)):
            raise ValueError(
                "LevelSetMeshAdaptation.values must be one finite scalar per scoped vertex; got nonfinite values or a different vertex count."
            )
        identities = (inside_region_id, outside_region_id, interface_id)
        if any(not isinstance(value, str) for value in identities):
            raise TypeError(
                "LevelSetMeshAdaptation.region_ids must be strings together with interface_id; got a non-string identity."
            )
        if (
            any(not value.strip() or value != value.strip() for value in identities)
            or len(set(identities)) != 3
        ):
            raise ValueError(
                f"LevelSetMeshAdaptation.region_ids must be distinct nonempty identifiers together with interface_id; got {identities!r}."
            )
        if not isinstance(accept_topology_change, bool):
            raise TypeError(
                f"LevelSetMeshAdaptation.accept_topology_change must be bool; got {type(accept_topology_change).__name__}."
            )
        if previous is not None and identities != (
            previous.inside_region_id,
            previous.outside_region_id,
            previous.interface_id,
        ):
            raise ValueError(
                "LevelSetMeshAdaptation.region_ids must preserve the previous region/interface identity together with interface_id; got different identities."
            )
        self.scope = scope
        self.values = jax.device_put(np.asarray(scalar, dtype=np.float64))
        self.inside_region_id, self.outside_region_id, self.interface_id = identities
        self.previous = previous
        self.accept_topology_change = accept_topology_change
        self.request_id = canonical_fingerprint(
            {
                "kind": "level-set-mesh-adaptation",
                "scope": scope.scope_id,
                "values": array_tree_fingerprint(scalar),
                "regions": identities,
                "previous": None if previous is None else previous.evidence_id,
                "accept_topology_change": accept_topology_change,
            }
        )


class _TetrahedralRows(NamedTuple):
    """Triangle and edge vertex rows aligned with the mesh face and edge entities."""

    faces: np.ndarray
    edges: np.ndarray


class _NativeTopology(NamedTuple):
    """Native exact subdivision; carriers only when the source is its carrier."""

    cells: np.ndarray
    parents: np.ndarray
    sources: np.ndarray
    weights: np.ndarray
    counters: LevelSetCounters
    coordinates: np.ndarray | None
    root_deviations: np.ndarray | None


class _Split(NamedTuple):
    """Certified exact subdivision: source vertex bytes, then root carriers."""

    coordinates: np.ndarray
    cells: np.ndarray
    parents: np.ndarray
    sources: np.ndarray
    weights: np.ndarray
    root_deviations: np.ndarray
    counters: LevelSetCounters


@final
class LevelSetEvidence(StrictModule, NonTrainableState):
    """Record P1 partitions, oriented zero facets, exact roots and topology events.

    ``topology_signature`` records strict inside/outside component counts, zero
    complex component count, Euler characteristic and maximum stratum dimension.
    Isolated vertices/edges equal to zero and tangential zero faces are retained.
    Zero face orientation is ±1 inside-to-outside, or 0 for a nonseparating face.
    Inserted root ``root_vertex_global_ids[r]`` is the exact P1 root of source
    edge ``root_source_global_ids[r]`` with exact homogeneous weights
    ``root_weights[r]`` (the swapped source magnitudes) over the exact positions
    of the source geometry ``source_geometry_id``; its published target carrier
    is the correctly rounded root and lies within the certified outward
    Euclidean bound ``root_deviations[r]``, which is zero exactly when the
    carrier is the root.
    """

    source_mesh_id: str = eqx.field(static=True)
    source_geometry_id: str = eqx.field(static=True)
    request_id: str = eqx.field(static=True)
    target_mesh_id: str = eqx.field(static=True)
    target_numeric_version: str = eqx.field(static=True)
    inside_region_id: str = eqx.field(static=True)
    outside_region_id: str = eqx.field(static=True)
    interface_id: str = eqx.field(static=True)
    cell_global_ids: Array
    cell_sides: Array
    interface_facet_global_ids: Array
    interface_orientations: Array
    zero_vertex_global_ids: Array
    root_vertex_global_ids: Array
    root_source_global_ids: Array
    root_weights: Array
    root_deviations: Array
    counters: Array
    topology_signature: tuple[int, int, int, int, int] = eqx.field(static=True)
    topology_changed: bool = eqx.field(static=True)
    topology_event_accepted: bool = eqx.field(static=True)
    inserted_roots: int = eqx.field(static=True)
    exact_zero_samples: int = eqx.field(static=True)
    work_units: int = eqx.field(static=True)
    maximum_carrier_deviation: float = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        prepared: PreparedMeshAdaptation,
        mesh: cell_mesh_owner.CellMesh,
        edit: CellTopologyEdit,
        split: _Split,
        /,
    ) -> None:
        request = _request(prepared)
        values, sides, facets = _partition(prepared, mesh, edit)
        signature = _topology_signature(mesh, values, sides, facets)
        previous = request.previous
        changed = previous is not None and signature != previous.topology_signature
        if changed and not request.accept_topology_change:
            raise ValueError(
                "LevelSetMeshAdaptation.accept_topology_change must be True when the moving zero set changes topology; got False."
            )
        counters = split.counters
        self.source_mesh_id = prepared.source.mesh.mesh_id
        self.source_geometry_id = cell_geometry_id(prepared.source.geometry)
        self.request_id = request.request_id
        self.target_mesh_id = mesh.mesh_id
        self.target_numeric_version = mesh.numeric_version
        self.inside_region_id = request.inside_region_id
        self.outside_region_id = request.outside_region_id
        self.interface_id = request.interface_id
        self.cell_global_ids = jax.device_put(
            np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64)
        )
        self.cell_sides = jax.device_put(np.asarray(sides, dtype=np.int8))
        self.interface_facet_global_ids = jax.device_put(
            np.asarray(np.asarray(mesh.entity_set(2).entity_ids)[facets], dtype=np.int64)
        )
        self.interface_orientations = jax.device_put(
            np.asarray(_interface_orientations(mesh, sides, facets), dtype=np.int8)
        )
        self.zero_vertex_global_ids = jax.device_put(
            np.asarray(np.asarray(mesh.vertex_global_ids)[values == 0.0], dtype=np.int64)
        )
        source_ids = np.asarray(prepared.source.mesh.vertex_global_ids)
        count = source_ids.size
        order = np.argsort(source_ids)
        root_sources = np.asarray(edit.stencil_sources[count:], dtype=np.int64)
        rows = order[np.searchsorted(source_ids[order], root_sources)]
        root_values = _source_values(prepared)[rows]
        weights = np.asarray(split.weights[count:], dtype=np.float64)
        deviations = np.asarray(split.root_deviations, dtype=np.float64)
        # The certified carriers are the published ones: a later geometry
        # rebinding may not substitute another carrier for a certified root.
        target_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
        target_order = np.argsort(target_ids)
        target_rows = target_order[
            np.minimum(
                np.searchsorted(target_ids[target_order], edit.vertex_global_ids),
                target_ids.size - 1,
            )
        ]
        if not np.array_equal(
            target_ids[target_rows], edit.vertex_global_ids
        ) or not np.array_equal(
            np.asarray(mesh.coordinates, dtype=np.float64)[target_rows].view(np.uint64),
            np.asarray(split.coordinates, dtype=np.float64).view(np.uint64),
        ):
            raise ValueError(
                "LevelSetEvidence.root_deviations must certify the published target carriers; got target coordinates that differ from the certified source and root carriers."
            )
        # The unrounded source-owned law: opposite nonzero endpoint values and the
        # swapped magnitudes as homogeneous weights, so the P1 value is zero.
        if (
            root_sources.shape != (counters.inserted, 2)
            or deviations.shape != (counters.inserted,)
            or not np.all(root_values != 0.0)
            or not np.all(np.signbit(root_values[:, 0]) != np.signbit(root_values[:, 1]))
            or not np.array_equal(weights, np.abs(root_values[:, ::-1]))
            or not np.all(np.isfinite(deviations))
            or np.any(deviations < 0.0)
        ):
            raise ValueError(
                "LevelSetEvidence.root_weights must be the exact source-owned P1 roots of crossing source edges together with certified carrier deviations; got a different root law."
            )
        self.root_vertex_global_ids = jax.device_put(
            np.asarray(edit.vertex_global_ids[count:], dtype=np.int64)
        )
        self.root_source_global_ids = jax.device_put(root_sources)
        self.root_weights = jax.device_put(weights)
        self.root_deviations = jax.device_put(deviations)
        counter_values = np.asarray(
            [value for _, value in counters.items()], dtype=np.int64
        )
        self.counters = jax.device_put(np.asarray(counter_values, dtype=np.int64))
        self.topology_signature = signature
        self.topology_changed = changed
        self.topology_event_accepted = changed and request.accept_topology_change
        self.inserted_roots = counters.inserted
        self.exact_zero_samples = counters.zero_vertices
        self.work_units = counters.work_units
        self.maximum_carrier_deviation = float(np.max(deviations, initial=0.0))
        self.evidence_id = canonical_fingerprint(self._scientific_payload())

    def _scientific_payload(self) -> dict[str, object]:
        return {
            "kind": "native-level-set-evidence",
            "request": self.request_id,
            "source": self.source_mesh_id,
            "source_geometry": self.source_geometry_id,
            "target": self.target_mesh_id,
            "target_revision": self.target_numeric_version,
            "cells": array_tree_fingerprint((self.cell_global_ids, self.cell_sides)),
            "facets": array_tree_fingerprint(
                (self.interface_facet_global_ids, self.interface_orientations)
            ),
            "zero_vertices": array_tree_fingerprint(self.zero_vertex_global_ids),
            "roots": array_tree_fingerprint(
                (
                    self.root_vertex_global_ids,
                    self.root_source_global_ids,
                    self.root_weights,
                    self.root_deviations,
                )
            ),
            "topology": self.topology_signature,
            "event_accepted": self.topology_event_accepted,
            "counters": array_tree_fingerprint(self.counters),
            "maximum_carrier_deviation": self.maximum_carrier_deviation,
        }

    def validate_source_integrity(self) -> None:
        """Authenticate retained scientific evidence after archive reconstruction."""
        counters = np.asarray(self.counters)
        roots = np.asarray(self.root_vertex_global_ids)
        weights = np.asarray(self.root_weights)
        deviations = np.asarray(self.root_deviations)
        if (
            counters.ndim != 1
            or counters.size < 5
            or self.inserted_roots != int(counters[2])
            or self.exact_zero_samples != int(counters[3])
            or self.work_units != int(counters[4])
            or self.topology_event_accepted != self.topology_changed
        ):
            raise ValueError(
                "LevelSetEvidence.counters must be consistent with the retained topology signature together with topology_events; got inconsistent retained counters or events."
            )
        if (
            roots.shape != (self.inserted_roots,)
            or np.asarray(self.root_source_global_ids).shape != (roots.size, 2)
            or weights.shape != (roots.size, 2)
            or deviations.shape != (roots.size,)
            or not np.all(np.isfinite(weights))
            or np.any(weights <= 0.0)
            or not np.all(np.isfinite(deviations))
            or np.any(deviations < 0.0)
            or self.maximum_carrier_deviation != float(np.max(deviations, initial=0.0))
            or not np.all(np.isin(roots, np.asarray(self.zero_vertex_global_ids)))
        ):
            raise ValueError(
                "LevelSetEvidence.root_weights must retain one positive rational homogeneous root and certified deviation per inserted zero vertex; got inconsistent retained roots."
            )
        if canonical_fingerprint(self._scientific_payload()) != self.evidence_id:
            raise ValueError(
                "LevelSetEvidence.evidence_id must match the retained scientific content; got a stale scientific identity."
            )

    @checked
    def require_current(self, mesh: cell_mesh_owner.CellMesh, /) -> None:
        """Require matching mesh content identity and numerical revision."""
        self.validate_source_integrity()
        if (
            mesh.mesh_id != self.target_mesh_id
            or mesh.numeric_version != self.target_numeric_version
            or not np.array_equal(
                np.asarray(mesh.entity_set(3).entity_ids),
                np.asarray(self.cell_global_ids),
            )
        ):
            raise ValueError(
                "LevelSetEvidence.target_mesh_id must match the queried mesh together with target_numeric_version; got stale evidence or another mesh."
            )
        sides = np.asarray(self.cell_sides)
        zero_vertices = np.isin(
            np.asarray(mesh.vertex_global_ids), np.asarray(self.zero_vertex_global_ids)
        )
        zero_values = np.where(zero_vertices, 0.0, 1.0)
        facets = np.all(zero_vertices[_tetrahedral_connectivity(mesh).faces], axis=1)
        if (
            sides.shape != self.cell_global_ids.shape
            or not np.all(np.isin(sides, (-1, 1)))
            or not np.array_equal(
                np.asarray(mesh.vertex_global_ids)[zero_vertices],
                np.asarray(self.zero_vertex_global_ids),
            )
            or not np.array_equal(
                np.asarray(mesh.entity_set(2).entity_ids)[facets],
                np.asarray(self.interface_facet_global_ids),
            )
            or self.topology_signature
            != _topology_signature(mesh, zero_values, sides, facets)
            or not np.array_equal(
                _interface_orientations(mesh, sides, facets),
                np.asarray(self.interface_orientations),
            )
            or int(self.counters[0]) != mesh.coordinates.shape[0]
            or int(self.counters[1]) != mesh.entity_set(3).count
        ):
            raise ValueError(
                "LevelSetEvidence.partition must match the retained target partition; got contradictory sides, interface facets, adjacency, or topology."
            )


def _request(prepared: PreparedMeshAdaptation, /) -> LevelSetMeshAdaptation:
    match prepared.request:
        case LevelSetMeshAdaptation() as request:
            return request
        case _:
            raise TypeError(
                f"PreparedMeshAdaptation.request must be LevelSetMeshAdaptation for the native level-set route; got {type(prepared.request).__name__}."
            )


def _source_values(prepared: PreparedMeshAdaptation, /) -> np.ndarray:
    request = _request(prepared)
    mesh = prepared.source.mesh
    resolve_mesh_scope(mesh, request.scope)
    ids = np.asarray(request.scope.entity_ids, dtype=np.int64)
    if ids.size != mesh.coordinates.shape[0]:
        raise ValueError(
            "LevelSetMeshAdaptation.scope must cover every source vertex; got a different vertex count."
        )
    return np.asarray(request.values, dtype=np.float64)[
        np.searchsorted(ids, np.asarray(mesh.vertex_global_ids))
    ]


def validate_level_set_adaptation(
    prepared: PreparedMeshAdaptation,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> None:
    """Validate the level-set request at the adaptation preparation boundary."""
    with ExitStack() as owner:
        if budget is None:
            limits = prepared.policy.limits
            ticket = owner.enter_context(
                HostBatchTicket(
                    limits.maximum_work_units,
                    queries_upper=limits.maximum_geometry_queries,
                    bytes_upper=limits.maximum_scratch_bytes,
                )
            )
            budget = CoordinateEnclosureBudget(
                ticket,
                maximum_work_units=limits.maximum_work_units,
                maximum_memory_bytes=limits.maximum_scratch_bytes,
            )
            owner.enter_context(budget.activate())
        mesh = prepared.source.mesh
        request = _request(prepared)
        if (
            mesh.topological_dimension != 3
            or mesh.ambient_dimension != 3
            or any(block.cell_kind != "tetrahedron" for block in mesh.blocks)
        ):
            raise ValueError(
                "PreparedMeshAdaptation.source.mesh must be a tetrahedral volume mesh for native level-set insertion; got an unsupported mesh dimension or family."
            )
        if not is_affine_cell_geometry(mesh, prepared.source.geometry, budget=budget):
            raise ValueError(
                "PreparedMeshAdaptation.source.geometry must be affine for P1 zero insertion; got non-affine source geometry."
            )
        values = _source_values(prepared)
        if any(
            np.any(np.all(values[np.asarray(block.vertices)] == 0.0, axis=1))
            for block in mesh.blocks
        ):
            raise ValueError(
                "LevelSetMeshAdaptation.values must permit an exclusive inside/outside assignment per source cell; got an identically zero source cell."
            )
        previous = request.previous
        if previous is not None:
            ticket = budget.ticket
            if isinstance(ticket, NativeBatchTicket):
                # Replay predicates must consume the reserved execution phase,
                # not the scope's unreserved remainder.
                with ticket.native_target():
                    previous.require_current(mesh)
            else:
                previous.require_current(mesh)
        existing = {
            *prepared.source.patches.names,
            *(
                value.name
                for value in (*prepared.source.zones, *prepared.source.labels)
            ),
        }
        previous_names = (
            set()
            if previous is None
            else {
                previous.inside_region_id,
                previous.outside_region_id,
                previous.interface_id,
            }
        )
        if (
            existing.intersection(
                (
                    request.inside_region_id,
                    request.outside_region_id,
                    request.interface_id,
                )
            )
            - previous_names
        ):
            raise ValueError(
                "LevelSetMeshAdaptation.region_ids must avoid existing mesh organization identities together with interface_id; got an identity collision."
            )
        periodic = mesh.periodic_topology
        if periodic is not None and not np.array_equal(
            values, values[np.asarray(periodic.vertex_representatives)]
        ):
            raise ValueError(
                "LevelSetMeshAdaptation.values must agree on periodic vertex orbits; got unequal orbit values."
            )


def _tetrahedral_connectivity(mesh: cell_mesh_owner.CellMesh, /) -> _TetrahedralRows:
    """Narrow the admitted volume carrier at each native topology boundary.

    Exact root geometry publishes one block per distinct child action, so an
    all-tetrahedron target may carry packed polyhedral incidence; its faces are
    then the triangles of the same face entities.
    """
    match mesh.connectivity:
        case TetrahedralConnectivity(faces=faces, edges=edges):
            return _TetrahedralRows(
                np.asarray(faces, dtype=np.int64), np.asarray(edges, dtype=np.int64)
            )
        case PolyhedralConnectivity(
            face_vertex_offsets=offsets, face_vertex_values=values, edges=edges
        ) if all(block.cell_kind == "tetrahedron" for block in mesh.blocks):
            offsets_ = np.asarray(offsets, dtype=np.int64)
            if not np.all(np.diff(offsets_) == 3) or offsets_[0] != 0:
                raise ValueError(
                    "CellMesh.connectivity must have triangular faces for native level-set topology; got a non-triangular face."
                )
            return _TetrahedralRows(
                np.asarray(values, dtype=np.int64).reshape((-1, 3)),
                np.asarray(edges, dtype=np.int64),
            )
        case _:
            raise ValueError(
                "CellMesh.connectivity must be tetrahedral for native level-set topology; got a different connectivity kind."
            )


def _exact_source_positions(
    prepared: PreparedMeshAdaptation,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[ExactScalar, ...], ...] | None:
    """Return exact source vertex positions, or None when the carrier is exactly the source.

    The structural owner proof is the fast path. Otherwise the affine source
    map is evaluated once through its prepared source bank; positions equal to
    the binary64 carrier keep native carrier output, while any other unrounded
    source (for example a non-dyadic root of an earlier adaptation) owns its
    roots. Non-dyadic positions alone never refuse.
    """
    mesh = prepared.source.mesh
    geometry = prepared.source.geometry
    if affine_geometry_is_carrier(mesh, geometry):
        return None
    positions = affine_vertex_positions(mesh, geometry, budget=budget)
    # One rational lift and comparison per carrier coordinate.
    budget.reserve(2 * index(np.asarray(mesh.coordinates).size))
    return None if vertex_positions_are_carrier(positions, mesh) else positions


def _native_split(
    prepared: PreparedMeshAdaptation,
    /,
    *,
    output_mode: LevelSetOutputMode,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
) -> _NativeTopology:
    mesh = prepared.source.mesh
    connectivity = _tetrahedral_connectivity(mesh)
    points = np.ascontiguousarray(mesh.coordinates, dtype=np.float64)
    values = np.ascontiguousarray(_source_values(prepared), dtype=np.float64)
    cells = np.ascontiguousarray(
        np.concatenate([np.asarray(block.vertices) for block in mesh.blocks]),
        dtype=np.int32,
    )
    faces = np.ascontiguousarray(connectivity.faces, dtype=np.int32)
    edges = np.ascontiguousarray(connectivity.edges, dtype=np.int32)
    protection = np.ascontiguousarray(
        prepared.constraints.protected_edge_mask, dtype=np.int8
    )
    edge_ids = np.sort(np.asarray(mesh.vertex_global_ids)[edges], axis=1)
    periodic = mesh.periodic_topology
    if periodic is None:
        edge_order = np.lexsort((edge_ids[:, 1], edge_ids[:, 0]))
    else:
        edge_order = np.lexsort(
            (edge_ids[:, 1], edge_ids[:, 0], np.asarray(periodic.orbits(1)[0]))
        )
    edges = np.ascontiguousarray(edges[edge_order], dtype=np.int32)
    protection = np.ascontiguousarray(protection[edge_order], dtype=np.int8)
    limits = prepared.policy.limits
    edge_values = values[edges]
    # count_nonzero returns a NumPy scalar; static native metadata is a Python int.
    crossings = index(
        np.count_nonzero(
            (np.min(edge_values, axis=1) < 0.0) & (np.max(edge_values, axis=1) > 0.0)
        )
    )
    vertex_capacity = min(points.shape[0] + crossings, limits.maximum_vertices)
    cell_values = values[cells]
    roots_per_cell = np.count_nonzero(cell_values < 0.0, axis=1) * np.count_nonzero(
        cell_values > 0.0, axis=1
    )
    # The native split bisects each crossing source edge once; a cell gains one
    # child per descendant containing that edge, independently of its
    # neighbours. With k = negatives x positives crossings, a 1/1, 1/2 or 1/3
    # sign split shares one apex (k + 1 children) and a 2/2 split yields 6 in
    # every edge order. Uncut cells keep one output slot.
    combinatorial_bound = np.asarray((1, 2, 3, 4, 6), dtype=np.int64)
    cell_capacity = min(
        int(np.sum(combinatorial_bound[roots_per_cell])), limits.maximum_cells
    )
    # Conservative bound for native source-validation tables, incidence lists
    # and unrounded root temporaries, not merely Python output arrays. Refuse before
    # allocating the work buffers.
    scratch_bound = 4096 * (
        vertex_capacity + cell_capacity + faces.shape[0] + edges.shape[0]
    )
    if scratch_bound > limits.maximum_scratch_bytes:
        raise ValueError(
            f"LevelSetMeshAdaptation.working_storage must satisfy maximum_scratch_bytes; got required={scratch_bound}, capacity={limits.maximum_scratch_bytes}."
        )
    remaining = scope.remaining()
    ticket = scope.batch(
        min(limits.maximum_work_units, remaining.remaining_work_units),
        bytes_upper=scratch_bound,
        cavity_upper=min(limits.maximum_cavity_cells, remaining.remaining_cavity_cells),
    )
    carriers = output_mode is LevelSetOutputMode.FULL_CARRIERS
    with budget.bind_ticket(ticket):
        output_points = (
            np.empty((vertex_capacity, 3), dtype=np.float64) if carriers else None
        )
        output_cells = np.empty((cell_capacity, 4), dtype=np.int32)
        parents = np.empty((cell_capacity,), dtype=np.int32)
        sources = np.empty((vertex_capacity, 2), dtype=np.int32)
        weights = np.empty((vertex_capacity, 2), dtype=np.float64)
        deviations = np.empty((vertex_capacity,), dtype=np.float64) if carriers else None
        counter_slots = np.zeros((6,), dtype=np.int64)
        budget.retain_basis(
            tuple(
                array
                for array in (
                    output_points,
                    output_cells,
                    parents,
                    sources,
                    weights,
                    deviations,
                    counter_slots,
                )
                if array is not None
            )
        )
        with ticket.native_target():
            run = level_set_split_3d(
                LevelSetSplitRequest(
                    points=points,
                    values=values,
                    cells=cells,
                    faces=faces,
                    edges=edges,
                    protected_edges=protection,
                    max_vertices=vertex_capacity,
                    max_cells=cell_capacity,
                    work_limit=ticket.remaining_work_units,
                    cavity_limit=limits.maximum_cavity_cells,
                    output_mode=output_mode,
                    output_points=output_points,
                    output_cells=output_cells,
                    output_parents=parents,
                    output_sources=sources,
                    output_weights=weights,
                    output_deviations=deviations,
                    counter_slots=counter_slots,
                )
            )
        status = run.status
        counters = run.counters
        if status is not MeshcoreStatus.OK:
            raise MeshcoreError(
                status,
                f"LevelSetMeshAdaptation.native_split must satisfy its declared native contracts; got status={status.name}, failed_edge={counters.failed_edge}, work={counters.work_units}.",
            )
        nv, nc = counters.vertices, counters.cells
        return _NativeTopology(
            output_cells[:nc],
            parents[:nc],
            sources[:nv],
            weights[:nv],
            counters,
            None if output_points is None else output_points[:nv],
            None if deviations is None else deviations[points.shape[0] : nv],
        )


def _certified_split(
    prepared: PreparedMeshAdaptation,
    topology: _NativeTopology,
    positions: tuple[tuple[ExactScalar, ...], ...] | None,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> _Split:
    """Attach carriers certified from the exact source to the native topology.

    Native carriers are certified by the kernel when the source carrier is the
    exact source. Otherwise the input vertex bytes are kept, each root is
    the correctly rounded exact root over the exact source positions with its
    upward Euclidean deviation, and every target cell and carrier is certified
    with the native exact predicates exactly as the kernel certifies its own.
    """
    if positions is None:
        if topology.coordinates is None or topology.root_deviations is None:
            raise RuntimeError(
                "LevelSetMeshAdaptation.native_split must publish carriers for an exact source carrier; got topology-only output."
            )
        return _Split(
            topology.coordinates,
            topology.cells,
            topology.parents,
            topology.sources,
            topology.weights,
            topology.root_deviations,
            topology.counters,
        )
    source = np.asarray(prepared.source.mesh.coordinates, dtype=np.float64)
    count = source.shape[0]
    roots, deviations = exact_root_points(
        positions,
        topology.sources[count:],
        topology.weights[count:],
        budget=budget,
    )
    coordinates = np.concatenate((source, roots), axis=0)
    _certify_carriers(coordinates, roots, topology, budget=budget)
    return _Split(
        coordinates,
        topology.cells,
        topology.parents,
        topology.sources,
        topology.weights,
        deviations,
        topology.counters,
    )


def _certify_carriers(
    coordinates: np.ndarray,
    roots: np.ndarray,
    topology: _NativeTopology,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> None:
    """Refuse carriers outside the meshcore coordinate domain, nonpositive cells and coincidences."""
    counters = topology.counters
    minimum, maximum = meshcore_exact_domain()
    magnitude = np.abs(roots)
    if np.any(
        (magnitude != 0.0)
        & ((magnitude < np.ldexp(1.0, minimum)) | (magnitude > np.ldexp(1.0, maximum)))
    ):
        raise MeshcoreError(
            MeshcoreStatus.INVALID_INPUT,
            f"LevelSetMeshAdaptation.exact_carriers must lie in the meshcore exact arithmetic domain; got status=INVALID_INPUT, inserted={counters.inserted}.",
        )
    cells = np.asarray(topology.cells, dtype=np.int64)
    count = coordinates.shape[0]
    levels = max(1, (count - 1).bit_length() + 1)
    with budget.temporary_scope():
        # Corner gathers, one exact orientation per cell and one sorted
        # coincidence pass, as the kernel charges its own certification.
        budget.reserve(
            16 * cells.shape[0] + count * levels,
            cells.size * 3 * 8 + cells.shape[0] + count * (3 * 8 + 8),
        )
        corners = coordinates[cells]
        ticket = budget.ticket
        with (
            ticket.native_target()
            if isinstance(ticket, NativeBatchTicket)
            else nullcontext()
        ):
            signs = exact_orient3d(
                corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]
            )
        order = np.lexsort((coordinates[:, 2], coordinates[:, 1], coordinates[:, 0]))
        ordered = coordinates[order]
        coincident = np.all(ordered[1:] == ordered[:-1], axis=1)
    if np.any(signs <= 0) or np.any(coincident):
        raise MeshcoreError(
            MeshcoreStatus.DEGENERATE_INPUT,
            f"LevelSetMeshAdaptation.exact_carriers must form positively oriented cells with distinct carriers; got status=DEGENERATE_INPUT, inserted={counters.inserted}.",
        )


def _edit(
    prepared: PreparedMeshAdaptation,
    split: _Split,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> CellTopologyEdit:
    mesh = prepared.source.mesh
    old_vertices = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    nv = old_vertices.size
    vertex_ids = np.concatenate(
        (
            old_vertices,
            np.arange(
                int(np.max(old_vertices)) + 1,
                int(np.max(old_vertices)) + 1 + split.coordinates.shape[0] - nv,
                dtype=np.int64,
            ),
        )
    )
    sources = old_vertices[split.sources]
    source_cells = np.concatenate([np.asarray(block.vertices) for block in mesh.blocks])
    source_ids = np.concatenate([np.asarray(block.global_ids) for block in mesh.blocks])
    cell_ids = np.arange(
        int(np.max(source_ids)) + 1,
        int(np.max(source_ids)) + 1 + split.cells.shape[0],
        dtype=np.int64,
    )
    unchanged = np.all(
        np.sort(split.cells, axis=1) == np.sort(source_cells[split.parents], axis=1),
        axis=1,
    )
    cell_ids[unchanged] = source_ids[split.parents[unchanged]]
    blocks = []
    offset = 0
    order_parts = []
    for block in mesh.blocks:
        selected = np.flatnonzero(
            (split.parents >= offset) & (split.parents < offset + block.cell_count)
        )
        selected = selected[np.argsort(cell_ids[selected], kind="stable")]
        order_parts.append(selected)
        blocks.append(
            TopologyEditBlock(
                block.name,
                "tetrahedron",
                "tetrahedron",
                split.cells[selected],
                cell_ids[selected],
            )
        )
        offset += block.cell_count
    order = np.concatenate(order_parts)
    target_cells = split.cells[order]
    target_ids = cell_ids[order]
    parents = split.parents[order]
    relations = [
        _support_relations(mesh, degree, target_cells, vertex_ids, split.sources)
        for degree in range(3)
    ]
    relations.append(
        EntityRelations(
            3,
            source_ids[parents, None],
            target_ids[:, None],
            np.where(
                unchanged[order],
                EntityLineageKind.PRESERVED,
                EntityLineageKind.REFINED_FROM,
            ).astype(np.int32),
        )
    )
    # Exact homogeneous roots own the geometry; rounded barycentrics are only
    # their correctly rounded views.
    valid = np.ones(sources.shape, dtype=np.bool_)
    barycentrics = rounded_homogeneous_barycentrics(split.weights, valid, budget=budget)
    reference_vertices, reference_homogeneous = exact_nested_reference_vertices(
        sources[target_cells],
        split.weights[target_cells],
        old_vertices[source_cells[parents]],
        budget=budget,
    )
    witnesses = NestedReferenceWitnesses(
        target_ids, source_ids[parents], reference_vertices, reference_homogeneous
    )
    periodic_orbits = None
    if mesh.periodic_topology is not None:
        carrier = cell_mesh_owner.CellMesh(
            split.coordinates,
            tuple(
                reference_cell_types_owner.CellBlock(
                    block.name, block.cell_kind, block.cells, global_ids=block.cell_ids
                )
                for block in blocks
            ),
            vertex_global_ids=vertex_ids,
        )
        representative_ids, exponents = _periodic_target_orbits(mesh, vertex_ids, sources)
        periodic_orbits = periodic_owner.periodic_vertex_orbit_witness(
            mesh, carrier, representative_ids, exponents, None, budget=budget
        )
    return CellTopologyEdit(
        "nested_refinement",
        split.coordinates,
        vertex_ids,
        tuple(blocks),
        sources,
        barycentrics,
        valid,
        tuple(relations),
        refinement=witnesses,
        periodic_orbits=periodic_orbits,
        stencil_homogeneous=split.weights,
    )


def _support_relations(
    mesh: cell_mesh_owner.CellMesh,
    degree: int,
    cells: np.ndarray,
    vertex_ids: np.ndarray,
    source_rows: np.ndarray,
    /,
) -> EntityRelations:
    width = 1 if degree == 0 else degree + 1
    if degree == 0:
        return EntityRelations(
            0,
            np.zeros((0, 1), dtype=np.int64),
            np.zeros((0, 1), dtype=np.int64),
            np.zeros((0,), dtype=np.int32),
        )

    keys = {
        tuple(sorted(vertex_ids[cell[list(local)]]))
        for cell in cells
        for local in combinations(range(4), width)
    }
    old = {tuple(key): key for key in entity_keys(mesh, degree)}
    source_ids = np.asarray(mesh.vertex_global_ids)
    row_by_id = {int(identifier): row for row, identifier in enumerate(vertex_ids)}
    source_keys, target_keys = [], []
    for key in sorted(keys):
        support = tuple(
            sorted(
                {
                    int(source_ids[source])
                    for identifier in key
                    for source in source_rows[row_by_id[int(identifier)]]
                }
            )
        )
        if support in old and support != key:
            source_keys.append(support)
            target_keys.append(key)
    return EntityRelations(
        degree,
        np.asarray(source_keys, dtype=np.int64).reshape((-1, width)),
        np.asarray(target_keys, dtype=np.int64).reshape((-1, width)),
        np.full((len(source_keys),), EntityLineageKind.REFINED_FROM, dtype=np.int32),
    )


def _periodic_target_orbits(
    source: cell_mesh_owner.CellMesh, vertex_ids: np.ndarray, sources: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Return global representative IDs and group exponents for zero-insertion rows.

    Old rows keep their scientific quotient membership. A born vertex lies on a
    split source edge; all born vertices of one quotient-edge orbit form one new
    vertex orbit whose representative is its smallest (first issued)
    global ID.
    """
    periodic = source.periodic_topology
    if periodic is None:
        raise ValueError(
            "CellMesh.periodic_topology must be supplied for periodic zero-insertion orbits; got None."
        )
    old_ids = np.asarray(source.vertex_global_ids, dtype=np.int64)
    count = old_ids.size
    old_shifts = np.asarray(periodic.vertex_shifts, dtype=np.int64)
    if old_shifts.ndim != 2 or old_shifts.shape[0] != count:
        raise ValueError(
            f"CellMesh.periodic_topology.vertex_shifts must have one exponent row per vertex; got shape={old_shifts.shape}, vertices={count}."
        )
    roots = np.asarray(periodic.vertex_representatives, dtype=np.int64)
    representatives = np.empty(vertex_ids.shape, dtype=np.int64)
    shifts = np.zeros((vertex_ids.size, old_shifts.shape[1]), dtype=np.int64)
    representatives[:count] = old_ids[roots]
    shifts[:count] = old_shifts
    order = np.argsort(old_ids, kind="stable")
    groups: dict[tuple[int, ...], tuple[int, np.ndarray]] = {}
    for row in range(count, vertex_ids.size):
        a, b = (
            int(value) for value in order[np.searchsorted(old_ids[order], sources[row])]
        )
        first_key = (
            int(roots[a]),
            int(roots[b]),
            *map(int, old_shifts[b] - old_shifts[a]),
        )
        second_key = (
            int(roots[b]),
            int(roots[a]),
            *map(int, old_shifts[a] - old_shifts[b]),
        )
        key, anchor = (
            (first_key, old_shifts[a])
            if first_key <= second_key
            else (second_key, old_shifts[b])
        )
        representative, origin = groups.setdefault(key, (row, anchor))
        representatives[row] = vertex_ids[representative]
        shifts[row] = anchor - origin
    return representatives, shifts


def _partition(
    prepared: PreparedMeshAdaptation,
    mesh: cell_mesh_owner.CellMesh,
    edit: CellTopologyEdit,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    connectivity = _tetrahedral_connectivity(mesh)
    values = np.concatenate(
        (
            _source_values(prepared),
            np.zeros(
                (mesh.coordinates.shape[0] - prepared.source.mesh.coordinates.shape[0],),
                dtype=np.float64,
            ),
        )
    )
    cells = np.concatenate([np.asarray(block.vertices) for block in mesh.blocks])
    ids = np.concatenate([np.asarray(block.global_ids) for block in mesh.blocks])
    cell_values = values[cells]
    low, high = np.min(cell_values, axis=1), np.max(cell_values, axis=1)
    if np.any((low < 0.0) & (high > 0.0)) or np.any((low == 0.0) & (high == 0.0)):
        raise ValueError(
            "LevelSetMeshAdaptation.target_partition must have an exclusive side partition; got a crossing or identically zero target cell."
        )
    block_sides = np.where(low < 0.0, -1, 1).astype(np.int8)
    order = np.argsort(ids)
    sides = block_sides[
        order[np.searchsorted(ids[order], np.asarray(mesh.entity_set(3).entity_ids))]
    ]
    relation = mesh.topology.incidences[2].relation
    valid = np.asarray(relation.valid)
    face_rows = np.asarray(relation.source_indices)[valid]
    incident_sides = sides[np.asarray(relation.target_indices)[valid]]
    periodic = mesh.periodic_topology
    orbit = (
        np.arange(mesh.entity_set(2).count)
        if periodic is None
        else np.asarray(periodic.orbits(2)[0])
    )
    negative = np.zeros((mesh.entity_set(2).count,), dtype=np.bool_)
    positive = negative.copy()
    negative[orbit[face_rows[incident_sides < 0]]] = True
    positive[orbit[face_rows[incident_sides > 0]]] = True
    separating = (negative & positive)[orbit]
    facets = np.all(values[np.asarray(connectivity.faces)] == 0.0, axis=1)
    if np.any(separating & ~facets):
        raise ValueError(
            "LevelSetMeshAdaptation.target_partition must support inside/outside adjacency with a zero facet; got separating cells without a zero facet."
        )
    return values, sides, facets


def _interface_orientations(
    mesh: cell_mesh_owner.CellMesh, sides: np.ndarray, facets: np.ndarray, /
) -> np.ndarray:
    """Return inside-to-outside signs, with zero for nonseparating faces."""
    connectivity = _tetrahedral_connectivity(mesh)
    selected_faces = np.flatnonzero(facets)
    if selected_faces.size == 0:
        return np.empty((0,), dtype=np.int8)
    cells = np.concatenate([np.asarray(block.vertices) for block in mesh.blocks])
    ids = np.concatenate([np.asarray(block.global_ids) for block in mesh.blocks])
    order = np.argsort(ids)
    cells = cells[
        order[np.searchsorted(ids[order], np.asarray(mesh.entity_set(3).entity_ids))]
    ]
    relation = mesh.topology.incidences[2].relation
    valid = np.asarray(relation.valid)
    owner: dict[int, int] = {}
    periodic = mesh.periodic_topology
    orbit = (
        np.arange(facets.size) if periodic is None else np.asarray(periodic.orbits(2)[0])
    )
    incident_sides: dict[int, set[int]] = {}
    for face, cell in zip(
        np.asarray(relation.source_indices)[valid],
        np.asarray(relation.target_indices)[valid],
        strict=True,
    ):
        owner.setdefault(int(face), int(cell))
        incident_sides.setdefault(int(orbit[face]), set()).add(int(sides[cell]))
    face_vertices = np.asarray(connectivity.faces)[selected_faces]
    cell_rows = np.asarray([owner[int(face)] for face in selected_faces], dtype=np.int64)
    apex = np.asarray(
        [
            next(int(vertex) for vertex in cells[row] if vertex not in face)
            for row, face in zip(cell_rows, face_vertices, strict=True)
        ],
        dtype=np.int64,
    )
    points = np.asarray(mesh.coordinates)
    triangles = points[face_vertices]
    signs = exact_orient3d(
        triangles[:, 0], triangles[:, 1], triangles[:, 2], points[apex]
    )
    if np.any(signs == 0):
        raise ValueError(
            "LevelSetMeshAdaptation.interface_geometry must be nondegenerate; got degenerate incident geometry."
        )
    separating = np.asarray(
        [len(incident_sides[int(orbit[face])]) == 2 for face in selected_faces]
    )
    return np.where(separating, signs * sides[cell_rows], 0).astype(np.int8)


def _components(nodes: set[int], links: list[tuple[int, int]], /) -> int:
    neighbors = {node: [] for node in nodes}
    for first, second in links:
        if first in neighbors and second in neighbors:
            neighbors[first].append(second)
            neighbors[second].append(first)
    count = 0
    unseen = set(nodes)
    while unseen:
        stack = [unseen.pop()]
        count += 1
        while stack:
            for neighbor in neighbors[stack.pop()]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
    return count


def _topology_signature(
    mesh: cell_mesh_owner.CellMesh,
    values: np.ndarray,
    sides: np.ndarray,
    facets: np.ndarray,
    /,
) -> tuple[int, int, int, int, int]:
    connectivity = _tetrahedral_connectivity(mesh)
    relation = mesh.topology.incidences[2].relation
    valid = np.asarray(relation.valid)
    face_rows = np.asarray(relation.source_indices)[valid]
    cell_rows = np.asarray(relation.target_indices)[valid]
    periodic = mesh.periodic_topology
    cell_orbits = (
        np.arange(sides.size) if periodic is None else np.asarray(periodic.orbits(3)[0])
    )
    face_orbits = (
        np.arange(facets.size) if periodic is None else np.asarray(periodic.orbits(2)[0])
    )
    owners: dict[int, set[int]] = {}
    for face, cell in zip(face_orbits[face_rows], cell_orbits[cell_rows], strict=True):
        owners.setdefault(int(face), set()).add(int(cell))
    zero_faces = set(map(int, face_orbits[facets]))
    links = [
        (min(rows), max(rows))
        for face, rows in owners.items()
        if len(rows) == 2 and face not in zero_faces
    ]
    inside = _components(set(map(int, cell_orbits[sides < 0])), links)
    outside = _components(set(map(int, cell_orbits[sides > 0])), links)
    zero_vertices = np.flatnonzero(values == 0.0)
    all_edges = np.asarray(connectivity.edges)
    zero_edge_rows = np.flatnonzero(np.all(values[all_edges] == 0.0, axis=1))
    edges = all_edges[zero_edge_rows]
    dimension = (
        2 if np.any(facets) else 1 if edges.shape[0] else 0 if zero_vertices.size else -1
    )
    if periodic is None:
        vertices = set(map(int, zero_vertices))
        links = [(int(edge[0]), int(edge[1])) for edge in edges]
        return (
            inside,
            outside,
            _components(vertices, links),
            len(vertices) - edges.shape[0] + int(np.count_nonzero(facets)),
            dimension,
        )
    vertex_orbits = np.asarray(periodic.vertex_representatives)
    vertices = set(map(int, vertex_orbits[zero_vertices]))
    lifted_edges = vertex_orbits[edges]
    links = [(int(edge[0]), int(edge[1])) for edge in lifted_edges]
    edge_count = np.unique(np.asarray(periodic.orbits(1)[0])[zero_edge_rows]).size
    face_count = np.unique(face_orbits[facets]).size
    return (
        inside,
        outside,
        _components(vertices, links),
        len(vertices) - edge_count + face_count,
        dimension,
    )


def _interface_adjacent_sides(
    mesh: cell_mesh_owner.CellMesh, sides: np.ndarray, facets: np.ndarray, /
) -> tuple[int, ...]:
    """Return distinct sides bounded by each zero facet with one incident cell per side.

    Tangential or mixed facets have no distinct-zone adjacency, so none is declared.
    """
    relation = mesh.topology.incidences[2].relation
    valid = np.asarray(relation.valid)
    face_rows = np.asarray(relation.source_indices)[valid]
    cell_rows = np.asarray(relation.target_indices)[valid]
    selected = facets[face_rows]
    incident: dict[int, list[int]] = {}
    for face, cell in zip(face_rows[selected], cell_rows[selected], strict=True):
        incident.setdefault(int(face), []).append(int(sides[cell]))
    patterns = {tuple(sorted(values)) for values in incident.values()}
    if len(patterns) != 1:
        return ()
    (pattern,) = patterns
    return pattern if len(set(pattern)) == len(pattern) else ()


def level_set_organization(
    prepared: PreparedMeshAdaptation,
    mesh: cell_mesh_owner.CellMesh,
    patches: tuple[MeshPatch, ...],
    zones: tuple[MeshZone, ...],
    labels: tuple[MeshLabel, ...],
    edit: CellTopologyEdit,
    /,
) -> tuple[tuple[MeshPatch, ...], tuple[MeshZone, ...], tuple[MeshLabel, ...]]:
    """Construct level-set organization before target certification."""
    request = _request(prepared)
    values, sides, facets = _partition(prepared, mesh, edit)
    previous = request.previous
    signature = _topology_signature(mesh, values, sides, facets)
    if (
        previous is not None
        and signature != previous.topology_signature
        and not request.accept_topology_change
    ):
        raise ValueError(
            "LevelSetMeshAdaptation.accept_topology_change must be True when the moving zero set changes topology; got False."
        )
    retired = (
        set()
        if previous is None
        else {
            previous.inside_region_id,
            previous.outside_region_id,
            previous.interface_id,
        }
    )
    patches = tuple(value for value in patches if value.name not in retired)
    zones = tuple(value for value in zones if value.name not in retired)
    labels = tuple(value for value in labels if value.name not in retired)
    material = any(zone.role is MeshZoneRole.REGION for zone in zones)
    new_zones, new_labels = [], []
    for sign, identity in (
        (-1, request.inside_region_id),
        (1, request.outside_region_id),
    ):
        ids = np.asarray(mesh.entity_set(3).entity_ids)[sides == sign]
        if ids.size:
            scope = MeshingScope(
                mesh.mesh_id,
                mesh.numeric_version,
                MeshingEntityKind.MESH,
                3,
                mesh.entity_set(3).entity_set_id,
                ids,
            )
            if material:
                new_labels.append(MeshLabel(identity, scope))
            else:
                new_zones.append(MeshZone(identity, MeshZoneRole.REGION, scope))
    if np.any(facets):
        scope = MeshingScope(
            mesh.mesh_id,
            mesh.numeric_version,
            MeshingEntityKind.MESH,
            2,
            mesh.entity_set(2).entity_set_id,
            np.asarray(mesh.entity_set(2).entity_ids)[facets],
        )
        adjacent = _interface_adjacent_sides(mesh, sides, facets)
        zone_sides = {request.inside_region_id: -1, request.outside_region_id: 1}
        adjacent_zone_ids = tuple(
            zone.zone_id for zone in new_zones if zone_sides[zone.name] in adjacent
        )
        patches += (
            MeshPatch(
                request.interface_id,
                scope,
                connected=signature[2] == 1,
                adjacent_zone_ids=adjacent_zone_ids,
            ),
        )
    return patches, zones + tuple(new_zones), labels + tuple(new_labels)


def execute_level_set_route(
    prepared: PreparedMeshAdaptation,
    /,
    *,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
) -> adaptation_owner.RouteOutcome:
    """Execute the level-set route admitted by execute_mesh_adaptation."""

    # Lazy: _level_set <-> _adaptation import cycle.
    from . import _adaptation as adaptation_owner

    with _level_set_coordinate_budget(prepared, scope=scope, budget=budget):
        validate_level_set_adaptation(prepared, budget=budget)
        positions = _exact_source_positions(prepared, budget=budget)
    topology = _native_split(
        prepared,
        output_mode=LevelSetOutputMode.FULL_CARRIERS
        if positions is None
        else LevelSetOutputMode.TOPOLOGY_ONLY,
        scope=scope,
        budget=budget,
    )
    with _level_set_coordinate_budget(prepared, scope=scope, budget=budget):
        split = _certified_split(prepared, topology, positions, budget=budget)
        edit = _edit(prepared, split, budget=budget)
    native = adaptation_owner.finalize_native(
        prepared,
        edit,
        MeshTransitionKind.REFINE,
        conservative=True,
        scope=scope,
        budget=budget,
    )
    evidence = LevelSetEvidence(prepared, native.target.mesh, edit, split)
    limits = prepared.policy.limits
    if (
        native.target.mesh.entity_set(1).count > limits.maximum_edges
        or native.target.mesh.entity_set(2).count > limits.maximum_faces
    ):
        raise ValueError(
            "LevelSetMeshAdaptation.target_partition must satisfy edge/face limits; got excessive target edges or faces."
        )
    data_bytes = native.target.mesh.coordinates.size * 8 + sum(
        block.vertices.size * 8 for block in native.target.mesh.blocks
    )
    if data_bytes > limits.maximum_data_bytes:
        raise ValueError(
            f"LevelSetMeshAdaptation.target_partition must satisfy maximum_data_bytes; got bytes={data_bytes}, capacity={limits.maximum_data_bytes}."
        )
    return adaptation_owner.RouteOutcome(
        adaptation_owner.MeshAdaptationStatus.COMPLETE,
        native.target,
        native.transition,
        native.lineage,
        native.stencil,
        native.transfer,
        None,
        evidence,
        None,
    )


__all__ = ["LevelSetMeshAdaptation", "LevelSetEvidence"]
