#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native existing-mesh P1 zero interfaces and conservative W08 transitions.

The scalar is the declared piecewise-affine vertex field, not a sampled oracle
claiming hidden-component completeness. Native roots split protected source
strata through meshcore cavity transactions. Material identity is orthogonal to
level-set side identity: material assignments inherit exact cell lineage.
Accepted moving interfaces bind the explicit target numerical revision even
when coordinates and connectivity, and therefore content identity, are unchanged.
Tangential zero faces remain interface patches without declaring distinct-zone
adjacency when both incident cells belong to the same side.
"""

from __future__ import annotations

from typing import final, NamedTuple, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._meshcore import exact_orient3d, load_meshcore, MeshcoreError, MeshcoreStatus
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellBlock, CellMesh
from ..discretization._cell_complex import TetrahedralConnectivity
from ..discretization._cell_geometry_transfer import (
    is_affine_cell_geometry,
    NestedReferenceWitnesses,
)
from ._lineage import EntityLineageKind, MeshTransitionKind
from ._organization import MeshLabel, MeshPatch, MeshZone, MeshZoneRole
from ._scope import MeshingEntityKind, MeshingScope, resolve_mesh_scope
from ._topology_edit import (
    CellTopologyEdit,
    entity_keys,
    EntityRelations,
    nested_reference_vertices,
    TopologyEditBlock,
)


if TYPE_CHECKING:
    from ._adaptation import _RouteOutcome, PreparedMeshAdaptation


@final
class LevelSetMeshAdaptation(StrictModule, NonTrainableState):
    """Insert the zero set of one mesh-bound scalar P1 field.

    ``values`` follow ``scope.entity_ids``, not incidental mesh row order. Exact
    zero samples remain existing vertices. ``previous`` binds the last accepted
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
        if not isinstance(scope, MeshingScope):
            raise TypeError("scope must be MeshingScope.")
        if scope.entity_kind is not MeshingEntityKind.MESH or scope.entity_dimension != 0:
            raise ValueError("A level set must be bound to mesh vertices.")
        scalar = np.asarray(values, dtype=np.float64)
        if scalar.shape != scope.entity_ids.shape or not np.all(np.isfinite(scalar)):
            raise ValueError(
                "Level-set values must be one finite scalar per scoped vertex."
            )
        identities = (inside_region_id, outside_region_id, interface_id)
        if any(not isinstance(value, str) for value in identities):
            raise TypeError("Level-set identities must be strings.")
        if (
            any(not value.strip() or value != value.strip() for value in identities)
            or len(set(identities)) != 3
        ):
            raise ValueError(
                "Inside, outside and interface identities must be distinct nonempty identifiers."
            )
        if previous is not None and not isinstance(previous, LevelSetEvidence):
            raise TypeError("previous must be LevelSetEvidence or None.")
        if not isinstance(accept_topology_change, bool):
            raise TypeError("accept_topology_change must be bool.")
        if previous is not None and identities != (
            previous.inside_region_id,
            previous.outside_region_id,
            previous.interface_id,
        ):
            raise ValueError(
                "A moving level set must preserve its scientific region/interface identity."
            )
        self.scope = scope
        self.values = jnp.asarray(scalar, dtype=jnp.float64)
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


class _Split(NamedTuple):
    coordinates: np.ndarray
    cells: np.ndarray
    parents: np.ndarray
    sources: np.ndarray
    weights: np.ndarray
    counters: np.ndarray


@final
class LevelSetEvidence(StrictModule, NonTrainableState):
    """P1 partition, oriented zero facets, root residuals and topology events.

    ``topology_signature`` records strict inside/outside component counts, zero
    complex component count, Euler characteristic and maximum stratum dimension.
    Exact-zero isolated vertices/edges and tangential zero faces are retained.
    Zero face orientation is ±1 inside-to-outside, or 0 for a nonseparating face.
    """

    source_mesh_id: str = eqx.field(static=True)
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
    counters: Array
    topology_signature: tuple[int, int, int, int, int] = eqx.field(static=True)
    topology_changed: bool = eqx.field(static=True)
    topology_event_accepted: bool = eqx.field(static=True)
    inserted_roots: int = eqx.field(static=True)
    exact_zero_samples: int = eqx.field(static=True)
    work_units: int = eqx.field(static=True)
    maximum_root_residual: float = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        prepared: PreparedMeshAdaptation,
        mesh: CellMesh,
        edit: CellTopologyEdit,
        counters: np.ndarray,
        /,
    ) -> None:
        request = _request(prepared)
        values, sides, facets = _partition(prepared, mesh, edit)
        signature = _topology_signature(mesh, values, sides, facets)
        previous = request.previous
        changed = previous is not None and signature != previous.topology_signature
        if changed and not request.accept_topology_change:
            raise ValueError(
                "The moving zero set changes topology; explicit accept_topology_change is required."
            )
        self.source_mesh_id = prepared.source.mesh.mesh_id
        self.request_id = request.request_id
        self.target_mesh_id = mesh.mesh_id
        self.target_numeric_version = mesh.numeric_version
        self.inside_region_id = request.inside_region_id
        self.outside_region_id = request.outside_region_id
        self.interface_id = request.interface_id
        self.cell_global_ids = jnp.asarray(mesh.entity_set(3).entity_ids, dtype=jnp.int64)
        self.cell_sides = jnp.asarray(sides, dtype=jnp.int8)
        self.interface_facet_global_ids = jnp.asarray(
            np.asarray(mesh.entity_set(2).entity_ids)[facets], dtype=jnp.int64
        )
        self.interface_orientations = jnp.asarray(
            _interface_orientations(mesh, sides, facets), dtype=jnp.int8
        )
        self.zero_vertex_global_ids = jnp.asarray(
            np.asarray(mesh.vertex_global_ids)[values == 0.0], dtype=jnp.int64
        )
        self.counters = jnp.asarray(counters, dtype=jnp.int64)
        self.topology_signature = signature
        self.topology_changed = changed
        self.topology_event_accepted = changed and request.accept_topology_change
        self.inserted_roots = int(counters[2])
        self.exact_zero_samples = int(counters[3])
        self.work_units = int(counters[4])
        source_ids = np.asarray(prepared.source.mesh.vertex_global_ids)
        order = np.argsort(source_ids)
        rows = order[np.searchsorted(source_ids[order], edit.stencil_sources)]
        root_values = np.sum(
            edit.stencil_weights * _source_values(prepared)[rows], axis=1
        )[source_ids.size :]
        self.maximum_root_residual = float(np.max(np.abs(root_values), initial=0.0))
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "native-level-set-evidence",
                "request": request.request_id,
                "source": self.source_mesh_id,
                "target": self.target_mesh_id,
                "target_revision": self.target_numeric_version,
                "cells": array_tree_fingerprint((self.cell_global_ids, self.cell_sides)),
                "facets": array_tree_fingerprint(
                    (self.interface_facet_global_ids, self.interface_orientations)
                ),
                "zero_vertices": array_tree_fingerprint(self.zero_vertex_global_ids),
                "topology": signature,
                "event_accepted": self.topology_event_accepted,
                "counters": array_tree_fingerprint(counters),
                "maximum_root_residual": self.maximum_root_residual,
            }
        )

    def validate_source_integrity(self) -> None:
        """Authenticate retained scientific evidence after archive reconstruction."""
        counters = np.asarray(self.counters)
        if (
            counters.ndim != 1
            or counters.size < 5
            or self.inserted_roots != int(counters[2])
            or self.exact_zero_samples != int(counters[3])
            or self.work_units != int(counters[4])
            or self.topology_event_accepted != self.topology_changed
            or not np.isfinite(self.maximum_root_residual)
            or self.maximum_root_residual < 0.0
        ):
            raise ValueError(
                "Level-set evidence has inconsistent retained counters or events."
            )
        expected = canonical_fingerprint(
            {
                "kind": "native-level-set-evidence",
                "request": self.request_id,
                "source": self.source_mesh_id,
                "target": self.target_mesh_id,
                "target_revision": self.target_numeric_version,
                "cells": array_tree_fingerprint((self.cell_global_ids, self.cell_sides)),
                "facets": array_tree_fingerprint(
                    (self.interface_facet_global_ids, self.interface_orientations)
                ),
                "zero_vertices": array_tree_fingerprint(self.zero_vertex_global_ids),
                "topology": self.topology_signature,
                "event_accepted": self.topology_event_accepted,
                "counters": array_tree_fingerprint(counters),
                "maximum_root_residual": self.maximum_root_residual,
            }
        )
        if expected != self.evidence_id:
            raise ValueError("Level-set evidence scientific identity is stale.")

    def require_current(self, mesh: CellMesh, /) -> None:
        """Refuse any other mesh or revision; content-equal simplex meshes share ``mesh_id``."""
        if not isinstance(mesh, CellMesh):
            raise TypeError("mesh must be CellMesh.")
        self.validate_source_integrity()
        if (
            mesh.mesh_id != self.target_mesh_id
            or mesh.numeric_version != self.target_numeric_version
            or not np.array_equal(
                np.asarray(mesh.entity_set(3).entity_ids),
                np.asarray(self.cell_global_ids),
            )
        ):
            raise ValueError("Level-set evidence is stale or belongs to another mesh.")
        sides = np.asarray(self.cell_sides)
        zero_vertices = np.isin(
            np.asarray(mesh.vertex_global_ids), np.asarray(self.zero_vertex_global_ids)
        )
        zero_values = np.where(zero_vertices, 0.0, 1.0)
        facets = np.all(
            zero_vertices[np.asarray(_tetrahedral_connectivity(mesh).faces)], axis=1
        )
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
                "Level-set evidence contradicts its retained target partition."
            )


def _request(prepared: PreparedMeshAdaptation, /) -> LevelSetMeshAdaptation:
    if not isinstance(prepared.request, LevelSetMeshAdaptation):
        raise TypeError("The native level-set route requires LevelSetMeshAdaptation.")
    return prepared.request


def _source_values(prepared: PreparedMeshAdaptation, /) -> np.ndarray:
    request = _request(prepared)
    mesh = prepared.source.mesh
    resolve_mesh_scope(mesh, request.scope)
    ids = np.asarray(request.scope.entity_ids, dtype=np.int64)
    if ids.size != mesh.coordinates.shape[0]:
        raise ValueError("A level set must cover every source vertex.")
    return np.asarray(request.values, dtype=np.float64)[
        np.searchsorted(ids, np.asarray(mesh.vertex_global_ids))
    ]


def validate_level_set_adaptation(prepared: PreparedMeshAdaptation, /) -> None:
    """Admission called by the canonical adaptation preparation boundary."""
    mesh = prepared.source.mesh
    request = _request(prepared)
    if (
        mesh.topological_dimension != 3
        or mesh.ambient_dimension != 3
        or any(block.cell_kind != "tetrahedron" for block in mesh.blocks)
    ):
        raise ValueError("Native level-set insertion requires a tetrahedral volume mesh.")
    if not is_affine_cell_geometry(mesh, prepared.source.geometry):
        raise ValueError("P1 zero insertion requires affine source geometry.")
    values = _source_values(prepared)
    if any(
        np.any(np.all(values[np.asarray(block.vertices)] == 0.0, axis=1))
        for block in mesh.blocks
    ):
        raise ValueError(
            "An identically zero source cell has no exclusive inside/outside assignment."
        )
    previous = request.previous
    if previous is not None:
        previous.require_current(mesh)
    existing = {
        value.name
        for value in (
            *prepared.source.patches,
            *prepared.source.zones,
            *prepared.source.labels,
        )
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
            (request.inside_region_id, request.outside_region_id, request.interface_id)
        )
        - previous_names
    ):
        raise ValueError("Level-set identities collide with existing mesh organization.")
    periodic = mesh.periodic_topology
    if periodic is not None and not np.array_equal(
        values, values[np.asarray(periodic.vertex_representatives)]
    ):
        raise ValueError("Periodic level-set values must agree exactly on vertex orbits.")


def _tetrahedral_connectivity(mesh: CellMesh, /) -> TetrahedralConnectivity:
    """Narrow the admitted volume carrier at each native topology boundary."""
    connectivity = mesh.connectivity
    if not isinstance(connectivity, TetrahedralConnectivity):
        raise ValueError("Native level-set topology requires tetrahedral connectivity.")
    return connectivity


def _native_split(prepared: PreparedMeshAdaptation, /) -> _Split:
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
    crossings = np.count_nonzero(
        (np.min(edge_values, axis=1) < 0.0) & (np.max(edge_values, axis=1) > 0.0)
    )
    vertex_capacity = min(points.shape[0] + crossings, limits.maximum_vertices)
    cell_values = values[cells]
    roots_per_cell = np.count_nonzero(cell_values < 0.0, axis=1) * np.count_nonzero(
        cell_values > 0.0, axis=1
    )
    # A source tetrahedron and its k edge roots admit at most C(4+k, 4)
    # distinct tetrahedra. Uncut cells need just one output slot.
    combinatorial_bound = np.asarray((1, 5, 15, 35, 70), dtype=np.int64)
    cell_capacity = min(
        int(np.sum(combinatorial_bound[roots_per_cell])), limits.maximum_cells
    )
    # Conservative bound includes native slot/constraint tables and staged cavities,
    # not merely Python output arrays. Refuse before allocating the work buffers.
    scratch_bound = 4096 * (
        vertex_capacity + cell_capacity + faces.shape[0] + edges.shape[0]
    )
    if scratch_bound > limits.maximum_scratch_bytes:
        raise ValueError(
            "Native level-set working storage exceeds maximum_scratch_bytes."
        )
    output_points = np.empty((vertex_capacity, 3), dtype=np.float64)
    output_cells = np.empty((cell_capacity, 4), dtype=np.int32)
    parents = np.empty((cell_capacity,), dtype=np.int32)
    sources = np.empty((vertex_capacity, 2), dtype=np.int32)
    weights = np.empty((vertex_capacity, 2), dtype=np.float64)
    counters = np.zeros((6,), dtype=np.int64)
    status = MeshcoreStatus(
        load_meshcore()["phx_mc_level_set_split_3d"](
            points.shape[0],
            points.ctypes.data,
            values.ctypes.data,
            cells.shape[0],
            cells.ctypes.data,
            faces.shape[0],
            faces.ctypes.data,
            edges.shape[0],
            edges.ctypes.data,
            protection.ctypes.data,
            vertex_capacity,
            cell_capacity,
            limits.maximum_work_units,
            limits.maximum_cavity_cells,
            output_points.ctypes.data,
            output_cells.ctypes.data,
            parents.ctypes.data,
            sources.ctypes.data,
            weights.ctypes.data,
            counters.ctypes.data,
        )
    )
    if status is not MeshcoreStatus.OK:
        raise MeshcoreError(
            status,
            f"Native zero insertion refused source edge {counters[5]}; work={counters[4]}.",
        )
    nv, nc = int(counters[0]), int(counters[1])
    return _Split(
        output_points[:nv],
        output_cells[:nc],
        parents[:nc],
        sources[:nv],
        weights[:nv],
        counters,
    )


def _edit(prepared: PreparedMeshAdaptation, split: _Split, /) -> CellTopologyEdit:
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
    witnesses = NestedReferenceWitnesses(
        target_ids,
        source_ids[parents],
        nested_reference_vertices(
            sources[target_cells],
            split.weights[target_cells],
            old_vertices[source_cells[parents]],
        ),
    )
    periodic_orbits = None
    if mesh.periodic_topology is not None:
        from ._periodic import periodic_vertex_orbit_witness

        carrier = CellMesh(
            split.coordinates,
            tuple(
                CellBlock(
                    block.name, block.cell_kind, block.cells, global_ids=block.cell_ids
                )
                for block in blocks
            ),
            vertex_global_ids=vertex_ids,
        )
        representative_ids, exponents = _periodic_target_orbits(mesh, vertex_ids, sources)
        periodic_orbits = periodic_vertex_orbit_witness(
            mesh, carrier, representative_ids, exponents, None
        )
    return CellTopologyEdit(
        "nested_refinement",
        split.coordinates,
        vertex_ids,
        tuple(blocks),
        sources,
        split.weights,
        np.ones(sources.shape, dtype=np.bool_),
        tuple(relations),
        refinement=witnesses,
        periodic_orbits=periodic_orbits,
    )


def _support_relations(
    mesh: CellMesh,
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
    from itertools import combinations

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
    source: CellMesh, vertex_ids: np.ndarray, sources: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Global representative IDs and group exponents of every zero-insertion target row.

    Old rows keep their scientific quotient membership. A born vertex lies on a
    split source edge; all born vertices of one quotient-edge orbit form one new
    vertex orbit whose canonical representative is its smallest (first issued)
    global ID.
    """
    periodic = source.periodic_topology
    if periodic is None:
        raise ValueError("Periodic zero-insertion orbits require a periodic source.")
    old_ids = np.asarray(source.vertex_global_ids, dtype=np.int64)
    count = old_ids.size
    old_shifts = np.asarray(periodic.vertex_shifts, dtype=np.int64)
    if old_shifts.ndim != 2 or old_shifts.shape[0] != count:
        raise ValueError("Periodic vertex shifts must have one exponent row per vertex.")
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
    prepared: PreparedMeshAdaptation, mesh: CellMesh, edit: CellTopologyEdit, /
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
            "Native zero insertion did not produce an exclusive side partition."
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
        raise ValueError("Inside/outside adjacency is not supported by a zero facet.")
    return values, sides, facets


def _interface_orientations(
    mesh: CellMesh, sides: np.ndarray, facets: np.ndarray, /
) -> np.ndarray:
    """Inside-to-outside signs; zero marks tangential, nonseparating zero faces."""
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
        raise ValueError("A zero interface has degenerate incident geometry.")
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
    mesh: CellMesh, values: np.ndarray, sides: np.ndarray, facets: np.ndarray, /
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
    mesh: CellMesh, sides: np.ndarray, facets: np.ndarray, /
) -> tuple[int, ...]:
    """Distinct sides every zero facet bounds with exactly one incident cell each.

    Tangential or mixed facets have no exact zone adjacency, so none is declared.
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
    mesh: CellMesh,
    patches: tuple[MeshPatch, ...],
    zones: tuple[MeshZone, ...],
    labels: tuple[MeshLabel, ...],
    edit: CellTopologyEdit,
    /,
) -> tuple[tuple[MeshPatch, ...], tuple[MeshZone, ...], tuple[MeshLabel, ...]]:
    """Canonical organization hook run before target certification."""
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
            "The moving zero set changes topology; explicit accept_topology_change is required."
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


def execute_level_set_route(prepared: PreparedMeshAdaptation, /) -> _RouteOutcome:
    """Route executor; public execution remains execute_mesh_adaptation."""
    from ._adaptation import _finalize_native, _RouteOutcome, MeshAdaptationStatus

    validate_level_set_adaptation(prepared)
    split = _native_split(prepared)
    edit = _edit(prepared, split)
    native = _finalize_native(
        prepared, edit, MeshTransitionKind.REFINE, conservative=True
    )
    evidence = LevelSetEvidence(prepared, native.target.mesh, edit, split.counters)
    limits = prepared.policy.limits
    if (
        native.target.mesh.entity_set(1).count > limits.maximum_edges
        or native.target.mesh.entity_set(2).count > limits.maximum_faces
    ):
        raise ValueError("The level-set partition exceeds edge/face limits.")
    data_bytes = native.target.mesh.coordinates.size * 8 + sum(
        block.vertices.size * 8 for block in native.target.mesh.blocks
    )
    if data_bytes > limits.maximum_data_bytes:
        raise ValueError("The level-set partition exceeds maximum_data_bytes.")
    return _RouteOutcome(
        MeshAdaptationStatus.COMPLETE,
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
