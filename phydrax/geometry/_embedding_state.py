#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Shared embedding state, findings, limits, and host facet preparation."""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, fields
from fractions import Fraction
from typing import assert_never, final, Literal, Protocol, TYPE_CHECKING, TypeAlias

import equinox as eqx
import numpy as np

from .._bvh import bvh_overlap_pair_blocks, prepare_bvh
from .._fingerprint import canonical_fingerprint
from .._meshcore import MeshcoreError, MeshcoreStatus, NativeExecutionLimits
from .._resource_ticket import HostBatchTicket
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import canonical_identifier, nonnegative_integer, positive_integer
from ..discretization._cell_complex import (
    PolygonalConnectivity,
    PolyhedralConnectivity,
    TetrahedralConnectivity,
)
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_mesh import CellMesh
from ..discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResource,
    CoordinateEnclosureResourceError,
)
from ..discretization._coordinate_enclosure import CoordinateSourceBank
from ..discretization._coordinate_identity import coordinate_scope_key
from ..discretization._hexahedral import HexahedralConnectivity
from ..discretization._reference_cell import reference_cell_topology
from ..typing import checked, parse
from ._mesh_check_names import MeshCertificateCheck


if TYPE_CHECKING:
    from ..discretization._cell_geometry_predecessor import CellGeometryPredecessor
    from ._global_embedding import MappedBoundaryDegreeEvidence
    from ._periodic_mapped_restriction import PeriodicMappedRestrictionEvidence

MeshFindingStatus: TypeAlias = Literal["violated", "unresolved"]


MeshCertificateEntityKind: TypeAlias = Literal[
    "mesh",
    "cell",
    "facet",
    "source_cell",
    "source_facet",
    "source_region",
    "source_sample",
]


MeshCertificateResource: TypeAlias = (
    CoordinateEnclosureResource | Literal["distance_evaluations"]
)


type CoordinateMapScope = Literal["affine", "mapped"]


@final
class CoordinateScope(StrictModule, NonTrainableState):
    """Return the coordinate-map scope, mapped cell rows, and prepared source bank.

    The cell rows are one host int64 vector. Its emptiness determines the scope.
    Source preparation establishes the bank's scalar coefficient structure; this record
    retains the array and bank without conversion or copying.
    """

    coordinate_scope: CoordinateMapScope = eqx.field(static=True)
    mapped_cells: np.ndarray
    source_values: CoordinateSourceBank

    def __init__(
        self,
        coordinate_scope: CoordinateMapScope,
        mapped_cells: np.ndarray,
        source_values: CoordinateSourceBank,
        /,
    ) -> None:
        scope = parse(
            coordinate_scope, CoordinateMapScope, "CoordinateScope.coordinate_scope"
        )
        if mapped_cells.ndim != 1 or mapped_cells.dtype != np.dtype(np.int64):
            raise ValueError(
                "CoordinateScope.mapped_cells must be a host int64 vector; "
                f"got shape={mapped_cells.shape}, dtype={mapped_cells.dtype}"
            )
        if (scope == "mapped") != bool(mapped_cells.size):
            raise ValueError(
                "CoordinateScope.coordinate_scope must agree with mapped_cells emptiness; "
                f"got scope={scope}, cells={mapped_cells.size}"
            )
        self.coordinate_scope = scope
        self.mapped_cells = mapped_cells
        self.source_values = source_values


@final
class MeshCertificateLimits(StrictModule, NonTrainableState):
    """Work budgets of the mesh certificates; exhaustion is reported UNRESOLVED."""

    maximum_candidate_pairs: int = eqx.field(static=True)
    maximum_ray_tests: int = eqx.field(static=True)
    maximum_source_samples: int = eqx.field(static=True)
    maximum_distance_evaluations: int = eqx.field(static=True)
    maximum_subdivision_depth: int = eqx.field(static=True)
    maximum_subdivision_pieces: int = eqx.field(static=True)
    maximum_bernstein_nodes: int = eqx.field(static=True)
    maximum_periodic_images: int = eqx.field(static=True)
    maximum_work_units: int = eqx.field(static=True)
    maximum_scratch_bytes: int = eqx.field(static=True)
    limits_id: str = eqx.field(static=True, init=False)

    def __init__(
        self,
        *,
        maximum_candidate_pairs: int = 5_000_000,
        maximum_ray_tests: int = 50_000_000,
        maximum_source_samples: int = 2_000_000,
        maximum_distance_evaluations: int = 200_000_000,
        maximum_subdivision_depth: int = 12,
        maximum_subdivision_pieces: int = 1_000_000,
        maximum_bernstein_nodes: int = 4096,
        maximum_periodic_images: int = 100_000,
        maximum_work_units: int = 200_000_000,
        maximum_scratch_bytes: int = 512 * 1024**2,
    ) -> None:
        values = {
            name: positive_integer(value, name)
            for name, value in (
                ("maximum_candidate_pairs", maximum_candidate_pairs),
                ("maximum_ray_tests", maximum_ray_tests),
                ("maximum_source_samples", maximum_source_samples),
                ("maximum_distance_evaluations", maximum_distance_evaluations),
                ("maximum_subdivision_pieces", maximum_subdivision_pieces),
                ("maximum_bernstein_nodes", maximum_bernstein_nodes),
                ("maximum_periodic_images", maximum_periodic_images),
                ("maximum_work_units", maximum_work_units),
                ("maximum_scratch_bytes", maximum_scratch_bytes),
            )
        }
        values["maximum_subdivision_depth"] = nonnegative_integer(
            maximum_subdivision_depth, "MeshCertificateLimits.maximum_subdivision_depth"
        )
        self.maximum_candidate_pairs = values["maximum_candidate_pairs"]
        self.maximum_ray_tests = values["maximum_ray_tests"]
        self.maximum_source_samples = values["maximum_source_samples"]
        self.maximum_distance_evaluations = values["maximum_distance_evaluations"]
        self.maximum_subdivision_depth = values["maximum_subdivision_depth"]
        self.maximum_subdivision_pieces = values["maximum_subdivision_pieces"]
        self.maximum_bernstein_nodes = values["maximum_bernstein_nodes"]
        self.maximum_periodic_images = values["maximum_periodic_images"]
        self.maximum_work_units = values["maximum_work_units"]
        self.maximum_scratch_bytes = values["maximum_scratch_bytes"]
        self.limits_id = canonical_fingerprint(
            {"kind": "mesh-certificate-limits", **values}
        )

    def native_limits(self) -> NativeExecutionLimits:
        """Lower certificate work, query, and storage limits to native allowances."""
        return NativeExecutionLimits(
            self.maximum_work_units,
            self.maximum_distance_evaluations,
            0,
            self.maximum_scratch_bytes,
            math.inf,
        )


@contextmanager
@checked
def coordinate_certificate_request(
    limits: MeshCertificateLimits,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> Iterator[CoordinateEnclosureBudget]:
    """Bind one certificate request to an explicit rational-arithmetic view."""
    if budget is not None:
        with (
            budget.measurement(),
            budget.activate(),
            budget.bound_stage(limits.maximum_work_units, limits.maximum_scratch_bytes),
        ):
            yield budget
        return
    ticket = HostBatchTicket(
        limits.maximum_work_units,
        queries_upper=limits.maximum_distance_evaluations,
        bytes_upper=limits.maximum_scratch_bytes,
    )
    view = CoordinateEnclosureBudget(
        ticket,
        maximum_work_units=limits.maximum_work_units,
        maximum_memory_bytes=limits.maximum_scratch_bytes,
    )
    with ticket, view.measurement(), view.activate():
        yield view


@final
class NativeCertificateRefusal(StrictModule, NonTrainableState):
    """Retain the native status, batch bounds, and root refusal evidence."""

    status: MeshcoreStatus = eqx.field(static=True)
    admitted_upper: tuple[int, int, int, int] | None = eqx.field(static=True)
    requested_upper: tuple[int, int, int, int] | None = eqx.field(static=True)
    root_receipt: tuple[int, int, int, int, int, int, int, int] | None = eqx.field(
        static=True
    )
    refusal_id: str = eqx.field(static=True)

    @checked
    def __init__(self, error: MeshcoreError, /) -> None:
        if error.status not in (MeshcoreStatus.CAPACITY_EXCEEDED, MeshcoreStatus.TIMEOUT):
            raise ValueError(
                "NativeCertificateRefusal.status must indicate resource exhaustion; "
                f"got {error.status.name}."
            )
        receipt = error.batch_refusal_evidence
        if receipt is not None and receipt.shape != (8,):
            raise ValueError(
                "NativeCertificateRefusal.root_receipt must contain eight counters; "
                f"got shape={receipt.shape}."
            )
        root_receipt = (
            None
            if receipt is None
            else (
                int(receipt[0]),
                int(receipt[1]),
                int(receipt[2]),
                int(receipt[3]),
                int(receipt[4]),
                int(receipt[5]),
                int(receipt[6]),
                int(receipt[7]),
            )
        )
        self.status = error.status
        self.admitted_upper = error.batch_admitted_upper
        self.requested_upper = error.batch_requested_upper
        self.root_receipt = root_receipt
        self.refusal_id = canonical_fingerprint(
            {
                "kind": "native-certificate-refusal",
                "status": int(error.status),
                "admitted_upper": error.batch_admitted_upper,
                "requested_upper": error.batch_requested_upper,
                "root_receipt": root_receipt,
            }
        )


@final
class MeshCertificateFinding(StrictModule, NonTrainableState):
    """One violated or unresolved check with the mesh or source entities it concerns.

    ``entity_ids`` are mesh global ids for ``cell``/``facet`` entities and
    declared-source row indices for ``source_*`` entities.

    Coordinate and distance-query resource refusals retain integer ``limit``,
    ``requested``, and ``completed`` counts plus the resource kind.
    Achieved expression quantities come from the measured coordinate phase.
    ``*_at_catch`` storage quantities describe the ledger after scope unwinding;
    its peak remains the measured storage upper bound, not process RSS.
    Native refusals retain status, available admitted/requested batch bounds,
    and the root counter evidence in ``native_refusal``. The certificate stays
    unresolved, and report consumers classify its blocking resource finding
    as resource exhaustion.
    """

    check: MeshCertificateCheck = eqx.field(static=True)
    status: MeshFindingStatus = eqx.field(static=True)
    entity_kind: MeshCertificateEntityKind = eqx.field(static=True)
    entity_ids: tuple[int, ...] = eqx.field(static=True)
    resource: MeshCertificateResource | None = eqx.field(static=True)
    requested: tuple[tuple[str, int], ...] = eqx.field(static=True)
    achieved: tuple[tuple[str, int], ...] = eqx.field(static=True)
    observations: tuple[tuple[str, int], ...] = eqx.field(static=True)
    native_refusal: NativeCertificateRefusal | None
    finding_id: str = eqx.field(static=True)

    def __init__(
        self,
        check: str,
        status: MeshFindingStatus,
        entity_kind: MeshCertificateEntityKind,
        entity_ids: tuple[int, ...] = (),
        /,
        *,
        resource: MeshCertificateResource | None = None,
        requested: tuple[tuple[str, int], ...] = (),
        achieved: tuple[tuple[str, int], ...] = (),
        observations: tuple[tuple[str, int], ...] = (),
        native_refusal: NativeCertificateRefusal | None = None,
    ) -> None:
        name = parse(check, MeshCertificateCheck, "MeshCertificateFinding.check")
        status_ = parse(status, MeshFindingStatus, "MeshCertificateFinding.status")
        kind = parse(
            entity_kind, MeshCertificateEntityKind, "MeshCertificateFinding.entity_kind"
        )
        identifiers = tuple(sorted({int(value) for value in entity_ids}))

        def quantities(
            values: tuple[tuple[str, int], ...], label: str, /
        ) -> tuple[tuple[str, int], ...]:
            field = f"MeshCertificateFinding.{label}"
            if not isinstance(values, tuple):
                raise TypeError(
                    f"{field} must be an immutable quantity tuple; got {type(values).__name__}"
                )
            rows: list[tuple[str, int]] = []
            for item in values:
                if not isinstance(item, tuple) or len(item) != 2:
                    raise TypeError(
                        f"{field} must contain named integer pairs; got an invalid quantity row"
                    )
                key = canonical_identifier(item[0], f"{label} quantity")
                rows.append((key, nonnegative_integer(item[1], f"{label}:{key}")))
            if len({key for key, _ in rows}) != len(rows):
                raise ValueError(
                    f"{field} must have unique quantity keys; got repeated keys"
                )
            return tuple(sorted(rows))

        resource_ = (
            None
            if resource is None
            else parse(
                resource, MeshCertificateResource, "MeshCertificateFinding.resource"
            )
        )
        requested_ = quantities(requested, "requested")
        achieved_ = quantities(achieved, "achieved")
        observations_ = quantities(observations, "observations")
        if resource_ is None:
            if requested_ or achieved_:
                raise ValueError(
                    "MeshCertificateFinding.resource must identify requested and achieved quantities; got None"
                )
        else:
            wanted, completed = dict(requested_), dict(achieved_)
            if set(wanted) != {"limit", "requested"} or "completed" not in completed:
                raise ValueError(
                    "MeshCertificateFinding.quantities must include limit, requested, and completed counts; got missing quantities"
                )
            if (
                wanted["requested"] <= wanted["limit"]
                or completed["completed"] > wanted["limit"]
            ):
                raise ValueError(
                    "MeshCertificateFinding.quantities must describe requested above limit and completed within limit; got inconsistent counts"
                )
        self.check = name
        self.status = status_
        self.entity_kind = kind
        self.entity_ids = identifiers
        self.resource = resource_
        self.requested = requested_
        self.achieved = achieved_
        self.observations = observations_
        self.native_refusal = native_refusal
        self.finding_id = canonical_fingerprint(
            {
                "kind": "mesh-certificate-finding",
                "check": name,
                "status": status_,
                "entity_kind": kind,
                "entity_ids": identifiers,
                "observations": observations_,
                **(
                    {}
                    if native_refusal is None
                    else {"native_refusal": native_refusal.refusal_id}
                ),
                **(
                    {}
                    if resource_ is None
                    else {
                        "resource": resource_,
                        "requested": requested_,
                        "achieved": achieved_,
                    }
                ),
            }
        )


@dataclass(frozen=True)
class Facets:
    """Oriented facet occurrences with pairing stored as packed occurrence ranges."""

    rows: np.ndarray
    signs: np.ndarray
    cells: np.ndarray
    group: np.ndarray
    counts: np.ndarray
    order: np.ndarray
    group_offsets: np.ndarray

    @property
    def groups(self) -> Iterator[np.ndarray]:
        """Yield read-only occurrence views in canonical facet-group order."""
        for index in range(self.group_offsets.size - 1):
            group = self.order[
                self.group_offsets[index] : self.group_offsets[index + 1]
            ]
            group.flags.writeable = False
            yield group

    @property
    def boundary(self) -> np.ndarray:
        return self.counts[self.group] == 1


def csr_rows(offsets: np.ndarray, values: np.ndarray, /) -> np.ndarray:
    sizes = np.diff(offsets)
    rows = np.repeat(np.arange(sizes.size), sizes)
    rank = np.arange(values.size) - offsets[rows]
    result = np.full((sizes.size, int(np.max(sizes))), -1, dtype=np.int64)
    result[rows, rank] = values
    return result


def padded_facet_rows(blocks: list[np.ndarray], /) -> np.ndarray:
    width = max(value.shape[1] for value in blocks)
    return np.concatenate(
        [
            np.pad(value, ((0, 0), (0, width - value.shape[1])), constant_values=-1)
            for value in blocks
        ]
    )


def polygon_edges(rows: np.ndarray, valid: np.ndarray, /) -> tuple:
    """Directed edges of counterclockwise loops with valid-prefix padding."""

    lengths = np.sum(valid, axis=1)
    column = np.arange(rows.shape[1])
    following = rows[np.arange(rows.shape[0])[:, None], (column + 1) % lengths[:, None]]
    slots = column[None, :] < lengths[:, None]
    owners = np.broadcast_to(np.arange(rows.shape[0])[:, None], rows.shape)
    return np.stack((rows[slots], following[slots]), axis=1), owners[slots]


def polyhedral_facets(
    connectivity: PolyhedralConnectivity, owners: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    faces = np.asarray(connectivity.cell_face_values, dtype=np.int64)
    offsets = np.asarray(connectivity.cell_face_offsets, dtype=np.int64)
    signs = np.asarray(connectivity.cell_face_sign_values, dtype=np.float64)
    incidence = np.repeat(np.arange(offsets.size - 1), np.diff(offsets))
    chosen = np.isin(incidence, owners)
    loops = csr_rows(
        np.asarray(connectivity.face_vertex_offsets, dtype=np.int64),
        np.asarray(connectivity.face_vertex_values, dtype=np.int64),
    )[faces[chosen]]
    lengths = np.sum(loops >= 0, axis=1)
    reverse = signs[chosen] < 0.0
    for length in np.unique(lengths[reverse]):
        rows = reverse & (lengths == length)
        loops[rows, :length] = loops[rows, :length][:, ::-1]
    return loops, incidence[chosen]


def _polyhedral_connectivity(mesh: CellMesh, /) -> PolyhedralConnectivity:
    """Return the connectivity required by a declared polyhedral block."""
    match mesh.connectivity:
        case PolyhedralConnectivity() as connectivity:
            return connectivity
        case value:
            raise TypeError(
                "mesh_facets.mesh.connectivity must be polyhedral for a polyhedral block; "
                f"got {type(value).__name__}"
            )


@checked
def mesh_facets(
    mesh: CellMesh,
    /,
    *,
    geometry: CellGeometrySpec | None = None,
    budget: CoordinateEnclosureBudget,
) -> Facets:
    dimension = mesh.topological_dimension

    ledger = budget
    source_key = None
    if geometry is not None:
        ledger.reserve(1)
        source_key = coordinate_scope_key(mesh, geometry)
        cached = ledger.facet_cache.get(source_key)
        if cached is not None:
            return cached
        if any(block.cell_kind == "polyhedron" for block in mesh.blocks):
            connectivity = _polyhedral_connectivity(mesh)
            occurrences = connectivity.cell_face_values.size
        else:
            occurrences = sum(
                block.vertices.size
                if dimension <= 2
                else block.cell_count
                * len(reference_cell_topology(block.cell_kind).entities[2])
                for block in mesh.blocks
            )
        ledger.reserve(0, 1024 + 256 * occurrences)
    rows = []
    signs = []
    cells = []
    cursor = 0
    for block in mesh.blocks:
        count = block.cell_count
        owners = np.arange(cursor, cursor + count)
        vertices = np.asarray(block.vertices, dtype=np.int64)
        if block.cell_kind == "polyhedron":
            connectivity = _polyhedral_connectivity(mesh)
            loops, incidence = polyhedral_facets(connectivity, owners)
            rows.append(loops)
            cells.append(incidence)
            signs.append(np.zeros((incidence.size,), dtype=np.int8))
        elif dimension == 1:
            rows.append(vertices.reshape(-1, 1))
            cells.append(np.repeat(owners, 2))
            signs.append(np.tile(np.asarray((-1, 1), dtype=np.int8), count))
        elif dimension == 2:
            edges, local = polygon_edges(
                vertices, np.asarray(block.vertex_valid, dtype=np.bool_)
            )
            rows.append(edges)
            cells.append(owners[local])
            signs.append(np.zeros((local.size,), dtype=np.int8))
        else:
            local_facets = reference_cell_topology(block.cell_kind).entities[2]
            for face in local_facets:
                values = np.full((count, 4), -1, dtype=np.int64)
                values[:, : len(face)] = vertices[:, np.asarray(face)]
                rows.append(values)
                cells.append(owners)
                signs.append(np.zeros((count,), dtype=np.int8))
        cursor += count
    facet_rows = padded_facet_rows(rows)
    keys = np.sort(np.where(facet_rows < 0, np.iinfo(np.int64).max, facet_rows), axis=1)
    _, group, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    incidence = group.reshape(-1)
    if source_key is not None:
        ledger.reserve(incidence.size, 256 + 32 * incidence.size)
    order = np.argsort(incidence, kind="stable")
    group_offsets = np.empty((counts.size + 1,), dtype=np.int64)
    group_offsets[0] = 0
    group_offsets[1:-1] = np.flatnonzero(np.diff(incidence[order])) + 1
    group_offsets[-1] = order.size
    prepared = Facets(
        facet_rows,
        np.concatenate(signs),
        np.concatenate(cells),
        incidence,
        counts,
        order,
        group_offsets,
    )
    if source_key is not None:
        arrays = (
            prepared.rows,
            prepared.signs,
            prepared.cells,
            prepared.group,
            prepared.counts,
            prepared.order,
            prepared.group_offsets,
        )
        for array in arrays:
            array.flags.writeable = False
        ledger.retain_basis((source_key, prepared, arrays))
        ledger.facet_cache[source_key] = prepared
    return prepared


def sorted_facet_keys(rows: np.ndarray, width: int, /) -> np.ndarray:
    padded = np.full((rows.shape[0], width), -1, dtype=np.int64)
    padded[:, : rows.shape[1]] = rows
    return np.sort(np.where(padded < 0, np.iinfo(np.int64).max, padded), axis=1)


@checked
def facet_entity_ids(mesh: CellMesh, rows: np.ndarray, /) -> np.ndarray:
    """Mesh facet entity global ids of facet vertex rows."""

    dimension = mesh.topological_dimension
    if dimension == 1:
        return np.asarray(mesh.vertex_global_ids, dtype=np.int64)[rows[:, 0]]
    connectivity = mesh.connectivity
    match connectivity:
        case PolyhedralConnectivity():
            table = csr_rows(
                np.asarray(connectivity.face_vertex_offsets, dtype=np.int64),
                np.asarray(connectivity.face_vertex_values, dtype=np.int64),
            )
        case PolygonalConnectivity():
            table = np.asarray(connectivity.edges, dtype=np.int64)
        case TetrahedralConnectivity() | HexahedralConnectivity():
            table = np.asarray(connectivity.faces, dtype=np.int64)
        case _:
            raise TypeError(
                "facet_entity_ids.mesh.connectivity must provide facet topology; "
                f"got {type(connectivity).__name__}"
            )
    width = max(table.shape[1], rows.shape[1])
    known = sorted_facet_keys(table, width)
    query = sorted_facet_keys(rows, width)
    view = np.dtype((np.void, known.dtype.itemsize * width))
    known_view = np.ascontiguousarray(known).view(view).reshape(-1)
    query_view = np.ascontiguousarray(query).view(view).reshape(-1)
    order = np.argsort(known_view, kind="stable")
    position = order[np.searchsorted(known_view[order], query_view)]
    if not np.array_equal(known[position], query):
        raise RuntimeError(
            "facet_entity_ids.rows must identify facets in the mesh topology; got unmatched rows"
        )
    return np.asarray(mesh.entity_set(dimension - 1).entity_ids, dtype=np.int64)[position]


def candidate_pairs(
    lower: np.ndarray,
    upper: np.ndarray,
    capacity: int,
    /,
    *,
    other: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Closed-box candidate pairs (``first < second`` for self queries)."""

    empty = np.empty((0,), dtype=np.int64)
    if lower.shape[0] == 0 or (other is not None and other[0].shape[0] == 0):
        return empty, empty, False
    first_bvh = prepare_bvh(lower, upper, dtype=np.float64)
    second_bvh = first_bvh if other is None else prepare_bvh(*other, dtype=np.float64)
    firsts = [empty]
    seconds = [empty]
    remaining = capacity
    for first, second in bvh_overlap_pair_blocks(
        first_bvh, second_bvh, include_touching=True
    ):
        if other is None:
            keep = first < second
            first = first[keep]
            second = second[keep]
        if first.size > remaining:
            return np.concatenate(firsts), np.concatenate(seconds), True
        remaining -= first.size
        firsts.append(first)
        seconds.append(second)
    return np.concatenate(firsts), np.concatenate(seconds), False


def boxes(points: np.ndarray, rows: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
    corners = points[rows]
    if points.dtype == object:
        lower, upper = (
            np.asarray(np.min(corners, axis=1), dtype=np.float64),
            np.asarray(np.max(corners, axis=1), dtype=np.float64),
        )
        return np.nextafter(lower, -np.inf), np.nextafter(upper, np.inf)
    return np.min(corners, axis=1), np.max(corners, axis=1)


@final
class PeriodicAffineQuotientEvidence(StrictModule, NonTrainableState):
    """Bind full-source affinity, positive validity, and closed quotient traces."""

    mesh_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    periodic_topology_id: str = eqx.field(static=True)
    validity_certificate_id: str = eqx.field(static=True)
    affinity_certificate_id: str = eqx.field(static=True)
    facet_trace_certificate_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh_id: str,
        geometry_id: str,
        periodic_topology_id: str,
        validity_certificate_id: str,
        affinity_certificate_id: str,
        facet_trace_certificate_id: str,
    ) -> None:
        values = tuple(
            canonical_identifier(value, f"PeriodicAffineQuotientEvidence.{name}")
            for name, value in (
                ("mesh_id", mesh_id),
                ("geometry_id", geometry_id),
                ("periodic_topology_id", periodic_topology_id),
                ("validity_certificate_id", validity_certificate_id),
                ("affinity_certificate_id", affinity_certificate_id),
                ("facet_trace_certificate_id", facet_trace_certificate_id),
            )
        )
        (
            self.mesh_id,
            self.geometry_id,
            self.periodic_topology_id,
            self.validity_certificate_id,
            self.affinity_certificate_id,
            self.facet_trace_certificate_id,
        ) = values
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "periodic-affine-quotient-evidence",
                "mesh": self.mesh_id,
                "geometry": self.geometry_id,
                "periodic_topology": self.periodic_topology_id,
                "validity": self.validity_certificate_id,
                "affinity": self.affinity_certificate_id,
                "facet_traces": self.facet_trace_certificate_id,
            }
        )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[PeriodicAffineQuotientEvidence],
        value: PeriodicAffineQuotientEvidence,
        /,
    ) -> PeriodicAffineQuotientEvidence:
        """Regenerate the binding hash and reject stale restored identities."""
        rebuilt = cls(
            value.mesh_id,
            value.geometry_id,
            value.periodic_topology_id,
            value.validity_certificate_id,
            value.affinity_certificate_id,
            value.facet_trace_certificate_id,
        )
        if rebuilt.evidence_id != value.evidence_id:
            raise ValueError(
                "PeriodicAffineQuotientEvidence.evidence_id must bind every restored field; "
                f"got {value.evidence_id!r}"
            )
        return rebuilt


@final
class PeriodicAffineTorusDegreeEvidence(StrictModule, NonTrainableState):
    """Bind current affine-torus premises and compute degree from exact areas."""

    mesh_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    periodic_topology_id: str = eqx.field(static=True)
    input_binding_id: str = eqx.field(static=True)
    affinity_certificate_id: str = eqx.field(static=True)
    manifold_certificate_id: str = eqx.field(static=True)
    trace_certificate_id: str = eqx.field(static=True)
    orientation_certificate_id: str = eqx.field(static=True)
    pullback_area: tuple[int, int] = eqx.field(static=True)
    lattice_area: tuple[int, int] = eqx.field(static=True)
    covering_degree: int = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh_id: str,
        geometry_id: str,
        periodic_topology_id: str,
        input_binding_id: str,
        affinity_certificate_id: str,
        manifold_certificate_id: str,
        trace_certificate_id: str,
        orientation_certificate_id: str,
        pullback_area: tuple[int, int],
        lattice_area: tuple[int, int],
    ) -> None:
        names = (
            "mesh_id", "geometry_id", "periodic_topology_id", "input_binding_id",
            "affinity_certificate_id", "manifold_certificate_id",
            "trace_certificate_id", "orientation_certificate_id",
        )
        values = tuple(
            canonical_identifier(value, f"PeriodicAffineTorusDegreeEvidence.{name}")
            for name, value in zip(
                names,
                (mesh_id, geometry_id, periodic_topology_id, input_binding_id,
                 affinity_certificate_id, manifold_certificate_id,
                 trace_certificate_id, orientation_certificate_id),
                strict=True,
            )
        )
        areas = tuple(
            Fraction(
                positive_integer(value[0], f"PeriodicAffineTorusDegreeEvidence.{name}.numerator"),
                positive_integer(value[1], f"PeriodicAffineTorusDegreeEvidence.{name}.denominator"),
            )
            for name, value in (("pullback_area", pullback_area), ("lattice_area", lattice_area))
        )
        degree = areas[0] / areas[1]
        if degree.denominator != 1 or degree.numerator != 1:
            raise ValueError(
                "PeriodicAffineTorusDegreeEvidence.exact_integral must compute integer degree one; "
                f"got {degree}"
            )
        (
            self.mesh_id, self.geometry_id, self.periodic_topology_id,
            self.input_binding_id, self.affinity_certificate_id,
            self.manifold_certificate_id, self.trace_certificate_id,
            self.orientation_certificate_id,
        ) = values
        self.pullback_area = (areas[0].numerator, areas[0].denominator)
        self.lattice_area = (areas[1].numerator, areas[1].denominator)
        self.covering_degree = degree.numerator
        self.evidence_id = canonical_fingerprint({
            "kind": "periodic-affine-torus-degree-one",
            **dict(zip(names, values, strict=True)),
            "pullback_area": self.pullback_area,
            "lattice_area": self.lattice_area,
            "covering_degree": self.covering_degree,
        })

    @classmethod
    @checked
    def from_stored_fields(
        cls, value: PeriodicAffineTorusDegreeEvidence, /
    ) -> PeriodicAffineTorusDegreeEvidence:
        rebuilt = cls(
            value.mesh_id, value.geometry_id, value.periodic_topology_id,
            value.input_binding_id, value.affinity_certificate_id,
            value.manifold_certificate_id, value.trace_certificate_id,
            value.orientation_certificate_id, value.pullback_area, value.lattice_area,
        )
        if any(getattr(rebuilt, field.name) != getattr(value, field.name) for field in fields(value)):
            raise ValueError(
                "PeriodicAffineTorusDegreeEvidence.fields must bind the current exact degree and every premise; got changed stored evidence"
            )
        return rebuilt


def _exact_fraction(value: tuple[int, int], name: str, /) -> Fraction:
    numerator, denominator = value
    if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
        raise ValueError(f"{name} must be an exact integer ratio; got {value!r}")
    result = Fraction(numerator, denominator)
    if (result.numerator, result.denominator) != (numerator, denominator):
        raise ValueError(f"{name} must be a reduced integer ratio; got {value!r}")
    return result


type PeriodicPartitionDirection = Literal["refine", "coarsen"]


@final
class PeriodicRestrictionPartitionEvidence(StrictModule, NonTrainableState):
    """Bind the exact parent partition that transfers a source quotient embedding.

    ``direction`` identifies whether the current target consists of the children
    (refinement) or parents (coarsening). Each child names one partition parent,
    its closed sign cell and exact volume. Every parent's volume equals its
    children's sum; all source-certificate bindings remain in source orientation.
    """

    mesh_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    periodic_topology_id: str = eqx.field(static=True)
    validity_certificate_id: str = eqx.field(static=True)
    facet_trace_certificate_id: str = eqx.field(static=True)
    boundary_subdivision_certificate_id: str = eqx.field(static=True)
    identification_id: str = eqx.field(static=True)
    source_mesh_id: str = eqx.field(static=True)
    source_geometry_id: str = eqx.field(static=True)
    source_validity_certificate_id: str = eqx.field(static=True)
    source_embedding_certificate_id: str = eqx.field(static=True)
    planes: tuple[tuple[tuple[int, int], ...], ...] = eqx.field(static=True)
    child_cell_ids: tuple[int, ...] = eqx.field(static=True)
    parent_cell_ids: tuple[int, ...] = eqx.field(static=True)
    sign_vectors: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    child_volumes: tuple[tuple[int, int], ...] = eqx.field(static=True)
    partition_cell_ids: tuple[int, ...] = eqx.field(static=True)
    partition_volumes: tuple[tuple[int, int], ...] = eqx.field(static=True)
    direction: PeriodicPartitionDirection = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh_id: str,
        geometry_id: str,
        periodic_topology_id: str,
        validity_certificate_id: str,
        facet_trace_certificate_id: str,
        boundary_subdivision_certificate_id: str,
        identification_id: str,
        source_mesh_id: str,
        source_geometry_id: str,
        source_validity_certificate_id: str,
        source_embedding_certificate_id: str,
        planes: tuple[tuple[tuple[int, int], ...], ...],
        child_cell_ids: tuple[int, ...],
        parent_cell_ids: tuple[int, ...],
        sign_vectors: tuple[tuple[int, ...], ...],
        child_volumes: tuple[tuple[int, int], ...],
        partition_cell_ids: tuple[int, ...],
        partition_volumes: tuple[tuple[int, int], ...],
        *,
        direction: PeriodicPartitionDirection,
    ) -> None:
        names = (
            "mesh_id", "geometry_id", "periodic_topology_id",
            "validity_certificate_id", "facet_trace_certificate_id",
            "boundary_subdivision_certificate_id", "identification_id", "source_mesh_id",
            "source_geometry_id", "source_validity_certificate_id",
            "source_embedding_certificate_id",
        )
        identifiers = tuple(
            canonical_identifier(value, f"PeriodicRestrictionPartitionEvidence.{name}")
            for name, value in zip(
                names,
                (
                    mesh_id, geometry_id, periodic_topology_id,
                    validity_certificate_id, facet_trace_certificate_id,
                    boundary_subdivision_certificate_id, identification_id, source_mesh_id,
                    source_geometry_id, source_validity_certificate_id,
                    source_embedding_certificate_id,
                ),
                strict=True,
            )
        )
        prefix = "PeriodicRestrictionPartitionEvidence"
        direction = parse(direction, PeriodicPartitionDirection, f"{prefix}.direction")
        exact_planes = tuple(
            tuple(_exact_fraction(value, f"{prefix}.planes") for value in row)
            for row in planes
        )
        if not exact_planes or any(
            len(row) != 4 or not any(row[:3]) for row in exact_planes
        ) or len(set(exact_planes)) != len(exact_planes):
            raise ValueError(
                f"{prefix}.planes must be distinct exact planes with nonzero normals; got {planes!r}"
            )
        count = len(child_cell_ids)
        if not count or (
            len(parent_cell_ids), len(sign_vectors), len(child_volumes)
        ) != (count, count, count) or len(set(child_cell_ids)) != count:
            raise ValueError(
                f"{prefix}.children must name each partition child once with one parent, sign vector and volume; got {count} children"
            )
        if not partition_cell_ids or len(partition_volumes) != len(partition_cell_ids) or len(
            set(partition_cell_ids)
        ) != len(partition_cell_ids):
            raise ValueError(f"{prefix}.partition_cells must name each parent once with its volume")
        parents = dict(
            zip(
                partition_cell_ids,
                (_exact_fraction(value, f"{prefix}.partition_volumes") for value in partition_volumes),
                strict=True,
            )
        )
        totals = dict.fromkeys(parents, Fraction(0))
        signs: dict[int, set[tuple[int, ...]]] = {parent: set() for parent in parents}
        for parent, vector, volume in zip(
            parent_cell_ids, sign_vectors, child_volumes, strict=True
        ):
            exact = _exact_fraction(volume, f"{prefix}.child_volumes")
            if (
                parent not in parents
                or exact <= 0
                or len(vector) != len(planes)
                or any(value not in (-1, 1) for value in vector)
                or vector in signs[parent]
            ):
                raise ValueError(
                    f"{prefix}.children must be positive closed sign cells, unique within one declared parent; got parent {parent!r}"
                )
            signs[parent].add(vector)
            totals[parent] += exact
        if any(totals[parent] != volume or volume <= 0 for parent, volume in parents.items()):
            raise ValueError(
                f"{prefix}.partition_volumes must equal the exact sum of their children's volumes"
            )
        (
            self.mesh_id, self.geometry_id, self.periodic_topology_id,
            self.validity_certificate_id, self.facet_trace_certificate_id,
            self.boundary_subdivision_certificate_id, self.identification_id,
            self.source_mesh_id, self.source_geometry_id,
            self.source_validity_certificate_id, self.source_embedding_certificate_id,
        ) = identifiers
        self.planes = tuple(tuple(tuple(value) for value in row) for row in planes)
        self.child_cell_ids = tuple(int(value) for value in child_cell_ids)
        self.parent_cell_ids = tuple(int(value) for value in parent_cell_ids)
        self.sign_vectors = tuple(tuple(int(value) for value in row) for row in sign_vectors)
        self.child_volumes = tuple(tuple(value) for value in child_volumes)
        self.partition_cell_ids = tuple(int(value) for value in partition_cell_ids)
        self.partition_volumes = tuple(tuple(value) for value in partition_volumes)
        self.direction = direction
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "periodic-restriction-partition-evidence",
                **dict(zip(names, identifiers, strict=True)),
                "direction": direction,
                "planes": self.planes,
                "children": tuple(
                    zip(
                        self.child_cell_ids, self.parent_cell_ids, self.sign_vectors,
                        self.child_volumes, strict=True,
                    )
                ),
                "parents": tuple(zip(self.partition_cell_ids, self.partition_volumes, strict=True)),
            }
        )

    @classmethod
    @checked
    def from_stored_fields(
        cls, value: PeriodicRestrictionPartitionEvidence, /
    ) -> PeriodicRestrictionPartitionEvidence:
        """Recompute the exact partition arithmetic and every binding."""
        rebuilt = cls(
            value.mesh_id, value.geometry_id, value.periodic_topology_id,
            value.validity_certificate_id, value.facet_trace_certificate_id,
            value.boundary_subdivision_certificate_id, value.identification_id,
            value.source_mesh_id, value.source_geometry_id,
            value.source_validity_certificate_id, value.source_embedding_certificate_id,
            value.planes, value.child_cell_ids, value.parent_cell_ids,
            value.sign_vectors, value.child_volumes, value.partition_cell_ids,
            value.partition_volumes, direction=value.direction,
        )
        if any(getattr(rebuilt, field.name) != getattr(value, field.name) for field in fields(value)):
            raise ValueError(
                "PeriodicRestrictionPartitionEvidence.fields must bind the exact partition and every premise; got changed stored evidence"
            )
        return rebuilt


@final
class MappedRestrictionRootEmbeddingEvidence(StrictModule, NonTrainableState):
    """Bind an imported root-union certificate to the target's positive validity."""

    imported_certificate_id: str = eqx.field(static=True)
    target_binding_id: str = eqx.field(static=True)
    validity_certificate_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        imported_certificate_id: str,
        target_binding_id: str,
        validity_certificate_id: str,
    ) -> None:
        self.imported_certificate_id = canonical_identifier(
            imported_certificate_id,
            "MappedRestrictionRootEmbeddingEvidence.imported_certificate_id",
        )
        self.target_binding_id = canonical_identifier(
            target_binding_id, "MappedRestrictionRootEmbeddingEvidence.target_binding_id"
        )
        self.validity_certificate_id = canonical_identifier(
            validity_certificate_id,
            "MappedRestrictionRootEmbeddingEvidence.validity_certificate_id",
        )
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "mapped-restriction-root-embedding-evidence",
                "imported_certificate": self.imported_certificate_id,
                "target_binding": self.target_binding_id,
                "validity": self.validity_certificate_id,
            }
        )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[MappedRestrictionRootEmbeddingEvidence],
        value: MappedRestrictionRootEmbeddingEvidence,
        /,
    ) -> MappedRestrictionRootEmbeddingEvidence:
        """Regenerate the imported-proof binding and reject stale restored identities."""
        rebuilt = cls(
            value.imported_certificate_id,
            value.target_binding_id,
            value.validity_certificate_id,
        )
        if rebuilt.evidence_id != value.evidence_id:
            raise ValueError(
                "MappedRestrictionRootEmbeddingEvidence.evidence_id must bind every restored field; "
                f"got {value.evidence_id!r}"
            )
        return rebuilt


@dataclass
class EmbeddingState:
    findings: list[MeshCertificateFinding]
    checks: list[str]
    candidate_pairs: int = 0
    ray_tests: int = 0
    subdivision_pieces: int = 0
    subdivision_depth: int = 0
    boundary_facets: int = 0
    shells: int = 0
    periodic_images: int = 0
    source_expression_work_units: int = 0
    source_expression_peak_bytes: int = 0
    # Hypothesis (a) of the boundary-degree theorem: the bound validity
    # certificate proves det DF > 0 on every closed reference cell.
    closed_cell_orientation: bool = False
    boundary_degree: MappedBoundaryDegreeEvidence | None = None
    periodic_affine: PeriodicAffineQuotientEvidence | None = None
    periodic_affine_degree: PeriodicAffineTorusDegreeEvidence | None = None
    periodic_restriction: PeriodicRestrictionPartitionEvidence | None = None
    periodic_mapped_restriction: PeriodicMappedRestrictionEvidence | None = None
    imported_root_embedding_certificate_id: str | None = None

    def add(
        self,
        check: str,
        status: MeshFindingStatus,
        kind: MeshCertificateEntityKind,
        identifiers: np.ndarray | tuple[int, ...] = (),
        /,
        *,
        resource_error: CoordinateEnclosureResourceError | MeshcoreError | None = None,
        expression_budget: CoordinateEnclosureBudget | None = None,
    ) -> None:
        values = tuple(int(value) for value in np.asarray(identifiers).reshape(-1))
        achieved: tuple[tuple[str, int], ...] = ()
        requested: tuple[tuple[str, int], ...] = ()
        resource: MeshCertificateResource | None = None
        native_refusal: NativeCertificateRefusal | None = None
        match resource_error:
            case MeshcoreError():
                native_refusal = NativeCertificateRefusal(resource_error)
            case CoordinateEnclosureResourceError():
                phase_evidence = resource_error.phase_evidence
                if phase_evidence is None:
                    raise RuntimeError(
                        "EmbeddingState.resource_error.phase_evidence must contain measured coordinate work; got None"
                    )
                resource = resource_error.resource
                requested = (
                    ("limit", int(resource_error.limit)),
                    ("requested", int(resource_error.requested)),
                )
                achieved = (("completed", int(resource_error.completed)),)
                if expression_budget is not None:
                    achieved += (
                        (
                            "source_expression_work_units",
                            phase_evidence.work_units,
                        ),
                        (
                            "source_expression_peak_bytes",
                            phase_evidence.peak_bytes_upper,
                        ),
                        (
                            "source_expression_native_charged_work_units",
                            phase_evidence.native_charged_work_units,
                        ),
                        (
                            "source_expression_retained_basis_bytes_at_catch",
                            expression_budget.retained_basis_bytes,
                        ),
                        (
                            "source_expression_temporary_bytes_upper_at_catch",
                            expression_budget.temporary_bytes_upper,
                        ),
                    )
            case None:
                pass
            case invalid:
                assert_never(invalid)
        self.findings.append(
            MeshCertificateFinding(
                check,
                status,
                kind,
                values,
                resource=resource,
                requested=requested,
                achieved=achieved,
                native_refusal=native_refusal,
            )
        )

    @property
    def clean(self) -> bool:
        return not self.findings


ContactStatus: TypeAlias = Literal["separated", "overlap", "unresolved"]


@final
class ContactDecision(StrictModule, NonTrainableState):
    """Contact classification, unresolved check, and consumed subdivision work."""

    status: ContactStatus = eqx.field(static=True)
    check: MeshCertificateCheck | None = eqx.field(static=True)
    work: int = eqx.field(static=True)

    def __init__(
        self,
        status: ContactStatus,
        work: int,
        /,
        *,
        check: MeshCertificateCheck | None = None,
    ) -> None:
        status = parse(status, ContactStatus, "ContactDecision.status")
        if check is not None:
            check = parse(check, MeshCertificateCheck, "ContactDecision.check")
            if status != "unresolved":
                raise ValueError(
                    f"ContactDecision.check must accompany unresolved status; got {status}"
                )
        self.status = status
        self.check = check
        self.work = nonnegative_integer(work, "ContactDecision.work")

    def unresolved_check(self) -> MeshCertificateCheck:
        """Return the refusal check required before publishing a contact finding."""
        if self.status != "unresolved" or self.check is None:
            raise ValueError(
                f"ContactDecision.check must identify an unresolved finding; got status={self.status}, check={self.check}"
            )
        return self.check


class MappedEmbeddingCertifier(Protocol):
    """Request-local mapped proof capability for rational-source dispatch."""

    def __call__(
        self,
        state: EmbeddingState,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        cell_ids: np.ndarray,
        limits: MeshCertificateLimits,
        /,
        *,
        budget: CoordinateEnclosureBudget,
        predecessor: CellGeometryPredecessor | None = None,
    ) -> None: ...
