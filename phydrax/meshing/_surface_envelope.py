# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Declared unsigned-distance wrapping, never an inferred repair of a solid.

The repaired domain is the sublevel set of a native sampled distance field.
A soup remains a soup; the separately audited extracted surface receives new
identity. Two directed distance bounds and strict source inclusion are required
before a wrapping result can be published.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass, replace
from enum import Enum
from fractions import Fraction
from itertools import chain
from math import nextafter, sqrt
from types import MethodType
from typing import Any, assert_never, final, Literal

import equinox as eqx
import numpy as np
from jax import Array

from phydrax.meshing._contracts import (
    MeshingLimits,
    MeshingProviderInfo,
    VolumeMeshingSpec,
)
from phydrax.meshing._failures import MeshingFailure, MeshingFailureCategory
from phydrax.meshing._host_banks import family_host_array

from .. import _meshcore
from .._fingerprint import (
    array_tree_fingerprint,
    canonical_fingerprint,
    canonical_json,
    tree_content_equal,
)
from .._geometry_predicates import exact_charge
from .._host_storage import declared_integer_storage_bytes, declared_object_storage_bytes
from .._identity import _named_records, _static_payload, SemanticProvenance
from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import (
    canonical_identifier,
    nonnegative_integer,
    positive_integer,
    unique_identifiers,
)
from ..discretization._cell_geometry_validity import CellValidityPolicy
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..exterior._form_type import FormType, FormValueSpec
from ..geometry._domain_coverage import DomainCoverageCertificate, PiecewiseLinearDomain
from ..geometry._global_embedding import GlobalEmbeddingCertificate
from ..geometry.surface._contracts import SurfaceAuditPolicy, SurfaceMetadata
from ..geometry.surface._model import SurfaceModel, SurfaceRealization
from ..typing import (
    checked,
    ConvertibleToArray,
    Dim,
    HostFloat64,
    HostInt64,
    parse,
    Scope,
)
from ._association_state import GeometryAssociation
from ._measurements import (
    measure_phase,
    NativeExecutionRecord,
    NativeMeshingPhaseRecorder,
    phase_started,
    record_elapsed,
)
from ._native_preparation import native_preparation_ledger
from ._result import CellMeshingResult, MeshingComplianceReport
from ._trace import MeshingStageKind, MeshingStageReport, MeshingStageStatus, MeshingTrace
from ._volume_generation import (
    bind_volume_execution,
    finalize_plc_volume,
    native_volume_checkpoint,
    native_volume_execution_budget,
    native_volume_source_execution_budget,
    NativeVolumeSchedule,
    plc_source_rows,
    PlcCarrier,
    VolumeConstruction,
)
from .providers import (
    _native_sources as native_sources_owner,
    _native_volume as native_volume_owner,
)
from .providers._native_route_checks import check_deadline


class SoupVertexDim(Dim, minimum=1):
    """Raw soup coordinate rows, without manifold assumptions."""


class SoupFaceDim(Dim, minimum=1):
    """Raw soup triangle occurrences; duplicate occurrences are retained."""


class SoupFeatureDim(Dim):
    """Explicitly declared source feature edges."""


class EnvelopeVolumeVertexDim(Dim, minimum=4):
    """Vertices of the retained extraction cut-cell carrier."""


class EnvelopeVolumeCellDim(Dim, minimum=1):
    """Positively oriented tetrahedra of the retained extraction carrier."""


class EnvelopeFeaturePermission(str, Enum):
    PRESERVE = "preserve"
    ALLOW_ROUNDING = "allow_rounding"


class EnvelopeTopologyPermission(str, Enum):
    PRESERVE = "preserve"
    ALLOW_CHANGE = "allow_change"


@final
class RawTriangleSoup(StrictModule, NonTrainableState):
    """Finite indexed triangles with explicit provenance, not a SurfaceModel.

    Open, duplicate, nonmanifold, inconsistently oriented and zero-area faces
    are admitted as distance primitives, not silently deleted or reoriented.
    """

    __strict_contract__ = True
    vertices: HostFloat64[SoupVertexDim, Literal[3]]
    triangles: HostInt64[SoupFaceDim, Literal[3]]
    feature_edges: HostInt64[SoupFeatureDim, Literal[2]]
    metadata: SurfaceMetadata
    soup_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        vertices: ConvertibleToArray,
        triangles: ConvertibleToArray,
        metadata: SurfaceMetadata,
        /,
        *,
        feature_edges: ConvertibleToArray | None = None,
    ) -> None:
        scope = Scope()
        points = parse(
            np.array(vertices, dtype=np.float64, copy=True),
            HostFloat64[SoupVertexDim, Literal[3]],
            "vertices",
            scope=scope,
        )
        faces_raw = np.asarray(triangles)
        edges_raw = (
            np.zeros((0, 2), dtype=np.int64)
            if feature_edges is None
            else np.asarray(feature_edges)
        )
        if not np.issubdtype(faces_raw.dtype, np.integer) or not np.issubdtype(
            edges_raw.dtype, np.integer
        ):
            raise TypeError(
                "RawTriangleSoup.indices must have integer dtype; got noninteger triangle or feature indices"
            )
        faces = parse(
            np.array(faces_raw, dtype=np.int64, copy=True),
            HostInt64[SoupFaceDim, Literal[3]],
            "triangles",
            scope=scope,
        )
        edges = parse(
            np.array(edges_raw, dtype=np.int64, copy=True),
            HostInt64[SoupFeatureDim, Literal[2]],
            "feature_edges",
            scope=scope,
        )
        if not np.all(np.isfinite(points)):
            raise ValueError(
                "RawTriangleSoup.vertices must be finite; got nonfinite coordinates"
            )
        if (
            np.any(faces < 0)
            or np.any(faces >= points.shape[0])
            or np.any(edges < 0)
            or np.any(edges >= points.shape[0])
        ):
            raise ValueError(
                "RawTriangleSoup.indices must name declared coordinate rows; got out-of-range indices"
            )
        represented_edges = {
            tuple(sorted((int(face[local]), int(face[(local + 1) % 3]))))
            for face in faces
            for local in range(3)
        }
        declared_edges = tuple(
            tuple(sorted((int(edge[0]), int(edge[1])))) for edge in edges
        )
        if len(set(declared_edges)) != len(declared_edges) or any(
            a == b or (a, b) not in represented_edges for a, b in declared_edges
        ):
            raise ValueError(
                "RawTriangleSoup.feature_edges must be distinct noncollapsed represented edges; got invalid feature edge declarations"
            )
        if metadata.cell_tags and len(metadata.cell_tags) != faces.shape[0]:
            raise ValueError(
                "RawTriangleSoup.metadata.cell_tags must name every triangle occurrence; got misaligned cell tags"
            )
        points.setflags(write=False)
        faces.setflags(write=False)
        edges.setflags(write=False)
        self.vertices, self.triangles, self.feature_edges = points, faces, edges
        self.metadata = metadata
        self.soup_id = canonical_fingerprint(
            {
                "kind": "raw-triangle-soup",
                "metadata": metadata.metadata_id,
                "vertices": array_tree_fingerprint(points),
                "triangles": array_tree_fingerprint(faces),
                "features": array_tree_fingerprint(edges),
            }
        )


@final
class SurfaceEnvelopePolicy(StrictModule, NonTrainableState):
    """Physical offset/tolerance, permissions and finite native resource bounds.

    ``offset`` explicitly chooses sampled unsigned-distance thickening, including
    for open sheets. ``certificate_spacing`` bounds the barycentric source
    covering radius; ``spacing`` bounds the extraction grid cells. Neither
    sampling nor wrapping selects an intended interior of a dirty shell.
    Volume budgets are finite even when omitted: ``max_volume_vertices``
    defaults to ``max_vertices + 7*max_samples`` and ``max_tetrahedra`` to
    ``48*max_samples``, bounding the retained grid/cut-cell carrier.
    """

    offset: float = eqx.field(static=True)
    maximum_deviation: float = eqx.field(static=True)
    spacing: float = eqx.field(static=True)
    certificate_spacing: float = eqx.field(static=True)
    feature_permission: EnvelopeFeaturePermission = eqx.field(static=True)
    topology_permission: EnvelopeTopologyPermission = eqx.field(static=True)
    max_samples: int = eqx.field(static=True)
    max_work_units: int = eqx.field(static=True)
    max_vertices: int = eqx.field(static=True)
    max_triangles: int = eqx.field(static=True)
    max_volume_vertices: int = eqx.field(static=True)
    max_tetrahedra: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        offset: float,
        maximum_deviation: float,
        spacing: float,
        certificate_spacing: float,
        feature_permission: EnvelopeFeaturePermission,
        topology_permission: EnvelopeTopologyPermission,
        max_samples: int,
        max_work_units: int,
        max_vertices: int,
        max_triangles: int,
        max_volume_vertices: int | None = None,
        max_tetrahedra: int | None = None,
    ) -> None:
        values = tuple(
            float(value)
            for value in (offset, maximum_deviation, spacing, certificate_spacing)
        )
        if any(not np.isfinite(value) or value <= 0 for value in values):
            raise ValueError(
                "SurfaceEnvelopePolicy.distances must be finite and positive; got invalid distance bounds"
            )
        radius, deviation, step, certificate = values
        interpolation_error = np.sqrt(3.0) * step + certificate
        if radius <= interpolation_error:
            raise ValueError(
                "SurfaceEnvelopePolicy.offset must exceed grid-cell diameter plus source covering radius; got insufficient inclusion offset"
            )
        if radius + interpolation_error > deviation:
            raise ValueError(
                "SurfaceEnvelopePolicy.maximum_deviation must cover offset plus sampling and interpolation error; got insufficient deviation bound"
            )
        feature_permission = parse(
            feature_permission,
            EnvelopeFeaturePermission,
            "SurfaceEnvelopePolicy.feature_permission",
        )
        topology_permission = parse(
            topology_permission,
            EnvelopeTopologyPermission,
            "SurfaceEnvelopePolicy.topology_permission",
        )
        limits = (max_samples, max_work_units, max_vertices, max_triangles)
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in limits
        ):
            raise TypeError(
                "SurfaceEnvelopePolicy.budgets must be integers; got noninteger budgets"
            )
        if any(value < 1 for value in limits):
            raise ValueError(
                "SurfaceEnvelopePolicy.budgets must be positive finite bounds; got nonpositive budgets"
            )
        volume_vertices = (
            max_vertices + 7 * max_samples
            if max_volume_vertices is None
            else max_volume_vertices
        )
        tetrahedra = 48 * max_samples if max_tetrahedra is None else max_tetrahedra
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in (volume_vertices, tetrahedra)
        ):
            raise TypeError(
                "SurfaceEnvelopePolicy.volume_budgets must be integers; got noninteger volume budgets"
            )
        limits = (*limits, volume_vertices, tetrahedra)
        if any(value < 1 or value > np.iinfo(np.int64).max for value in limits):
            raise ValueError(
                "SurfaceEnvelopePolicy.budgets must be positive signed-int64 bounds; got out-of-range budgets"
            )
        self.offset, self.maximum_deviation, self.spacing, self.certificate_spacing = (
            values
        )
        self.feature_permission, self.topology_permission = (
            feature_permission,
            topology_permission,
        )
        (
            self.max_samples,
            self.max_work_units,
            self.max_vertices,
            self.max_triangles,
            self.max_volume_vertices,
            self.max_tetrahedra,
        ) = tuple(int(value) for value in limits)
        self.policy_id = canonical_fingerprint(
            {
                "kind": "unsigned-distance-envelope-policy",
                "distances": values,
                "feature_permission": feature_permission.value,
                "topology_permission": topology_permission.value,
                "budgets": limits,
            }
        )

    def native_limits(self) -> _meshcore.NativeExecutionLimits:
        """Lower the envelope's work cap and existing uncapped mechanical axes."""
        return _meshcore.NativeExecutionLimits(
            self.max_work_units,
            np.iinfo(np.uint64).max,
            np.iinfo(np.uint64).max,
            np.iinfo(np.intp).max,
            float("inf"),
        )


@dataclass(frozen=True, slots=True)
class EnvelopeTopology:
    """Triangle-occurrence incidence diagnostics, not inferred source homology.

    For a defective soup, ``euler_characteristic`` is only the count V-E+F:
    duplicate and degenerate triangle occurrences contribute to F, and the
    counts do not certify a valid complex or a unique intended geometry.
    Component and boundary counts likewise describe supplied index incidence.
    For the separately audited repaired closed surface, V-E+F has its usual
    Euler interpretation. Comparing these records never proves a homology
    transition or a source-to-repaired homeomorphism.
    """

    vertices: int
    edges: int
    faces: int
    components: int
    boundary_edges: int
    nonmanifold_edges: int
    degenerate_faces: int
    duplicate_faces: int
    euler_characteristic: int


def _degenerate_faces_exact(
    points: np.ndarray,
    faces: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> int:
    """Count rational dyadic collinearity, including binary64 underflow cases."""
    exact_points = []
    for index, row in enumerate(points):
        _surface_envelope_topology_checkpoint(budget, index)
        exact_points.append(
            (
                Fraction.from_float(float(row[0])),
                Fraction.from_float(float(row[1])),
                Fraction.from_float(float(row[2])),
            )
        )
    degenerate = 0
    for index, face in enumerate(faces):
        _surface_envelope_topology_checkpoint(budget, index)
        a, b, c = (exact_points[int(face[local])] for local in range(3))
        ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
        ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
        if (
            ab[0] * ac[1] == ab[1] * ac[0]
            and ab[0] * ac[2] == ab[2] * ac[0]
            and ab[1] * ac[2] == ab[2] * ac[1]
        ):
            degenerate += 1
    return degenerate


def _surface_envelope_topology_checkpoint(
    budget: CoordinateEnclosureBudget | None, index: int, /
) -> None:
    # Match the coordinate ledger's existing 256-admission checkpoint cadence.
    if budget is not None and index % 256 == 0:
        budget.ticket.checkpoint()


def _surface_envelope_topology_bounds(
    vertex_count: int, face_count: int, /
) -> tuple[int, int]:
    """Bound the retained graph, NumPy sorts and binary64 rational collinearity."""
    floating = np.finfo(np.float64)
    bits = max(floating.maxexp, 1 - floating.minexp + floating.nmant)
    fraction = 256 + 2 * declared_integer_storage_bytes(bits)
    index = declared_integer_storage_bytes(max(vertex_count, face_count).bit_length())
    # Three Fraction coordinates and their tuple remain live for every vertex.
    points = 256 + vertex_count * (16 + 304 + 3 * fraction)
    # At most 3F edge keys/incidence lists and 6F adjacency links coexist with
    # visited/pending indexes. The record bound also covers dictionary resizing.
    graph = (24 * face_count + 16) * (256 + 4 * index + 64)
    # Six differences and paired products need only per-face rational scratch;
    # numerator/denominator intermediates contain at most four input bit bounds.
    rational = 32 * (256 + 2 * declared_integer_storage_bytes(4 * bits + 8))
    storage = points + graph + rational + 256 * face_count
    # Work follows the retained-owner convention: one unit per visited vertex,
    # face and edge record (at most three per face).
    work = vertex_count + 4 * face_count
    return work, storage


def _topology(
    points: np.ndarray,
    faces: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> EnvelopeTopology:
    if budget is None:
        return _surface_envelope_topology(points, faces, budget=None)
    work, storage = _surface_envelope_topology_bounds(points.shape[0], faces.shape[0])
    with exact_charge(work, storage, budget=budget):
        topology = _surface_envelope_topology(points, faces, budget=budget)
        budget.ticket.checkpoint()
        return topology


def _surface_envelope_topology(
    points: np.ndarray,
    faces: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget | None,
) -> EnvelopeTopology:
    edges: dict[tuple[int, int], list[int]] = {}
    neighbors: list[list[int]] = [[] for _ in range(faces.shape[0])]
    for face_index, face in enumerate(faces):
        _surface_envelope_topology_checkpoint(budget, face_index)
        for local in range(3):
            a, b = int(face[local]), int(face[(local + 1) % 3])
            edges.setdefault((min(a, b), max(a, b)), []).append(face_index)
    neighbor_visits = 0
    for edge_index, incidents in enumerate(edges.values()):
        _surface_envelope_topology_checkpoint(budget, edge_index)
        first = incidents[0]
        for other in incidents[1:]:
            _surface_envelope_topology_checkpoint(budget, neighbor_visits)
            neighbor_visits += 1
            neighbors[first].append(other)
            neighbors[other].append(first)
    visited: set[int] = set()
    components = 0
    visited_count = 0
    for first in range(faces.shape[0]):
        if first in visited:
            continue
        components += 1
        pending = [first]
        visited.add(first)
        while pending:
            _surface_envelope_topology_checkpoint(budget, visited_count)
            visited_count += 1
            for other in neighbors[pending.pop()]:
                _surface_envelope_topology_checkpoint(budget, neighbor_visits)
                neighbor_visits += 1
                if other not in visited:
                    visited.add(other)
                    pending.append(other)
    _surface_envelope_topology_checkpoint(budget, 0)
    used_vertices = np.unique(faces).size
    duplicates = faces.shape[0] - np.unique(np.sort(faces, axis=1), axis=0).shape[0]
    _surface_envelope_topology_checkpoint(budget, 0)
    return EnvelopeTopology(
        vertices=used_vertices,
        edges=len(edges),
        faces=faces.shape[0],
        components=components,
        boundary_edges=sum(len(rows) == 1 for rows in edges.values()),
        nonmanifold_edges=sum(len(rows) > 2 for rows in edges.values()),
        degenerate_faces=_degenerate_faces_exact(points, faces, budget=budget),
        duplicate_faces=duplicates,
        euler_characteristic=used_vertices - len(edges) + faces.shape[0],
    )


@dataclass(frozen=True, slots=True)
class SurfaceEnvelopeEvidence:
    source_id: str
    source_revision: str
    source_geometry_id: str
    repaired_model_id: str
    policy_id: str
    source_to_repaired_upper: float
    repaired_to_source_upper: float
    source_inclusion_margin: float
    interpolation_error: float
    source_topology: EnvelopeTopology
    repaired_topology: EnvelopeTopology
    rounded_feature_edges: int
    lost_selection_ids: tuple[str, ...]
    lost_interface_ids: tuple[str, ...]
    discarded_cell_tags: tuple[str, ...]
    grid_samples: int
    work_units: int
    carrier_vertices: int
    carrier_tetrahedra: int
    certificate_samples: int
    evidence_id: str


def _surface_envelope_source_id(
    source_id: str, source_geometry_id: str, policy_id: str, /
) -> str:
    return canonical_fingerprint(
        {
            "kind": "repaired-envelope-source",
            "source": source_id,
            "source_geometry": source_geometry_id,
            "policy": policy_id,
        }
    )


def _surface_envelope_revision(
    source_geometry_id: str,
    policy_id: str,
    vertices: np.ndarray,
    triangles: np.ndarray,
    /,
) -> str:
    return canonical_fingerprint(
        {
            "kind": "surface-envelope-revision",
            "source_geometry": source_geometry_id,
            "policy": policy_id,
            "vertices": array_tree_fingerprint(vertices),
            "triangles": array_tree_fingerprint(triangles),
        }
    )


def _surface_envelope_evidence_id(
    evidence: SurfaceEnvelopeEvidence, surface_vertices: int, /
) -> str:
    """Authenticate the source-to-repaired transition, not target mesh statistics."""
    counts = _meshcore.SurfaceEnvelopeCounts(
        surface_vertices=surface_vertices,
        surface_triangles=evidence.repaired_topology.faces,
        grid_samples=evidence.grid_samples,
        work_units=evidence.work_units,
        source_samples=evidence.certificate_samples,
        volume_vertices=evidence.carrier_vertices,
        tetrahedra=evidence.carrier_tetrahedra,
    )
    return canonical_fingerprint(
        {
            "kind": "surface-envelope-evidence",
            "source": evidence.source_geometry_id,
            "repaired": evidence.repaired_model_id,
            "policy": evidence.policy_id,
            "bounds": array_tree_fingerprint(
                np.asarray(
                    [
                        evidence.source_to_repaired_upper,
                        evidence.repaired_to_source_upper,
                        evidence.source_inclusion_margin,
                        evidence.interpolation_error,
                    ],
                    dtype=np.float64,
                )
            ),
            "counts": array_tree_fingerprint(
                np.asarray([value for _, value in counts.items()], dtype=np.int64)
            ),
            "source_topology": asdict(evidence.source_topology),
            "repaired_topology": asdict(evidence.repaired_topology),
            "rounded_features": evidence.rounded_feature_edges,
            "lost_selections": evidence.lost_selection_ids,
            "lost_interfaces": evidence.lost_interface_ids,
            "discarded_tags": evidence.discarded_cell_tags,
        }
    )


_SURFACE_ENVELOPE_REPAIR_ACHIEVED_NAMES = (
    "envelope:source_to_repaired_upper",
    "envelope:repaired_to_source_upper",
    "envelope:source_inclusion_margin",
    "envelope:interpolation_error",
    "envelope:original_components",
    "envelope:repaired_components",
    "envelope:original_boundary_edges",
    "envelope:repaired_boundary_edges",
    "envelope:rounded_feature_edges",
    "envelope:grid_samples",
    "envelope:certificate_samples",
    "envelope:work_units",
    "envelope:carrier_vertices",
    "envelope:carrier_tetrahedra",
)
_SURFACE_ENVELOPE_REPAIR_METRIC_NAMES = frozenset(
    (
        *_SURFACE_ENVELOPE_REPAIR_ACHIEVED_NAMES,
        "envelope:maximum_deviation",
        "envelope:offset",
        "envelope:minimum_inclusion_margin",
    )
)


def _surface_envelope_achieved(
    evidence: SurfaceEnvelopeEvidence, /
) -> tuple[tuple[str, float], ...]:
    return tuple(
        zip(
            _SURFACE_ENVELOPE_REPAIR_ACHIEVED_NAMES,
            (
                evidence.source_to_repaired_upper,
                evidence.repaired_to_source_upper,
                evidence.source_inclusion_margin,
                evidence.interpolation_error,
                evidence.source_topology.components,
                evidence.repaired_topology.components,
                evidence.source_topology.boundary_edges,
                evidence.repaired_topology.boundary_edges,
                evidence.rounded_feature_edges,
                evidence.grid_samples,
                evidence.certificate_samples,
                evidence.work_units,
                evidence.carrier_vertices,
                evidence.carrier_tetrahedra,
            ),
            strict=True,
        )
    )


def _surface_envelope_requested(
    policy: SurfaceEnvelopePolicy, /
) -> tuple[tuple[str, float], ...]:
    return (
        ("envelope:maximum_deviation", policy.maximum_deviation),
        ("envelope:offset", policy.offset),
        ("envelope:minimum_inclusion_margin", 0.0),
    )


def _require_surface_envelope_repair_metrics(
    result: CellMeshingResult,
    evidence: SurfaceEnvelopeEvidence,
    policy: SurfaceEnvelopePolicy,
    /,
) -> None:
    actual = tuple(
        tuple(item for item in group if item[0] in _SURFACE_ENVELOPE_REPAIR_METRIC_NAMES)
        for group in (result.compliance.requested, result.compliance.achieved)
    )
    expected = (
        _surface_envelope_requested(policy),
        tuple(
            (name, float(value)) for name, value in _surface_envelope_achieved(evidence)
        ),
    )
    if actual != expected or canonical_json(actual) != canonical_json(expected):
        raise ValueError(
            "SurfaceEnvelope.repair_provenance.metrics must retain the ordered requested and achieved repair inventories; got missing, duplicate, reordered or changed repair metrics"
        )


def _surface_envelope_policy_payload(policy: SurfaceEnvelopePolicy, /) -> dict[str, Any]:
    return {
        "policy_id": policy.policy_id,
        "interpretation": "sampled_unsigned_distance_sublevel",
        "offset": policy.offset,
        "maximum_deviation": policy.maximum_deviation,
        "spacing": policy.spacing,
        "certificate_spacing": policy.certificate_spacing,
        "feature_permission": policy.feature_permission.value,
        "topology_permission": policy.topology_permission.value,
        "max_samples": policy.max_samples,
        "max_work_units": policy.max_work_units,
        "max_vertices": policy.max_vertices,
        "max_triangles": policy.max_triangles,
        "max_volume_vertices": policy.max_volume_vertices,
        "max_tetrahedra": policy.max_tetrahedra,
    }


def _surface_envelope_repair_stage(
    evidence: SurfaceEnvelopeEvidence, source_id: str, source_revision: str, /
) -> MeshingStageReport:
    return MeshingStageReport(
        MeshingStageKind.TOPOLOGY_REPAIR,
        MeshingStageStatus.PASSED,
        input_ids=(
            evidence.source_id,
            evidence.source_revision,
            evidence.source_geometry_id,
            evidence.policy_id,
        ),
        output_ids=(
            source_id,
            source_revision,
            evidence.repaired_model_id,
            evidence.evidence_id,
        ),
        created_count=evidence.repaired_topology.faces,
        deleted_count=evidence.source_topology.faces,
    )


@final
class SurfaceEnvelope(StrictModule, NonTrainableState):
    __strict_contract__ = True
    source: RawTriangleSoup | SurfaceModel
    policy: SurfaceEnvelopePolicy
    repaired: SurfaceRealization
    execution_evidence: NativeExecutionRecord | None
    carrier_vertices: HostFloat64[EnvelopeVolumeVertexDim, Literal[3]]
    carrier_tetrahedra: HostInt64[EnvelopeVolumeCellDim, Literal[4]]
    carrier_id: str = eqx.field(static=True)
    evidence: SurfaceEnvelopeEvidence = eqx.field(static=True)
    envelope_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source: RawTriangleSoup | SurfaceModel,
        policy: SurfaceEnvelopePolicy,
        repaired: SurfaceRealization,
        evidence: SurfaceEnvelopeEvidence,
        carrier_vertices: ConvertibleToArray,
        carrier_tetrahedra: ConvertibleToArray,
        /,
        *,
        execution_evidence: NativeExecutionRecord | None = None,
    ) -> None:
        geometry_id = (
            source.soup_id if isinstance(source, RawTriangleSoup) else source.model_id
        )
        if (
            evidence.source_geometry_id != geometry_id
            or evidence.repaired_model_id != repaired.model.model_id
            or evidence.policy_id != policy.policy_id
        ):
            raise ValueError(
                "SurfaceEnvelope.evidence.binding must match source, repaired geometry and policy; got different evidence IDs"
            )
        distances = (
            evidence.source_to_repaired_upper,
            evidence.repaired_to_source_upper,
            evidence.source_inclusion_margin,
            evidence.interpolation_error,
        )
        if any(not np.isfinite(value) or value < 0 for value in distances):
            raise ValueError(
                "SurfaceEnvelope.evidence.distances must be finite and nonnegative; got invalid distance evidence"
            )
        if (evidence.source_id, evidence.source_revision) != (
            source.metadata.source_id,
            source.metadata.source_revision,
        ) or not repaired.policy.require_closed:
            raise ValueError(
                "SurfaceEnvelope.evidence.source must preserve source identity and certify a closed repaired surface; got different identity or open realization"
            )
        if (
            evidence.source_inclusion_margin <= 0
            or max(evidence.source_to_repaired_upper, evidence.repaired_to_source_upper)
            > policy.maximum_deviation
        ):
            raise ValueError(
                "SurfaceEnvelope.evidence.certificate must certify inclusion and the deviation bound; got unmet certificate"
            )
        scope = Scope()
        points = parse(
            np.array(carrier_vertices, dtype=np.float64, copy=True),
            HostFloat64[EnvelopeVolumeVertexDim, Literal[3]],
            "carrier_vertices",
            scope=scope,
        )
        raw_cells = np.asarray(carrier_tetrahedra)
        if not np.issubdtype(raw_cells.dtype, np.integer):
            raise TypeError(
                "SurfaceEnvelope.carrier_tetrahedra must have integer dtype; got noninteger indices"
            )
        cells = parse(
            np.array(raw_cells, dtype=np.int64, copy=True),
            HostInt64[EnvelopeVolumeCellDim, Literal[4]],
            "carrier_tetrahedra",
            scope=scope,
        )
        if (
            not np.all(np.isfinite(points))
            or np.any(cells < 0)
            or np.any(cells >= points.shape[0])
        ):
            raise ValueError(
                "SurfaceEnvelope.carrier must have finite coordinates and valid tetrahedron indices; got invalid carrier arrays"
            )
        boundary_points = np.asarray(repaired.mesh.coordinates, dtype=np.float64)
        boundary_triangles = np.asarray(repaired.mesh.blocks[0].vertices, dtype=np.int64)
        if not np.array_equal(points[: boundary_points.shape[0]], boundary_points):
            raise ValueError(
                "SurfaceEnvelope.carrier_vertices must preserve the repaired surface vertex prefix; got different boundary coordinates"
            )
        corners = points[cells]
        if np.any(
            _meshcore.exact_orient3d(
                corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]
            )
            <= 0
        ):
            raise ValueError(
                "SurfaceEnvelope.carrier_tetrahedra must have positive orientation; got nonpositive tetrahedra"
            )
        all_faces = np.concatenate(
            (
                cells[:, [1, 3, 2]],
                cells[:, [0, 2, 3]],
                cells[:, [0, 3, 1]],
                cells[:, [0, 1, 2]],
            )
        )
        unique_faces, face_counts = np.unique(
            np.sort(all_faces, axis=1), axis=0, return_counts=True
        )
        if np.any(face_counts > 2) or not np.array_equal(
            unique_faces[face_counts == 1],
            np.unique(np.sort(boundary_triangles, axis=1), axis=0),
        ):
            raise ValueError(
                "SurfaceEnvelope.carrier_boundary must contain every extraction skin triangle and no additional exterior face; got different or nonmanifold boundary"
            )
        if (
            (points.shape[0], cells.shape[0])
            != (evidence.carrier_vertices, evidence.carrier_tetrahedra)
            or points.shape[0] > policy.max_volume_vertices
            or cells.shape[0] > policy.max_tetrahedra
        ):
            raise ValueError(
                "SurfaceEnvelope.carrier_counts must match count evidence and fit volume budgets; got different or excessive counts"
            )
        points.setflags(write=False)
        cells.setflags(write=False)
        self.carrier_vertices, self.carrier_tetrahedra = points, cells
        self.carrier_id = canonical_fingerprint(
            {
                "kind": "surface-envelope-cut-cell-carrier",
                "surface": repaired.model.model_id,
                "points": array_tree_fingerprint(points),
                "tetrahedra": array_tree_fingerprint(cells),
            }
        )
        self.source, self.policy, self.repaired, self.evidence = (
            source,
            policy,
            repaired,
            evidence,
        )
        self.envelope_id = canonical_fingerprint(
            {
                "kind": "surface-envelope",
                "evidence": evidence.evidence_id,
                "realization": repaired.realization_id,
                "carrier": self.carrier_id,
            }
        )
        if execution_evidence is not None:
            if type(execution_evidence) is not NativeExecutionRecord:
                raise TypeError(
                    "SurfaceEnvelope.execution_evidence must be a NativeExecutionRecord; got a different record type"
                )
            execution_evidence.require_valid()
            if execution_evidence.owner_id != self.envelope_id:
                raise ValueError(
                    "SurfaceEnvelope.execution_evidence.owner_id must match the envelope ID; got another wrapping owner"
                )
        self.execution_evidence = execution_evidence

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[SurfaceEnvelope], value: SurfaceEnvelope, /
    ) -> SurfaceEnvelope:
        """Rebuild a stored envelope and authenticate its retained execution record."""
        fresh = wrap_surface_envelope(value.source, value.policy)
        return cls(
            fresh.source,
            fresh.policy,
            fresh.repaired,
            fresh.evidence,
            fresh.carrier_vertices,
            fresh.carrier_tetrahedra,
            execution_evidence=value.execution_evidence,
        )


@checked
def wrap_surface_envelope(
    source: RawTriangleSoup | SurfaceModel,
    policy: SurfaceEnvelopePolicy,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> SurfaceEnvelope:
    """Native finite-budget distance thickening with two-sided certificates."""
    scope = _meshcore.NativeExecutionScope(policy.native_limits())
    with scope, scope.host_workspace() as storage:
        storage.retain_owner(source)
        envelope = _wrap_surface_envelope(
            source, policy, scope=scope, record_phase=record_phase
        )
        storage.retain_owner(envelope)
    if scope.evidence is None:
        raise RuntimeError(
            "SurfaceEnvelope.execution_evidence must retain the ended native record; got missing ended execution evidence"
        )
    receipt = NativeExecutionRecord(scope.evidence, owner_id=envelope.envelope_id)
    envelope = eqx.tree_at(
        lambda value: value.execution_evidence,
        envelope,
        receipt,
        is_leaf=lambda value: value is None,
    )
    parent = scope.parent
    if parent is not None:
        with parent.host_workspace() as storage:
            ledger = native_preparation_ledger(parent)
            ledger.remember_receipt(parent, receipt)
            ledger.remember_source(parent, envelope.envelope_id)
            storage.retain_owner(envelope)
    return envelope


def _require_surface_envelope_wrapping_policy(policy: SurfaceEnvelopePolicy, /) -> None:
    match policy.feature_permission:
        case EnvelopeFeaturePermission.PRESERVE:
            raise ValueError(
                "SurfaceEnvelopePolicy.feature_permission must allow rounding for unsigned-distance wrapping; got feature preservation requested"
            )
        case EnvelopeFeaturePermission.ALLOW_ROUNDING:
            pass
        case invalid_permission:
            assert_never(invalid_permission)
    match policy.topology_permission:
        case EnvelopeTopologyPermission.PRESERVE:
            raise ValueError(
                "SurfaceEnvelopePolicy.topology_permission must allow topology change for unsigned-distance wrapping; got no homeomorphism certificate"
            )
        case EnvelopeTopologyPermission.ALLOW_CHANGE:
            pass
        case invalid_permission:
            assert_never(invalid_permission)


def _require_surface_envelope_native_bounds(
    source_upper: float,
    repaired_upper: float,
    inclusion_margin: float,
    interpolation_error: float,
    policy: SurfaceEnvelopePolicy,
    /,
) -> None:
    # Native prepare_envelope_grid/build_covering_cloud use unconditional
    # nextafter at each positive sum. Source coordinate magnitudes can only
    # increase this policy-derived roundoff floor.
    denorm = nextafter(0.0, np.inf)
    roundoff = nextafter(
        max(
            256.0 * float(np.finfo(np.float64).eps) * (policy.offset + policy.spacing),
            256.0 * denorm,
        ),
        np.inf,
    )
    diameter = nextafter(nextafter(sqrt(3.0), np.inf) * policy.spacing, np.inf)
    arithmetic = nextafter(4.0 * roundoff, np.inf)
    covering = nextafter(denorm + roundoff, np.inf)
    initial_error = nextafter(
        nextafter(diameter + policy.certificate_spacing, np.inf) + arithmetic, np.inf
    )
    error_floor = nextafter(
        nextafter(nextafter(diameter + covering, np.inf) + arithmetic, np.inf), np.inf
    )
    source_floor = nextafter(
        nextafter(nextafter(denorm + covering, np.inf) + arithmetic, np.inf), np.inf
    )
    if (
        not all(
            np.isfinite(value) and value > 0.0
            for value in (
                source_upper,
                repaired_upper,
                inclusion_margin,
                interpolation_error,
            )
        )
        or repaired_upper != nextafter(policy.offset + interpolation_error, np.inf)
        or inclusion_margin != nextafter(policy.offset - interpolation_error, -np.inf)
        or max(source_upper, repaired_upper) > policy.maximum_deviation
        or not policy.offset > initial_error
        or interpolation_error < error_floor
        or source_upper < source_floor
    ):
        raise ValueError(
            "SurfaceEnvelope.repair_provenance.bounds must retain positive finite native outward-rounded bounds; got impossible wrapping distances"
        )


def _wrap_surface_envelope(
    source: RawTriangleSoup | SurfaceModel,
    policy: SurfaceEnvelopePolicy,
    /,
    *,
    scope: _meshcore.NativeExecutionScope,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> SurfaceEnvelope:
    """Run the represented wrapping construction in its already admitted root."""
    _require_surface_envelope_wrapping_policy(policy)
    if isinstance(source, RawTriangleSoup):
        points, faces, source_geometry_id = (
            source.vertices,
            source.triangles,
            source.soup_id,
        )
        feature_count, selection_ids, interface_ids = (
            source.feature_edges.shape[0],
            (),
            (),
        )
    else:
        source.prepare()
        points = np.asarray(source.mesh.coordinates, dtype=np.float64)
        faces = np.asarray(source.mesh.blocks[0].vertices, dtype=np.int64)
        source_geometry_id = source.model_id
        feature_count = 0
        selection_ids = tuple(selection.selection_id for selection in source.selections)
        interface_ids = tuple(interface.interface_id for interface in source.interfaces)
    measurement_started = phase_started(record_phase)
    native_work: int | None = None
    try:
        native = _meshcore.surface_envelope(
            points,
            faces,
            offset=policy.offset,
            spacing=policy.spacing,
            certificate_spacing=policy.certificate_spacing,
            max_samples=policy.max_samples,
            max_work_units=policy.max_work_units,
            max_vertices=policy.max_vertices,
            max_triangles=policy.max_triangles,
            max_volume_vertices=policy.max_volume_vertices,
            max_tetrahedra=policy.max_tetrahedra,
        )
        vertices = native.surface_points
        triangles = native.surface_faces
        counts = native.counts
        bounds = native.bounds
        carrier_vertices = native.volume_points
        carrier_tetrahedra = native.cells
        if record_phase is not None:
            native_work = counts.work_units
    finally:
        record_elapsed(
            record_phase,
            "envelope_construction",
            measurement_started,
            work_units=native_work,
        )
    if max(float(bounds[0]), float(bounds[1])) > policy.maximum_deviation:
        raise ValueError(
            "SurfaceEnvelope.deviation_certificate must fit the declared maximum deviation; got uncertified two-sided deviation"
        )
    _require_surface_envelope_native_bounds(
        float(bounds[0]),
        float(bounds[1]),
        float(bounds[2]),
        float(bounds[3]),
        policy,
    )
    original_topology, repaired_topology = (
        _topology(points, faces),
        _topology(vertices, triangles),
    )
    revision = _surface_envelope_revision(
        source_geometry_id, policy.policy_id, vertices, triangles
    )
    metadata = SurfaceMetadata(
        source_id=_surface_envelope_source_id(
            source.metadata.source_id, source_geometry_id, policy.policy_id
        ),
        source_revision=revision,
        coordinate_contract=source.metadata.coordinate_contract,
        provenance=(
            *source.metadata.provenance,
            f"unsigned-distance-envelope:{source_geometry_id}:{policy.policy_id}",
        ),
    )
    model = SurfaceModel.from_triangles(
        vertices, triangles, metadata, numeric_version=revision
    )
    realization = model.prepare(
        SurfaceAuditPolicy(
            require_closed=True,
            maximum_vertices=policy.max_vertices,
            maximum_cells=policy.max_triangles,
        )
    )
    evidence = SurfaceEnvelopeEvidence(
        source_id=source.metadata.source_id,
        source_revision=source.metadata.source_revision,
        source_geometry_id=source_geometry_id,
        repaired_model_id=model.model_id,
        policy_id=policy.policy_id,
        source_to_repaired_upper=float(bounds[0]),
        repaired_to_source_upper=float(bounds[1]),
        source_inclusion_margin=float(bounds[2]),
        interpolation_error=float(bounds[3]),
        source_topology=original_topology,
        repaired_topology=repaired_topology,
        rounded_feature_edges=feature_count,
        lost_selection_ids=selection_ids,
        lost_interface_ids=interface_ids,
        discarded_cell_tags=source.metadata.cell_tags,
        grid_samples=counts.grid_samples,
        work_units=counts.work_units,
        carrier_vertices=counts.volume_vertices,
        carrier_tetrahedra=counts.tetrahedra,
        certificate_samples=counts.source_samples,
        evidence_id="",
    )
    evidence = replace(
        evidence,
        evidence_id=_surface_envelope_evidence_id(evidence, vertices.shape[0]),
    )
    return SurfaceEnvelope(
        source, policy, realization, evidence, carrier_vertices, carrier_tetrahedra
    )


@checked
def envelope_volume_support_issues(
    source: native_sources_owner.NativeSurfaceEnvelopeSource,
    specification: VolumeMeshingSpec,
    /,
) -> list[str]:

    issues = native_volume_owner.volume_support_issues(source.plc_source, specification)
    boundary = specification.boundary_scope
    if (boundary.source_id, boundary.source_revision) != (
        source.source_id,
        source.source_revision,
    ):
        issues.append(
            "scopes bound to the repaired envelope identity rather than the input soup"
        )
    return issues


@final
class PreparedSurfaceEnvelopeVolume(StrictModule, NonTrainableState):
    schedule: NativeVolumeSchedule
    specification_id: str = eqx.field(static=True)
    source_binding_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source: native_sources_owner.NativeSurfaceEnvelopeSource,
        specification: VolumeMeshingSpec,
        schedule: NativeVolumeSchedule,
        /,
    ) -> None:
        issues = envelope_volume_support_issues(source, specification)
        if issues:
            raise ValueError(
                "PreparedSurfaceEnvelopeVolume.constraints must be supported; got "
                + "; ".join(issues)
            )
        self.schedule, self.source_binding_id = schedule, source.binding_id
        self.specification_id = specification.specification_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-surface-envelope-volume",
                "source": source.binding_id,
                "specification": specification.specification_id,
                "schedule": schedule.schedule_id,
            }
        )


@checked
def surface_envelope_source_entities(
    source: native_sources_owner.NativeSurfaceEnvelopeSource,
    /,
    *,
    scope: _meshcore.NativeExecutionScope,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Input triangles, polygon IDs and PLC edge identity table.

    Both native carrier insertion and represented-source transfer consume this
    same namespace. No identity is inferred from coordinate proximity.
    """

    complex_ = source.plc_source.complex
    triangles = np.asarray(
        source.envelope.repaired.mesh.blocks[0].vertices, dtype=np.int64
    )

    if complex_.polygon_offsets.size != triangles.shape[0] + 1:
        raise ValueError(
            "SurfaceEnvelope.facets must preserve extraction skin facets; got different PLC facets"
        )
    with scope.batch(7 * triangles.shape[0] - 1) as ticket:
        ticket.consume(triangles.shape[0])
        for polygon, face in enumerate(triangles):
            first, last = complex_.polygon_offsets[polygon : polygon + 2]
            if last - first != 3 or not np.array_equal(
                complex_.polygon_vertices[first:last], face
            ):
                raise ValueError(
                    "SurfaceEnvelope.facets must preserve extraction skin facets; got different PLC facets"
                )
        occurrences = family_host_array(
            (3 * triangles.shape[0], 2), np.int64, scope=scope
        )
        ticket.consume(occurrences.shape[0])
        for slot, (first, last) in enumerate(((0, 1), (1, 2), (2, 0))):
            rows = occurrences[
                slot * triangles.shape[0] : (slot + 1) * triangles.shape[0]
            ]
            np.minimum(triangles[:, first], triangles[:, last], out=rows[:, 0])
            np.maximum(triangles[:, first], triangles[:, last], out=rows[:, 1])
        # Structured heapsort is in-place and has no data-sized sorting workspace.
        occurrences.view([("first", np.int64), ("last", np.int64)]).reshape(-1).sort(
            order=("first", "last"),
            kind="heapsort",
        )
        distinct = 1
        ticket.consume(occurrences.shape[0] - 1)
        for row in range(1, occurrences.shape[0]):
            if not np.array_equal(occurrences[row], occurrences[distinct - 1]):
                occurrences[distinct] = occurrences[row]
                distinct += 1
        edges = occurrences[:distinct]
        polygons = family_host_array((triangles.shape[0],), np.int64, scope=scope)
        for polygon in range(polygons.size):
            polygons[polygon] = polygon
    return triangles, polygons, edges


@checked
def generate_surface_envelope_volume(
    source: native_sources_owner.NativeSurfaceEnvelopeSource,
    specification: VolumeMeshingSpec,
    schedule: NativeVolumeSchedule,
    /,
    *,
    validity_policy: CellValidityPolicy,
    scope: _meshcore.NativeExecutionScope,
    workspace: _meshcore.NativeHostStorageWorkspace,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> VolumeConstruction:
    """Construct the retained carrier inside the cumulative allowance."""

    receipt = source.envelope.execution_evidence
    with native_volume_source_execution_budget(
        specification.limits,
        receipt,
        source.envelope.envelope_id,
        scope=scope,
    ):
        workspace.retain_owner(source)
        with native_volume_execution_budget(specification.limits) as execution:
            construction = _generate_surface_envelope_volume(
                source,
                specification,
                schedule,
                validity_policy=validity_policy,
                scope=execution,
                workspace=workspace,
                record_phase=record_phase,
            )
    evidence = execution.evidence
    if evidence is None:
        raise RuntimeError(
            "SurfaceEnvelopeVolume.execution_evidence must retain the ended construction record; got missing execution evidence"
        )
    return bind_volume_execution(
        construction,
        evidence,
        0 if receipt is None else int(np.asarray(receipt.total_work_units)),
        0 if receipt is None else int(np.asarray(receipt.total_geometry_queries)),
        preparation_evidence=receipt,
    )


def _surface_envelope_carrier_source(
    source: native_sources_owner.NativeSurfaceEnvelopeSource,
    specification: VolumeMeshingSpec,
    triangles: np.ndarray,
    polygons: np.ndarray,
    edges: np.ndarray,
    *,
    scope: _meshcore.NativeExecutionScope,
    workspace: _meshcore.NativeHostStorageWorkspace,
) -> _meshcore.TetMeshSourceComplex:
    """Declare extraction-skin rows and rational indexed endpoint witnesses."""

    rows = plc_source_rows(
        source.plc_source.complex,
        specification,
        triangles,
        polygons,
        edges,
        np.empty((0, 2), dtype=np.int64),
    )
    points, faces, face_labels, face_bounds = (
        rows.points,
        rows.faces,
        rows.triangle_facets,
        rows.face_tolerances,
    )
    segments, segment_ids, segment_bounds = (
        rows.segments,
        rows.segment_ids,
        rows.segment_tolerances,
    )
    count = source.envelope.carrier_vertices.shape[0]
    with scope.batch(count + 2 * segments.shape[0]) as ticket:
        strata = family_host_array((count,), np.int8, scope=scope)
        entities = family_host_array((count,), np.int32, scope=scope)
        parameters = family_host_array((count, 2), np.float64, scope=scope)
        ticket.consume(count + 2 * segments.shape[0])
        strata.fill(0)
        entities.fill(-1)
        parameters.fill(0.0)
        for row, segment in enumerate(segments):
            for endpoint, vertex in enumerate(segment):
                if strata[vertex] == 0:
                    strata[vertex] = 1
                    entities[vertex] = row
                    parameters[vertex, 0] = endpoint
        if np.any(strata[: points.shape[0]] == 0):
            raise ValueError(
                "SurfaceEnvelopeVolume.source_vertices must retain source stratum ownership; got an unowned skin vertex"
            )
        face_ids, face_groups = np.unique(face_labels, return_inverse=True)
        declared = _meshcore.TetMeshSourceComplex(
            points,
            faces,
            face_groups.astype(np.int32),
            face_bounds,
            segments,
            np.arange(segments.shape[0], dtype=np.int32),
            segment_bounds,
            strata,
            entities,
            parameters,
            face_ids,
            segment_ids,
        )
        workspace.retain_owner(declared)
    return declared


def _generate_surface_envelope_volume(
    source: native_sources_owner.NativeSurfaceEnvelopeSource,
    specification: VolumeMeshingSpec,
    schedule: NativeVolumeSchedule,
    /,
    *,
    validity_policy: CellValidityPolicy,
    scope: _meshcore.NativeExecutionScope,
    workspace: _meshcore.NativeHostStorageWorkspace,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> VolumeConstruction:
    prepared = PreparedSurfaceEnvelopeVolume(source, specification, schedule)
    envelope = source.envelope
    limits = specification.limits
    evidence = envelope.evidence
    if (
        envelope.carrier_vertices.shape[0] > limits.maximum_vertices
        or envelope.carrier_tetrahedra.shape[0] > limits.maximum_cells
        or evidence.work_units >= limits.maximum_work_units
    ):
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "SurfaceEnvelopeVolume.carrier must fit the volume construction budget; got excessive vertices, cells, or work",
            stage=MeshingStageKind.VOLUME_FILL.value,
            achieved=(
                ("vertices", envelope.carrier_vertices.shape[0]),
                ("tetrahedra", envelope.carrier_tetrahedra.shape[0]),
                ("work_units", evidence.work_units),
            ),
        )
    complex_ = source.plc_source.complex
    native_volume_checkpoint(limits, MeshingStageKind.VOLUME_FILL, scope=scope)
    triangles, polygons, edges = surface_envelope_source_entities(source, scope=scope)

    regions = family_host_array(
        (envelope.carrier_tetrahedra.shape[0],), np.int32, scope=scope
    )
    edge_ids = family_host_array((edges.shape[0],), np.int64, scope=scope)
    with scope.batch(regions.size + edge_ids.size) as ticket:
        ticket.consume(regions.size + edge_ids.size)
        regions.fill(0)
        for edge in range(edge_ids.size):
            edge_ids[edge] = edge
    native_volume_checkpoint(limits, MeshingStageKind.VOLUME_FILL, scope=scope)
    declared_source = _surface_envelope_carrier_source(
        source,
        specification,
        triangles,
        polygons,
        edges,
        scope=scope,
        workspace=workspace,
    )
    state: _meshcore.TetMesh3D | None = None
    try:
        with measure_phase(record_phase, "topology_construction"):
            state = _meshcore.TetMesh3D(
                envelope.carrier_vertices,
                envelope.carrier_tetrahedra,
                regions,
                triangles,
                polygons,
                edges,
                edge_ids,
                boundary_policy=complex_.boundary,
                max_vertices=limits.maximum_vertices,
                max_tetrahedra=limits.maximum_cells,
                max_scratch_bytes=limits.maximum_scratch_bytes,
                source=declared_source,
            )
    except BaseException as error:
        # Recorder failures are consumer errors; they must not retain a newly
        # created mutable native state before the finalizer assumes ownership.
        if state is not None:
            state.close()
        if (
            isinstance(error, _meshcore.MeshcoreError)
            and error.status is _meshcore.MeshcoreStatus.CAPACITY_EXCEEDED
        ):
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "SurfaceEnvelopeVolume.carrier must fit the native scratch allowance; got allocation refusal",
                stage=MeshingStageKind.VOLUME_FILL.value,
                requested=(("maximum_scratch_bytes", limits.maximum_scratch_bytes),),
                provider_code=error.status.name,
            ) from error
        raise
    if state is None:
        raise RuntimeError(
            "SurfaceEnvelopeVolume.carrier must retain the constructed native state; got missing state"
        )
    stage = MeshingStageReport(
        MeshingStageKind.VOLUME_FILL,
        MeshingStageStatus.PASSED,
        input_ids=(envelope.envelope_id, prepared.prepared_id),
        output_ids=(envelope.carrier_id,),
        created_count=envelope.carrier_tetrahedra.shape[0],
    )
    return finalize_plc_volume(
        PlcCarrier(
            state=state,
            complex=complex_,
            source_id=source.plc_source.source_id,
            source_revision=source.plc_source.source_revision,
            input_id=prepared.prepared_id,
            input_triangles=triangles,
            input_polygons=polygons,
            plc_edges=edges,
            diagonal_polygons=np.empty((0,), dtype=np.int64),
            construction_stage=stage,
            construction_counters=(
                ("grid_samples", evidence.grid_samples),
                ("certificate_samples", evidence.certificate_samples),
                ("carrier_vertices", evidence.carrier_vertices),
                ("carrier_tetrahedra", evidence.carrier_tetrahedra),
                ("work_units", evidence.work_units),
            ),
            construction_work_units=evidence.work_units,
            source=declared_source,
        ),
        specification,
        schedule,
        validity_policy=validity_policy,
        scope=scope,
        record_phase=record_phase,
    )


@checked
def execute_surface_envelope_route(
    source: native_sources_owner.NativeSurfaceEnvelopeSource,
    specification: VolumeMeshingSpec,
    prepared: PreparedSurfaceEnvelopeVolume,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    /,
    *,
    scope: _meshcore.NativeExecutionScope,
    workspace: _meshcore.NativeHostStorageWorkspace,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> CellMeshingResult:
    """Keep construction, source certification and publication in one root."""
    receipt = source.envelope.execution_evidence
    with native_volume_source_execution_budget(
        specification.limits,
        receipt,
        source.envelope.envelope_id,
        scope=scope,
    ):
        workspace.retain_owner(source)
        return _execute_surface_envelope_route(
            source,
            specification,
            prepared,
            coordinate_contract,
            provider,
            plan_id,
            scope=scope,
            workspace=workspace,
            record_phase=record_phase,
        )


def _execute_surface_envelope_route(
    source: native_sources_owner.NativeSurfaceEnvelopeSource,
    specification: VolumeMeshingSpec,
    prepared: PreparedSurfaceEnvelopeVolume,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    /,
    *,
    scope: _meshcore.NativeExecutionScope,
    workspace: _meshcore.NativeHostStorageWorkspace,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> CellMeshingResult:
    """Publish PLC acceptance while retaining repair evidence."""

    started = scope.operation_started

    if (
        prepared.source_binding_id != source.binding_id
        or prepared.specification_id != specification.specification_id
    ):
        raise ValueError(
            "PreparedSurfaceEnvelopeVolume.binding must match source and specification IDs; got different prepared route binding"
        )
    if (
        coordinate_contract.spatial_id
        != source.envelope.source.metadata.coordinate_contract.spatial_id
    ):
        raise ValueError(
            "SurfaceEnvelopeVolume.coordinate_contract must preserve declared source units; got different spatial ID"
        )

    volume = native_volume_owner.PreparedPlcVolume(
        source.plc_source, specification, prepared.schedule
    )
    construction = _generate_surface_envelope_volume(
        source,
        specification,
        prepared.schedule,
        scope=scope,
        workspace=workspace,
        validity_policy=native_volume_owner.plc_volume_audit_policy().validity_policy,
        record_phase=record_phase,
    )
    check_deadline(started, specification.limits, MeshingStageKind.VOLUME_FILL)
    result = native_volume_owner.execute_volume_route(
        source.plc_source,
        specification,
        volume,
        coordinate_contract,
        provider,
        plan_id,
        scope=scope,
        workspace=workspace,
        construction=construction,
        record_phase=record_phase,
    )
    check_deadline(started, specification.limits, MeshingStageKind.CERTIFICATION)
    evidence = source.envelope.evidence
    compliance = MeshingComplianceReport(
        specification.specification_id,
        issues=result.compliance.issues,
        requested=(
            *result.compliance.requested,
            *_surface_envelope_requested(source.envelope.policy),
        ),
        achieved=(
            *result.compliance.achieved,
            *_surface_envelope_achieved(evidence),
        ),
    )
    policy = source.envelope.policy
    provenance = SemanticProvenance(
        {
            "kind": "native-surface-envelope-volume",
            "plan": plan_id,
            "envelope": source.envelope.envelope_id,
            "preserved_carrier": source.envelope.carrier_id,
            "wrapping_evidence": asdict(evidence),
            "wrapping_policy": _surface_envelope_policy_payload(policy),
            "exact_repaired_volume": result.provenance.content_id,
        },
        resource_ids=result.provenance.resource_ids,
    )
    wrapping_stage = _surface_envelope_repair_stage(
        evidence, source.source_id, source.source_revision
    )
    trace = MeshingTrace(
        (wrapping_stage, *result.trace.stages), binding=result.trace.binding
    )
    publication = CellMeshingResult(
        result.mesh,
        result.geometry,
        result.coordinate_contract,
        result.audit,
        result.quality,
        compliance,
        trace,
        result.provider,
        result.runtime,
        result.derivative_mode,
        provenance,
        boundary=result.boundary,
        patches=result.patches,
        zones=result.zones,
        labels=result.labels,
        attributes=result.attributes,
        associations=result.associations,
        adapter_reports=result.adapter_reports,
        certification=result.certification,
        region_evidence=result.region_evidence,
        surface_source=result.surface_source,
    )
    check_deadline(
        started, specification.limits, MeshingStageKind.SPECIFICATION_COMPLIANCE
    )
    return publication


def _repair_provenance_mapping(value: Any, /) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("kind") != "mapping":
        raise ValueError(
            "SurfaceEnvelope.repair_provenance must retain named repair records; got another semantic payload"
        )
    records = _named_records(value["items"], "SurfaceEnvelope.repair_provenance.records")
    if set(value) != {"kind", "items"} or value["items"] != [
        list(record) for record in records
    ]:
        raise ValueError(
            "SurfaceEnvelope.repair_provenance.records must retain unique named records in declared order; got changed record ordering or fields"
        )
    return dict(records)


def _admit_surface_envelope_record(
    value: object,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    storage_admission: Callable[[int, int, int], None] | None = None,
) -> None:
    """Admit full declared owners before replay copies, sorting or fingerprinting.

    Work follows the retained-owner convention: one unit per visited owner.
    Immutable text is borrowed by the constructors, so a repeated reference
    admits only its expanded serialization buffers.
    """
    # Lazy: the model owner registry also registers meshing source records.
    from .._model._structure import _model_owner_children

    def admit(work: int, metadata: int = 0, buffers: int = 0) -> None:
        if storage_admission is None:
            budget.reserve(work, metadata + buffers)
        else:
            storage_admission(work, metadata, buffers)

    admit(0, 1024)
    pending: list[tuple[int | None, Iterator[object]]] = [(None, iter((value,)))]
    active: set[int] = set()
    text_owners: dict[int, str] = {}
    while pending:
        owner, children = pending[-1]
        try:
            current = next(children)
        except StopIteration:
            pending.pop()
            if owner is not None:
                active.remove(owner)
            continue
        if isinstance(current, str) and text_owners.get(id(current)) is current:
            # JSON still emits this occurrence: escaping uses at most twelve
            # ASCII bytes per code point, held as text and UTF-8 together.
            admit(0, 128, 24 * (len(current) + 2))
            continue
        admit(1)
        descendants: Iterator[object]
        if isinstance(current, (Array, np.ndarray)):
            count = int(current.size)
            # Reconstruction may normalize narrow banks to int64/float64 and
            # retain source, host, device and fingerprint temporaries together.
            copied = max(int(current.nbytes), 8 * count)
            admit(0, 8 * 256, 8 * copied)
            descendants = iter((current.dtype,))
            if isinstance(current, np.ndarray) and current.dtype.hasobject:
                descendants = chain((current.dtype,), current.flat)
        else:
            storage = 128 + 8 * declared_object_storage_bytes(current)
            text_buffers = 32 * len(current) if isinstance(current, str) else 0
            admit(0, storage - text_buffers, text_buffers)
            if isinstance(current, str):
                text_owners[id(current)] = current
            if isinstance(current, np.dtype):
                descendants = iter((current.fields, current.subdtype, current.metadata))
            elif isinstance(current, Fraction):
                descendants = iter((current.numerator, current.denominator))
            elif isinstance(current, Enum):
                # Constructor parsing/equality uses the existing enum value;
                # neither rebuilds its class/name/value object graph.
                descendants = iter(())
            elif isinstance(current, np.generic):
                descendants = iter((current.dtype, current.item()))
            elif isinstance(current, memoryview):
                descendants = iter((current.obj,))
            elif isinstance(current, FormValueSpec):
                descendants = iter(
                    (current.form_type, current.proxy, current.value_spec_id)
                )
            elif isinstance(current, FormType):
                descendants = iter(
                    (
                        current.dimension,
                        current.degree,
                        current.twist,
                        current.fiber_shape,
                        current.ambient_dimension,
                        current.form_type_id,
                    )
                )
            elif current is None or isinstance(
                current, (str, bytes, bytearray, int, float, complex, type)
            ):
                continue
            elif is_dataclass(current) or isinstance(
                current, (Mapping, tuple, list, set, frozenset, MethodType)
            ):
                descendants = iter(_model_owner_children(current))
            else:
                raise TypeError(
                    "SurfaceEnvelope.repair_preparation.owner must have declared model fields; got an unsupported host owner"
                )
        identifier = id(current)
        if identifier in active:
            raise ValueError(
                "SurfaceEnvelope.repair_preparation.owner must be acyclic; got recursive retained evidence"
            )
        active.add(identifier)
        pending.append((identifier, descendants))


@contextmanager
def _surface_envelope_record_preparation(
    records: Iterable[object], /, *, budget: CoordinateEnclosureBudget
) -> Iterator[None]:
    with budget.temporary_scope():
        budget.reserve(1, 512)
        for record in records:
            _admit_surface_envelope_record(record, budget=budget)
        yield
        budget.ticket.checkpoint()


def _surface_envelope_certificate_inputs(
    result: CellMeshingResult, /
) -> Iterator[object]:
    yield result.certification
    yield result.audit.report_id
    yield result.mesh.vertex_global_ids
    for block in result.mesh.blocks:
        yield block.global_ids
    for entities in result.mesh.topology.entity_sets:
        yield entities.entity_set_id
    yield result.geometry.geometry_layout_id
    if result.geometry.exact_source is not None:
        yield result.geometry.exact_source.source_id
    if result.geometry.periodic_source is not None:
        yield result.geometry.periodic_source.source_id


@contextmanager
def _admit_surface_envelope_geometry_replay(
    result: CellMeshingResult, /, *, budget: CoordinateEnclosureBudget
) -> Iterator[tuple[int, int]]:
    geometry = result.geometry
    banks = (
        (geometry,)
        if geometry.storage is not None
        else (geometry.elements, geometry.geometry_dofs, geometry.coordinates)
    )
    metadata_upper = 0
    maximum_buffer = 0
    bank_buffer = 0
    with budget.live_storage() as storage:

        def admit(work: int, metadata: int, buffers: int) -> None:
            nonlocal metadata_upper, maximum_buffer, bank_buffer
            next_metadata = metadata_upper + metadata
            next_buffer = bank_buffer + buffers
            next_maximum = max(maximum_buffer, next_buffer)
            storage.set_bound(next_metadata + next_maximum, work=work)
            metadata_upper = next_metadata
            bank_buffer = next_buffer
            maximum_buffer = next_maximum

        for bank in banks:
            bank_buffer = 0
            _admit_surface_envelope_record(
                bank,
                budget=budget,
                storage_admission=admit,
            )
        # Keep metadata and the shared buffer through the consuming replays.
        yield metadata_upper, maximum_buffer


def _certified_surface_envelope_domain(
    result: CellMeshingResult, /, *, budget: CoordinateEnclosureBudget
) -> PiecewiseLinearDomain:
    report = result.certification
    if (
        report is None
        or not report.passed
        or not result.audit.passed
        or not result.compliance.passed
        or report.embedding is None
        or report.coverage is None
        or not isinstance(report.request.domain, PiecewiseLinearDomain)
    ):
        raise ValueError(
            "SurfaceEnvelope.repair_provenance.certification must independently certify the repaired domain; got missing or undecided volume evidence"
        )
    # Reconstruct policies, request assignments and their geometry bindings under
    # the caller's allowance; no retained limit is permitted to authorize it.
    from ..lifecycle._meshing_source_records import (
        reconstruct_stored_fields,
        register_meshing_source_types,
    )

    with _surface_envelope_record_preparation((report.request.limits,), budget=budget):
        fresh_limits = replace(report.request.limits)
        if not tree_content_equal(fresh_limits, report.request.limits):
            raise ValueError(
                "SurfaceEnvelope.repair_provenance.limits must match reconstructed policy fields; got stale limit identity"
            )
        if (
            report.embedding.binding.limits_id != fresh_limits.limits_id
            or report.coverage.binding.limits_id != fresh_limits.limits_id
        ):
            raise ValueError(
                "SurfaceEnvelope.repair_provenance.limits must match both certificate bindings; got foreign limit identities"
            )
        del fresh_limits
    with (
        _surface_envelope_record_preparation(
            _surface_envelope_certificate_inputs(result), budget=budget
        ),
        _admit_surface_envelope_geometry_replay(result, budget=budget),
    ):
        report.request.validate_source_integrity(budget=budget)
        register_meshing_source_types()
        rebuilt_request = reconstruct_stored_fields(
            report.request,
            bindings={
                "mesh": result.mesh,
                "geometry": result.geometry,
                "budget": budget,
            },
        )
        del rebuilt_request
        for binding in (report.embedding.binding, report.coverage.binding):
            rebuilt_binding = reconstruct_stored_fields(
                binding,
                bindings={
                    "mesh": result.mesh,
                    "geometry": result.geometry,
                    "limits": report.request.limits,
                },
            )
            del rebuilt_binding
        domain = report.request.domain
        embedding = GlobalEmbeddingCertificate.from_stored_fields(report.embedding)
        coverage = DomainCoverageCertificate.from_stored_fields(report.coverage, domain)
        if not tree_content_equal(embedding, report.embedding) or not tree_content_equal(
            coverage, report.coverage
        ):
            raise ValueError(
                "SurfaceEnvelope.repair_provenance.certificates must match reconstructed evidence; got changed embedding or coverage content"
            )
        reconstructed_report = reconstruct_stored_fields(
            report,
            bindings={
                "mesh": result.mesh,
                "geometry": result.geometry,
                "audit": result.audit,
            },
        )
        del reconstructed_report
        if (
            embedding.status != "certified"
            or coverage.status != "certified"
            or coverage.domain_id != domain.domain_id
            or coverage.embedding_certificate_id != embedding.certificate_id
            or coverage.covered_source_facet_count != coverage.source_facet_count
            or coverage.region_ids != domain.region_ids
        ):
            raise ValueError(
                "SurfaceEnvelope.repair_provenance.domain must bind coverage of every repaired-domain facet; got foreign or incomplete coverage"
            )
        del embedding, coverage
        budget.ticket.checkpoint()
    return domain


def _require_surface_envelope_payload_schema(payload: dict[str, Any], /) -> None:
    common = {
        "kind",
        "envelope",
        "preserved_carrier",
        "wrapping_evidence",
        "wrapping_policy",
        "exact_repaired_volume",
    }
    history = (
        ("plan",)
        if payload["kind"] == "native-surface-envelope-volume"
        else ("source", "repair_provenance")
    )
    if set(payload) != common | set(history):
        raise ValueError(
            "SurfaceEnvelope.repair_provenance.fields must retain every declared owner field; got missing or foreign history fields"
        )
    for name in ("envelope", "preserved_carrier", "exact_repaired_volume", *history):
        canonical_identifier(payload[name], f"SurfaceEnvelope.repair_provenance.{name}")


def _surface_envelope_provenance_bounds(
    character_count: int, resource_storage: int, /
) -> tuple[int, int]:
    """Bound both owner parses and their normalization/hash traversals."""
    extent = character_count + resource_storage + 1
    # JSON has at most one value/slot per input character. Sixteen simultaneous
    # parsed, normalized and serialized copies cover the owner and consumer;
    # metadata and integer bounds use the existing declared-storage convention.
    metadata = declared_object_storage_bytes({}) + 64 + 16
    storage = 16 * (metadata * extent + declared_integer_storage_bytes(4 * extent))
    # Work follows the retained-owner convention: each of the two parses visits
    # at most one decoded owner per input slot.
    work = 2 * extent
    return work, storage


@contextmanager
def _surface_envelope_repair_payload(
    result: CellMeshingResult,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> Iterator[dict[str, Any] | None]:
    """Keep authenticated decoded records inside their admitted host lifetime."""
    provenance = result.provenance
    resources = budget.live_object_storage_upper(provenance.resource_ids)
    work, storage = _surface_envelope_provenance_bounds(
        len(provenance.content_json), resources
    )
    with exact_charge(work, storage, budget=budget):
        groups = result.compliance.requested, result.compliance.achieved
        metric_count = sum(len(group) for group in groups)
        budget.reserve(metric_count, 8 * (256 + 64 * metric_count))
        for group in groups:
            for name, _ in group:
                budget.reserve(1, 8 * declared_object_storage_bytes(name))
        SemanticProvenance.from_stored_fields(provenance)
        budget.ticket.checkpoint()
        encoded = json.loads(provenance.content_json)
        budget.ticket.checkpoint()
        yield _classify_surface_envelope_repair_payload(result, encoded)
        budget.ticket.checkpoint()


def _classify_surface_envelope_repair_payload(
    result: CellMeshingResult, encoded: Any, /
) -> dict[str, Any] | None:
    payload = (
        _repair_provenance_mapping(encoded)
        if isinstance(encoded, dict) and encoded.get("kind") == "mapping"
        else {}
    )
    if payload.get("kind") in (
        "native-surface-envelope-volume",
        "native-adapted-surface-envelope-volume",
    ):
        _require_surface_envelope_payload_schema(payload)
        return payload
    # No retained source-kind field distinguishes an envelope from another PLC.
    # Only this owner's declared requested/achieved metric schema reserves repair
    # history here; generic topology repairs and unrelated payload keys do not.
    if any(
        name in _SURFACE_ENVELOPE_REPAIR_METRIC_NAMES
        for group in (result.compliance.requested, result.compliance.achieved)
        for name, _ in group
    ):
        raise ValueError(
            "SurfaceEnvelope.repair_provenance.owner must retain the declared repair owner while its evidence remains; got a missing or foreign owner"
        )
    return None


def _validate_surface_envelope_topology_counts(topology: EnvelopeTopology, /) -> None:
    for name in ("vertices", "edges", "faces", "components"):
        positive_integer(getattr(topology, name), f"SurfaceEnvelope.topology.{name}")
    for name in (
        "boundary_edges",
        "nonmanifold_edges",
        "degenerate_faces",
        "duplicate_faces",
    ):
        nonnegative_integer(getattr(topology, name), f"SurfaceEnvelope.topology.{name}")
    if (
        topology.vertices > 3 * topology.faces
        or topology.edges > 3 * topology.faces
        or topology.components > topology.faces
        or topology.boundary_edges + topology.nonmanifold_edges > topology.edges
        or topology.degenerate_faces > topology.faces
        or topology.duplicate_faces >= topology.faces
        or topology.euler_characteristic
        != topology.vertices - topology.edges + topology.faces
    ):
        raise ValueError(
            "SurfaceEnvelope.repair_provenance.topology must retain occurrence-count identities; got impossible topology counters"
        )


def _validate_surface_envelope_evidence_fields(
    fields: dict[str, Any], policy: SurfaceEnvelopePolicy, /
) -> None:
    """Validate the native counters, not similarly named unrelated policy axes."""
    _require_surface_envelope_wrapping_policy(policy)
    _require_surface_envelope_native_bounds(
        fields["source_to_repaired_upper"],
        fields["repaired_to_source_upper"],
        fields["source_inclusion_margin"],
        fields["interpolation_error"],
        policy,
    )
    original, repaired = fields["source_topology"], fields["repaired_topology"]
    _validate_surface_envelope_topology_counts(original)
    _validate_surface_envelope_topology_counts(repaired)
    for name in (
        "grid_samples",
        "certificate_samples",
        "work_units",
        "carrier_vertices",
        "carrier_tetrahedra",
    ):
        positive_integer(fields[name], f"SurfaceEnvelope.repair_provenance.{name}")
    nonnegative_integer(
        fields["rounded_feature_edges"],
        "SurfaceEnvelope.repair_provenance.rounded_feature_edges",
    )
    # Native prepare_envelope_grid has three extents >=2. Covering-cloud
    # samples share sample_limit with that grid and emit >=3 points per face.
    grid, samples = fields["grid_samples"], fields["certificate_samples"]
    # Work::spend counts cloud creation, every grid/cloud pair and every
    # cloud/repaired-vertex certificate pair, in addition to carrier extraction.
    minimum_work = samples * (1 + grid + repaired.vertices)
    if (
        grid < 8
        or samples < 3 * original.faces
        or grid + samples > policy.max_samples
        or fields["work_units"] < minimum_work
        or fields["work_units"] > policy.max_work_units
        or repaired.vertices > policy.max_vertices
        or repaired.faces > policy.max_triangles
        or fields["carrier_vertices"] < max(4, repaired.vertices)
        or fields["carrier_vertices"] > policy.max_volume_vertices
        or fields["carrier_tetrahedra"] > policy.max_tetrahedra
        or fields["rounded_feature_edges"] > np.iinfo(np.intp).max
        or fields["rounded_feature_edges"] > original.edges
    ):
        raise ValueError(
            "SurfaceEnvelope.repair_provenance.counters must fit native construction semantics and policy; got impossible or excessive repair work/storage"
        )
    # Input feature edges are distinct represented edges. Nonempty tag banks
    # are face-aligned; tag whitespace and repeated values remain source data.
    for name in ("lost_selection_ids", "lost_interface_ids"):
        unique_identifiers(fields[name], f"SurfaceEnvelope.{name}", allow_empty=True)
    tags = fields["discarded_cell_tags"]
    if (tags and len(tags) != original.faces) or any(
        type(tag) is not str or not tag for tag in tags
    ):
        raise ValueError(
            "SurfaceEnvelope.repair_provenance.history must retain face-aligned nonempty string cell tags; got invalid discarded tags"
        )
    if fields["rounded_feature_edges"] and (
        fields["lost_selection_ids"] or fields["lost_interface_ids"]
    ):
        raise ValueError(
            "SurfaceEnvelope.repair_provenance.history must retain one source-kind contract; got raw-soup features with surface-model selection/interface history"
        )


def _decode_surface_envelope_repair(
    payload: dict[str, Any], /
) -> tuple[SurfaceEnvelopeEvidence, SurfaceEnvelopePolicy]:
    fields = _repair_provenance_mapping(payload["wrapping_evidence"])
    for name in ("source_topology", "repaired_topology"):
        topology = _repair_provenance_mapping(fields[name])
        fields[name] = EnvelopeTopology(
            **{key: int(value) for key, value in topology.items()}
        )
    for name in ("lost_selection_ids", "lost_interface_ids", "discarded_cell_tags"):
        if not isinstance(fields[name], dict) or fields[name].get("kind") != "tuple":
            raise ValueError(
                "SurfaceEnvelope.repair_provenance.evidence must retain tuple declarations; got another source-history representation"
            )
        fields[name] = tuple(fields[name]["items"])
    for name in (
        "source_to_repaired_upper",
        "repaired_to_source_upper",
        "source_inclusion_margin",
        "interpolation_error",
    ):
        fields[name] = float(fields[name])
    for name in (
        "rounded_feature_edges",
        "grid_samples",
        "work_units",
        "carrier_vertices",
        "carrier_tetrahedra",
        "certificate_samples",
    ):
        fields[name] = int(fields[name])
    for name in (
        "source_id",
        "source_revision",
        "source_geometry_id",
        "repaired_model_id",
        "policy_id",
        "evidence_id",
    ):
        canonical_identifier(
            fields[name], f"SurfaceEnvelope.repair_provenance.evidence.{name}"
        )
    policy_fields = _repair_provenance_mapping(payload["wrapping_policy"])
    policy_id = policy_fields.pop("policy_id")
    interpretation = policy_fields.pop("interpretation")
    policy_fields["feature_permission"] = EnvelopeFeaturePermission(
        policy_fields["feature_permission"]
    )
    policy_fields["topology_permission"] = EnvelopeTopologyPermission(
        policy_fields["topology_permission"]
    )
    policy = SurfaceEnvelopePolicy(**policy_fields)
    _validate_surface_envelope_evidence_fields(fields, policy)
    evidence = SurfaceEnvelopeEvidence(**fields)
    if (
        interpretation != "sampled_unsigned_distance_sublevel"
        or policy.policy_id != policy_id
        or evidence.policy_id != policy.policy_id
    ):
        raise ValueError(
            "SurfaceEnvelope.repair_provenance.policy must authenticate its interpretation and identity; got another wrapping policy"
        )
    for name, canonical in (
        ("wrapping_evidence", asdict(evidence)),
        ("wrapping_policy", _surface_envelope_policy_payload(policy)),
    ):
        expected = _static_payload(canonical, f"SurfaceEnvelope.{name}")
        if canonical_json(payload[name]) != canonical_json(expected):
            raise ValueError(
                "SurfaceEnvelope.repair_provenance.representation must retain the owner's fields and scalar types without normalization; got normalized or extra repair fields"
            )
    return evidence, policy


def _authenticate_surface_envelope_repair(
    evidence: SurfaceEnvelopeEvidence,
    policy: SurfaceEnvelopePolicy,
    domain: PiecewiseLinearDomain,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> None:
    topology = _topology(domain.vertices, domain.facets, budget=budget)
    if (
        evidence.evidence_id
        != _surface_envelope_evidence_id(evidence, domain.vertices.shape[0])
        or evidence.source_inclusion_margin <= 0.0
        or min(
            evidence.source_to_repaired_upper,
            evidence.repaired_to_source_upper,
            evidence.interpolation_error,
        )
        < 0.0
        or max(evidence.source_to_repaired_upper, evidence.repaired_to_source_upper)
        > policy.maximum_deviation
        or evidence.repaired_topology != topology
        or topology.vertices != domain.vertices.shape[0]
        or any(
            (
                topology.boundary_edges,
                topology.nonmanifold_edges,
                topology.degenerate_faces,
                topology.duplicate_faces,
            )
        )
    ):
        raise ValueError(
            "SurfaceEnvelope.repair_provenance.evidence must authenticate its source-to-repaired transition and policy; got changed or uncertified wrapping evidence"
        )


def _require_surface_envelope_associations(
    result: CellMeshingResult,
    source_id: str,
    source_revision: str,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> None:
    if not result.associations:
        raise ValueError(
            "SurfaceEnvelope.repair_provenance.associations must retain repaired source bindings; got no associations"
        )
    for association in result.associations:
        rebuilt = GeometryAssociation.from_stored_fields(association, budget=budget)
        budget.ticket.checkpoint()
        if (rebuilt.source_id, rebuilt.source_revision) != (
            source_id,
            source_revision,
        ):
            raise ValueError(
                "SurfaceEnvelope.repair_provenance.associations must match the repaired source revision; got foreign source bindings"
            )
        del rebuilt


def _bind_surface_envelope_repair(
    result: CellMeshingResult,
    evidence: SurfaceEnvelopeEvidence,
    policy: SurfaceEnvelopePolicy,
    domain: PiecewiseLinearDomain,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> MeshingStageReport:
    # The retained declared PLC, not the adapted mesh, supplies these arrays.
    # NativeSurfaceEnvelopeSource preserves the repaired triangle order exactly.
    source_id = _surface_envelope_source_id(
        evidence.source_id, evidence.source_geometry_id, policy.policy_id
    )
    source_revision = _surface_envelope_revision(
        evidence.source_geometry_id, policy.policy_id, domain.vertices, domain.facets
    )
    _require_surface_envelope_associations(
        result, source_id, source_revision, budget=budget
    )
    stage = _surface_envelope_repair_stage(evidence, source_id, source_revision)
    _require_surface_envelope_repair_metrics(result, evidence, policy)
    if domain.source_id != source_id or stage not in result.trace.stages:
        raise ValueError(
            "SurfaceEnvelope.repair_provenance.source must bind the same repaired PLC, ancestry and directed bounds; got foreign or stripped source history"
        )
    return stage


@contextmanager
def _retained_surface_envelope_repair(
    result: CellMeshingResult,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> Iterator[
    tuple[
        dict[str, Any],
        SurfaceEnvelopeEvidence,
        SurfaceEnvelopePolicy,
        MeshingStageReport,
        PiecewiseLinearDomain,
    ]
    | None
]:
    with _surface_envelope_repair_payload(result, budget=budget) as payload:
        if payload is None:
            yield None
            return
        evidence, policy = _decode_surface_envelope_repair(payload)
        domain = _certified_surface_envelope_domain(result, budget=budget)
        _authenticate_surface_envelope_repair(evidence, policy, domain, budget=budget)
        stage = _bind_surface_envelope_repair(
            result, evidence, policy, domain, budget=budget
        )
        yield payload, evidence, policy, stage, domain


def _require_surface_envelope_repair_provenance(
    source: native_sources_owner.NativeSurfaceEnvelopeSource,
    result: CellMeshingResult,
    /,
    *,
    limits: MeshingLimits,
) -> None:
    """Authenticate retained repair history against its independently kept author."""
    report = result.certification
    if report is None:
        raise ValueError(
            "SurfaceEnvelope.repair_provenance.certification must retain the declared request limits; got no volume certificate"
        )
    # This allowance is independently supplied by the caller's specification.
    from .providers._native_publication import coordinate_budget, coordinate_phase

    with _meshcore.NativeExecutionScope(limits.native_limits()) as scope:
        budget = coordinate_budget(limits)
        with budget.retained_storage(lambda amount: scope.batch(0, bytes_upper=amount)):
            with (
                coordinate_phase(scope, limits, budget),
                budget.measurement(),
                budget.activate(),
            ):
                with _retained_surface_envelope_repair(result, budget=budget) as retained:
                    if retained is None:
                        raise ValueError(
                            "SurfaceEnvelope.repair_provenance must retain its declared envelope owner; got another provenance kind"
                        )
                    payload, evidence, policy, stage, _ = retained
                    if (
                        evidence != source.envelope.evidence
                        or policy.policy_id != source.envelope.policy.policy_id
                        or payload["envelope"] != source.envelope.envelope_id
                        or payload["preserved_carrier"] != source.envelope.carrier_id
                        or stage.output_ids[:2]
                        != (source.source_id, source.source_revision)
                    ):
                        raise ValueError(
                            "SurfaceEnvelope.repair_provenance must match the retained authored envelope; got foreign repair provenance"
                        )


def _carry_surface_envelope_repair(
    source: CellMeshingResult,
    target: CellMeshingResult,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> CellMeshingResult:
    """Retain only authenticated repair invariants after target certification."""
    with _retained_surface_envelope_repair(source, budget=budget) as retained:
        if retained is None:
            return target
        payload, evidence, policy, stage, domain = retained
        target_domain = _certified_surface_envelope_domain(target, budget=budget)
        _require_surface_envelope_associations(
            target, *stage.output_ids[:2], budget=budget
        )
        if (
            target_domain.domain_id != domain.domain_id
            or target.coordinate_contract.spatial_id
            != source.coordinate_contract.spatial_id
        ):
            raise ValueError(
                "SurfaceEnvelope.repair_provenance.target must certify the same repaired PLC and source revision; got foreign target evidence"
            )
        if any(
            name in _SURFACE_ENVELOPE_REPAIR_METRIC_NAMES
            for group in (target.compliance.requested, target.compliance.achieved)
            for name, _ in group
        ):
            raise ValueError(
                "SurfaceEnvelope.repair_provenance.metrics must be absent from the freshly certified adaptation target; got an already populated repair inventory"
            )
        compliance = MeshingComplianceReport(
            target.compliance.specification_id,
            issues=target.compliance.issues,
            requested=(
                *target.compliance.requested,
                *_surface_envelope_requested(policy),
            ),
            achieved=(*target.compliance.achieved, *_surface_envelope_achieved(evidence)),
        )
        provenance = SemanticProvenance(
            {
                "kind": "native-adapted-surface-envelope-volume",
                "source": source.result_id,
                "repair_provenance": source.provenance.semantic_id,
                "envelope": payload["envelope"],
                "preserved_carrier": payload["preserved_carrier"],
                "wrapping_evidence": asdict(evidence),
                "wrapping_policy": _surface_envelope_policy_payload(policy),
                "exact_repaired_volume": target.provenance.content_id,
            },
            resource_ids=target.provenance.resource_ids,
        )
        # The new JSON record survives the decoded input's temporary lifetime.
        budget.retain_owner(provenance)
        return CellMeshingResult(
            target.mesh,
            target.geometry,
            target.coordinate_contract,
            target.audit,
            target.quality,
            compliance,
            MeshingTrace((stage, *target.trace.stages), binding=target.trace.binding),
            target.provider,
            target.runtime,
            target.derivative_mode,
            provenance,
            boundary=target.boundary,
            patches=target.patches,
            zones=target.zones,
            labels=target.labels,
            attributes=target.attributes,
            associations=target.associations,
            adapter_reports=target.adapter_reports,
            certification=target.certification,
            region_evidence=target.region_evidence,
            region_boundary_evidence=target.region_boundary_evidence,
            surface_source=target.surface_source,
            collective_evidence=target.collective_evidence,
            storage_binding=target.storage_binding,
            collective_certificates=target.collective_certificates,
            execution_evidence=target.execution_evidence,
        )


__all__ = [
    "EnvelopeFeaturePermission",
    "EnvelopeTopologyPermission",
    "EnvelopeTopology",
    "PreparedSurfaceEnvelopeVolume",
    "RawTriangleSoup",
    "SurfaceEnvelope",
    "SurfaceEnvelopeEvidence",
    "SurfaceEnvelopePolicy",
    "envelope_volume_support_issues",
    "execute_surface_envelope_route",
    "generate_surface_envelope_volume",
    "surface_envelope_source_entities",
    "wrap_surface_envelope",
]
