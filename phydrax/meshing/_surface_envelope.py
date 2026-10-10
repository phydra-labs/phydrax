# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Declared unsigned-distance wrapping, never an inferred repair of a solid.

The repaired domain is the sublevel set of a native sampled distance field.
A soup remains a soup; the separately audited extracted surface receives new
identity. Two directed distance bounds and strict source inclusion are required
before a wrapping result can be published.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from enum import Enum
from fractions import Fraction
from time import monotonic
from typing import final, Literal, TYPE_CHECKING

import equinox as eqx
import numpy as np

from .. import _meshcore
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._identity import SemanticProvenance
from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._cell_geometry_validity import CellValidityPolicy
from ..geometry.surface._contracts import SurfaceAuditPolicy, SurfaceMetadata
from ..geometry.surface._model import SurfaceModel, SurfaceRealization
from ..typing import ConvertibleToArray, Dim, HostFloat64, HostInt64, parse, Scope
from ._contracts import (
    MeshingFailure,
    MeshingFailureCategory,
    MeshingProviderInfo,
    VolumeMeshingSpec,
)
from ._measurements import (
    measure_phase,
    NativeExecutionRecord,
    NativeMeshingPhaseRecorder,
    phase_started,
    record_elapsed,
)
from ._result import CellMeshingResult, MeshingComplianceReport
from ._trace import MeshingStageKind, MeshingStageReport, MeshingStageStatus, MeshingTrace
from ._volume_generation import (
    finalize_plc_volume,
    native_volume_checkpoint,
    native_volume_source_execution_budget,
    NativeVolumeSchedule,
    VolumeConstruction,
)


if TYPE_CHECKING:
    from .providers._native_sources import NativeSurfaceEnvelopeSource


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

    def __init__(
        self,
        vertices: ConvertibleToArray,
        triangles: ConvertibleToArray,
        metadata: SurfaceMetadata,
        /,
        *,
        feature_edges: ConvertibleToArray | None = None,
    ) -> None:
        if not isinstance(metadata, SurfaceMetadata):
            raise TypeError("metadata must be SurfaceMetadata.")
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
            raise TypeError("Triangle and feature indices must be integers.")
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
            raise ValueError("Soup coordinates must be finite.")
        if (
            np.any(faces < 0)
            or np.any(faces >= points.shape[0])
            or np.any(edges < 0)
            or np.any(edges >= points.shape[0])
        ):
            raise ValueError("Soup indices must name declared coordinate rows.")
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
                "Feature edges must be distinct noncollapsed represented triangle edges."
            )
        if metadata.cell_tags and len(metadata.cell_tags) != faces.shape[0]:
            raise ValueError("Soup cell tags must name every triangle occurrence.")
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
            raise ValueError("Envelope distances must be finite and positive.")
        radius, deviation, step, certificate = values
        interpolation_error = np.sqrt(3.0) * step + certificate
        if radius <= interpolation_error:
            raise ValueError(
                "Offset must exceed grid-cell diameter plus source covering radius for certified inclusion."
            )
        if radius + interpolation_error > deviation:
            raise ValueError(
                "Maximum deviation must cover offset plus sampling/interpolation error."
            )
        if not isinstance(
            feature_permission, EnvelopeFeaturePermission
        ) or not isinstance(topology_permission, EnvelopeTopologyPermission):
            raise TypeError(
                "Envelope feature and topology permissions must be explicit enums."
            )
        limits = (max_samples, max_work_units, max_vertices, max_triangles)
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in limits
        ):
            raise TypeError("Envelope budgets must be integers.")
        if any(value < 1 for value in limits):
            raise ValueError("Envelope budgets must be positive finite bounds.")
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
            raise TypeError("Envelope volume budgets must be integers.")
        limits = (*limits, volume_vertices, tetrahedra)
        if any(value < 1 or value > np.iinfo(np.int64).max for value in limits):
            raise ValueError("Envelope budgets must be positive signed-int64 bounds.")
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


def _degenerate_faces_exact(points: np.ndarray, faces: np.ndarray, /) -> int:
    """Count exact dyadic collinearity, including binary64 underflow cases."""
    exact_points = [
        (
            Fraction.from_float(float(row[0])),
            Fraction.from_float(float(row[1])),
            Fraction.from_float(float(row[2])),
        )
        for row in points
    ]
    degenerate = 0
    for face in faces:
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


def _topology(points: np.ndarray, faces: np.ndarray, /) -> EnvelopeTopology:
    edges: dict[tuple[int, int], list[int]] = {}
    neighbors: list[list[int]] = [[] for _ in range(faces.shape[0])]
    for face_index, face in enumerate(faces):
        for local in range(3):
            a, b = int(face[local]), int(face[(local + 1) % 3])
            edges.setdefault((min(a, b), max(a, b)), []).append(face_index)
    for incidents in edges.values():
        first = incidents[0]
        for other in incidents[1:]:
            neighbors[first].append(other)
            neighbors[other].append(first)
    visited: set[int] = set()
    components = 0
    for first in range(faces.shape[0]):
        if first in visited:
            continue
        components += 1
        pending = [first]
        visited.add(first)
        while pending:
            for other in neighbors[pending.pop()]:
                if other not in visited:
                    visited.add(other)
                    pending.append(other)
    used_vertices = np.unique(faces).size
    duplicates = faces.shape[0] - np.unique(np.sort(faces, axis=1), axis=0).shape[0]
    return EnvelopeTopology(
        used_vertices,
        len(edges),
        faces.shape[0],
        components,
        sum(len(rows) == 1 for rows in edges.values()),
        sum(len(rows) > 2 for rows in edges.values()),
        _degenerate_faces_exact(points, faces),
        duplicates,
        used_vertices - len(edges) + faces.shape[0],
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
        if (
            not isinstance(source, (RawTriangleSoup, SurfaceModel))
            or not isinstance(policy, SurfaceEnvelopePolicy)
            or not isinstance(repaired, SurfaceRealization)
            or not isinstance(evidence, SurfaceEnvelopeEvidence)
        ):
            raise TypeError(
                "SurfaceEnvelope requires typed source, policy, realization and evidence."
            )
        geometry_id = (
            source.soup_id if isinstance(source, RawTriangleSoup) else source.model_id
        )
        if (
            evidence.source_geometry_id != geometry_id
            or evidence.repaired_model_id != repaired.model.model_id
            or evidence.policy_id != policy.policy_id
        ):
            raise ValueError(
                "Envelope evidence is bound to different geometry or policy."
            )
        distances = (
            evidence.source_to_repaired_upper,
            evidence.repaired_to_source_upper,
            evidence.source_inclusion_margin,
            evidence.interpolation_error,
        )
        if any(not np.isfinite(value) or value < 0 for value in distances):
            raise ValueError("Envelope distance evidence must be finite and nonnegative.")
        if (evidence.source_id, evidence.source_revision) != (
            source.metadata.source_id,
            source.metadata.source_revision,
        ) or not repaired.policy.require_closed:
            raise ValueError(
                "Envelope evidence must preserve source identity and certify a closed repaired surface."
            )
        if (
            evidence.source_inclusion_margin <= 0
            or max(evidence.source_to_repaired_upper, evidence.repaired_to_source_upper)
            > policy.maximum_deviation
        ):
            raise ValueError("Envelope inclusion or deviation certificate is unmet.")
        scope = Scope()
        points = parse(
            np.array(carrier_vertices, dtype=np.float64, copy=True),
            HostFloat64[EnvelopeVolumeVertexDim, Literal[3]],
            "carrier_vertices",
            scope=scope,
        )
        raw_cells = np.asarray(carrier_tetrahedra)
        if not np.issubdtype(raw_cells.dtype, np.integer):
            raise TypeError("Envelope carrier tetrahedron indices must be integers.")
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
                "Envelope carrier must have finite valid indexed tetrahedra."
            )
        boundary_points = np.asarray(repaired.mesh.coordinates, dtype=np.float64)
        boundary_triangles = np.asarray(repaired.mesh.blocks[0].vertices, dtype=np.int64)
        if not np.array_equal(points[: boundary_points.shape[0]], boundary_points):
            raise ValueError(
                "Carrier boundary root prefix must exactly preserve repaired surface vertex identity."
            )
        corners = points[cells]
        if np.any(
            _meshcore.exact_orient3d(
                corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]
            )
            <= 0
        ):
            raise ValueError(
                "Retained envelope carrier tetrahedra must be strictly positive."
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
                "Retained envelope carrier must preserve every authoritative skin triangle and no other exterior face."
            )
        if (
            (points.shape[0], cells.shape[0])
            != (evidence.carrier_vertices, evidence.carrier_tetrahedra)
            or points.shape[0] > policy.max_volume_vertices
            or cells.shape[0] > policy.max_tetrahedra
        ):
            raise ValueError(
                "Envelope carrier count evidence or finite volume budget is unmet."
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
                    "Envelope execution evidence must be its exact native phase record."
                )
            execution_evidence.require_valid()
            if execution_evidence.owner_id != self.envelope_id:
                raise ValueError(
                    "Envelope execution evidence binds another original wrapping owner."
                )
        self.execution_evidence = execution_evidence


def wrap_surface_envelope(
    source: RawTriangleSoup | SurfaceModel,
    policy: SurfaceEnvelopePolicy,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> SurfaceEnvelope:
    """Native finite-budget distance thickening with two-sided certificates."""
    if not isinstance(source, (RawTriangleSoup, SurfaceModel)) or not isinstance(
        policy, SurfaceEnvelopePolicy
    ):
        raise TypeError("Wrapping requires its typed original source and policy.")
    active = _meshcore.current_native_execution_budget()
    if active is not None:
        workspace = _meshcore.current_native_host_workspace()
        with (
            active.host_workspace() if workspace is None else nullcontext(workspace)
        ) as storage:
            storage.retain_owner(source)
            envelope = _wrap_surface_envelope(source, policy, record_phase=record_phase)
            storage.retain_owner(envelope)
        active._prepared_envelope_sources.add(envelope.envelope_id)
        return envelope
    # The source policy declares work/sample bounds, but no wall or byte cap.
    # Capture their actual receipt without inventing finite source controls;
    # a later volume request must admit this completed preparation honestly.
    with _meshcore.NativeExecutionBudget(
        max_work=policy.max_work_units,
        max_geometry_queries=np.iinfo(np.uint64).max,
        max_cavity_cells=np.iinfo(np.uint64).max,
        max_scratch_bytes=np.iinfo(np.intp).max,
        max_wall_seconds=float("inf"),
    ) as budget:
        with budget.host_workspace() as storage:
            storage.retain_owner(source)
            envelope = _wrap_surface_envelope(source, policy, record_phase=record_phase)
            storage.retain_owner(envelope)
    if budget.evidence is None:
        raise RuntimeError(
            "Standalone wrapping lost its actual ended execution evidence."
        )
    receipt = NativeExecutionRecord(budget.evidence, owner_id=envelope.envelope_id)
    return eqx.tree_at(
        lambda value: value.execution_evidence,
        envelope,
        receipt,
        is_leaf=lambda value: value is None,
    )


def _wrap_surface_envelope(
    source: RawTriangleSoup | SurfaceModel,
    policy: SurfaceEnvelopePolicy,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> SurfaceEnvelope:
    """Run the represented wrapping theorem in its already admitted root."""
    if not isinstance(source, (RawTriangleSoup, SurfaceModel)):
        raise TypeError("source must explicitly be RawTriangleSoup or SurfaceModel.")
    if not isinstance(policy, SurfaceEnvelopePolicy):
        raise TypeError("policy must be SurfaceEnvelopePolicy.")
    if policy.feature_permission is EnvelopeFeaturePermission.PRESERVE:
        raise ValueError(
            "Unsigned-distance wrapping moves the represented surface; exact feature preservation is incompatible. Declare rounding permission."
        )
    if policy.topology_permission is EnvelopeTopologyPermission.PRESERVE:
        raise ValueError(
            "Unsigned-distance wrapping has no homeomorphism certificate; topology-change permission is required."
        )
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
        vertices, triangles, counts, bounds, carrier_vertices, carrier_tetrahedra = (
            _meshcore.surface_envelope(
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
        )
        if record_phase is not None:
            native_work = int(counts[3])
    finally:
        record_elapsed(
            record_phase,
            "envelope_construction",
            measurement_started,
            work_units=native_work,
        )
    if max(float(bounds[0]), float(bounds[1])) > policy.maximum_deviation:
        raise ValueError(
            "Two-sided envelope deviation could not be certified within the declared tolerance."
        )
    original_topology, repaired_topology = (
        _topology(points, faces),
        _topology(vertices, triangles),
    )
    revision = canonical_fingerprint(
        {
            "kind": "surface-envelope-revision",
            "source_geometry": source_geometry_id,
            "policy": policy.policy_id,
            "vertices": array_tree_fingerprint(vertices),
            "triangles": array_tree_fingerprint(triangles),
        }
    )
    metadata = SurfaceMetadata(
        source_id=canonical_fingerprint(
            {
                "kind": "repaired-envelope-source",
                "source": source.metadata.source_id,
                "source_geometry": source_geometry_id,
                "policy": policy.policy_id,
            }
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
    evidence_id = canonical_fingerprint(
        {
            "kind": "surface-envelope-evidence",
            "source": source_geometry_id,
            "repaired": model.model_id,
            "policy": policy.policy_id,
            "bounds": array_tree_fingerprint(bounds),
            "counts": array_tree_fingerprint(counts),
            "source_topology": asdict(original_topology),
            "repaired_topology": asdict(repaired_topology),
            "rounded_features": feature_count,
            "lost_selections": selection_ids,
            "lost_interfaces": interface_ids,
            "discarded_tags": source.metadata.cell_tags,
        }
    )
    evidence = SurfaceEnvelopeEvidence(
        source.metadata.source_id,
        source.metadata.source_revision,
        source_geometry_id,
        model.model_id,
        policy.policy_id,
        float(bounds[0]),
        float(bounds[1]),
        float(bounds[2]),
        float(bounds[3]),
        original_topology,
        repaired_topology,
        feature_count,
        selection_ids,
        interface_ids,
        source.metadata.cell_tags,
        int(counts[2]),
        int(counts[3]),
        int(counts[5]),
        int(counts[6]),
        int(counts[4]),
        evidence_id,
    )
    return SurfaceEnvelope(
        source, policy, realization, evidence, carrier_vertices, carrier_tetrahedra
    )


def envelope_volume_support_issues(
    source: NativeSurfaceEnvelopeSource, specification: VolumeMeshingSpec, /
) -> list[str]:
    from .providers._native_sources import NativeSurfaceEnvelopeSource
    from .providers._native_volume import volume_support_issues

    if not isinstance(source, NativeSurfaceEnvelopeSource) or not isinstance(
        specification, VolumeMeshingSpec
    ):
        raise TypeError(
            "Envelope volume admission requires typed source and specification."
        )
    issues = volume_support_issues(source.plc_source, specification)
    boundary = specification.boundary_scope
    if (boundary.source_id, boundary.source_revision) != (
        source.source_id,
        source.source_revision,
    ):
        issues.append(
            "scopes bound to the repaired envelope identity, not the original soup"
        )
    return issues


@final
class PreparedSurfaceEnvelopeVolume(StrictModule, NonTrainableState):
    schedule: NativeVolumeSchedule
    specification_id: str = eqx.field(static=True)
    source_binding_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: NativeSurfaceEnvelopeSource,
        specification: VolumeMeshingSpec,
        schedule: NativeVolumeSchedule,
        /,
    ) -> None:
        issues = envelope_volume_support_issues(source, specification)
        if issues:
            raise ValueError(
                "Unsupported envelope volume constraints: " + "; ".join(issues)
            )
        if not isinstance(schedule, NativeVolumeSchedule):
            raise TypeError("schedule must be NativeVolumeSchedule.")
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


def surface_envelope_source_entities(
    source: NativeSurfaceEnvelopeSource,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Authoritative input triangles, polygon IDs and PLC edge identity table.

    Both native carrier insertion and represented-source transfer consume this
    same namespace. No identity is inferred from coordinate proximity.
    """
    from .providers._native_sources import NativeSurfaceEnvelopeSource

    if not isinstance(source, NativeSurfaceEnvelopeSource):
        raise TypeError("source must be NativeSurfaceEnvelopeSource.")
    complex_ = source.plc_source.complex
    triangles = np.asarray(
        source.envelope.repaired.mesh.blocks[0].vertices, dtype=np.int64
    )
    from ._quad_generation import _family_host_array

    if complex_.polygon_offsets.size != triangles.shape[0] + 1:
        raise ValueError(
            "Envelope source facets differ from the authoritative extraction skin."
        )
    budget = _meshcore.current_native_execution_budget()
    if budget is not None:
        budget.charge(work=triangles.shape[0])
    for polygon, face in enumerate(triangles):
        first, last = complex_.polygon_offsets[polygon : polygon + 2]
        if last - first != 3 or not np.array_equal(
            complex_.polygon_vertices[first:last], face
        ):
            raise ValueError(
                "Envelope source facets differ from the authoritative extraction skin."
            )
    occurrences = _family_host_array((3 * triangles.shape[0], 2), np.int64)
    if budget is not None:
        budget.charge(work=occurrences.shape[0])
    for slot, (first, last) in enumerate(((0, 1), (1, 2), (2, 0))):
        rows = occurrences[slot * triangles.shape[0] : (slot + 1) * triangles.shape[0]]
        np.minimum(triangles[:, first], triangles[:, last], out=rows[:, 0])
        np.maximum(triangles[:, first], triangles[:, last], out=rows[:, 1])
    # Structured heapsort is in-place and has no data-sized sorting workspace.
    occurrences.view([("first", np.int64), ("last", np.int64)]).reshape(-1).sort(
        order=("first", "last"),
        kind="heapsort",
    )
    distinct = 1
    if budget is not None:
        budget.charge(work=occurrences.shape[0] - 1)
    for row in range(1, occurrences.shape[0]):
        if not np.array_equal(occurrences[row], occurrences[distinct - 1]):
            occurrences[distinct] = occurrences[row]
            distinct += 1
    edges = occurrences[:distinct]
    polygons = _family_host_array((triangles.shape[0],), np.int64)
    for polygon in range(polygons.size):
        polygons[polygon] = polygon
    return triangles, polygons, edges


def generate_surface_envelope_volume(
    source: NativeSurfaceEnvelopeSource,
    specification: VolumeMeshingSpec,
    schedule: NativeVolumeSchedule,
    /,
    *,
    validity_policy: CellValidityPolicy,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> VolumeConstruction:
    """Construct the retained carrier inside the original cumulative allowance."""
    from ._volume_generation import _bind_volume_execution

    receipt = source.envelope.execution_evidence
    with native_volume_source_execution_budget(
        specification.limits,
        receipt,
        source.envelope.envelope_id,
    ) as budget:
        workspace = _meshcore.current_native_host_workspace()
        with (
            budget.host_workspace() if workspace is None else nullcontext(workspace)
        ) as storage:
            storage.retain_owner(source)
            construction = _generate_surface_envelope_volume(
                source,
                specification,
                schedule,
                validity_policy=validity_policy,
                record_phase=record_phase,
            )
    if budget.evidence is None:
        return construction
    return _bind_volume_execution(
        construction,
        budget,
        0 if receipt is None else int(np.asarray(receipt.total_work_units)),
        0 if receipt is None else int(np.asarray(receipt.total_geometry_queries)),
        preparation_evidence=receipt,
    )


def _surface_envelope_carrier_source(
    source: NativeSurfaceEnvelopeSource,
    specification: VolumeMeshingSpec,
    triangles: np.ndarray,
    polygons: np.ndarray,
    edges: np.ndarray,
) -> _meshcore.TetMeshSourceComplex:
    """Declare extraction-skin rows and exact indexed endpoint witnesses."""
    from ._quad_generation import _family_host_array
    from ._volume_generation import _plc_source_rows

    points, faces, face_labels, face_bounds, segments, segment_ids, segment_bounds = (
        _plc_source_rows(
            source.plc_source.complex,
            specification,
            triangles,
            polygons,
            edges,
            np.empty((0, 2), dtype=np.int64),
        )
    )
    count = source.envelope.carrier_vertices.shape[0]
    strata = _family_host_array((count,), np.int8)
    entities = _family_host_array((count,), np.int32)
    parameters = _family_host_array((count, 2), np.float64)
    budget = _meshcore.current_native_execution_budget()
    if budget is not None:
        budget.charge(work=count + 2 * segments.shape[0])
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
        raise ValueError("Envelope extraction skin has an unowned source vertex.")
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
    storage = _meshcore.current_native_host_workspace()
    if storage is not None:
        storage.retain_owner(declared)
    return declared


def _generate_surface_envelope_volume(
    source: NativeSurfaceEnvelopeSource,
    specification: VolumeMeshingSpec,
    schedule: NativeVolumeSchedule,
    /,
    *,
    validity_policy: CellValidityPolicy,
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
            "Preserved envelope carrier exceeds the volume construction budget.",
            stage=MeshingStageKind.VOLUME_FILL.value,
            achieved=(
                ("vertices", envelope.carrier_vertices.shape[0]),
                ("tetrahedra", envelope.carrier_tetrahedra.shape[0]),
                ("work_units", evidence.work_units),
            ),
        )
    complex_ = source.plc_source.complex
    native_volume_checkpoint(limits, MeshingStageKind.VOLUME_FILL)
    triangles, polygons, edges = surface_envelope_source_entities(source)
    from ._quad_generation import _family_host_array

    regions = _family_host_array((envelope.carrier_tetrahedra.shape[0],), np.int32)
    edge_ids = _family_host_array((edges.shape[0],), np.int64)
    budget = _meshcore.current_native_execution_budget()
    if budget is not None:
        budget.charge(work=regions.size + edge_ids.size)
    regions.fill(0)
    for edge in range(edge_ids.size):
        edge_ids[edge] = edge
    native_volume_checkpoint(limits, MeshingStageKind.VOLUME_FILL)
    declared_source = _surface_envelope_carrier_source(
        source,
        specification,
        triangles,
        polygons,
        edges,
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
                "Envelope carrier construction exceeds its native scratch allocation allowance.",
                stage=MeshingStageKind.VOLUME_FILL.value,
                requested=(("maximum_scratch_bytes", limits.maximum_scratch_bytes),),
                provider_code=error.status.name,
            ) from error
        raise
    if state is None:
        raise RuntimeError("Native carrier construction returned no owned state.")
    stage = MeshingStageReport(
        MeshingStageKind.VOLUME_FILL,
        MeshingStageStatus.PASSED,
        input_ids=(envelope.envelope_id, prepared.prepared_id),
        output_ids=(envelope.carrier_id,),
        created_count=envelope.carrier_tetrahedra.shape[0],
    )
    return finalize_plc_volume(
        state,
        complex_,
        specification,
        schedule,
        validity_policy=validity_policy,
        source_id=source.plc_source.source_id,
        source_revision=source.plc_source.source_revision,
        input_id=prepared.prepared_id,
        input_triangles=triangles,
        input_polygons=polygons,
        plc_edges=edges,
        diagonal_polygons=np.empty((0,), dtype=np.int64),
        construction_stage=stage,
        source=declared_source,
        construction_counters=(
            ("grid_samples", evidence.grid_samples),
            ("certificate_samples", evidence.certificate_samples),
            ("carrier_vertices", evidence.carrier_vertices),
            ("carrier_tetrahedra", evidence.carrier_tetrahedra),
            ("work_units", evidence.work_units),
        ),
        construction_work_units=evidence.work_units,
        record_phase=record_phase,
    )


def execute_surface_envelope_route(
    source: NativeSurfaceEnvelopeSource,
    specification: VolumeMeshingSpec,
    prepared: PreparedSurfaceEnvelopeVolume,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> CellMeshingResult:
    """Keep construction, source certification and publication in one root."""
    from .providers._native_publication import bind_native_execution_result

    receipt = source.envelope.execution_evidence
    started = monotonic() - (
        0.0 if receipt is None else float(np.asarray(receipt.total_elapsed_seconds))
    )
    with native_volume_source_execution_budget(
        specification.limits,
        receipt,
        source.envelope.envelope_id,
    ) as budget:
        workspace = _meshcore.current_native_host_workspace()
        with (
            budget.host_workspace() if workspace is None else nullcontext(workspace)
        ) as storage:
            storage.retain_owner(source)
            result = _execute_surface_envelope_route(
                source,
                specification,
                prepared,
                coordinate_contract,
                provider,
                plan_id,
                record_phase=record_phase,
            )
    if budget.evidence is None:
        return result
    return bind_native_execution_result(
        result,
        budget.evidence,
        specification.limits,
        started,
        preparation_evidence=receipt,
    )


def _execute_surface_envelope_route(
    source: NativeSurfaceEnvelopeSource,
    specification: VolumeMeshingSpec,
    prepared: PreparedSurfaceEnvelopeVolume,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
) -> CellMeshingResult:
    """Publish canonical PLC acceptance while retaining original repair meaning."""
    from .providers._native_publication import check_deadline
    from .providers._native_volume import (
        execute_volume_route,
        plc_volume_audit_policy,
        PreparedPlcVolume,
    )

    started = monotonic()

    if (
        not isinstance(prepared, PreparedSurfaceEnvelopeVolume)
        or prepared.source_binding_id != source.binding_id
        or prepared.specification_id != specification.specification_id
    ):
        raise ValueError("Prepared envelope route binds another source or specification.")
    if (
        coordinate_contract.spatial_id
        != source.envelope.source.metadata.coordinate_contract.spatial_id
    ):
        raise ValueError(
            "Envelope volume coordinates must preserve the declared source units."
        )
    volume = PreparedPlcVolume(source.plc_source, specification, prepared.schedule)
    construction = _generate_surface_envelope_volume(
        source,
        specification,
        prepared.schedule,
        validity_policy=plc_volume_audit_policy().validity_policy,
        record_phase=record_phase,
    )
    check_deadline(started, specification.limits, MeshingStageKind.VOLUME_FILL)
    result = execute_volume_route(
        source.plc_source,
        specification,
        volume,
        coordinate_contract,
        provider,
        plan_id,
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
            ("envelope:maximum_deviation", source.envelope.policy.maximum_deviation),
            ("envelope:offset", source.envelope.policy.offset),
            ("envelope:minimum_inclusion_margin", 0.0),
        ),
        achieved=(
            *result.compliance.achieved,
            ("envelope:source_to_repaired_upper", evidence.source_to_repaired_upper),
            ("envelope:repaired_to_source_upper", evidence.repaired_to_source_upper),
            ("envelope:source_inclusion_margin", evidence.source_inclusion_margin),
            ("envelope:interpolation_error", evidence.interpolation_error),
            ("envelope:original_components", evidence.source_topology.components),
            ("envelope:repaired_components", evidence.repaired_topology.components),
            ("envelope:original_boundary_edges", evidence.source_topology.boundary_edges),
            (
                "envelope:repaired_boundary_edges",
                evidence.repaired_topology.boundary_edges,
            ),
            ("envelope:rounded_feature_edges", evidence.rounded_feature_edges),
            ("envelope:grid_samples", evidence.grid_samples),
            ("envelope:certificate_samples", evidence.certificate_samples),
            ("envelope:work_units", evidence.work_units),
            ("envelope:carrier_vertices", evidence.carrier_vertices),
            ("envelope:carrier_tetrahedra", evidence.carrier_tetrahedra),
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
            "wrapping_policy": {
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
            },
            "exact_repaired_volume": result.provenance.content_id,
        },
        resource_ids=result.provenance.resource_ids,
    )
    wrapping_stage = MeshingStageReport(
        MeshingStageKind.TOPOLOGY_REPAIR,
        MeshingStageStatus.PASSED,
        input_ids=(
            evidence.source_id,
            evidence.source_revision,
            evidence.source_geometry_id,
            evidence.policy_id,
        ),
        output_ids=(
            source.source_id,
            source.source_revision,
            evidence.repaired_model_id,
            evidence.evidence_id,
        ),
        created_count=evidence.repaired_topology.faces,
        deleted_count=evidence.source_topology.faces,
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
