#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"native PLC-restricted polyhedral publication.\n\nConstruction ancestry and numerical optimization diagnostics accompany the\nexisting CellMeshingResult. Independent affine validity, global embedding and\nmaterial-domain coverage remain mandatory; unresolved checks never publish.\nVEM/FV are explicit downstream consumers, not hidden allocations in generation.\n"

from __future__ import annotations

from dataclasses import asdict, replace
from fractions import Fraction
from math import prod
from typing import Any, assert_never, final, Literal

import equinox as eqx
import jax
import numpy as np

from phydrax.discretization import CellMesh, PolyhedralConnectivity
from phydrax.meshing._association_state import (
    GeometryAssociation,
    GeometryAssociationKind,
    GeometrySourceEntityRole,
)
from phydrax.meshing._certification_schedule import MeshCertificationSchedule
from phydrax.meshing._contracts import (
    MeshingLimits,
    MeshingProviderInfo,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing._failures import MeshingFailure, MeshingFailureCategory
from phydrax.meshing._host_banks import entity_key
from phydrax.typing import checked, parse

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._host_storage import declared_integer_storage_bytes
from ..._meshcore import (
    MeshcoreError,
    MeshcoreStatus,
    NativeExecutionScope,
    NativeHostStorageWorkspace,
    PlcRecoveryFailure,
    RestrictedPowerFailure,
)
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from ...discretization._coordinate_operands import CoordinateOperandCollector
from ...geometry._periodic_power_preparation import PeriodicPowerImageCapacityRefusal
from ...geometry._triangulation import PeriodicPowerSourceRefusal
from .._association_routes import AssociationPropagationError
from .._measurements import (
    measure_phase,
    NativeMeshingPhaseMeasurement,
    NativeMeshingPhaseRecorder,
    phase_started,
    record_elapsed,
)
from .._organization_records import MeshLabel, MeshPatch, MeshZone, MeshZoneRole
from .._plc_mapped_support import certify_mapped_plc_associations
from .._plc_source import PiecewiseLinearComplex
from .._polyhedral_generation import (
    generate_polyhedral_volume,
    NativePolyhedralSchedule,
    PolyhedralConstruction,
    PolyhedralGenerationError,
    require_polyhedral_connectivity,
    sqrt_upper,
)
from .._result import CellMeshingResult, MeshingComplianceReport
from .._scope import MeshingEntityKind
from .._sizing import UniformSizeControl
from .._trace import (
    MeshingDiagnostic,
    MeshingDiagnosticSeverity,
    MeshingStageKind,
    MeshingStageReport,
    MeshingStageStatus,
)
from .._volume_generation import (
    native_volume_checkpoint,
    prepare_plc_source,
    PreparedPlcSource,
    recovery_failure,
    region_seed_points,
)
from . import _native_sources as sources_owner, _native_volume as volume_owner
from ._native_publication import (
    coordinate_budget,
    coordinate_phase,
    NativeCertificationRequest,
    NativePublicationExecution,
    NativePublicationOrganization,
    publish_native_result,
)
from ._native_route_checks import (
    check_deadline,
    edge_size_evidence,
    uniform_size_compliance,
)
from ._native_route_metadata import mesh_entity_scope, NativeRouteMetadata
from ._native_route_policy import publication_policy


PowerConstructionPhase = Literal[
    "regular_triangulation",
    "power_adjacency",
    "power_clipping",
    "power_components",
    "power_publication",
]


@checked
def polyhedral_support_issues(
    source: sources_owner.NativePlcSource | sources_owner.NativePolyhedralSource,
    specification: VolumeMeshingSpec,
    /,
) -> list[str]:
    """Check structural support; numerical periodic proof uses the execution ledger."""
    issues = [
        issue
        for issue in volume_owner.volume_support_issues(source, specification)
        if issue
        not in (
            "a tetrahedral volume target",
            "affine tetrahedra in ambient dimension three",
            "the simplex fill strategy",
            "periodic constraints",
        )
    ]
    target = specification.target
    if set((*target.cell_families.required, *target.cell_families.preferred)) != {
        "polyhedron"
    }:
        issues.append("a polyhedral volume target")
    if (
        target.topological_dimension != 3
        or target.ambient_dimension != 3
        or target.geometry_order != 1
    ):
        issues.append("affine planar-face polyhedra in ambient dimension three")
    if specification.fill_strategy is not VolumeFillStrategy.POLYHEDRAL:
        issues.append("the polyhedral fill strategy")
    for constraint in specification.periodic_constraints:
        for scope in (constraint.source_scope, constraint.target_scope):
            identifiers = np.asarray(scope.entity_ids)
            if (
                scope.entity_dimension != 2
                or scope.entity_kind is not MeshingEntityKind.GEOMETRY
                or scope.source_id != source.source_id
                or scope.source_revision != source.source_revision
                or np.any(identifiers < 0)
                or np.any(identifiers >= source.complex.facet_count)
            ):
                issues.append("periodic scopes bound to original PLC facets")
    # Exact commuting-action admission belongs to generation's actual request
    # ledger. A standalone preparation ticket neither enforces that request nor
    # earns reusable preparation evidence.
    return issues


@final
class PreparedPolyhedralVolume(StrictModule, NonTrainableState):
    """Bound source, specification, sites and the numerical refinement policy."""

    schedule: NativePolyhedralSchedule = eqx.field(static=True)
    sites: np.ndarray | None
    weights: np.ndarray | None
    source_binding_id: str = eqx.field(static=True)
    specification_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source: sources_owner.NativePlcSource | sources_owner.NativePolyhedralSource,
        specification: VolumeMeshingSpec,
        schedule: NativePolyhedralSchedule | None = None,
        /,
    ) -> None:
        issues = polyhedral_support_issues(source, specification)
        if issues:
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                "; ".join(issues),
                stage="preparation",
            )
        policy = NativePolyhedralSchedule() if schedule is None else schedule
        limits = specification.limits
        control = specification.size_controls[0]
        match control:
            case UniformSizeControl():
                pass
            case _:
                raise TypeError(
                    "polyhedral.size_controls must contain a uniform whole-domain control; got a different control kind"
                )
        maximum = (
            control.maximum_size
            if control.maximum_size is not None
            else control.target_size
        )
        # A hard upper edge-size request is enforced by the conservative whole
        # cell diameter bound. No request is weakened to site spacing.
        policy = replace(
            policy,
            maximum_vertices=min(policy.maximum_vertices, limits.maximum_vertices),
            maximum_cells=min(policy.maximum_cells, limits.maximum_cells),
            maximum_work_units=min(policy.maximum_work_units, limits.maximum_work_units),
            maximum_scratch_bytes=min(
                policy.maximum_scratch_bytes, limits.maximum_scratch_bytes
            ),
            maximum_cell_diameter=maximum
            if policy.maximum_cell_diameter is None
            else min(maximum, policy.maximum_cell_diameter),
        )
        self.schedule = policy
        self.sites = (
            source.sites
            if isinstance(source, sources_owner.NativePolyhedralSource)
            else None
        )
        self.weights = (
            source.weights
            if isinstance(source, sources_owner.NativePolyhedralSource)
            else None
        )
        self.source_binding_id = source.binding_id
        self.specification_id = specification.specification_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "native-polyhedral-prepared",
                "source": source.binding_id,
                "specification": specification.specification_id,
                "schedule": asdict(policy),
                "sites": array_tree_fingerprint((self.sites, self.weights)),
            }
        )


def _protected_vertices(specification: VolumeMeshingSpec) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                int(vertex)
                for feature in specification.protected_features
                if feature.scope.entity_dimension == 0
                for vertex in np.asarray(feature.scope.entity_ids)
            }
        )
    )


def _polyhedral_construction_inputs(
    source: sources_owner.NativePlcSource | sources_owner.NativePolyhedralSource,
    specification: VolumeMeshingSpec,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Admit the seed stack and protected-vertex set before materializing them."""
    seeds = len(specification.region_seeds) + len(specification.hole_seeds)
    protected = sum(
        feature.scope.entity_ids.size
        for feature in specification.protected_features
        if feature.scope.entity_dimension == 0
    )
    integer = declared_integer_storage_bytes(
        max(source.complex.vertices.shape[0], 1).bit_length()
    )
    output = 4 * 256 + 3 * 16 + 28 * seeds + (16 + integer) * protected
    # Two seed pointer lists, borrowed coordinate views, stack and astype
    # copies; protected set, sorted list and final tuple coexist at conversion.
    scratch = 4 * 256 + seeds * (2 * 16 + 256 + 2 * 24 + integer)
    scratch += 3 * (256 + protected * (64 + integer))
    with budget.temporary_scope(), budget.live_storage() as storage:
        storage.set_bound(output, work=seeds + protected)
        budget.reserve(0, scratch)
        points, regions = region_seed_points(source.complex, specification)
        result = points, regions, _protected_vertices(specification)
        budget.retain_owner(result, temporary_storage_credit=storage)
        return result


def _vertex_association(
    source: sources_owner.NativePlcSource | sources_owner.NativePolyhedralSource,
    specification: VolumeMeshingSpec | None,
    construction: PolyhedralConstruction,
    selected_faces: np.ndarray,
    face_residuals: np.ndarray,
) -> GeometryAssociation:

    mesh, c = construction.mesh, require_polyhedral_connectivity(construction.mesh)
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    carriers = construction.diagram.construction.vertex_carriers
    original_count = source.complex.vertices.shape[0]
    dimensions = np.full(points.shape[0], 3, dtype=np.int64)
    sources = np.full(points.shape[0], -1, dtype=np.int64)
    residuals = np.zeros(points.shape[0], dtype=np.float64)
    ambiguous = np.zeros(points.shape[0], dtype=np.bool_)
    original_rows: dict[int, int] = {}
    for vertex, carrier in enumerate(carriers):
        carrier = np.asarray(carrier, dtype=np.int32)
        indices = carrier[carrier >= 0]
        if indices.size == 1 and indices[0] < original_count:
            index = int(indices[0])
            dimensions[vertex], sources[vertex] = 0, index
            original_rows[index] = vertex
            squared = sum(
                (
                    (Fraction(float(a)) - Fraction(float(b))) ** 2
                    for a, b in zip(
                        points[vertex], source.complex.vertices[index], strict=True
                    )
                ),
                Fraction(0),
            )
            residuals[vertex] = sqrt_upper(squared)

    def assign(
        rows: np.ndarray, dimension: int, source_index: int, residual: float
    ) -> None:
        same = (dimensions[rows] == dimension) & (sources[rows] >= 0)
        ambiguous[rows[same & (sources[rows] != source_index)]] = True
        matches = rows[same & (sources[rows] == source_index)]
        residuals[matches] = np.maximum(residuals[matches], residual)
        replace = rows[(dimensions[rows] > dimension) | (sources[rows] < 0)]
        dimensions[replace], sources[replace] = dimension, source_index
        residuals[replace], ambiguous[replace] = residual, False

    offsets, vertices = (
        np.asarray(c.cell_vertex_offsets),
        np.asarray(c.cell_vertex_values),
    )
    for cell, (a, b) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        assign(vertices[a:b], 3, int(construction.cell_regions[cell]), 0.0)
    offsets, vertices = (
        np.asarray(c.face_vertex_offsets),
        np.asarray(c.face_vertex_values),
    )
    for face, residual in zip(selected_faces, face_residuals, strict=True):
        a, b = offsets[face : face + 2]
        assign(vertices[a:b], 2, int(construction.face_facets[face]), float(residual))
    edges = np.asarray(c.edges, dtype=np.int64)
    for edge in np.flatnonzero(construction.feature_edge_sources >= 0):
        source_edge = int(construction.feature_edge_sources[edge])
        assign(
            edges[edge],
            1,
            source_edge,
            float(construction.feature_maximum_distance[source_edge]),
        )
    missing = tuple(
        vertex
        for vertex in (
            () if specification is None else _protected_vertices(specification)
        )
        if vertex not in original_rows
    )
    if missing:
        raise MeshingFailure(
            MeshingFailureCategory.AUDIT_FAILED,
            "polyhedral.protected_vertices must retain source carrier ancestry; got missing vertices",
            stage="geometry_association",
            entity_ids=missing,
        )
    for feature in () if specification is None else specification.protected_features:
        if feature.scope.entity_dimension == 0:
            bad = tuple(
                int(vertex)
                for vertex in np.asarray(feature.scope.entity_ids)
                if residuals[original_rows[int(vertex)]] > feature.maximum_deviation
            )
            if bad:
                raise MeshingFailure(
                    MeshingFailureCategory.AUDIT_FAILED,
                    "polyhedral.protected_vertices.displacement must not exceed maximum_deviation; got excessive displacement",
                    stage="geometry_association",
                    entity_ids=bad,
                )
    unresolved = ambiguous | (sources < 0)
    if np.any(unresolved):
        raise MeshingFailure(
            MeshingFailureCategory.AUDIT_FAILED,
            "polyhedral.vertex.source_ancestry must identify one source stratum; got missing or ambiguous ancestry",
            stage="geometry_association",
            entity_ids=tuple(map(int, np.asarray(c.vertex_global_ids)[unresolved])),
        )
    # These dimensions were assigned from the validated volume PLC carriers:
    # source points, curve edges, surface facets, and material regions.
    roles = (
        GeometrySourceEntityRole.VERTEX,
        GeometrySourceEntityRole.EDGE,
        GeometrySourceEntityRole.FACET,
        GeometrySourceEntityRole.REGION,
    )
    return GeometryAssociation(
        GeometryAssociationKind.PIECEWISE_LINEAR,
        source.source_id,
        source.source_revision,
        mesh.entity_set(0).entity_set_id,
        c.vertex_global_ids,
        tuple(
            entity_key(source.source_revision, roles[int(dimension)].value, int(index))
            for dimension, index in zip(dimensions, sources, strict=True)
        ),
        residuals,
        exact=False,
        source_dimensions=dimensions,
        source_indices=sources,
        source_entity_roles=tuple(roles[int(dimension)] for dimension in dimensions),
    )


def _polyhedral_organization_bounds(
    source: sources_owner.NativePlcSource | sources_owner.NativePolyhedralSource,
    specification: VolumeMeshingSpec | None,
    construction: PolyhedralConstruction,
) -> tuple[int, int, int]:
    """Declared host bounds from bank shapes, not process/device memory estimates.

    Association rows have ten numeric banks, a dictionary entry triple and
    expanded name/role/path columns. All four strata are bounded by their full
    mesh entity counts, including rows that subsequently prove interior. Scope
    memberships include every requested patch, even overlapping patches.
    """
    mesh = construction.mesh
    c = require_polyhedral_connectivity(mesh)
    vertices, edges, faces, cells = (
        mesh.coordinates.shape[0], c.edge_count, c.face_count, c.cell_count
    )
    rows = vertices + edges + faces + cells
    patch_controls = () if specification is None else specification.patch_controls
    records = len(source.complex.region_ids) + source.complex.facet_count
    records += len(patch_controls) + 4
    memberships = cells + (4 + len(patch_controls)) * faces + edges
    # entity_key is revision:role:index. Decimal index width is bounded by the
    # concrete source bank lengths; no expanded association array is built here.
    index_width = len(str(max(
        source.complex.vertices.shape[0], source.complex.segments.shape[0],
        construction.feature_source_edges.shape[0], source.complex.facet_count,
        len(source.complex.region_ids), 1,
    )))
    key_width = len(source.source_revision) + len("region") + index_width + 2
    names = sum(len(name) for name in source.complex.region_ids)
    names += sum(
        len(control.name) + sum(map(len, control.adjacent_region_names))
        for control in patch_controls
    )
    if specification is not None:
        for values in (specification.region_seeds, specification.region_controls):
            names += sum(
                len(value.material_id) if value.material_id is not None else 0
                for value in values
            )
    # Each record and its scope have at most 32 fixed scalar/string/container
    # owners (256-byte declared metadata, 64-character digest text); memberships
    # allow an int64 bank plus its logical array owner. Association entry triples
    # allow three container owners and generated key text per row, without
    # relying on interning or dictionary deduplication.
    record_storage = records * 32 * (256 + 4 * 64)
    scope_storage = 2 * (256 * records + 8 * memberships)
    association_storage = 4 * (16 * 256 + 10 * 256)
    association_storage += rows * (
        10 * 8 + 3 * (256 + 3 * 16) + 256 + 4 * key_width
    )
    output = (
        256 + 16 * records + record_storage + scope_storage
        + association_storage + 4 * names
        + 4 * (256 + 4 * (len(source.source_id) + len(source.source_revision)))
    )
    # Host coordinate/connectivity copies; per-row masks, index selections and
    # numeric association validation banks; edge lookup dict/tuple/int owners.
    incidence = c.face_vertex_values.size + c.cell_vertex_values.size
    bank_copies = 24 * vertices + 16 * edges + 8 * (cells + faces + incidence)
    numeric_scratch = 4 * (10 * 256 + 10 * 8 * rows)
    # At most eight loop coordinate triples coexist in gathered, rolled,
    # difference, cross-product and normal-reduction banks.
    numeric_scratch += 16 * memberships + 8 * 24 * incidence
    integer = declared_integer_storage_bytes(max(vertices, edges, 1).bit_length())
    feature_entries = 0 if specification is None else sum(
        feature.scope.entity_ids.size for feature in specification.protected_features
    )
    # Protected-source ID expansion can exceed a single mesh stratum when
    # controls overlap. Bound the declared entries, not only distinct IDs.
    numeric_scratch += feature_entries * (3 * 8 + 64 + 256 + integer)
    edge_lookup = 256 + edges * (64 + 256 + 2 * 16 + 3 * integer)
    vertex_lookup = 256 + vertices * (64 + 2 * integer)
    carrier_width = construction.diagram.construction.vertex_carriers.shape[1]
    numeric_scratch += 3 * 256 + (4 + 1 + 4) * carrier_width
    # Constructor normalization, role expansion and fingerprint text use the
    # same Unicode/UTF-8 buffer allowance as association reconstruction.
    text_scratch = 24 * rows * (2 * key_width + len("region") + 16)
    # Vertex-carrier displacement compares three binary64 components exactly.
    # Binary64 numerator/denominator widths are <=1075; differences, squares and
    # their three-term sum fit eight such widths. At most 32 scalar fractions
    # (metadata plus numerator and denominator) coexist during that expression.
    rational_scratch = 32 * (
        256 + 2 * declared_integer_storage_bytes(8 * 1075)
    )
    scratch = (
        bank_copies + numeric_scratch + edge_lookup + vertex_lookup + text_scratch
        + rational_scratch + 32 * 256
    )
    work = rows + incidence + memberships + edges + feature_entries
    work += cells * len(source.complex.region_ids)
    work += faces * (source.complex.facet_count + len(patch_controls))
    return output, scratch, work


@checked
def organization(
    source: sources_owner.NativePlcSource | sources_owner.NativePolyhedralSource,
    specification: VolumeMeshingSpec | None,
    construction: PolyhedralConstruction,
    tolerance: float,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> NativeRouteMetadata:
    """Build source rows after admitting their concrete topology and text bounds."""
    if budget is None:
        return _organization(source, specification, construction, tolerance)
    output, scratch, work = _polyhedral_organization_bounds(
        source, specification, construction
    )
    with budget.temporary_scope(), budget.live_storage() as storage:
        storage.set_bound(output, work=work)
        budget.reserve(0, scratch)
        result = _organization(source, specification, construction, tolerance)
        budget.retain_owner(result, temporary_storage_credit=storage)
        return result


def _organization(
    source: sources_owner.NativePlcSource | sources_owner.NativePolyhedralSource,
    specification: VolumeMeshingSpec | None,
    construction: PolyhedralConstruction,
    tolerance: float,
) -> NativeRouteMetadata:
    mesh, complex_ = construction.mesh, source.complex
    connectivity = require_polyhedral_connectivity(mesh)
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    cell_ids = np.asarray(connectivity.cell_global_ids, dtype=np.int64)
    face_ids = np.asarray(connectivity.face_global_ids, dtype=np.int64)
    edge_ids = np.asarray(connectivity.edge_global_ids, dtype=np.int64)

    seeds = (
        {}
        if specification is None
        else {seed.region_name: seed for seed in specification.region_seeds}
    )
    controls = (
        {}
        if specification is None
        else {control.region_name: control for control in specification.region_controls}
    )
    zones = tuple(
        MeshZone(
            name,
            MeshZoneRole.REGION,
            mesh_entity_scope(
                mesh, 3, np.sort(cell_ids[construction.cell_regions == region])
            ),
            material_id=(
                controls[name].material_id
                if name in controls
                else seeds[name].material_id
                if name in seeds
                else None
            ),
            region_role=(
                controls[name].role
                if name in controls
                else seeds[name].role
                if name in seeds
                else None
            ),
        )
        for region, name in enumerate(complex_.region_ids)
    )
    facets = construction.face_facets
    selected = np.flatnonzero(facets >= 0)
    patches = tuple(
        MeshPatch(
            f"facet:{facet}",
            mesh_entity_scope(mesh, 2, np.sort(face_ids[facets == facet])),
        )
        for facet in np.unique(facets[selected])
    )
    zone_ids = {zone.name: zone.zone_id for zone in zones}
    for control in () if specification is None else specification.patch_controls:
        rows = face_ids[np.isin(facets, np.asarray(control.scope.entity_ids))]
        if control.required and not rows.size:
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "polyhedral.patch_controls must publish face fragments for every required patch; got an empty patch",
                stage="geometry_association",
                entity_ids=tuple(map(int, np.asarray(control.scope.entity_ids))),
            )
        if rows.size:
            patches += (
                MeshPatch(
                    control.name,
                    mesh_entity_scope(mesh, 2, np.sort(rows)),
                    adjacent_zone_ids=tuple(
                        zone_ids[name] for name in control.adjacent_region_names
                    ),
                ),
            )
    labels = tuple(
        MeshLabel(name, mesh_entity_scope(mesh, dimension, np.sort(rows)))
        for name, dimension, rows in (
            ("interface", 2, face_ids[construction.interface_faces]),
            ("internal_sheet", 2, face_ids[construction.sheet_faces]),
            (
                "internal_curve",
                1,
                edge_ids[
                    (construction.feature_edge_sources >= 0)
                    & (construction.feature_edge_sources < complex_.segments.shape[0])
                ],
            ),
        )
        if rows.size
    )
    if construction.seam_face_pairs is not None:
        paired = np.unique(construction.seam_face_pairs[:, :2])
        labels += (
            MeshLabel("periodic", mesh_entity_scope(mesh, 2, np.sort(face_ids[paired]))),
        )
    phase = _polyhedral_associations(
        source,
        specification,
        construction,
        tolerance,
        points,
        cell_ids,
        face_ids,
        edge_ids,
        facets,
        selected,
    )
    associations = phase.associations
    return NativeRouteMetadata(zones, patches, labels, associations)


def _native_resource_quantities(error: MeshcoreError, /) -> tuple[tuple[str, float], ...]:
    quantities = [("native_status", float(error.status))]
    if error.work_counters is not None:
        quantities.extend(
            (f"native_execution:{name}", float(value))
            for name, value in error.work_counters.items()
        )
    if error.memory_counters is not None:
        quantities.extend(
            (f"native_memory:{name}", float(value))
            for name, value in error.memory_counters.items()
        )
    axes = ("work_units", "geometry_queries", "cavity_cells", "scratch_bytes")
    for prefix, values in (
        ("native_requested", error.batch_requested_upper),
        ("native_admitted", error.batch_admitted_upper),
    ):
        if values is not None:
            quantities.extend(
                (f"{prefix}:{axis}", float(value))
                for axis, value in zip(axes, values, strict=True)
            )
    if error.batch_refusal_evidence is not None:
        quantities.extend(
            (f"native_remaining:{axis}", float(value))
            for axis, value in zip(
                axes, error.batch_refusal_evidence[:4], strict=True
            )
        )
        quantities.extend(
            (f"native_refusal_slot:{index}", float(value))
            for index, value in enumerate(error.batch_refusal_evidence[4:], start=4)
        )
    if error.elapsed_seconds is not None:
        quantities.append(("native_elapsed_seconds", error.elapsed_seconds))
    if error.maximum_seconds is not None:
        quantities.append(("native_maximum_seconds", error.maximum_seconds))
    return tuple(quantities)


def _periodic_source_failure(
    error: PeriodicPowerSourceRefusal, stage: str, /
) -> MeshingFailure:
    native = construction_failure(error.native_refusal, stage).evidence
    quantities = list(native.achieved)
    if error.requested_work is not None:
        quantities.append(("periodic_requested_work_units", float(error.requested_work)))
    if error.completed_work is not None:
        quantities.append(("periodic_completed_work_units", float(error.completed_work)))
    return MeshingFailure(
        native.category,
        f"{native.message}; periodic source={error.preparation.source_id}",
        provider_code=native.provider_code,
        stage=native.stage,
        entity_ids=native.entity_ids,
        locations=native.locations,
        requested=(
            *native.requested,
            ("periodic_call_work_units", float(error.maximum_work)),
        ),
        achieved=tuple(quantities),
        checkpoint_id=native.checkpoint_id,
        logical_findings=native.logical_findings,
    )


@checked
def construction_failure(
    error: MeshcoreError | PolyhedralGenerationError | CoordinateEnclosureResourceError,
    stage: str,
    /,
) -> MeshingFailure:
    """Preserve native resource/scientific status at generation and adaptation."""
    match error:
        case CoordinateEnclosureResourceError():
            return MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                str(error),
                stage=stage,
                requested=((error.resource, float(error.limit)),),
                achieved=(
                    ("requested", float(error.requested)),
                    ("completed", float(error.completed)),
                ),
            )
        case PeriodicPowerImageCapacityRefusal():
            return MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                str(error),
                stage=stage,
                requested=(("site_images", float(error.maximum_images)),),
                achieved=(
                    *_native_resource_quantities(error),
                    ("requested_site_images", float(error.requested_images)),
                    ("completed_site_images", float(error.completed_images)),
                ),
            )
        case PeriodicPowerSourceRefusal():
            return _periodic_source_failure(error, stage)
        case MeshcoreError() if error.status is MeshcoreStatus.TIMEOUT:
            return MeshingFailure(
                MeshingFailureCategory.TIMED_OUT,
                str(error),
                stage=stage,
                achieved=_native_resource_quantities(error),
            )
        case PlcRecoveryFailure():
            return recovery_failure(error)
        case RestrictedPowerFailure():
            return MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED
                if error.status == MeshcoreStatus.CAPACITY_EXCEEDED
                else MeshingFailureCategory.AUDIT_FAILED,
                str(error),
                stage=stage,
                entity_ids=tuple(
                    int(value) for value in error.failure[1:3] if value >= 0
                ),
                requested=() if error.capacity is None else (
                    (f"maximum_{error.capacity.axis}", error.capacity.remaining),
                ),
                achieved=(
                    *_native_resource_quantities(error),
                    *tuple(
                        (f"power_failure_{index}", float(value))
                        for index, value in enumerate(error.failure)
                    ),
                    *tuple(
                        (f"power_counter_{index}", float(value))
                        for index, (_, value) in enumerate(error.counters.items())
                    ),
                    *tuple(
                        (f"native_preflight:{name}", float(value))
                        for name, value in error.preflight.items()
                    ),
                    *(() if error.capacity is None else (
                        (f"required_{error.capacity.axis}", error.capacity.required),
                        (f"remaining_{error.capacity.axis}", error.capacity.remaining),
                    )),
                ),
            )
        case PolyhedralGenerationError():
            match error.achieved:
                case dict():
                    quantities = tuple(
                        (str(name), float(value))
                        for name, value in error.achieved.items()
                    )
                case np.ndarray():
                    quantities = tuple(
                        (f"unmet_entity_{index}", float(value))
                        for index, value in enumerate(error.achieved.flat)
                    )
                case int() | float():
                    quantities = (("consumed_work", float(error.achieved)),)
                case None:
                    quantities = ()
                case invalid:
                    assert_never(invalid)
            return MeshingFailure(
                error.reason.failure_category,
                str(error),
                stage=stage,
                entity_ids=error.entities,
                achieved=quantities,
            )
        case MeshcoreError():
            return MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED
                if error.status == MeshcoreStatus.CAPACITY_EXCEEDED
                else MeshingFailureCategory.AUDIT_FAILED,
                str(error),
                stage=stage,
                achieved=_native_resource_quantities(error),
            )
        case _:
            raise TypeError(
                f"polyhedral.error must carry a native construction refusal; got {type(error).__name__}"
            )


def _execute_polyhedral_route(
    source: sources_owner.NativePlcSource | sources_owner.NativePolyhedralSource,
    specification: VolumeMeshingSpec,
    prepared: PreparedPolyhedralVolume,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    scope: NativeExecutionScope,
    workspace: NativeHostStorageWorkspace,
    coordinate_operands: CoordinateOperandCollector | None = None,
) -> CellMeshingResult:
    "Construct, independently certify, and publish the result."
    if (
        prepared.source_binding_id != source.binding_id
        or prepared.specification_id != specification.specification_id
    ):
        raise ValueError(
            "PreparedPolyhedralVolume.binding must match the source and specification IDs; got mismatched IDs"
        )
    started = scope.operation_started
    # Construction and publication share one request ledger, so publication
    # reuses source witnesses this request already earned and retained.
    budget = coordinate_budget(specification.limits)
    budget.capture_coordinate_operands = coordinate_operands is not None
    budget.capture_polynomial_operands = (
        coordinate_operands is not None and coordinate_operands.capture_polynomials
    )
    budget.capture_signed_determinant_operands = (
        coordinate_operands is not None and coordinate_operands.capture_signed_determinants
    )
    with budget.retained_storage(lambda _size: scope.ticket_view()):
        with budget.temporary_scope():
            with budget.bind_ticket(scope.ticket_view()):
                # Route wrappers borrow the scientific owners admitted below.
                budget.reserve(
                    0, 128 * (256 + 4 * 64)
                    + 4 * (
                        len(source.source_id) + len(source.source_revision) + len(plan_id)
                    ),
                )
            result = _publish_polyhedral_route(
                source,
                specification,
                prepared,
                coordinate_contract,
                provider,
                plan_id,
                _construct_polyhedral_route(
                    source, specification, prepared, started, record_phase, scope, budget
                ),
                record_phase=record_phase,
                scope=scope,
                workspace=workspace,
                budget=budget,
            )
            if coordinate_operands is not None:
                with coordinate_phase(scope, specification.limits, budget):
                    coordinate_operands.capture(
                        result.mesh, result.geometry, budget, scope,
                        input_owners=(source, specification, prepared),
                        coordinate_contract=result.coordinate_contract,
                    )
        # Retire route scratch before transferring the still-live retained
        # reservation. The old ledger cannot mutate it after this point.
        budget.handover_retained_storage(workspace.adopt_retained_storage)
        return result


def _publish_polyhedral_route(
    source: sources_owner.NativePlcSource | sources_owner.NativePolyhedralSource,
    specification: VolumeMeshingSpec,
    prepared: PreparedPolyhedralVolume,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    constructed: _PolyhedralConstructionPhase,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None,
    scope: NativeExecutionScope,
    workspace: NativeHostStorageWorkspace,
    budget: CoordinateEnclosureBudget,
) -> CellMeshingResult:
    "Associate, independently certify, and publish one constructed route."
    started = scope.operation_started
    construction, mesh = constructed.construction, constructed.mesh
    with budget.bind_ticket(scope.ticket_view()):
        compliance = _polyhedral_compliance(
            construction, specification, record_phase, budget=budget
        ).compliance
    with measure_phase(record_phase, "geometry_association"):
        with budget.bind_ticket(scope.ticket_view()):
            metadata = organization(
                source, specification, construction, prepared.schedule.feature_tolerance,
                budget=budget,
            )
        zones = metadata.zones
        patches = metadata.patches
        labels = metadata.labels
        certified = certify_polyhedral_associations(
            source,
            coordinate_contract,
            construction,
            metadata.associations,
            NativeCertificationRequest(
                MeshCertificationSchedule("volume_plc"),
                source.source_id,
                source.source_revision,
                specification.limits,
                domain=construction.domain,
                cell_regions=construction.cell_regions,
            ),
            record_phase=record_phase,
            scope=scope,
            budget=budget,
        )
        associations = certified.associations
    with budget.bind_ticket(scope.ticket_view()):
        stages = _polyhedral_stage_reports(source, prepared, construction, budget=budget)
    with budget.bind_ticket(scope.ticket_view()):
        provenance = _polyhedral_provenance(
            source, prepared, plan_id, construction, budget=budget
        )
    check_deadline(started, specification.limits, MeshingStageKind.GEOMETRY_AUDIT)
    return publish_native_result(
        mesh,
        coordinate_contract,
        compliance,
        stages,
        provider,
        provenance,
        certified.request,
        policy=publication_policy("polyhedral"),
        organization=NativePublicationOrganization(
            patches=patches, zones=zones, labels=labels, associations=associations
        ),
        execution=NativePublicationExecution(
            scope=scope, workspace=workspace, record_phase=record_phase, budget=budget
        ),
        geometry=construction.geometry,
        covered_coordinate_storage=True,
    )


def _polyhedral_stage_reports(
    source: sources_owner.NativePlcSource | sources_owner.NativePolyhedralSource,
    prepared: PreparedPolyhedralVolume,
    construction: PolyhedralConstruction,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[MeshingStageReport, ...]:
    """Own the fixed report records and concrete native-counter diagnostics."""
    data = construction.diagram.construction
    # The canonical field bank already exists on the counter record's type.
    # items() constructs tuples, so defer that expansion until after admission.
    count = len(type(data.counters).__dataclass_fields__)
    key_width = len("power_counter_") + len(str(max(count, 1)))
    # Two stage records, one diagnostic, their enums, IDs and tuple fields fit
    # 64 fixed metadata/string owners. Counter rows add a triple, generated
    # name and two boxed float values each.
    output = 64 * (256 + 4 * 64)
    output += count * (256 + 3 * 16 + 256 + 4 * key_width + 2 * 256)
    with budget.temporary_scope(), budget.live_storage() as storage:
        storage.set_bound(output, work=count)
        # Input/normalized report tuples coexist during construction; UTF-8 and
        # Unicode fingerprint buffers use the declared text-width allowance.
        budget.reserve(0, output + 24 * (1024 + count * (key_width + 64)))
        diagnostics = (
            MeshingDiagnostic(
                MeshingDiagnosticSeverity.INFO,
                "Reciprocal restricted components; non-star-visible sites decomposed into convex pieces. Site ancestry is not a field interpolation stencil.",
                quantities=tuple(
                    (f"power_counter_{i}", float(x), float(x))
                    for i, (_, x) in enumerate(data.counters.items())
                ),
            ),
        )
        stages = (
            MeshingStageReport(
                MeshingStageKind.SOURCE_INSPECTION,
                MeshingStageStatus.PASSED,
                input_ids=(source.binding_id,),
                output_ids=(prepared.prepared_id,),
            ),
            MeshingStageReport(
                MeshingStageKind.VOLUME_FILL,
                MeshingStageStatus.PASSED,
                input_ids=(prepared.prepared_id,),
                output_ids=(construction.construction_id, construction.mesh.mesh_id),
                created_count=require_polyhedral_connectivity(construction.mesh).cell_count,
                diagnostics=diagnostics,
            ),
        )
        budget.retain_owner(stages, temporary_storage_credit=storage)
        return stages


def _tolist_storage_upper(shape: tuple[int, ...], dtype: np.dtype) -> int:
    """Bound ndarray.tolist's actual nested lists and boxed numeric leaves."""
    containers, slots, parents = 0, 0, 1
    for width in shape:
        containers += parents
        slots += parents * width
        parents *= width
    scalar = max(256, declared_integer_storage_bytes(dtype.itemsize * 8))
    return 256 * containers + 16 * slots + scalar * parents


def _polyhedral_provenance_arrays(
    construction: PolyhedralConstruction,
) -> tuple[tuple[str, np.ndarray, bool], ...]:
    """Borrow numeric banks; the boolean marks publication-order selection."""
    data = construction.diagram.construction
    return (
        ("site_ancestry", construction.site_parents, False),
        ("cell_sites", construction.cell_sites, False),
        ("cell_components", construction.component_ids, False),
        ("cell_regions", construction.cell_regions, False),
        ("face_facets", construction.face_facets, False),
        ("face_recovery_triangles", construction.face_recovery_triangles, False),
        ("feature_edge_sources", construction.feature_edge_sources, False),
        ("feature_source_edges", construction.feature_source_edges, False),
        ("carrier_points", construction.recovery.points, False),
        ("carrier_tetrahedra", construction.recovery.tetrahedra, False),
        ("vertex_carriers", data.vertex_carriers, False),
        ("feature_distance_upper", construction.feature_maximum_distance, False),
        ("feature_coverage_gap_upper", construction.feature_coverage_gap, False),
        ("requested_site_volumes", construction.requested_site_volumes, False),
        ("achieved_site_volumes", construction.achieved_site_volumes, False),
        ("construction_cell_volumes", data.cell_volumes, True),
        ("construction_cell_first_moments", data.cell_moments, True),
        ("construction_cell_site_second_moments", data.cell_second_moments, True),
        ("piece_sites", construction.diagram.piece_original_sites, False),
        ("piece_image_sites", data.piece_sites, False),
        ("piece_tetrahedra", data.piece_tets, False),
        ("piece_cells", data.piece_cells, False),
        ("piece_volumes", data.piece_volumes, False),
        ("cell_image_sites", data.cell_sites, True),
        ("cell_image_exponents", construction.diagram.cell_image_exponents, True),
        ("piece_image_exponents", construction.diagram.piece_image_exponents, False),
        ("site_points", construction.diagram.points, False),
        ("site_weights", construction.diagram.weights, False),
    )


def _polyhedral_provenance(
    source: sources_owner.NativePlcSource | sources_owner.NativePolyhedralSource,
    prepared: PreparedPolyhedralVolume,
    plan_id: str,
    construction: PolyhedralConstruction,
    *,
    budget: CoordinateEnclosureBudget,
) -> dict[str, Any]:
    """Admit every boxed provenance payload before indexing or calling tolist.

    Retained storage is declared Python-list/scalar structure and logical output
    banks. This does not describe compiler, device, or process-RSS enforcement.
    """
    c = require_polyhedral_connectivity(construction.mesh)
    data = construction.diagram.construction
    with budget.temporary_scope(), budget.live_storage() as storage:
        # The descriptor tuple has exactly 28 triples and borrows its keys.
        # It precedes no numeric payload allocation.
        budget.reserve(0, 256 + 28 * (16 + 256 + 3 * 16))
        arrays = _polyhedral_provenance_arrays(construction)
        # At most 48 mapping fields, scalar metadata and <=64-character keys.
        # Count every key and value independently, including borrowed strings.
        output = 256 + 64 * 48 + 48 * 2 * (256 + 4 * 64)
        output += 4 * (
            len(source.binding_id) + len(plan_id) + len(prepared.specification_id)
            + len(construction.construction_id)
        )
        output += sum(
            declared_integer_storage_bytes(int(value).bit_length())
            for value in (
                construction.refinement_steps, construction.lloyd_steps,
                construction.work_units, data.source_work_units,
                construction.diagram.preparation_work_units,
                construction.diagram.adjacency_work_units,
            )
        )
        work, scratch = 0, 256 + 8 * c.cell_count
        for _, array, selected in arrays:
            shape = (c.cell_count, *array.shape[1:]) if selected else array.shape
            output += _tolist_storage_upper(shape, array.dtype)
            work += prod(shape)
            if selected:
                scratch = max(
                    scratch,
                    512 + 8 * c.cell_count + prod(shape) * array.dtype.itemsize,
                )
        constraints = construction.periodic_constraints
        output += 256 + 16 * len(constraints)
        output += sum(256 + 4 * len(value.constraint_id) for value in constraints)
        for pairs in (construction.seam_face_pairs, construction.seam_vertex_pairs):
            if pairs is not None:
                output += _tolist_storage_upper(pairs.shape, pairs.dtype)
                work += pairs.size
        storage.set_bound(output, work=work)
        budget.reserve(0, scratch)
        provenance: dict[str, Any] = {
            "kind": "native-polyhedral-volume-mesh",
            "route": "plc_restricted_power",
            "source": source.binding_id,
            "plan": plan_id,
            "specification": prepared.specification_id,
            "construction": construction.construction_id,
            "feature_tolerance": prepared.schedule.feature_tolerance,
            "weight_relative_residual": construction.optimization_relative_residual,
            "refinement_steps": construction.refinement_steps,
            "lloyd_steps": construction.lloyd_steps,
            "work_units": construction.work_units,
            "periodic_constraints": [value.constraint_id for value in constraints],
            "periodic_seam_faces": None if construction.seam_face_pairs is None
            else construction.seam_face_pairs.tolist(),
            "periodic_seam_vertices": None if construction.seam_vertex_pairs is None
            else construction.seam_vertex_pairs.tolist(),
            "source_work_units": data.source_work_units,
            "preparation_work_units": construction.diagram.preparation_work_units,
            "adjacency_work_units": construction.diagram.adjacency_work_units,
        }
        order = np.asarray(c.cell_global_ids)
        for name, array, selected in arrays:
            provenance[name] = (array[order] if selected else array).tolist()
        budget.retain_owner(provenance, temporary_storage_credit=storage)
        return provenance


@final
class CertifiedPolyhedralAssociations(StrictModule):
    """Exact source rows and the acceptance request carrying their support proof."""

    associations: tuple[GeometryAssociation, ...]
    request: NativeCertificationRequest


def certify_polyhedral_associations(
    source: sources_owner.NativePlcSource | sources_owner.NativePolyhedralSource,
    coordinate_contract: SpatialCoordinateContract,
    construction: PolyhedralConstruction,
    associations: tuple[GeometryAssociation, ...],
    request: NativeCertificationRequest,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
) -> CertifiedPolyhedralAssociations:
    """Earn exact PLC rows on the published exact power geometry.

    Construction rows name declared strata with rounded residual evidence. They
    are republished as exact only after every vertex, edge, face loop and cell
    is proved against the named stratum of the authored PLC on the rational
    exact-power vertices; the proof's acceptance evidence is reused by
    certification instead of being recomputed.
    """
    limits = request.limits
    support_source = _polyhedral_support_source(
        source.complex, source.source_id, source.source_revision, coordinate_contract,
        limits, record_phase, scope=scope, budget=budget,
    )
    if support_source.association_transfer.domain.domain_id != construction.domain.domain_id:
        raise MeshingFailure(
            MeshingFailureCategory.ASSOCIATION_FAILED,
            "polyhedral.support.domain_id must equal publication.domain_id; got different PLC domains",
            stage=MeshingStageKind.GEOMETRY_ASSOCIATION.value,
        )
    try:
        output, scratch, work = _polyhedral_organization_bounds(
            source, None, construction
        )
        with (
            coordinate_phase(scope, limits, budget),
            budget.temporary_scope(),
            budget.live_storage() as storage,
        ):
            # Mapped source proof constructs replacement source columns and
            # association rows too; its exact arithmetic keeps its own charges
            # on this same ledger, while these nonrational banks coexist.
            storage.set_bound(output, work=work)
            budget.reserve(0, scratch)
            proof = certify_mapped_plc_associations(
                support_source.association_transfer,
                construction.mesh,
                construction.geometry,
                associations,
                certificate_limits=request.certificate_limits,
                budget=budget,
            )
            budget.retain_owner(proof.associations, temporary_storage_credit=storage)
    except AssociationPropagationError as error:
        raise MeshingFailure(
            MeshingFailureCategory.ASSOCIATION_FAILED,
            str(error),
            stage=MeshingStageKind.GEOMETRY_ASSOCIATION.value,
            entity_ids=tuple(int(identifier) for identifier in error.target_ids.tolist()),
        ) from error
    except ValueError as error:
        raise MeshingFailure(
            MeshingFailureCategory.ASSOCIATION_FAILED,
            str(error),
            stage=MeshingStageKind.GEOMETRY_ASSOCIATION.value,
        ) from error
    return CertifiedPolyhedralAssociations(
        proof.associations, replace(request, prepared=proof.support.prepared)
    )


def _polyhedral_support_source(
    plc: PiecewiseLinearComplex,
    source_id: str,
    source_revision: str,
    coordinate_contract: SpatialCoordinateContract,
    limits: MeshingLimits,
    record_phase: NativeMeshingPhaseRecorder | None,
    *,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
) -> PreparedPlcSource:
    """Bound the source-only host tables around native PLC source admission."""
    vertices, incidence = plc.vertices.shape[0], plc.polygon_vertices.size
    # A source polygon has at most its corner count triangles. Its original
    # edges and all triangulation edges, plus authored segments, bound the
    # source edge union without first constructing that union.
    triangles, edges = incidence, 4 * incidence + plc.segments.shape[0]
    facets, regions = plc.facet_count, plc.region_count
    table_bytes = (
        24 * vertices + 16 * edges + 24 * triangles + 8 * triangles
        + 16 * facets + 8 * (vertices + edges + regions)
    )
    integer = declared_integer_storage_bytes(max(vertices, edges, 1).bit_length())
    # Incidence validation uses triangle/edge tuple keys and scalar index
    # columns. Three corners per triangle and two per edge are explicit.
    incidence_records = 3 * triangles + 2 * edges
    scratch = 4 * (table_bytes + 12 * 256)
    scratch += incidence_records * (64 + 256 + 3 * (16 + integer))
    with (
        budget.bind_ticket(scope.ticket_view()),
        budget.temporary_scope(),
        budget.live_storage() as storage,
    ):
        budget.reserve(0, 2 * 256)
        borrowed = budget.live_object_storage_upper(plc)
        borrowed += budget.live_object_storage_upper(coordinate_contract)
        # Transfer geometry tables, domain snapshot, source polygon indices,
        # and fixed prepared-source/domain/transfer metadata owners.
        output = borrowed + 3 * table_bytes + 8 * triangles + 64 * 256
        storage.set_bound(output, work=vertices + incidence_records)
        budget.reserve(0, scratch)
        with budget.ticket.native_target():
            result = prepare_plc_source(
                plc, source_id, source_revision, coordinate_contract,
                limits=limits, record_phase=record_phase, scope=scope,
                coordinate_budget=budget,
            )
        budget.retain_owner(result, temporary_storage_credit=storage)
        return result


@checked
def execute_polyhedral_route(
    source: sources_owner.NativePlcSource | sources_owner.NativePolyhedralSource,
    specification: VolumeMeshingSpec,
    prepared: PreparedPolyhedralVolume,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    plan_id: str,
    /,
    *,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    scope: NativeExecutionScope,
    workspace: NativeHostStorageWorkspace,
    coordinate_operands: CoordinateOperandCollector | None = None,
) -> CellMeshingResult:
    "Keep construction, association and publication in one scope."
    try:
        native_volume_checkpoint(
            specification.limits, MeshingStageKind.SOURCE_INSPECTION, scope=scope
        )
        result = _execute_polyhedral_route(
            source,
            specification,
            prepared,
            coordinate_contract,
            provider,
            plan_id,
            record_phase=record_phase,
            scope=scope,
            workspace=workspace,
            coordinate_operands=coordinate_operands,
        )
        native_volume_checkpoint(
            specification.limits, MeshingStageKind.CERTIFICATION, scope=scope
        )
        return result
    except (
        MeshcoreError,
        CoordinateEnclosureResourceError,
        PolyhedralGenerationError,
    ) as error:
        raise construction_failure(error, "native_execution") from error


__all__ = [
    "PreparedPolyhedralVolume",
    "polyhedral_support_issues",
    "execute_polyhedral_route",
]


@final
class _PolyhedralAssociationPhase(StrictModule):
    associations: tuple[GeometryAssociation, ...]


def _polyhedral_associations(
    source: sources_owner.NativePlcSource | sources_owner.NativePolyhedralSource,
    specification: VolumeMeshingSpec | None,
    construction: PolyhedralConstruction,
    tolerance: float,
    points: np.ndarray,
    cell_ids: np.ndarray,
    face_ids: np.ndarray,
    edge_ids: np.ndarray,
    facets: np.ndarray,
    selected: np.ndarray,
) -> _PolyhedralAssociationPhase:
    """Polyhedral associations."""
    mesh = construction.mesh
    connectivity = require_polyhedral_connectivity(mesh)
    phase = _polyhedral_facet_residuals(
        connectivity,
        construction,
        points,
        edge_ids,
        face_ids,
        facets,
        selected,
        tolerance,
    )
    edge_facets = phase.edge_facets
    edge_residuals = phase.edge_residuals
    face_residuals = phase.face_residuals
    orientations = phase.orientations
    edge_vertices = phase.edge_vertices
    feature_sources = construction.feature_edge_sources
    combined = construction.feature_maximum_distance + construction.feature_coverage_gap
    combined = np.where(combined == 0, 0.0, np.nextafter(combined, np.inf))
    for feature in () if specification is None else specification.protected_features:
        identifiers = np.asarray(feature.scope.entity_ids, dtype=np.int64)
        if feature.scope.entity_dimension == 1:
            bad = identifiers[combined[identifiers] > feature.maximum_deviation]
        elif feature.scope.entity_dimension == 2:
            bad = np.asarray(
                [
                    int(facet)
                    for facet in identifiers
                    if np.max(face_residuals[facets[selected] == facet], initial=0.0)
                    > feature.maximum_deviation
                ],
                dtype=np.int64,
            )
        else:
            continue
        if bad.size:
            raise MeshingFailure(
                MeshingFailureCategory.AUDIT_FAILED,
                "polyhedral.protected_features.deviation must not exceed maximum_deviation; got excessive feature distance",
                stage="geometry_association",
                entity_ids=tuple(map(int, bad)),
            )
    boundary_edges = np.flatnonzero((edge_facets >= 0) | (feature_sources >= 0))
    revision = source.source_revision
    edge_dimensions = np.where(feature_sources[boundary_edges] >= 0, 1, 2)
    edge_sources = np.where(
        feature_sources[boundary_edges] >= 0,
        feature_sources[boundary_edges],
        edge_facets[boundary_edges],
    )
    edge_orientations = np.zeros(boundary_edges.size, dtype=np.int8)
    for row, edge in enumerate(boundary_edges):
        if feature_sources[edge] >= 0:
            pair = construction.feature_source_edges[feature_sources[edge]]
            direction = (
                construction.recovery.points[pair[1]]
                - construction.recovery.points[pair[0]]
            )
            mesh_edge = edge_vertices[edge]
            edge_orientations[row] = (
                1
                if np.dot(points[mesh_edge[1]] - points[mesh_edge[0]], direction) > 0
                else -1
            )
    associations = (
        _vertex_association(
            source, specification, construction, selected, face_residuals
        ),
        GeometryAssociation(
            GeometryAssociationKind.PIECEWISE_LINEAR,
            source.source_id,
            revision,
            mesh.entity_set(3).entity_set_id,
            cell_ids,
            tuple(
                entity_key(revision, "region", int(x)) for x in construction.cell_regions
            ),
            np.zeros(cell_ids.size, dtype=np.float64),
            exact=True,
            source_dimensions=np.full(cell_ids.size, 3, dtype=np.int64),
            source_indices=construction.cell_regions,
            source_entity_roles=(GeometrySourceEntityRole.REGION,) * cell_ids.size,
        ),
        GeometryAssociation(
            GeometryAssociationKind.PIECEWISE_LINEAR,
            source.source_id,
            revision,
            mesh.entity_set(2).entity_set_id,
            face_ids[selected],
            tuple(entity_key(revision, "facet", int(facets[x])) for x in selected),
            face_residuals,
            exact=False,
            orientations=orientations,
            source_dimensions=np.full(selected.size, 2, dtype=np.int64),
            source_indices=facets[selected],
            source_entity_roles=(GeometrySourceEntityRole.FACET,) * selected.size,
        ),
        GeometryAssociation(
            GeometryAssociationKind.PIECEWISE_LINEAR,
            source.source_id,
            revision,
            mesh.entity_set(1).entity_set_id,
            edge_ids[boundary_edges],
            tuple(
                entity_key(
                    revision,
                    "edge" if feature_sources[x] >= 0 else "facet",
                    int(
                        feature_sources[x] if feature_sources[x] >= 0 else edge_facets[x]
                    ),
                )
                for x in boundary_edges
            ),
            np.asarray(
                [
                    construction.feature_maximum_distance[feature_sources[x]]
                    if feature_sources[x] >= 0
                    else edge_residuals[x]
                    for x in boundary_edges
                ]
            ),
            exact=False,
            source_dimensions=edge_dimensions,
            source_indices=edge_sources,
            orientations=edge_orientations,
            source_entity_roles=tuple(
                GeometrySourceEntityRole.EDGE
                if feature_sources[edge] >= 0
                else GeometrySourceEntityRole.FACET
                for edge in boundary_edges
            ),
        ),
    )
    return _PolyhedralAssociationPhase(associations)


@final
class _PolyhedralCompliancePhase(StrictModule):
    compliance: MeshingComplianceReport


def _polyhedral_compliance(
    construction: PolyhedralConstruction,
    specification: VolumeMeshingSpec,
    record_phase: NativeMeshingPhaseRecorder | None,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> _PolyhedralCompliancePhase:
    """Admit coordinate copies, vector/length/growth banks and report text."""
    if budget is None:
        return _unbudgeted_polyhedral_compliance(construction, specification, record_phase)
    output, scratch, work = _polyhedral_compliance_bounds(construction, specification)
    with budget.temporary_scope(), budget.live_storage() as storage:
        storage.set_bound(output, work=work)
        budget.reserve(0, scratch)
        result = _unbudgeted_polyhedral_compliance(
            construction, specification, record_phase
        )
        budget.retain_owner(result, temporary_storage_credit=storage)
        return result


def _polyhedral_compliance_bounds(
    construction: PolyhedralConstruction, specification: VolumeMeshingSpec
) -> tuple[int, int, int]:
    c = require_polyhedral_connectivity(construction.mesh)
    vertices, edges = construction.mesh.coordinates.shape[0], c.edge_count
    # Coordinate/edge host copies, two gathered endpoints, vectors and norm
    # multiplication/reduction; length/quantile copies; vertex min/max/ratio
    # banks and masks. Count all banks together rather than assuming aliasing.
    scratch = 24 * vertices + 16 * edges + 4 * 24 * edges
    scratch += 4 * 8 * edges + 5 * 8 * vertices + 3 * vertices + 16 * 256
    # At most four requested and eight achieved/issue entries, with key text
    # derived from the actual control ID and the longest fixed metric suffix.
    text = len(specification.size_controls[0].control_id) + 64
    output = 32 * (256 + 4 * text + 256 + 2 * 16)
    output += 8 * 256 + 4 * len(specification.specification_id)
    return output, scratch, vertices + 8 * edges


def _unbudgeted_polyhedral_compliance(
    construction: PolyhedralConstruction,
    specification: VolumeMeshingSpec,
    record_phase: NativeMeshingPhaseRecorder | None,
) -> _PolyhedralCompliancePhase:
    """Compute the unchanged uniform-size evidence on admitted host banks."""
    mesh = construction.mesh
    c = require_polyhedral_connectivity(mesh)
    compliance_started = phase_started(record_phase)
    lengths, growth = edge_size_evidence(
        np.asarray(mesh.coordinates), np.asarray(c.edges)
    )
    size_control = specification.size_controls[0]
    match size_control:
        case UniformSizeControl():
            pass
        case _:
            raise TypeError(
                "polyhedral.size_controls must contain a uniform control; got a different control kind"
            )
    requested, achieved, issues = uniform_size_compliance(
        size_control, specification.size_compliance, lengths, growth
    )
    achieved.extend(
        (
            ("cell_diameter_upper", construction.maximum_cell_diameter),
            (
                "volume_optimization_relative_residual",
                construction.optimization_relative_residual,
            ),
        )
    )
    compliance = MeshingComplianceReport(
        specification.specification_id,
        issues=tuple(issues),
        requested=tuple(requested),
        achieved=tuple(achieved),
    )
    record_elapsed(record_phase, "compliance", compliance_started)
    return _PolyhedralCompliancePhase(compliance)


@final
class _PolyhedralFacetPhase(StrictModule):
    edge_facets: np.ndarray
    edge_residuals: np.ndarray
    face_residuals: np.ndarray
    orientations: np.ndarray
    edge_vertices: np.ndarray


def _polyhedral_facet_residuals(
    connectivity: PolyhedralConnectivity,
    construction: PolyhedralConstruction,
    points: np.ndarray,
    edge_ids: np.ndarray,
    face_ids: np.ndarray,
    facets: np.ndarray,
    selected: np.ndarray,
    tolerance: float,
) -> _PolyhedralFacetPhase:
    """Polyhedral facet residuals."""
    offsets, vertices = (
        np.asarray(connectivity.face_vertex_offsets),
        np.asarray(connectivity.face_vertex_values),
    )
    edge_vertices = np.asarray(jax.device_get(connectivity.edges))
    edge_rows = {
        tuple(sorted(map(int, edge))): row for row, edge in enumerate(edge_vertices)
    }
    edge_facets = np.full(edge_ids.size, -1, dtype=np.int64)
    edge_residuals = np.zeros(edge_ids.size, dtype=np.float64)
    face_residuals = np.zeros(selected.size, dtype=np.float64)
    orientations = np.zeros(selected.size, dtype=np.int8)
    for row, face in enumerate(selected):
        facet = int(facets[face])
        triangle = int(construction.face_recovery_triangles[face])
        source_points = construction.recovery.points[
            construction.recovery.faces[triangle]
        ]
        normal = np.cross(
            source_points[1] - source_points[0], source_points[2] - source_points[0]
        )
        length = np.linalg.norm(normal)
        loop = vertices[offsets[face] : offsets[face + 1]]
        face_residuals[row] = float(
            np.max(np.abs((points[loop] - source_points[0]) @ normal)) / length
        )
        normal_mesh = np.sum(
            np.cross(
                points[loop] - points[loop[0]],
                np.roll(points[loop], -1, axis=0) - points[loop[0]],
            ),
            axis=0,
        )
        orientations[row] = 1 if np.dot(normal_mesh, normal) > 0 else -1
        for a, b in zip(loop, np.roll(loop, -1), strict=True):
            edge_facets[edge_rows[tuple(sorted((int(a), int(b))))]] = facet
            edge = edge_rows[tuple(sorted((int(a), int(b))))]
            edge_residuals[edge] = max(edge_residuals[edge], face_residuals[row])
    if np.any(face_residuals > tolerance):
        raise MeshingFailure(
            MeshingFailureCategory.AUDIT_FAILED,
            "polyhedral.face_residuals must not exceed feature_tolerance; got excessive source-plane residuals",
            stage="geometry_association",
            entity_ids=tuple(map(int, face_ids[selected[face_residuals > tolerance]])),
        )
    return _PolyhedralFacetPhase(
        edge_facets, edge_residuals, face_residuals, orientations, edge_vertices
    )


@final
class _PolyhedralConstructionPhase(StrictModule):
    construction: PolyhedralConstruction
    mesh: CellMesh
    c: PolyhedralConnectivity


def _construct_polyhedral_route(
    source: sources_owner.NativePlcSource | sources_owner.NativePolyhedralSource,
    specification: VolumeMeshingSpec,
    prepared: PreparedPolyhedralVolume,
    started: float,
    record_phase: NativeMeshingPhaseRecorder | None,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
) -> _PolyhedralConstructionPhase:
    """Construct polyhedral route."""

    def native_phase(
        phase: str, elapsed: float, work_units: int | None, invocations: int
    ) -> None:
        if record_phase is not None:
            record_phase(
                NativeMeshingPhaseMeasurement(
                    parse(phase, PowerConstructionPhase, "polyhedral.phase"),
                    elapsed,
                    work_units,
                    invocations,
                )
            )

    construction_started = phase_started(record_phase)
    try:
        with budget.bind_ticket(scope.ticket_view()):
            local_seeds, seed_regions, protected = _polyhedral_construction_inputs(
                source, specification, budget=budget
            )
        construction = generate_polyhedral_volume(
            source.complex,
            sites=prepared.sites,
            weights=prepared.weights,
            schedule=prepared.schedule,
            source_id=source.source_id,
            seeds=local_seeds,
            seed_regions=seed_regions,
            protected_vertices=protected,
            periodic_constraints=specification.periodic_constraints,
            record_native_phase=None if record_phase is None else native_phase,
            scope=scope,
            budget=budget,
            limits=specification.limits,
        )
    except (
        MeshcoreError,
        PolyhedralGenerationError,
        CoordinateEnclosureResourceError,
    ) as error:
        raise construction_failure(error, "volume_fill") from error
    record_elapsed(
        record_phase,
        "construction",
        construction_started,
        work_units=construction.work_units,
    )
    check_deadline(started, specification.limits, MeshingStageKind.VOLUME_FILL)
    mesh = construction.mesh
    c = require_polyhedral_connectivity(mesh)
    limits = specification.limits
    capacities = (
        ("vertices", limits.maximum_vertices, mesh.coordinates.shape[0]),
        ("edges", limits.maximum_edges, c.edge_count),
        ("faces", limits.maximum_faces, c.face_count),
        ("cells", limits.maximum_cells, c.cell_count),
        (
            "connectivity_entries",
            limits.maximum_connectivity_entries,
            sum(
                value.size
                for value in (
                    c.cell_vertex_values,
                    c.face_vertex_values,
                    c.cell_face_values,
                )
            ),
        ),
    )
    for name, maximum, achieved in capacities:
        if achieved > maximum:
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                f"polyhedral.entity_capacity must not exceed maximum_{name}={maximum}; got {achieved}",
                stage="volume_fill",
                requested=((name, maximum),),
                achieved=((name, achieved),),
            )
    return _PolyhedralConstructionPhase(construction, mesh, c)
