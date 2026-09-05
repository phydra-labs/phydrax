#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class InterfaceSide(str, Enum):
    """The two trace sides induced by an oriented surface."""

    MINUS = "minus"
    PLUS = "plus"


class SurfacePreparationStatus(str, Enum):
    """Typed reason that an authoritative surface could not be realized."""

    INVALID_INPUT = "invalid_input"
    NONFINITE_GEOMETRY = "nonfinite_geometry"
    UNSUPPORTED_TOPOLOGY = "unsupported_topology"
    NONMANIFOLD_TOPOLOGY = "nonmanifold_topology"
    INCONSISTENT_ORIENTATION = "inconsistent_orientation"
    NONORIENTABLE_TOPOLOGY = "nonorientable_topology"
    AUDIT_REJECTED = "audit_rejected"
    TOPOLOGY_CHANGED = "topology_changed"


class SurfacePreparationError(ValueError):
    """Fail-closed surface preparation error with retained audit evidence."""

    def __init__(
        self,
        status: SurfacePreparationStatus,
        message: str,
        /,
        *,
        issues: Sequence[str] = (),
        report: SurfaceAuditReport | None = None,
    ):
        if not isinstance(status, SurfacePreparationStatus):
            raise TypeError("status must be a SurfacePreparationStatus.")
        message_ = str(message)
        if not message_:
            raise ValueError("Surface preparation errors require a message.")
        self.status = status
        self.issues = tuple(str(issue) for issue in issues)
        self.report = report
        super().__init__(message_)


class SurfaceMetadata(StrictModule, NonTrainableState):
    """Static units and source provenance for one authoritative surface."""

    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    coordinate_contract: SpatialCoordinateContract
    provenance: tuple[str, ...] = eqx.field(static=True)
    cell_tags: tuple[str, ...] = eqx.field(static=True)
    metadata_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        source_id: str,
        source_revision: str,
        coordinate_contract: SpatialCoordinateContract,
        provenance: Sequence[str],
        cell_tags: Sequence[str] = (),
    ):
        source = str(source_id)
        revision = str(source_revision)
        if not isinstance(coordinate_contract, SpatialCoordinateContract):
            raise TypeError(
                "Surface coordinate_contract must be a SpatialCoordinateContract."
            )
        history = tuple(str(entry) for entry in provenance)
        tags = tuple(str(tag) for tag in cell_tags)
        if not source or not revision:
            raise ValueError("Surface source_id and source_revision must be non-empty.")
        if not history or any(not entry for entry in history):
            raise ValueError("Surface provenance must contain non-empty entries.")
        if any(not tag for tag in tags):
            raise ValueError("Surface cell tags must be non-empty.")
        self.source_id = source
        self.source_revision = revision
        self.coordinate_contract = coordinate_contract
        self.provenance = history
        self.cell_tags = tags
        self.metadata_id = canonical_fingerprint(
            {
                "kind": "surface-metadata",
                "source_id": source,
                "source_revision": revision,
                "coordinate_contract": coordinate_contract.spatial_id,
                "provenance": history,
                "cell_tags": tags,
            }
        )


class SurfaceSelection(StrictModule, NonTrainableState):
    """Named cell selection bound to one exact CellMesh entity set."""

    name: str = eqx.field(static=True)
    role: str = eqx.field(static=True)
    cell_entity_set_id: str = eqx.field(static=True)
    cell_global_ids: Array
    selection_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        cell_global_ids: ArrayLike,
        /,
        *,
        cell_entity_set_id: str,
        role: str = "surface",
    ):
        name_ = str(name)
        role_ = str(role)
        entity_set = str(cell_entity_set_id)
        identifiers = np.asarray(cell_global_ids)
        if not name_ or not role_ or not entity_set:
            raise ValueError("Surface selection identity and role must be non-empty.")
        if identifiers.ndim != 1 or not np.issubdtype(identifiers.dtype, np.integer):
            raise TypeError("Surface selection cell IDs must be one integer array.")
        identifiers = identifiers.astype(np.int64, copy=False)
        if identifiers.size == 0:
            raise ValueError("Surface selections must contain at least one cell.")
        if np.any(identifiers < 0) or np.unique(identifiers).size != identifiers.size:
            raise ValueError(
                "Surface selection cell IDs must be unique and non-negative."
            )
        identifiers = np.sort(identifiers, kind="stable")
        self.name = name_
        self.role = role_
        self.cell_entity_set_id = entity_set
        self.cell_global_ids = jnp.asarray(identifiers, dtype=jnp.int64)
        self.selection_id = canonical_fingerprint(
            {
                "kind": "surface-selection-v1",
                "name": name_,
                "role": role_,
                "cell_entity_set_id": entity_set,
                "cell_global_ids": array_tree_fingerprint(identifiers),
            }
        )


class SurfaceInterface(StrictModule, NonTrainableState):
    """Oriented two-region interface over an exact surface selection."""

    name: str = eqx.field(static=True)
    support: SurfaceSelection
    minus_region: str = eqx.field(static=True)
    plus_region: str = eqx.field(static=True)
    interface_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        support: SurfaceSelection,
        /,
        *,
        minus_region: str,
        plus_region: str,
    ):
        name_ = str(name)
        minus = str(minus_region)
        plus = str(plus_region)
        if not isinstance(support, SurfaceSelection):
            raise TypeError("Surface interface support must be a SurfaceSelection.")
        if not name_ or not minus or not plus:
            raise ValueError("Surface interface name and regions must be non-empty.")
        if minus == plus:
            raise ValueError("Surface interface sides must name distinct regions.")
        self.name = name_
        self.support = support
        self.minus_region = minus
        self.plus_region = plus
        self.interface_id = canonical_fingerprint(
            {
                "kind": "surface-interface-v1",
                "name": name_,
                "support": support.selection_id,
                "minus_region": minus,
                "plus_region": plus,
            }
        )

    def region(self, side: InterfaceSide, /) -> str:
        if not isinstance(side, InterfaceSide):
            raise TypeError("side must be an InterfaceSide.")
        return self.minus_region if side is InterfaceSide.MINUS else self.plus_region


class SurfaceAuditPolicy(StrictModule, NonTrainableState):
    """Scale-aware metric, orientation, and host-capacity audit policy."""

    minimum_face_area: float = eqx.field(static=True)
    relative_degeneracy_tolerance: float = eqx.field(static=True)
    minimum_closed_volume: float = eqx.field(static=True)
    require_closed: bool = eqx.field(static=True)
    require_outward_orientation: bool = eqx.field(static=True)
    maximum_vertices: int | None = eqx.field(static=True)
    maximum_cells: int | None = eqx.field(static=True)
    maximum_edges: int | None = eqx.field(static=True)
    maximum_components: int | None = eqx.field(static=True)
    maximum_boundary_edges: int | None = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        minimum_face_area: float = 0.0,
        relative_degeneracy_tolerance: float = 64.0 * np.finfo(float).eps,
        minimum_closed_volume: float = 0.0,
        require_closed: bool = False,
        require_outward_orientation: bool = False,
        maximum_vertices: int | None = None,
        maximum_cells: int | None = None,
        maximum_edges: int | None = None,
        maximum_components: int | None = None,
        maximum_boundary_edges: int | None = None,
    ):
        minimum_area = float(minimum_face_area)
        relative = float(relative_degeneracy_tolerance)
        minimum_volume = float(minimum_closed_volume)
        if (
            not np.isfinite(minimum_area)
            or minimum_area < 0.0
            or not np.isfinite(relative)
            or relative < 0.0
            or not np.isfinite(minimum_volume)
            or minimum_volume < 0.0
        ):
            raise ValueError("Surface metric tolerances must be finite and non-negative.")
        capacities = (
            maximum_vertices,
            maximum_cells,
            maximum_edges,
            maximum_components,
            maximum_boundary_edges,
        )
        if any(limit is not None and int(limit) <= 0 for limit in capacities):
            raise ValueError("Surface audit capacities must be positive when supplied.")
        normalized = tuple(None if limit is None else int(limit) for limit in capacities)
        self.minimum_face_area = minimum_area
        self.relative_degeneracy_tolerance = relative
        self.minimum_closed_volume = minimum_volume
        self.require_closed = bool(require_closed)
        self.require_outward_orientation = bool(require_outward_orientation)
        (
            self.maximum_vertices,
            self.maximum_cells,
            self.maximum_edges,
            self.maximum_components,
            self.maximum_boundary_edges,
        ) = normalized
        self.policy_id = canonical_fingerprint(
            {
                "kind": "surface-audit-policy-v1",
                "minimum_face_area": minimum_area,
                "relative_degeneracy_tolerance": relative,
                "minimum_closed_volume": minimum_volume,
                "require_closed": bool(require_closed),
                "require_outward_orientation": bool(require_outward_orientation),
                "capacities": normalized,
            }
        )


class SurfaceOrientationRepair(StrictModule, NonTrainableState):
    """Explicit mapping from input triangle orientation to repaired orientation."""

    source_face_indices: Array
    orientation_signs: Array
    component_ids: Array
    source_topology_id: str = eqx.field(static=True)
    repaired_topology_id: str = eqx.field(static=True)
    repair_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        source_face_indices: ArrayLike,
        orientation_signs: ArrayLike,
        component_ids: ArrayLike,
        source_topology_id: str,
        repaired_topology_id: str,
    ):
        source = np.asarray(source_face_indices, dtype=np.int64)
        signs = np.asarray(orientation_signs, dtype=np.int8)
        components = np.asarray(component_ids, dtype=np.int32)
        source_topology = str(source_topology_id)
        repaired_topology = str(repaired_topology_id)
        if (
            source.ndim != 1
            or signs.shape != source.shape
            or components.shape != source.shape
        ):
            raise ValueError("Orientation repair arrays must share one rank-1 shape.")
        if np.unique(source).size != source.size or np.any(source < 0):
            raise ValueError(
                "Orientation repair source indices must be unique and non-negative."
            )
        if np.any((signs != 1) & (signs != -1)) or np.any(components < 0):
            raise ValueError("Orientation repair signs and component IDs are invalid.")
        if not source_topology or not repaired_topology:
            raise ValueError("Orientation repair topology identities must be non-empty.")
        self.source_face_indices = jnp.asarray(source, dtype=jnp.int64)
        self.orientation_signs = jnp.asarray(signs, dtype=jnp.int8)
        self.component_ids = jnp.asarray(components, dtype=jnp.int32)
        self.source_topology_id = source_topology
        self.repaired_topology_id = repaired_topology
        self.repair_id = canonical_fingerprint(
            {
                "kind": "surface-orientation-repair-v1",
                "source_topology_id": source_topology,
                "repaired_topology_id": repaired_topology,
                "source_face_indices": array_tree_fingerprint(source),
                "orientation_signs": array_tree_fingerprint(signs),
                "component_ids": array_tree_fingerprint(components),
            }
        )


class SurfaceChartMappingEvidence(StrictModule, NonTrainableState):
    """Exact deterministic chart-local to CellMesh-global cell-ID mapping."""

    chart_ids: Array
    cell_global_ids: Array
    cell_entity_set_id: str = eqx.field(static=True)
    mapping_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        chart_ids: ArrayLike,
        cell_global_ids: ArrayLike,
        cell_entity_set_id: str,
    ):
        charts = np.asarray(chart_ids, dtype=np.int32)
        cells = np.asarray(cell_global_ids, dtype=np.int64)
        entity_set = str(cell_entity_set_id)
        if charts.ndim != 1 or cells.shape != charts.shape or charts.size == 0:
            raise ValueError(
                "Surface chart mapping arrays must share one non-empty shape."
            )
        if not np.array_equal(charts, np.arange(charts.size, dtype=np.int32)):
            raise ValueError("Surface chart IDs must be contiguous local indices.")
        if np.any(cells < 0) or np.unique(cells).size != cells.size or not entity_set:
            raise ValueError(
                "Surface chart mapping requires exact unique global cell IDs."
            )
        self.chart_ids = jnp.asarray(charts, dtype=jnp.int32)
        self.cell_global_ids = jnp.asarray(cells, dtype=jnp.int64)
        self.cell_entity_set_id = entity_set
        self.mapping_id = canonical_fingerprint(
            {
                "kind": "surface-chart-mapping-v1",
                "chart_ids": array_tree_fingerprint(charts),
                "cell_global_ids": array_tree_fingerprint(cells),
                "cell_entity_set_id": entity_set,
            }
        )


class SurfaceAuditReport(StrictModule, NonTrainableState):
    """Immutable computed evidence for one exact surface realization."""

    finite: Array
    topology_valid: Array
    manifold: Array
    metric_valid: Array
    orientation_consistent: Array
    capacity_valid: Array
    outward_orientation_valid: Array
    valid: Array
    face_areas: Array
    component_ids: Array
    component_closed: Array
    component_signed_volumes: Array
    vertex_count: int = eqx.field(static=True)
    cell_count: int = eqx.field(static=True)
    edge_count: int = eqx.field(static=True)
    component_count: int = eqx.field(static=True)
    boundary_edge_count: int = eqx.field(static=True)
    boundary_loop_count: int = eqx.field(static=True)
    classification: str = eqx.field(static=True)
    component_classification: tuple[str, ...] = eqx.field(static=True)
    issues: tuple[str, ...] = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    metadata_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        finite: bool,
        topology_valid: bool,
        manifold: bool,
        metric_valid: bool,
        orientation_consistent: bool,
        capacity_valid: bool,
        outward_orientation_valid: bool,
        face_areas: ArrayLike,
        component_ids: ArrayLike,
        component_closed: ArrayLike,
        component_signed_volumes: ArrayLike,
        vertex_count: int,
        edge_count: int,
        boundary_edge_count: int,
        boundary_loop_count: int,
        classification: str,
        issues: Sequence[str],
        model_id: str,
        metadata_id: str,
        topology_id: str,
        geometry_id: str,
        policy_id: str,
        policy_accepts_open: bool,
    ):
        areas = np.asarray(face_areas, dtype=float)
        components = np.asarray(component_ids, dtype=np.int32)
        closed = np.asarray(component_closed, dtype=bool)
        volumes = np.asarray(component_signed_volumes, dtype=float)
        if areas.ndim != 1 or areas.size == 0:
            raise ValueError("Surface audit face areas must be a non-empty rank-1 array.")
        if components.shape != areas.shape:
            raise ValueError(
                "Surface audit component IDs must contain one value per face."
            )
        component_count = int(closed.size)
        if component_count <= 0 or volumes.shape != closed.shape:
            raise ValueError("Surface audit component evidence is inconsistent.")
        if np.any(components < 0) or np.any(components >= component_count):
            raise ValueError("Surface audit component IDs are out of range.")
        classification_ = str(classification)
        issues_ = tuple(str(issue) for issue in issues)
        if not classification_ or any(not issue for issue in issues_):
            raise ValueError(
                "Surface audit classifications and issues must be non-empty."
            )
        identifiers = (
            str(model_id),
            str(metadata_id),
            str(topology_id),
            str(geometry_id),
            str(policy_id),
        )
        if any(not identifier for identifier in identifiers):
            raise ValueError("Surface audit identities must be non-empty.")
        accepted = bool(
            finite
            and topology_valid
            and manifold
            and metric_valid
            and orientation_consistent
            and capacity_valid
            and outward_orientation_valid
            and (policy_accepts_open or bool(np.all(closed)))
        )
        component_kinds = tuple("closed" if value else "open" for value in closed)
        self.finite = jnp.asarray(finite, dtype=bool)
        self.topology_valid = jnp.asarray(topology_valid, dtype=bool)
        self.manifold = jnp.asarray(manifold, dtype=bool)
        self.metric_valid = jnp.asarray(metric_valid, dtype=bool)
        self.orientation_consistent = jnp.asarray(orientation_consistent, dtype=bool)
        self.capacity_valid = jnp.asarray(capacity_valid, dtype=bool)
        self.outward_orientation_valid = jnp.asarray(
            outward_orientation_valid, dtype=bool
        )
        self.valid = jnp.asarray(accepted, dtype=bool)
        self.face_areas = jnp.asarray(areas, dtype=float)
        self.component_ids = jnp.asarray(components, dtype=jnp.int32)
        self.component_closed = jnp.asarray(closed, dtype=bool)
        self.component_signed_volumes = jnp.asarray(volumes, dtype=float)
        self.vertex_count = int(vertex_count)
        self.cell_count = int(areas.size)
        self.edge_count = int(edge_count)
        self.component_count = component_count
        self.boundary_edge_count = int(boundary_edge_count)
        self.boundary_loop_count = int(boundary_loop_count)
        self.classification = classification_
        self.component_classification = component_kinds
        self.issues = issues_
        (
            self.model_id,
            self.metadata_id,
            self.topology_id,
            self.geometry_id,
            self.policy_id,
        ) = identifiers
        self.report_id = canonical_fingerprint(
            {
                "kind": "surface-audit-report-v1",
                "model_id": identifiers[0],
                "metadata_id": identifiers[1],
                "topology_id": identifiers[2],
                "geometry_id": identifiers[3],
                "policy_id": identifiers[4],
                "checks": (
                    bool(finite),
                    bool(topology_valid),
                    bool(manifold),
                    bool(metric_valid),
                    bool(orientation_consistent),
                    bool(capacity_valid),
                    bool(outward_orientation_valid),
                    accepted,
                ),
                "face_areas": array_tree_fingerprint(areas),
                "component_ids": array_tree_fingerprint(components),
                "component_closed": array_tree_fingerprint(closed),
                "component_signed_volumes": array_tree_fingerprint(volumes),
                "counts": (
                    int(vertex_count),
                    int(areas.size),
                    int(edge_count),
                    component_count,
                    int(boundary_edge_count),
                    int(boundary_loop_count),
                ),
                "classification": classification_,
                "issues": issues_,
            }
        )

    @property
    def closed(self) -> bool:
        return bool(np.all(np.asarray(self.component_closed)))

    @property
    def open(self) -> bool:
        return not self.closed


class SurfaceValidityCertificate(StrictModule, NonTrainableState):
    """Fail-closed validity certificate bound to one computed audit report."""

    model_id: str = eqx.field(static=True)
    metadata_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    report_id: str = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)
    valid: Array

    def __init__(self, report: SurfaceAuditReport, /):
        if not isinstance(report, SurfaceAuditReport):
            raise TypeError("Surface validity certificates require an audit report.")
        if not bool(np.asarray(report.valid)):
            raise ValueError("Invalid surface audits cannot produce a certificate.")
        self.model_id = report.model_id
        self.metadata_id = report.metadata_id
        self.topology_id = report.topology_id
        self.geometry_id = report.geometry_id
        self.policy_id = report.policy_id
        self.report_id = report.report_id
        self.valid = jnp.asarray(True, dtype=bool)
        self.certificate_id = canonical_fingerprint(
            {
                "model_id": report.model_id,
                "metadata_id": report.metadata_id,
                "kind": "surface-validity-certificate-v1",
                "topology_id": report.topology_id,
                "geometry_id": report.geometry_id,
                "policy_id": report.policy_id,
                "report_id": report.report_id,
            }
        )


__all__ = [
    "InterfaceSide",
    "SurfaceAuditPolicy",
    "SurfaceAuditReport",
    "SurfaceChartMappingEvidence",
    "SurfaceInterface",
    "SurfaceMetadata",
    "SurfaceOrientationRepair",
    "SurfacePreparationError",
    "SurfacePreparationStatus",
    "SurfaceSelection",
    "SurfaceValidityCertificate",
]
