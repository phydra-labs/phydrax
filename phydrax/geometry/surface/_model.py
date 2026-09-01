#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization._cell_mesh import CellMesh
from .._atlas import AbstractBoundaryMap, BoundaryAtlas
from ._contracts import (
    SurfaceAuditPolicy,
    SurfaceAuditReport,
    SurfaceChartMappingEvidence,
    SurfaceInterface,
    SurfaceMetadata,
    SurfaceOrientationRepair,
    SurfacePreparationError,
    SurfacePreparationStatus,
    SurfaceSelection,
    SurfaceValidityCertificate,
)


if TYPE_CHECKING:
    from ._intersection import PlaneSurfaceSection


def _triangle_faces(mesh: CellMesh, /) -> np.ndarray:
    connectivity = mesh.connectivity
    faces = np.asarray(connectivity.cell_vertices, dtype=np.int32)
    kinds = np.asarray(connectivity.cell_kinds, dtype=np.int32)
    if faces.ndim != 2 or kinds.shape != (faces.shape[0],) or np.any(kinds != 3):
        raise SurfacePreparationError(
            SurfacePreparationStatus.UNSUPPORTED_TOPOLOGY,
            "Authoritative surfaces currently require affine triangle cells only.",
        )
    return faces[:, :3]


def _edge_records(faces: np.ndarray, /) -> dict[tuple[int, int], list[tuple[int, int]]]:
    records: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for face_index, face in enumerate(faces):
        for local in range(3):
            start = int(face[local])
            stop = int(face[(local + 1) % 3])
            key = (min(start, stop), max(start, stop))
            direction = 1 if (start, stop) == key else -1
            records.setdefault(key, []).append((face_index, direction))
    return records


def _face_components(
    face_count: int,
    records: dict[tuple[int, int], list[tuple[int, int]]],
    /,
) -> tuple[np.ndarray, int]:
    neighbours: list[list[int]] = [[] for _ in range(face_count)]
    for incidents in records.values():
        incident_faces = tuple(entry[0] for entry in incidents)
        for index, first in enumerate(incident_faces):
            for second in incident_faces[index + 1 :]:
                neighbours[first].append(second)
                neighbours[second].append(first)
    components = np.full((face_count,), -1, dtype=np.int32)
    component_count = 0
    for first_face in range(face_count):
        if components[first_face] >= 0:
            continue
        components[first_face] = component_count
        pending = [first_face]
        while pending:
            face = pending.pop()
            for neighbour in neighbours[face]:
                if components[neighbour] < 0:
                    components[neighbour] = component_count
                    pending.append(neighbour)
        component_count += 1
    return components, component_count


def _orientation_solution(
    faces: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int], list[tuple[int, int]]]]:
    records = _edge_records(faces)
    if any(len(incidents) > 2 for incidents in records.values()):
        raise SurfacePreparationError(
            SurfacePreparationStatus.NONMANIFOLD_TOPOLOGY,
            "A surface edge has more than two incident triangles.",
        )
    components, _ = _face_components(faces.shape[0], records)
    neighbours: list[list[tuple[int, int]]] = [[] for _ in range(faces.shape[0])]
    for incidents in records.values():
        if len(incidents) != 2:
            continue
        (first, first_direction), (second, second_direction) = incidents
        relation = -first_direction * second_direction
        neighbours[first].append((second, relation))
        neighbours[second].append((first, relation))
    signs = np.zeros((faces.shape[0],), dtype=np.int8)
    for first_face in range(faces.shape[0]):
        if signs[first_face] != 0:
            continue
        signs[first_face] = 1
        pending = [first_face]
        while pending:
            face = pending.pop()
            for neighbour, relation in neighbours[face]:
                required = int(signs[face]) * relation
                if signs[neighbour] == 0:
                    signs[neighbour] = required
                    pending.append(neighbour)
                elif signs[neighbour] != required:
                    raise SurfacePreparationError(
                        SurfacePreparationStatus.NONORIENTABLE_TOPOLOGY,
                        "Triangle adjacency constraints are not globally orientable.",
                    )
    return signs, components, records


def _repair_orientations(
    points: np.ndarray,
    faces: np.ndarray,
    /,
    *,
    orient_closed_outward: bool,
) -> tuple[np.ndarray, SurfaceOrientationRepair]:
    signs, components, _ = _orientation_solution(faces)
    repaired = faces.copy()
    repaired[signs < 0] = repaired[signs < 0][:, [0, 2, 1]]
    if orient_closed_outward:
        repaired_records = _edge_records(repaired)
        component_count = int(np.max(components)) + 1
        for component in range(component_count):
            component_faces = np.flatnonzero(components == component)
            component_edges = {
                key
                for key, incidents in repaired_records.items()
                if any(entry[0] in component_faces for entry in incidents)
            }
            closed = all(len(repaired_records[edge]) == 2 for edge in component_edges)
            if not closed:
                continue
            triangles = points[repaired[component_faces]]
            signed_volume = float(
                np.sum(
                    np.sum(
                        triangles[:, 0] * np.cross(triangles[:, 1], triangles[:, 2]),
                        axis=1,
                    )
                )
                / 6.0
            )
            if signed_volume < 0.0:
                repaired[component_faces] = repaired[component_faces][:, [0, 2, 1]]
                signs[component_faces] *= -1
    repair = SurfaceOrientationRepair(
        source_face_indices=np.arange(faces.shape[0], dtype=np.int64),
        orientation_signs=signs,
        component_ids=components,
        source_topology_id=canonical_fingerprint(
            {
                "kind": "surface-orientation-source-v1",
                "faces": array_tree_fingerprint(faces),
            }
        ),
        repaired_topology_id=canonical_fingerprint(
            {
                "kind": "surface-orientation-repaired-v1",
                "faces": array_tree_fingerprint(repaired),
            }
        ),
    )
    return repaired, repair


def _validate_triangle_arrays(
    coordinates: ArrayLike, triangles: ArrayLike, /
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(coordinates, dtype=float)
    faces = np.asarray(triangles)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != 3:
        raise SurfacePreparationError(
            SurfacePreparationStatus.INVALID_INPUT,
            "Surface coordinates must have shape (num_vertices > 0, 3).",
        )
    if not np.all(np.isfinite(points)):
        raise SurfacePreparationError(
            SurfacePreparationStatus.NONFINITE_GEOMETRY,
            "Surface coordinates must contain only finite values.",
        )
    if faces.ndim != 2 or faces.shape[0] == 0 or faces.shape[1] != 3:
        raise SurfacePreparationError(
            SurfacePreparationStatus.INVALID_INPUT,
            "Surface triangles must have shape (num_faces > 0, 3).",
        )
    if not np.issubdtype(faces.dtype, np.integer):
        raise SurfacePreparationError(
            SurfacePreparationStatus.INVALID_INPUT,
            "Surface triangle indices must be integers.",
        )
    faces = faces.astype(np.int32, copy=False)
    if np.any(faces < 0) or np.any(faces >= points.shape[0]):
        raise SurfacePreparationError(
            SurfacePreparationStatus.INVALID_INPUT,
            "Surface triangles index undeclared vertices.",
        )
    if np.any(np.diff(np.sort(faces, axis=1), axis=1) == 0):
        raise SurfacePreparationError(
            SurfacePreparationStatus.INVALID_INPUT,
            "Every surface triangle must reference three distinct vertices.",
        )
    if np.unique(np.sort(faces, axis=1), axis=0).shape[0] != faces.shape[0]:
        raise SurfacePreparationError(
            SurfacePreparationStatus.INVALID_INPUT,
            "Surface topology cannot contain duplicate triangles.",
        )
    return points, faces


def _validated_global_ids(
    name: str, value: ArrayLike | None, count: int, /
) -> np.ndarray | None:
    if value is None:
        return None
    identifiers = np.asarray(value)
    if identifiers.shape != (count,) or not np.issubdtype(identifiers.dtype, np.integer):
        raise SurfacePreparationError(
            SurfacePreparationStatus.INVALID_INPUT,
            f"{name} must be one integer ID per source entity.",
        )
    identifiers = identifiers.astype(np.int64, copy=False)
    if np.any(identifiers < 0) or np.unique(identifiers).size != count:
        raise SurfacePreparationError(
            SurfacePreparationStatus.INVALID_INPUT,
            f"{name} must contain unique non-negative IDs.",
        )
    return identifiers


def _boundary_loop_count(
    records: dict[tuple[int, int], list[tuple[int, int]]], /
) -> tuple[int, bool]:
    boundary = tuple(edge for edge, incidents in records.items() if len(incidents) == 1)
    if not boundary:
        return 0, True
    neighbours: dict[int, list[int]] = {}
    for start, stop in boundary:
        neighbours.setdefault(start, []).append(stop)
        neighbours.setdefault(stop, []).append(start)
    if any(len(values) != 2 for values in neighbours.values()):
        return 0, False
    remaining = set(neighbours)
    loop_count = 0
    while remaining:
        first = min(remaining)
        pending = [first]
        remaining.remove(first)
        while pending:
            vertex = pending.pop()
            for neighbour in neighbours[vertex]:
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    pending.append(neighbour)
        loop_count += 1
    return loop_count, True


def _surface_audit(
    model: SurfaceModel,
    mesh: CellMesh,
    policy: SurfaceAuditPolicy,
    /,
) -> SurfaceAuditReport:
    points = np.asarray(mesh.coordinates, dtype=float)
    faces = _triangle_faces(mesh)
    records = _edge_records(faces)
    components, component_count = _face_components(faces.shape[0], records)
    finite = bool(np.all(np.isfinite(points)))
    manifold = all(len(incidents) <= 2 for incidents in records.values())
    orientation_consistent = all(
        len(incidents) != 2 or incidents[0][1] != incidents[1][1]
        for incidents in records.values()
    )
    used_vertices = np.unique(faces.reshape((-1,)))
    boundary_loop_count, boundary_regular = _boundary_loop_count(records)
    topology_valid = bool(
        faces.ndim == 2
        and faces.shape[1] == 3
        and used_vertices.size == points.shape[0]
        and boundary_regular
    )

    triangles = points[faces]
    first_edges = triangles[:, 1] - triangles[:, 0]
    second_edges = triangles[:, 2] - triangles[:, 0]
    third_edges = triangles[:, 2] - triangles[:, 1]
    doubled_areas = np.linalg.norm(np.cross(first_edges, second_edges), axis=1)
    face_areas = 0.5 * doubled_areas
    maximum_edge_squared = np.maximum.reduce(
        (
            np.sum(first_edges * first_edges, axis=1),
            np.sum(second_edges * second_edges, axis=1),
            np.sum(third_edges * third_edges, axis=1),
        )
    )
    area_threshold = np.maximum(
        policy.minimum_face_area,
        0.5 * policy.relative_degeneracy_tolerance * maximum_edge_squared,
    )
    metric_valid = bool(
        finite and np.all(np.isfinite(face_areas)) and np.all(face_areas > area_threshold)
    )

    component_closed = np.ones((component_count,), dtype=bool)
    for incidents in records.values():
        if len(incidents) != 2:
            for face, _ in incidents:
                component_closed[components[face]] = False
    component_volumes = np.zeros((component_count,), dtype=float)
    signed_face_volumes = (
        np.sum(
            triangles[:, 0] * np.cross(triangles[:, 1], triangles[:, 2]),
            axis=1,
        )
        / 6.0
    )
    for component in range(component_count):
        if component_closed[component]:
            component_volumes[component] = float(
                np.sum(signed_face_volumes[components == component])
            )
    volume_valid = bool(
        np.all(
            (~component_closed)
            | (np.abs(component_volumes) >= policy.minimum_closed_volume)
        )
    )
    outward_orientation_valid = bool(
        volume_valid
        and (
            not policy.require_outward_orientation
            or np.all(
                component_closed & (component_volumes > policy.minimum_closed_volume)
            )
        )
    )

    boundary_edge_count = sum(len(incidents) == 1 for incidents in records.values())
    capacity_pairs = (
        (points.shape[0], policy.maximum_vertices),
        (faces.shape[0], policy.maximum_cells),
        (len(records), policy.maximum_edges),
        (component_count, policy.maximum_components),
        (boundary_edge_count, policy.maximum_boundary_edges),
    )
    capacity_valid = all(
        limit is None or count <= limit for count, limit in capacity_pairs
    )
    if model.interfaces:
        classification = "interface"
    elif component_count > 1:
        classification = "component"
    elif component_closed[0]:
        classification = "closed"
    else:
        classification = "open"

    issues: list[str] = []
    if not finite:
        issues.append("coordinates are non-finite")
    if not topology_valid:
        issues.append("topology has unused vertices or non-loop boundary incidence")
    if not manifold:
        issues.append("an edge has more than two incident cells")
    if not metric_valid:
        issues.append("one or more triangle metrics are degenerate or non-finite")
    if not orientation_consistent:
        issues.append("adjacent triangles have inconsistent orientation")
    if not capacity_valid:
        issues.append("surface exceeds an audit capacity")
    if not volume_valid:
        issues.append("a closed component is below the minimum enclosed volume")
    if policy.require_outward_orientation and not outward_orientation_valid:
        issues.append("closed components are not all outward oriented")
    if policy.require_closed and not bool(np.all(component_closed)):
        issues.append("audit policy requires every component to be closed")

    return SurfaceAuditReport(
        finite=finite,
        topology_valid=topology_valid,
        manifold=manifold,
        metric_valid=metric_valid,
        orientation_consistent=orientation_consistent,
        capacity_valid=capacity_valid,
        outward_orientation_valid=outward_orientation_valid,
        face_areas=face_areas,
        component_ids=components,
        component_closed=component_closed,
        component_signed_volumes=component_volumes,
        vertex_count=points.shape[0],
        edge_count=len(records),
        boundary_edge_count=boundary_edge_count,
        boundary_loop_count=boundary_loop_count,
        classification=classification,
        issues=issues,
        model_id=model.model_id,
        metadata_id=model.metadata.metadata_id,
        topology_id=mesh.topology_id,
        geometry_id=mesh.geometry_id,
        policy_id=policy.policy_id,
        policy_accepts_open=not policy.require_closed,
    )


class _CellMeshTriangleMap(AbstractBoundaryMap):
    """Concrete affine triangle provider backed directly by one CellMesh."""

    mesh: CellMesh

    def __init__(self, mesh: CellMesh, /):
        self.mesh = mesh

    @property
    def num_charts(self) -> int:
        return int(self.mesh.connectivity.cell_count)

    @property
    def reference_dimension(self) -> int:
        return 2

    @property
    def ambient_dimension(self) -> int:
        return 3

    def map(self, chart_indices: Array, reference: Array, /) -> Array:
        faces = self.mesh.connectivity.cell_vertices[chart_indices, :3]
        triangles = self.mesh.coordinates[faces]
        first = reference[..., :1]
        second = reference[..., 1:2]
        return (
            triangles[..., 0, :]
            + first * (triangles[..., 1, :] - triangles[..., 0, :])
            + (1.0 - first) * second * (triangles[..., 2, :] - triangles[..., 0, :])
        )

    def jacobian(self, chart_indices: Array, reference: Array, /) -> Array:
        faces = self.mesh.connectivity.cell_vertices[chart_indices, :3]
        triangles = self.mesh.coordinates[faces]
        doubled_area = jnp.linalg.norm(
            jnp.cross(
                triangles[..., 1, :] - triangles[..., 0, :],
                triangles[..., 2, :] - triangles[..., 0, :],
            ),
            axis=-1,
        )
        return doubled_area * (1.0 - reference[..., 0])


class SurfaceModel(StrictModule, NonTrainableState):
    """Authoritative triangular surface definition over canonical CellMesh topology."""

    mesh: CellMesh
    metadata: SurfaceMetadata
    selections: tuple[SurfaceSelection, ...]
    interfaces: tuple[SurfaceInterface, ...]
    orientation_repair: SurfaceOrientationRepair | None
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: CellMesh,
        metadata: SurfaceMetadata,
        /,
        *,
        selections: Sequence[SurfaceSelection] = (),
        interfaces: Sequence[SurfaceInterface] = (),
        orientation_repair: SurfaceOrientationRepair | None = None,
    ):
        if not isinstance(mesh, CellMesh):
            raise TypeError("SurfaceModel mesh must be a CellMesh.")
        if not isinstance(metadata, SurfaceMetadata):
            raise TypeError("SurfaceModel metadata must be SurfaceMetadata.")
        if mesh.topological_dimension != 2 or mesh.ambient_dimension != 3:
            raise SurfacePreparationError(
                SurfacePreparationStatus.UNSUPPORTED_TOPOLOGY,
                "SurfaceModel requires topological dimension 2 in ambient dimension 3.",
            )
        faces = _triangle_faces(mesh)
        if metadata.cell_tags and len(metadata.cell_tags) != faces.shape[0]:
            raise ValueError("Surface metadata cell_tags must contain one tag per cell.")
        selections_ = tuple(selections)
        interfaces_ = tuple(interfaces)
        if not all(isinstance(value, SurfaceSelection) for value in selections_):
            raise TypeError("selections must contain SurfaceSelection values.")
        if not all(isinstance(value, SurfaceInterface) for value in interfaces_):
            raise TypeError("interfaces must contain SurfaceInterface values.")
        if orientation_repair is not None and not isinstance(
            orientation_repair, SurfaceOrientationRepair
        ):
            raise TypeError(
                "orientation_repair must be SurfaceOrientationRepair or None."
            )
        if len({value.name for value in selections_}) != len(selections_):
            raise ValueError("Surface selection names must be unique.")
        if len({value.name for value in interfaces_}) != len(interfaces_):
            raise ValueError("Surface interface names must be unique.")
        cell_entities = mesh.entity_set(2)
        authoritative_ids = np.asarray(cell_entities.entity_ids, dtype=np.int64)
        for selection in (
            *selections_,
            *(interface.support for interface in interfaces_),
        ):
            if selection.cell_entity_set_id != cell_entities.entity_set_id:
                raise ValueError(
                    "Surface selection is bound to a different cell entity set."
                )
            if not np.all(
                np.isin(np.asarray(selection.cell_global_ids), authoritative_ids)
            ):
                raise ValueError("Surface selection contains non-authoritative cell IDs.")
        self.mesh = mesh
        self.metadata = metadata
        self.selections = selections_
        self.interfaces = interfaces_
        self.orientation_repair = orientation_repair
        self.model_id = canonical_fingerprint(
            {
                "kind": "surface-model-v1",
                "topology_id": mesh.topology_id,
                "metadata_id": metadata.metadata_id,
                "selection_ids": tuple(value.selection_id for value in selections_),
                "interface_ids": tuple(value.interface_id for value in interfaces_),
                "orientation_repair_id": (
                    None if orientation_repair is None else orientation_repair.repair_id
                ),
            }
        )

    @classmethod
    def from_triangles(
        cls,
        coordinates: ArrayLike,
        triangles: ArrayLike,
        metadata: SurfaceMetadata,
        /,
        *,
        vertex_global_ids: ArrayLike | None = None,
        cell_global_ids: ArrayLike | None = None,
        numeric_version: str = "0",
        repair_orientation: bool = False,
        orient_closed_outward: bool = True,
    ) -> SurfaceModel:
        if not isinstance(metadata, SurfaceMetadata):
            raise TypeError("metadata must be SurfaceMetadata.")
        points, faces = _validate_triangle_arrays(coordinates, triangles)
        signs, _, _ = _orientation_solution(faces)
        repair = None
        if repair_orientation:
            faces, repair = _repair_orientations(
                points,
                faces,
                orient_closed_outward=bool(orient_closed_outward),
            )
        elif np.any(signs < 0):
            raise SurfacePreparationError(
                SurfacePreparationStatus.INCONSISTENT_ORIENTATION,
                "Adjacent triangles require explicit orientation repair.",
            )
        vertex_ids = _validated_global_ids(
            "vertex_global_ids", vertex_global_ids, points.shape[0]
        )
        cell_ids = _validated_global_ids(
            "cell_global_ids", cell_global_ids, faces.shape[0]
        )
        version = str(numeric_version)
        if not version:
            raise SurfacePreparationError(
                SurfacePreparationStatus.INVALID_INPUT,
                "Surface numeric_version must be non-empty.",
            )
        mesh = CellMesh.from_triangles(
            points,
            faces,
            vertex_global_ids=vertex_ids,
            cell_global_ids=cell_ids,
            numeric_version=version,
        )
        return cls(mesh, metadata, orientation_repair=repair)

    def bind_selection(
        self,
        name: str,
        cell_global_ids: ArrayLike,
        /,
        *,
        role: str = "surface",
    ) -> SurfaceSelection:
        return SurfaceSelection(
            name,
            cell_global_ids,
            cell_entity_set_id=self.mesh.entity_set(2).entity_set_id,
            role=role,
        )

    def with_selection(self, selection: SurfaceSelection, /) -> SurfaceModel:
        if not isinstance(selection, SurfaceSelection):
            raise TypeError("selection must be a SurfaceSelection.")
        return SurfaceModel(
            self.mesh,
            self.metadata,
            selections=(*self.selections, selection),
            interfaces=self.interfaces,
            orientation_repair=self.orientation_repair,
        )

    def with_interface(self, interface: SurfaceInterface, /) -> SurfaceModel:
        if not isinstance(interface, SurfaceInterface):
            raise TypeError("interface must be a SurfaceInterface.")
        return SurfaceModel(
            self.mesh,
            self.metadata,
            selections=self.selections,
            interfaces=(*self.interfaces, interface),
            orientation_repair=self.orientation_repair,
        )

    def selection(self, name: str, /) -> SurfaceSelection:
        requested = str(name)
        for selection in self.selections:
            if selection.name == requested:
                return selection
        raise KeyError(f"Unknown surface selection {requested!r}.")

    def interface(self, name: str, /) -> SurfaceInterface:
        requested = str(name)
        for interface in self.interfaces:
            if interface.name == requested:
                return interface
        raise KeyError(f"Unknown surface interface {requested!r}.")

    def audit(self, policy: SurfaceAuditPolicy | None = None, /) -> SurfaceAuditReport:
        policy_ = SurfaceAuditPolicy() if policy is None else policy
        if not isinstance(policy_, SurfaceAuditPolicy):
            raise TypeError("policy must be SurfaceAuditPolicy or None.")
        return _surface_audit(self, self.mesh, policy_)

    def prepare(self, policy: SurfaceAuditPolicy | None = None, /) -> SurfaceRealization:
        policy_ = SurfaceAuditPolicy() if policy is None else policy
        if not isinstance(policy_, SurfaceAuditPolicy):
            raise TypeError("policy must be SurfaceAuditPolicy or None.")
        return SurfaceRealization._prepare(self, self.mesh, policy_)


class SurfaceRealization(StrictModule, NonTrainableState):
    """Audited fixed-topology execution realization with no coordinate copy."""

    model: SurfaceModel
    mesh: CellMesh
    policy: SurfaceAuditPolicy
    audit: SurfaceAuditReport
    certificate: SurfaceValidityCertificate
    chart_mapping: SurfaceChartMappingEvidence
    realization_id: str = eqx.field(static=True)

    def __init__(
        self,
        model: SurfaceModel,
        mesh: CellMesh,
        policy: SurfaceAuditPolicy,
        audit: SurfaceAuditReport,
        certificate: SurfaceValidityCertificate,
        chart_mapping: SurfaceChartMappingEvidence,
        /,
    ):
        if not isinstance(model, SurfaceModel) or not isinstance(mesh, CellMesh):
            raise TypeError("Surface realization requires SurfaceModel and CellMesh.")
        if not isinstance(policy, SurfaceAuditPolicy) or not isinstance(
            audit, SurfaceAuditReport
        ):
            raise TypeError("Surface realization requires policy and audit evidence.")
        if not isinstance(certificate, SurfaceValidityCertificate) or not isinstance(
            chart_mapping, SurfaceChartMappingEvidence
        ):
            raise TypeError(
                "Surface realization requires certificate and chart mapping evidence."
            )
        if mesh.topology_id != model.mesh.topology_id:
            raise SurfacePreparationError(
                SurfacePreparationStatus.TOPOLOGY_CHANGED,
                "Surface realization topology differs from its authoritative model.",
            )
        if (
            audit.model_id != model.model_id
            or audit.metadata_id != model.metadata.metadata_id
            or certificate.model_id != model.model_id
            or certificate.metadata_id != model.metadata.metadata_id
            or audit.topology_id != mesh.topology_id
            or audit.geometry_id != mesh.geometry_id
            or audit.policy_id != policy.policy_id
            or certificate.report_id != audit.report_id
            or certificate.topology_id != mesh.topology_id
            or certificate.geometry_id != mesh.geometry_id
        ):
            raise ValueError("Surface realization evidence is not bound to its mesh.")
        cell_entities = mesh.entity_set(2)
        if (
            chart_mapping.cell_entity_set_id != cell_entities.entity_set_id
            or not np.array_equal(
                np.asarray(chart_mapping.cell_global_ids, dtype=np.int64),
                np.asarray(cell_entities.entity_ids, dtype=np.int64),
            )
        ):
            raise ValueError(
                "Surface chart mapping is not bound to the mesh cell entity set."
            )
        self.model = model
        self.mesh = mesh
        self.policy = policy
        self.audit = audit
        self.certificate = certificate
        self.chart_mapping = chart_mapping
        self.realization_id = canonical_fingerprint(
            {
                "kind": "surface-realization-v1",
                "model_id": model.model_id,
                "geometry_id": mesh.geometry_id,
                "certificate_id": certificate.certificate_id,
                "chart_mapping_id": chart_mapping.mapping_id,
            }
        )

    @classmethod
    def _prepare(
        cls,
        model: SurfaceModel,
        mesh: CellMesh,
        policy: SurfaceAuditPolicy,
        /,
    ) -> SurfaceRealization:
        report = _surface_audit(model, mesh, policy)
        if not bool(np.asarray(report.valid)):
            raise SurfacePreparationError(
                SurfacePreparationStatus.AUDIT_REJECTED,
                "Surface realization was rejected by its audit policy.",
                issues=report.issues,
                report=report,
            )
        certificate = SurfaceValidityCertificate(report)
        cell_entities = mesh.entity_set(2)
        cell_ids = np.asarray(cell_entities.entity_ids, dtype=np.int64)
        chart_mapping = SurfaceChartMappingEvidence(
            chart_ids=np.arange(cell_ids.size, dtype=np.int32),
            cell_global_ids=cell_ids,
            cell_entity_set_id=cell_entities.entity_set_id,
        )
        return cls(model, mesh, policy, report, certificate, chart_mapping)

    @property
    def metadata(self) -> SurfaceMetadata:
        return self.model.metadata

    @property
    def selections(self) -> tuple[SurfaceSelection, ...]:
        return self.model.selections

    @property
    def interfaces(self) -> tuple[SurfaceInterface, ...]:
        return self.model.interfaces

    @property
    def classification(self) -> str:
        return self.audit.classification

    def selection(self, name: str, /) -> SurfaceSelection:
        return self.model.selection(name)

    def interface(self, name: str, /) -> SurfaceInterface:
        return self.model.interface(name)

    @property
    def boundary_atlas(self) -> BoundaryAtlas:
        tags = (
            self.model.metadata.cell_tags
            if self.model.metadata.cell_tags
            else tuple("surface" for _ in range(self.chart_mapping.chart_ids.size))
        )
        return BoundaryAtlas(
            _CellMeshTriangleMap(self.mesh),
            source_entity_ids=self.chart_mapping.chart_ids,
            source_id=self.mesh.geometry_id,
            physical_tags=tags,
        )

    def chart_ids_for(self, selection: SurfaceSelection, /) -> Array:
        if not isinstance(selection, SurfaceSelection):
            raise TypeError("selection must be a SurfaceSelection.")
        if selection.cell_entity_set_id != self.chart_mapping.cell_entity_set_id:
            raise ValueError("Surface selection is not bound to this chart mapping.")
        mask = np.isin(
            np.asarray(self.chart_mapping.cell_global_ids, dtype=np.int64),
            np.asarray(selection.cell_global_ids, dtype=np.int64),
        )
        if np.count_nonzero(mask) != selection.cell_global_ids.size:
            raise ValueError(
                "Surface selection is not fully represented by this realization."
            )
        return self.chart_mapping.chart_ids[jnp.asarray(mask)]

    def selection_atlas(self, selection: SurfaceSelection | str, /) -> BoundaryAtlas:
        selection_ = (
            self.model.selection(selection) if isinstance(selection, str) else selection
        )
        chart_ids = self.chart_ids_for(selection_)
        return self.boundary_atlas.select(entity_ids=np.asarray(chart_ids).tolist())

    def interface_atlas(self, interface: SurfaceInterface | str, /) -> BoundaryAtlas:
        interface_ = (
            self.model.interface(interface) if isinstance(interface, str) else interface
        )
        if not isinstance(interface_, SurfaceInterface):
            raise TypeError("interface must be SurfaceInterface or an interface name.")
        return self.selection_atlas(interface_.support)

    def refresh(
        self,
        coordinates: ArrayLike,
        /,
        *,
        numeric_version: str,
        policy: SurfaceAuditPolicy | None = None,
    ) -> SurfaceRealization:
        points = np.asarray(coordinates, dtype=float)
        if points.shape != self.mesh.coordinates.shape:
            raise SurfacePreparationError(
                SurfacePreparationStatus.TOPOLOGY_CHANGED,
                "Fixed-topology refresh must preserve the coordinate array shape.",
            )
        if not np.all(np.isfinite(points)):
            raise SurfacePreparationError(
                SurfacePreparationStatus.NONFINITE_GEOMETRY,
                "Fixed-topology refresh coordinates must be finite.",
            )
        version = str(numeric_version)
        if not version:
            raise SurfacePreparationError(
                SurfacePreparationStatus.INVALID_INPUT,
                "Surface numeric_version must be non-empty.",
            )
        refreshed = self.mesh.with_coordinates(points, numeric_version=version)
        if refreshed.topology_id != self.mesh.topology_id:
            raise SurfacePreparationError(
                SurfacePreparationStatus.TOPOLOGY_CHANGED,
                "Fixed-topology refresh changed the CellMesh topology identity.",
            )
        policy_ = self.policy if policy is None else policy
        if not isinstance(policy_, SurfaceAuditPolicy):
            raise TypeError("policy must be SurfaceAuditPolicy or None.")
        return SurfaceRealization._prepare(self.model, refreshed, policy_)

    def intersect_plane(
        self,
        origin: ArrayLike,
        normal: ArrayLike,
        /,
        *,
        tolerance: float | None = None,
    ) -> PlaneSurfaceSection:
        from ._intersection import intersect_plane_surface

        return intersect_plane_surface(self, origin, normal, tolerance=tolerance)


__all__ = ["SurfaceModel", "SurfaceRealization"]
