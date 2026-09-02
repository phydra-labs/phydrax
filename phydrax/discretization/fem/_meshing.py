#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module, util
from pathlib import Path
from typing import Literal

import equinox as eqx
import numpy as np

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...geometry.brep import BRepEntityId, BRepSource
from ...geometry.brep._occt import read_occt_shape
from ...geometry.surface import SurfaceMetadata, SurfaceModel
from .._cell_mesh import CellMesh


CADFEMCellFamily = Literal["triangle", "tetrahedron"]


class CADFEMMeshingPolicy(StrictModule, NonTrainableState):
    """Deterministic bounded envelope for one CAD-to-FEM mesh generation."""

    provider: str = eqx.field(static=True)
    topological_dimension: int = eqx.field(static=True)
    cell_family: CADFEMCellFamily = eqx.field(static=True)
    target_size: float = eqx.field(static=True)
    minimum_size: float = eqx.field(static=True)
    maximum_size: float = eqx.field(static=True)
    curvature_sizing: bool = eqx.field(static=True)
    proximity_sizing: bool = eqx.field(static=True)
    maximum_growth_rate: float = eqx.field(static=True)
    algorithm_2d: int = eqx.field(static=True)
    algorithm_3d: int = eqx.field(static=True)
    length_unit: str = eqx.field(static=True)
    maximum_entities: int = eqx.field(static=True)
    maximum_vertices: int = eqx.field(static=True)
    maximum_elements: int = eqx.field(static=True)
    maximum_conversion_bytes: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        topological_dimension: int,
        cell_family: CADFEMCellFamily,
        target_size: float,
        provider: str = "gmsh",
        minimum_size: float | None = None,
        maximum_size: float | None = None,
        curvature_sizing: bool = True,
        proximity_sizing: bool = False,
        maximum_growth_rate: float = 1.3,
        algorithm_2d: int = 6,
        algorithm_3d: int = 1,
        length_unit: str = "m",
        maximum_entities: int = 100_000,
        maximum_vertices: int = 2_000_000,
        maximum_elements: int = 5_000_000,
        maximum_conversion_bytes: int = 2_000_000_000,
    ):
        dimension = int(topological_dimension)
        family = str(cell_family)
        size = float(target_size)
        minimum = size if minimum_size is None else float(minimum_size)
        maximum = size if maximum_size is None else float(maximum_size)
        growth = float(maximum_growth_rate)
        if str(provider) != "gmsh":
            raise ValueError("The bounded CAD meshing provider is exactly 'gmsh'.")
        if dimension not in (2, 3):
            raise ValueError("topological_dimension must be two or three.")
        expected = "triangle" if dimension == 2 else "tetrahedron"
        if family != expected:
            raise ValueError(
                f"The bounded dimension-{dimension} tuple requires {expected} cells."
            )
        if (
            not np.isfinite(size)
            or not np.isfinite(minimum)
            or not np.isfinite(maximum)
            or minimum <= 0.0
            or size <= 0.0
            or maximum < size
            or minimum > size
        ):
            raise ValueError("CAD mesh sizes must satisfy 0 < min <= target <= max.")
        if not np.isfinite(growth) or growth < 1.0:
            raise ValueError("maximum_growth_rate must be finite and at least one.")
        capacities = (
            int(maximum_entities),
            int(maximum_vertices),
            int(maximum_elements),
            int(maximum_conversion_bytes),
        )
        if any(value <= 0 for value in capacities):
            raise ValueError("CAD meshing capacities must be positive.")
        unit = str(length_unit)
        if not unit:
            raise ValueError("length_unit must be non-empty.")
        self.provider = "gmsh"
        self.topological_dimension = dimension
        self.cell_family = family
        self.target_size = size
        self.minimum_size = minimum
        self.maximum_size = maximum
        self.curvature_sizing = bool(curvature_sizing)
        self.proximity_sizing = bool(proximity_sizing)
        self.maximum_growth_rate = growth
        self.algorithm_2d = int(algorithm_2d)
        self.algorithm_3d = int(algorithm_3d)
        self.length_unit = unit
        self.maximum_entities = capacities[0]
        self.maximum_vertices = capacities[1]
        self.maximum_elements = capacities[2]
        self.maximum_conversion_bytes = capacities[3]
        self.policy_id = canonical_fingerprint(
            {
                "kind": "cad-fem-meshing-policy",
                "provider": "gmsh",
                "dimension": dimension,
                "family": family,
                "sizes": [minimum, size, maximum],
                "curvature": bool(curvature_sizing),
                "proximity": bool(proximity_sizing),
                "growth": growth,
                "algorithms": [int(algorithm_2d), int(algorithm_3d)],
                "unit": unit,
                "capacities": list(capacities),
            }
        )


@dataclass(frozen=True, slots=True)
class CADMeshAssociation:
    """Stable CAD ownership for generated vertices, cells, and boundary faces."""

    vertex_entities: tuple[BRepEntityId | None, ...]
    cell_entities: tuple[BRepEntityId, ...]
    boundary_face_entities: tuple[BRepEntityId, ...]
    cell_physical_tags: tuple[int, ...]
    boundary_physical_tags: tuple[int, ...]
    unresolved_vertex_count: int
    ambiguous_vertex_count: int
    association_id: str


class CADFEMMeshEvidence(StrictModule, NonTrainableState):
    """Provenance, geometry, conformity, and resource evidence for one mesh."""

    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    provider: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    vertex_count: int = eqx.field(static=True)
    cell_count: int = eqx.field(static=True)
    boundary_face_count: int = eqx.field(static=True)
    conversion_bytes: int = eqx.field(static=True)
    minimum_jacobian: float = eqx.field(static=True)
    maximum_jacobian: float = eqx.field(static=True)
    total_measure: float = eqx.field(static=True)
    maximum_boundary_residual: float = eqx.field(static=True)
    outward_oriented: bool = eqx.field(static=True)
    conforming: bool = eqx.field(static=True)
    complete_association: bool = eqx.field(static=True)
    provider_workspace_bounded: bool = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class CADFEMMeshResult(StrictModule, NonTrainableState):
    mesh: CellMesh
    boundary: SurfaceModel
    association: CADMeshAssociation = eqx.field(static=True)
    evidence: CADFEMMeshEvidence


@dataclass(frozen=True, slots=True)
class _ElementRows:
    tags: np.ndarray
    vertices: np.ndarray
    entity_tags: np.ndarray


def _element_rows(gmsh, dimension: int, expected_type: int, /) -> _ElementRows:
    records: list[tuple[int, np.ndarray, int]] = []
    for _, entity_tag in sorted(gmsh.model.getEntities(dimension)):
        element_types, element_tags, node_tags = gmsh.model.mesh.getElements(
            dimension, entity_tag
        )
        for element_type, tags, nodes in zip(
            element_types, element_tags, node_tags, strict=True
        ):
            if int(element_type) != expected_type:
                raise ValueError(
                    f"Gmsh produced unsupported element type {int(element_type)}."
                )
            properties = gmsh.model.mesh.getElementProperties(int(element_type))
            node_count = int(properties[3])
            rows = np.asarray(nodes, dtype=np.int64).reshape((-1, node_count))
            for tag, row in zip(np.asarray(tags, dtype=np.int64), rows, strict=True):
                records.append((int(tag), row, int(entity_tag)))
    if not records:
        raise ValueError(f"Gmsh generated no dimension-{dimension} elements.")
    records.sort(key=lambda value: value[0])
    return _ElementRows(
        tags=np.asarray([value[0] for value in records], dtype=np.int64),
        vertices=np.stack([value[1] for value in records]),
        entity_tags=np.asarray([value[2] for value in records], dtype=np.int32),
    )


def _oriented_tetrahedra(
    points: np.ndarray, cells: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    tetrahedra = points[cells]
    matrices = np.stack(
        (
            tetrahedra[:, 1] - tetrahedra[:, 0],
            tetrahedra[:, 2] - tetrahedra[:, 0],
            tetrahedra[:, 3] - tetrahedra[:, 0],
        ),
        axis=-1,
    )
    determinants = np.linalg.det(matrices)
    if np.any(~np.isfinite(determinants)) or np.any(determinants == 0.0):
        raise ValueError("Gmsh generated a degenerate tetrahedron.")
    oriented = cells.copy()
    negative = determinants < 0.0
    oriented[negative, 2], oriented[negative, 3] = (
        cells[negative, 3],
        cells[negative, 2],
    )
    return oriented, np.abs(determinants)


def _triangle_measures(points: np.ndarray, faces: np.ndarray, /) -> np.ndarray:
    triangles = points[faces]
    doubled = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=-1,
    )
    if np.any(~np.isfinite(doubled)) or np.any(doubled <= 0.0):
        raise ValueError("Gmsh generated a degenerate triangle.")
    return 0.5 * doubled


def _entity_ids(
    revision: str, kind: str, entity_tags: np.ndarray, /
) -> tuple[BRepEntityId, ...]:
    return tuple(BRepEntityId(revision, kind, int(tag) - 1) for tag in entity_tags)


def mesh_brep_for_fem(
    source: BRepSource,
    policy: CADFEMMeshingPolicy,
    /,
) -> CADFEMMeshResult:
    """Generate one revision-checked affine triangle/tetrahedron FEM mesh.

    This host-only operation has no query-tessellation fallback.  Gmsh owns CAD
    meshing; PhydraX bounds and validates the converted canonical result.  Gmsh's
    internal workspace is not hard-capped by its Python API and is reported
    truthfully as unbounded; element/count/converted-byte caps are enforced.
    """

    if not isinstance(source, BRepSource):
        raise TypeError("source must be BRepSource.")
    if not isinstance(policy, CADFEMMeshingPolicy):
        raise TypeError("policy must be CADFEMMeshingPolicy.")
    report = source.report
    source_path = Path(report.source_id)
    if not source_path.is_file():
        raise ValueError(
            "CAD FEM meshing requires a reopenable STEP/IGES/BREP source path."
        )
    _, source_format, current_revision = read_occt_shape(source_path)
    if (
        current_revision != report.source_revision
        or source_format != report.source_format
    ):
        raise ValueError("The CAD source revision changed since BRep import.")
    if (
        report.num_faces + report.num_edges + report.num_vertices
        > policy.maximum_entities
    ):
        raise ValueError("CAD entity capacity is exceeded before provider meshing.")
    if util.find_spec("gmsh") is None:
        raise RuntimeError("The requested gmsh CAD meshing provider is unavailable.")
    gmsh = import_module("gmsh")
    if gmsh.isInitialized():
        raise RuntimeError("Deterministic CAD meshing requires an unowned Gmsh session.")

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.option.setNumber("Mesh.Algorithm", policy.algorithm_2d)
        gmsh.option.setNumber("Mesh.Algorithm3D", policy.algorithm_3d)
        gmsh.option.setNumber("Mesh.MeshSizeMin", policy.minimum_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", policy.maximum_size)
        gmsh.option.setNumber(
            "Mesh.MeshSizeFromCurvature", 1 if policy.curvature_sizing else 0
        )
        gmsh.option.setNumber(
            "Mesh.MeshSizeFromPoints", 1 if policy.proximity_sizing else 0
        )
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.model.add("phydrax-cad-fem")
        imported = gmsh.model.occ.importShapes(str(source_path))
        gmsh.model.occ.synchronize()
        top_entities = sorted(gmsh.model.getEntities(policy.topological_dimension))
        if not top_entities:
            raise ValueError("The CAD source has no requested top-dimensional entities.")
        if len(gmsh.model.getEntities()) > policy.maximum_entities:
            raise ValueError("Imported CAD entities exceed maximum_entities.")
        for dimension, tag in top_entities:
            gmsh.model.addPhysicalGroup(dimension, [tag], tag=tag)
        gmsh.model.mesh.generate(policy.topological_dimension)

        node_tags, node_coordinates, _ = gmsh.model.mesh.getNodes()
        node_tags = np.asarray(node_tags, dtype=np.int64)
        order = np.argsort(node_tags, kind="stable")
        node_tags = node_tags[order]
        points = np.asarray(node_coordinates, dtype=float).reshape((-1, 3))[order]
        if points.shape[0] > policy.maximum_vertices:
            raise ValueError("Generated CAD mesh exceeds maximum_vertices.")
        node_to_local = {int(tag): index for index, tag in enumerate(node_tags)}

        top_type = 2 if policy.topological_dimension == 2 else 4
        top = _element_rows(gmsh, policy.topological_dimension, top_type)
        if top.tags.size > policy.maximum_elements:
            raise ValueError("Generated CAD mesh exceeds maximum_elements.")
        top_vertices = np.asarray(
            [[node_to_local[int(tag)] for tag in row] for row in top.vertices],
            dtype=np.int32,
        )
        if policy.topological_dimension == 3:
            boundary_rows = _element_rows(gmsh, 2, 2)
            boundary_vertices = np.asarray(
                [
                    [node_to_local[int(tag)] for tag in row]
                    for row in boundary_rows.vertices
                ],
                dtype=np.int32,
            )
            top_vertices, jacobians = _oriented_tetrahedra(points, top_vertices)
            total_measure = float(np.sum(jacobians) / 6.0)
            minimum_jacobian = float(np.min(jacobians))
            maximum_jacobian = float(np.max(jacobians))
            mesh = CellMesh.from_tetrahedra(
                points,
                top_vertices,
                vertex_global_ids=node_tags,
                cell_global_ids=top.tags,
                numeric_version=report.source_revision,
            )
        else:
            boundary_rows = top
            boundary_vertices = top_vertices
            areas = _triangle_measures(points, top_vertices)
            total_measure = float(np.sum(areas))
            minimum_jacobian = float(np.min(2.0 * areas))
            maximum_jacobian = float(np.max(2.0 * areas))
            mesh = CellMesh.from_triangles(
                points,
                top_vertices,
                vertex_global_ids=node_tags,
                cell_global_ids=top.tags,
                numeric_version=report.source_revision,
            )

        face_tags = tuple(f"brep-face:{int(tag)}" for tag in boundary_rows.entity_tags)
        metadata = SurfaceMetadata(
            source_id=report.source_id,
            source_revision=report.source_revision,
            length_unit=policy.length_unit,
            provenance=("gmsh-occ-affine", policy.policy_id),
            cell_tags=face_tags,
        )
        boundary = SurfaceModel.from_triangles(
            points,
            boundary_vertices,
            metadata,
            vertex_global_ids=node_tags,
            cell_global_ids=boundary_rows.tags,
            numeric_version=report.source_revision,
            repair_orientation=True,
            orient_closed_outward=policy.topological_dimension == 3,
        )
        if policy.topological_dimension == 2:
            mesh = boundary.mesh

        vertex_owners: list[set[tuple[int, int]]] = [set() for _ in node_tags]
        boundary_residual = 0.0
        for dimension in range(policy.topological_dimension + 1):
            for _, entity_tag in sorted(gmsh.model.getEntities(dimension)):
                tags, coordinates, _ = gmsh.model.mesh.getNodes(
                    dimension, entity_tag, True, False
                )
                tags_ = np.asarray(tags, dtype=np.int64)
                coordinates_ = np.asarray(coordinates, dtype=float).reshape((-1, 3))
                for tag in tags_:
                    local = node_to_local.get(int(tag))
                    if local is not None:
                        vertex_owners[local].add((dimension, int(entity_tag)))
                if dimension == 2 and coordinates_.size:
                    closest, _ = gmsh.model.getClosestPoint(
                        dimension, entity_tag, coordinates_.reshape((-1,)).tolist()
                    )
                    closest_ = np.asarray(closest, dtype=float).reshape((-1, 3))
                    boundary_residual = max(
                        boundary_residual,
                        float(np.max(np.linalg.norm(coordinates_ - closest_, axis=1))),
                    )
        vertex_entities: list[BRepEntityId | None] = []
        ambiguous = 0
        unresolved = 0
        kind_for_dimension = {0: "vertex", 1: "edge", 2: "face", 3: "solid"}
        for owners in vertex_owners:
            if not owners:
                vertex_entities.append(None)
                unresolved += 1
                continue
            minimum_dimension = min(value[0] for value in owners)
            selected = sorted(value for value in owners if value[0] == minimum_dimension)
            ambiguous += int(len(selected) > 1)
            dimension, tag = selected[0]
            vertex_entities.append(
                BRepEntityId(
                    report.source_revision, kind_for_dimension[dimension], tag - 1
                )
            )

        conversion_bytes = int(
            points.nbytes
            + top_vertices.nbytes
            + boundary_vertices.nbytes
            + node_tags.nbytes
            + top.tags.nbytes
            + boundary_rows.tags.nbytes
        )
        if conversion_bytes > policy.maximum_conversion_bytes:
            raise ValueError("Generated CAD mesh exceeds maximum_conversion_bytes.")
        cell_kind = "face" if policy.topological_dimension == 2 else "solid"
        association_id = canonical_fingerprint(
            {
                "kind": "cad-mesh-association",
                "source_revision": report.source_revision,
                "vertices": [
                    None if value is None else [value.kind, value.index]
                    for value in vertex_entities
                ],
                "cell_entities": array_tree_fingerprint(top.entity_tags),
                "boundary_entities": array_tree_fingerprint(boundary_rows.entity_tags),
                "cell_tags": array_tree_fingerprint(top.tags),
                "boundary_tags": array_tree_fingerprint(boundary_rows.tags),
            }
        )
        association = CADMeshAssociation(
            vertex_entities=tuple(vertex_entities),
            cell_entities=_entity_ids(report.source_revision, cell_kind, top.entity_tags),
            boundary_face_entities=_entity_ids(
                report.source_revision, "face", boundary_rows.entity_tags
            ),
            cell_physical_tags=tuple(int(value) for value in top.entity_tags),
            boundary_physical_tags=tuple(
                int(value) for value in boundary_rows.entity_tags
            ),
            unresolved_vertex_count=unresolved,
            ambiguous_vertex_count=ambiguous,
            association_id=association_id,
        )
        complete_association = unresolved == 0
        evidence_id = canonical_fingerprint(
            {
                "kind": "cad-fem-mesh-evidence",
                "source_revision": report.source_revision,
                "policy": policy.policy_id,
                "mesh": mesh.mesh_id,
                "boundary": boundary.model_id,
                "association": association_id,
                "counts": [points.shape[0], top.tags.size, boundary_rows.tags.size],
                "jacobian": [minimum_jacobian, maximum_jacobian],
                "measure": total_measure,
                "boundary_residual": boundary_residual,
                "conversion_bytes": conversion_bytes,
                "provider_workspace_bounded": False,
            }
        )
        evidence = CADFEMMeshEvidence(
            source_id=report.source_id,
            source_revision=report.source_revision,
            provider="gmsh",
            policy_id=policy.policy_id,
            vertex_count=int(points.shape[0]),
            cell_count=int(top.tags.size),
            boundary_face_count=int(boundary_rows.tags.size),
            conversion_bytes=conversion_bytes,
            minimum_jacobian=minimum_jacobian,
            maximum_jacobian=maximum_jacobian,
            total_measure=total_measure,
            maximum_boundary_residual=boundary_residual,
            outward_oriented=True,
            conforming=True,
            complete_association=complete_association,
            provider_workspace_bounded=False,
            evidence_id=evidence_id,
        )
        if not complete_association:
            raise ValueError("Generated mesh contains vertices without CAD association.")
        return CADFEMMeshResult(
            mesh=mesh,
            boundary=boundary,
            association=association,
            evidence=evidence,
        )
    finally:
        gmsh.finalize()


__all__ = [
    "CADFEMCellFamily",
    "CADFEMMeshingPolicy",
    "CADFEMMeshEvidence",
    "CADFEMMeshResult",
    "CADMeshAssociation",
    "mesh_brep_for_fem",
]
