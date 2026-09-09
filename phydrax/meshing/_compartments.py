#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Compartment-aware multi-surface tetrahedralization and zone certification."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from importlib import import_module, metadata

import numpy as np

from .._fingerprint import canonical_fingerprint
from .._identity import SemanticProvenance
from ..discretization import CellMesh, TetrahedralConnectivity
from ..geometry import CompartmentComplex
from ..geometry.surface import SurfaceModel
from ..imaging import CompartmentSurfaceResult, LabelVolume
from ._audit import CellMeshAuditPolicy
from ._canonical import certify_cell_mesh
from ._contracts import (
    MeshingDerivativeMode,
    MeshingExecutionMode,
    MeshingFailure,
    MeshingFailureCategory,
    MeshingLimits,
)
from ._organization import MeshPatch, MeshZone, MeshZoneRole, validate_mesh_zones
from ._result import CellMeshingResult, MeshingComplianceReport, MeshingRuntimeInfo
from ._scope import MeshingEntityKind, MeshingScope
from ._trace import MeshingStageKind, MeshingStageReport, MeshingStageStatus, MeshingTrace
from .providers._ftetwild import FTetWildOptions, FTetWildProvider


_COMPARTMENT_FTETWILD_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class CompartmentMeshingSpec:
    labels: LabelVolume
    compartments: CompartmentComplex
    outer_surface: SurfaceModel
    interfaces: CompartmentSurfaceResult
    target_edge_length: float
    options: FTetWildOptions = field(default_factory=FTetWildOptions)
    limits: MeshingLimits = field(default_factory=MeshingLimits)
    audit_policy: CellMeshAuditPolicy = field(default_factory=CellMeshAuditPolicy)
    specification_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.labels, LabelVolume):
            raise TypeError("labels must be LabelVolume.")
        if not isinstance(self.compartments, CompartmentComplex):
            raise TypeError("compartments must be CompartmentComplex.")
        if self.compartments.source_revision != self.labels.label_volume_id:
            raise ValueError("Compartment and label revisions differ.")
        self.compartments.require_valid_adjacency()
        if not isinstance(self.outer_surface, SurfaceModel):
            raise TypeError("outer_surface must be SurfaceModel.")
        if not isinstance(self.interfaces, CompartmentSurfaceResult):
            raise TypeError("interfaces must be CompartmentSurfaceResult.")
        if self.interfaces.complex.complex_id != self.compartments.complex_id:
            raise ValueError("Interface surfaces and compartment complex differ.")
        spatial = self.labels.asset.spatial_affine.coordinate_contract.spatial_id
        if self.outer_surface.metadata.coordinate_contract.spatial_id != spatial or any(
            value.surface.metadata.coordinate_contract.spatial_id != spatial
            for value in self.interfaces.surfaces
        ):
            raise ValueError(
                "Compartment surfaces and labels require one spatial contract."
            )
        target = float(self.target_edge_length)
        if not np.isfinite(target) or target <= 0.0:
            raise ValueError("target_edge_length must be finite and positive.")
        if not isinstance(self.options, FTetWildOptions):
            raise TypeError("options must be FTetWildOptions.")
        if not isinstance(self.limits, MeshingLimits):
            raise TypeError("limits must be MeshingLimits.")
        if not isinstance(self.audit_policy, CellMeshAuditPolicy):
            raise TypeError("audit_policy must be CellMeshAuditPolicy.")
        object.__setattr__(self, "target_edge_length", target)
        object.__setattr__(
            self,
            "specification_id",
            canonical_fingerprint(
                {
                    "kind": "compartment-meshing-specification",
                    "labels": self.labels.label_volume_id,
                    "compartments": self.compartments.complex_id,
                    "outer_surface": self.outer_surface.mesh.mesh_id,
                    "interfaces": self.interfaces.extraction_id,
                    "target_edge_length": target.hex(),
                    "options": (
                        self.options.envelope_distance,
                        self.options.maximum_iterations,
                        self.options.stop_quality,
                        self.options.maximum_threads,
                        self.options.skip_simplify,
                        self.options.coarsen,
                    ),
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class CompartmentMeshingResult:
    result: CellMeshingResult
    zones: tuple[MeshZone, ...]
    interfaces: tuple[MeshPatch, ...]
    cell_compartment_ids: tuple[str, ...]
    adjacency_pairs: tuple[tuple[str, str], ...]
    specification_id: str
    result_id: str


def _surface_arrays(surface: SurfaceModel, /) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(surface.mesh.coordinates, dtype=np.float64)
    faces = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int32) for block in surface.mesh.blocks]
    )
    return points, faces


class FTetWildCompartmentProvider:
    """Insert all compartment surfaces once, then classify tetrahedra by labels."""

    def execute(
        self, specification: CompartmentMeshingSpec, /
    ) -> CompartmentMeshingResult:
        if not isinstance(specification, CompartmentMeshingSpec):
            raise TypeError("specification must be CompartmentMeshingSpec.")
        surfaces = (specification.outer_surface,) + tuple(
            value.surface for value in specification.interfaces.surfaces
        )
        vertices, faces = zip(
            *(_surface_arrays(value) for value in surfaces), strict=True
        )
        outer_points = vertices[0]
        diagonal = float(np.linalg.norm(np.ptp(outer_points, axis=0)))
        if not np.isfinite(diagonal) or diagonal <= 0.0:
            raise MeshingFailure(
                MeshingFailureCategory.INVALID_SOURCE,
                "Compartment outer-surface extent is invalid.",
            )
        if sum(len(value) for value in vertices) > specification.limits.maximum_vertices:
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "Compartment source exceeds the vertex budget.",
            )
        try:
            native = import_module("wildmeshing")
            version = metadata.version("wildmeshing")
        except (ImportError, metadata.PackageNotFoundError) as error:
            raise MeshingFailure(
                MeshingFailureCategory.PROVIDER_UNAVAILABLE,
                "Install wildmeshing>=0.4.1 for compartment tetrahedralization.",
            ) from error
        options = specification.options
        with _COMPARTMENT_FTETWILD_LOCK:
            try:
                tetrahedralizer = native.Tetrahedralizer(
                    stop_quality=options.stop_quality,
                    max_its=options.maximum_iterations,
                    max_threads=options.maximum_threads,
                    epsilon=options.envelope_distance / diagonal,
                    edge_length_r=specification.target_edge_length / diagonal,
                    skip_simplify=options.skip_simplify,
                    coarsen=options.coarsen,
                )
                tetrahedralizer.set_log_level(6)
                tetrahedralizer.set_meshes(list(vertices), list(faces))
                tetrahedralizer.tetrahedralize()
                output_points, output_cells, _ = tetrahedralizer.get_tet_mesh(
                    all_mesh=False,
                    use_input_for_wn=True,
                    manifold_surface=True,
                    correct_surface_orientation=True,
                )
            except (RuntimeError, ValueError, TypeError) as error:
                raise MeshingFailure(
                    MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                    f"fTetWild {version}: {error}",
                ) from error
        points = np.asarray(output_points, dtype=float)
        cells = np.asarray(output_cells)
        if (
            cells.ndim != 2
            or cells.shape[1] != 4
            or not np.issubdtype(cells.dtype, np.integer)
        ):
            raise MeshingFailure(
                MeshingFailureCategory.CONVERSION_FAILED,
                "fTetWild returned non-tetrahedral compartment output.",
            )
        used, remapped = np.unique(cells, return_inverse=True)
        points = points[used]
        cells = remapped.reshape(cells.shape)
        determinant = np.linalg.det(points[cells[:, 1:]] - points[cells[:, :1]])
        cells[determinant < 0.0, :2] = cells[determinant < 0.0, 1::-1]
        mesh = CellMesh.from_tetrahedra(
            points,
            cells,
            numeric_version=specification.specification_id,
        )
        if (
            len(points) > specification.limits.maximum_vertices
            or len(cells) > specification.limits.maximum_cells
        ):
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "Compartment output exceeds entity budgets.",
            )
        certified = certify_cell_mesh(
            mesh,
            specification.labels.asset.spatial_affine.coordinate_contract,
            audit_policy=specification.audit_policy,
        )
        mesh = certified.mesh
        points = np.asarray(mesh.coordinates)
        cells = np.concatenate([np.asarray(block.vertices) for block in mesh.blocks])
        centroids = np.mean(points[cells], axis=1)
        voxel = specification.labels.asset.spatial_affine.world_to_index(centroids)
        nearest = np.rint(voxel).astype(np.int64)
        image_shape = np.asarray(specification.labels.asset.values.shape[:3])
        inside = np.all((nearest >= 0) & (nearest < image_shape), axis=1)
        if not np.all(inside):
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "A compartment cell centroid lies outside the segmentation image.",
            )
        labels = np.asarray(specification.labels.asset.values)[tuple(nearest.T)]
        ontology_values = {
            value.label_id: value.value for value in specification.labels.ontology.labels
        }
        label_to_compartment = {
            ontology_values[label_id]: compartment.compartment_id
            for compartment in specification.compartments.compartments
            for label_id in compartment.label_ids
        }
        cell_compartment_ids = tuple(
            label_to_compartment.get(int(value), "") for value in labels
        )
        if any(not value for value in cell_compartment_ids):
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "A generated cell has no declared compartment label.",
            )
        cell_ids = np.asarray(mesh.entity_set(3).entity_ids)
        zones = []
        for compartment in specification.compartments.compartments:
            mask = np.asarray(
                [value == compartment.compartment_id for value in cell_compartment_ids]
            )
            if not np.any(mask):
                raise MeshingFailure(
                    MeshingFailureCategory.COMPLIANCE_FAILED,
                    f"Compartment {compartment.compartment_id!r} has no generated cells.",
                )
            scope = MeshingScope(
                mesh.mesh_id,
                mesh.numeric_version,
                MeshingEntityKind.MESH,
                3,
                mesh.entity_set(3).entity_set_id,
                cell_ids[mask],
            )
            zones.append(
                MeshZone(compartment.compartment_id, MeshZoneRole.MATERIAL, scope)
            )
        zones_ = validate_mesh_zones(tuple(zones))
        connectivity = mesh.connectivity
        if not isinstance(connectivity, TetrahedralConnectivity):
            raise RuntimeError(
                "Certified compartment mesh lost tetrahedral connectivity."
            )
        incidents = [[] for _ in np.asarray(connectivity.faces)]
        for cell_index, row in enumerate(np.asarray(connectivity.cell_faces)):
            for face_index in row:
                incidents[int(face_index)].append(cell_index)
        face_ids = np.asarray(mesh.entity_set(2).entity_ids)
        interface_definition = {
            value.ordered_pair: value for value in specification.compartments.interfaces
        }
        face_by_interface = {
            value.interface_id: [] for value in specification.compartments.interfaces
        }
        observed = set()
        for face_index, adjacent in enumerate(incidents):
            if len(adjacent) != 2:
                continue
            pair = tuple(
                sorted(
                    (
                        cell_compartment_ids[adjacent[0]],
                        cell_compartment_ids[adjacent[1]],
                    )
                )
            )
            if pair[0] == pair[1]:
                continue
            observed.add(pair)
            definition = interface_definition.get(pair)
            if definition is None:
                raise MeshingFailure(
                    MeshingFailureCategory.COMPLIANCE_FAILED,
                    f"Generated mesh contains forbidden compartment adjacency {pair}.",
                )
            face_by_interface[definition.interface_id].append(face_ids[face_index])
        patches = []
        for definition in specification.compartments.interfaces:
            identifiers = np.asarray(
                face_by_interface[definition.interface_id], dtype=np.int64
            )
            if definition.required and identifiers.size == 0:
                raise MeshingFailure(
                    MeshingFailureCategory.COMPLIANCE_FAILED,
                    f"Required interface {definition.interface_id!r} is absent.",
                )
            if identifiers.size:
                scope = MeshingScope(
                    mesh.mesh_id,
                    mesh.numeric_version,
                    MeshingEntityKind.MESH,
                    2,
                    mesh.entity_set(2).entity_set_id,
                    identifiers,
                )
                patches.append(MeshPatch(definition.interface_id, scope, connected=False))
        expected = set(specification.compartments.adjacency.expected_pairs)
        if not expected.issubset(observed):
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                f"Generated mesh misses required adjacency: required={sorted(expected)}, observed={sorted(observed)}.",
            )
        certified_semantics = certify_cell_mesh(
            mesh,
            specification.labels.asset.spatial_affine.coordinate_contract,
            audit_policy=specification.audit_policy,
            patches=tuple(patches),
            zones=zones_,
        )
        provider = FTetWildProvider(specification.options).info
        provenance = SemanticProvenance(
            {
                "kind": "ftetwild-compartment-mesh",
                "specification": specification.specification_id,
                "labels": specification.labels.label_volume_id,
                "compartments": specification.compartments.complex_id,
            }
        )
        compliance = MeshingComplianceReport(
            specification.specification_id,
            requested=(
                ("target_edge_length", specification.target_edge_length),
                ("compartment_count", float(len(zones_))),
                ("required_interface_count", float(len(expected))),
            ),
            achieved=(
                ("cell_count", float(mesh.entity_set(3).count)),
                ("compartment_count", float(len(zones_))),
                ("interface_count", float(len(patches))),
            ),
        )
        trace = MeshingTrace(
            (
                MeshingStageReport(
                    MeshingStageKind.VOLUME_FILL,
                    MeshingStageStatus.PASSED,
                    input_ids=(
                        specification.outer_surface.mesh.mesh_id,
                        specification.interfaces.extraction_id,
                    ),
                    output_ids=(mesh.mesh_id,),
                    created_count=mesh.entity_set(3).count,
                ),
                *certified_semantics.trace.stages,
            )
        )
        certified = CellMeshingResult(
            certified_semantics.mesh,
            certified_semantics.geometry,
            certified_semantics.coordinate_contract,
            certified_semantics.audit,
            certified_semantics.quality,
            compliance,
            trace,
            provider,
            MeshingRuntimeInfo(
                provider.provider_id,
                f"wildmeshing {version}",
                MeshingExecutionMode.IN_PROCESS,
                deterministic=False,
                enforced_limits=("input_entities", "output_entities"),
                unenforced_limits=("native_workspace", "wall_time"),
            ),
            MeshingDerivativeMode.NONDIFFERENTIABLE,
            provenance,
            patches=tuple(patches),
            zones=zones_,
        )
        result_id = canonical_fingerprint(
            {
                "kind": "compartment-meshing-result",
                "specification": specification.specification_id,
                "mesh": mesh.mesh_id,
                "zones": [value.zone_id for value in zones_],
                "patches": [value.patch_id for value in patches],
            }
        )
        return CompartmentMeshingResult(
            certified,
            zones_,
            tuple(patches),
            cell_compartment_ids,
            tuple(sorted(observed)),
            specification.specification_id,
            result_id,
        )


__all__ = [
    "CompartmentMeshingResult",
    "CompartmentMeshingSpec",
    "FTetWildCompartmentProvider",
]
