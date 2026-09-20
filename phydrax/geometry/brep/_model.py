#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass, field

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from .._atlas import AbstractBoundaryMap, BoundaryAtlas, TrimDomain
from ._patches import AbstractSurfacePatch


@dataclass(frozen=True, slots=True, order=True)
class BRepEntityId:
    """Stable source-revision-scoped identity for one B-Rep entity."""

    source_revision: str
    kind: str
    index: int

    def __post_init__(self):
        if not isinstance(self.source_revision, str):
            raise TypeError("source_revision must be a string.")
        if not self.source_revision:
            raise ValueError("source_revision must be non-empty.")
        if not isinstance(self.kind, str):
            raise TypeError("kind must be a string.")
        if self.kind not in ("solid", "face", "edge", "vertex"):
            raise ValueError("kind must be solid, face, edge, or vertex.")
        if isinstance(self.index, bool) or not isinstance(self.index, int):
            raise TypeError("index must be an integer.")
        if self.index < 0:
            raise ValueError("index must be non-negative.")


@dataclass(frozen=True, slots=True)
class BRepImportReport:
    """Host-side physical provenance and approximation limits for one import."""

    source_id: str
    source_digest: str
    source_format: str
    coordinate_contract: SpatialCoordinateContract
    import_policy_id: str
    num_solids: int
    num_faces: int
    num_edges: int
    num_vertices: int
    num_triangles: int
    linear_deflection: float
    angular_deflection: float
    trim_samples_per_edge: int
    converted_surface_count: int
    source_revision: str = field(init=False)

    def __post_init__(self):
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("source_id must be a non-empty string.")
        if (
            not isinstance(self.source_digest, str)
            or len(self.source_digest) != 64
            or self.source_digest != self.source_digest.lower()
            or any(
                character not in "0123456789abcdef" for character in self.source_digest
            )
        ):
            raise ValueError("source_digest must be a lowercase SHA-256 digest.")
        if not isinstance(self.source_format, str) or not self.source_format:
            raise ValueError("source_format must be a non-empty string.")
        if not isinstance(self.coordinate_contract, SpatialCoordinateContract):
            raise TypeError("coordinate_contract must be a SpatialCoordinateContract.")
        if (
            not isinstance(self.import_policy_id, str)
            or len(self.import_policy_id) != 64
            or self.import_policy_id != self.import_policy_id.lower()
            or any(
                character not in "0123456789abcdef" for character in self.import_policy_id
            )
        ):
            raise ValueError("import_policy_id must be a lowercase SHA-256 digest.")
        counts = (
            self.num_solids,
            self.num_faces,
            self.num_edges,
            self.num_vertices,
            self.num_triangles,
            self.converted_surface_count,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in counts):
            raise TypeError("B-Rep import counts must be integers.")
        if any(value < 0 for value in counts):
            raise ValueError("B-Rep import counts must be non-negative.")
        if self.num_faces == 0:
            raise ValueError("A B-Rep import report requires at least one face.")
        if self.converted_surface_count > self.num_faces:
            raise ValueError("converted_surface_count cannot exceed num_faces.")
        if (
            not np.isfinite(self.linear_deflection)
            or not np.isfinite(self.angular_deflection)
            or self.linear_deflection <= 0.0
            or self.angular_deflection <= 0.0
        ):
            raise ValueError("Meshing deflections must be finite and positive.")
        if isinstance(self.trim_samples_per_edge, bool) or not isinstance(
            self.trim_samples_per_edge, int
        ):
            raise TypeError("trim_samples_per_edge must be an integer.")
        if self.trim_samples_per_edge < 3:
            raise ValueError("trim_samples_per_edge must be at least three.")
        object.__setattr__(
            self,
            "source_revision",
            canonical_fingerprint(
                {
                    "kind": "brep-source-revision",
                    "source_digest": self.source_digest,
                    "spatial_id": self.coordinate_contract.spatial_id,
                    "import_policy_id": self.import_policy_id,
                }
            ),
        )


class BRepTopology(StrictModule):
    """Immutable incidence relations with stable local entity ordering."""

    face_edges: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    edge_faces: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    face_wires: tuple[tuple[tuple[int, ...], ...], ...] = eqx.field(static=True)
    solid_faces: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    solid_face_orientations: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    face_solids: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    num_vertices: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        face_edges: tuple[tuple[int, ...], ...],
        edge_faces: tuple[tuple[int, ...], ...],
        face_wires: tuple[tuple[tuple[int, ...], ...], ...],
        solid_faces: tuple[tuple[int, ...], ...],
        solid_face_orientations: tuple[tuple[int, ...], ...],
        num_vertices: int,
    ):
        face_count = len(face_edges)
        if len(face_wires) != face_count:
            raise ValueError("face_wires must contain one entry per face.")
        if len(solid_faces) != len(solid_face_orientations):
            raise ValueError("solid_face_orientations must contain one entry per solid.")
        if int(num_vertices) < 0:
            raise ValueError("num_vertices must be non-negative.")
        face_solids: list[list[int]] = [[] for _ in range(face_count)]
        for solid_index, (indices, orientations) in enumerate(
            zip(solid_faces, solid_face_orientations, strict=True)
        ):
            if len(indices) != len(orientations):
                raise ValueError(
                    "Each solid's face orientations must align with its faces."
                )
            if len(set(indices)) != len(indices):
                raise ValueError("A solid cannot contain the same face more than once.")
            if any(index < 0 or index >= face_count for index in indices):
                raise ValueError("A solid references an absent global face.")
            if any(orientation not in (-1, 1) for orientation in orientations):
                raise ValueError("Solid face orientations must be -1 or 1.")
            for face_index in indices:
                face_solids[face_index].append(solid_index)
        self.face_edges = face_edges
        self.edge_faces = edge_faces
        self.face_wires = face_wires
        self.solid_faces = solid_faces
        self.solid_face_orientations = solid_face_orientations
        self.face_solids = tuple(tuple(indices) for indices in face_solids)
        self.num_vertices = int(num_vertices)

    @property
    def num_faces(self) -> int:
        return len(self.face_edges)

    @property
    def num_edges(self) -> int:
        return len(self.edge_faces)

    @property
    def num_solids(self) -> int:
        return len(self.solid_faces)


class BRepBoundaryMap(AbstractBoundaryMap):
    """Dispatch heterogeneous JAX surface patches over normalized face charts."""

    patches: tuple[AbstractSurfacePatch, ...]
    parameter_bounds: Array

    def __init__(
        self,
        patches: tuple[AbstractSurfacePatch, ...],
        parameter_bounds: Array,
    ):
        bounds = jnp.asarray(parameter_bounds, dtype=jnp.float64)
        if not patches:
            raise ValueError("A BRepBoundaryMap requires at least one patch.")
        if bounds.shape != (len(patches), 2, 2):
            raise ValueError("parameter_bounds must have shape (num_faces, 2, 2).")
        bounds_host = np.asarray(bounds)
        if not np.all(np.isfinite(bounds_host)) or np.any(
            bounds_host[:, 1, :] <= bounds_host[:, 0, :]
        ):
            raise ValueError(
                "Every surface parameter interval must be finite and nonempty."
            )
        self.patches = patches
        self.parameter_bounds = bounds

    @property
    def num_charts(self) -> int:
        return len(self.patches)

    @property
    def reference_dimension(self) -> int:
        return 2

    @property
    def ambient_dimension(self) -> int:
        return 3

    def _map_one(self, chart_index: Array, reference: Array) -> Array:
        bounds = self.parameter_bounds[chart_index]
        parameters = bounds[0] + reference * (bounds[1] - bounds[0])
        branches = tuple(
            lambda coordinate, patch=patch: patch.evaluate(coordinate)
            for patch in self.patches
        )
        return jax.lax.switch(chart_index, branches, parameters)

    def map(self, chart_indices: Array, reference: Array, /) -> Array:
        indices = jnp.asarray(chart_indices, dtype=jnp.int32)
        reference_ = jnp.asarray(reference, dtype=self.parameter_bounds.dtype)
        leading = indices.shape
        values = jax.vmap(self._map_one)(
            indices.reshape((-1,)), reference_.reshape((-1, 2))
        )
        return values.reshape((*leading, 3))

    def jacobian(self, chart_indices: Array, reference: Array, /) -> Array:
        indices = jnp.asarray(chart_indices, dtype=jnp.int32)
        reference_ = jnp.asarray(reference, dtype=self.parameter_bounds.dtype)
        leading = indices.shape
        differential = jax.vmap(
            lambda index, coordinate: jax.jacfwd(
                lambda value: self._map_one(index, value)
            )(coordinate)
        )(indices.reshape((-1,)), reference_.reshape((-1, 2)))
        jacobian = jnp.linalg.norm(
            jnp.cross(differential[..., :, 0], differential[..., :, 1]), axis=-1
        )
        return jacobian.reshape(leading)


class BRepModel(StrictModule):
    """JAX-compatible B-Rep realization plus its reported query tessellation."""

    patches: tuple[AbstractSurfacePatch, ...]
    parameter_bounds: Array
    orientation: Array
    trim_domains: tuple[TrimDomain | None, ...]
    topology: BRepTopology
    mesh_vertices: Array
    mesh_faces: Array
    triangle_face_ids: Array
    triangle_parameters: Array
    physical_tags: tuple[str, ...] = eqx.field(static=True)
    report: BRepImportReport = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        patches: tuple[AbstractSurfacePatch, ...],
        parameter_bounds: Array,
        orientation: Array,
        trim_domains: tuple[TrimDomain | None, ...],
        topology: BRepTopology,
        coordinate_contract: SpatialCoordinateContract,
        mesh_vertices: Array,
        mesh_faces: Array,
        triangle_face_ids: Array,
        triangle_parameters: Array,
        physical_tags: tuple[str, ...],
        report: BRepImportReport,
    ):
        if not isinstance(topology, BRepTopology):
            raise TypeError("topology must be a BRepTopology.")
        if not isinstance(report, BRepImportReport):
            raise TypeError("report must be a BRepImportReport.")
        if not isinstance(coordinate_contract, SpatialCoordinateContract):
            raise TypeError("coordinate_contract must be a SpatialCoordinateContract.")
        if coordinate_contract.spatial_id != report.coordinate_contract.spatial_id:
            raise ValueError(
                "The model and import report coordinate contracts must match."
            )
        face_count = len(patches)
        bounds = jnp.asarray(parameter_bounds, dtype=jnp.float64)
        orientation_ = jnp.asarray(orientation, dtype=jnp.float64).reshape((-1,))
        vertices = jnp.asarray(mesh_vertices, dtype=jnp.float64)
        faces = jnp.asarray(mesh_faces, dtype=jnp.int32)
        face_ids = jnp.asarray(triangle_face_ids, dtype=jnp.int32).reshape((-1,))
        parameters = jnp.asarray(triangle_parameters, dtype=jnp.float64)
        bounds_host = np.asarray(bounds)
        orientation_host = np.asarray(orientation_)
        vertices_host = np.asarray(vertices)
        faces_host = np.asarray(faces)
        face_ids_host = np.asarray(face_ids)
        parameters_host = np.asarray(parameters)
        if face_count == 0:
            raise ValueError("A BRepModel requires at least one face.")
        if topology.num_faces != face_count:
            raise ValueError("topology must contain one entry per face.")
        if bounds.shape != (face_count, 2, 2):
            raise ValueError("parameter_bounds must contain one 2D box per face.")
        if not np.all(np.isfinite(bounds_host)) or np.any(
            bounds_host[:, 1, :] <= bounds_host[:, 0, :]
        ):
            raise ValueError(
                "Every surface parameter interval must be finite and nonempty."
            )
        if orientation_.shape != (face_count,):
            raise ValueError("orientation must contain one sign per face.")
        if not np.all(np.isin(orientation_host, (-1.0, 1.0))):
            raise ValueError("orientation entries must be -1 or 1.")
        if len(trim_domains) != face_count or len(physical_tags) != face_count:
            raise ValueError("Trim domains and physical tags must align with faces.")
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("mesh_vertices must have shape (num_vertices, 3).")
        if not np.all(np.isfinite(vertices_host)):
            raise ValueError("mesh_vertices must be finite.")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("mesh_faces must have shape (num_triangles, 3).")
        if faces_host.size and (
            np.min(faces_host) < 0 or np.max(faces_host) >= vertices.shape[0]
        ):
            raise ValueError("mesh_faces reference an absent mesh vertex.")
        if face_ids.shape != (faces.shape[0],):
            raise ValueError("triangle_face_ids must align with mesh faces.")
        if face_ids_host.size and (
            np.min(face_ids_host) < 0 or np.max(face_ids_host) >= face_count
        ):
            raise ValueError("triangle_face_ids reference an absent B-Rep face.")
        if parameters.shape != (faces.shape[0], 3, 2):
            raise ValueError("triangle_parameters must have shape (num_triangles, 3, 2).")
        if not np.all(np.isfinite(parameters_host)):
            raise ValueError("triangle_parameters must be finite.")
        expected_report_counts = (
            topology.num_solids,
            face_count,
            topology.num_edges,
            topology.num_vertices,
            faces.shape[0],
        )
        report_counts = (
            report.num_solids,
            report.num_faces,
            report.num_edges,
            report.num_vertices,
            report.num_triangles,
        )
        if report_counts != expected_report_counts:
            raise ValueError(
                "The import report entity counts must match the B-Rep model."
            )
        self.patches = patches
        self.parameter_bounds = bounds
        self.orientation = orientation_
        self.trim_domains = trim_domains
        self.topology = topology
        self.mesh_vertices = vertices
        self.mesh_faces = faces
        self.triangle_face_ids = face_ids
        self.triangle_parameters = parameters
        self.physical_tags = physical_tags
        self.report = report
        self.model_id = canonical_fingerprint(
            {
                "kind": "brep-model",
                "source_revision": report.source_revision,
                "query_representation": {
                    "linear_deflection": report.linear_deflection,
                    "angular_deflection": report.angular_deflection,
                    "trim_samples_per_edge": report.trim_samples_per_edge,
                    "parameter_bounds": bounds,
                    "orientation": orientation_,
                    "trim_domains": tuple(
                        None
                        if domain is None
                        else {"outer": domain.outer, "holes": domain.holes}
                        for domain in trim_domains
                    ),
                    "mesh_vertices": vertices,
                    "mesh_faces": faces,
                    "triangle_face_ids": face_ids,
                    "triangle_parameters": parameters,
                },
                "topology_representation": {
                    "face_edges": topology.face_edges,
                    "edge_faces": topology.edge_faces,
                    "face_wires": topology.face_wires,
                    "solid_faces": topology.solid_faces,
                    "solid_face_orientations": topology.solid_face_orientations,
                    "num_vertices": topology.num_vertices,
                },
                "patch_types": tuple(type(patch).__name__ for patch in patches),
                "physical_tags": physical_tags,
            }
        )

    @property
    def source_id(self) -> str:
        return self.report.source_id

    @property
    def source_digest(self) -> str:
        return self.report.source_digest

    @property
    def source_revision(self) -> str:
        return self.report.source_revision

    @property
    def coordinate_contract(self) -> SpatialCoordinateContract:
        return self.report.coordinate_contract

    @property
    def import_policy_id(self) -> str:
        return self.report.import_policy_id

    @property
    def face_ids(self) -> tuple[BRepEntityId, ...]:
        return tuple(
            BRepEntityId(self.source_revision, "face", index)
            for index in range(len(self.patches))
        )

    @property
    def edge_ids(self) -> tuple[BRepEntityId, ...]:
        return tuple(
            BRepEntityId(self.source_revision, "edge", index)
            for index in range(self.topology.num_edges)
        )

    @property
    def vertex_ids(self) -> tuple[BRepEntityId, ...]:
        return tuple(
            BRepEntityId(self.source_revision, "vertex", index)
            for index in range(self.topology.num_vertices)
        )

    @property
    def solid_ids(self) -> tuple[BRepEntityId, ...]:
        return tuple(
            BRepEntityId(self.source_revision, "solid", index)
            for index in range(self.topology.num_solids)
        )

    @property
    def boundary_atlas(self) -> BoundaryAtlas:
        return BoundaryAtlas(
            BRepBoundaryMap(self.patches, self.parameter_bounds),
            source_entity_ids=jnp.arange(len(self.patches), dtype=jnp.int32),
            source_id=self.source_id,
            physical_tags=self.physical_tags,
            orientation=self.orientation,
            trim_domains=self.trim_domains,
        )


__all__ = [
    "BRepBoundaryMap",
    "BRepEntityId",
    "BRepImportReport",
    "BRepModel",
    "BRepTopology",
]
