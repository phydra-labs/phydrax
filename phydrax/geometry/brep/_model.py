#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._strict import StrictModule
from .._atlas import AbstractBoundaryMap, BoundaryAtlas, TrimDomain
from ._patches import AbstractSurfacePatch


@dataclass(frozen=True, slots=True, order=True)
class BRepEntityId:
    """Stable source-revision-scoped identity for one B-Rep entity."""

    source_revision: str
    kind: str
    index: int


@dataclass(frozen=True, slots=True)
class BRepImportReport:
    """Host-side provenance and approximation limits for one imported B-Rep."""

    source_id: str
    source_revision: str
    source_format: str
    num_faces: int
    num_edges: int
    num_vertices: int
    num_triangles: int
    linear_deflection: float
    angular_deflection: float
    trim_samples_per_edge: int
    converted_surface_count: int


class BRepTopology(StrictModule):
    """Immutable incidence relations with stable local entity ordering."""

    face_edges: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    edge_faces: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    face_wires: tuple[tuple[tuple[int, ...], ...], ...] = eqx.field(static=True)
    num_vertices: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        face_edges: tuple[tuple[int, ...], ...],
        edge_faces: tuple[tuple[int, ...], ...],
        face_wires: tuple[tuple[tuple[int, ...], ...], ...],
        num_vertices: int,
    ):
        self.face_edges = face_edges
        self.edge_faces = edge_faces
        self.face_wires = face_wires
        self.num_vertices = int(num_vertices)

    @property
    def num_faces(self) -> int:
        return len(self.face_edges)

    @property
    def num_edges(self) -> int:
        return len(self.edge_faces)


class BRepBoundaryMap(AbstractBoundaryMap):
    """Dispatch heterogeneous JAX surface patches over normalized face charts."""

    patches: tuple[AbstractSurfacePatch, ...]
    parameter_bounds: Array

    def __init__(
        self,
        patches: tuple[AbstractSurfacePatch, ...],
        parameter_bounds: Array,
    ):
        bounds = jnp.asarray(parameter_bounds, dtype=float)
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
    """JAX-compatible B-Rep realization plus a watertight query tessellation."""

    patches: tuple[AbstractSurfacePatch, ...]
    parameter_bounds: Array
    orientation: Array
    trim_domains: tuple[TrimDomain | None, ...]
    topology: BRepTopology
    mesh_vertices: Array
    mesh_faces: Array
    triangle_face_ids: Array
    triangle_parameters: Array
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    physical_tags: tuple[str, ...] = eqx.field(static=True)
    report: BRepImportReport = eqx.field(static=True)

    def __init__(
        self,
        *,
        patches: tuple[AbstractSurfacePatch, ...],
        parameter_bounds: Array,
        orientation: Array,
        trim_domains: tuple[TrimDomain | None, ...],
        topology: BRepTopology,
        mesh_vertices: Array,
        mesh_faces: Array,
        triangle_face_ids: Array,
        triangle_parameters: Array,
        source_id: str,
        source_revision: str,
        physical_tags: tuple[str, ...],
        report: BRepImportReport,
    ):
        face_count = len(patches)
        bounds = jnp.asarray(parameter_bounds, dtype=float)
        orientation_ = jnp.asarray(orientation, dtype=float).reshape((-1,))
        vertices = jnp.asarray(mesh_vertices, dtype=float)
        faces = jnp.asarray(mesh_faces, dtype=jnp.int32)
        face_ids = jnp.asarray(triangle_face_ids, dtype=jnp.int32).reshape((-1,))
        parameters = jnp.asarray(triangle_parameters, dtype=float)
        if bounds.shape != (face_count, 2, 2):
            raise ValueError("parameter_bounds must contain one 2D box per face.")
        if orientation_.shape != (face_count,):
            raise ValueError("orientation must contain one sign per face.")
        if len(trim_domains) != face_count or len(physical_tags) != face_count:
            raise ValueError("Trim domains and physical tags must align with faces.")
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("mesh_vertices must have shape (num_vertices, 3).")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("mesh_faces must have shape (num_triangles, 3).")
        if face_ids.shape != (faces.shape[0],):
            raise ValueError("triangle_face_ids must align with mesh faces.")
        if parameters.shape != (faces.shape[0], 3, 2):
            raise ValueError("triangle_parameters must have shape (num_triangles, 3, 2).")
        self.patches = patches
        self.parameter_bounds = bounds
        self.orientation = orientation_
        self.trim_domains = trim_domains
        self.topology = topology
        self.mesh_vertices = vertices
        self.mesh_faces = faces
        self.triangle_face_ids = face_ids
        self.triangle_parameters = parameters
        self.source_id = source_id
        self.source_revision = source_revision
        self.physical_tags = physical_tags
        self.report = report

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
