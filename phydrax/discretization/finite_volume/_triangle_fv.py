#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...linalg import ArraySpace, DiagonalPairing
from .._cell_complex import PolygonalConnectivity
from .._cell_mesh import CellMesh
from .._core import (
    DiscretizationCapability,
    DiscretizationKey,
    DiscretizationRole,
    PreparationReport,
)
from .._lifecycle import (
    AbstractDiscretizationPlan,
    AbstractPreparedDiscretization,
    validate_prepared_metadata,
)
from .._measure import DiscreteMeasure
from .._spaces import DiscreteFieldSpace, EntityDofLayout
from .._support import DiscreteSupport
from .._triangular import triangle_connectivity, TriangleConnectivity
from ._geometry_protocol import FiniteVolumeFaceBlock
from ._structured import _component_names


def _normalized_triangles(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    points = vertices[triangles]
    signed_twice_area = (points[:, 1, 0] - points[:, 0, 0]) * (
        points[:, 2, 1] - points[:, 0, 1]
    ) - (points[:, 1, 1] - points[:, 0, 1]) * (points[:, 2, 0] - points[:, 0, 0])
    if np.any(~np.isfinite(signed_twice_area)) or np.any(signed_twice_area == 0.0):
        raise ValueError("Triangle mesh contains nonfinite or degenerate cells.")
    normalized = triangles.copy()
    reverse = signed_twice_area < 0.0
    normalized[reverse, 1], normalized[reverse, 2] = (
        normalized[reverse, 2].copy(),
        normalized[reverse, 1].copy(),
    )
    canonical_cells = np.sort(normalized, axis=1)
    if np.unique(canonical_cells, axis=0).shape[0] != normalized.shape[0]:
        raise ValueError("Triangle mesh contains duplicate cells.")
    return normalized


def _owner_neighbour(connectivity: PolygonalConnectivity, cell_count: int):
    cell_edges = np.asarray(connectivity.cell_edges, dtype=np.int32)
    cell_signs = np.asarray(connectivity.cell_edge_signs, dtype=float)
    edge_count = np.asarray(connectivity.edges).shape[0]
    owner = np.full((edge_count,), -1, dtype=np.int32)
    neighbour = np.full((edge_count,), -1, dtype=np.int32)
    owner_sign = np.zeros((edge_count,), dtype=float)
    for cell in range(cell_count):
        for local in range(3):
            edge = int(cell_edges[cell, local])
            if owner[edge] < 0:
                owner[edge] = cell
                owner_sign[edge] = cell_signs[cell, local]
            else:
                if neighbour[edge] >= 0:
                    raise ValueError("Triangle mesh contains a non-manifold edge.")
                if cell_signs[cell, local] == owner_sign[edge]:
                    raise ValueError(
                        "Interior triangle incidences must have opposite orientation."
                    )
                neighbour[edge] = cell
    return owner, neighbour, owner_sign


def evaluate_triangle_fv_geometry(
    vertices: ArrayLike,
    triangles: ArrayLike,
    connectivity: PolygonalConnectivity,
    owner: ArrayLike,
    owner_sign: ArrayLike,
    /,
):
    points = jnp.asarray(vertices)
    cells = jnp.asarray(triangles, dtype=jnp.int32)
    cell_points = points[cells]
    cross = (cell_points[:, 1, 0] - cell_points[:, 0, 0]) * (
        cell_points[:, 2, 1] - cell_points[:, 0, 1]
    ) - (cell_points[:, 1, 1] - cell_points[:, 0, 1]) * (
        cell_points[:, 2, 0] - cell_points[:, 0, 0]
    )
    area = 0.5 * cross
    area = eqx.error_if(
        area,
        jnp.any(~jnp.isfinite(area) | (area <= 0.0)),
        "Triangle FV geometry requires positive finite cell area.",
    )
    cell_centers = jnp.mean(cell_points, axis=1)
    edges = jnp.asarray(connectivity.edges, dtype=jnp.int32)
    edge_points = points[edges]
    face_centers = 0.5 * (edge_points[:, 0] + edge_points[:, 1])
    tangent = edge_points[:, 1] - edge_points[:, 0]
    canonical_normal = jnp.stack((tangent[:, 1], -tangent[:, 0]), axis=-1)
    area_vectors = jnp.asarray(owner_sign)[:, None] * canonical_normal
    face_measures = jnp.linalg.norm(area_vectors, axis=-1)
    face_measures = eqx.error_if(
        face_measures,
        jnp.any(~jnp.isfinite(face_measures) | (face_measures <= 0.0)),
        "Triangle FV geometry requires positive finite edge measures.",
    )
    owner_centers = cell_centers[jnp.asarray(owner, dtype=jnp.int32)]
    outward = jnp.sum((face_centers - owner_centers) * area_vectors, axis=-1)
    area_vectors = eqx.error_if(
        area_vectors,
        jnp.any(outward <= 0.0),
        "Triangle face area vector must point outward from owner.",
    )
    closure = jnp.zeros_like(cell_centers)
    cell_edges = jnp.asarray(connectivity.cell_edges[:, :3], dtype=jnp.int32)
    cell_signs = jnp.asarray(connectivity.cell_edge_signs[:, :3])
    closure = closure.at[jnp.repeat(jnp.arange(cells.shape[0]), 3)].add(
        (cell_signs[..., None] * canonical_normal[cell_edges]).reshape((-1, 2))
    )
    return area, cell_centers, face_centers, area_vectors, face_measures, closure


class TriangleFiniteVolumeQualityReport(StrictModule):
    minimum_area: Array
    maximum_area: Array
    minimum_edge_measure: Array
    maximum_aspect_ratio: Array
    maximum_nonorthogonality_degrees: Array
    maximum_closure_residual: Array
    worst_cell: Array


class TriangleFiniteVolumePlan(AbstractDiscretizationPlan):
    mesh: CellMesh
    patch_names: tuple[str, ...] = eqx.field(static=True)
    patch_edges: tuple[Array, ...]
    field_name: str = eqx.field(static=True)
    component_names: tuple[str, ...] = eqx.field(static=True)
    key: DiscretizationKey
    capabilities: tuple[DiscretizationCapability, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        vertices: ArrayLike,
        triangles: ArrayLike,
        /,
        *,
        boundary_patches: Mapping[str, ArrayLike] | None = None,
        field_name: str = "state",
        component_names: Sequence[str] = ("value",),
    ):
        points = np.asarray(vertices, dtype=float)
        cells = np.asarray(triangles, dtype=np.int32)
        if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
            raise ValueError("Triangle FV vertices must have shape (n >= 3, 2).")
        if cells.ndim != 2 or cells.shape[1] != 3 or cells.shape[0] == 0:
            raise ValueError("Triangle FV cells must have shape (m > 0, 3).")
        if np.any(cells < 0) or np.any(cells >= points.shape[0]):
            raise ValueError("Triangle FV connectivity indexes invalid vertices.")
        cells = _normalized_triangles(points, cells)
        mesh = CellMesh.from_triangles(points, cells)
        connectivity = mesh.connectivity
        edges = np.asarray(connectivity.edges, dtype=np.int32)
        boundary_mask = np.asarray(connectivity.boundary_edges, dtype=bool)
        patches = {} if boundary_patches is None else dict(boundary_patches)
        if not patches:
            patches = {"boundary": edges[boundary_mask]}
        names = tuple(sorted(str(name) for name in patches))
        patch_values = tuple(np.asarray(patches[name], dtype=np.int32) for name in names)
        edge_lookup = {tuple(edge): index for index, edge in enumerate(edges)}
        assigned = np.zeros((edges.shape[0],), dtype=np.int32)
        normalized_patch_edges = []
        for name, values in zip(names, patch_values, strict=True):
            if not name or values.ndim != 2 or values.shape[1] != 2:
                raise ValueError("Boundary patch edges must have shape (k, 2).")
            indices = []
            for edge in values:
                key = tuple(sorted((int(edge[0]), int(edge[1]))))
                if key not in edge_lookup:
                    raise ValueError(f"Boundary patch edge {key!r} is not in the mesh.")
                edge_index = edge_lookup[key]
                if not boundary_mask[edge_index]:
                    raise ValueError("Physical patch cannot contain an interior edge.")
                assigned[edge_index] += 1
                indices.append(edge_index)
            normalized_patch_edges.append(np.asarray(indices, dtype=np.int32))
        if np.any(assigned[boundary_mask] != 1):
            raise ValueError("Every triangle boundary edge requires exactly one patch.")
        field = str(field_name)
        if not field:
            raise ValueError("field_name must be non-empty.")
        components = _component_names(component_names)
        key = DiscretizationKey("triangle_finite_volume", DiscretizationRole.PHYSICAL)
        capabilities = (
            DiscretizationCapability.RECONSTRUCTION,
            DiscretizationCapability.TRACE,
            DiscretizationCapability.CONSERVATIVE_FLUX,
            DiscretizationCapability.BOUNDARY_INTEGRAL,
            DiscretizationCapability.MATRIX_FREE,
            DiscretizationCapability.DIFFERENTIABLE_GEOMETRY,
        )
        self.mesh = mesh
        self.patch_names = names
        self.patch_edges = tuple(jnp.asarray(value) for value in normalized_patch_edges)
        self.field_name = field
        self.component_names = components
        self.key = key
        self.capabilities = capabilities
        self.plan_id = canonical_fingerprint(
            {
                "kind": "triangle-finite-volume-plan",
                "mesh": mesh.mesh_id,
                "patches": {
                    name: array_tree_fingerprint(value)
                    for name, value in zip(names, normalized_patch_edges, strict=True)
                },
                "field": field,
                "components": list(components),
            }
        )

    @property
    def vertices(self) -> Array:
        return self.mesh.coordinates

    @property
    def triangles(self) -> Array:
        return self.mesh.blocks[0].vertices

    def prepare(self, /, *, numeric_version: str = "0"):
        return TriangleFiniteVolumeDiscretization(self, numeric_version=numeric_version)


class TriangleFiniteVolumeDiscretization(AbstractPreparedDiscretization):
    mesh: CellMesh
    triangle_connectivity: TriangleConnectivity
    face_block: FiniteVolumeFaceBlock
    face_blocks: tuple[FiniteVolumeFaceBlock, ...]
    cell_volumes: Array
    cell_centers: Array
    face_centers: Array
    area_vectors: Array
    face_measures: Array
    face_quadrature_points: Array
    face_quadrature_weights: Array
    owner_cells: Array
    owner_signs: Array
    neighbour_cells: Array
    boundary_patch_ids: Array
    boundary_patch_names: tuple[str, ...] = eqx.field(static=True)
    cell_space: DiscreteFieldSpace
    face_space: DiscreteFieldSpace
    component_names: tuple[str, ...] = eqx.field(static=True)
    key: DiscretizationKey
    support: DiscreteSupport
    field_spaces: tuple[DiscreteFieldSpace, ...]
    measures: tuple[DiscreteMeasure, ...]
    capabilities: tuple[DiscretizationCapability, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)
    preparation: PreparationReport
    quality: TriangleFiniteVolumeQualityReport

    def __init__(self, plan: TriangleFiniteVolumePlan, /, *, numeric_version: str = "0"):
        if not isinstance(plan, TriangleFiniteVolumePlan):
            raise TypeError("plan must be TriangleFiniteVolumePlan.")
        mesh = plan.mesh
        points = np.asarray(mesh.coordinates)
        cells = np.asarray(mesh.blocks[0].vertices, dtype=np.int32)
        connectivity = triangle_connectivity(cells, points.shape[0])
        topology = mesh.topology
        owner, neighbour, owner_sign = _owner_neighbour(connectivity, cells.shape[0])
        area, centers, face_centers, area_vectors, measures, closure = (
            evaluate_triangle_fv_geometry(
                plan.vertices, plan.triangles, connectivity, owner, owner_sign
            )
        )
        edge_points = plan.vertices[jnp.asarray(connectivity.edges)]
        half_gauss_offset = (edge_points[:, 1] - edge_points[:, 0]) / (
            2.0 * jnp.sqrt(3.0)
        )
        face_quadrature_points = jnp.stack(
            (
                face_centers - half_gauss_offset,
                face_centers + half_gauss_offset,
            ),
            axis=1,
        )
        face_quadrature_weights = jnp.broadcast_to(
            0.5 * measures[:, None],
            (measures.size, 2),
        )
        boundary_patch_ids = np.full((owner.shape[0],), -1, dtype=np.int32)
        for patch_id, edge_indices in enumerate(plan.patch_edges):
            boundary_patch_ids[np.asarray(edge_indices, dtype=np.int32)] = patch_id
        support = mesh.support
        components = len(plan.component_names)
        cell_entities = topology.entity_sets[2]
        face_entities = topology.entity_sets[1]
        cell_shape = (cells.shape[0], components)
        cell_weights = jnp.broadcast_to(area[:, None], cell_shape)
        cell_space = DiscreteFieldSpace(
            plan.field_name,
            support.support_id,
            EntityDofLayout(
                cell_entities.entity_set_id,
                cells.shape[0],
                cells.shape[0],
                component_shape=(components,),
            ),
            ArraySpace(cell_shape, pairing=DiagonalPairing(cell_weights)),
            representation="cell_average",
            conformity="discontinuous",
            reconstruction_id=canonical_fingerprint(
                {"kind": "triangle-cell-average", "plan": plan.plan_id}
            ),
        )
        face_shape = (owner.shape[0], components)
        face_space = DiscreteFieldSpace(
            f"{plan.field_name}_face_flux",
            support.support_id,
            EntityDofLayout(
                face_entities.entity_set_id,
                owner.shape[0],
                owner.shape[0],
                component_shape=(components,),
            ),
            ArraySpace(
                face_shape,
                pairing=DiagonalPairing(jnp.broadcast_to(measures[:, None], face_shape)),
            ),
            representation="flux_moment",
            conformity="Hdiv",
            trace_space_id=cell_space.field_space_id,
        )
        face_block = FiniteVolumeFaceBlock(
            face_ids=jnp.arange(owner.shape[0], dtype=jnp.int32),
            owner_cells=jnp.asarray(owner),
            neighbour_cells=jnp.asarray(neighbour),
            boundary_patch_ids=jnp.asarray(boundary_patch_ids),
            face_centers=face_centers,
            area_vectors=area_vectors,
            face_measures=measures,
            active_mask=jnp.ones((owner.shape[0],), dtype=bool),
            block_id=canonical_fingerprint(
                {"kind": "triangle-face-block", "plan": plan.plan_id}
            ),
        )
        quality = _triangle_quality(
            plan.vertices,
            plan.triangles,
            centers,
            area_vectors,
            measures,
            closure,
            owner,
            neighbour,
        )
        preparation = PreparationReport(
            capabilities=plan.capabilities,
            diagnostics=(
                "triangle cell areas are positive",
                "face area vectors point outward from owners",
                "boundary patches are complete",
            ),
            resource_counts={
                "vertices": points.shape[0],
                "faces": owner.shape[0],
                "cells": cells.shape[0],
                "boundary_faces": int(np.sum(neighbour < 0)),
            },
        )
        measures_metadata = (
            DiscreteMeasure(
                "triangle_cell_area",
                support.support_id,
                cell_entities.entity_set_id,
                area,
            ),
            DiscreteMeasure(
                "triangle_face_measure",
                support.support_id,
                face_entities.entity_set_id,
                measures,
            ),
        )
        spaces, measures_, capabilities = validate_prepared_metadata(
            key=plan.key,
            support=support,
            field_spaces=(cell_space, face_space),
            measures=measures_metadata,
            capabilities=plan.capabilities,
            preparation=preparation,
        )
        version = str(numeric_version)
        if not version:
            raise ValueError("numeric_version must be non-empty.")
        self.mesh = mesh
        self.triangle_connectivity = connectivity
        self.face_block = face_block
        self.face_blocks = (face_block,)
        self.cell_volumes = area
        self.cell_centers = centers
        self.face_centers = face_centers
        self.area_vectors = area_vectors
        self.face_measures = measures
        self.face_quadrature_points = face_quadrature_points
        self.face_quadrature_weights = face_quadrature_weights
        self.owner_cells = jnp.asarray(owner)
        self.owner_signs = jnp.asarray(owner_sign)
        self.neighbour_cells = jnp.asarray(neighbour)
        self.boundary_patch_ids = jnp.asarray(boundary_patch_ids)
        self.boundary_patch_names = plan.patch_names
        self.cell_space = cell_space
        self.face_space = face_space
        self.component_names = plan.component_names
        self.key = plan.key
        self.support = support
        self.field_spaces = spaces
        self.measures = measures_
        self.capabilities = capabilities
        self.plan_id = plan.plan_id
        self.numeric_version = version
        self.preparation = preparation
        self.quality = quality
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-triangle-finite-volume",
                "plan": plan.plan_id,
                "topology": mesh.topology_id,
                "geometry": mesh.geometry_id,
                "numeric_version": version,
            }
        )

    @property
    def vertices(self) -> Array:
        return self.mesh.coordinates

    @property
    def triangles(self) -> Array:
        return self.mesh.blocks[0].vertices

    @property
    def topology(self):
        return self.mesh.topology

    @property
    def connectivity(self) -> TriangleConnectivity:
        return self.triangle_connectivity

    @property
    def cell_count(self) -> int:
        return int(self.triangles.shape[0])

    @property
    def component_count(self) -> int:
        return len(self.component_names)

    @property
    def state_shape(self) -> tuple[int, ...]:
        return (self.cell_count, self.component_count)


def _triangle_quality(
    vertices,
    triangles,
    centers,
    area_vectors,
    face_measures,
    closure,
    owner,
    neighbour,
):
    points = jnp.asarray(vertices)[jnp.asarray(triangles, dtype=jnp.int32)]
    lengths = jnp.linalg.norm(jnp.roll(points, -1, axis=1) - points, axis=-1)
    area = 0.5 * jnp.abs(
        (points[:, 1, 0] - points[:, 0, 0]) * (points[:, 2, 1] - points[:, 0, 1])
        - (points[:, 1, 1] - points[:, 0, 1]) * (points[:, 2, 0] - points[:, 0, 0])
    )
    altitude = 2.0 * area[:, None] / lengths
    aspect = jnp.max(lengths, axis=1) / jnp.min(altitude, axis=1)
    owner_ = jnp.asarray(owner, dtype=jnp.int32)
    neighbour_ = jnp.asarray(neighbour, dtype=jnp.int32)
    interior = neighbour_ >= 0
    connector = centers[jnp.maximum(neighbour_, 0)] - centers[owner_]
    denominator = jnp.linalg.norm(connector, axis=-1) * face_measures
    cosine = jnp.abs(jnp.sum(connector * area_vectors, axis=-1)) / jnp.where(
        denominator > 0.0, denominator, 1.0
    )
    angle = jnp.degrees(jnp.arccos(jnp.clip(cosine, 0.0, 1.0)))
    maximum_nonorthogonality = jnp.max(jnp.where(interior, angle, 0.0))
    return TriangleFiniteVolumeQualityReport(
        minimum_area=jnp.min(area),
        maximum_area=jnp.max(area),
        minimum_edge_measure=jnp.min(lengths),
        maximum_aspect_ratio=jnp.max(aspect),
        maximum_nonorthogonality_degrees=maximum_nonorthogonality,
        maximum_closure_residual=jnp.max(jnp.linalg.norm(closure, axis=-1)),
        worst_cell=jnp.argmax(aspect),
    )


__all__ = [
    "TriangleFiniteVolumeDiscretization",
    "TriangleFiniteVolumePlan",
    "TriangleFiniteVolumeQualityReport",
    "evaluate_triangle_fv_geometry",
]
