#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState


SmoothingPatchKind = Literal["cell", "edge", "node", "axisymmetric"]
SmoothingEnergyEvidence = Literal[
    "none",
    "empirical-lower-like",
    "empirical-upper-like",
    "proved-under-explicit-assumptions",
]


class SmoothingPatchLayout(StrictModule, NonTrainableState):
    """Fixed-capacity polygonal smoothing patches as affine vertex combinations."""

    patch_kind: SmoothingPatchKind = eqx.field(static=True)
    owner_entities: Array
    dof_routes: Array
    dof_valid: Array
    vertex_sources: Array
    vertex_coefficients: Array
    vertex_valid: Array
    boundary_edges: Array
    boundary_valid: Array
    boundary_shape_values: Array
    rule_points: Array
    rule_weights: Array
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        patch_kind: SmoothingPatchKind,
        owner_entities: ArrayLike,
        dof_routes: ArrayLike,
        dof_valid: ArrayLike,
        vertex_sources: ArrayLike,
        vertex_coefficients: ArrayLike,
        vertex_valid: ArrayLike,
        boundary_edges: ArrayLike,
        boundary_valid: ArrayLike,
        boundary_shape_values: ArrayLike,
        rule_points: ArrayLike,
        rule_weights: ArrayLike,
        /,
    ):
        owners = np.asarray(owner_entities, dtype=np.int32)
        routes = np.asarray(dof_routes, dtype=np.int32)
        route_valid = np.asarray(dof_valid, dtype=bool)
        sources = np.asarray(vertex_sources, dtype=np.int32)
        coefficients = np.asarray(vertex_coefficients, dtype=float)
        vertices_valid = np.asarray(vertex_valid, dtype=bool)
        edges = np.asarray(boundary_edges, dtype=np.int32)
        edges_valid = np.asarray(boundary_valid, dtype=bool)
        shape_values = np.asarray(boundary_shape_values, dtype=float)
        parameters = np.asarray(rule_points, dtype=float)
        weights = np.asarray(rule_weights, dtype=float)
        if patch_kind not in ("cell", "edge", "node", "axisymmetric"):
            raise ValueError("Unknown smoothing patch kind.")
        if owners.ndim != 1 or routes.ndim != 2 or routes.shape[0] != owners.size:
            raise ValueError("Smoothing owner/DOF routes have incompatible shapes.")
        if route_valid.shape != routes.shape or np.any(routes[route_valid] < 0):
            raise ValueError("Smoothing DOF routes or validity are invalid.")
        if sources.ndim != 3 or coefficients.shape != sources.shape:
            raise ValueError(
                "Patch vertex source/coefficient arrays must be rank-3 peers."
            )
        if vertices_valid.shape != sources.shape[:2] or sources.shape[0] != owners.size:
            raise ValueError("Patch vertex validity has incompatible shape.")
        active_source = vertices_valid[..., None] & (coefficients != 0.0)
        if np.any(sources[active_source] < 0):
            raise ValueError("Active patch-vertex sources must be non-negative.")
        if edges.ndim != 3 or edges.shape[-1] != 2 or edges.shape[0] != owners.size:
            raise ValueError("Boundary edges must have shape (patches, pieces, 2).")
        if edges_valid.shape != edges.shape[:2]:
            raise ValueError("Boundary validity must match patch/piece axes.")
        if shape_values.ndim != 4 or shape_values.shape[:2] != edges.shape[:2]:
            raise ValueError(
                "Boundary shape values need patch/piece/quadrature/DOF axes."
            )
        if (
            shape_values.shape[-1] != routes.shape[-1]
            or parameters.shape != (shape_values.shape[2],)
            or weights.shape != (shape_values.shape[2],)
        ):
            raise ValueError("Boundary shape/rule dimensions are incompatible.")
        if np.any(np.sum(vertices_valid, axis=1) < 3):
            raise ValueError(
                "Every 2-D smoothing patch requires at least three vertices."
            )
        self.patch_kind = patch_kind
        self.owner_entities = jnp.asarray(owners)
        self.dof_routes = jnp.asarray(routes)
        self.dof_valid = jnp.asarray(route_valid)
        self.vertex_sources = jnp.asarray(sources)
        self.vertex_coefficients = jnp.asarray(coefficients)
        self.vertex_valid = jnp.asarray(vertices_valid)
        self.boundary_edges = jnp.asarray(edges)
        self.boundary_valid = jnp.asarray(edges_valid)
        self.boundary_shape_values = jnp.asarray(shape_values)
        self.rule_points = jnp.asarray(parameters)
        self.rule_weights = jnp.asarray(weights)
        self.layout_id = canonical_fingerprint(
            {
                "kind": "smoothing-patch-layout",
                "patch_kind": patch_kind,
                "owners": array_tree_fingerprint(owners),
                "dofs": array_tree_fingerprint(routes),
                "dof_valid": array_tree_fingerprint(route_valid),
                "vertex_sources": array_tree_fingerprint(sources),
                "vertex_coefficients": array_tree_fingerprint(coefficients),
                "vertex_valid": array_tree_fingerprint(vertices_valid),
                "boundary_edges": array_tree_fingerprint(edges),
                "boundary_valid": array_tree_fingerprint(edges_valid),
                "shape_values": array_tree_fingerprint(shape_values),
                "rule_points": array_tree_fingerprint(parameters),
                "rule_weights": array_tree_fingerprint(weights),
            }
        )


class SmoothingPatchGeometry(StrictModule):
    patch_vertices: Array
    area: Array
    centroid: Array
    boundary_points: Array
    boundary_lengths: Array
    boundary_normals: Array
    valid: Array


class SmoothingEvidence(StrictModule, NonTrainableState):
    positive_measure: Array
    closure_defect: Array
    partition_defect: Array
    affine_reproduction_defect: Array
    rigid_mode_count: int = eqx.field(static=True)
    extra_near_zero_modes: int = eqx.field(static=True)
    minimum_constrained_eigenvalue: Array
    energy_evidence: SmoothingEnergyEvidence = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        positive_measure: ArrayLike,
        closure_defect: ArrayLike,
        partition_defect: ArrayLike,
        affine_reproduction_defect: ArrayLike,
        rigid_mode_count: int,
        extra_near_zero_modes: int,
        minimum_constrained_eigenvalue: ArrayLike,
        /,
        *,
        energy_evidence: SmoothingEnergyEvidence = "none",
    ):
        positive = jnp.asarray(positive_measure, dtype=bool)
        closure = jnp.asarray(closure_defect)
        partition = jnp.asarray(partition_defect)
        affine = jnp.asarray(affine_reproduction_defect)
        minimum = jnp.asarray(minimum_constrained_eigenvalue)
        rigid = int(rigid_mode_count)
        extra = int(extra_near_zero_modes)
        if rigid < 0 or extra < 0 or minimum.shape != ():
            raise ValueError("Smoothing spectral evidence is invalid.")
        self.positive_measure = positive
        self.closure_defect = closure
        self.partition_defect = partition
        self.affine_reproduction_defect = affine
        self.rigid_mode_count = rigid
        self.extra_near_zero_modes = extra
        self.minimum_constrained_eigenvalue = minimum
        self.energy_evidence = energy_evidence
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "smoothing-evidence",
                "rigid_mode_count": rigid,
                "extra_near_zero_modes": extra,
                "energy_evidence": energy_evidence,
                "shapes": {
                    "positive": list(positive.shape),
                    "closure": list(closure.shape),
                    "partition": list(partition.shape),
                    "affine": list(affine.shape),
                },
            }
        )


def evaluate_smoothing_geometry(
    layout: SmoothingPatchLayout,
    coordinates: ArrayLike,
    /,
) -> SmoothingPatchGeometry:
    if not isinstance(layout, SmoothingPatchLayout):
        raise TypeError("layout must be SmoothingPatchLayout.")
    points = jnp.asarray(coordinates)
    safe_sources = jnp.where(layout.vertex_sources >= 0, layout.vertex_sources, 0)
    gathered = points[safe_sources]
    patch_vertices = jnp.sum(gathered * layout.vertex_coefficients[..., None], axis=2)
    coordinate_dimension = patch_vertices.shape[-1]
    start_indices = jnp.broadcast_to(
        layout.boundary_edges[..., 0, None],
        layout.boundary_edges.shape[:2] + (coordinate_dimension,),
    )
    end_indices = jnp.broadcast_to(
        layout.boundary_edges[..., 1, None],
        layout.boundary_edges.shape[:2] + (coordinate_dimension,),
    )
    start = jnp.take_along_axis(patch_vertices, start_indices, axis=1)
    end = jnp.take_along_axis(patch_vertices, end_indices, axis=1)
    tangent = end - start
    length = jnp.sqrt(jnp.sum(tangent**2, axis=-1))
    safe_length = jnp.where(layout.boundary_valid, length, 1.0)
    normal = jnp.stack((tangent[..., 1], -tangent[..., 0]), axis=-1)
    normal = normal / safe_length[..., None]
    normal = jnp.where(layout.boundary_valid[..., None], normal, 0.0)
    length = jnp.where(layout.boundary_valid, length, 0.0)
    cross = start[..., 0] * end[..., 1] - end[..., 0] * start[..., 1]
    cross = jnp.where(layout.boundary_valid, cross, 0.0)
    area = 0.5 * jnp.abs(jnp.sum(cross, axis=1))
    centroid = (
        jnp.sum(patch_vertices * layout.vertex_valid[..., None], axis=1)
        / jnp.sum(layout.vertex_valid, axis=1)[:, None]
    )
    parameter = layout.rule_points
    boundary_points = (1.0 - parameter)[None, None, :, None] * start[
        :, :, None, :
    ] + parameter[None, None, :, None] * end[:, :, None, :]
    valid = (
        jnp.isfinite(area)
        & (area > 0.0)
        & jnp.all(jnp.isfinite(patch_vertices), axis=(1, 2))
    )
    return SmoothingPatchGeometry(
        patch_vertices=patch_vertices,
        area=area,
        centroid=centroid,
        boundary_points=boundary_points,
        boundary_lengths=length,
        boundary_normals=normal,
        valid=valid,
    )


__all__ = [
    "SmoothingEnergyEvidence",
    "SmoothingEvidence",
    "SmoothingPatchGeometry",
    "SmoothingPatchKind",
    "SmoothingPatchLayout",
    "evaluate_smoothing_geometry",
]
