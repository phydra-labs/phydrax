#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._cell_complex import PolygonalConnectivity
from .._conservation_boundary import AbstractConservationBoundary
from ._geometry_protocol import (
    FiniteVolumeGeometryStatus,
    FiniteVolumeStageFaceBlock,
    FiniteVolumeStageFaceLayout,
    FiniteVolumeStageGeometryEvidence,
    FiniteVolumeStageMetrics,
)
from ._physical_boundaries import SlipWallBoundary
from ._unstructured import UnstructuredFiniteVolumeDiscretization
from ._unstructured_embedded_boundary import EmbeddedBoundaryMetrics


class UnstructuredEmbeddedBoundarySet(StrictModule, NonTrainableState):
    """Complete stationary cut-wall policy ownership indexed by body tag."""

    body_tags: tuple[int, ...] = eqx.field(static=True)
    boundaries: tuple[SlipWallBoundary, ...]
    boundary_set_id: str = eqx.field(static=True)

    def __init__(
        self,
        boundaries: Mapping[int, AbstractConservationBoundary],
        /,
    ):
        if not isinstance(boundaries, Mapping):
            raise TypeError("boundaries must map embedded body tags to policies.")
        normalized: dict[int, AbstractConservationBoundary] = {}
        for raw_tag, boundary in boundaries.items():
            if not isinstance(raw_tag, (int, np.integer)) or isinstance(
                raw_tag, (bool, np.bool_)
            ):
                raise TypeError("Embedded body tags must be nonnegative integers.")
            tag = int(raw_tag)
            if tag < 0:
                raise ValueError("Embedded body tags must be nonnegative integers.")
            if tag in normalized:
                raise ValueError("Embedded body tags must be unique.")
            normalized[tag] = boundary
        if not normalized:
            raise ValueError("Embedded boundary policies must not be empty.")

        tags = tuple(sorted(normalized))
        policies: list[SlipWallBoundary] = []
        for tag in tags:
            policy = normalized[tag]
            if not isinstance(policy, SlipWallBoundary):
                raise TypeError(
                    "Stationary embedded cut faces currently require SlipWallBoundary."
                )
            policies.append(policy)
        self.body_tags = tags
        self.boundaries = tuple(policies)
        self.boundary_set_id = canonical_fingerprint(
            {
                "kind": "unstructured-embedded-boundary-set",
                "bodies": [
                    {"tag": tag, "boundary": policy.boundary_id}
                    for tag, policy in zip(tags, policies, strict=True)
                ],
            }
        )


def _physical_stage_block(
    discretization: UnstructuredFiniteVolumeDiscretization,
    metrics: EmbeddedBoundaryMetrics,
    block,
    topology_epoch_id: str,
    block_index: int,
    /,
) -> FiniteVolumeStageFaceBlock | None:
    source_face_ids = np.asarray(block.face_ids)
    if np.any(source_face_ids < 0) or np.any(
        source_face_ids >= discretization.face_measures.size
    ):
        raise ValueError("Physical face-block IDs index outside embedded apertures.")

    source_owners = np.asarray(block.owner_cells)
    source_neighbours = np.asarray(block.neighbour_cells)
    active_cells = np.asarray(metrics.active_fluid_cells, dtype=bool)
    aperture = np.asarray(metrics.face_open_fraction)[source_face_ids]
    internal = source_neighbours >= 0
    safe_neighbours = np.where(internal, source_neighbours, 0)
    route_active = (
        np.asarray(block.active_mask, dtype=bool)
        & (aperture > 0.0)
        & (np.asarray(block.face_measures) * aperture > 0.0)
        & active_cells[source_owners]
        & (~internal | active_cells[safe_neighbours])
    )
    route_rows = np.flatnonzero(route_active).astype(np.int32)
    if route_rows.size == 0:
        return None

    compacted_face_ids = source_face_ids[route_rows]
    route_indices = jnp.asarray(route_rows, dtype=jnp.int32)
    face_ids = jnp.asarray(compacted_face_ids, dtype=jnp.int32)
    owners = jnp.asarray(block.owner_cells, dtype=jnp.int32)[route_indices]
    neighbours = jnp.asarray(block.neighbour_cells, dtype=jnp.int32)[route_indices]
    policy_ids = jnp.asarray(block.boundary_patch_ids, dtype=jnp.int32)[route_indices]
    compact_aperture = metrics.face_open_fraction[face_ids]
    measures = metrics.open_face_measures[face_ids]

    edge_vertex_ids = np.asarray(
        discretization.connectivity.edges,
        dtype=np.int32,
    )[compacted_face_ids]
    edge_points = np.asarray(discretization.vertices)[edge_vertex_ids]
    edge_values = np.asarray(metrics.vertex_values)[edge_vertex_ids]
    open_fraction = np.asarray(metrics.face_open_fraction)[compacted_face_ids]
    partial = open_fraction < 1.0
    open_start = edge_points[:, 0].copy()
    open_stop = edge_points[:, 1].copy()
    tangent = edge_points[:, 1] - edge_points[:, 0]
    start_is_fluid = edge_values[:, 0] >= 0.0
    start_partial = partial & start_is_fluid
    stop_partial = partial & ~start_is_fluid
    open_stop[start_partial] = (
        edge_points[start_partial, 0]
        + open_fraction[start_partial, None] * tangent[start_partial]
    )
    open_start[stop_partial] = (
        edge_points[stop_partial, 1]
        - open_fraction[stop_partial, None] * tangent[stop_partial]
    )
    open_tangent = open_stop - open_start
    open_centers = 0.5 * (open_start + open_stop)
    gauss_offset = open_tangent / (2.0 * np.sqrt(3.0))
    open_points = np.stack(
        (open_centers - gauss_offset, open_centers + gauss_offset),
        axis=1,
    )
    partial_mask = jnp.asarray(partial, dtype=bool)
    source_centers = block.face_centers[route_indices]
    centers = jnp.where(
        partial_mask[:, None],
        jnp.asarray(open_centers, dtype=source_centers.dtype),
        source_centers,
    )
    source_points = discretization.face_quadrature_points[face_ids]
    points = jnp.where(
        partial_mask[:, None, None],
        jnp.asarray(open_points, dtype=source_points.dtype),
        source_points,
    )
    area_vectors = block.area_vectors[route_indices] * compact_aperture[:, None]
    source_weights = discretization.face_quadrature_weights[face_ids]
    open_weights = jnp.broadcast_to(
        0.5 * measures[:, None],
        source_weights.shape,
    )
    weights = jnp.where(partial_mask[:, None], open_weights, source_weights)
    active_mask = jnp.ones((route_rows.size,), dtype=bool)
    spatial_shape = tuple(centers.shape)
    quadrature_shape = tuple(weights.shape)
    layout = FiniteVolumeStageFaceLayout(
        face_ids=face_ids,
        owner_cells=owners,
        neighbour_cells=neighbours,
        active_mask=active_mask,
        boundary_policy_ids=policy_ids,
        boundary_policy_count=len(discretization.boundary_patch_names),
        spatial_shape=spatial_shape,
        quadrature_shape=quadrature_shape,
        block_id=canonical_fingerprint(
            {
                "kind": "stationary-embedded-physical-stage-face-route",
                "topology_epoch": topology_epoch_id,
                "topology": discretization.topology_id,
                "metrics": metrics.metrics_id,
                "block_index": block_index,
                "face_ids": array_tree_fingerprint(face_ids),
                "owner_cells": array_tree_fingerprint(owners),
                "neighbour_cells": array_tree_fingerprint(neighbours),
                "boundary_policy_ids": array_tree_fingerprint(policy_ids),
                "boundary_policy_count": len(discretization.boundary_patch_names),
                "active_mask": array_tree_fingerprint(active_mask),
                "spatial_shape": spatial_shape,
                "quadrature_shape": quadrature_shape,
            }
        ),
        block_kind="physical",
    )
    return FiniteVolumeStageFaceBlock(
        layout=layout,
        face_centers=centers,
        area_vectors=area_vectors,
        face_measures=measures,
        quadrature_points=points,
        quadrature_weights=weights,
        quadrature_grid_normal_velocity=jnp.zeros_like(weights),
    )


def _cut_stage_block(
    discretization: UnstructuredFiniteVolumeDiscretization,
    metrics: EmbeddedBoundaryMetrics,
    boundary_set: UnstructuredEmbeddedBoundarySet,
    topology_epoch_id: str,
    policy_ids,
    cut_face_start: int,
    /,
) -> FiniteVolumeStageFaceBlock | None:
    active = (
        np.asarray(metrics.cut_face_active, dtype=bool)
        & np.asarray(metrics.active_fluid_cells, dtype=bool)
        & (np.asarray(metrics.cut_face_measures) > 0.0)
    )
    route_rows = np.flatnonzero(active).astype(np.int32)
    if route_rows.size == 0:
        return None
    if cut_face_start + int(route_rows[-1]) > np.iinfo(np.int32).max:
        raise ValueError("Embedded cut-face IDs exceed the int32 route range.")

    route_indices = jnp.asarray(route_rows, dtype=jnp.int32)
    face_ids = jnp.asarray(cut_face_start + route_rows, dtype=jnp.int32)
    owners = route_indices
    neighbours = jnp.full((route_rows.size,), -1, dtype=jnp.int32)
    compact_policy_ids = jnp.asarray(policy_ids, dtype=jnp.int32)[route_indices]
    measures = metrics.cut_face_measures[route_indices]
    centers = metrics.cut_face_centers[route_indices]
    normals = metrics.cut_face_normals[route_indices]
    weights = measures[:, None]
    active_mask = jnp.ones((route_rows.size,), dtype=bool)
    spatial_shape = tuple(centers.shape)
    quadrature_shape = tuple(weights.shape)
    layout = FiniteVolumeStageFaceLayout(
        face_ids=face_ids,
        owner_cells=owners,
        neighbour_cells=neighbours,
        active_mask=active_mask,
        boundary_policy_ids=compact_policy_ids,
        boundary_policy_count=len(boundary_set.boundaries),
        spatial_shape=spatial_shape,
        quadrature_shape=quadrature_shape,
        block_id=canonical_fingerprint(
            {
                "kind": "stationary-embedded-cut-stage-face-route",
                "topology_epoch": topology_epoch_id,
                "topology": discretization.topology_id,
                "metrics": metrics.metrics_id,
                "boundary_set": boundary_set.boundary_set_id,
                "face_ids": array_tree_fingerprint(face_ids),
                "owner_cells": array_tree_fingerprint(owners),
                "neighbour_cells": array_tree_fingerprint(neighbours),
                "boundary_policy_ids": array_tree_fingerprint(compact_policy_ids),
                "boundary_policy_count": len(boundary_set.boundaries),
                "active_mask": array_tree_fingerprint(active_mask),
                "spatial_shape": spatial_shape,
                "quadrature_shape": quadrature_shape,
            }
        ),
        block_kind="cut",
    )
    return FiniteVolumeStageFaceBlock(
        layout=layout,
        face_centers=centers,
        area_vectors=normals * measures[:, None],
        face_measures=measures,
        quadrature_points=centers[:, None, :],
        quadrature_weights=weights,
        quadrature_grid_normal_velocity=jnp.zeros_like(weights),
    )


def _translated_evidence(
    discretization: UnstructuredFiniteVolumeDiscretization,
    metrics: EmbeddedBoundaryMetrics,
    evidence_version,
    /,
) -> FiniteVolumeStageGeometryEvidence:
    embedded = metrics.evidence
    aperture_failed = (
        embedded.aperture_closure_defect > embedded.aperture_closure_tolerance
    ).astype(jnp.int32)
    owners = jnp.asarray(discretization.owner_cells, dtype=jnp.int32)
    neighbours = jnp.asarray(discretization.neighbour_cells, dtype=jnp.int32)
    aperture_failure_by_cell = jnp.zeros(
        (discretization.cell_count,),
        dtype=jnp.int32,
    )
    aperture_failure_by_cell = aperture_failure_by_cell.at[owners].max(aperture_failed)
    internal = neighbours >= 0
    safe_neighbours = jnp.where(internal, neighbours, 0)
    aperture_failure_by_cell = aperture_failure_by_cell.at[safe_neighbours].max(
        jnp.where(internal, aperture_failed, 0)
    )
    aperture_defect = aperture_failure_by_cell.astype(metrics.fluid_cell_volumes.dtype)
    aperture_tolerance = jnp.zeros_like(aperture_defect)
    evidence_policy_id = canonical_fingerprint(
        {
            "kind": "stationary-embedded-stage-geometry-evidence-policy",
            "metric_policy": metrics.stabilization_policy_id,
            "translation": {
                "coordinate_effective_volume": "embedded-volume-closure",
                "face_closure": "embedded-cut-face-closure",
                "gcl_identity": "incident-aperture-closure-failure-indicator",
                "expected_order": 0,
                "proposed_reduction_factor": 1.0,
            },
            "numeric_version": discretization.numeric_version,
        }
    )
    return FiniteVolumeStageGeometryEvidence(
        coordinate_effective_volume_defect=embedded.volume_closure_defect,
        coordinate_effective_volume_tolerance=embedded.volume_closure_tolerance,
        face_closure_defect=embedded.cut_face_closure_defect,
        face_closure_tolerance=embedded.cut_face_closure_tolerance,
        gcl_identity_defect=aperture_defect,
        gcl_identity_tolerance=aperture_tolerance,
        expected_order=0,
        proposed_reduction_factor=1.0,
        passed=embedded.passed,
        status=jnp.where(
            embedded.passed,
            int(FiniteVolumeGeometryStatus.SUCCESS),
            int(FiniteVolumeGeometryStatus.FAILED),
        ),
        evidence_version=evidence_version,
        policy_id=evidence_policy_id,
    )


def lower_embedded_stage_metrics(
    discretization: UnstructuredFiniteVolumeDiscretization,
    metrics: EmbeddedBoundaryMetrics,
    boundary_set: UnstructuredEmbeddedBoundarySet,
    topology_epoch_id: str,
    geometry_version,
    evidence_version,
    *,
    time=0.0,
) -> FiniteVolumeStageMetrics:
    """Host-lower stationary 2-D cut geometry into compact stage face blocks."""

    if not isinstance(discretization, UnstructuredFiniteVolumeDiscretization):
        raise TypeError("discretization must be UnstructuredFiniteVolumeDiscretization.")
    if discretization.cell_dimension != 2 or not isinstance(
        discretization.connectivity, PolygonalConnectivity
    ):
        raise ValueError("Embedded stage lowering requires stationary 2-D polygons.")
    if not isinstance(metrics, EmbeddedBoundaryMetrics):
        raise TypeError("metrics must be EmbeddedBoundaryMetrics.")
    if not isinstance(boundary_set, UnstructuredEmbeddedBoundarySet):
        raise TypeError("boundary_set must be UnstructuredEmbeddedBoundarySet.")
    if (
        not isinstance(topology_epoch_id, str)
        or not topology_epoch_id
        or (topology_epoch_id != topology_epoch_id.strip())
    ):
        raise ValueError("topology_epoch_id must be a non-empty canonical string.")
    if (
        metrics.prepared_id != discretization.prepared_id
        or metrics.topology_id != discretization.topology_id
        or metrics.geometry_id != discretization.geometry_id
    ):
        raise ValueError(
            "Embedded metrics and discretization must share prepared topology and geometry."
        )
    if metrics.fluid_cell_centers.shape != discretization.cell_centers.shape:
        raise ValueError(
            "Embedded fluid centroids must match the discretization cell-center shape."
        )

    body_tags = np.asarray(metrics.body_tags)
    if (
        body_tags.shape != (discretization.cell_count,)
        or body_tags.dtype.kind not in "iu"
    ):
        raise ValueError("Embedded body tags must contain one integer per cell.")
    known_tags = np.asarray(boundary_set.body_tags, dtype=body_tags.dtype)
    matches = body_tags[:, None] == known_tags[None, :]
    complete = np.all(np.sum(matches, axis=1) == 1) and np.all(np.any(matches, axis=0))
    if not complete:
        raise ValueError(
            "Embedded boundary policies must cover every metric body tag exactly."
        )
    policy_ids = jnp.asarray(np.argmax(matches, axis=1), dtype=jnp.int32)

    physical_blocks_list: list[FiniteVolumeStageFaceBlock] = []
    for block_index, block in enumerate(discretization.face_blocks):
        stage_block = _physical_stage_block(
            discretization,
            metrics,
            block,
            topology_epoch_id,
            block_index,
        )
        if stage_block is not None:
            physical_blocks_list.append(stage_block)
    physical_blocks = tuple(physical_blocks_list)
    cut_block = _cut_stage_block(
        discretization,
        metrics,
        boundary_set,
        topology_epoch_id,
        policy_ids,
        int(discretization.face_measures.size),
    )
    stage_blocks = physical_blocks if cut_block is None else (*physical_blocks, cut_block)
    original_cell_centers = jnp.asarray(
        discretization.cell_centers,
        dtype=metrics.fluid_cell_centers.dtype,
    )
    active_cells = metrics.active_fluid_cells[:, None]
    active_cut_cells = active_cells & metrics.cut_cells[:, None]
    safe_cell_centers = jnp.where(
        active_cells,
        original_cell_centers,
        jnp.zeros_like(original_cell_centers),
    )
    stage_cell_centers = jnp.where(
        active_cut_cells,
        metrics.fluid_cell_centers,
        safe_cell_centers,
    )
    geometry_layout_id = canonical_fingerprint(
        {
            "kind": "stationary-embedded-stage-geometry-layout",
            "topology": discretization.topology_id,
            "geometry_family": discretization.geometry_id,
            "metrics": metrics.metrics_id,
            "boundary_set": boundary_set.boundary_set_id,
            "cell_shape": tuple(metrics.fluid_cell_volumes.shape),
            "cell_dtype": str(metrics.fluid_cell_volumes.dtype),
            "blocks": [
                {
                    "face_ids": array_tree_fingerprint(block.layout.face_ids),
                    "owner_cells": array_tree_fingerprint(block.layout.owner_cells),
                    "neighbour_cells": array_tree_fingerprint(
                        block.layout.neighbour_cells
                    ),
                    "active_mask": array_tree_fingerprint(block.layout.active_mask),
                    "boundary_policy_ids": array_tree_fingerprint(
                        block.layout.boundary_policy_ids
                    ),
                    "boundary_policy_count": block.layout.boundary_policy_count,
                    "block_kind": block.layout.block_kind,
                    "spatial_shape": block.layout.spatial_shape,
                    "quadrature_shape": block.layout.quadrature_shape,
                }
                for block in stage_blocks
            ],
        }
    )
    evidence = _translated_evidence(
        discretization,
        metrics,
        evidence_version,
    )
    zero_volume_rate = jnp.zeros_like(metrics.fluid_cell_volumes)
    return FiniteVolumeStageMetrics(
        topology_epoch_id=topology_epoch_id,
        geometry_family_id=discretization.geometry_id,
        geometry_layout_id=geometry_layout_id,
        geometry_version=geometry_version,
        time=jnp.asarray(time, dtype=metrics.fluid_cell_volumes.dtype),
        effective_cell_volumes=metrics.fluid_cell_volumes,
        coordinate_effective_cell_volumes=metrics.fluid_cell_volumes,
        mesh_volume_rate=zero_volume_rate,
        cell_centers=stage_cell_centers,
        active_cell_mask=metrics.active_fluid_cells,
        face_blocks=stage_blocks,
        evidence=evidence,
    )


__all__ = [
    "UnstructuredEmbeddedBoundarySet",
    "lower_embedded_stage_metrics",
]
