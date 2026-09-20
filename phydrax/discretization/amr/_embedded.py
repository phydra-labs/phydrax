#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Sharp 2-D apertured embedded-boundary realization over mapped patch geometry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._sharp_clipping import (
    clip_positive_polygon,
    open_positive_segment,
    polygon_measure_centroid,
)
from ._geometry import VariablePatchGeometryState


EmbeddedLevelSet = Callable[[Array, Array, Any], ArrayLike]


class VariablePatchEmbeddedBoundaryEvidence(StrictModule, NonTrainableState):
    """Host-certified volume and apertured-face closure evidence."""

    topology_epoch_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    body_id: str = eqx.field(static=True)
    cut_cell_count: int = eqx.field(static=True)
    small_cell_count: int = eqx.field(static=True)
    maximum_volume_closure_defect: float = eqx.field(static=True)
    maximum_face_closure_defect: float = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        topology_epoch_id: str,
        geometry_layout_id: str,
        body_id: str,
        cut_cell_count: int,
        small_cell_count: int,
        maximum_volume_closure_defect: float,
        maximum_face_closure_defect: float,
        /,
    ):
        volume_defect = float(maximum_volume_closure_defect)
        face_defect = float(maximum_face_closure_defect)
        if (
            not topology_epoch_id
            or not geometry_layout_id
            or not body_id
            or int(cut_cell_count) < 0
            or int(small_cell_count) < 0
            or not np.isfinite(volume_defect)
            or volume_defect < 0.0
            or not np.isfinite(face_defect)
            or face_defect < 0.0
        ):
            raise ValueError("Variable patch embedded-boundary evidence is invalid.")
        self.topology_epoch_id = topology_epoch_id
        self.geometry_layout_id = geometry_layout_id
        self.body_id = body_id
        self.cut_cell_count = int(cut_cell_count)
        self.small_cell_count = int(small_cell_count)
        self.maximum_volume_closure_defect = volume_defect
        self.maximum_face_closure_defect = face_defect
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "variable-patch-embedded-boundary-evidence",
                "epoch": topology_epoch_id,
                "geometry_layout": geometry_layout_id,
                "body": body_id,
                "cut_cells": int(cut_cell_count),
                "small_cells": int(small_cell_count),
                "volume_closure": volume_defect,
                "face_closure": face_defect,
            }
        )


class VariablePatchEmbeddedBoundaryMetrics(StrictModule, NonTrainableState):
    """Bounded sharp cut-cell and background-face aperture metrics.

    Level/bucket/axis arrays preserve the geometry state's static envelope signatures.
    Shared reference faces are clipped once through a canonical host face-key cache.
    """

    geometry: VariablePatchGeometryState
    volume_fraction: tuple[tuple[Array, ...], ...]
    fluid_cell_volumes: tuple[tuple[Array, ...], ...]
    fluid_cell_centers: tuple[tuple[Array, ...], ...]
    cut_face_centers: tuple[tuple[Array, ...], ...]
    cut_face_normals: tuple[tuple[Array, ...], ...]
    cut_face_measures: tuple[tuple[Array, ...], ...]
    cut_face_active: tuple[tuple[Array, ...], ...]
    open_face_fractions: tuple[tuple[tuple[Array, ...], ...], ...]
    open_face_measures: tuple[tuple[tuple[Array, ...], ...], ...]
    open_face_centers: tuple[tuple[tuple[Array, ...], ...], ...]
    open_face_endpoints: tuple[tuple[tuple[Array, ...], ...], ...]
    body_tags: tuple[tuple[Array, ...], ...]
    small_cells: tuple[tuple[Array, ...], ...]
    evidence: VariablePatchEmbeddedBoundaryEvidence
    metrics_id: str = eqx.field(static=True)


class VariablePatchEmbeddedBoundaryPlan(StrictModule, NonTrainableState):
    """Host-only exact linear-edge clipping for stationary 2-D patch epochs."""

    geometry_plan_id: str = eqx.field(static=True)
    level_set: EmbeddedLevelSet = eqx.field(static=True)
    body_id: str = eqx.field(static=True)
    body_tag: int = eqx.field(static=True)
    fluid_sign: Literal["positive", "negative"] = eqx.field(static=True)
    minimum_volume_fraction: float = eqx.field(static=True)
    closure_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        geometry_plan_id: str,
        level_set: EmbeddedLevelSet,
        body_id: str,
        /,
        *,
        body_tag: int = 0,
        fluid_sign: Literal["positive", "negative"] = "positive",
        minimum_volume_fraction: float = 0.1,
        closure_tolerance: float = 1.0e-12,
    ):
        identifier = str(geometry_plan_id)
        body = str(body_id)
        threshold = float(minimum_volume_fraction)
        tolerance = float(closure_tolerance)
        if (
            not identifier
            or not callable(level_set)
            or not body
            or int(body_tag) < 0
            or fluid_sign not in ("positive", "negative")
            or not np.isfinite(threshold)
            or threshold <= 0.0
            or threshold > 1.0
            or not np.isfinite(tolerance)
            or tolerance < 0.0
        ):
            raise ValueError("Variable patch embedded-boundary plan is invalid.")
        self.geometry_plan_id = identifier
        self.level_set = level_set
        self.body_id = body
        self.body_tag = int(body_tag)
        self.fluid_sign = fluid_sign
        self.minimum_volume_fraction = threshold
        self.closure_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "variable-patch-embedded-boundary-plan",
                "geometry": identifier,
                "body": body,
                "body_tag": int(body_tag),
                "fluid_sign": fluid_sign,
                "minimum_volume_fraction": threshold,
                "closure_tolerance": tolerance,
            }
        )

    def prepare(
        self,
        geometry: VariablePatchGeometryState,
        args: Any = None,
        /,
    ) -> VariablePatchEmbeddedBoundaryMetrics:
        if not isinstance(geometry, VariablePatchGeometryState):
            raise TypeError("Variable patch EB realization requires geometry state.")
        if geometry.plan.plan_id != self.geometry_plan_id:
            raise ValueError(
                "Embedded-boundary plan is bound to another patch geometry plan."
            )
        if isinstance(geometry.valid, jax.core.Tracer) or not bool(geometry.valid):
            raise ValueError(
                "Embedded boundaries require an accepted host geometry state."
            )
        if any(level.dimension != 2 for level in geometry.plan.topology.plan.levels):
            raise ValueError(
                "Variable patch sharp clipping currently supports 2-D patches."
            )
        fractions_by_level = []
        volumes_by_level = []
        centers_by_level = []
        cut_centers_by_level = []
        cut_normals_by_level = []
        cut_measures_by_level = []
        cut_active_by_level = []
        open_fractions_by_level = []
        open_measures_by_level = []
        open_centers_by_level = []
        open_endpoints_by_level = []
        tags_by_level = []
        small_by_level = []
        cut_count = 0
        small_count = 0
        maximum_volume_defect = 0.0
        maximum_face_defect = 0.0
        for level, (
            level_plan,
            metadata,
            vertices,
            cell_centers,
            cell_volumes,
            active,
        ) in enumerate(
            zip(
                geometry.plan.topology.plan.levels,
                geometry.plan.topology.levels,
                geometry.vertex_coordinates,
                geometry.cell_centers,
                geometry.cell_volumes,
                geometry.active_cell_masks,
                strict=True,
            )
        ):
            face_cache: dict[
                tuple[int, tuple[int, int]],
                tuple[float, float, np.ndarray, np.ndarray, np.ndarray],
            ] = {}
            level_fraction = []
            level_volumes = []
            level_centers = []
            level_cut_centers = []
            level_cut_normals = []
            level_cut_measures = []
            level_cut_active = []
            level_open_fractions = []
            level_open_measures = []
            level_open_centers = []
            level_open_endpoints = []
            level_tags = []
            level_small = []
            for bucket_index, bucket in enumerate(level_plan.buckets):
                envelope = bucket.signature.envelope_shape
                shape = (bucket.lane_capacity,) + envelope
                fraction = np.zeros(shape, dtype=np.float64)
                fluid_volume = np.zeros(shape, dtype=np.float64)
                fluid_center = np.zeros(shape + (2,), dtype=np.float64)
                cut_center = np.zeros(shape + (2,), dtype=np.float64)
                cut_normal = np.zeros(shape + (2,), dtype=np.float64)
                cut_measure = np.zeros(shape, dtype=np.float64)
                cut_active = np.zeros(shape, dtype=np.bool_)
                tags = np.full(shape, -1, dtype=np.int32)
                small = np.zeros(shape, dtype=np.bool_)
                face_shapes = (
                    (bucket.lane_capacity, envelope[0] + 1, envelope[1]),
                    (bucket.lane_capacity, envelope[0], envelope[1] + 1),
                )
                open_fraction = [
                    np.zeros(face_shape, dtype=np.float64) for face_shape in face_shapes
                ]
                open_measure = [
                    np.zeros(face_shape, dtype=np.float64) for face_shape in face_shapes
                ]
                open_center = [
                    np.zeros(face_shape + (2,), dtype=np.float64)
                    for face_shape in face_shapes
                ]
                open_endpoints = [
                    np.zeros(face_shape + (2, 2), dtype=np.float64)
                    for face_shape in face_shapes
                ]
                vertex = np.asarray(vertices[bucket_index])
                full_volume = np.asarray(cell_volumes[bucket_index])
                full_center = np.asarray(cell_centers[bucket_index])
                active_cells = np.asarray(active[bucket_index], dtype=np.bool_)
                patch_lower = np.asarray(metadata.lower[bucket_index], dtype=np.int32)
                for lane in range(bucket.lane_capacity):
                    if not np.any(active_cells[lane]):
                        continue
                    for local in np.argwhere(active_cells[lane]):
                        i, j = (int(value) for value in local)
                        index = (i, j)
                        polygon = np.asarray(
                            (
                                vertex[lane, i, j],
                                vertex[lane, i + 1, j],
                                vertex[lane, i + 1, j + 1],
                                vertex[lane, i, j + 1],
                            )
                        )
                        values = np.asarray(
                            self.level_set(jnp.asarray(polygon), geometry.time, args),
                            dtype=np.float64,
                        )
                        if values.shape != (4,) or np.any(~np.isfinite(values)):
                            raise ValueError(
                                "Embedded level set must return four finite corner values."
                            )
                        if self.fluid_sign == "negative":
                            values = -values
                        if np.any(values == 0.0):
                            raise ValueError(
                                "Embedded cut classification is ambiguous at a vertex."
                            )
                        global_cell = tuple(patch_lower[lane] + local)
                        edge_specs = (
                            (
                                0,
                                (global_cell[0], global_cell[1]),
                                polygon[3],
                                polygon[0],
                                values[3],
                                values[0],
                                (i, j),
                            ),
                            (
                                0,
                                (global_cell[0] + 1, global_cell[1]),
                                polygon[1],
                                polygon[2],
                                values[1],
                                values[2],
                                (i + 1, j),
                            ),
                            (
                                1,
                                (global_cell[0], global_cell[1]),
                                polygon[0],
                                polygon[1],
                                values[0],
                                values[1],
                                (i, j),
                            ),
                            (
                                1,
                                (global_cell[0], global_cell[1] + 1),
                                polygon[2],
                                polygon[3],
                                values[2],
                                values[3],
                                (i, j + 1),
                            ),
                        )
                        closure_vector = np.zeros((2,), dtype=np.float64)
                        for (
                            axis,
                            face_key,
                            start,
                            stop,
                            first,
                            second,
                            face_index,
                        ) in edge_specs:
                            cache_key = (axis, face_key)
                            metric = face_cache.get(cache_key)
                            if metric is None:
                                metric = open_positive_segment(start, stop, first, second)
                                face_cache[cache_key] = metric
                            (
                                fraction_value,
                                measure_value,
                                center_value,
                                first_open,
                                second_open,
                            ) = metric
                            route = (lane,) + face_index
                            open_fraction[axis][route] = fraction_value
                            open_measure[axis][route] = measure_value
                            open_center[axis][route] = center_value
                            open_endpoints[axis][route] = np.asarray(
                                (first_open, second_open)
                            )
                            edge = stop - start
                            closure_vector += fraction_value * np.asarray(
                                (edge[1], -edge[0])
                            )
                        clipped, intersections = clip_positive_polygon(polygon, values)
                        if len(intersections) not in (0, 2):
                            raise ValueError(
                                "Embedded patch cell has ambiguous edge crossings."
                            )
                        clipped_volume, clipped_center = polygon_measure_centroid(clipped)
                        solid_polygon, _ = clip_positive_polygon(polygon, -values)
                        solid_volume, _ = polygon_measure_centroid(solid_polygon)
                        total = float(full_volume[(lane,) + index])
                        if np.all(values > 0.0):
                            clipped_volume = total
                            solid_volume = 0.0
                            clipped_center = np.asarray(full_center[(lane,) + index])
                        elif np.all(values < 0.0):
                            clipped_volume = 0.0
                            solid_volume = total
                        volume_defect = abs(clipped_volume + solid_volume - total)
                        maximum_volume_defect = max(maximum_volume_defect, volume_defect)
                        if volume_defect > self.closure_tolerance * max(total, 1.0):
                            raise ValueError(
                                "Embedded patch cell violates fluid/solid volume closure."
                            )
                        fluid_volume[(lane,) + index] = clipped_volume
                        fraction[(lane,) + index] = clipped_volume / total
                        fluid_center[(lane,) + index] = clipped_center
                        tags[(lane,) + index] = self.body_tag
                        if 0.0 < clipped_volume < total:
                            if len(intersections) != 2:
                                raise ValueError(
                                    "Embedded cut cell lacks one linear cut segment."
                                )
                            first, second = intersections
                            midpoint = 0.5 * (first + second)
                            tangent = second - first
                            measure = np.linalg.norm(tangent)
                            if not np.isfinite(measure) or measure <= 0.0:
                                raise ValueError(
                                    "Embedded cut segment has invalid measure."
                                )
                            normal = np.asarray((tangent[1], -tangent[0])) / measure
                            if np.dot(normal, midpoint - clipped_center) < 0.0:
                                normal = -normal
                            if not np.isfinite(normal).all() or not np.isclose(
                                np.linalg.norm(normal),
                                1.0,
                                rtol=0.0,
                                atol=1.0e-12,
                            ):
                                raise ValueError(
                                    "Embedded cut normal is not unit length."
                                )
                            cut_center[(lane,) + index] = midpoint
                            cut_normal[(lane,) + index] = normal
                            cut_measure[(lane,) + index] = measure
                            cut_active[(lane,) + index] = True
                            closure_vector += normal * measure
                            cut_count += 1
                        face_defect = float(np.linalg.norm(closure_vector))
                        maximum_face_defect = max(maximum_face_defect, face_defect)
                        if face_defect > self.closure_tolerance * max(
                            np.sqrt(total), 1.0
                        ):
                            raise ValueError(
                                "Embedded patch cell violates apertured face closure."
                            )
                        small[(lane,) + index] = (
                            0.0 < fraction[(lane,) + index] < self.minimum_volume_fraction
                        )
                        small_count += int(small[(lane,) + index])
                level_fraction.append(jnp.asarray(fraction))
                level_volumes.append(jnp.asarray(fluid_volume))
                level_centers.append(jnp.asarray(fluid_center))
                level_cut_centers.append(jnp.asarray(cut_center))
                level_cut_normals.append(jnp.asarray(cut_normal))
                level_cut_measures.append(jnp.asarray(cut_measure))
                level_cut_active.append(jnp.asarray(cut_active))
                level_open_fractions.append(
                    tuple(jnp.asarray(value) for value in open_fraction)
                )
                level_open_measures.append(
                    tuple(jnp.asarray(value) for value in open_measure)
                )
                level_open_centers.append(
                    tuple(jnp.asarray(value) for value in open_center)
                )
                level_open_endpoints.append(
                    tuple(jnp.asarray(value) for value in open_endpoints)
                )
                level_tags.append(jnp.asarray(tags))
                level_small.append(jnp.asarray(small))
            fractions_by_level.append(tuple(level_fraction))
            volumes_by_level.append(tuple(level_volumes))
            centers_by_level.append(tuple(level_centers))
            cut_centers_by_level.append(tuple(level_cut_centers))
            cut_normals_by_level.append(tuple(level_cut_normals))
            cut_measures_by_level.append(tuple(level_cut_measures))
            cut_active_by_level.append(tuple(level_cut_active))
            open_fractions_by_level.append(tuple(level_open_fractions))
            open_measures_by_level.append(tuple(level_open_measures))
            open_centers_by_level.append(tuple(level_open_centers))
            open_endpoints_by_level.append(tuple(level_open_endpoints))
            tags_by_level.append(tuple(level_tags))
            small_by_level.append(tuple(level_small))
        evidence = VariablePatchEmbeddedBoundaryEvidence(
            geometry.plan.topology.epoch.epoch_id,
            geometry.plan.geometry_layout_id,
            self.body_id,
            cut_count,
            small_count,
            maximum_volume_defect,
            maximum_face_defect,
        )
        return VariablePatchEmbeddedBoundaryMetrics(
            geometry,
            tuple(fractions_by_level),
            tuple(volumes_by_level),
            tuple(centers_by_level),
            tuple(cut_centers_by_level),
            tuple(cut_normals_by_level),
            tuple(cut_measures_by_level),
            tuple(cut_active_by_level),
            tuple(open_fractions_by_level),
            tuple(open_measures_by_level),
            tuple(open_centers_by_level),
            tuple(open_endpoints_by_level),
            tuple(tags_by_level),
            tuple(small_by_level),
            evidence,
            canonical_fingerprint(
                {
                    "kind": "variable-patch-embedded-boundary-metrics",
                    "plan": self.plan_id,
                    "geometry_revision": int(np.asarray(geometry.revision)),
                    "evidence": evidence.evidence_id,
                }
            ),
        )


__all__ = [
    "VariablePatchEmbeddedBoundaryEvidence",
    "VariablePatchEmbeddedBoundaryMetrics",
    "VariablePatchEmbeddedBoundaryPlan",
]
