#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._cell_complex import PolygonalConnectivity
from ._unstructured import UnstructuredFiniteVolumeDiscretization


LevelSetField = Callable[[Array, Any], ArrayLike]

_EMBEDDED_FLUID_VERTEX_CAPACITY = 5


def _unique_points(points, tolerance, /):
    unique = []
    for point in points:
        if not any(np.linalg.norm(point - existing) <= tolerance for existing in unique):
            unique.append(point)
    return unique


def _clip_positive_polygon(vertices, values, /):
    output = []
    intersections = []
    for index in range(vertices.shape[0]):
        start = vertices[index]
        stop = vertices[(index + 1) % vertices.shape[0]]
        start_value = float(values[index])
        stop_value = float(values[(index + 1) % vertices.shape[0]])
        start_inside = start_value >= 0.0
        stop_inside = stop_value >= 0.0
        if start_inside:
            output.append(start)
        if start_inside != stop_inside:
            fraction = start_value / (start_value - stop_value)
            point = start + fraction * (stop - start)
            output.append(point)
            intersections.append(point)
    scale = max(np.max(np.linalg.norm(vertices - vertices[0], axis=-1)), 1.0)
    tolerance = 128.0 * np.finfo(float).eps * scale
    output = _unique_points(output, tolerance)
    intersections = _unique_points(intersections, tolerance)
    return np.asarray(output), intersections


def _polygon_measure_centroid(vertices, /):
    if vertices.shape[0] < 3:
        return 0.0, np.zeros((2,))
    following = np.roll(vertices, -1, axis=0)
    cross = vertices[:, 0] * following[:, 1] - following[:, 0] * vertices[:, 1]
    twice_area = np.sum(cross)
    if twice_area <= 0.0:
        raise ValueError("Clipped fluid polygon must retain positive orientation.")
    area = 0.5 * twice_area
    centroid = np.sum((vertices + following) * cross[:, None], axis=0) / (
        3.0 * twice_area
    )
    return area, centroid


class EmbeddedBoundaryStatus(IntEnum):
    """Portable validity status for one stationary embedded-boundary metric set."""

    SUCCESS = 0
    FAILED = 1


class EmbeddedBoundaryEvidence(StrictModule, NonTrainableState):
    """Closure and small-cell evidence bound to prepared cut geometry."""

    volume_closure_defect: Array
    volume_closure_tolerance: Array
    aperture_closure_defect: Array
    aperture_closure_tolerance: Array
    cut_face_closure_defect: Array
    cut_face_closure_tolerance: Array
    fluid_polygon_measure_defect: Array
    fluid_polygon_measure_tolerance: Array
    open_segment_measure_defect: Array
    open_segment_measure_tolerance: Array
    small_cell_count: Array
    minimum_nonzero_volume_fraction: Array
    passed: Array
    status: Array


class EmbeddedBoundaryReport(StrictModule):
    total_fluid_volume: Array
    cut_cell_count: Array
    solid_cell_count: Array
    minimum_nonzero_volume_fraction: Array
    maximum_fluid_closure_residual: Array


class EmbeddedBoundaryStabilizationPolicy(StrictModule, NonTrainableState):
    """Immutable thresholds for a separate conservative redistribution operator."""

    minimum_volume_fraction: float = eqx.field(static=True)
    maximum_recipients: int = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        minimum_volume_fraction: float = 0.1,
        maximum_recipients: int = 4,
        absolute_tolerance: float = 1.0e-12,
        relative_tolerance: float = 1.0e-12,
    ):
        minimum = float(minimum_volume_fraction)
        if not isinstance(maximum_recipients, (int, np.integer)) or isinstance(
            maximum_recipients, (bool, np.bool_)
        ):
            raise ValueError("maximum_recipients must be a positive integer.")
        recipients = int(maximum_recipients)
        absolute = float(absolute_tolerance)
        relative = float(relative_tolerance)
        if not np.isfinite(minimum) or minimum <= 0.0 or minimum > 1.0:
            raise ValueError("minimum_volume_fraction must be finite and in (0, 1].")
        if recipients <= 0:
            raise ValueError("maximum_recipients must be a positive integer.")
        if not np.isfinite(absolute) or absolute < 0.0:
            raise ValueError("absolute_tolerance must be finite and nonnegative.")
        if not np.isfinite(relative) or relative < 0.0:
            raise ValueError("relative_tolerance must be finite and nonnegative.")
        self.minimum_volume_fraction = minimum
        self.maximum_recipients = recipients
        self.absolute_tolerance = absolute
        self.relative_tolerance = relative
        self.policy_id = canonical_fingerprint(
            {
                "kind": "embedded-boundary-stabilization-policy",
                "minimum_volume_fraction": minimum,
                "maximum_recipients": recipients,
                "absolute_tolerance": absolute,
                "relative_tolerance": relative,
            }
        )


class EmbeddedBoundaryMetrics(StrictModule, NonTrainableState):
    volume_fraction: Array
    fluid_cell_volumes: Array
    fluid_cell_centers: Array
    active_fluid_cells: Array
    cut_cells: Array
    fluid_polygon_vertices: Array
    fluid_polygon_valid: Array
    face_open_fraction: Array
    open_face_measures: Array
    open_face_segment_endpoints: Array
    cut_face_centers: Array
    cut_face_normals: Array
    cut_face_measures: Array
    cut_face_active: Array
    body_tags: Array
    safe_inverse_fluid_volume: Array
    vertex_values: Array
    evidence: EmbeddedBoundaryEvidence
    report: EmbeddedBoundaryReport
    prepared_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    field_id: str = eqx.field(static=True)
    body_tag: int = eqx.field(static=True)
    stabilization_policy_id: str = eqx.field(static=True)
    metrics_id: str = eqx.field(static=True)

    def validate_contact_angle_bindings(
        self,
        contact_angles: Any,
        plic_id: str,
        /,
    ) -> None:
        """Validate explicit contact-angle ownership against prepared geometry."""

        from ._contact_angle import EmbeddedBoundaryContactAngleSet

        if not isinstance(contact_angles, EmbeddedBoundaryContactAngleSet):
            raise TypeError("contact_angles must be an EmbeddedBoundaryContactAngleSet.")
        contact_angles.validate_body_tags(np.asarray(self.body_tags))
        if self.geometry_id != contact_angles.geometry_id:
            raise ValueError(
                "Contact-angle policies belong to stale or different embedded geometry."
            )
        contact_angles.validate_bindings(self.geometry_id, str(plic_id))


class EmbeddedBoundaryPlan(StrictModule, NonTrainableState):
    """Exact linear-edge clipping of a certified 2-D polygonal level set."""

    discretization: UnstructuredFiniteVolumeDiscretization
    level_set: LevelSetField = eqx.field(static=True)
    field_id: str = eqx.field(static=True)
    body_tag: int = eqx.field(static=True)
    stabilization_policy: EmbeddedBoundaryStabilizationPolicy
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: UnstructuredFiniteVolumeDiscretization,
        level_set: LevelSetField,
        /,
        *,
        field_id: str,
        body_tag: int = 0,
        stabilization_policy: EmbeddedBoundaryStabilizationPolicy | None = None,
    ):
        if not isinstance(discretization, UnstructuredFiniteVolumeDiscretization):
            raise TypeError("Embedded boundaries require unstructured FV geometry.")
        if discretization.cell_dimension != 2 or not isinstance(
            discretization.connectivity, PolygonalConnectivity
        ):
            raise ValueError(
                "Embedded-boundary clipping currently supports 2-D polygons."
            )
        if not callable(level_set):
            raise TypeError("level_set must be callable.")
        identifier = str(field_id)
        if not identifier:
            raise ValueError("field_id must be non-empty.")
        tag = int(body_tag)
        if tag < 0:
            raise ValueError("body_tag must be nonnegative.")
        if stabilization_policy is None:
            policy = EmbeddedBoundaryStabilizationPolicy()
        elif isinstance(stabilization_policy, EmbeddedBoundaryStabilizationPolicy):
            policy = stabilization_policy
        else:
            raise TypeError(
                "stabilization_policy must be an EmbeddedBoundaryStabilizationPolicy."
            )
        self.discretization = discretization
        self.level_set = level_set
        self.field_id = identifier
        self.body_tag = tag
        self.stabilization_policy = policy
        self.plan_id = canonical_fingerprint(
            {
                "kind": "embedded-boundary-plan",
                "prepared": discretization.prepared_id,
                "topology": discretization.topology_id,
                "geometry": discretization.geometry_id,
                "field": identifier,
                "body_tag": tag,
                "stabilization_policy": policy.policy_id,
            }
        )

    def prepare(self, args: Any = None, /) -> EmbeddedBoundaryMetrics:
        geometry = self.discretization
        connectivity = geometry.connectivity
        if not isinstance(connectivity, PolygonalConnectivity):
            raise TypeError("Embedded-boundary connectivity must be polygonal.")
        vertices = np.asarray(geometry.vertices)
        level_values = np.asarray(
            self.level_set(jnp.asarray(vertices), args), dtype=float
        )
        if level_values.shape != (vertices.shape[0],) or np.any(
            ~np.isfinite(level_values)
        ):
            raise ValueError(
                "Embedded level_set must return one finite value per vertex."
            )

        cell_vertices = np.asarray(connectivity.cell_vertices, dtype=np.int32)
        cell_kinds = np.asarray(connectivity.cell_kinds, dtype=np.int32)
        cell_count = geometry.cell_count
        cell_volumes = np.asarray(geometry.cell_volumes, dtype=float)
        fluid_volumes = np.zeros((cell_count,))
        solid_volumes = np.zeros((cell_count,))
        fluid_centers = np.zeros((cell_count, 2))
        cut_centers = np.zeros((cell_count, 2))
        cut_normals = np.zeros((cell_count, 2))
        cut_measures = np.zeros((cell_count,))
        cut_active = np.zeros((cell_count,), dtype=bool)
        fluid_polygon_vertices = np.zeros(
            (cell_count, _EMBEDDED_FLUID_VERTEX_CAPACITY, 2)
        )
        fluid_polygon_valid = np.zeros(
            (cell_count, _EMBEDDED_FLUID_VERTEX_CAPACITY), dtype=bool
        )
        for cell in range(cell_count):
            arity = int(cell_kinds[cell])
            indices = cell_vertices[cell, :arity]
            polygon = vertices[indices]
            values = level_values[indices]
            if np.all(values == 0.0):
                raise ValueError(
                    f"Embedded boundary in cell {cell} is identically zero and "
                    "has ambiguous crossings."
                )
            clipped, intersections = _clip_positive_polygon(polygon, values)
            if len(intersections) not in (0, 2):
                raise ValueError(
                    f"Embedded boundary in cell {cell} has ambiguous edge crossings."
                )
            clipped_count = clipped.shape[0]
            if clipped_count > _EMBEDDED_FLUID_VERTEX_CAPACITY:
                raise ValueError(
                    f"Embedded fluid polygon in cell {cell} exceeds fixed capacity."
                )
            if clipped_count:
                fluid_polygon_vertices[cell, :clipped_count] = clipped
                fluid_polygon_valid[cell, :clipped_count] = True
            fluid_volume, fluid_center = _polygon_measure_centroid(clipped)
            solid_polygon, _ = _clip_positive_polygon(polygon, -values)
            solid_volume, _ = _polygon_measure_centroid(solid_polygon)
            if np.all(values >= 0.0):
                fluid_volume = cell_volumes[cell]
                solid_volume = 0.0
            elif np.all(values < 0.0):
                fluid_volume = 0.0
                solid_volume = cell_volumes[cell]
            fluid_volumes[cell] = fluid_volume
            solid_volumes[cell] = solid_volume
            fluid_centers[cell] = fluid_center
            if len(intersections) == 2 and 0.0 < fluid_volume < cell_volumes[cell]:
                first, second = intersections
                midpoint = 0.5 * (first + second)
                tangent = second - first
                measure = np.linalg.norm(tangent)
                if not np.isfinite(measure) or measure <= 0.0:
                    raise ValueError(
                        f"Embedded cut face in cell {cell} has invalid measure."
                    )
                normal = np.asarray((tangent[1], -tangent[0])) / measure
                if np.dot(normal, midpoint - fluid_center) < 0.0:
                    normal = -normal
                normal_norm = np.linalg.norm(normal)
                outward_projection = np.dot(normal, midpoint - fluid_center)
                if (
                    not np.isfinite(normal_norm)
                    or abs(normal_norm - 1.0) > 256.0 * np.finfo(float).eps
                    or outward_projection <= 0.0
                ):
                    raise ValueError(
                        f"Embedded cut face in cell {cell} must have a unit "
                        "outward normal."
                    )
                cut_centers[cell] = midpoint
                cut_normals[cell] = normal
                cut_measures[cell] = measure
                cut_active[cell] = True

        fraction = fluid_volumes / cell_volumes
        if (
            np.any(~np.isfinite(fraction))
            or np.any(fraction < 0.0)
            or np.any(fraction > 1.0)
        ):
            raise ValueError("Embedded fluid volume fractions lie outside [0, 1].")
        active = fraction > 0.0
        if np.any(fluid_volumes[~active] != 0.0) or np.any(fluid_volumes[active] <= 0.0):
            raise ValueError(
                "Inactive fluid volumes must be zero and active volumes positive."
            )

        edges = np.asarray(connectivity.edges, dtype=np.int32)
        edge_values = level_values[edges]
        edge_points = vertices[edges]
        open_face_segment_endpoints = np.zeros((edges.shape[0], 2, 2))
        open_fraction = np.empty((edges.shape[0],))
        solid_fraction = np.empty((edges.shape[0],))
        for face, (start_value, stop_value) in enumerate(edge_values):
            if start_value == 0.0 and stop_value == 0.0:
                raise ValueError(
                    f"Embedded boundary coincides with face {face}; crossings are "
                    "ambiguous."
                )
            if start_value >= 0.0 and stop_value >= 0.0:
                open_fraction[face] = 1.0
            elif start_value < 0.0 and stop_value < 0.0:
                open_fraction[face] = 0.0
            else:
                crossing = start_value / (start_value - stop_value)
                open_fraction[face] = crossing if start_value >= 0.0 else 1.0 - crossing
            if start_value <= 0.0 and stop_value <= 0.0:
                solid_fraction[face] = 1.0
            elif start_value > 0.0 and stop_value > 0.0:
                solid_fraction[face] = 0.0
            else:
                crossing = start_value / (start_value - stop_value)
                solid_fraction[face] = crossing if start_value <= 0.0 else 1.0 - crossing
            if open_fraction[face] > 0.0:
                first_point, second_point = edge_points[face]
                if start_value >= 0.0 and stop_value >= 0.0:
                    open_face_segment_endpoints[face] = edge_points[face]
                else:
                    crossing = start_value / (start_value - stop_value)
                    crossing_point = first_point + crossing * (second_point - first_point)
                    if start_value >= 0.0:
                        open_face_segment_endpoints[face, 0] = first_point
                        open_face_segment_endpoints[face, 1] = crossing_point
                    else:
                        open_face_segment_endpoints[face, 0] = crossing_point
                        open_face_segment_endpoints[face, 1] = second_point
        if (
            np.any(~np.isfinite(open_fraction))
            or np.any(open_fraction < 0.0)
            or np.any(open_fraction > 1.0)
        ):
            raise ValueError("Embedded face-open fractions lie outside [0, 1].")
        face_measures = np.asarray(geometry.face_measures, dtype=float)
        open_measures = open_fraction * face_measures
        if np.any(~np.isfinite(open_measures)) or not np.array_equal(
            open_measures, open_fraction * face_measures
        ):
            raise ValueError(
                "Embedded open-face measures must equal open fractions times "
                "base-face measures."
            )

        tangent = edge_points[:, 1] - edge_points[:, 0]
        canonical_area = np.stack((tangent[:, 1], -tangent[:, 0]), axis=-1)
        cell_edges = np.asarray(connectivity.cell_edges, dtype=np.int32)
        cell_signs = np.asarray(connectivity.cell_edge_signs)
        cell_valid = np.asarray(connectivity.cell_edge_valid, dtype=bool)

        policy = self.stabilization_policy
        body_tags = np.full((cell_count,), self.body_tag, dtype=np.int32)
        if body_tags.shape != (cell_count,) or np.any(body_tags != self.body_tag):
            raise ValueError("Embedded boundary body tags must cover every cell.")

        target_dtype = jnp.asarray(fraction).dtype
        numpy_target_dtype = np.dtype(target_dtype)
        volume_fraction_array = jnp.asarray(fraction, dtype=target_dtype)
        fluid_volume_array = jnp.asarray(fluid_volumes, dtype=target_dtype)
        fluid_center_array = jnp.asarray(fluid_centers, dtype=target_dtype)
        active_array = jnp.asarray(active)
        cut_array = jnp.asarray(cut_active)
        fluid_polygon_array = jnp.asarray(fluid_polygon_vertices, dtype=target_dtype)
        fluid_polygon_valid_array = jnp.asarray(fluid_polygon_valid)
        open_fraction_array = jnp.asarray(open_fraction, dtype=target_dtype)
        face_measure_array = jnp.asarray(face_measures, dtype=target_dtype)
        open_measure_array = open_fraction_array * face_measure_array
        open_face_segment_array = jnp.asarray(
            open_face_segment_endpoints, dtype=target_dtype
        )
        cut_center_array = jnp.asarray(cut_centers, dtype=target_dtype)
        cut_normal_array = jnp.asarray(cut_normals, dtype=target_dtype)
        cut_measure_array = jnp.asarray(cut_measures, dtype=target_dtype)
        body_tag_array = jnp.asarray(body_tags)
        vertex_value_array = jnp.asarray(level_values, dtype=target_dtype)

        target_fraction = np.asarray(volume_fraction_array)
        target_fluid_volumes = np.asarray(fluid_volume_array)
        target_fluid_centers = np.asarray(fluid_center_array)
        target_active = np.asarray(active_array)
        target_cut_active = np.asarray(cut_array)
        target_fluid_polygon_vertices = np.asarray(fluid_polygon_array)
        target_fluid_polygon_valid = np.asarray(fluid_polygon_valid_array, dtype=bool)
        target_open_fraction = np.asarray(open_fraction_array)
        target_open_measures = np.asarray(open_measure_array)
        target_open_face_segments = np.asarray(open_face_segment_array)
        target_cut_centers = np.asarray(cut_center_array)
        target_cut_normals = np.asarray(cut_normal_array)
        target_cut_measures = np.asarray(cut_measure_array)
        target_vertex_values = np.asarray(vertex_value_array)
        if (
            np.any(~np.isfinite(target_fraction))
            or np.any(target_fraction < 0.0)
            or np.any(target_fraction > 1.0)
            or np.any(~np.isfinite(target_fluid_volumes))
            or np.any(~np.isfinite(target_fluid_centers))
            or np.any(target_fraction[target_active] <= 0.0)
            or np.any(target_fluid_volumes[target_active] <= 0.0)
            or np.any(target_fraction[~target_active] != 0.0)
            or np.any(target_fluid_volumes[~target_active] != 0.0)
            or np.any(target_fluid_centers[~target_active] != 0.0)
        ):
            raise ValueError(
                "Embedded active fluid volumes, fractions, and centroids must remain "
                "finite and positive where applicable, with exact inactive zeros, "
                "in the target dtype."
            )
        polygon_counts: np.ndarray = np.asarray(
            np.sum(target_fluid_polygon_valid, axis=1, dtype=np.int32),
            dtype=np.int32,
        )
        compact_polygon_valid = target_fluid_polygon_valid == (
            np.arange(_EMBEDDED_FLUID_VERTEX_CAPACITY)[None, :] < polygon_counts[:, None]
        )
        if (
            target_fluid_polygon_vertices.shape
            != (cell_count, _EMBEDDED_FLUID_VERTEX_CAPACITY, 2)
            or target_fluid_polygon_valid.shape
            != (cell_count, _EMBEDDED_FLUID_VERTEX_CAPACITY)
            or np.any(~np.isfinite(target_fluid_polygon_vertices))
            or np.any(~compact_polygon_valid)
            or np.any(
                (polygon_counts[target_active] < 3)
                | (polygon_counts[target_active] > _EMBEDDED_FLUID_VERTEX_CAPACITY)
            )
            or np.any(polygon_counts[~target_active] != 0)
            or np.any(target_fluid_polygon_vertices[~target_fluid_polygon_valid] != 0.0)
        ):
            raise ValueError(
                "Embedded fluid polygons must remain finite compact fixed-capacity "
                "geometry in the target dtype, with exact inactive and padding zeros."
            )
        if (
            np.any(~np.isfinite(target_open_fraction))
            or np.any(target_open_fraction < 0.0)
            or np.any(target_open_fraction > 1.0)
            or np.any(~np.isfinite(target_open_measures))
            or np.any(target_open_measures < 0.0)
            or np.any(target_open_fraction[open_fraction > 0.0] <= 0.0)
            or np.any(target_open_measures[open_fraction > 0.0] <= 0.0)
            or np.any(target_open_measures[open_fraction == 0.0] != 0.0)
        ):
            raise ValueError(
                "Embedded face fractions and open measures must remain finite and "
                "positive when open, with exact closed-face zeros, in the target "
                "dtype."
            )
        target_open_segment_measures = np.linalg.norm(
            target_open_face_segments[:, 1] - target_open_face_segments[:, 0],
            axis=-1,
        )
        target_open = target_open_measures > 0.0
        if (
            target_open_face_segments.shape != (edges.shape[0], 2, 2)
            or np.any(~np.isfinite(target_open_face_segments))
            or np.any(target_open_segment_measures[target_open] <= 0.0)
            or np.any(target_open_face_segments[~target_open] != 0.0)
        ):
            raise ValueError(
                "Embedded open-face segments must be finite and nondegenerate when "
                "open, with exact closed-face zeros."
            )
        normal_norm = np.linalg.norm(target_cut_normals[target_cut_active], axis=-1)
        normal_tolerance = 256.0 * np.finfo(numpy_target_dtype).eps
        if (
            np.any(~np.isfinite(target_cut_centers))
            or np.any(~np.isfinite(target_cut_normals))
            or np.any(~np.isfinite(target_cut_measures))
            or np.any(target_cut_measures[target_cut_active] <= 0.0)
            or np.any(np.abs(normal_norm - 1.0) > normal_tolerance)
            or np.any(target_cut_centers[~target_cut_active] != 0.0)
            or np.any(target_cut_normals[~target_cut_active] != 0.0)
            or np.any(target_cut_measures[~target_cut_active] != 0.0)
            or np.any(~np.isfinite(target_vertex_values))
        ):
            raise ValueError(
                "Embedded cut centers, unit normals, and measures must remain finite "
                "and nondegenerate, with exact inactive zeros, in the target dtype."
            )

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            target_safe_inverse = np.zeros_like(target_fluid_volumes)
            target_safe_inverse[target_active] = 1.0 / target_fluid_volumes[target_active]
        safe_inverse_array = jnp.asarray(target_safe_inverse, dtype=target_dtype)
        target_safe_inverse = np.asarray(safe_inverse_array)
        if np.any(target_safe_inverse[~target_active] != 0.0) or np.any(
            ~np.isfinite(target_safe_inverse[target_active])
            | (target_safe_inverse[target_active] <= 0.0)
        ):
            raise ValueError(
                "Inactive inverse fluid volumes must remain zero and active inverses "
                "finite and positive in the target dtype."
            )

        target_cell_volumes = np.asarray(jnp.asarray(cell_volumes, dtype=target_dtype))
        target_solid_volumes = np.asarray(jnp.asarray(solid_volumes, dtype=target_dtype))
        target_solid_fraction = np.asarray(
            jnp.asarray(solid_fraction, dtype=target_dtype)
        )
        target_face_measures = np.asarray(face_measure_array)
        target_canonical_area = np.asarray(
            jnp.asarray(canonical_area, dtype=target_dtype)
        )
        target_fluid_polygon_measures = np.zeros((cell_count,), dtype=numpy_target_dtype)
        for cell in range(cell_count):
            count = int(polygon_counts[cell])
            target_fluid_polygon_measures[cell] = _polygon_measure_centroid(
                target_fluid_polygon_vertices[cell, :count]
            )[0]
        target_closure = np.zeros((cell_count, 2), dtype=numpy_target_dtype)
        target_closure_scale = np.zeros((cell_count,), dtype=numpy_target_dtype)
        for cell in range(cell_count):
            valid = cell_valid[cell]
            local_edges = cell_edges[cell, valid]
            target_closure[cell] = np.sum(
                np.asarray(cell_signs[cell, valid, None], dtype=numpy_target_dtype)
                * target_open_fraction[local_edges, None]
                * target_canonical_area[local_edges],
                axis=0,
                dtype=numpy_target_dtype,
            )
            target_closure[cell] += target_cut_normals[cell] * target_cut_measures[cell]
            target_closure_scale[cell] = (
                np.sum(
                    target_open_measures[local_edges],
                    dtype=numpy_target_dtype,
                )
                + target_cut_measures[cell]
            )

        target_absolute_tolerance = np.asarray(
            policy.absolute_tolerance, dtype=numpy_target_dtype
        )
        target_relative_tolerance = np.asarray(
            policy.relative_tolerance, dtype=numpy_target_dtype
        )
        machine_epsilon = np.asarray(
            np.finfo(numpy_target_dtype).eps, dtype=numpy_target_dtype
        )
        volume_operation_count = np.asarray(32 * cell_kinds + 8, dtype=numpy_target_dtype)
        aperture_operation_count = np.asarray(16.0, dtype=numpy_target_dtype)
        cut_operation_count = np.asarray(
            32 * (np.sum(cell_valid, axis=1, dtype=np.int32) + 1),
            dtype=numpy_target_dtype,
        )

        volume_defect = np.abs(
            target_fluid_volumes + target_solid_volumes - target_cell_volumes
        )
        requested_volume_tolerance = (
            target_absolute_tolerance + target_relative_tolerance * target_cell_volumes
        )
        volume_roundoff_floor = (
            machine_epsilon
            * volume_operation_count
            * (
                np.abs(target_fluid_volumes)
                + np.abs(target_solid_volumes)
                + np.abs(target_cell_volumes)
            )
        )
        volume_tolerance = np.maximum(requested_volume_tolerance, volume_roundoff_floor)

        aperture_defect = np.abs(
            (target_open_fraction + target_solid_fraction - 1.0) * target_face_measures
        )
        requested_aperture_tolerance = (
            target_absolute_tolerance + target_relative_tolerance * target_face_measures
        )
        aperture_roundoff_floor = (
            machine_epsilon
            * aperture_operation_count
            * (
                np.abs(target_open_fraction)
                + np.abs(target_solid_fraction)
                + np.asarray(1.0, dtype=numpy_target_dtype)
            )
            * target_face_measures
        )
        aperture_tolerance = np.maximum(
            requested_aperture_tolerance, aperture_roundoff_floor
        )
        fluid_polygon_defect = np.abs(
            target_fluid_polygon_measures - target_fluid_volumes
        )
        fluid_polygon_tolerance = volume_tolerance
        open_segment_defect = np.abs(target_open_segment_measures - target_open_measures)
        open_segment_tolerance = aperture_tolerance

        cut_closure_defect = np.linalg.norm(target_closure, axis=-1)
        requested_cut_closure_tolerance = (
            target_absolute_tolerance + target_relative_tolerance * target_closure_scale
        )
        cut_closure_roundoff_floor = (
            machine_epsilon * cut_operation_count * target_closure_scale
        )
        cut_closure_tolerance = np.maximum(
            requested_cut_closure_tolerance, cut_closure_roundoff_floor
        )
        realized_tolerance_policy_id = canonical_fingerprint(
            {
                "kind": "embedded-boundary-realized-tolerance-policy",
                "schema_version": 2,
                "stabilization_policy": policy.policy_id,
                "metric_dtype": numpy_target_dtype.str,
                "machine_epsilon": float(machine_epsilon),
                "volume_operation_multiplier": 32,
                "volume_fixed_operation_count": 8,
                "aperture_operation_count": 16,
                "cut_closure_operation_multiplier": 32,
                "fluid_polygon_operation_multiplier": 32,
                "open_segment_operation_count": 16,
            }
        )
        evidence_values = (
            volume_defect,
            volume_tolerance,
            aperture_defect,
            aperture_tolerance,
            cut_closure_defect,
            cut_closure_tolerance,
            fluid_polygon_defect,
            fluid_polygon_tolerance,
            open_segment_defect,
            open_segment_tolerance,
        )
        if any(np.any(~np.isfinite(value)) for value in evidence_values):
            raise ValueError(
                "Embedded closure evidence must remain finite in the target dtype."
            )

        target_small_cells = target_active & (
            target_fraction
            < np.asarray(policy.minimum_volume_fraction, dtype=numpy_target_dtype)
        )
        small_cell_count = np.asarray(np.sum(target_small_cells), dtype=np.int32)
        target_nonzero = target_fraction[target_active]
        minimum_fraction = (
            np.min(target_nonzero)
            if target_nonzero.size
            else np.asarray(0.0, dtype=numpy_target_dtype)
        )
        total_fluid_volume = jnp.sum(fluid_volume_array)
        if not np.isfinite(np.asarray(total_fluid_volume)):
            raise ValueError(
                "Embedded total fluid volume must remain finite in the target dtype."
            )
        passed = np.asarray(
            np.all(volume_defect <= volume_tolerance)
            and np.all(aperture_defect <= aperture_tolerance)
            and np.all(cut_closure_defect <= cut_closure_tolerance)
            and np.all(fluid_polygon_defect <= fluid_polygon_tolerance)
            and np.all(open_segment_defect <= open_segment_tolerance),
            dtype=bool,
        )
        status = np.asarray(
            int(
                EmbeddedBoundaryStatus.SUCCESS
                if bool(passed)
                else EmbeddedBoundaryStatus.FAILED
            ),
            dtype=np.int32,
        )
        evidence = EmbeddedBoundaryEvidence(
            volume_closure_defect=jnp.asarray(volume_defect, dtype=target_dtype),
            volume_closure_tolerance=jnp.asarray(volume_tolerance, dtype=target_dtype),
            aperture_closure_defect=jnp.asarray(aperture_defect, dtype=target_dtype),
            aperture_closure_tolerance=jnp.asarray(
                aperture_tolerance, dtype=target_dtype
            ),
            cut_face_closure_defect=jnp.asarray(cut_closure_defect, dtype=target_dtype),
            cut_face_closure_tolerance=jnp.asarray(
                cut_closure_tolerance, dtype=target_dtype
            ),
            fluid_polygon_measure_defect=jnp.asarray(
                fluid_polygon_defect, dtype=target_dtype
            ),
            fluid_polygon_measure_tolerance=jnp.asarray(
                fluid_polygon_tolerance, dtype=target_dtype
            ),
            open_segment_measure_defect=jnp.asarray(
                open_segment_defect, dtype=target_dtype
            ),
            open_segment_measure_tolerance=jnp.asarray(
                open_segment_tolerance, dtype=target_dtype
            ),
            small_cell_count=jnp.asarray(small_cell_count),
            minimum_nonzero_volume_fraction=jnp.asarray(
                minimum_fraction, dtype=target_dtype
            ),
            passed=jnp.asarray(passed),
            status=jnp.asarray(status),
        )
        report = EmbeddedBoundaryReport(
            total_fluid_volume=total_fluid_volume,
            cut_cell_count=jnp.sum(cut_array, dtype=jnp.int32),
            solid_cell_count=jnp.sum(~active_array, dtype=jnp.int32),
            minimum_nonzero_volume_fraction=evidence.minimum_nonzero_volume_fraction,
            maximum_fluid_closure_residual=jnp.max(evidence.cut_face_closure_defect),
        )
        metrics_id = canonical_fingerprint(
            {
                "kind": "embedded-boundary-metrics",
                "prepared": geometry.prepared_id,
                "topology": geometry.topology_id,
                "geometry": geometry.geometry_id,
                "field": self.field_id,
                "body_tag": self.body_tag,
                "stabilization_policy": policy.policy_id,
                "metric_dtype": numpy_target_dtype.str,
                "realized_tolerance_policy": realized_tolerance_policy_id,
                "vertex_values": array_tree_fingerprint(vertex_value_array),
                "metric_arrays": array_tree_fingerprint(
                    {
                        "volume_fraction": volume_fraction_array,
                        "fluid_cell_volumes": fluid_volume_array,
                        "fluid_cell_centers": fluid_center_array,
                        "active_fluid_cells": active_array,
                        "cut_cells": cut_array,
                        "fluid_polygon_vertices": fluid_polygon_array,
                        "fluid_polygon_valid": fluid_polygon_valid_array,
                        "face_open_fraction": open_fraction_array,
                        "open_face_measures": open_measure_array,
                        "open_face_segment_endpoints": open_face_segment_array,
                        "cut_face_centers": cut_center_array,
                        "cut_face_normals": cut_normal_array,
                        "cut_face_measures": cut_measure_array,
                        "cut_face_active": cut_array,
                        "body_tags": body_tag_array,
                        "safe_inverse_fluid_volume": safe_inverse_array,
                    }
                ),
                "evidence": array_tree_fingerprint(evidence),
            }
        )
        return EmbeddedBoundaryMetrics(
            volume_fraction=volume_fraction_array,
            fluid_cell_volumes=fluid_volume_array,
            fluid_cell_centers=fluid_center_array,
            active_fluid_cells=active_array,
            cut_cells=cut_array,
            fluid_polygon_vertices=fluid_polygon_array,
            fluid_polygon_valid=fluid_polygon_valid_array,
            face_open_fraction=open_fraction_array,
            open_face_measures=open_measure_array,
            open_face_segment_endpoints=open_face_segment_array,
            cut_face_centers=cut_center_array,
            cut_face_normals=cut_normal_array,
            cut_face_measures=cut_measure_array,
            cut_face_active=cut_array,
            body_tags=body_tag_array,
            safe_inverse_fluid_volume=safe_inverse_array,
            vertex_values=vertex_value_array,
            evidence=evidence,
            report=report,
            prepared_id=geometry.prepared_id,
            topology_id=geometry.topology_id,
            geometry_id=geometry.geometry_id,
            field_id=self.field_id,
            body_tag=self.body_tag,
            stabilization_policy_id=policy.policy_id,
            metrics_id=metrics_id,
        )


__all__ = [
    "EmbeddedBoundaryEvidence",
    "EmbeddedBoundaryMetrics",
    "EmbeddedBoundaryPlan",
    "EmbeddedBoundaryReport",
    "EmbeddedBoundaryStabilizationPolicy",
    "EmbeddedBoundaryStatus",
]
