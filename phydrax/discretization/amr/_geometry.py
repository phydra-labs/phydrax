#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Reference-map stationary and fixed-connectivity ALE geometry for patch buckets."""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import determinant_small_linear, SmallLinearSolvePlan
from ._variable import VariablePatchHierarchyTopology


CoordinateMap = Callable[[Array, Array, object], Array]
_INT32_MAX = np.iinfo(np.int32).max


def _validated_revision(value: ArrayLike, /) -> Array:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("Patch geometry revision must be a scalar signed integer.")
    if isinstance(value, (int, np.integer)):
        integer = int(value)
        if integer < 0 or integer > _INT32_MAX:
            raise ValueError(
                "Patch geometry revision must be nonnegative and representable as int32."
            )
    elif isinstance(value, np.ndarray):
        if value.shape != () or value.dtype.kind != "i":
            raise ValueError("Patch geometry revision must be a scalar signed integer.")
        integer = int(value.item())
        if integer < 0 or integer > _INT32_MAX:
            raise ValueError(
                "Patch geometry revision must be nonnegative and representable as int32."
            )
    revision = jnp.asarray(value)
    if revision.shape != () or revision.dtype.kind != "i":
        raise ValueError("Patch geometry revision must be a scalar signed integer.")
    return eqx.error_if(
        revision,
        (revision < 0) | (revision > _INT32_MAX),
        "Patch geometry revision must be nonnegative and representable as int32.",
    ).astype(jnp.int32)


def _corners(vertices: Array, dimension: int, /) -> Array:
    """Return tensor-cell corners ordered x-fastest: 00,10,01,11, ... ."""
    cell_shape = tuple(size - 1 for size in vertices.shape[1:-1])
    order = tuple(
        tuple((index >> axis) & 1 for axis in range(dimension))
        for index in range(2**dimension)
    )
    return jnp.stack(
        tuple(
            vertices[
                (slice(None),)
                + tuple(
                    slice(bit, bit + cell_shape[axis]) for axis, bit in enumerate(bits)
                )
                + (slice(None),)
            ]
            for bits in order
        ),
        axis=-2,
    )


def _volume(
    corners: Array,
    dimension: int,
    determinant_plan: SmallLinearSolvePlan,
    /,
) -> Array:
    if dimension == 1:
        return corners[..., 1, 0] - corners[..., 0, 0]
    if dimension == 2:
        p00, p10, p01, p11 = (corners[..., index, :] for index in range(4))
        return 0.5 * (
            p00[..., 0] * p10[..., 1]
            - p00[..., 1] * p10[..., 0]
            + p10[..., 0] * p11[..., 1]
            - p10[..., 1] * p11[..., 0]
            + p11[..., 0] * p01[..., 1]
            - p11[..., 1] * p01[..., 0]
            + p01[..., 0] * p00[..., 1]
            - p01[..., 1] * p00[..., 0]
        )
    if dimension != 3:
        raise ValueError("Patch geometry supports one, two, or three dimensions.")
    p000, p100, p010, p110, p001, p101, p011, p111 = (
        corners[..., index, :] for index in range(8)
    )

    def tetra(a: Array, b: Array, c: Array, d: Array, /) -> Array:
        return (
            determinant_small_linear(
                determinant_plan,
                jnp.stack((b - a, c - a, d - a), axis=-1),
            )
            / 6.0
        )

    return sum(
        (
            tetra(p000, p100, p010, p001),
            tetra(p100, p110, p010, p111),
            tetra(p100, p010, p001, p111),
            tetra(p100, p001, p101, p111),
            tetra(p010, p001, p111, p011),
        )
    )


def _orientation(
    corners: Array,
    dimension: int,
    determinant_plan: SmallLinearSolvePlan,
    /,
) -> Array:
    if dimension <= 2:
        return _volume(corners, dimension, determinant_plan)
    p000, p100, p010, p110, p001, p101, p011, p111 = (
        corners[..., index, :] for index in range(8)
    )

    def tetra(a: Array, b: Array, c: Array, d: Array, /) -> Array:
        return (
            determinant_small_linear(
                determinant_plan,
                jnp.stack((b - a, c - a, d - a), axis=-1),
            )
            / 6.0
        )

    return jnp.min(
        jnp.stack(
            (
                tetra(p000, p100, p010, p001),
                tetra(p100, p110, p010, p111),
                tetra(p100, p010, p001, p111),
                tetra(p100, p001, p101, p111),
                tetra(p010, p001, p111, p011),
            ),
            axis=-1,
        ),
        axis=-1,
    )


def _swept_volume_rate(corners: Array, velocities: Array, dimension: int, /) -> Array:
    """Independent face-sweep form of the discrete geometric conservation law."""
    if dimension == 1:
        return velocities[..., 1, 0] - velocities[..., 0, 0]
    center = jnp.mean(corners, axis=-2)
    if dimension == 2:
        p00, p10, p01, p11 = (corners[..., index, :] for index in range(4))
        v00, v10, v01, v11 = (velocities[..., index, :] for index in range(4))
        edges = (
            (p00, p10, v00, v10),
            (p10, p11, v10, v11),
            (p11, p01, v11, v01),
            (p01, p00, v01, v00),
        )
        return sum(
            jnp.sum(
                0.5
                * (left_velocity + right_velocity)
                * jnp.stack(
                    (
                        right[..., 1] - left[..., 1],
                        left[..., 0] - right[..., 0],
                    ),
                    axis=-1,
                ),
                axis=-1,
            )
            for left, right, left_velocity, right_velocity in edges
        )
    if dimension != 3:
        raise ValueError("Patch geometry supports one, two, or three dimensions.")
    face_indices = (
        (0, 2, 6, 4),
        (1, 5, 7, 3),
        (0, 4, 5, 1),
        (2, 3, 7, 6),
        (0, 1, 3, 2),
        (4, 6, 7, 5),
    )
    result = jnp.zeros(center.shape[:-1], dtype=corners.dtype)
    for indices in face_indices:
        face = corners[..., indices, :]
        face_velocity = velocities[..., indices, :]
        p00, p10, p11, p01 = (face[..., index, :] for index in range(4))
        area = 0.5 * (jnp.cross(p10 - p00, p11 - p00) + jnp.cross(p11 - p00, p01 - p00))
        face_center = jnp.mean(face, axis=-2)
        area = jnp.where(
            (jnp.sum(area * (face_center - center), axis=-1) < 0.0)[..., None],
            -area,
            area,
        )
        result = result + jnp.sum(jnp.mean(face_velocity, axis=-2) * area, axis=-1)
    return result


class VariablePatchGeometryState(StrictModule):
    """Traceable physical cell metrics over fixed variable-patch connectivity."""

    plan: VariablePatchGeometryPlan
    time: Array
    revision: Array
    vertex_coordinates: tuple[tuple[Array, ...], ...]
    cell_centers: tuple[tuple[Array, ...], ...]
    cell_volumes: tuple[tuple[Array, ...], ...]
    mesh_volume_rates: tuple[tuple[Array, ...], ...]
    orientation_minima: tuple[tuple[Array, ...], ...]
    gcl_defects: tuple[tuple[Array, ...], ...]
    active_cell_masks: tuple[tuple[Array, ...], ...]
    valid: Array


class VariablePatchGeometryPlan(StrictModule, NonTrainableState):
    """Static reference vertices and traceable mapped/ALE metric evaluator.

    ``coordinate_map(reference_point, time, args)`` is the sole physical-map owner
    at this seam.  Inactive bucket-envelope vertices are routed through a safe
    in-domain reference point before invoking the map, so padding cannot poison
    a metric state.
    """

    topology: VariablePatchHierarchyTopology
    coordinate_map: CoordinateMap = eqx.field(static=True)
    coordinate_map_id: str = eqx.field(static=True)
    geometry_family_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    determinant_plan: SmallLinearSolvePlan
    reference_vertices: tuple[tuple[Array, ...], ...]
    active_cell_masks: tuple[tuple[Array, ...], ...]
    active_vertex_masks: tuple[tuple[Array, ...], ...]
    gcl_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        topology: VariablePatchHierarchyTopology,
        coordinate_map: CoordinateMap,
        coordinate_map_id: str,
        /,
        *,
        geometry_family_id: str = "mapped-patch",
        geometry_layout_id: str = "reference-bucket-layout",
        gcl_tolerance: float = 1.0e-10,
    ):
        if not isinstance(topology, VariablePatchHierarchyTopology):
            raise TypeError("Patch geometry requires VariablePatchHierarchyTopology.")
        if not callable(coordinate_map):
            raise TypeError("Patch coordinate_map must be callable.")
        map_id = str(coordinate_map_id)
        family = str(geometry_family_id)
        layout = str(geometry_layout_id)
        tolerance = float(gcl_tolerance)
        dimension = len(topology.plan.grid.shape)
        if (
            not map_id
            or not family
            or not layout
            or not np.isfinite(tolerance)
            or tolerance <= 0.0
        ):
            raise ValueError(
                "Patch geometry identifiers and GCL tolerance must be valid."
            )
        references: list[tuple[Array, ...]] = []
        cell_masks: list[tuple[Array, ...]] = []
        vertex_masks: list[tuple[Array, ...]] = []
        lower_bounds = np.asarray(
            [axis.bounds[0] for axis in topology.plan.grid.structured_axes],
            dtype=np.float64,
        )
        for level_plan, metadata, spacing in zip(
            topology.plan.levels,
            topology.levels,
            topology.plan.level_spacings,
            strict=True,
        ):
            level_references = []
            level_cell_masks = []
            level_vertex_masks = []
            for bucket_index, bucket in enumerate(level_plan.buckets):
                envelope = bucket.signature.envelope_shape
                vertex_shape = tuple(value + 1 for value in envelope)
                vertex_local = np.stack(
                    np.meshgrid(
                        *(np.arange(value, dtype=np.float64) for value in vertex_shape),
                        indexing="ij",
                    ),
                    axis=-1,
                )
                cell_local = np.stack(
                    np.meshgrid(
                        *(np.arange(value, dtype=np.int32) for value in envelope),
                        indexing="ij",
                    ),
                    axis=-1,
                )
                lower = np.asarray(metadata.lower[bucket_index], dtype=np.float64)
                extent = np.asarray(metadata.extent[bucket_index], dtype=np.int32)
                lanes = bucket.lane_capacity
                reference = (
                    lower.reshape((lanes,) + (1,) * dimension + (dimension,))
                    + vertex_local.reshape((1,) + vertex_shape + (dimension,))
                    * np.asarray(spacing, dtype=np.float64).reshape(
                        (1,) + (1,) * dimension + (dimension,)
                    )
                    + lower_bounds.reshape((1,) + (1,) * dimension + (dimension,))
                )
                active_lane = np.asarray(metadata.active[bucket_index], dtype=np.bool_)
                cell_valid = active_lane[(slice(None),) + (None,) * dimension]
                vertex_valid = active_lane[(slice(None),) + (None,) * dimension]
                for axis in range(dimension):
                    cell_valid = cell_valid & (
                        cell_local[..., axis][None]
                        < extent[:, axis][(slice(None),) + (None,) * dimension]
                    )
                    vertex_valid = vertex_valid & (
                        vertex_local[..., axis][None]
                        <= extent[:, axis][(slice(None),) + (None,) * dimension]
                    )
                safe_reference = np.where(
                    vertex_valid[..., None],
                    reference,
                    lower_bounds.reshape((1,) + (1,) * dimension + (dimension,)),
                )
                level_references.append(jnp.asarray(safe_reference))
                level_cell_masks.append(jnp.asarray(cell_valid))
                level_vertex_masks.append(jnp.asarray(vertex_valid))
            references.append(tuple(level_references))
            cell_masks.append(tuple(level_cell_masks))
            vertex_masks.append(tuple(level_vertex_masks))
        determinant = SmallLinearSolvePlan(dimension)
        self.topology = topology
        self.coordinate_map = coordinate_map
        self.coordinate_map_id = map_id
        self.geometry_family_id = family
        self.geometry_layout_id = layout
        self.determinant_plan = determinant
        self.reference_vertices = tuple(references)
        self.active_cell_masks = tuple(cell_masks)
        self.active_vertex_masks = tuple(vertex_masks)
        self.gcl_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "variable-patch-geometry-plan",
                "epoch": topology.epoch.epoch_id,
                "coordinate_map": map_id,
                "geometry_family": family,
                "geometry_layout": layout,
                "determinant": determinant.plan_id,
                "references": [
                    [array_tree_fingerprint(value) for value in level]
                    for level in references
                ],
            }
        )

    def state(
        self,
        time: ArrayLike,
        args: object = None,
        /,
        *,
        revision: ArrayLike = 0,
    ) -> VariablePatchGeometryState:
        time_ = jnp.asarray(time)
        revision_ = _validated_revision(revision)
        if time_.shape != ():
            raise ValueError("Patch geometry time must be scalar.")
        vertices_by_level = []
        centers_by_level = []
        volumes_by_level = []
        rates_by_level = []
        defects_by_level = []
        orientations_by_level = []
        valid_terms = [jnp.isfinite(time_)]
        for level, (references, cell_masks, vertex_masks) in enumerate(
            zip(
                self.reference_vertices,
                self.active_cell_masks,
                self.active_vertex_masks,
                strict=True,
            )
        ):
            dimension = self.topology.plan.levels[level].dimension
            level_vertices = []
            level_centers = []
            level_volumes = []
            level_rates = []
            level_orientations = []
            level_defects = []
            for reference, cell_active, vertex_active in zip(
                references,
                cell_masks,
                vertex_masks,
                strict=True,
            ):
                flat_reference = reference.reshape((-1, dimension))

                def map_points(
                    tau: Array,
                    *,
                    flat_reference: Array = flat_reference,
                    reference: Array = reference,
                ) -> Array:
                    mapped = jax.vmap(
                        lambda point: jnp.asarray(self.coordinate_map(point, tau, args))
                    )(flat_reference)
                    return mapped.reshape(reference.shape)

                vertices = map_points(time_)
                if vertices.shape != reference.shape:
                    raise ValueError(
                        "Patch coordinate_map must preserve ambient dimension."
                    )
                velocity = jax.jvp(
                    map_points,
                    (time_,),
                    (jnp.ones_like(time_),),
                )[1]
                corners = _corners(vertices, dimension)
                velocity_corners = _corners(velocity, dimension)
                volume = _volume(corners, dimension, self.determinant_plan)
                orientation = _orientation(
                    corners,
                    dimension,
                    self.determinant_plan,
                )
                rate = jax.jvp(
                    lambda value, dimension=dimension: _volume(
                        _corners(value, dimension),
                        dimension,
                        self.determinant_plan,
                    ),
                    (vertices,),
                    (velocity,),
                )[1]
                swept = _swept_volume_rate(corners, velocity_corners, dimension)
                defect = jnp.abs(rate - swept)
                scale = jnp.maximum(jnp.maximum(jnp.abs(rate), jnp.abs(swept)), 1.0)
                valid = (
                    jnp.all(
                        ~cell_active
                        | (jnp.isfinite(volume) & (volume > 0.0) & (orientation > 0.0))
                    )
                    & jnp.all(~cell_active | jnp.isfinite(rate))
                    & jnp.all(~cell_active | (defect <= self.gcl_tolerance * scale))
                    & jnp.all(~vertex_active[..., None] | jnp.isfinite(vertices))
                )
                valid_terms.append(valid)
                level_vertices.append(vertices)
                level_centers.append(
                    jnp.where(
                        cell_active[..., None],
                        jnp.mean(corners, axis=-2),
                        0.0,
                    )
                )
                level_volumes.append(jnp.where(cell_active, volume, 0.0))
                level_rates.append(jnp.where(cell_active, rate, 0.0))
                level_orientations.append(jnp.where(cell_active, orientation, 0.0))
                level_defects.append(jnp.where(cell_active, defect, 0.0))
            vertices_by_level.append(tuple(level_vertices))
            centers_by_level.append(tuple(level_centers))
            volumes_by_level.append(tuple(level_volumes))
            rates_by_level.append(tuple(level_rates))
            orientations_by_level.append(tuple(level_orientations))
            defects_by_level.append(tuple(level_defects))
        return VariablePatchGeometryState(
            self,
            time_,
            revision_,
            tuple(vertices_by_level),
            tuple(centers_by_level),
            tuple(volumes_by_level),
            tuple(rates_by_level),
            tuple(orientations_by_level),
            tuple(defects_by_level),
            self.active_cell_masks,
            jnp.all(jnp.stack(tuple(valid_terms))),
        )


__all__ = ["VariablePatchGeometryPlan", "VariablePatchGeometryState"]
