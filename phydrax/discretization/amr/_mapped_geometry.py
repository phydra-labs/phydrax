#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""High-order mapped patch metrics and nonconforming physical mortars."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import product
from math import prod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._canonical import canonicalize_patch_hierarchy, CanonicalPatchHierarchy
from ._core import BlockHierarchyTopology
from ._geometry import _validated_revision
from ._variable import VariablePatchHierarchyTopology


CoordinateMap = Callable[[Array, Array, Any], ArrayLike]
ReferenceTraceMap = Callable[[Array, Array, Any], ArrayLike]
SurfaceMap = Callable[[Array, Array, Any], ArrayLike]


class PatchCoordinateMapSet(StrictModule, NonTrainableState):
    """Stable default map plus explicit per-logical-patch overrides."""

    default_map: CoordinateMap = eqx.field(static=True)
    default_map_id: str = eqx.field(static=True)
    patch_ids: tuple[str, ...] = eqx.field(static=True)
    patch_maps: tuple[CoordinateMap, ...] = eqx.field(static=True)
    patch_map_ids: tuple[str, ...] = eqx.field(static=True)
    map_set_id: str = eqx.field(static=True)

    def __init__(
        self,
        default_map: CoordinateMap,
        default_map_id: str,
        /,
        *,
        patch_maps: Mapping[str, tuple[CoordinateMap, str]] | None = None,
    ):
        default_id = str(default_map_id)
        if not callable(default_map) or not default_id:
            raise ValueError(
                "Patch map sets require a callable default map and stable ID."
            )
        entries = () if patch_maps is None else tuple(sorted(patch_maps.items()))
        patch_ids: list[str] = []
        maps: list[CoordinateMap] = []
        map_ids: list[str] = []
        for patch_id, value in entries:
            map_, map_id = value
            patch = str(patch_id)
            identifier = str(map_id)
            if not patch or not callable(map_) or not identifier:
                raise ValueError("Patch map overrides require patch and map identifiers.")
            patch_ids.append(patch)
            maps.append(map_)
            map_ids.append(identifier)
        self.default_map = default_map
        self.default_map_id = default_id
        self.patch_ids = tuple(patch_ids)
        self.patch_maps = tuple(maps)
        self.patch_map_ids = tuple(map_ids)
        self.map_set_id = canonical_fingerprint(
            {
                "kind": "patch-coordinate-map-set",
                "default": default_id,
                "patches": list(zip(patch_ids, map_ids, strict=True)),
            }
        )

    def map_for(self, patch_id: str, /) -> CoordinateMap:
        patch = str(patch_id)
        if patch in self.patch_ids:
            return self.patch_maps[self.patch_ids.index(patch)]
        return self.default_map

    def map_id_for(self, patch_id: str, /) -> str:
        patch = str(patch_id)
        if patch in self.patch_ids:
            return self.patch_map_ids[self.patch_ids.index(patch)]
        return self.default_map_id

    def evaluate(
        self,
        patch_id: str,
        reference_point: Array,
        time: Array,
        args: Any,
        /,
    ) -> Array:
        return jnp.asarray(self.map_for(patch_id)(reference_point, time, args))


class MappedMortarEvidence(StrictModule, NonTrainableState):
    """Physical trace agreement and finite measure for one mortar."""

    maximum_owner_mismatch: Array
    maximum_neighbor_mismatch: Array
    minimum_measure: Array
    tolerance: float = eqx.field(static=True)
    valid: Array
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        maximum_owner_mismatch: ArrayLike,
        maximum_neighbor_mismatch: ArrayLike,
        minimum_measure: ArrayLike,
        tolerance: float,
        /,
    ):
        tolerance_ = float(tolerance)
        owner = jnp.asarray(maximum_owner_mismatch)
        neighbor = jnp.asarray(maximum_neighbor_mismatch)
        measure = jnp.asarray(minimum_measure)
        if owner.shape != () or neighbor.shape != () or measure.shape != ():
            raise ValueError("Mapped mortar evidence values must be scalar.")
        if not np.isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("Mapped mortar tolerance must be positive and finite.")
        valid = (
            jnp.isfinite(owner)
            & jnp.isfinite(neighbor)
            & jnp.isfinite(measure)
            & (owner <= tolerance_)
            & (neighbor <= tolerance_)
            & (measure > 0.0)
        )
        self.maximum_owner_mismatch = owner
        self.maximum_neighbor_mismatch = neighbor
        self.minimum_measure = measure
        self.tolerance = tolerance_
        self.valid = valid
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "mapped-mortar-evidence",
                "tolerance": tolerance_,
            }
        )


class MappedMortarGeometry(StrictModule):
    """Common physical quadrature for one nonconforming patch interface."""

    quadrature_points: Array
    quadrature_weights: Array
    weighted_area_vectors: Array
    owner_reference_points: Array
    neighbor_reference_points: Array
    evidence: MappedMortarEvidence
    mortar_id: str = eqx.field(static=True)
    owner_patch_id: str = eqx.field(static=True)
    neighbor_patch_id: str = eqx.field(static=True)


class MappedMortarPlan(StrictModule, NonTrainableState):
    """Explicit common-surface contract for two nonconforming patch charts."""

    owner_patch_id: str = eqx.field(static=True)
    neighbor_patch_id: str = eqx.field(static=True)
    owner_reference_map: ReferenceTraceMap = eqx.field(static=True)
    neighbor_reference_map: ReferenceTraceMap = eqx.field(static=True)
    surface_map: SurfaceMap = eqx.field(static=True)
    parameter_bounds: tuple[tuple[float, float], ...] = eqx.field(static=True)
    quadrature_order: int = eqx.field(static=True)
    orientation: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    mortar_id: str = eqx.field(static=True)

    def __init__(
        self,
        owner_patch_id: str,
        neighbor_patch_id: str,
        owner_reference_map: ReferenceTraceMap,
        neighbor_reference_map: ReferenceTraceMap,
        surface_map: SurfaceMap,
        parameter_bounds: Sequence[Sequence[float]],
        /,
        *,
        quadrature_order: int = 3,
        orientation: int = 1,
        tolerance: float = 1.0e-10,
    ):
        owner = str(owner_patch_id)
        neighbor = str(neighbor_patch_id)
        bounds = tuple(tuple(float(value) for value in pair) for pair in parameter_bounds)
        order = int(quadrature_order)
        orientation_ = int(orientation)
        tolerance_ = float(tolerance)
        if (
            not owner
            or not neighbor
            or owner == neighbor
            or not callable(owner_reference_map)
            or not callable(neighbor_reference_map)
            or not callable(surface_map)
            or not bounds
            or any(
                len(pair) != 2
                or not np.isfinite(pair[0])
                or not np.isfinite(pair[1])
                or pair[1] <= pair[0]
                for pair in bounds
            )
            or order <= 0
            or orientation_ not in (-1, 1)
            or not np.isfinite(tolerance_)
            or tolerance_ <= 0.0
        ):
            raise ValueError(
                "Mapped mortar chart, bounds, orientation, or tolerance is invalid."
            )
        self.owner_patch_id = owner
        self.neighbor_patch_id = neighbor
        self.owner_reference_map = owner_reference_map
        self.neighbor_reference_map = neighbor_reference_map
        self.surface_map = surface_map
        self.parameter_bounds = bounds
        self.quadrature_order = order
        self.orientation = orientation_
        self.tolerance = tolerance_
        self.mortar_id = canonical_fingerprint(
            {
                "kind": "mapped-patch-mortar",
                "owner": owner,
                "neighbor": neighbor,
                "bounds": bounds,
                "quadrature_order": order,
                "orientation": orientation_,
                "tolerance": tolerance_,
            }
        )

    def prepare(
        self,
        maps: PatchCoordinateMapSet,
        time: ArrayLike = 0.0,
        args: Any = None,
        /,
    ) -> MappedMortarGeometry:
        if not isinstance(maps, PatchCoordinateMapSet):
            raise TypeError("Mapped mortar preparation requires PatchCoordinateMapSet.")
        time_ = jnp.asarray(time)
        if time_.shape != ():
            raise ValueError("Mapped mortar time must be scalar.")
        nodes, weights = np.polynomial.legendre.leggauss(self.quadrature_order)
        axis_nodes = [
            0.5 * ((upper - lower) * nodes + upper + lower)
            for lower, upper in self.parameter_bounds
        ]
        axis_weights = [
            0.5 * (upper - lower) * weights for lower, upper in self.parameter_bounds
        ]
        parameters = np.asarray(tuple(product(*axis_nodes)), dtype=np.float64)
        quadrature_weights = np.asarray(
            [prod(values) for values in product(*axis_weights)], dtype=np.float64
        )

        def surface(parameter):
            return jnp.asarray(self.surface_map(parameter, time_, args))

        parameter_array = jnp.asarray(parameters)
        points = jax.vmap(surface)(parameter_array)
        jacobians = jax.vmap(jax.jacfwd(surface))(parameter_array)
        spatial_dimension = points.shape[-1]
        parameter_dimension = len(self.parameter_bounds)
        if spatial_dimension == 3 and parameter_dimension == 2:
            area = jnp.cross(jacobians[:, :, 0], jacobians[:, :, 1])
        elif spatial_dimension == 2 and parameter_dimension == 1:
            tangent = jacobians[:, :, 0]
            area = jnp.stack((tangent[:, 1], -tangent[:, 0]), axis=-1)
        else:
            raise ValueError("Mapped mortars require a codimension-one 2-D or 3-D chart.")
        area = self.orientation * area
        measure = jnp.linalg.norm(area, axis=-1)
        weighted_area = area * jnp.asarray(quadrature_weights)[:, None]
        owner_reference = jax.vmap(
            lambda parameter: jnp.asarray(
                self.owner_reference_map(parameter, time_, args)
            )
        )(parameter_array)
        neighbor_reference = jax.vmap(
            lambda parameter: jnp.asarray(
                self.neighbor_reference_map(parameter, time_, args)
            )
        )(parameter_array)
        owner_points = jax.vmap(
            lambda reference: maps.evaluate(self.owner_patch_id, reference, time_, args)
        )(owner_reference)
        neighbor_points = jax.vmap(
            lambda reference: maps.evaluate(
                self.neighbor_patch_id, reference, time_, args
            )
        )(neighbor_reference)
        owner_mismatch = jnp.max(jnp.linalg.norm(owner_points - points, axis=-1))
        neighbor_mismatch = jnp.max(jnp.linalg.norm(neighbor_points - points, axis=-1))
        evidence = MappedMortarEvidence(
            owner_mismatch,
            neighbor_mismatch,
            jnp.min(measure),
            self.tolerance,
        )
        return MappedMortarGeometry(
            quadrature_points=points,
            quadrature_weights=jnp.asarray(quadrature_weights) * measure,
            weighted_area_vectors=weighted_area,
            owner_reference_points=owner_reference,
            neighbor_reference_points=neighbor_reference,
            evidence=evidence,
            mortar_id=self.mortar_id,
            owner_patch_id=self.owner_patch_id,
            neighbor_patch_id=self.neighbor_patch_id,
        )


class CanonicalMappedGeometryEvidence(StrictModule, NonTrainableState):
    """Per-cell orientation, metric-closure, and GCL evidence."""

    minimum_jacobian: tuple[tuple[Array, ...], ...]
    face_closure_defect: tuple[tuple[Array, ...], ...]
    gcl_defect: tuple[tuple[Array, ...], ...]
    valid: Array
    tolerance: float = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        minimum_jacobian: Sequence[Sequence[ArrayLike]],
        face_closure_defect: Sequence[Sequence[ArrayLike]],
        gcl_defect: Sequence[Sequence[ArrayLike]],
        valid: ArrayLike,
        tolerance: float,
        /,
    ):
        jacobian = tuple(
            tuple(jnp.asarray(value) for value in level) for level in minimum_jacobian
        )
        closure = tuple(
            tuple(jnp.asarray(value) for value in level) for level in face_closure_defect
        )
        gcl = tuple(tuple(jnp.asarray(value) for value in level) for level in gcl_defect)
        valid_ = jnp.asarray(valid)
        tolerance_ = float(tolerance)
        if valid_.shape != () or valid_.dtype.kind != "b":
            raise ValueError("Mapped geometry validity must be scalar Boolean.")
        if not np.isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("Mapped geometry tolerance must be positive and finite.")
        self.minimum_jacobian = jacobian
        self.face_closure_defect = closure
        self.gcl_defect = gcl
        self.valid = valid_
        self.tolerance = tolerance_
        self.evidence_id = canonical_fingerprint(
            {"kind": "canonical-mapped-geometry-evidence", "tolerance": tolerance_}
        )


class CanonicalMappedGeometryState(StrictModule):
    """Traceable high-order cell and face metrics over canonical patch buckets."""

    time: Array
    revision: Array
    cell_volumes: tuple[tuple[Array, ...], ...]
    cell_centers: tuple[tuple[Array, ...], ...]
    mesh_volume_rates: tuple[tuple[Array, ...], ...]
    face_quadrature_points: tuple[tuple[Array, ...], ...]
    face_weighted_area_vectors: tuple[tuple[Array, ...], ...]
    face_grid_velocities: tuple[tuple[Array, ...], ...]
    evidence: CanonicalMappedGeometryEvidence
    topology_id: str = eqx.field(static=True)
    geometry_family_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)


class CanonicalMappedGeometryPlan(StrictModule, NonTrainableState):
    """Tensor-quadrature metric evaluator for canonical mapped patch hierarchies."""

    hierarchy: CanonicalPatchHierarchy
    maps: PatchCoordinateMapSet
    reference_lower_bounds: tuple[float, ...] = eqx.field(static=True)
    active_cell_indices: tuple[tuple[tuple[tuple[int, ...], ...], ...], ...] = eqx.field(
        static=True
    )
    quadrature_order: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    geometry_family_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        topology: BlockHierarchyTopology | VariablePatchHierarchyTopology,
        maps: PatchCoordinateMapSet,
        /,
        *,
        quadrature_order: int = 3,
        tolerance: float = 1.0e-9,
    ):
        hierarchy = canonicalize_patch_hierarchy(topology)
        order = int(quadrature_order)
        tolerance_ = float(tolerance)
        if not isinstance(maps, PatchCoordinateMapSet):
            raise TypeError("Canonical mapped geometry requires PatchCoordinateMapSet.")
        if order <= 0 or not np.isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("Mapped geometry quadrature and tolerance must be positive.")
        lower_bounds = tuple(
            float(np.asarray(axis.bounds[0]))
            for axis in hierarchy.topology.plan.grid.structured_axes
        )
        active_indices = tuple(
            tuple(
                tuple(
                    tuple(index)
                    for index in np.argwhere(
                        np.asarray(bucket.cell_active, dtype=np.bool_)
                    )
                )
                for bucket in level.buckets
            )
            for level in hierarchy.levels
        )
        self.hierarchy = hierarchy
        self.maps = maps
        self.reference_lower_bounds = lower_bounds
        self.active_cell_indices = active_indices
        self.quadrature_order = order
        self.tolerance = tolerance_
        self.geometry_family_id = canonical_fingerprint(
            {"kind": "canonical-mapped-family", "maps": maps.map_set_id}
        )
        self.geometry_layout_id = canonical_fingerprint(
            {
                "kind": "canonical-mapped-layout",
                "hierarchy": hierarchy.layout_id,
                "quadrature_order": order,
            }
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "canonical-mapped-geometry-plan",
                "hierarchy": hierarchy.hierarchy_id,
                "maps": maps.map_set_id,
                "quadrature_order": order,
                "reference_lower_bounds": lower_bounds,
                "active_cell_counts": [
                    [len(indices) for indices in level] for level in active_indices
                ],
                "tolerance": tolerance_,
            }
        )

    def _cell_metrics(
        self,
        patch_id: str,
        lower: np.ndarray,
        spacing: np.ndarray,
        time: Array,
        args: Any,
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array]:
        dimension = lower.size
        nodes_host, weights_host = np.polynomial.legendre.leggauss(self.quadrature_order)
        unit_nodes = 0.5 * (nodes_host + 1.0)
        unit_weights = 0.5 * weights_host
        cell_parameters = np.asarray(tuple(product(*((unit_nodes,) * dimension))))
        cell_weights = np.asarray(
            [prod(values) for values in product(*((unit_weights,) * dimension))]
        ) * prod(spacing)
        references = jnp.asarray(lower + cell_parameters * spacing)

        def mapped(reference, stage_time):
            return self.maps.evaluate(patch_id, reference, stage_time, args)

        points = jax.vmap(lambda reference: mapped(reference, time))(references)
        jacobians = jax.vmap(
            lambda reference: jax.jacfwd(lambda point: mapped(point, time))(reference)
        )(references)
        determinants = jax.vmap(jnp.linalg.det)(jacobians)
        weighted_jacobian = determinants * jnp.asarray(cell_weights)
        volume = jnp.sum(weighted_jacobian)
        center = jnp.sum(points * weighted_jacobian[:, None], axis=0) / volume

        face_parameters = np.asarray(
            tuple(product(*((unit_nodes,) * max(0, dimension - 1)))), dtype=np.float64
        ).reshape((-1, max(0, dimension - 1)))
        face_reference_weight = np.asarray(
            [
                prod(values)
                for values in product(*((unit_weights,) * max(0, dimension - 1)))
            ],
            dtype=np.float64,
        )
        face_point_blocks = []
        face_area_blocks = []
        face_velocity_blocks = []
        face_flux_sum = jnp.zeros((dimension,), dtype=points.dtype)
        mesh_volume_rate = jnp.zeros((), dtype=points.dtype)
        for axis in range(dimension):
            tangential = tuple(value for value in range(dimension) if value != axis)
            reference_weight = face_reference_weight * prod(
                spacing[value] for value in tangential
            )
            for side in (0, 1):
                reference_values = np.zeros(
                    (face_parameters.shape[0], dimension), dtype=np.float64
                )
                reference_values[:, axis] = float(side)
                for local_axis, global_axis in enumerate(tangential):
                    reference_values[:, global_axis] = face_parameters[:, local_axis]
                face_references = jnp.asarray(lower + reference_values * spacing)
                face_points = jax.vmap(lambda reference: mapped(reference, time))(
                    face_references
                )
                face_jacobians = jax.vmap(
                    lambda reference: jax.jacfwd(lambda point: mapped(point, time))(
                        reference
                    )
                )(face_references)
                if dimension == 3:
                    columns = tuple(face_jacobians[:, :, value] for value in range(3))
                    cofactors = (
                        jnp.cross(columns[1], columns[2]),
                        jnp.cross(columns[2], columns[0]),
                        jnp.cross(columns[0], columns[1]),
                    )
                    area = cofactors[axis]
                elif dimension == 2:
                    matrix = face_jacobians
                    cofactors = (
                        jnp.stack((matrix[:, 1, 1], -matrix[:, 0, 1]), axis=-1),
                        jnp.stack((-matrix[:, 1, 0], matrix[:, 0, 0]), axis=-1),
                    )
                    area = cofactors[axis]
                else:
                    area = jnp.ones((face_references.shape[0], 1), dtype=points.dtype)
                sign = -1.0 if side == 0 else 1.0
                weighted_area = sign * area * jnp.asarray(reference_weight)[:, None]
                velocities = jax.vmap(
                    lambda reference: jax.jvp(
                        lambda stage_time: mapped(reference, stage_time),
                        (time,),
                        (jnp.ones_like(time),),
                    )[1]
                )(face_references)
                face_point_blocks.append(face_points)
                face_area_blocks.append(weighted_area)
                face_velocity_blocks.append(velocities)
                face_flux_sum = face_flux_sum + jnp.sum(weighted_area, axis=0)
                mesh_volume_rate = mesh_volume_rate + jnp.sum(
                    jnp.sum(velocities * weighted_area, axis=-1)
                )

        def volume_at(stage_time):
            stage_jacobians = jax.vmap(
                lambda reference: jax.jacfwd(lambda point: mapped(point, stage_time))(
                    reference
                )
            )(references)
            stage_determinants = jax.vmap(jnp.linalg.det)(stage_jacobians)
            return jnp.sum(stage_determinants * jnp.asarray(cell_weights))

        volume_rate = jax.jvp(
            volume_at,
            (time,),
            (jnp.ones_like(time),),
        )[1]
        return (
            volume,
            center,
            volume_rate,
            jnp.min(determinants),
            jnp.linalg.norm(face_flux_sum),
            jnp.abs(volume_rate - mesh_volume_rate),
            jnp.stack(face_point_blocks),
            jnp.stack(face_area_blocks),
            jnp.stack(face_velocity_blocks),
        )

    def evaluate(
        self,
        time: ArrayLike,
        args: Any = None,
        /,
        *,
        revision: ArrayLike = 0,
    ) -> CanonicalMappedGeometryState:
        time_ = jnp.asarray(time)
        revision_ = _validated_revision(revision)
        if time_.shape != ():
            raise ValueError("Mapped geometry time must be scalar.")
        lower_bounds = np.asarray(self.reference_lower_bounds, dtype=np.float64)
        level_volumes = []
        level_centers = []
        level_rates = []
        level_points = []
        level_areas = []
        level_velocities = []
        level_jacobian = []
        level_closure = []
        level_gcl = []
        valid_terms = []
        for level_index, level in enumerate(self.hierarchy.levels):
            spacing = np.asarray(level.spacing, dtype=np.float64)
            bucket_volumes = []
            bucket_centers = []
            bucket_rates = []
            bucket_points = []
            bucket_areas = []
            bucket_velocities = []
            bucket_jacobian = []
            bucket_closure = []
            bucket_gcl = []
            for bucket_index, bucket in enumerate(level.buckets):
                shape = (
                    bucket.plan.lane_capacity,
                ) + bucket.plan.signature.envelope_shape
                dimension = self.hierarchy.dimension
                face_count = 2 * dimension
                face_quadrature_count = self.quadrature_order ** max(0, dimension - 1)
                volumes = jnp.zeros(shape)
                centers = jnp.zeros(shape + (dimension,))
                rates = jnp.zeros(shape)
                points = jnp.zeros(shape + (face_count, face_quadrature_count, dimension))
                areas = jnp.zeros_like(points)
                velocities = jnp.zeros_like(points)
                jacobian = jnp.zeros(shape)
                closure = jnp.zeros(shape)
                gcl = jnp.zeros(shape)
                for index in self.active_cell_indices[level_index][bucket_index]:
                    lane = index[0]
                    local = index[1:]
                    box = bucket.boxes[lane]
                    if box is None:
                        raise RuntimeError(
                            "Static mapped active index lost its patch box."
                        )
                    reference_lower = lower_bounds + spacing * (
                        np.asarray(box.lower, dtype=np.float64)
                        + np.asarray(local, dtype=np.float64)
                    )
                    (
                        volume,
                        center,
                        rate,
                        minimum_jacobian,
                        closure_defect,
                        gcl_defect,
                        face_points,
                        face_areas,
                        face_velocity,
                    ) = self._cell_metrics(
                        box.box_id,
                        reference_lower,
                        spacing,
                        time_,
                        args,
                    )
                    volumes = volumes.at[index].set(volume)
                    centers = centers.at[index].set(center)
                    rates = rates.at[index].set(rate)
                    points = points.at[index].set(face_points)
                    areas = areas.at[index].set(face_areas)
                    velocities = velocities.at[index].set(face_velocity)
                    jacobian = jacobian.at[index].set(minimum_jacobian)
                    closure = closure.at[index].set(closure_defect)
                    gcl = gcl.at[index].set(gcl_defect)
                active_array = jnp.asarray(bucket.cell_active)
                valid_terms.append(
                    jnp.all(
                        (~active_array)
                        | (
                            (jacobian > 0.0)
                            & (closure <= self.tolerance)
                            & (gcl <= self.tolerance)
                        )
                    )
                )
                bucket_volumes.append(volumes)
                bucket_centers.append(centers)
                bucket_rates.append(rates)
                bucket_points.append(points)
                bucket_areas.append(areas)
                bucket_velocities.append(velocities)
                bucket_jacobian.append(jacobian)
                bucket_closure.append(closure)
                bucket_gcl.append(gcl)
            level_volumes.append(tuple(bucket_volumes))
            level_centers.append(tuple(bucket_centers))
            level_rates.append(tuple(bucket_rates))
            level_points.append(tuple(bucket_points))
            level_areas.append(tuple(bucket_areas))
            level_velocities.append(tuple(bucket_velocities))
            level_jacobian.append(tuple(bucket_jacobian))
            level_closure.append(tuple(bucket_closure))
            level_gcl.append(tuple(bucket_gcl))
        evidence = CanonicalMappedGeometryEvidence(
            level_jacobian,
            level_closure,
            level_gcl,
            jnp.all(jnp.stack(valid_terms)),
            self.tolerance,
        )
        return CanonicalMappedGeometryState(
            time=time_,
            revision=revision_,
            cell_volumes=tuple(level_volumes),
            cell_centers=tuple(level_centers),
            mesh_volume_rates=tuple(level_rates),
            face_quadrature_points=tuple(level_points),
            face_weighted_area_vectors=tuple(level_areas),
            face_grid_velocities=tuple(level_velocities),
            evidence=evidence,
            topology_id=self.hierarchy.topology_id,
            geometry_family_id=self.geometry_family_id,
            geometry_layout_id=self.geometry_layout_id,
        )


class MappedMortarFluxResult(StrictModule):
    """Integrated owner-to-neighbor mortar flux and conservative scatter."""

    integrated_flux: Array
    content_rate: Array
    conservation_defect: Array
    maximum_speed: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class MappedMortarFluxPlan(StrictModule, NonTrainableState):
    """Conservative arbitrary-normal flux integration on one physical mortar."""

    geometry: MappedMortarGeometry
    owner_cell: int = eqx.field(static=True)
    neighbor_cell: int = eqx.field(static=True)
    cell_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        geometry: MappedMortarGeometry,
        owner_cell: int,
        neighbor_cell: int,
        cell_count: int,
        /,
    ):
        owner = int(owner_cell)
        neighbor = int(neighbor_cell)
        count = int(cell_count)
        if (
            not isinstance(geometry, MappedMortarGeometry)
            or owner < 0
            or neighbor < 0
            or owner >= count
            or neighbor >= count
            or owner == neighbor
        ):
            raise ValueError("Mapped mortar flux routes are invalid.")
        self.geometry = geometry
        self.owner_cell = owner
        self.neighbor_cell = neighbor
        self.cell_count = count
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mapped-mortar-flux-plan",
                "mortar": geometry.mortar_id,
                "owner": owner,
                "neighbor": neighbor,
                "cell_count": count,
            }
        )

    def evaluate(
        self,
        system: Any,
        numerical_flux: Any,
        owner_trace: ArrayLike,
        neighbor_trace: ArrayLike,
        args: Any = None,
        /,
    ) -> MappedMortarFluxResult:
        from ..finite_volume._riemann import AbstractArbitraryNormalNumericalFluxPlan

        if not isinstance(numerical_flux, AbstractArbitraryNormalNumericalFluxPlan):
            raise TypeError("Mapped mortars require arbitrary-normal numerical flux.")
        owner = jnp.asarray(owner_trace)
        neighbor = jnp.asarray(neighbor_trace, dtype=owner.dtype)
        quadrature_count = self.geometry.quadrature_weights.size
        component_count = int(system.component_count)
        expected = (quadrature_count, component_count)
        if owner.shape == (component_count,):
            owner = jnp.broadcast_to(owner, expected)
        if neighbor.shape == (component_count,):
            neighbor = jnp.broadcast_to(neighbor, expected)
        if owner.shape != expected or neighbor.shape != expected:
            raise ValueError(
                "Mapped mortar traces must provide one state per quadrature point."
            )
        weights = self.geometry.quadrature_weights.astype(owner.dtype)
        area = self.geometry.weighted_area_vectors.astype(owner.dtype)
        normals = area / weights[:, None]
        flux = numerical_flux.normal_face_flux(
            system,
            owner,
            neighbor,
            normals,
            args,
        )
        integrated = jnp.sum(flux.normal_flux * weights[:, None], axis=0)
        content_rate = jnp.zeros(
            (self.cell_count, component_count), dtype=integrated.dtype
        )
        content_rate = content_rate.at[self.owner_cell].add(-integrated)
        content_rate = content_rate.at[self.neighbor_cell].add(integrated)
        defect = jnp.sum(content_rate, axis=0)
        successful = (
            self.geometry.evidence.valid
            & jnp.all(jnp.isfinite(integrated))
            & jnp.all(jnp.isfinite(flux.max_speed))
            & jnp.all(
                jnp.abs(defect)
                <= 32.0
                * jnp.finfo(integrated.dtype).eps
                * jnp.maximum(jnp.abs(integrated), 1.0)
            )
        )
        return MappedMortarFluxResult(
            integrated_flux=integrated,
            content_rate=content_rate,
            conservation_defect=defect,
            maximum_speed=jnp.max(flux.max_speed),
            successful=successful,
            plan_id=self.plan_id,
        )


__all__ = [
    "CanonicalMappedGeometryEvidence",
    "CanonicalMappedGeometryPlan",
    "CanonicalMappedGeometryState",
    "MappedMortarEvidence",
    "MappedMortarGeometry",
    "MappedMortarFluxPlan",
    "MappedMortarFluxResult",
    "MappedMortarPlan",
    "PatchCoordinateMapSet",
]
