#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from itertools import product
from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import ArraySpace, DiagonalPairing
from .._axis import broadcasted_grid
from .._core import DiscretizationCapability, DiscretizationKey, PreparationReport
from .._lifecycle import (
    AbstractDiscretizationPlan,
    AbstractPreparedDiscretization,
    validate_prepared_metadata,
)
from .._measure import DiscreteMeasure
from .._spaces import DiscreteFieldSpace, TensorDofLayout
from .._support import DiscreteSupport
from .._tensor_entities import TensorEntityLayout
from .._tensor_support import PreparedTensorGrid
from ._structured import FiniteVolumeDiscretization


CoordinateMap = Callable[[Array], ArrayLike]


def _mapped_vertices(
    base: FiniteVolumeDiscretization, coordinate_map: CoordinateMap, /
) -> Array:
    edge_axes = tuple(
        axis.bounds[0]
        + jnp.concatenate(
            (
                jnp.zeros((1,), dtype=axis.interval_widths.dtype),
                jnp.cumsum(axis.interval_widths),
            )
        )
        for axis in base.grid.structured_axes
    )
    reference = broadcasted_grid(edge_axes)
    flat = reference.reshape((-1, len(edge_axes)))
    mapped = jax.vmap(lambda point: jnp.asarray(coordinate_map(point)))(flat)
    if mapped.shape != flat.shape:
        raise ValueError("coordinate_map must preserve the spatial coordinate dimension.")
    return mapped.reshape(reference.shape)


def _corner(
    vertices: Array, bits: tuple[int, ...], cell_shape: tuple[int, ...], /
) -> Array:
    index = tuple(slice(bit, bit + cell_shape[axis]) for axis, bit in enumerate(bits))
    return vertices[index]


def _cross2(left: Array, right: Array, /) -> Array:
    return left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]


def _cell_geometry(
    vertices: Array, cell_shape: tuple[int, ...], /
) -> tuple[Array, Array]:
    dimension = len(cell_shape)
    corners = {
        bits: _corner(vertices, bits, cell_shape)
        for bits in product((0, 1), repeat=dimension)
    }
    center = jnp.mean(jnp.stack(tuple(corners.values()), axis=0), axis=0)
    if dimension == 1:
        signed = corners[(1,)][..., 0] - corners[(0,)][..., 0]
        volume = eqx.error_if(
            signed,
            jnp.any(~jnp.isfinite(signed) | (signed <= 0.0)),
            "Mapped one-dimensional cells require positive orientation.",
        )
        return center, volume
    if dimension == 2:
        p00 = corners[(0, 0)]
        p10 = corners[(1, 0)]
        p11 = corners[(1, 1)]
        p01 = corners[(0, 1)]
        signed = 0.5 * (
            _cross2(p00, p10) + _cross2(p10, p11) + _cross2(p11, p01) + _cross2(p01, p00)
        )
        volume = eqx.error_if(
            signed,
            jnp.any(~jnp.isfinite(signed) | (signed <= 0.0)),
            "Mapped quadrilateral cells require positive orientation.",
        )
        return center, volume
    if dimension != 3:
        raise ValueError("Mapped finite volumes support one, two, or three dimensions.")

    def tetra(a: Array, b: Array, c: Array, d: Array) -> Array:
        return jnp.linalg.det(jnp.stack((b - a, c - a, d - a), axis=-1)) / 6.0

    p000 = corners[(0, 0, 0)]
    p100 = corners[(1, 0, 0)]
    p010 = corners[(0, 1, 0)]
    p110 = corners[(1, 1, 0)]
    p001 = corners[(0, 0, 1)]
    p101 = corners[(1, 0, 1)]
    p011 = corners[(0, 1, 1)]
    p111 = corners[(1, 1, 1)]
    parts = (
        tetra(p000, p100, p010, p001),
        tetra(p100, p110, p010, p111),
        tetra(p100, p010, p001, p111),
        tetra(p100, p001, p101, p111),
        tetra(p010, p001, p111, p011),
    )
    volume = sum(jnp.abs(part) for part in parts)
    orientation = jnp.linalg.det(
        jnp.stack(
            (
                0.25 * (p100 + p110 + p101 + p111 - p000 - p010 - p001 - p011),
                0.25 * (p010 + p110 + p011 + p111 - p000 - p100 - p001 - p101),
                0.25 * (p001 + p101 + p011 + p111 - p000 - p100 - p010 - p110),
            ),
            axis=-1,
        )
    )
    volume = eqx.error_if(
        volume,
        jnp.any(~jnp.isfinite(volume) | (volume <= 0.0) | (orientation <= 0.0)),
        "Mapped hexahedral cells require positive orientation and volume.",
    )
    return center, volume


def _face_geometry(
    vertices: Array,
    base: FiniteVolumeDiscretization,
    axis: int,
    /,
    *,
    include_periodic_endpoint: bool = False,
) -> tuple[Array, Array, Array]:
    dimension = len(base.cell_shape)
    tangential = tuple(index for index in range(dimension) if index != axis)
    face_shape_full = list(base.face_layouts[axis].shape)
    if base.grid.structured_axes[axis].periodic:
        face_shape_full[axis] += 1

    def face_corner(bits: tuple[int, ...]) -> Array:
        index = []
        bit_by_axis = {tangential[i]: bits[i] for i in range(len(tangential))}
        for current_axis in range(dimension):
            if current_axis == axis:
                index.append(slice(0, face_shape_full[current_axis]))
            else:
                bit = bit_by_axis[current_axis]
                index.append(slice(bit, bit + base.cell_shape[current_axis]))
        return vertices[tuple(index)]

    corners = tuple(face_corner(bits) for bits in product((0, 1), repeat=dimension - 1))
    center = jnp.mean(jnp.stack(corners, axis=0), axis=0)
    if dimension == 1:
        area_vector = jnp.ones(center.shape)
    elif dimension == 2:
        tangent = corners[1] - corners[0]
        area_vector = (
            jnp.stack((tangent[..., 1], -tangent[..., 0]), axis=-1)
            if axis == 0
            else jnp.stack((-tangent[..., 1], tangent[..., 0]), axis=-1)
        )
    else:
        p00, p01, p10, p11 = corners
        first = jnp.cross(p10 - p00, p11 - p00)
        second = jnp.cross(p11 - p00, p01 - p00)
        area_vector = 0.5 * (first + second)
        if axis == 1:
            area_vector = -area_vector
    if base.grid.structured_axes[axis].periodic and not include_periodic_endpoint:
        keep = [slice(None)] * center.ndim
        keep[axis] = slice(0, base.face_layouts[axis].shape[axis])
        center = center[tuple(keep)]
        area_vector = area_vector[tuple(keep)]
    measure = jnp.linalg.norm(area_vector, axis=-1)
    measure = eqx.error_if(
        measure,
        jnp.any(~jnp.isfinite(measure) | (measure <= 0.0)),
        "Mapped finite-volume faces require finite positive measure.",
    )
    return center, measure, area_vector


def evaluate_mapped_finite_volume_geometry(
    reference: FiniteVolumeDiscretization,
    coordinate_map: CoordinateMap,
    /,
) -> tuple[
    Array,
    Array,
    Array,
    tuple[Array, ...],
    tuple[Array, ...],
    tuple[Array, ...],
]:
    """Pure fixed-topology mapped vertices, cells, and directional faces."""
    if not isinstance(reference, FiniteVolumeDiscretization) or not callable(
        coordinate_map
    ):
        raise TypeError(
            "Mapped geometry evaluation requires reference geometry and a map."
        )
    vertices = _mapped_vertices(reference, coordinate_map)
    cell_centers, cell_volumes = _cell_geometry(vertices, reference.cell_shape)
    face_geometry = tuple(
        _face_geometry(vertices, reference, axis)
        for axis in range(len(reference.cell_shape))
    )
    return (
        vertices,
        cell_centers,
        cell_volumes,
        tuple(value[0] for value in face_geometry),
        tuple(value[1] for value in face_geometry),
        tuple(value[2] for value in face_geometry),
    )


class MappedPeriodicSeam(StrictModule, NonTrainableState):
    """Prepared, geometry-certified periodic isometry for one mapped axis."""

    axis: int = eqx.field(static=True)
    rotation: Array
    translation: Array
    tolerance: float = eqx.field(static=True)
    seam_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)

    def image(self, coordinates: ArrayLike, /, *, inverse: bool = False) -> Array:
        points = jnp.asarray(coordinates)
        rotation = self.rotation.T if inverse else self.rotation
        shifted = points - self.translation if inverse else points
        mapped = jnp.matmul(shifted, rotation.T)
        return mapped if inverse else mapped + self.translation

    def transform_conserved(self, state: ArrayLike, /, *, inverse: bool = False) -> Array:
        """Rotate the momentum block of a density-momentum-energy state."""
        value = jnp.asarray(state)
        dimension = self.rotation.shape[0]
        if value.shape[-1] < dimension + 1:
            raise ValueError("Conserved state does not contain a momentum block.")
        rotation = self.rotation.T if inverse else self.rotation
        momentum = jnp.matmul(value[..., 1 : dimension + 1], rotation.T)
        return value.at[..., 1 : dimension + 1].set(momentum)


class MappedPeriodicSeamPlan(StrictModule, NonTrainableState):
    """Declared Euclidean isometry pairing the ends of one mapped axis."""

    axis: int = eqx.field(static=True)
    rotation: Array
    translation: Array
    tolerance: float = eqx.field(static=True)
    seam_id: str = eqx.field(static=True)

    def __init__(
        self,
        axis: int,
        rotation: ArrayLike,
        translation: ArrayLike,
        /,
        *,
        tolerance: float = 1.0e-10,
    ):
        axis_ = int(axis)
        rotation_ = np.asarray(rotation, dtype=float)
        translation_ = np.asarray(translation, dtype=float)
        tolerance_ = float(tolerance)
        if (
            rotation_.ndim != 2
            or rotation_.shape[0] != rotation_.shape[1]
            or translation_.shape != (rotation_.shape[0],)
        ):
            raise ValueError("Seam rotation and translation dimensions must agree.")
        if not np.isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("Seam tolerance must be positive and finite.")
        if not np.all(np.isfinite(rotation_)) or not np.all(np.isfinite(translation_)):
            raise ValueError("Seam isometry must be finite.")
        identity = np.eye(rotation_.shape[0])
        if not np.allclose(rotation_.T @ rotation_, identity, atol=tolerance_, rtol=0.0):
            raise ValueError("Mapped periodic seam rotation must be orthogonal.")
        if axis_ < 0 or axis_ >= rotation_.shape[0]:
            raise ValueError("Mapped periodic seam axis is out of range.")
        self.axis = axis_
        self.rotation = jnp.asarray(rotation_)
        self.translation = jnp.asarray(translation_)
        self.tolerance = tolerance_
        self.seam_id = canonical_fingerprint(
            {
                "kind": "mapped-periodic-seam",
                "axis": axis_,
                "rotation": rotation_.tolist(),
                "translation": translation_.tolist(),
                "tolerance": tolerance_,
            }
        )

    def prepare(
        self, discretization: "MappedFiniteVolumeDiscretization", /
    ) -> MappedPeriodicSeam:
        if not isinstance(discretization, MappedFiniteVolumeDiscretization):
            raise TypeError("Mapped periodic seams require prepared mapped geometry.")
        if self.rotation.shape != (len(discretization.cell_shape),) * 2:
            raise ValueError("Seam isometry dimension does not match mapped geometry.")
        axis = self.axis
        centers, measures, area_vectors = _face_geometry(
            discretization.mapped_vertices,
            discretization.reference,
            axis,
            include_periodic_endpoint=True,
        )
        upper_index = centers.shape[axis] - 1
        lower_centers = np.asarray(jax.device_get(jnp.take(centers, 0, axis=axis)))
        upper_centers = np.asarray(
            jax.device_get(jnp.take(centers, upper_index, axis=axis))
        )
        mapped_lower = lower_centers @ np.asarray(self.rotation).T + np.asarray(
            self.translation
        )
        if not np.allclose(mapped_lower, upper_centers, atol=self.tolerance, rtol=0.0):
            raise ValueError("Mapped periodic seam face coordinates do not match.")
        lower_measure = np.asarray(jax.device_get(jnp.take(measures, 0, axis=axis)))
        upper_measure = np.asarray(
            jax.device_get(jnp.take(measures, upper_index, axis=axis))
        )
        if not np.allclose(lower_measure, upper_measure, atol=self.tolerance, rtol=0.0):
            raise ValueError("Mapped periodic seam face measures do not match.")
        lower_area = np.asarray(jax.device_get(jnp.take(area_vectors, 0, axis=axis)))
        upper_area = np.asarray(
            jax.device_get(jnp.take(area_vectors, upper_index, axis=axis))
        )
        lower_normal = -lower_area / lower_measure[..., None]
        upper_normal = upper_area / upper_measure[..., None]
        mapped_normal = lower_normal @ np.asarray(self.rotation).T
        if not np.allclose(mapped_normal, -upper_normal, atol=self.tolerance, rtol=0.0):
            raise ValueError("Mapped periodic seam normals do not oppose.")
        lower_cells = np.asarray(
            jax.device_get(jnp.take(discretization.cell_centers, 0, axis=axis))
        )
        upper_cells = np.asarray(
            jax.device_get(
                jnp.take(
                    discretization.cell_centers,
                    discretization.cell_shape[axis] - 1,
                    axis=axis,
                )
            )
        )
        seam_distance = np.sum((mapped_lower - upper_cells) * upper_normal, axis=-1)
        seam_distance += np.sum((lower_centers - lower_cells) * lower_normal, axis=-1)
        if np.any(~np.isfinite(seam_distance)) or np.any(seam_distance <= 0.0):
            raise ValueError("Mapped periodic seam distance must be positive and finite.")
        return MappedPeriodicSeam(
            axis=self.axis,
            rotation=self.rotation,
            translation=self.translation,
            tolerance=self.tolerance,
            seam_id=self.seam_id,
            geometry_id=discretization.prepared_id,
        )


class MappedFiniteVolumePlan(AbstractDiscretizationPlan):
    """Stationary coordinate map over structured reference control volumes."""

    reference: FiniteVolumeDiscretization
    grid: PreparedTensorGrid
    field_name: str = eqx.field(static=True)
    component_names: tuple[str, ...] = eqx.field(static=True)
    coordinate_map: CoordinateMap = eqx.field(static=True)
    mapping_id: str = eqx.field(static=True)
    periodic_seams: tuple[MappedPeriodicSeamPlan, ...]
    key: DiscretizationKey
    capabilities: tuple[DiscretizationCapability, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        reference: FiniteVolumeDiscretization,
        coordinate_map: CoordinateMap,
        /,
        *,
        mapping_id: str,
        periodic_seams: tuple[MappedPeriodicSeamPlan, ...] = (),
    ):
        if not isinstance(reference, FiniteVolumeDiscretization) or not callable(
            coordinate_map
        ):
            raise TypeError("Mapped finite volumes require reference geometry and a map.")
        identifier = str(mapping_id)
        if not identifier:
            raise ValueError("mapping_id must be non-empty.")
        seams = tuple(periodic_seams)
        if any(not isinstance(seam, MappedPeriodicSeamPlan) for seam in seams):
            raise TypeError("periodic_seams must contain MappedPeriodicSeamPlan values.")
        axes = tuple(seam.axis for seam in seams)
        if len(set(axes)) != len(axes):
            raise ValueError("Mapped periodic seam axes must be unique.")
        if any(seam.rotation.shape[0] != len(reference.cell_shape) for seam in seams):
            raise ValueError("Mapped periodic seam dimension must match the grid.")
        if any(not reference.grid.structured_axes[axis].periodic for axis in axes):
            raise ValueError("Mapped periodic seams may bind only periodic axes.")
        self.reference = reference
        self.grid = reference.grid
        self.field_name = reference.field_name
        self.component_names = reference.component_names
        self.coordinate_map = coordinate_map
        self.mapping_id = identifier
        self.periodic_seams = seams
        self.key = reference.key
        self.capabilities = reference.capabilities
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mapped-finite-volume-plan",
                "reference": reference.prepared_id,
                "mapping": identifier,
                "periodic_seams": tuple(seam.seam_id for seam in seams),
            }
        )

    def prepare(
        self, /, *, numeric_version: str = "0"
    ) -> "MappedFiniteVolumeDiscretization":
        return MappedFiniteVolumeDiscretization(self, numeric_version=numeric_version)


class MappedFiniteVolumeDiscretization(AbstractPreparedDiscretization):
    """Prepared stationary mapped control-volume geometry."""

    grid: PreparedTensorGrid
    cell_layout: TensorEntityLayout
    face_layouts: tuple[TensorEntityLayout, ...]
    cell_centers: Array
    cell_volumes: Array
    face_centers: tuple[Array, ...]
    face_measures: tuple[Array, ...]
    face_area_vectors: tuple[Array, ...]
    cell_space: DiscreteFieldSpace
    face_spaces: tuple[DiscreteFieldSpace, ...]
    field_name: str = eqx.field(static=True)
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
    reference: FiniteVolumeDiscretization
    mapped_vertices: Array
    mapping_id: str = eqx.field(static=True)
    periodic_seams: tuple[MappedPeriodicSeam, ...]

    def __init__(
        self,
        plan: MappedFiniteVolumePlan,
        /,
        *,
        numeric_version: str = "0",
    ):
        if not isinstance(plan, MappedFiniteVolumePlan):
            raise TypeError("plan must be a MappedFiniteVolumePlan.")
        reference = plan.reference
        (
            vertices,
            cell_centers,
            cell_volumes,
            face_centers,
            face_measures,
            face_area_vectors,
        ) = evaluate_mapped_finite_volume_geometry(reference, plan.coordinate_map)
        component_count = reference.component_count
        cell_shape = reference.cell_shape + (component_count,)
        cell_space = DiscreteFieldSpace(
            reference.field_name,
            reference.support.support_id,
            TensorDofLayout(
                reference.grid.axis_names,
                reference.cell_shape,
                component_shape=(component_count,),
                location_id=reference.cell_layout.location_id,
            ),
            ArraySpace(
                cell_shape,
                pairing=DiagonalPairing(
                    jnp.broadcast_to(cell_volumes[..., None], cell_shape)
                ),
            ),
            representation="cell_average",
            conformity="discontinuous",
            reconstruction_id=reference.cell_space.reconstruction_id,
        )
        face_spaces = tuple(
            DiscreteFieldSpace(
                reference.face_spaces[axis].name,
                reference.support.support_id,
                TensorDofLayout(
                    reference.grid.axis_names,
                    layout.shape,
                    component_shape=(component_count,),
                    location_id=layout.location_id,
                ),
                ArraySpace(
                    layout.shape + (component_count,),
                    pairing=DiagonalPairing(
                        jnp.broadcast_to(
                            face_measures[axis][..., None],
                            layout.shape + (component_count,),
                        )
                    ),
                ),
                representation="flux_moment",
                conformity="Hdiv",
                trace_space_id=cell_space.field_space_id,
            )
            for axis, layout in enumerate(reference.face_layouts)
        )
        measures = (
            DiscreteMeasure(
                "mapped_finite_volume_cell",
                reference.support.support_id,
                reference.cell_layout.entity_set_id,
                cell_volumes.reshape((-1,)),
            ),
            *tuple(
                DiscreteMeasure(
                    f"mapped_finite_volume_{reference.grid.axis_names[axis]}_face",
                    reference.support.support_id,
                    layout.entity_set_id,
                    face_measures[axis].reshape((-1,)),
                )
                for axis, layout in enumerate(reference.face_layouts)
            ),
        )
        preparation = PreparationReport(
            capabilities=plan.capabilities,
            diagnostics=(
                "mapped cells have positive orientation and measure",
                "mapped face area vectors use canonical reference-axis orientation",
                "mapped topology is fixed during differentiation",
            ),
            resource_counts={
                "cells": prod(reference.cell_shape),
                "components": component_count,
                "faces": sum(prod(layout.shape) for layout in reference.face_layouts),
            },
        )
        spaces, measures_, capabilities = validate_prepared_metadata(
            key=plan.key,
            support=reference.support,
            field_spaces=(cell_space, *face_spaces),
            measures=measures,
            capabilities=plan.capabilities,
            preparation=preparation,
        )
        version = str(numeric_version)
        if not version:
            raise ValueError("numeric_version must be non-empty.")
        self.grid = reference.grid
        self.cell_layout = reference.cell_layout
        self.face_layouts = reference.face_layouts
        self.cell_centers = cell_centers
        self.cell_volumes = cell_volumes
        self.face_centers = face_centers
        self.face_measures = face_measures
        self.face_area_vectors = face_area_vectors
        self.cell_space = cell_space
        self.face_spaces = face_spaces
        self.field_name = reference.field_name
        self.component_names = reference.component_names
        self.key = plan.key
        self.support = reference.support
        self.field_spaces = spaces
        self.measures = measures_
        self.capabilities = capabilities
        self.plan_id = plan.plan_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-mapped-finite-volume",
                "plan": plan.plan_id,
                "mapping": plan.mapping_id,
                "numeric_version": version,
                "periodic_seams": tuple(seam.seam_id for seam in plan.periodic_seams),
            }
        )
        self.numeric_version = version
        self.preparation = preparation
        self.reference = reference
        self.mapped_vertices = vertices
        self.mapping_id = plan.mapping_id
        self.periodic_seams = tuple(seam.prepare(self) for seam in plan.periodic_seams)

    @property
    def cell_shape(self) -> tuple[int, ...]:
        return self.cell_layout.shape

    @property
    def component_count(self) -> int:
        return len(self.component_names)

    @property
    def state_shape(self) -> tuple[int, ...]:
        return self.cell_shape + (self.component_count,)

    def outward_normal(self, axis: int, side: str, /) -> Array:
        if not 0 <= int(axis) < len(self.cell_shape) or side not in (
            "lower",
            "upper",
        ):
            raise ValueError("axis and side must identify one mapped boundary.")
        axis_ = int(axis)
        index = 0 if side == "lower" else self.face_layouts[axis_].shape[axis_] - 1
        area_vector = jnp.take(self.face_area_vectors[axis_], index, axis=axis_)
        measure = jnp.take(self.face_measures[axis_], index, axis=axis_)
        sign = -1.0 if side == "lower" else 1.0
        return sign * area_vector / measure[..., None]


__all__ = [
    "MappedFiniteVolumeDiscretization",
    "MappedFiniteVolumePlan",
    "MappedPeriodicSeam",
    "MappedPeriodicSeamPlan",
    "evaluate_mapped_finite_volume_geometry",
]
