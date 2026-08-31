#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from itertools import combinations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..sparse import EdgeRelation, SparseLinearMap
from ._cochain import CochainBoundaryKind, CochainDiscretization
from ._tensor_support import PreparedTensorGrid
from ._topology import CellComplexTopology, EntitySet, OrientedIncidence


class StructuredCochainBridge(StrictModule, NonTrainableState):
    """Cartesian tensor entities assembled into one oriented cubical cochain complex."""

    grid: PreparedTensorGrid
    cochain: CochainDiscretization
    orientations: tuple[tuple[tuple[int, ...], ...], ...] = eqx.field(static=True)
    orientation_shapes: tuple[tuple[tuple[int, ...], ...], ...] = eqx.field(static=True)
    orientation_offsets: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    directional_signs: tuple[tuple[Array, ...], ...]
    bridge_id: str = eqx.field(static=True)

    def __init__(self, grid: PreparedTensorGrid, /):
        if not isinstance(grid, PreparedTensorGrid):
            raise TypeError("Structured cochain bridge requires PreparedTensorGrid.")
        dimension = len(grid.shape)
        if dimension not in (1, 2, 3):
            raise ValueError(
                "Structured cochain bridge supports dimensions one through three."
            )
        orientation_values = tuple(
            tuple(combinations(range(dimension), degree))
            for degree in range(dimension + 1)
        )
        shapes = []
        offsets = []
        entity_sets = []
        coordinates = []
        primal_measures = []
        dual_measures = []
        boundary_masks = []
        index_maps = []
        for degree, degree_orientations in enumerate(orientation_values):
            degree_shapes = []
            degree_offsets = []
            coordinate_parts = []
            primal_parts = []
            dual_parts = []
            boundary_parts = []
            index_map = {}
            offset = 0
            for orientation in degree_orientations:
                shape = tuple(
                    grid.structured_axes[axis].interval_centers.size
                    if axis in orientation
                    else grid.structured_axes[axis].point_coordinates.size
                    for axis in range(dimension)
                )
                degree_shapes.append(shape)
                degree_offsets.append(offset)
                mesh = np.meshgrid(
                    *tuple(
                        np.asarray(
                            grid.structured_axes[axis].interval_centers
                            if axis in orientation
                            else grid.structured_axes[axis].point_coordinates
                        )
                        for axis in range(dimension)
                    ),
                    indexing="ij",
                )
                coordinate_parts.append(
                    np.stack(tuple(value.reshape((-1,)) for value in mesh), axis=-1)
                )
                primal = np.ones(shape)
                dual = np.ones(shape)
                boundary = np.zeros(shape, dtype=bool)
                for axis in range(dimension):
                    structured_axis = grid.structured_axes[axis]
                    if axis in orientation:
                        measure = np.asarray(structured_axis.interval_widths)
                    else:
                        measure = np.ones(structured_axis.point_coordinates.shape)
                        dual_measure = np.asarray(structured_axis.point_measures)
                        reshape = [1] * dimension
                        reshape[axis] = dual_measure.size
                        dual = dual * dual_measure.reshape(reshape)
                        if not structured_axis.periodic:
                            lower: list[slice | int] = [slice(None)] * dimension
                            upper: list[slice | int] = [slice(None)] * dimension
                            lower[axis] = 0
                            upper[axis] = shape[axis] - 1
                            boundary[tuple(lower)] = True
                            boundary[tuple(upper)] = True
                    reshape = [1] * dimension
                    reshape[axis] = measure.size
                    primal = primal * measure.reshape(reshape)
                count = int(np.prod(shape))
                for local, index in enumerate(np.ndindex(*shape)):
                    index_map[(orientation, index)] = offset + local
                offset += count
                primal_parts.append(primal.reshape((-1,)))
                dual_parts.append(dual.reshape((-1,)))
                boundary_parts.append(boundary.reshape((-1,)))
            shapes.append(tuple(degree_shapes))
            offsets.append(tuple(degree_offsets))
            count = offset
            entity_sets.append(
                EntitySet(
                    f"structured_{degree}_cells",
                    degree,
                    np.arange(count, dtype=np.int64),
                )
            )
            coordinates.append(jnp.asarray(np.concatenate(coordinate_parts, axis=0)))
            primal_measures.append(jnp.asarray(np.concatenate(primal_parts)))
            dual_measures.append(jnp.asarray(np.concatenate(dual_parts)))
            boundary_masks.append(jnp.asarray(np.concatenate(boundary_parts)))
            index_maps.append(index_map)
        incidences = []
        directional_signs = []
        for degree in range(1, dimension + 1):
            source_indices = []
            target_indices = []
            signs = []
            route_axes = []
            for orientation in orientation_values[degree]:
                shape = shapes[degree][orientation_values[degree].index(orientation)]
                for upper_index in np.ndindex(*shape):
                    target = index_maps[degree][(orientation, upper_index)]
                    for position, axis in enumerate(orientation):
                        lower_orientation = tuple(
                            value for value in orientation if value != axis
                        )
                        lower_index = list(upper_index)
                        lower_index[axis] = upper_index[axis]
                        upper_boundary_index = list(upper_index)
                        lower_shape = shapes[degree - 1][
                            orientation_values[degree - 1].index(lower_orientation)
                        ]
                        upper_boundary_index[axis] = (
                            (upper_index[axis] + 1) % lower_shape[axis]
                            if grid.structured_axes[axis].periodic
                            else upper_index[axis] + 1
                        )
                        orientation_sign = -1.0 if position % 2 else 1.0
                        source_indices.extend(
                            (
                                index_maps[degree - 1][
                                    (lower_orientation, tuple(lower_index))
                                ],
                                index_maps[degree - 1][
                                    (lower_orientation, tuple(upper_boundary_index))
                                ],
                            )
                        )
                        target_indices.extend((target, target))
                        signs.extend((-orientation_sign, orientation_sign))
                        route_axes.extend((axis, axis))
            relation = EdgeRelation(
                np.asarray(source_indices, dtype=np.int32),
                np.asarray(target_indices, dtype=np.int32),
                source_size=entity_sets[degree - 1].count,
                target_size=entity_sets[degree].count,
            )
            incidences.append(
                OrientedIncidence(
                    degree,
                    entity_sets[degree - 1],
                    entity_sets[degree],
                    relation,
                    np.asarray(signs),
                )
            )
            route_axes_array = np.asarray(route_axes, dtype=np.int32)
            signs_array = np.asarray(signs)
            directional_signs.append(
                tuple(
                    jnp.asarray(np.where(route_axes_array == axis, signs_array, 0.0))
                    for axis in range(dimension)
                )
            )
        topology = CellComplexTopology(entity_sets, incidences)
        hodge = tuple(
            dual / primal
            for dual, primal in zip(dual_measures, primal_measures, strict=True)
        )
        cochain = CochainDiscretization(
            topology,
            hodge,
            primal_measures=primal_measures,
            dual_measures=dual_measures,
            boundary_masks=boundary_masks,
            coordinates=coordinates,
            plan_id=canonical_fingerprint(
                {
                    "kind": "structured-cochain-plan",
                    "grid": grid.prepared_id,
                }
            ),
        )
        self.grid = grid
        self.cochain = cochain
        self.orientations = orientation_values
        self.orientation_shapes = tuple(shapes)
        self.orientation_offsets = tuple(offsets)
        self.directional_signs = tuple(directional_signs)
        self.bridge_id = canonical_fingerprint(
            {
                "kind": "structured-cochain-bridge",
                "grid": grid.prepared_id,
                "cochain": cochain.prepared_id,
            }
        )

    @property
    def dimension(self) -> int:
        return len(self.grid.shape)

    def pack(self, degree: int, components: tuple[ArrayLike, ...], /) -> Array:
        degree_ = int(degree)
        if degree_ < 0 or degree_ > self.dimension:
            raise ValueError("Cochain degree is outside the structured dimension.")
        shapes = self.orientation_shapes[degree_]
        if len(components) != len(shapes):
            raise ValueError("One oriented component is required per degree orientation.")
        arrays = []
        for value, shape in zip(components, shapes, strict=True):
            array = jnp.asarray(value)
            if array.shape != shape:
                raise ValueError(f"Oriented cochain component must have shape {shape}.")
            arrays.append(array.reshape((-1,)))
        return arrays[0] if len(arrays) == 1 else jnp.concatenate(tuple(arrays))

    def unpack(self, degree: int, values: ArrayLike, /) -> tuple[Array, ...]:
        degree_ = int(degree)
        value = jnp.asarray(values)
        expected = self.cochain.cell_counts[degree_]
        if value.shape != (expected,):
            raise ValueError(f"Degree-{degree_} cochain must have shape ({expected},).")
        output = []
        shapes = self.orientation_shapes[degree_]
        offsets = self.orientation_offsets[degree_]
        for shape, offset in zip(shapes, offsets, strict=True):
            count = int(np.prod(shape))
            output.append(value[offset : offset + count].reshape(shape))
        return tuple(output)

    def pack_edge_circulation(
        self, components: tuple[ArrayLike, ArrayLike, ArrayLike], /
    ) -> Array:
        """Integrate Cartesian vector components along oriented primal edges."""

        if self.dimension != 3:
            raise ValueError(
                "Edge-circulation packing requires a three-dimensional bridge."
            )
        measures = self.unpack(1, self.cochain.primal_measures[1])
        return self.pack(
            1,
            tuple(
                jnp.asarray(component) * measure
                for component, measure in zip(components, measures, strict=True)
            ),
        )

    def unpack_edge_circulation(self, values: ArrayLike, /) -> tuple[Array, Array, Array]:
        """Recover Cartesian edge-tangent values from integrated circulations."""

        if self.dimension != 3:
            raise ValueError(
                "Edge-circulation unpacking requires a three-dimensional bridge."
            )
        integrated = self.unpack(1, values)
        measures = self.unpack(1, self.cochain.primal_measures[1])
        return tuple(
            value / measure for value, measure in zip(integrated, measures, strict=True)
        )

    def pack_face_flux(
        self, components: tuple[ArrayLike, ArrayLike, ArrayLike], /
    ) -> Array:
        """Integrate ``(Bx, By, Bz)`` through oriented Cartesian primal faces."""

        if self.dimension != 3:
            raise ValueError("Face-flux packing requires a three-dimensional bridge.")
        bx, by, bz = (jnp.asarray(component) for component in components)
        measure_xy, measure_xz, measure_yz = self.unpack(
            2, self.cochain.primal_measures[2]
        )
        return self.pack(
            2,
            (
                bz * measure_xy,
                -by * measure_xz,
                bx * measure_yz,
            ),
        )

    def unpack_face_flux(self, values: ArrayLike, /) -> tuple[Array, Array, Array]:
        """Recover ``(Bx, By, Bz)`` normal-face values from integrated fluxes."""

        if self.dimension != 3:
            raise ValueError("Face-flux unpacking requires a three-dimensional bridge.")
        flux_xy, flux_xz, flux_yz = self.unpack(2, values)
        measure_xy, measure_xz, measure_yz = self.unpack(
            2, self.cochain.primal_measures[2]
        )
        return (
            flux_yz / measure_yz,
            -flux_xz / measure_xz,
            flux_xy / measure_xy,
        )

    def exterior_derivative(
        self,
        degree: int,
        values: ArrayLike,
        /,
        *,
        boundary_policy: CochainBoundaryKind = "absolute",
    ) -> Array:
        return self.cochain.exterior_derivative(
            degree,
            values,
            boundary_policy=boundary_policy,
        )

    def directional_exterior_derivative(
        self,
        degree: int,
        values: ArrayLike,
        axis: int,
        /,
        *,
        boundary_policy: CochainBoundaryKind = "absolute",
    ) -> Array:
        degree_ = int(degree)
        axis_ = int(axis)
        if degree_ < 0 or degree_ >= self.dimension:
            raise ValueError("Directional exterior derivative degree is invalid.")
        if axis_ < 0 or axis_ >= self.dimension:
            raise ValueError("Directional exterior derivative axis is invalid.")
        _, value = self.cochain._values(degree_, values)
        source = jnp.where(
            self.cochain.active_mask(degree_, boundary_policy),
            value,
            0,
        )
        incidence = self.cochain.topology.incidences[degree_]
        operator = SparseLinearMap(
            incidence.relation,
            self.directional_signs[degree_][axis_],
            operator_id=canonical_fingerprint(
                {
                    "kind": "directional-exterior-derivative",
                    "bridge": self.bridge_id,
                    "degree": degree_,
                    "axis": axis_,
                }
            ),
        )
        output = operator.mv(source)
        return jnp.where(
            self.cochain.active_mask(degree_ + 1, boundary_policy),
            output,
            0,
        )

    def codifferential(
        self,
        degree: int,
        values: ArrayLike,
        /,
        *,
        boundary_policy: CochainBoundaryKind = "absolute",
    ) -> Array:
        return self.cochain.codifferential(
            degree,
            values,
            boundary_policy=boundary_policy,
        )

    def directional_codifferential(
        self,
        degree: int,
        values: ArrayLike,
        axis: int,
        /,
        *,
        boundary_policy: CochainBoundaryKind = "absolute",
    ) -> Array:
        degree_ = int(degree)
        axis_ = int(axis)
        if degree_ <= 0 or degree_ > self.dimension:
            raise ValueError("Directional codifferential degree is invalid.")
        if axis_ < 0 or axis_ >= self.dimension:
            raise ValueError("Directional codifferential axis is invalid.")
        _, value = self.cochain._values(degree_, values)
        source = jnp.where(
            self.cochain.active_mask(degree_, boundary_policy),
            value,
            0,
        )
        weighted = self.cochain.apply_hodge(degree_, source)
        incidence = self.cochain.topology.incidences[degree_ - 1]
        operator = SparseLinearMap(
            incidence.relation,
            self.directional_signs[degree_ - 1][axis_],
            operator_id=canonical_fingerprint(
                {
                    "kind": "directional-codifferential-boundary",
                    "bridge": self.bridge_id,
                    "degree": degree_,
                    "axis": axis_,
                }
            ),
        )
        output = self.cochain.solve_hodge(
            degree_ - 1,
            operator.transpose_mv(weighted),
        )
        return jnp.where(
            self.cochain.active_mask(degree_ - 1, boundary_policy),
            output,
            0,
        )

    def laplace_de_rham(
        self,
        degree: int,
        values: ArrayLike,
        /,
        *,
        boundary_policy: CochainBoundaryKind = "absolute",
    ) -> Array:
        return self.cochain.laplace_de_rham(
            degree,
            values,
            boundary_policy=boundary_policy,
        )


__all__ = ["StructuredCochainBridge"]
