#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from itertools import combinations
from math import prod
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..exterior._complex import ComplexBoundary
from ..exterior._form_type import FormTwist
from ..linalg import (
    AbstractLinearOperator,
    ArraySpace,
    HilbertComplex,
    HodgeLaplacianPart,
    OperatorCapabilities,
    OperatorProperties,
)
from ..typing import checked
from ._cell_complex import cubical_cell_complex, CubicalCellComplex
from ._cell_de_rham import AbstractCellDeRhamComplex
from ._cochain import CochainDiscretization, DiagonalHodge
from ._tensor_support import PreparedTensorGrid
from ._topology import CellComplexTopology


@final
class StructuredCochainResourcePolicy(StrictModule):
    """Host-preparation limits for combinatorial cubical cochains."""

    maximum_entities: int = eqx.field(static=True)
    maximum_incidence_routes: int = eqx.field(static=True)
    maximum_coordinate_values: int = eqx.field(static=True)
    maximum_preparation_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_entities: int = 5_000_000,
        maximum_incidence_routes: int = 20_000_000,
        maximum_coordinate_values: int = 20_000_000,
        maximum_preparation_bytes: int = 1 << 30,
    ) -> None:
        values = (
            maximum_entities,
            maximum_incidence_routes,
            maximum_coordinate_values,
            maximum_preparation_bytes,
        )
        if any(value <= 0 for value in values):
            raise ValueError("Structured cochain resource limits must be positive.")
        self.maximum_entities = maximum_entities
        self.maximum_incidence_routes = maximum_incidence_routes
        self.maximum_coordinate_values = maximum_coordinate_values
        self.maximum_preparation_bytes = maximum_preparation_bytes


@final
class StructuredDifferentialOperator(AbstractLinearOperator):
    """Prepared cubical incidence action using tensor slices rather than routes."""

    source: ArraySpace
    target: ArraySpace
    source_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    target_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    source_offsets: tuple[int, ...] = eqx.field(static=True)
    target_offsets: tuple[int, ...] = eqx.field(static=True)
    terms: tuple[tuple[tuple[int, int, int], ...], ...] = eqx.field(static=True)
    periodic: tuple[bool, ...] = eqx.field(static=True)
    axis: int | None = eqx.field(static=True)

    def __init__(
        self,
        product: CubicalCellComplex,
        degree: int,
        /,
        *,
        source: ArraySpace,
        target: ArraySpace,
        axis: int | None = None,
    ) -> None:
        if degree < 0 or degree >= len(product.shape):
            raise ValueError("Structured differential degree is invalid.")
        if axis is not None and (axis < 0 or axis >= len(product.shape)):
            raise ValueError("Structured differential axis is invalid.")
        source_orientations = product.orientations[degree]
        targets = product.orientations[degree + 1]
        terms = tuple(
            tuple(
                (
                    source_orientations.index(
                        tuple(a for a in orientation if a != direction)
                    ),
                    direction,
                    -1 if position % 2 else 1,
                )
                for position, direction in enumerate(orientation)
                if axis is None or axis == direction
            )
            for orientation in targets
        )
        source_shapes = product.orientation_shapes[degree]
        target_shapes = product.orientation_shapes[degree + 1]
        if source.shape != (sum(prod(shape) for shape in source_shapes),):
            raise ValueError("Structured differential source shape is invalid.")
        if target.shape != (sum(prod(shape) for shape in target_shapes),):
            raise ValueError("Structured differential target shape is invalid.")
        self.source = source
        self.target = target
        self.source_shapes = source_shapes
        self.target_shapes = target_shapes
        self.source_offsets = product.orientation_offsets[degree]
        self.target_offsets = product.orientation_offsets[degree + 1]
        self.terms = terms
        self.periodic = product.periodic
        self.axis = axis
        self.batch_shape = ()
        self.properties = OperatorProperties()
        self.capabilities = OperatorCapabilities(
            transpose=True, adjoint=True, materialize=True
        )
        self.operator_id = canonical_fingerprint(
            {
                "kind": "structured-differential",
                "topology": product.topology.topology_id,
                "degree": degree,
                "axis": axis,
            }
        )

    def mv(self, vector: ArrayLike, /) -> Array:
        value = self.source.validate(jnp.asarray(vector))
        components = tuple(
            value[offset : offset + prod(shape)].reshape(shape)
            for shape, offset in zip(self.source_shapes, self.source_offsets, strict=True)
        )
        output = []
        for shape, terms in zip(self.target_shapes, self.terms, strict=True):
            component = jnp.zeros(shape, dtype=value.dtype)
            for block, axis, sign in terms:
                source = components[block]
                if self.periodic[axis]:
                    difference = jnp.roll(source, -1, axis=axis) - source
                else:
                    lower = [slice(None)] * source.ndim
                    upper = [slice(None)] * source.ndim
                    lower[axis] = slice(0, -1)
                    upper[axis] = slice(1, None)
                    difference = source[tuple(upper)] - source[tuple(lower)]
                component = component + sign * difference
            output.append(component.reshape((-1,)))
        return jnp.concatenate(tuple(output))

    def transpose_mv(self, vector: ArrayLike, /) -> Array:
        value = self.target.validate(jnp.asarray(vector))
        output = [jnp.zeros(shape, dtype=value.dtype) for shape in self.source_shapes]
        for shape, offset, terms in zip(
            self.target_shapes, self.target_offsets, self.terms, strict=True
        ):
            component = value[offset : offset + prod(shape)].reshape(shape)
            for block, axis, sign in terms:
                if self.periodic[axis]:
                    difference = jnp.roll(component, 1, axis=axis) - component
                else:
                    lower = [(0, 0)] * component.ndim
                    upper = [(0, 0)] * component.ndim
                    lower[axis] = (0, 1)
                    upper[axis] = (1, 0)
                    difference = jnp.pad(component, upper) - jnp.pad(component, lower)
                output[block] = output[block] + sign * difference
        return jnp.concatenate(tuple(component.reshape((-1,)) for component in output))

    def adjoint_mv(self, vector: ArrayLike, /) -> Array:
        return self.source.inverse_riesz(self.transpose_mv(self.target.riesz(vector)))

    def _materialize(self, /) -> Array:
        return jax.vmap(self.mv)(jnp.eye(self.source.size, dtype=self.source.dtype)).T


@final
class StructuredCochainBridge(AbstractCellDeRhamComplex, NonTrainableState):
    """Cartesian tensor entities assembled into one oriented cubical cochain complex."""

    grid: PreparedTensorGrid
    cochain: CochainDiscretization
    product: CubicalCellComplex
    orientations: tuple[tuple[tuple[int, ...], ...], ...] = eqx.field(static=True)
    orientation_shapes: tuple[tuple[tuple[int, ...], ...], ...] = eqx.field(static=True)
    orientation_offsets: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    directional_differentials: tuple[tuple[StructuredDifferentialOperator, ...], ...]
    bridge_id: str = eqx.field(static=True)
    resource_policy: StructuredCochainResourcePolicy = eqx.field(static=True)
    entity_counts: tuple[int, ...] = eqx.field(static=True)
    incidence_route_count: int = eqx.field(static=True)
    preparation_bytes: int = eqx.field(static=True)

    @checked
    def __init__(
        self,
        grid: PreparedTensorGrid,
        /,
        *,
        resources: StructuredCochainResourcePolicy | None = None,
    ) -> None:
        resource_policy = (
            StructuredCochainResourcePolicy() if resources is None else resources
        )
        if not isinstance(resource_policy, StructuredCochainResourcePolicy):
            raise TypeError("resources must be a StructuredCochainResourcePolicy.")
        dimension = len(grid.shape)
        if dimension <= 0:
            raise ValueError("Structured cochain dimension must be positive.")
        orientation_values = tuple(
            tuple(combinations(range(dimension), degree))
            for degree in range(dimension + 1)
        )
        orientation_shape_values = tuple(
            tuple(
                tuple(
                    grid.structured_axes[axis].interval_centers.size
                    if axis in orientation
                    else grid.structured_axes[axis].point_coordinates.size
                    for axis in range(dimension)
                )
                for orientation in degree_orientations
            )
            for degree_orientations in orientation_values
        )
        entity_counts = tuple(
            sum(int(np.prod(shape)) for shape in degree_shapes)
            for degree_shapes in orientation_shape_values
        )
        total_entities = sum(entity_counts)
        incidence_route_count = sum(
            2 * degree * entity_counts[degree] for degree in range(1, dimension + 1)
        )
        coordinate_values = dimension * total_entities
        preparation_bytes = (
            coordinate_values
            * (np.dtype(np.float64).itemsize + np.dtype(np.int32).itemsize)
            + total_entities
            * (3 * np.dtype(np.float64).itemsize + np.dtype(np.int64).itemsize + 1)
            + incidence_route_count
            * (3 * np.dtype(np.int32).itemsize + np.dtype(np.float64).itemsize)
        )
        if total_entities > resource_policy.maximum_entities:
            raise ValueError(
                "Structured cochain exceeds maximum_entities: "
                f"required {total_entities}, allowed {resource_policy.maximum_entities}."
            )
        if incidence_route_count > resource_policy.maximum_incidence_routes:
            raise ValueError(
                "Structured cochain exceeds maximum_incidence_routes: "
                f"required {incidence_route_count}, "
                f"allowed {resource_policy.maximum_incidence_routes}."
            )
        if coordinate_values > resource_policy.maximum_coordinate_values:
            raise ValueError(
                "Structured cochain exceeds maximum_coordinate_values: "
                f"required {coordinate_values}, "
                f"allowed {resource_policy.maximum_coordinate_values}."
            )
        if preparation_bytes > resource_policy.maximum_preparation_bytes:
            raise ValueError(
                "Structured cochain exceeds maximum_preparation_bytes: "
                f"required {preparation_bytes}, "
                f"allowed {resource_policy.maximum_preparation_bytes}."
            )
        product = cubical_cell_complex(
            tuple(axis.point_coordinates.size for axis in grid.structured_axes),
            periodic=tuple(axis.periodic for axis in grid.structured_axes),
        )
        coordinates = []
        primal_measures = []
        dual_measures = []
        boundary_masks = []
        for degree, degree_orientations in enumerate(orientation_values):
            coordinate_parts = []
            primal_parts = []
            dual_parts = []
            boundary_parts = []
            for orientation in degree_orientations:
                shape = tuple(
                    grid.structured_axes[axis].interval_centers.size
                    if axis in orientation
                    else grid.structured_axes[axis].point_coordinates.size
                    for axis in range(dimension)
                )
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
                primal = np.ones(shape, dtype=np.float64)
                dual = np.ones(shape, dtype=np.float64)
                boundary = np.zeros(shape, dtype=np.bool_)
                for axis in range(dimension):
                    structured_axis = grid.structured_axes[axis]
                    if axis in orientation:
                        measure = np.asarray(structured_axis.interval_widths)
                    else:
                        measure = np.ones(
                            structured_axis.point_coordinates.shape, dtype=np.float64
                        )
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
                primal_parts.append(primal.reshape((-1,)))
                dual_parts.append(dual.reshape((-1,)))
                boundary_parts.append(boundary.reshape((-1,)))
            coordinates.append(jnp.asarray(np.concatenate(coordinate_parts, axis=0)))
            primal_measures.append(jnp.asarray(np.concatenate(primal_parts)))
            dual_measures.append(jnp.asarray(np.concatenate(dual_parts)))
            boundary_masks.append(jnp.asarray(np.concatenate(boundary_parts)))
        hodge = tuple(
            DiagonalHodge(dual / primal)
            for dual, primal in zip(dual_measures, primal_measures, strict=True)
        )
        spaces = tuple(
            ArraySpace((count,), dtype=coordinates[degree].dtype)
            for degree, count in enumerate(entity_counts)
        )
        differentials = tuple(
            StructuredDifferentialOperator(
                product, degree, source=spaces[degree], target=spaces[degree + 1]
            )
            for degree in range(dimension)
        )
        directional_differentials = tuple(
            tuple(
                StructuredDifferentialOperator(
                    product,
                    degree,
                    source=spaces[degree],
                    target=spaces[degree + 1],
                    axis=axis,
                )
                for axis in range(dimension)
            )
            for degree in range(dimension)
        )
        cochain = CochainDiscretization(
            product.topology,
            hodge,
            primal_measures=primal_measures,
            dual_measures=dual_measures,
            boundary_masks=boundary_masks,
            coordinates=coordinates,
            differentials=differentials,
        )
        self.grid = grid
        self.cochain = cochain
        self.product = product
        self.orientations = orientation_values
        self.orientation_shapes = product.orientation_shapes
        self.orientation_offsets = product.orientation_offsets
        self.directional_differentials = directional_differentials
        self.resource_policy = resource_policy
        self.entity_counts = entity_counts
        self.incidence_route_count = incidence_route_count
        self.preparation_bytes = preparation_bytes
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

    @property
    def topology(self) -> CellComplexTopology:
        return self.cochain.topology

    @property
    def boundary_masks(self) -> tuple[Array, ...]:
        return self.cochain.boundary_masks

    @property
    def realization_id(self) -> str:
        return self.bridge_id

    @property
    def primal_twist(self) -> FormTwist:
        return self.cochain.primal_twist

    def hilbert_complex(
        self, /, *, boundary: ComplexBoundary = "absolute"
    ) -> HilbertComplex:
        return self.cochain.hilbert_complex(boundary=boundary)

    def active_indices(
        self, degree: int, /, *, boundary: ComplexBoundary = "absolute"
    ) -> Array:
        return self.cochain.active_indices(degree, boundary=boundary)

    def hodge_star(self, degree: int, values: ArrayLike, /) -> Array:
        return self.cochain.hodge_star(degree, values)

    def inverse_hodge_star(self, degree: int, values: ArrayLike, /) -> Array:
        return self.cochain.inverse_hodge_star(degree, values)

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
        if degree_ < 0 or degree_ > self.dimension:
            raise ValueError("Cochain degree is outside the structured dimension.")
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

    def pack_normal_flux(self, components: tuple[ArrayLike, ...], /) -> Array:
        """Integrate a Cartesian vector through oriented codimension-one entities."""
        if len(components) != self.dimension:
            raise ValueError("One normal-flux component is required per dimension.")
        values = tuple(jnp.asarray(component) for component in components)
        if self.dimension == 1:
            return self.pack(0, values)
        degree = self.dimension - 1
        measures = self.unpack(degree, self.cochain.primal_measures[degree])
        normal_axes = tuple(
            next(axis for axis in range(self.dimension) if axis not in orientation)
            for orientation in self.orientations[degree]
        )
        return self.pack(
            degree,
            tuple(
                (-values[axis] if axis % 2 else values[axis]) * measure
                for axis, measure in zip(normal_axes, measures, strict=True)
            ),
        )

    def unpack_normal_flux(self, values: ArrayLike, /) -> tuple[Array, ...]:
        """Recover Cartesian normal fields from codimension-one flux integrals."""
        if self.dimension == 1:
            return self.unpack(0, values)
        degree = self.dimension - 1
        fluxes = self.unpack(degree, values)
        measures = self.unpack(degree, self.cochain.primal_measures[degree])
        normal_blocks = tuple(
            self.orientations[degree].index(
                tuple(
                    direction for direction in range(self.dimension) if direction != axis
                )
            )
            for axis in range(self.dimension)
        )
        return tuple(
            (-fluxes[block] if axis % 2 else fluxes[block]) / measures[block]
            for axis, block in enumerate(normal_blocks)
        )

    def pack_electromotive(
        self,
        components: tuple[ArrayLike, ...],
        /,
    ) -> Array:
        """Integrate the Faraday electromotive form for two or three dimensions."""
        if self.dimension == 2:
            if len(components) != 1:
                raise ValueError("Planar MHD requires one out-of-plane electromotive.")
            return self.pack(0, components)
        if self.dimension == 3:
            if len(components) != 3:
                raise ValueError("Three-dimensional MHD requires three edge components.")
            return self.pack_edge_circulation(components)
        if components:
            raise ValueError("One-dimensional MHD has no evolved EMF cochain.")
        return jnp.zeros((0,))

    def unpack_electromotive(self, values: ArrayLike, /) -> tuple[Array, ...]:
        if self.dimension == 2:
            return self.unpack(0, values)
        if self.dimension == 3:
            return self.unpack_edge_circulation(values)
        value = jnp.asarray(values)
        if value.shape != (0,):
            raise ValueError("One-dimensional EMF storage must be empty.")
        return ()

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
        integrated_x, integrated_y, integrated_z = self.unpack(1, values)
        measure_x, measure_y, measure_z = self.unpack(1, self.cochain.primal_measures[1])
        return (
            integrated_x / measure_x,
            integrated_y / measure_y,
            integrated_z / measure_z,
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
        boundary: ComplexBoundary = "absolute",
    ) -> Array:
        return self.cochain.exterior_derivative(
            degree,
            values,
            boundary=boundary,
        )

    def directional_exterior_derivative(
        self,
        degree: int,
        values: ArrayLike,
        axis: int,
        /,
        *,
        boundary: ComplexBoundary = "absolute",
    ) -> Array:
        degree_ = int(degree)
        axis_ = int(axis)
        if degree_ < 0 or degree_ >= self.dimension:
            raise ValueError("Directional exterior derivative degree is invalid.")
        if axis_ < 0 or axis_ >= self.dimension:
            raise ValueError("Directional exterior derivative axis is invalid.")
        value = self.cochain._values(degree_, values)
        source = jnp.where(
            self.cochain.active_mask(degree_, boundary),
            value,
            0,
        )
        operator = self.directional_differentials[degree_][axis_]
        output = operator.mv(source)
        return jnp.where(
            self.cochain.active_mask(degree_ + 1, boundary),
            output,
            0,
        )

    def codifferential(
        self,
        degree: int,
        values: ArrayLike,
        /,
        *,
        boundary: ComplexBoundary = "absolute",
    ) -> Array:
        return self.cochain.codifferential(
            degree,
            values,
            boundary=boundary,
        )

    def hodge_laplacian(
        self,
        degree: int,
        values: ArrayLike,
        /,
        *,
        boundary: ComplexBoundary = "absolute",
        part: HodgeLaplacianPart = "complete",
    ) -> Array:
        return self.cochain.hodge_laplacian(degree, values, boundary=boundary, part=part)

    def directional_codifferential(
        self,
        degree: int,
        values: ArrayLike,
        axis: int,
        /,
        *,
        boundary: ComplexBoundary = "absolute",
    ) -> Array:
        degree_ = int(degree)
        axis_ = int(axis)
        if degree_ <= 0 or degree_ > self.dimension:
            raise ValueError("Directional codifferential degree is invalid.")
        if axis_ < 0 or axis_ >= self.dimension:
            raise ValueError("Directional codifferential axis is invalid.")
        value = self.cochain._values(degree_, values)
        source = jnp.where(
            self.cochain.active_mask(degree_, boundary),
            value,
            0,
        )
        weighted = self.cochain.hodge_star(degree_, source)
        operator = self.directional_differentials[degree_ - 1][axis_]
        output = self.cochain.inverse_hodge_star(
            degree_ - 1,
            operator.transpose_mv(weighted),
        )
        return jnp.where(
            self.cochain.active_mask(degree_ - 1, boundary),
            output,
            0,
        )


__all__ = [
    "StructuredCochainResourcePolicy",
    "StructuredDifferentialOperator",
    "StructuredCochainBridge",
]
