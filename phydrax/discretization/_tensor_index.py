#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from itertools import product
from math import prod

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._axis import AxisDiscretization, TensorGridPlan
from ._measure import DiscreteMeasure
from ._tensor_entities import AxisEntityKind, StructuredAxis
from ._tensor_support import _fraction, GridLocation
from ._topology import TensorTopology


class TensorIndexLayout(StrictModule, NonTrainableState):
    """Factorized tensor-entity addressing without dense tensor arrays."""

    axis_names: tuple[str, ...] = eqx.field(static=True)
    axis_entities: tuple[AxisEntityKind, ...] = eqx.field(static=True)
    shape: tuple[int, ...] = eqx.field(static=True)
    strides: tuple[int, ...] = eqx.field(static=True)
    offsets: tuple[Fraction, ...] = eqx.field(static=True)
    coordinates_by_axis: tuple[Array, ...]
    measures_by_axis: tuple[Array, ...]
    lower_boundary_present: tuple[bool, ...] = eqx.field(static=True)
    upper_boundary_present: tuple[bool, ...] = eqx.field(static=True)
    location_id: str = eqx.field(static=True)
    entity_set_id: str = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        axis_names: Sequence[str],
        axes: Sequence[StructuredAxis],
        axis_entities: Sequence[AxisEntityKind],
        /,
    ) -> None:
        names = tuple(str(name) for name in axis_names)
        axes_ = tuple(axes)
        entities = tuple(axis_entities)
        if (
            not names
            or len(axes_) != len(names)
            or len(entities) != len(names)
            or any(entity not in ("point", "interval") for entity in entities)
        ):
            raise ValueError("Tensor index-layout factors must align with axes.")
        shape = tuple(
            axis.count(entity) for axis, entity in zip(axes_, entities, strict=True)
        )
        strides = tuple(prod(shape[axis + 1 :]) for axis in range(len(shape)))
        offsets = tuple(
            Fraction(0, 1) if entity == axis.primary_entity else Fraction(1, 2)
            for axis, entity in zip(axes_, entities, strict=True)
        )
        coordinates = tuple(
            axis.coordinates(entity) for axis, entity in zip(axes_, entities, strict=True)
        )
        measures = tuple(
            axis.measure(entity) for axis, entity in zip(axes_, entities, strict=True)
        )
        lower = tuple(
            entity == "point"
            and (
                axis.lower_endpoint_included
                or (axis.primary_entity == "interval" and not axis.periodic)
            )
            for axis, entity in zip(axes_, entities, strict=True)
        )
        upper = tuple(
            entity == "point"
            and (
                axis.upper_endpoint_included
                or (axis.primary_entity == "interval" and not axis.periodic)
            )
            for axis, entity in zip(axes_, entities, strict=True)
        )
        location_id = canonical_fingerprint(
            {
                "kind": "tensor-entity-location",
                "axis_names": list(names),
                "axis_entities": list(entities),
                "offsets": [[value.numerator, value.denominator] for value in offsets],
            }
        )
        entity_set_id = canonical_fingerprint(
            {
                "kind": "tensor-entity-set",
                "axes": [axis.axis_id for axis in axes_],
                "entities": list(entities),
                "shape": list(shape),
            }
        )
        self.axis_names = names
        self.axis_entities = entities
        self.shape = shape
        self.strides = strides
        self.offsets = offsets
        self.coordinates_by_axis = coordinates
        self.measures_by_axis = measures
        self.lower_boundary_present = lower
        self.upper_boundary_present = upper
        self.location_id = location_id
        self.entity_set_id = entity_set_id
        self.layout_id = canonical_fingerprint(
            {
                "kind": "tensor-entity-layout",
                "entity_set": entity_set_id,
                "location": location_id,
                "measure_shape": list(shape),
            }
        )

    @property
    def size(self) -> int:
        return prod(self.shape)

    def integer_coordinates(self, flat_indices: ArrayLike, /) -> tuple[Array, Array]:
        indices = jnp.asarray(flat_indices)
        if not jnp.issubdtype(indices.dtype, jnp.integer):
            raise TypeError("flat_indices must have an integer dtype.")
        supported = (indices >= 0) & (indices < self.size)
        safe = jnp.clip(indices, 0, self.size - 1)
        coordinates = jnp.stack(
            tuple(
                (safe // stride) % size
                for stride, size in zip(self.strides, self.shape, strict=True)
            ),
            axis=-1,
        ).astype(jnp.int32)
        return coordinates, supported

    def flat_indices(self, integer_coordinates: ArrayLike, /) -> tuple[Array, Array]:
        coordinates = jnp.asarray(integer_coordinates)
        if not jnp.issubdtype(coordinates.dtype, jnp.integer):
            raise TypeError("integer_coordinates must have an integer dtype.")
        if coordinates.ndim < 1 or coordinates.shape[-1] != len(self.shape):
            raise ValueError(
                "integer_coordinates must end with the tensor dimension "
                f"{len(self.shape)}."
            )
        supported = jnp.ones(coordinates.shape[:-1], dtype=bool)
        safe_axes = []
        for axis, size in enumerate(self.shape):
            value = coordinates[..., axis]
            supported = supported & (value >= 0) & (value < size)
            safe_axes.append(jnp.clip(value, 0, size - 1))
        flat = sum(
            value * stride for value, stride in zip(safe_axes, self.strides, strict=True)
        )
        return flat.astype(jnp.int32), supported

    def coordinates_at(self, flat_indices: ArrayLike, /) -> tuple[Array, Array]:
        integer, supported = self.integer_coordinates(flat_indices)
        coordinates = jnp.stack(
            tuple(
                axis_coordinates[integer[..., axis]]
                for axis, axis_coordinates in enumerate(self.coordinates_by_axis)
            ),
            axis=-1,
        )
        return coordinates, supported

    def measure_at(self, flat_indices: ArrayLike, /) -> tuple[Array, Array]:
        integer, supported = self.integer_coordinates(flat_indices)
        dtype = jnp.result_type(*tuple(value.dtype for value in self.measures_by_axis))
        measure = jnp.ones(integer.shape[:-1], dtype=dtype)
        for axis, factors in enumerate(self.measures_by_axis):
            measure = measure * factors[integer[..., axis]]
        return jnp.where(
            supported, measure, jnp.zeros((), dtype=measure.dtype)
        ), supported

    def boundary_at(
        self,
        flat_indices: ArrayLike,
        axis: str,
        side: str,
        /,
    ) -> tuple[Array, Array]:
        if axis not in self.axis_names:
            raise ValueError(f"Unknown tensor-grid axis {axis!r}.")
        if side not in ("lower", "upper"):
            raise ValueError("side must be 'lower' or 'upper'.")
        integer, supported = self.integer_coordinates(flat_indices)
        dimension = self.axis_names.index(axis)
        present = (
            self.lower_boundary_present[dimension]
            if side == "lower"
            else self.upper_boundary_present[dimension]
        )
        boundary_index = 0 if side == "lower" else self.shape[dimension] - 1
        selected = supported & present & (integer[..., dimension] == boundary_index)
        return selected, supported

    def materialize_coordinates(self, /) -> Array:
        mesh = jnp.meshgrid(*self.coordinates_by_axis, indexing="ij")
        return jnp.stack(mesh, axis=-1)

    def materialize_measure(self, /) -> Array:
        measure = jnp.ones(self.shape)
        for axis, factors in enumerate(self.measures_by_axis):
            reshape = [1] * len(self.shape)
            reshape[axis] = self.shape[axis]
            measure = measure * factors.reshape(reshape)
        return measure

    def materialize_boundary(self, axis: str, side: str, /) -> Array:
        flat = jnp.arange(self.size, dtype=jnp.int32)
        selected, _ = self.boundary_at(flat, axis, side)
        return selected.reshape(self.shape)


class TensorIndexMeasure(StrictModule, NonTrainableState):
    """Factorized physical measure for one tensor index layout."""

    layout: TensorIndexLayout
    support_id: str = eqx.field(static=True)
    measure_id: str = eqx.field(static=True)

    def __init__(self, layout: TensorIndexLayout, support_id: str, /) -> None:
        if not isinstance(layout, TensorIndexLayout):
            raise TypeError("layout must be a TensorIndexLayout.")
        support = str(support_id)
        if not support:
            raise ValueError("support_id must be non-empty.")
        self.layout = layout
        self.support_id = support
        self.measure_id = canonical_fingerprint(
            {
                "kind": "tensor-index-measure",
                "support": support,
                "entity_set": layout.entity_set_id,
                "layout": layout.layout_id,
            }
        )

    @property
    def total_mass(self) -> Array:
        value = jnp.asarray(1.0)
        for factors in self.layout.measures_by_axis:
            value = value * jnp.sum(factors)
        return value

    def weights_at(self, flat_indices: ArrayLike, /) -> tuple[Array, Array]:
        return self.layout.measure_at(flat_indices)

    def materialize(self, /) -> DiscreteMeasure:
        return DiscreteMeasure(
            "tensor-" + "-".join(self.layout.axis_entities),
            self.support_id,
            self.layout.entity_set_id,
            self.layout.materialize_measure().reshape((-1,)),
            normalization="physical",
        )


class PreparedTensorIndexSpace(StrictModule, NonTrainableState):
    """Virtual tensor grid storing axes and indexed layout metadata only."""

    axes: tuple[AxisDiscretization, ...]
    structured_axes: tuple[StructuredAxis, ...]
    entity_layouts: tuple[TensorIndexLayout, ...]
    primary_entity_layout: TensorIndexLayout
    measures: tuple[TensorIndexMeasure, ...]
    measure: TensorIndexMeasure
    centered_location: GridLocation
    topology: TensorTopology
    axis_names: tuple[str, ...] = eqx.field(static=True)
    periodic_axes: tuple[bool, ...] = eqx.field(static=True)
    shape: tuple[int, ...] = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        axes: Sequence[AxisDiscretization],
        /,
        *,
        axis_names: Sequence[str] | None = None,
        plan_id: str | None = None,
    ) -> None:
        axes_ = tuple(axes)
        if not axes_ or not all(isinstance(axis, AxisDiscretization) for axis in axes_):
            raise TypeError("axes must contain one or more AxisDiscretization values.")
        names = (
            tuple(f"axis{index}" for index in range(len(axes_)))
            if axis_names is None
            else tuple(str(name) for name in axis_names)
        )
        if (
            len(names) != len(axes_)
            or any(not name for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError(
                "axis_names must contain one unique non-empty name per axis."
            )
        structured = tuple(StructuredAxis(axis) for axis in axes_)
        layouts = tuple(
            TensorIndexLayout(names, structured, entities)
            for entities in product(("point", "interval"), repeat=len(names))
        )
        primary_kinds = tuple(axis.primary_entity for axis in structured)
        primary = next(
            layout for layout in layouts if layout.axis_entities == primary_kinds
        )
        periodic = tuple(bool(axis.periodic) for axis in axes_)
        topology = TensorTopology(names, primary.shape, periodic=periodic)
        embedding = canonical_fingerprint(
            {
                "kind": "structured-tensor-embedding",
                "axis_names": list(names),
                "structured_axes": [axis.axis_id for axis in structured],
            }
        )
        support_id = canonical_fingerprint(
            {
                "kind": "tensor-index-support",
                "topology": topology.topology_id,
                "embedding": embedding,
            }
        )
        measures = tuple(TensorIndexMeasure(layout, support_id) for layout in layouts)
        measure_by_layout = {
            layout.layout_id: measure
            for layout, measure in zip(layouts, measures, strict=True)
        }
        identifier = plan_id or canonical_fingerprint(
            {
                "kind": "tensor-index-space-plan",
                "axis_names": list(names),
                "primary_entities": list(primary_kinds),
                "shape": list(primary.shape),
                "periodic": list(periodic),
            }
        )
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-tensor-index-space",
                "plan": identifier,
                "support": support_id,
                "entity_layouts": [layout.layout_id for layout in layouts],
            }
        )
        self.axes = axes_
        self.structured_axes = structured
        self.entity_layouts = layouts
        self.primary_entity_layout = primary
        self.measures = measures
        self.measure = measure_by_layout[primary.layout_id]
        self.centered_location = GridLocation(
            names, primary.offsets, location_id=primary.location_id
        )
        self.topology = topology
        self.axis_names = names
        self.periodic_axes = periodic
        self.shape = primary.shape
        self.support_id = support_id
        self.plan_id = str(identifier)
        self.prepared_id = prepared_id

    @classmethod
    def from_plan(
        cls,
        plan: TensorGridPlan,
        bounds: ArrayLike,
        /,
    ) -> PreparedTensorIndexSpace:
        if not isinstance(plan, TensorGridPlan):
            raise TypeError("plan must be a TensorGridPlan.")
        limits = jnp.asarray(bounds, dtype=float)
        if limits.shape != (2, len(plan.axes)):
            raise ValueError(
                f"bounds must have shape {(2, len(plan.axes))}; got {limits.shape}."
            )
        return cls(
            tuple(
                axis.materialize(limits[0, index], limits[1, index])
                for index, axis in enumerate(plan.axes)
            ),
            axis_names=plan.axis_names,
            plan_id=plan.plan_id,
        )

    @property
    def size(self) -> int:
        return prod(self.shape)

    @property
    def stored_axis_values(self) -> int:
        return sum(
            int(coordinates.size + measures.size)
            for coordinates, measures in zip(
                self.primary_entity_layout.coordinates_by_axis,
                self.primary_entity_layout.measures_by_axis,
                strict=True,
            )
        )

    def entity_layout(
        self, axis_entities: Sequence[AxisEntityKind], /
    ) -> TensorIndexLayout:
        entities = tuple(axis_entities)
        for layout in self.entity_layouts:
            if layout.axis_entities == entities:
                return layout
        raise KeyError(f"Unknown tensor index layout {entities!r}.")

    def cells(self, /) -> TensorIndexLayout:
        return self.entity_layout(("interval",) * len(self.axis_names))

    def vertices(self, /) -> TensorIndexLayout:
        return self.entity_layout(("point",) * len(self.axis_names))

    def faces(self, axis: str, /) -> TensorIndexLayout:
        if axis not in self.axis_names:
            raise ValueError(f"Unknown face axis {axis!r}.")
        entities: list[AxisEntityKind] = ["interval"] * len(self.axis_names)
        entities[self.axis_names.index(axis)] = "point"
        return self.entity_layout(entities)

    def location(
        self,
        offsets: Sequence[Fraction | int | tuple[int, int]],
        /,
    ) -> GridLocation:
        values = tuple(_fraction(value) for value in offsets)
        for layout in self.entity_layouts:
            if layout.offsets == values:
                return GridLocation(
                    self.axis_names, values, location_id=layout.location_id
                )
        return GridLocation(self.axis_names, values)

    def layout_at(self, location: GridLocation, /) -> TensorIndexLayout:
        if (
            not isinstance(location, GridLocation)
            or location.axis_names != self.axis_names
        ):
            raise ValueError("Grid location does not belong to this tensor index space.")
        for layout in self.entity_layouts:
            if (
                layout.location_id == location.location_id
                or layout.offsets == location.offsets
            ):
                return layout
        raise ValueError("Grid location does not resolve to a tensor index layout.")

    def measure_for(self, layout: TensorIndexLayout, /) -> TensorIndexMeasure:
        for candidate, measure in zip(self.entity_layouts, self.measures, strict=True):
            if candidate.layout_id == layout.layout_id:
                return measure
        raise KeyError(f"Unknown tensor index measure {layout.layout_id!r}.")


__all__ = [
    "PreparedTensorIndexSpace",
    "TensorIndexLayout",
    "TensorIndexMeasure",
]
