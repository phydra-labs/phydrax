#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from itertools import product
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._axis import AxisDiscretization, AxisPrimaryEntity


AxisEntityKind: TypeAlias = Literal["point", "interval"]


class StructuredAxis(StrictModule, NonTrainableState):
    """Point/interval entities and dual measures for one structured axis."""

    bounds: Array
    point_coordinates: Array
    interval_centers: Array
    interval_widths: Array
    point_measures: Array
    primary_entity: AxisPrimaryEntity = eqx.field(static=True)
    periodic: bool = eqx.field(static=True)
    domain_kind: str = eqx.field(static=True)
    domain_id: str = eqx.field(static=True)
    lower_endpoint_included: bool = eqx.field(static=True)
    upper_endpoint_included: bool = eqx.field(static=True)
    axis_id: str = eqx.field(static=True)

    def __init__(self, axis: AxisDiscretization, /):
        if not isinstance(axis, AxisDiscretization):
            raise TypeError("axis must be an AxisDiscretization.")
        nodes = np.asarray(axis.nodes, dtype=float)
        if (
            nodes.size < 1
            or np.any(~np.isfinite(nodes))
            or (nodes.size > 1 and np.any(np.diff(nodes) <= 0.0))
        ):
            raise ValueError(
                "Structured axis nodes must be finite, nonempty, and increasing."
            )
        finite_bounds = axis.domain.finite_bounds
        if finite_bounds is not None:
            bounds = np.asarray(finite_bounds, dtype=float)
        elif axis.periodic:
            raise ValueError("Periodic structured axes require finite bounds.")
        else:
            bounds = np.asarray([nodes[0], nodes[-1]])
        if axis.primary_entity == "point":
            points = nodes
            if axis.periodic:
                period = float(bounds[1] - bounds[0])
                extended = np.concatenate((points, [points[0] + period]))
                widths = np.diff(extended)
                centers = points + 0.5 * widths
                centers = bounds[0] + np.mod(centers - bounds[0], period)
                geometric_measures = 0.5 * (widths + np.roll(widths, 1))
            else:
                if points.size < 2:
                    raise ValueError("Point-primary axes require at least two points.")
                widths = np.diff(points)
                centers = 0.5 * (points[:-1] + points[1:])
                geometric_measures = np.empty_like(points)
                geometric_measures[0] = 0.5 * widths[0]
                geometric_measures[-1] = 0.5 * widths[-1]
                if points.size > 2:
                    geometric_measures[1:-1] = 0.5 * (widths[:-1] + widths[1:])
            point_measures = (
                geometric_measures
                if axis.quad_weights is None
                else np.asarray(axis.quad_weights, dtype=float)
            )
        else:
            centers = nodes
            count = int(nodes.size)
            widths = (
                np.full((count,), float(bounds[1] - bounds[0]) / count)
                if axis.quad_weights is None
                else np.asarray(axis.quad_weights, dtype=float)
            )
            if (
                np.any(~np.isfinite(widths))
                or np.any(widths <= 0.0)
                or not np.isclose(
                    np.sum(widths),
                    bounds[1] - bounds[0],
                    rtol=1e-10,
                    atol=1e-12,
                )
            ):
                raise ValueError(
                    "Interval-primary axis measures must be positive and span bounds."
                )
            edge_coordinates = bounds[0] + np.concatenate(([0.0], np.cumsum(widths)))
            expected_centers = 0.5 * (edge_coordinates[:-1] + edge_coordinates[1:])
            if not np.allclose(centers, expected_centers, rtol=1e-10, atol=1e-12):
                raise ValueError(
                    "Interval-primary axis nodes must be cell centers for their measures."
                )
            points = edge_coordinates[:-1] if axis.periodic else edge_coordinates
            if axis.periodic:
                point_measures = 0.5 * (widths + np.roll(widths, 1))
            else:
                point_measures = np.empty((count + 1,), dtype=float)
                point_measures[0] = 0.5 * widths[0]
                point_measures[-1] = 0.5 * widths[-1]
                if count > 1:
                    point_measures[1:-1] = 0.5 * (widths[:-1] + widths[1:])
        if (
            np.any(~np.isfinite(widths))
            or np.any(widths <= 0.0)
            or np.any(~np.isfinite(point_measures))
            or np.any(point_measures < 0.0)
            or not np.any(point_measures > 0.0)
        ):
            raise ValueError(
                "Structured entity measures must be finite and non-negative "
                "with positive total support."
            )
        self.bounds = jnp.asarray(bounds)
        self.point_coordinates = jnp.asarray(points)
        self.interval_centers = jnp.asarray(centers)
        self.interval_widths = jnp.asarray(widths)
        self.point_measures = jnp.asarray(point_measures)
        self.primary_entity = axis.primary_entity
        self.periodic = axis.periodic
        self.domain_kind = axis.domain.kind
        self.domain_id = axis.domain.domain_id
        self.lower_endpoint_included = axis.lower_endpoint_included
        self.upper_endpoint_included = axis.upper_endpoint_included
        self.axis_id = canonical_fingerprint(
            {
                "kind": "structured-axis",
                "primary_entity": axis.primary_entity,
                "domain": axis.domain.domain_id,
                "lower_endpoint_included": axis.lower_endpoint_included,
                "upper_endpoint_included": axis.upper_endpoint_included,
                "bounds": array_tree_fingerprint(bounds),
                "points": array_tree_fingerprint(points),
                "interval_centers": array_tree_fingerprint(centers),
                "interval_widths": array_tree_fingerprint(widths),
                "point_measures": array_tree_fingerprint(point_measures),
            }
        )

    def count(self, kind: AxisEntityKind, /) -> int:
        if kind == "point":
            return int(self.point_coordinates.size)
        if kind == "interval":
            return int(self.interval_centers.size)
        raise ValueError("Unknown axis entity kind.")

    def coordinates(self, kind: AxisEntityKind, /) -> Array:
        if kind == "point":
            return self.point_coordinates
        if kind == "interval":
            return self.interval_centers
        raise ValueError("Unknown axis entity kind.")

    def measure(self, kind: AxisEntityKind, /) -> Array:
        if kind == "point":
            return self.point_measures
        if kind == "interval":
            return self.interval_widths
        raise ValueError("Unknown axis entity kind.")


class TensorEntityLayout(StrictModule, NonTrainableState):
    """Implicit tensor product of point/interval factors with exact shape and measure."""

    axis_names: tuple[str, ...] = eqx.field(static=True)
    axis_entities: tuple[AxisEntityKind, ...] = eqx.field(static=True)
    shape: tuple[int, ...] = eqx.field(static=True)
    offsets: tuple[Fraction, ...] = eqx.field(static=True)
    coordinates_by_axis: tuple[Array, ...]
    measure: Array
    lower_boundary_masks: tuple[Array, ...]
    upper_boundary_masks: tuple[Array, ...]
    location_id: str = eqx.field(static=True)
    entity_set_id: str = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        axis_names: Sequence[str],
        axes: Sequence[StructuredAxis],
        axis_entities: Sequence[AxisEntityKind],
        /,
    ):
        names = tuple(str(name) for name in axis_names)
        axes_ = tuple(axes)
        entities = tuple(axis_entities)
        if (
            not names
            or len(axes_) != len(names)
            or len(entities) != len(names)
            or any(entity not in ("point", "interval") for entity in entities)
        ):
            raise ValueError("Tensor entity layout factors must align with axes.")
        shape = tuple(
            axis.count(entity) for axis, entity in zip(axes_, entities, strict=True)
        )
        offsets = tuple(
            Fraction(0, 1) if entity == axis.primary_entity else Fraction(1, 2)
            for axis, entity in zip(axes_, entities, strict=True)
        )
        coordinates = tuple(
            axis.coordinates(entity) for axis, entity in zip(axes_, entities, strict=True)
        )
        measure = jnp.ones(shape)
        for index, (axis, entity) in enumerate(zip(axes_, entities, strict=True)):
            weights = axis.measure(entity)
            reshape = [1] * len(shape)
            reshape[index] = int(weights.size)
            measure = measure * weights.reshape(reshape)
        lower_masks = []
        upper_masks = []
        for dimension, (axis, entity) in enumerate(zip(axes_, entities, strict=True)):
            lower = jnp.zeros(shape, dtype=bool)
            upper = jnp.zeros(shape, dtype=bool)
            if entity == "point" and axis.lower_endpoint_included:
                lower_index: list[slice | int] = [slice(None)] * len(shape)
                lower_index[dimension] = 0
                lower = lower.at[tuple(lower_index)].set(True)
            if entity == "point" and axis.upper_endpoint_included:
                upper_index: list[slice | int] = [slice(None)] * len(shape)
                upper_index[dimension] = shape[dimension] - 1
                upper = upper.at[tuple(upper_index)].set(True)
            lower_masks.append(lower)
            upper_masks.append(upper)
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
        self.offsets = offsets
        self.coordinates_by_axis = coordinates
        self.measure = measure
        self.lower_boundary_masks = tuple(lower_masks)
        self.upper_boundary_masks = tuple(upper_masks)
        self.location_id = location_id
        self.entity_set_id = entity_set_id
        self.layout_id = canonical_fingerprint(
            {
                "kind": "tensor-entity-layout",
                "entity_set": entity_set_id,
                "location": location_id,
                "measure_shape": list(measure.shape),
            }
        )

    @classmethod
    def cells(
        cls,
        axis_names: Sequence[str],
        axes: Sequence[StructuredAxis],
        /,
    ) -> "TensorEntityLayout":
        return cls(axis_names, axes, ("interval",) * len(tuple(axis_names)))

    @classmethod
    def vertices(
        cls,
        axis_names: Sequence[str],
        axes: Sequence[StructuredAxis],
        /,
    ) -> "TensorEntityLayout":
        return cls(axis_names, axes, ("point",) * len(tuple(axis_names)))

    @classmethod
    def faces(
        cls,
        axis_names: Sequence[str],
        axes: Sequence[StructuredAxis],
        axis: str,
        /,
    ) -> "TensorEntityLayout":
        names = tuple(axis_names)
        axis_ = str(axis)
        if axis_ not in names:
            raise ValueError(f"Unknown face axis {axis_!r}.")
        entities: list[AxisEntityKind] = ["interval"] * len(names)
        entities[names.index(axis_)] = "point"
        return cls(names, axes, entities)


def all_tensor_entity_layouts(
    axis_names: Sequence[str],
    axes: Sequence[StructuredAxis],
    /,
) -> tuple[TensorEntityLayout, ...]:
    names = tuple(axis_names)
    return tuple(
        TensorEntityLayout(names, axes, entities)
        for entities in product(("point", "interval"), repeat=len(names))
    )


__all__ = [
    "AxisEntityKind",
    "StructuredAxis",
    "TensorEntityLayout",
    "all_tensor_entity_layouts",
]
