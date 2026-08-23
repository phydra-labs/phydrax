#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from math import prod

import equinox as eqx
import jax.numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import ArraySpace, DiagonalPairing
from ._axis import AxisDiscretization, broadcasted_grid, TensorGridPlan
from ._measure import DiscreteMeasure
from ._spaces import DiscreteFieldSpace, TensorDofLayout
from ._support import DiscreteSupport
from ._tensor_entities import (
    all_tensor_entity_layouts,
    AxisEntityKind,
    StructuredAxis,
    TensorEntityLayout,
)
from ._topology import TensorTopology


def _fraction(value: Fraction | int | tuple[int, int], /) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, tuple) and len(value) == 2:
        return Fraction(int(value[0]), int(value[1]))
    raise TypeError("Grid locations require Fraction, int, or (numerator, denominator).")


class GridLocation(StrictModule, NonTrainableState):
    """Exact rational offset from one tensor-grid primary entity."""

    axis_names: tuple[str, ...] = eqx.field(static=True)
    offsets: tuple[Fraction, ...] = eqx.field(static=True)
    location_id: str = eqx.field(static=True)

    def __init__(
        self,
        axis_names: Sequence[str],
        offsets: Sequence[Fraction | int | tuple[int, int]],
        /,
        *,
        location_id: str | None = None,
    ):
        names = tuple(str(name) for name in axis_names)
        values = tuple(_fraction(value) for value in offsets)
        if not names or any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("Grid location axis names must be unique and non-empty.")
        if len(values) != len(names):
            raise ValueError("Grid location requires one offset per axis.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "grid-location",
                    "axis_names": list(names),
                    "offsets": [[value.numerator, value.denominator] for value in values],
                }
            )
            if location_id is None
            else str(location_id)
        )
        if not identifier:
            raise ValueError("location_id must be non-empty.")
        self.axis_names = names
        self.offsets = values
        self.location_id = identifier

    @classmethod
    def centered(cls, axis_names: Sequence[str], /) -> "GridLocation":
        names = tuple(axis_names)
        return cls(names, (0,) * len(names))

    @classmethod
    def shifted(
        cls,
        axis_names: Sequence[str],
        axis: str,
        offset: Fraction | int | tuple[int, int] = Fraction(1, 2),
        /,
    ) -> "GridLocation":
        names = tuple(str(name) for name in axis_names)
        axis_ = str(axis)
        if axis_ not in names:
            raise ValueError(f"Unknown tensor-grid axis {axis_!r}.")
        values = [Fraction(0, 1)] * len(names)
        values[names.index(axis_)] = _fraction(offset)
        return cls(names, values)


class PreparedTensorGrid(StrictModule, NonTrainableState):
    """Prepared structured support with exact point/interval entity layouts."""

    axes: tuple[AxisDiscretization, ...]
    structured_axes: tuple[StructuredAxis, ...]
    entity_layouts: tuple[TensorEntityLayout, ...]
    primary_entity_layout: TensorEntityLayout
    axis_names: tuple[str, ...] = eqx.field(static=True)
    topology: TensorTopology
    support: DiscreteSupport
    measures: tuple[DiscreteMeasure, ...]
    measure: DiscreteMeasure
    points: Array
    quadrature_weights: Array
    centered_location: GridLocation
    shape: tuple[int, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        axes: Sequence[AxisDiscretization],
        /,
        *,
        axis_names: Sequence[str] | None = None,
        plan_id: str | None = None,
        embedding_id: str | None = None,
        prepared_id: str | None = None,
    ):
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
        structured_axes = tuple(StructuredAxis(axis) for axis in axes_)
        layouts = all_tensor_entity_layouts(names, structured_axes)
        primary_kinds = tuple(axis.primary_entity for axis in structured_axes)
        primary_layout = next(
            layout for layout in layouts if layout.axis_entities == primary_kinds
        )
        shape = primary_layout.shape
        periodic = tuple(bool(axis.periodic) for axis in axes_)
        topology = TensorTopology(names, shape, periodic=periodic)
        embedding = (
            canonical_fingerprint(
                {
                    "kind": "structured-tensor-embedding",
                    "axis_names": list(names),
                    "structured_axes": [axis.axis_id for axis in structured_axes],
                }
            )
            if embedding_id is None
            else str(embedding_id)
        )
        if not embedding:
            raise ValueError("embedding_id must be non-empty.")
        support = DiscreteSupport(topology, len(axes_), embedding)
        measures = tuple(
            DiscreteMeasure(
                "tensor-" + "-".join(layout.axis_entities),
                support.support_id,
                layout.entity_set_id,
                layout.measure.reshape((-1,)),
                normalization="physical",
            )
            for layout in layouts
        )
        measure_by_layout = {
            layout.layout_id: measure
            for layout, measure in zip(layouts, measures, strict=True)
        }
        primary_measure = measure_by_layout[primary_layout.layout_id]
        plan_identifier = (
            canonical_fingerprint(
                {
                    "kind": "tensor-grid-support-plan",
                    "axis_names": list(names),
                    "primary_entities": list(primary_kinds),
                    "shape": list(shape),
                    "periodic": list(periodic),
                }
            )
            if plan_id is None
            else str(plan_id)
        )
        resolved_prepared_id = (
            canonical_fingerprint(
                {
                    "kind": "prepared-tensor-grid",
                    "plan": plan_identifier,
                    "support": support.support_id,
                    "entity_layouts": [layout.layout_id for layout in layouts],
                    "measures": [measure.measure_id for measure in measures],
                }
            )
            if prepared_id is None
            else str(prepared_id)
        )
        if not plan_identifier or not resolved_prepared_id:
            raise ValueError("Tensor-grid plan and prepared IDs must be non-empty.")
        self.axes = axes_
        self.structured_axes = structured_axes
        self.entity_layouts = layouts
        self.primary_entity_layout = primary_layout
        self.axis_names = names
        self.topology = topology
        self.support = support
        self.measures = measures
        self.measure = primary_measure
        self.points = broadcasted_grid(primary_layout.coordinates_by_axis).reshape(
            (-1, len(axes_))
        )
        self.quadrature_weights = primary_layout.measure
        self.centered_location = GridLocation(
            names,
            primary_layout.offsets,
            location_id=primary_layout.location_id,
        )
        self.shape = shape
        self.plan_id = plan_identifier
        self.prepared_id = resolved_prepared_id

    @classmethod
    def from_plan(
        cls,
        plan: TensorGridPlan,
        bounds: ArrayLike,
        /,
    ) -> "PreparedTensorGrid":
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

    def entity_layout(
        self,
        axis_entities: Sequence[AxisEntityKind],
        /,
    ) -> TensorEntityLayout:
        entities = tuple(axis_entities)
        for layout in self.entity_layouts:
            if layout.axis_entities == entities:
                return layout
        raise KeyError(f"Unknown tensor entity layout {entities!r}.")

    def cells(self, /) -> TensorEntityLayout:
        return self.entity_layout(("interval",) * len(self.axis_names))

    def vertices(self, /) -> TensorEntityLayout:
        return self.entity_layout(("point",) * len(self.axis_names))

    def faces(self, axis: str, /) -> TensorEntityLayout:
        names = self.axis_names
        axis_ = str(axis)
        if axis_ not in names:
            raise ValueError(f"Unknown face axis {axis_!r}.")
        entities: list[AxisEntityKind] = ["interval"] * len(names)
        entities[names.index(axis_)] = "point"
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
                    self.axis_names,
                    values,
                    location_id=layout.location_id,
                )
        return GridLocation(self.axis_names, values)

    def layout_at(self, location: GridLocation, /) -> TensorEntityLayout:
        if (
            not isinstance(location, GridLocation)
            or location.axis_names != self.axis_names
        ):
            raise ValueError("Grid location does not belong to this tensor grid.")
        for layout in self.entity_layouts:
            if (
                layout.location_id == location.location_id
                or layout.offsets == location.offsets
            ):
                return layout
        raise ValueError("Grid location does not resolve to a structured entity layout.")

    def measure_for(self, layout: TensorEntityLayout, /) -> DiscreteMeasure:
        for candidate, measure in zip(self.entity_layouts, self.measures, strict=True):
            if candidate.layout_id == layout.layout_id:
                return measure
        raise KeyError(f"Unknown entity-layout measure {layout.layout_id!r}.")

    def field_space(
        self,
        name: str,
        /,
        *,
        location: GridLocation | None = None,
        entity_layout: TensorEntityLayout | None = None,
        component_shape: Sequence[int] = (),
        dtype: DTypeLike = float,
        representation: str = "point_value",
        conformity: str = "unrestricted",
    ) -> DiscreteFieldSpace:
        if location is not None and entity_layout is not None:
            resolved = self.layout_at(location)
            if resolved.layout_id != entity_layout.layout_id:
                raise ValueError("location and entity_layout do not identify one space.")
            layout_ = resolved
        elif entity_layout is not None:
            if not any(
                candidate.layout_id == entity_layout.layout_id
                for candidate in self.entity_layouts
            ):
                raise ValueError("entity_layout does not belong to this tensor grid.")
            layout_ = entity_layout
        elif location is not None:
            layout_ = self.layout_at(location)
        else:
            layout_ = self.primary_entity_layout
        location_ = GridLocation(
            self.axis_names,
            layout_.offsets,
            location_id=layout_.location_id,
        )
        components = tuple(int(size) for size in component_shape)
        if any(size <= 0 for size in components):
            raise ValueError("component_shape dimensions must be positive.")
        value_shape = layout_.shape + components
        pairing_weights = layout_.measure.reshape(layout_.shape + (1,) * len(components))
        pairing_weights = jnp.broadcast_to(pairing_weights, value_shape).astype(dtype)
        dof_layout = TensorDofLayout(
            self.axis_names,
            layout_.shape,
            component_shape=components,
            location_id=location_.location_id,
        )
        return DiscreteFieldSpace(
            str(name),
            self.support.support_id,
            dof_layout,
            ArraySpace(
                value_shape,
                dtype=dtype,
                pairing=DiagonalPairing(pairing_weights),
                space_id=canonical_fingerprint(
                    {
                        "kind": "tensor-grid-field-coordinates",
                        "support": self.support.support_id,
                        "entity_layout": layout_.layout_id,
                        "component_shape": list(components),
                        "dtype": str(jnp.dtype(dtype)),
                    }
                ),
            ),
            representation=representation,
            conformity=conformity,
            reconstruction_id=canonical_fingerprint(
                {
                    "kind": "tensor-grid-reconstruction",
                    "support": self.support.support_id,
                    "entity_layout": layout_.layout_id,
                }
            ),
        )


__all__ = ["GridLocation", "PreparedTensorGrid"]
