#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import prod

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ...linalg import ArraySpace, DiagonalPairing
from .._axis import broadcasted_grid
from .._core import (
    DiscretizationCapability,
    DiscretizationKey,
    DiscretizationRole,
    PreparationReport,
)
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


def _component_names(values: Sequence[str], /) -> tuple[str, ...]:
    names = tuple(str(value) for value in values)
    if not names or any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("component_names must contain unique non-empty names.")
    return names


def _coordinates(layout: TensorEntityLayout, /) -> Array:
    return broadcasted_grid(layout.coordinates_by_axis).reshape(
        layout.shape + (len(layout.axis_names),)
    )


def _face_measure(grid: PreparedTensorGrid, axis: int, /) -> Array:
    layout = grid.faces(grid.axis_names[axis])
    measure = jnp.ones(layout.shape)
    for other_axis, structured_axis in enumerate(grid.structured_axes):
        if other_axis == axis:
            weights = jnp.ones((layout.shape[other_axis],), dtype=measure.dtype)
        else:
            weights = structured_axis.interval_widths
        shape = [1] * len(layout.shape)
        shape[other_axis] = int(weights.size)
        measure = measure * weights.reshape(tuple(shape))
    return measure


class FiniteVolumePlan(AbstractDiscretizationPlan):
    """Structured cell-average finite-volume geometry plan."""

    grid: PreparedTensorGrid
    field_name: str = eqx.field(static=True)
    component_names: tuple[str, ...] = eqx.field(static=True)
    key: DiscretizationKey
    capabilities: tuple[DiscretizationCapability, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid: PreparedTensorGrid,
        /,
        *,
        field_name: str = "state",
        component_names: Sequence[str] = ("value",),
        key: DiscretizationKey | None = None,
        plan_id: str | None = None,
    ):
        if not isinstance(grid, PreparedTensorGrid):
            raise TypeError("FiniteVolumePlan requires a PreparedTensorGrid.")
        if any(axis.primary_entity != "interval" for axis in grid.structured_axes):
            raise ValueError("Finite-volume grids require interval-primary axes.")
        field = str(field_name)
        if not field:
            raise ValueError("field_name must be non-empty.")
        components = _component_names(component_names)
        key_ = (
            DiscretizationKey(
                "finite_volume",
                DiscretizationRole.PHYSICAL,
                domain_labels=grid.axis_names,
            )
            if key is None
            else key
        )
        if not isinstance(key_, DiscretizationKey):
            raise TypeError("key must be a DiscretizationKey.")
        capabilities = (
            DiscretizationCapability.RECONSTRUCTION,
            DiscretizationCapability.TRACE,
            DiscretizationCapability.CONSERVATIVE_FLUX,
            DiscretizationCapability.BOUNDARY_INTEGRAL,
            DiscretizationCapability.MATRIX_FREE,
            DiscretizationCapability.DIFFERENTIABLE_GEOMETRY,
        )
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "structured-finite-volume-plan",
                    "grid": grid.prepared_id,
                    "field": field,
                    "components": list(components),
                    "key": key_.key_id,
                }
            )
            if plan_id is None
            else str(plan_id)
        )
        if not identifier:
            raise ValueError("plan_id must be non-empty.")
        self.grid = grid
        self.field_name = field
        self.component_names = components
        self.key = key_
        self.capabilities = capabilities
        self.plan_id = identifier

    def prepare(self, /, *, numeric_version: str = "0") -> "FiniteVolumeDiscretization":
        return FiniteVolumeDiscretization(self, numeric_version=numeric_version)


class FiniteVolumeDiscretization(AbstractPreparedDiscretization):
    """Prepared structured control volumes with directional face geometry."""

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

    def __init__(
        self,
        plan: FiniteVolumePlan,
        /,
        *,
        numeric_version: str = "0",
    ):
        if not isinstance(plan, FiniteVolumePlan):
            raise TypeError("plan must be a FiniteVolumePlan.")
        grid = plan.grid
        cell_layout = grid.cells()
        face_layouts = tuple(grid.faces(name) for name in grid.axis_names)
        cell_volumes = jnp.ones(cell_layout.shape)
        for axis_index, axis in enumerate(grid.structured_axes):
            shape = [1] * len(cell_layout.shape)
            shape[axis_index] = int(axis.interval_widths.size)
            cell_volumes = cell_volumes * axis.interval_widths.reshape(tuple(shape))
        face_measures = tuple(
            _face_measure(grid, axis) for axis in range(len(grid.shape))
        )
        dimension = len(grid.shape)
        face_area_vectors = tuple(
            measure[..., None]
            * jnp.eye(dimension, dtype=measure.dtype)[axis].reshape(
                (1,) * measure.ndim + (dimension,)
            )
            for axis, measure in enumerate(face_measures)
        )
        cell_centers = _coordinates(cell_layout)
        face_centers = tuple(_coordinates(layout) for layout in face_layouts)
        component_count = len(plan.component_names)
        cell_shape = cell_layout.shape + (component_count,)
        cell_weights = jnp.broadcast_to(cell_volumes[..., None], cell_shape)
        reconstruction_id = canonical_fingerprint(
            {"kind": "finite-volume-cell-average", "plan": plan.plan_id}
        )
        cell_space = DiscreteFieldSpace(
            plan.field_name,
            grid.support.support_id,
            TensorDofLayout(
                grid.axis_names,
                cell_layout.shape,
                component_shape=(component_count,),
                location_id=cell_layout.location_id,
            ),
            ArraySpace(cell_shape, pairing=DiagonalPairing(cell_weights)),
            representation="cell_average",
            conformity="discontinuous",
            reconstruction_id=reconstruction_id,
        )
        face_spaces = tuple(
            DiscreteFieldSpace(
                f"{plan.field_name}_{grid.axis_names[axis]}_flux",
                grid.support.support_id,
                TensorDofLayout(
                    grid.axis_names,
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
            for axis, layout in enumerate(face_layouts)
        )
        measures = (
            DiscreteMeasure(
                "finite_volume_cell",
                grid.support.support_id,
                cell_layout.entity_set_id,
                cell_volumes.reshape((-1,)),
                normalization="physical",
            ),
            *tuple(
                DiscreteMeasure(
                    f"finite_volume_{grid.axis_names[axis]}_face",
                    grid.support.support_id,
                    layout.entity_set_id,
                    face_measures[axis].reshape((-1,)),
                    normalization="physical",
                )
                for axis, layout in enumerate(face_layouts)
            ),
        )
        preparation = PreparationReport(
            capabilities=plan.capabilities,
            diagnostics=(
                "cell measures are finite and positive",
                "directional face measures exclude the normal-axis dual measure",
                "internal faces use canonical positive-axis orientation",
            ),
            resource_counts={
                "cells": prod(cell_layout.shape),
                "components": component_count,
                "faces": sum(prod(layout.shape) for layout in face_layouts),
            },
        )
        spaces, measures_, capabilities = validate_prepared_metadata(
            key=plan.key,
            support=grid.support,
            field_spaces=(cell_space, *face_spaces),
            measures=measures,
            capabilities=plan.capabilities,
            preparation=preparation,
        )
        version = str(numeric_version)
        if not version:
            raise ValueError("numeric_version must be non-empty.")
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-structured-finite-volume",
                "plan": plan.plan_id,
                "cell_layout": cell_layout.layout_id,
                "face_layouts": [layout.layout_id for layout in face_layouts],
                "numeric_version": version,
            }
        )
        self.grid = grid
        self.cell_layout = cell_layout
        self.face_layouts = face_layouts
        self.cell_centers = cell_centers
        self.cell_volumes = cell_volumes
        self.face_centers = face_centers
        self.face_measures = face_measures
        self.face_area_vectors = face_area_vectors
        self.cell_space = cell_space
        self.face_spaces = face_spaces
        self.field_name = plan.field_name
        self.component_names = plan.component_names
        self.key = plan.key
        self.support = grid.support
        self.field_spaces = spaces
        self.measures = measures_
        self.capabilities = capabilities
        self.plan_id = plan.plan_id
        self.prepared_id = prepared_id
        self.numeric_version = version
        self.preparation = preparation

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
        if not 0 <= int(axis) < len(self.grid.shape) or side not in ("lower", "upper"):
            raise ValueError("axis and side must identify one finite-volume boundary.")
        sign = -1.0 if side == "lower" else 1.0
        return sign * jnp.eye(len(self.grid.shape))[int(axis)]


__all__ = ["FiniteVolumeDiscretization", "FiniteVolumePlan"]
