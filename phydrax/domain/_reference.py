#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array

from .._sampling import UnitCubeTransport
from ._components import Boundary, Interior, Selection
from ._dataset import DatasetDomain
from ._hyperrectangle import HyperRectangle
from ._probability import open_unit_interval, ProbabilityDomain
from ._scalar import AbstractScalarDomain
from .geometry1d._primitives import Interval1d


@dataclass(frozen=True, slots=True)
class _ExactReferenceTransport:
    reference_dimension: int
    _map: Callable[[Array], Any]

    def map(self, unit: Array, /) -> Any:
        return self._map(jnp.asarray(unit, dtype=float))


def _scalar_transport(
    factor: AbstractScalarDomain,
    component: Selection,
) -> UnitCubeTransport | None:
    if isinstance(component, Interior):
        if isinstance(factor, ProbabilityDomain):
            return _ExactReferenceTransport(
                1,
                lambda unit: jnp.asarray(
                    factor.distribution.icdf(open_unit_interval(unit[:, 0])),
                    dtype=float,
                ),
            )
        lower = jnp.asarray(factor.fixed("start"), dtype=float)
        upper = jnp.asarray(factor.fixed("end"), dtype=float)
        return _ExactReferenceTransport(
            1,
            lambda unit: lower + unit[:, 0] * (upper - lower),
        )
    if isinstance(component, Boundary):
        lower = jnp.asarray(factor.fixed("start"), dtype=float)
        upper = jnp.asarray(factor.fixed("end"), dtype=float)
        return _ExactReferenceTransport(
            1,
            lambda unit: jnp.where(unit[:, 0] < 0.5, lower, upper),
        )
    return None


def _interval_geometry_transport(
    factor: Interval1d,
    component: Selection,
) -> UnitCubeTransport | None:
    if isinstance(component, Interior):
        return _ExactReferenceTransport(
            1,
            lambda unit: factor.start + unit[:, :1] * (factor.end - factor.start),
        )
    if isinstance(component, Boundary):
        return _ExactReferenceTransport(
            1,
            lambda unit: jnp.where(
                unit[:, :1] < 0.5,
                factor.start,
                factor.end,
            ),
        )
    return None


def _box_boundary(factor: HyperRectangle, unit: Array, /) -> Array:
    dimension = int(factor.spatial_dim)
    if dimension == 1:
        return jnp.where(unit[:, :1] < 0.5, factor.lower, factor.upper)

    widths = factor.upper - factor.lower
    face_measures = jnp.prod(widths) / widths
    probabilities = jnp.repeat(face_measures, 2)
    probabilities = probabilities / jnp.sum(probabilities)
    face_ids = jnp.searchsorted(jnp.cumsum(probabilities), unit[:, 0], side="right")
    face_ids = jnp.minimum(face_ids, 2 * dimension - 1)
    axes = face_ids // 2
    sides = face_ids % 2
    local = unit[:, 1:]
    columns = []
    for column in range(dimension):
        lower_index = max(column - 1, 0)
        upper_index = min(column, dimension - 2)
        local_coordinate = jnp.where(
            axes < column,
            local[:, lower_index],
            local[:, upper_index],
        )
        interior = factor.lower[column] + local_coordinate * widths[column]
        boundary = jnp.where(
            sides == 0,
            factor.lower[column],
            factor.upper[column],
        )
        columns.append(jnp.where(axes == column, boundary, interior))
    return jnp.stack(columns, axis=1)


def _box_transport(
    factor: HyperRectangle,
    component: Selection,
) -> UnitCubeTransport | None:
    dimension = int(factor.spatial_dim)
    if isinstance(component, Interior):
        return _ExactReferenceTransport(
            dimension,
            lambda unit: factor.lower + unit * (factor.upper - factor.lower),
        )
    if isinstance(component, Boundary):
        return _ExactReferenceTransport(
            dimension,
            lambda unit: _box_boundary(factor, unit),
        )
    return None


def _dataset_transport(
    factor: DatasetDomain,
    component: Selection,
) -> UnitCubeTransport | None:
    if not isinstance(component, Interior):
        return None

    def map_dataset(unit: Array):
        indices = jnp.floor(unit[:, 0] * factor.size).astype(jnp.int32)
        indices = jnp.minimum(indices, factor.size - 1)
        return jax.tree_util.tree_map(lambda value: value[indices], factor.data)

    return _ExactReferenceTransport(1, map_dataset)


def reference_transport(
    factor: Any,
    component: Selection,
    /,
) -> UnitCubeTransport | None:
    """Return an exact target-measure transport for one domain factor."""
    if isinstance(factor, ProbabilityDomain):
        return _scalar_transport(factor, component)
    if isinstance(factor, AbstractScalarDomain):
        return _scalar_transport(factor, component)
    if isinstance(factor, Interval1d):
        return _interval_geometry_transport(factor, component)
    if isinstance(factor, HyperRectangle):
        return _box_transport(factor, component)
    if isinstance(factor, DatasetDomain):
        return _dataset_transport(factor, component)
    return None


__all__ = ["reference_transport"]
