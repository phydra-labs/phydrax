#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


NeutralSchemaKind: TypeAlias = Literal["geometry", "material", "field", "point_cloud"]


def _array(name: str, value: ArrayLike, /, *, rank: int | None = None) -> Array:
    array = jnp.asarray(value)
    if rank is not None and array.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}.")
    if not jnp.issubdtype(array.dtype, jnp.number) or bool(jnp.any(~jnp.isfinite(array))):
        raise ValueError(f"{name} must be finite numeric data.")
    return array


def _integer_array(name: str, value: ArrayLike, /) -> Array:
    host = np.asarray(value)
    if not np.issubdtype(host.dtype, np.integer):
        raise TypeError(f"{name} must have an integer dtype.")
    limits = np.iinfo(np.int32)
    if np.any(host < limits.min) or np.any(host > limits.max):
        raise ValueError(f"{name} contains values outside signed int32.")
    return jnp.asarray(host, dtype=jnp.int32)


def _portable_array(value: ArrayLike, /) -> Any:
    array = np.asarray(value)
    if np.iscomplexobj(array):
        return {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "real": np.real(array).tolist(),
            "imag": np.imag(array).tolist(),
        }
    return array.tolist()


class NeutralGeometrySchema(StrictModule, NonTrainableState):
    coordinates: Array
    cells: Array | None
    units: str = eqx.field(static=True)
    schema_id: str = eqx.field(static=True)

    def __init__(
        self,
        coordinates: ArrayLike,
        /,
        *,
        cells: ArrayLike | None = None,
        units: str = "dimensionless",
    ):
        coordinates_ = _array("coordinates", coordinates, rank=2)
        if jnp.iscomplexobj(coordinates_):
            raise TypeError("Geometry coordinates must be real-valued.")
        cells_ = None if cells is None else _integer_array("cells", cells)
        if cells_ is not None and (
            cells_.ndim != 2
            or bool(jnp.any(cells_ < 0))
            or bool(jnp.any(cells_ >= coordinates_.shape[0]))
        ):
            raise ValueError("Geometry cells contain invalid connectivity.")
        units_ = str(units)
        if not units_:
            raise ValueError("Geometry units must be nonempty.")
        self.coordinates = coordinates_
        self.cells = cells_
        self.units = units_
        self.schema_id = canonical_fingerprint(
            {
                "kind": "neutral-geometry-schema",
                "coordinates": array_tree_fingerprint(coordinates_),
                "cells": None if cells_ is None else array_tree_fingerprint(cells_),
                "units": units_,
            }
        )

    def to_dict(self, /) -> dict[str, Any]:
        return {
            "kind": "geometry",
            "coordinates": _portable_array(self.coordinates),
            "cells": None if self.cells is None else _portable_array(self.cells),
            "units": self.units,
            "schema_id": self.schema_id,
        }


class NeutralMaterialSchema(StrictModule, NonTrainableState):
    fields: tuple[tuple[str, Array], ...]
    units: tuple[tuple[str, str], ...] = eqx.field(static=True)
    schema_id: str = eqx.field(static=True)

    def __init__(
        self,
        fields: Mapping[str, ArrayLike],
        /,
        *,
        units: Mapping[str, str] | None = None,
    ):
        if not fields:
            raise ValueError("Neutral material schema requires fields.")
        arrays = tuple(
            (str(name), _array(str(name), value))
            for name, value in sorted(fields.items())
        )
        if any(not name for name, _ in arrays):
            raise ValueError("Material field names must be nonempty.")
        unit_map = (
            {}
            if units is None
            else {str(name): str(value) for name, value in units.items()}
        )
        unknown = set(unit_map) - {name for name, _ in arrays}
        if unknown:
            raise ValueError(f"Material units contain unknown fields {sorted(unknown)}.")
        units_ = tuple((name, unit_map.get(name, "dimensionless")) for name, _ in arrays)
        if any(not unit for _, unit in units_):
            raise ValueError("Material units must be nonempty.")
        self.fields = arrays
        self.units = units_
        self.schema_id = canonical_fingerprint(
            {
                "kind": "neutral-material-schema",
                "fields": {name: array_tree_fingerprint(value) for name, value in arrays},
                "units": dict(units_),
            }
        )

    def to_dict(self, /) -> dict[str, Any]:
        return {
            "kind": "material",
            "fields": {name: _portable_array(value) for name, value in self.fields},
            "units": dict(self.units),
            "schema_id": self.schema_id,
        }


class NeutralFieldSchema(StrictModule, NonTrainableState):
    name: str = eqx.field(static=True)
    values: Array
    cochain_degree: int | None = eqx.field(static=True)
    time: float | None = eqx.field(static=True)
    angular_frequency: float | None = eqx.field(static=True)
    units: str = eqx.field(static=True)
    schema_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        values: ArrayLike,
        /,
        *,
        cochain_degree: int | None = None,
        time: float | None = None,
        angular_frequency: float | None = None,
        units: str = "dimensionless",
    ):
        name_ = str(name)
        values_ = _array("field values", values)
        degree = None if cochain_degree is None else int(cochain_degree)
        if not name_ or (degree is not None and degree < 0):
            raise ValueError("Field name/cochain_degree are invalid.")
        if not str(units):
            raise ValueError("Field units must be nonempty.")
        time_ = None if time is None else float(time)
        frequency = None if angular_frequency is None else float(angular_frequency)
        if time_ is not None and not np.isfinite(time_):
            raise ValueError("Field time must be finite.")
        if frequency is not None and (not np.isfinite(frequency) or frequency < 0.0):
            raise ValueError("Field angular_frequency must be finite and nonnegative.")
        self.name = name_
        self.values = values_
        self.cochain_degree = degree
        self.time = time_
        self.angular_frequency = frequency
        self.units = str(units)
        self.schema_id = canonical_fingerprint(
            {
                "kind": "neutral-field-schema",
                "name": name_,
                "values": array_tree_fingerprint(values_),
                "degree": degree,
                "time": time_,
                "frequency": frequency,
                "units": self.units,
            }
        )

    def to_dict(self, /) -> dict[str, Any]:
        return {
            "kind": "field",
            "name": self.name,
            "values": _portable_array(self.values),
            "cochain_degree": self.cochain_degree,
            "time": self.time,
            "angular_frequency": self.angular_frequency,
            "units": self.units,
            "schema_id": self.schema_id,
        }


class NeutralPointCloudSchema(StrictModule, NonTrainableState):
    points: Array
    quadrature_weights: Array
    boundary_labels: Array
    neighborhoods: Array | None
    schema_id: str = eqx.field(static=True)

    def __init__(
        self,
        points: ArrayLike,
        quadrature_weights: ArrayLike,
        boundary_labels: ArrayLike,
        /,
        *,
        neighborhoods: ArrayLike | None = None,
    ):
        points_ = _array("point cloud", points, rank=2)
        if jnp.iscomplexobj(points_):
            raise TypeError("Point-cloud coordinates must be real-valued.")
        weights = _array("point weights", quadrature_weights, rank=1)
        labels = _integer_array("boundary_labels", boundary_labels)
        if weights.shape != points_.shape[:1] or labels.shape != points_.shape[:1]:
            raise ValueError("Point cloud weights/labels must match point count.")
        if bool(jnp.any(weights <= 0.0)) or bool(jnp.any(labels < 0)):
            raise ValueError("Point weights must be positive and labels nonnegative.")
        routes = (
            None
            if neighborhoods is None
            else _integer_array("neighborhoods", neighborhoods)
        )
        if routes is not None and (
            routes.ndim != 2
            or routes.shape[0] != points_.shape[0]
            or bool(jnp.any((routes < -1) | (routes >= points_.shape[0])))
        ):
            raise ValueError("Point neighborhoods are invalid.")
        self.points = points_
        self.quadrature_weights = weights
        self.boundary_labels = labels
        self.neighborhoods = routes
        self.schema_id = canonical_fingerprint(
            {
                "kind": "neutral-point-cloud-schema",
                "points": array_tree_fingerprint(points_),
                "weights": array_tree_fingerprint(weights),
                "labels": array_tree_fingerprint(labels),
                "neighborhoods": None
                if routes is None
                else array_tree_fingerprint(routes),
            }
        )

    def to_dict(self, /) -> dict[str, Any]:
        return {
            "kind": "point_cloud",
            "points": _portable_array(self.points),
            "quadrature_weights": _portable_array(self.quadrature_weights),
            "boundary_labels": _portable_array(self.boundary_labels),
            "neighborhoods": (
                None
                if self.neighborhoods is None
                else _portable_array(self.neighborhoods)
            ),
            "schema_id": self.schema_id,
        }


class NeutralAdapterBoundary(StrictModule, NonTrainableState):
    """Data-only adapter boundary that rejects runtime/code objects."""

    allowed_kinds: tuple[NeutralSchemaKind, ...] = eqx.field(static=True)

    def __init__(
        self,
        allowed_kinds: tuple[NeutralSchemaKind, ...] = (
            "geometry",
            "material",
            "field",
            "point_cloud",
        ),
    ):
        if not allowed_kinds or any(
            value not in ("geometry", "material", "field", "point_cloud")
            for value in allowed_kinds
        ):
            raise ValueError("Neutral adapter allowed_kinds are invalid.")
        self.allowed_kinds = tuple(allowed_kinds)

    def export(self, value: Any, /) -> dict[str, Any]:
        if not isinstance(
            value,
            (
                NeutralGeometrySchema,
                NeutralMaterialSchema,
                NeutralFieldSchema,
                NeutralPointCloudSchema,
            ),
        ):
            raise TypeError(
                "Adapters accept only neutral data schemas, never runtimes/operators."
            )
        payload = value.to_dict()
        if payload["kind"] not in self.allowed_kinds:
            raise ValueError("Neutral schema kind is not allowed by this boundary.")
        return payload


__all__ = [
    "NeutralAdapterBoundary",
    "NeutralFieldSchema",
    "NeutralGeometrySchema",
    "NeutralMaterialSchema",
    "NeutralPointCloudSchema",
    "NeutralSchemaKind",
]
