#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Modality-neutral sample-support contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Protocol, runtime_checkable

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._quantity import canonical_quantity_text
from ._time import SampleTimeAxis


def _shape(value, name: str, /) -> tuple[int, ...]:
    if isinstance(value, Integral):
        raise TypeError(f"{name} must be a sequence of positive integers.")
    shape = tuple(value)
    if any(isinstance(size, bool) or not isinstance(size, Integral) for size in shape):
        raise TypeError(f"{name} entries must be integers.")
    shape = tuple(int(size) for size in shape)
    if not shape or any(size < 1 for size in shape):
        raise ValueError(f"{name} must be non-empty and positive.")
    return shape


def _readonly(value: ArrayLike, name: str, /, *, dtype=None) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    if array.dtype.hasobject:
        raise TypeError(f"{name} must not use object dtype.")
    array.setflags(write=False)
    return array


def _real(value: ArrayLike, name: str, /) -> np.ndarray:
    original = np.asarray(value)
    array = _readonly(value, name, dtype=np.result_type(original.dtype, np.float64))
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"{name} must be real-valued.")
    return array


@runtime_checkable
class SampleSupport(Protocol):
    @property
    def sample_shape(self) -> tuple[int, ...]: ...

    @property
    def support_id(self) -> str: ...


@dataclass(frozen=True, slots=True)
class IndexSampleSupport:
    """Named dense sample axes without an asserted physical embedding."""

    sample_shape: tuple[int, ...]
    axis_labels: tuple[str, ...]
    time_axis: SampleTimeAxis | None = None
    time_axis_dimension: int | None = None
    frame_id: str | None = None
    support_id: str = field(init=False)

    def __post_init__(self) -> None:
        shape = _shape(self.sample_shape, "sample_shape")
        labels = tuple(
            canonical_quantity_text(label, "axis_label") for label in self.axis_labels
        )
        if len(labels) != len(shape) or len(labels) != len(set(labels)):
            raise ValueError("axis_labels must uniquely label every sample dimension.")
        dimension = self.time_axis_dimension
        if self.time_axis is None:
            if dimension is not None:
                raise ValueError("time_axis_dimension requires time_axis.")
        else:
            if not isinstance(self.time_axis, SampleTimeAxis):
                raise TypeError("time_axis must be SampleTimeAxis.")
            if isinstance(dimension, bool) or not isinstance(dimension, Integral):
                raise TypeError("time_axis_dimension must be an integer.")
            dimension = int(dimension)
            if dimension < 0 or dimension >= len(shape):
                raise ValueError("time_axis_dimension is outside sample_shape.")
            if shape[dimension] != self.time_axis.sample_count:
                raise ValueError("time-axis size disagrees with sample_shape.")
        frame = (
            None
            if self.frame_id is None
            else canonical_quantity_text(self.frame_id, "frame_id")
        )
        object.__setattr__(self, "sample_shape", shape)
        object.__setattr__(self, "axis_labels", labels)
        object.__setattr__(self, "time_axis_dimension", dimension)
        object.__setattr__(self, "frame_id", frame)
        object.__setattr__(
            self,
            "support_id",
            canonical_fingerprint(
                {
                    "kind": "index-sample-support",
                    "shape": list(shape),
                    "axes": list(labels),
                    "time": None
                    if self.time_axis is None
                    else self.time_axis.time_axis_id,
                    "time_dimension": dimension,
                    "frame": frame,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class PointSampleSupport:
    """Stable irregular physical sample points without implied quadrature measure."""

    points: np.ndarray
    sample_ids: tuple[str, ...]
    coordinate_contract: SpatialCoordinateContract
    sample_times: np.ndarray | None = None
    time_unit_id: str | None = None
    active_mask: np.ndarray | None = None
    sample_shape: tuple[int, ...] = field(init=False)
    support_id: str = field(init=False)

    def __post_init__(self) -> None:
        points = _real(self.points, "points")
        if points.ndim != 2 or points.shape[0] < 1 or points.shape[1] < 1:
            raise ValueError("points must have shape (sample_count, spatial_dimension).")
        if not isinstance(self.coordinate_contract, SpatialCoordinateContract):
            raise TypeError("coordinate_contract must be SpatialCoordinateContract.")
        identifiers = tuple(
            canonical_quantity_text(identifier, "sample_id")
            for identifier in self.sample_ids
        )
        if len(identifiers) != points.shape[0] or len(identifiers) != len(
            set(identifiers)
        ):
            raise ValueError("sample_ids must uniquely identify every point.")
        active = (
            np.ones((points.shape[0],), dtype=bool)
            if self.active_mask is None
            else _readonly(self.active_mask, "active_mask", dtype=bool)
        )
        if active.shape != (points.shape[0],):
            raise ValueError("active_mask must have shape (sample_count,).")
        if not np.all(np.where(active[:, None], np.isfinite(points), True)):
            raise ValueError("Active points must be finite.")
        times = None
        time_unit_id = self.time_unit_id
        if self.sample_times is not None:
            times = _real(self.sample_times, "sample_times")
            if times.shape != (points.shape[0],):
                raise ValueError("sample_times must have shape (sample_count,).")
            if time_unit_id is None:
                raise ValueError("sample_times require time_unit_id.")
            time_unit_id = canonical_quantity_text(time_unit_id, "time_unit_id")
        elif time_unit_id is not None:
            raise ValueError("time_unit_id requires sample_times.")
        active.setflags(write=False)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "sample_ids", identifiers)
        object.__setattr__(self, "active_mask", active)
        object.__setattr__(self, "sample_times", times)
        object.__setattr__(self, "time_unit_id", time_unit_id)
        object.__setattr__(self, "sample_shape", (points.shape[0],))
        object.__setattr__(
            self,
            "support_id",
            canonical_fingerprint(
                {
                    "kind": "point-sample-support",
                    "points": array_tree_fingerprint(points),
                    "sample_ids": list(identifiers),
                    "coordinate_contract": self.coordinate_contract.spatial_id,
                    "times": None if times is None else array_tree_fingerprint(times),
                    "time_unit": time_unit_id,
                    "active": array_tree_fingerprint(active),
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class RaySampleSupport:
    """Stable directed acquisition rays without attached measured ranges."""

    origins: np.ndarray
    directions: np.ndarray
    sample_ids: tuple[str, ...]
    coordinate_contract: SpatialCoordinateContract
    sample_times: np.ndarray | None = None
    time_unit_id: str | None = None
    active_mask: np.ndarray | None = None
    near: np.ndarray | None = None
    far: np.ndarray | None = None
    sample_shape: tuple[int, ...] = field(init=False)
    support_id: str = field(init=False)

    def __post_init__(self) -> None:
        origins = _real(self.origins, "origins")
        directions = _real(self.directions, "directions")
        if origins.ndim != 2 or origins.shape != directions.shape or origins.shape[0] < 1:
            raise ValueError(
                "origins and directions must share shape (sample_count, dimension)."
            )
        if not isinstance(self.coordinate_contract, SpatialCoordinateContract):
            raise TypeError("coordinate_contract must be SpatialCoordinateContract.")
        identifiers = tuple(
            canonical_quantity_text(identifier, "sample_id")
            for identifier in self.sample_ids
        )
        if len(identifiers) != origins.shape[0] or len(identifiers) != len(
            set(identifiers)
        ):
            raise ValueError("sample_ids must uniquely identify every ray.")
        active = (
            np.ones((origins.shape[0],), dtype=bool)
            if self.active_mask is None
            else _readonly(self.active_mask, "active_mask", dtype=bool)
        )
        if active.shape != (origins.shape[0],):
            raise ValueError("active_mask must have shape (sample_count,).")
        norms = np.linalg.norm(directions, axis=1)
        valid_geometry = np.isfinite(origins).all(axis=1) & np.isfinite(directions).all(
            axis=1
        )
        if np.any(active & (~valid_geometry | (norms <= 0.0))):
            raise ValueError(
                "Active rays must have finite origins and nonzero directions."
            )
        normalized = np.zeros_like(directions)
        safe = norms > 0.0
        normalized[safe] = directions[safe] / norms[safe, None]
        times = None
        time_unit_id = self.time_unit_id
        if self.sample_times is not None:
            times = _real(self.sample_times, "sample_times")
            if times.shape != (origins.shape[0],):
                raise ValueError("sample_times must have shape (sample_count,).")
            if time_unit_id is None:
                raise ValueError("sample_times require time_unit_id.")
            time_unit_id = canonical_quantity_text(time_unit_id, "time_unit_id")
        elif time_unit_id is not None:
            raise ValueError("time_unit_id requires sample_times.")
        near = (
            np.zeros((origins.shape[0],), dtype=origins.dtype)
            if self.near is None
            else _real(self.near, "near")
        )
        far = (
            np.full((origins.shape[0],), np.inf, dtype=origins.dtype)
            if self.far is None
            else _real(self.far, "far")
        )
        if near.shape != active.shape or far.shape != active.shape:
            raise ValueError("near and far must have shape (sample_count,).")
        if np.any(active & ((near < 0.0) | (far <= near))):
            raise ValueError("Active ray bounds require 0 <= near < far.")
        for array in (active, normalized, near, far):
            array.setflags(write=False)
        object.__setattr__(self, "origins", origins)
        object.__setattr__(self, "directions", normalized)
        object.__setattr__(self, "sample_ids", identifiers)
        object.__setattr__(self, "active_mask", active)
        object.__setattr__(self, "sample_times", times)
        object.__setattr__(self, "time_unit_id", time_unit_id)
        object.__setattr__(self, "near", near)
        object.__setattr__(self, "far", far)
        object.__setattr__(self, "sample_shape", (origins.shape[0],))
        object.__setattr__(
            self,
            "support_id",
            canonical_fingerprint(
                {
                    "kind": "ray-sample-support",
                    "origins": array_tree_fingerprint(origins),
                    "directions": array_tree_fingerprint(normalized),
                    "sample_ids": list(identifiers),
                    "coordinate_contract": self.coordinate_contract.spatial_id,
                    "times": None if times is None else array_tree_fingerprint(times),
                    "time_unit": time_unit_id,
                    "active": array_tree_fingerprint(active),
                    "near": array_tree_fingerprint(near),
                    "far": array_tree_fingerprint(far),
                }
            ),
        )


class PreparedPointSampleSupport(StrictModule, NonTrainableState):
    points: Array
    sample_times: Array | None
    active_mask: Array
    support_id: str = eqx.field(static=True)
    time_unit_id: str | None = eqx.field(static=True)


class PreparedRaySampleSupport(StrictModule, NonTrainableState):
    origins: Array
    directions: Array
    sample_times: Array | None
    active_mask: Array
    near: Array
    far: Array
    support_id: str = eqx.field(static=True)
    time_unit_id: str | None = eqx.field(static=True)


def prepare_point_support(support: PointSampleSupport, /) -> PreparedPointSampleSupport:
    if not isinstance(support, PointSampleSupport):
        raise TypeError("support must be PointSampleSupport.")
    return PreparedPointSampleSupport(
        jnp.asarray(support.points),
        None if support.sample_times is None else jnp.asarray(support.sample_times),
        jnp.asarray(support.active_mask),
        support.support_id,
        support.time_unit_id,
    )


def prepare_ray_support(support: RaySampleSupport, /) -> PreparedRaySampleSupport:
    if not isinstance(support, RaySampleSupport):
        raise TypeError("support must be RaySampleSupport.")
    return PreparedRaySampleSupport(
        jnp.asarray(support.origins),
        jnp.asarray(support.directions),
        None if support.sample_times is None else jnp.asarray(support.sample_times),
        jnp.asarray(support.active_mask),
        jnp.asarray(support.near),
        jnp.asarray(support.far),
        support.support_id,
        support.time_unit_id,
    )


__all__ = [
    "IndexSampleSupport",
    "PointSampleSupport",
    "PreparedPointSampleSupport",
    "PreparedRaySampleSupport",
    "RaySampleSupport",
    "SampleSupport",
    "prepare_point_support",
    "prepare_ray_support",
]
