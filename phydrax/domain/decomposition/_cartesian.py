#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise, product
from numbers import Real

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._domain import Domain, JointFactor
from .._function import DomainFunction
from .._hyperrectangle import HyperRectangle
from .._product_domain import ProductDomain
from .._scalar import ScalarInterval
from ..geometry1d import Interval1d
from ._cover import PairedSupport, SubdomainCover, SubdomainPatch


class _IdentityCoordinate(StrictModule, NonTrainableState):
    def __init__(self):
        pass

    def __call__(self, value, /, *, key=None, **kwargs):
        del key, kwargs
        return value


def _periodic_image(value, center, lower, upper, periodic):
    period = upper - lower
    wrapped = center + jnp.mod(value - center + 0.5 * period, period) - 0.5 * period
    return jnp.where(periodic, wrapped, value)


class _PeriodicCoordinate(StrictModule, NonTrainableState):
    lower: tuple[float, ...] = eqx.field(static=True)
    upper: tuple[float, ...] = eqx.field(static=True)
    center: tuple[float, ...] = eqx.field(static=True)
    periodic: tuple[bool, ...] = eqx.field(static=True)
    vector_coordinate: bool = eqx.field(static=True)
    direction: str = eqx.field(static=True)

    def __init__(
        self,
        lower: Sequence[float],
        upper: Sequence[float],
        center: Sequence[float],
        periodic: Sequence[bool],
        *,
        vector_coordinate: bool,
        direction: str,
    ):
        if direction not in ("to-local", "to-ambient"):
            raise ValueError("Unknown periodic coordinate direction.")
        self.lower = tuple(float(value) for value in lower)
        self.upper = tuple(float(value) for value in upper)
        self.center = tuple(float(value) for value in center)
        self.periodic = tuple(bool(value) for value in periodic)
        self.vector_coordinate = bool(vector_coordinate)
        self.direction = direction

    def __call__(self, coordinate, /, *, key=None, **kwargs):
        del key, kwargs
        value = jnp.asarray(coordinate)
        if not self.vector_coordinate:
            value = value[..., None]
        lower = jnp.asarray(self.lower, dtype=value.dtype)
        upper = jnp.asarray(self.upper, dtype=value.dtype)
        periodic = jnp.asarray(self.periodic)
        if self.direction == "to-local":
            center = jnp.asarray(self.center, dtype=value.dtype)
            result = _periodic_image(value, center, lower, upper, periodic)
        else:
            period = upper - lower
            wrapped = lower + jnp.mod(value - lower, period)
            result = jnp.where(periodic, wrapped, value)
        return result if self.vector_coordinate else result[..., 0]


class _BoxSupport(StrictModule, NonTrainableState):
    lower: tuple[float, ...] = eqx.field(static=True)
    upper: tuple[float, ...] = eqx.field(static=True)
    ambient_lower: tuple[float, ...] = eqx.field(static=True)
    ambient_upper: tuple[float, ...] = eqx.field(static=True)
    center: tuple[float, ...] = eqx.field(static=True)
    periodic: tuple[bool, ...] = eqx.field(static=True)
    vector_coordinate: bool = eqx.field(static=True)

    def __init__(
        self,
        lower: Sequence[float],
        upper: Sequence[float],
        *,
        ambient_lower: Sequence[float],
        ambient_upper: Sequence[float],
        center: Sequence[float],
        periodic: Sequence[bool],
        vector_coordinate: bool,
    ):
        self.lower = tuple(float(value) for value in lower)
        self.upper = tuple(float(value) for value in upper)
        self.ambient_lower = tuple(float(value) for value in ambient_lower)
        self.ambient_upper = tuple(float(value) for value in ambient_upper)
        self.center = tuple(float(value) for value in center)
        self.periodic = tuple(bool(value) for value in periodic)
        self.vector_coordinate = bool(vector_coordinate)

    def __call__(self, coordinate, /, *, key=None, **kwargs):
        del key, kwargs
        value = jnp.asarray(coordinate)
        if not self.vector_coordinate:
            value = value[..., None]
        value = _periodic_image(
            value,
            jnp.asarray(self.center, dtype=value.dtype),
            jnp.asarray(self.ambient_lower, dtype=value.dtype),
            jnp.asarray(self.ambient_upper, dtype=value.dtype),
            jnp.asarray(self.periodic),
        )
        lower = jnp.asarray(self.lower, dtype=value.dtype)
        upper = jnp.asarray(self.upper, dtype=value.dtype)
        return jnp.all((value >= lower) & (value <= upper), axis=-1)


class _BoxWindow(StrictModule, NonTrainableState):
    support_lower: tuple[float, ...] = eqx.field(static=True)
    core_lower: tuple[float, ...] = eqx.field(static=True)
    core_upper: tuple[float, ...] = eqx.field(static=True)
    support_upper: tuple[float, ...] = eqx.field(static=True)
    ambient_lower: tuple[float, ...] = eqx.field(static=True)
    ambient_upper: tuple[float, ...] = eqx.field(static=True)
    periodic: tuple[bool, ...] = eqx.field(static=True)
    vector_coordinate: bool = eqx.field(static=True)

    def __init__(
        self,
        support_lower: Sequence[float],
        core_lower: Sequence[float],
        core_upper: Sequence[float],
        support_upper: Sequence[float],
        *,
        ambient_lower: Sequence[float],
        ambient_upper: Sequence[float],
        periodic: Sequence[bool],
        vector_coordinate: bool,
    ):
        self.support_lower = tuple(float(value) for value in support_lower)
        self.core_lower = tuple(float(value) for value in core_lower)
        self.core_upper = tuple(float(value) for value in core_upper)
        self.support_upper = tuple(float(value) for value in support_upper)
        self.ambient_lower = tuple(float(value) for value in ambient_lower)
        self.ambient_upper = tuple(float(value) for value in ambient_upper)
        self.periodic = tuple(bool(value) for value in periodic)
        self.vector_coordinate = bool(vector_coordinate)

    @staticmethod
    def _smootherstep(value):
        return value**3 * (value * (value * 6.0 - 15.0) + 10.0)

    def __call__(self, coordinate, /, *, key=None, **kwargs):
        del key, kwargs
        value = jnp.asarray(coordinate)
        if not self.vector_coordinate:
            value = value[..., None]
        support_lower = jnp.asarray(self.support_lower, dtype=value.dtype)
        core_lower = jnp.asarray(self.core_lower, dtype=value.dtype)
        core_upper = jnp.asarray(self.core_upper, dtype=value.dtype)
        support_upper = jnp.asarray(self.support_upper, dtype=value.dtype)
        value = _periodic_image(
            value,
            0.5 * (core_lower + core_upper),
            jnp.asarray(self.ambient_lower, dtype=value.dtype),
            jnp.asarray(self.ambient_upper, dtype=value.dtype),
            jnp.asarray(self.periodic),
        )
        left_width = core_lower - support_lower
        right_width = support_upper - core_upper
        safe_left = jnp.where(left_width > 0.0, left_width, 1.0)
        safe_right = jnp.where(right_width > 0.0, right_width, 1.0)
        left_coordinate = jnp.clip((value - support_lower) / safe_left, 0.0, 1.0)
        right_coordinate = jnp.clip((support_upper - value) / safe_right, 0.0, 1.0)
        left = jnp.where(
            (value < core_lower) & (left_width > 0.0),
            self._smootherstep(left_coordinate),
            1.0,
        )
        right = jnp.where(
            (value > core_upper) & (right_width > 0.0),
            self._smootherstep(right_coordinate),
            1.0,
        )
        inside = jnp.all(
            (value >= support_lower) & (value <= support_upper),
            axis=-1,
        )
        result = jnp.prod(left * right, axis=-1)
        return jnp.where(inside, result, 0.0)


class _AffineCoordinate(StrictModule, NonTrainableState):
    center: tuple[float, ...] = eqx.field(static=True)
    half_width: tuple[float, ...] = eqx.field(static=True)
    vector_coordinate: bool = eqx.field(static=True)

    def __init__(
        self,
        lower: Sequence[float],
        upper: Sequence[float],
        *,
        vector_coordinate: bool,
    ):
        lower_ = np.asarray(lower, dtype=float)
        upper_ = np.asarray(upper, dtype=float)
        self.center = tuple(float(value) for value in 0.5 * (lower_ + upper_))
        self.half_width = tuple(float(value) for value in 0.5 * (upper_ - lower_))
        self.vector_coordinate = bool(vector_coordinate)

    def __call__(self, coordinate, /, *, key=None, **kwargs):
        del key, kwargs
        value = jnp.asarray(coordinate)
        center = jnp.asarray(self.center, dtype=value.dtype)
        half_width = jnp.asarray(self.half_width, dtype=value.dtype)
        normalized = (value - center) / half_width
        return normalized if self.vector_coordinate else normalized[..., 0]


class _FaceEmbedding(StrictModule, NonTrainableState):
    axis: int = eqx.field(static=True)
    boundary: float = eqx.field(static=True)
    dimension: int = eqx.field(static=True)

    def __init__(self, axis: int, boundary: float, dimension: int):
        self.axis = int(axis)
        self.boundary = float(boundary)
        self.dimension = int(dimension)

    def __call__(self, tangent, /, *, key=None, **kwargs):
        del key, kwargs
        tangent_ = jnp.asarray(tangent)
        boundary = jnp.asarray([self.boundary], dtype=tangent_.dtype)
        return jnp.concatenate(
            (tangent_[..., : self.axis], boundary, tangent_[..., self.axis :]),
            axis=-1,
        )


class AxisPartition(StrictModule, NonTrainableState):
    """Explicit nonuniform partition and overlap policy for one Cartesian axis."""

    boundaries: tuple[float, ...] = eqx.field(static=True)
    overlap_fraction: float = eqx.field(static=True)
    periodic: bool = eqx.field(static=True)

    def __init__(
        self,
        boundaries: Sequence[float],
        /,
        *,
        overlap_fraction: float = 0.0,
        periodic: bool = False,
    ):
        boundaries_ = tuple(float(value) for value in boundaries)
        if len(boundaries_) < 2:
            raise ValueError("AxisPartition requires at least two boundaries.")
        if any(not math.isfinite(value) for value in boundaries_):
            raise ValueError("AxisPartition boundaries must be finite.")
        if any(right <= left for left, right in pairwise(boundaries_)):
            raise ValueError("AxisPartition boundaries must be strictly increasing.")
        overlap = float(overlap_fraction)
        if not math.isfinite(overlap) or not 0.0 <= overlap < 0.5:
            raise ValueError("overlap_fraction must be finite and in [0, 0.5).")
        self.boundaries = boundaries_
        self.overlap_fraction = overlap
        self.periodic = bool(periodic)

    @property
    def count(self) -> int:
        return len(self.boundaries) - 1

    @classmethod
    def uniform(
        cls,
        start: float,
        end: float,
        count: int,
        /,
        *,
        overlap_fraction: float = 0.0,
        periodic: bool = False,
    ) -> AxisPartition:
        count_ = int(count)
        if count_ <= 0:
            raise ValueError("count must be positive.")
        return cls(
            np.linspace(float(start), float(end), count_ + 1),
            overlap_fraction=overlap_fraction,
            periodic=periodic,
        )


class BoxPartition(StrictModule, NonTrainableState):
    """Tensor-product axis partitions for one HyperRectangle factor."""

    axes: tuple[AxisPartition, ...]

    def __init__(self, axes: Sequence[AxisPartition], /):
        axes_ = tuple(axes)
        if not axes_ or any(not isinstance(axis, AxisPartition) for axis in axes_):
            raise TypeError("axes must be a non-empty sequence of AxisPartition objects.")
        self.axes = axes_

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(axis.count for axis in self.axes)


def _factor_bounds(
    factor: JointFactor,
    /,
) -> tuple[np.ndarray, np.ndarray, bool]:
    if isinstance(factor, ScalarInterval):
        return (
            np.asarray([factor.start], dtype=float),
            np.asarray([factor.end], dtype=float),
            False,
        )
    if isinstance(factor, Interval1d):
        return (
            np.asarray([factor.start], dtype=float),
            np.asarray([factor.end], dtype=float),
            True,
        )
    if isinstance(factor, HyperRectangle):
        return (
            np.asarray(factor.lower, dtype=float),
            np.asarray(factor.upper, dtype=float),
            True,
        )
    raise TypeError(
        "CartesianCoverPlan supports ScalarInterval, Interval1d, and HyperRectangle "
        "factors."
    )


def _factor_like(
    prototype: JointFactor,
    lower: Sequence[float],
    upper: Sequence[float],
    /,
) -> JointFactor:
    label = prototype.labels[0]
    lower_ = np.asarray(lower, dtype=float)
    upper_ = np.asarray(upper, dtype=float)
    if isinstance(prototype, ScalarInterval):
        return ScalarInterval(float(lower_[0]), float(upper_[0]), label=label)
    if isinstance(prototype, Interval1d):
        return Interval1d(float(lower_[0]), float(upper_[0]), label=label)
    if isinstance(prototype, HyperRectangle):
        return HyperRectangle(lower_, upper_, label=label)
    raise TypeError("Unsupported Cartesian factor.")


def _replace_factor(
    ambient: Domain,
    target: JointFactor,
    replacement: JointFactor,
    /,
) -> Domain:
    factors = tuple(
        replacement if factor is target else factor for factor in ambient.joint_factors
    )
    return factors[0] if len(factors) == 1 else ProductDomain(*factors)


def _identity_coordinates(domain: Domain, /) -> dict[str, DomainFunction]:
    return {
        label: domain.Function(label)(_IdentityCoordinate()) for label in domain.labels
    }


def _constant_coordinate(domain: Domain, value, /) -> DomainFunction:
    return DomainFunction(domain=domain, deps=(), func=jnp.asarray(value, dtype=float))


def _sequence_or_scalar(value, dimension: int, name: str, cast):
    if isinstance(value, (Real, bool)):
        return tuple(cast(value) for _ in range(dimension))
    values = tuple(cast(entry) for entry in value)
    if len(values) == 1 and dimension > 1:
        return values * dimension
    if len(values) != dimension:
        raise ValueError(f"{name} must have length {dimension}.")
    return values


def _normalize_boundary_input(boundaries, dimension: int):
    values = tuple(boundaries)
    if dimension == 1 and values and isinstance(values[0], Real):
        return (tuple(float(value) for value in values),)
    nested = tuple(tuple(float(value) for value in axis) for axis in values)
    if len(nested) != dimension:
        raise ValueError(f"boundaries must define {dimension} axes.")
    return nested


def _patch_id(index: tuple[int, ...]) -> str:
    return "patch-" + "-".join(f"{value:04d}" for value in index)


def _interface_factor(
    prototype: JointFactor,
    axis: int,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    label: str,
) -> JointFactor:
    dimension = int(lower.shape[0])
    if dimension == 1:
        return ScalarInterval(0.0, 1.0, label=label)
    tangent_lower = np.delete(lower, axis)
    tangent_upper = np.delete(upper, axis)
    return HyperRectangle(tangent_lower, tangent_upper, label=label)


def _interface_domain(
    ambient: Domain,
    split_factor: JointFactor,
    interface_factor: JointFactor,
    /,
) -> Domain:
    return _replace_factor(ambient, split_factor, interface_factor)


def _face_coordinates(
    interface_domain: Domain,
    split_factor: JointFactor,
    interface_factor: JointFactor,
    *,
    axis: int,
    boundary: float,
    dimension: int,
    vector_coordinate: bool,
) -> dict[str, DomainFunction]:
    coordinates: dict[str, DomainFunction] = {}
    split_label = split_factor.labels[0]
    if dimension == 1:
        value = jnp.asarray([boundary]) if vector_coordinate else jnp.asarray(boundary)
        coordinates[split_label] = _constant_coordinate(interface_domain, value)
    else:
        tangent_label = interface_factor.labels[0]
        coordinates[split_label] = interface_domain.Function(tangent_label)(
            _FaceEmbedding(axis, boundary, dimension)
        )
    for label in interface_domain.labels:
        if label != interface_factor.labels[0]:
            coordinates[label] = interface_domain.Function(label)(_IdentityCoordinate())
    return coordinates


class CartesianCoverPlan(StrictModule, NonTrainableState):
    """Exact Cartesian cover for one scalar or vector-valued box factor."""

    label: str = eqx.field(static=True)
    counts: tuple[int, ...] | None = eqx.field(static=True)
    boundaries: tuple[tuple[float, ...], ...] | None = eqx.field(static=True)
    axis_partitions: tuple[AxisPartition, ...] | None
    overlap_fraction: tuple[float, ...] = eqx.field(static=True)
    periodic: tuple[bool, ...] = eqx.field(static=True)
    cover_id: str | None = eqx.field(static=True)

    def __init__(
        self,
        label: str,
        num_subdomains: int | Sequence[int] | None = None,
        /,
        *,
        boundaries: Sequence[float] | Sequence[Sequence[float]] | None = None,
        axis_partitions: Sequence[AxisPartition] | None = None,
        overlap_fraction: float | Sequence[float] = 0.0,
        periodic: bool | Sequence[bool] = False,
        cover_id: str | None = None,
    ):
        if not isinstance(label, str) or not label:
            raise ValueError("label must be a non-empty string.")
        provided = sum(
            value is not None for value in (num_subdomains, boundaries, axis_partitions)
        )
        if provided != 1:
            raise ValueError(
                "Define exactly one of num_subdomains, boundaries, or axis_partitions."
            )
        if num_subdomains is None:
            counts = None
        elif isinstance(num_subdomains, int):
            if num_subdomains <= 0:
                raise ValueError("num_subdomains must be positive.")
            counts = (int(num_subdomains),)
        else:
            counts = tuple(int(value) for value in num_subdomains)
            if not counts or any(value <= 0 for value in counts):
                raise ValueError("num_subdomains entries must be positive.")
        boundaries_ = None if boundaries is None else tuple(boundaries)
        partitions_ = None if axis_partitions is None else tuple(axis_partitions)
        if partitions_ is not None and (
            not partitions_
            or any(not isinstance(value, AxisPartition) for value in partitions_)
        ):
            raise TypeError("axis_partitions must contain AxisPartition objects.")
        overlap_values = (
            (float(overlap_fraction),)
            if isinstance(overlap_fraction, Real)
            else tuple(float(value) for value in overlap_fraction)
        )
        periodic_values = (
            (bool(periodic),)
            if isinstance(periodic, bool)
            else tuple(bool(value) for value in periodic)
        )
        if cover_id is not None and (not isinstance(cover_id, str) or not cover_id):
            raise ValueError("cover_id must be a non-empty string or None.")
        self.label = label
        self.counts = counts
        self.boundaries = boundaries_
        self.axis_partitions = partitions_
        self.overlap_fraction = overlap_values
        self.periodic = periodic_values
        self.cover_id = cover_id

    def _partitions(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> tuple[AxisPartition, ...]:
        dimension = int(lower.shape[0])
        if self.axis_partitions is not None:
            if len(self.axis_partitions) != dimension:
                raise ValueError(
                    f"axis_partitions must define all {dimension} factor axes."
                )
            partitions = self.axis_partitions
        elif self.boundaries is not None:
            nested = _normalize_boundary_input(self.boundaries, dimension)
            overlaps = _sequence_or_scalar(
                self.overlap_fraction,
                dimension,
                "overlap_fraction",
                float,
            )
            periodic = _sequence_or_scalar(
                self.periodic,
                dimension,
                "periodic",
                bool,
            )
            partitions = tuple(
                AxisPartition(axis, overlap_fraction=overlap, periodic=periodic_)
                for axis, overlap, periodic_ in zip(
                    nested,
                    overlaps,
                    periodic,
                    strict=True,
                )
            )
        else:
            if self.counts is None:
                raise RuntimeError("Cartesian count configuration was lost.")
            counts = (
                self.counts * dimension
                if len(self.counts) == 1 and dimension > 1
                else self.counts
            )
            if len(counts) != dimension:
                raise ValueError(f"num_subdomains must have length {dimension}.")
            overlaps = _sequence_or_scalar(
                self.overlap_fraction,
                dimension,
                "overlap_fraction",
                float,
            )
            periodic = _sequence_or_scalar(
                self.periodic,
                dimension,
                "periodic",
                bool,
            )
            partitions = tuple(
                AxisPartition.uniform(
                    low,
                    high,
                    count,
                    overlap_fraction=overlap,
                    periodic=periodic_,
                )
                for low, high, count, overlap, periodic_ in zip(
                    lower,
                    upper,
                    counts,
                    overlaps,
                    periodic,
                    strict=True,
                )
            )
        for axis, (partition, low, high) in enumerate(
            zip(partitions, lower, upper, strict=True)
        ):
            if not np.isclose(partition.boundaries[0], low) or not np.isclose(
                partition.boundaries[-1], high
            ):
                raise ValueError(
                    f"Axis {axis} partition endpoints must match the factor bounds."
                )
        return partitions

    def build(self, ambient: Domain, /) -> SubdomainCover:
        if not isinstance(ambient, Domain):
            raise TypeError("ambient must be a Domain.")
        if self.label not in ambient.labels:
            raise KeyError(f"Label {self.label!r} is absent from the ambient domain.")
        factor = ambient.factor(self.label)
        if factor.labels != (self.label,):
            raise ValueError("Cartesian cover cannot split part of a coupled factor.")
        lower, upper, vector_coordinate = _factor_bounds(factor)
        dimension = int(lower.shape[0])
        partitions = self._partitions(lower, upper)
        shape = tuple(partition.count for partition in partitions)
        index_space = tuple(product(*(range(count) for count in shape)))

        core_bounds: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]] = {}
        support_bounds: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]] = {}
        patches: list[SubdomainPatch] = []
        for index in index_space:
            core_lower = np.asarray(
                [partitions[axis].boundaries[value] for axis, value in enumerate(index)]
            )
            core_upper = np.asarray(
                [
                    partitions[axis].boundaries[value + 1]
                    for axis, value in enumerate(index)
                ]
            )
            widths = core_upper - core_lower
            overlaps = np.asarray(
                [partition.overlap_fraction for partition in partitions]
            )
            periodic_axes = np.asarray(
                [partition.periodic for partition in partitions],
                dtype=bool,
            )
            raw_support_lower = core_lower - overlaps * widths
            raw_support_upper = core_upper + overlaps * widths
            support_lower = np.where(
                periodic_axes,
                raw_support_lower,
                np.maximum(lower, raw_support_lower),
            )
            support_upper = np.where(
                periodic_axes,
                raw_support_upper,
                np.minimum(upper, raw_support_upper),
            )
            center = 0.5 * (core_lower + core_upper)
            local_factor = _factor_like(factor, support_lower, support_upper)
            local_domain = _replace_factor(ambient, factor, local_factor)
            support = ambient.Function(self.label)(
                _BoxSupport(
                    support_lower,
                    support_upper,
                    ambient_lower=lower,
                    ambient_upper=upper,
                    center=center,
                    periodic=periodic_axes,
                    vector_coordinate=vector_coordinate,
                )
            ).with_metadata(
                support_lower=tuple(float(value) for value in support_lower),
                support_upper=tuple(float(value) for value in support_upper),
            )
            window_available = all(
                partition.count == 1 or partition.overlap_fraction > 0.0
                for partition in partitions
            )
            window = (
                ambient.Function(self.label)(
                    _BoxWindow(
                        support_lower,
                        core_lower,
                        core_upper,
                        support_upper,
                        ambient_lower=lower,
                        ambient_upper=upper,
                        periodic=periodic_axes,
                        vector_coordinate=vector_coordinate,
                    )
                ).with_metadata(
                    regularity_order=2,
                    coordinate_label=self.label,
                    core_lower=tuple(float(value) for value in core_lower),
                    core_upper=tuple(float(value) for value in core_upper),
                )
                if window_available
                else None
            )
            to_local = _identity_coordinates(ambient)
            to_ambient = _identity_coordinates(local_domain)
            if bool(np.any(periodic_axes)):
                to_local[self.label] = ambient.Function(self.label)(
                    _PeriodicCoordinate(
                        lower,
                        upper,
                        center,
                        periodic_axes,
                        vector_coordinate=vector_coordinate,
                        direction="to-local",
                    )
                )
                to_ambient[self.label] = local_domain.Function(self.label)(
                    _PeriodicCoordinate(
                        lower,
                        upper,
                        center,
                        periodic_axes,
                        vector_coordinate=vector_coordinate,
                        direction="to-ambient",
                    )
                )
            patch_id = _patch_id(index)
            patches.append(
                SubdomainPatch(
                    local_domain,
                    local_domain.component(),
                    support,
                    to_local,
                    to_ambient,
                    patch_id=patch_id,
                    window=window,
                )
            )
            core_bounds[index] = (core_lower, core_upper)
            support_bounds[index] = (support_lower, support_upper)

        by_index = {
            index: patch for index, patch in zip(index_space, patches, strict=True)
        }
        pairings: list[PairedSupport] = []
        for index in index_space:
            for axis, partition in enumerate(partitions):
                value = index[axis]
                if value + 1 < partition.count:
                    neighbor = tuple(
                        value_ + 1 if axis_ == axis else value_
                        for axis_, value_ in enumerate(index)
                    )
                    left_boundary = partition.boundaries[value + 1]
                    right_boundary = left_boundary
                    topology = "shared-interface"
                elif partition.periodic and partition.count > 1:
                    neighbor = tuple(
                        0 if axis_ == axis else value_
                        for axis_, value_ in enumerate(index)
                    )
                    left_boundary = partition.boundaries[-1]
                    right_boundary = partition.boundaries[0]
                    topology = "periodic-interface"
                else:
                    continue
                core_lower, core_upper = core_bounds[index]
                occupied = set(ambient.labels)
                stem = f"interface_{self.label}_{axis}_" + "_".join(map(str, index))
                interface_label = stem
                suffix = 0
                while interface_label in occupied:
                    suffix += 1
                    interface_label = f"{stem}_{suffix}"
                interface_factor = _interface_factor(
                    factor,
                    axis,
                    core_lower,
                    core_upper,
                    label=interface_label,
                )
                interface_domain = _interface_domain(
                    ambient,
                    factor,
                    interface_factor,
                )
                left_coordinates = _face_coordinates(
                    interface_domain,
                    factor,
                    interface_factor,
                    axis=axis,
                    boundary=left_boundary,
                    dimension=dimension,
                    vector_coordinate=vector_coordinate,
                )
                right_coordinates = _face_coordinates(
                    interface_domain,
                    factor,
                    interface_factor,
                    axis=axis,
                    boundary=right_boundary,
                    dimension=dimension,
                    vector_coordinate=vector_coordinate,
                )
                normal_array = np.zeros((dimension,), dtype=float)
                normal_array[axis] = 1.0
                normal_value = (
                    jnp.asarray(normal_array)
                    if vector_coordinate
                    else jnp.asarray(normal_array[0])
                )
                pairings.append(
                    PairedSupport(
                        interface_domain.component(),
                        left_coordinates,
                        right_coordinates,
                        pairing_id=(
                            f"interface-axis-{axis}-"
                            f"{by_index[index].patch_id}-"
                            f"{by_index[neighbor].patch_id}"
                        ),
                        left_patch_id=by_index[index].patch_id,
                        right_patch_id=by_index[neighbor].patch_id,
                        normal=_constant_coordinate(interface_domain, normal_value),
                        codimension=1,
                        topology=topology,
                    )
                )

        maximum_overlap = math.prod(
            2 if partition.count > 1 else 1 for partition in partitions
        )
        identifier = self.cover_id or canonical_fingerprint(
            {
                "kind": "cartesian-subdomain-cover",
                "ambient_type": type(ambient).__qualname__,
                "label": self.label,
                "partitions": tuple(
                    {
                        "boundaries": partition.boundaries,
                        "overlap_fraction": partition.overlap_fraction,
                        "periodic": partition.periodic,
                    }
                    for partition in partitions
                ),
            }
        )
        return SubdomainCover(
            ambient,
            patches,
            pairings,
            cover_id=identifier,
            exact_coverage=True,
            maximum_overlap=maximum_overlap,
        )


def cartesian_subdomain_cover(
    ambient: Domain,
    label: str,
    num_subdomains: int | Sequence[int] | None = None,
    /,
    *,
    boundaries: Sequence[float] | Sequence[Sequence[float]] | None = None,
    axis_partitions: Sequence[AxisPartition] | None = None,
    overlap_fraction: float | Sequence[float] = 0.0,
    periodic: bool | Sequence[bool] = False,
    cover_id: str | None = None,
) -> SubdomainCover:
    """Build an exact scalar-interval or HyperRectangle Cartesian cover."""
    return CartesianCoverPlan(
        label,
        num_subdomains,
        boundaries=boundaries,
        axis_partitions=axis_partitions,
        overlap_fraction=overlap_fraction,
        periodic=periodic,
        cover_id=cover_id,
    ).build(ambient)


def normalized_patch_coordinate(
    patch: SubdomainPatch,
    label: str,
    /,
) -> DomainFunction:
    """Return unclipped affine local coordinates in [-1, 1]."""
    if not isinstance(patch, SubdomainPatch):
        raise TypeError("patch must be a SubdomainPatch.")
    factor = patch.domain.factor(label)
    lower, upper, vector_coordinate = _factor_bounds(factor)
    return patch.domain.Function(label)(
        _AffineCoordinate(lower, upper, vector_coordinate=vector_coordinate)
    )


__all__ = [
    "AxisPartition",
    "BoxPartition",
    "CartesianCoverPlan",
    "cartesian_subdomain_cover",
    "normalized_patch_coordinate",
]
