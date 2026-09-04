#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..equations._les_closures import ResolvedLESFilter


FilterKind = Literal["identity", "box", "gaussian", "spectral_cutoff"]
FilterBoundary = Literal["periodic", "linear"]
LESFilterPairInput = Literal["primary-resolved"]


class LESFilterPair(StrictModule, NonTrainableState):
    """Ordered resolved/test-filter semantics without an executable filter claim.

    The pair deliberately carries the two core LES filter identities only. It does
    not equate index-space box averaging, modal projection, coarsening, or
    dealiasing. A later dynamic procedure must separately supply an implementation
    matching each declared filter.
    """

    primary_filter: ResolvedLESFilter
    test_filter: ResolvedLESFilter
    test_filter_input: LESFilterPairInput = eqx.field(static=True)
    pair_id: str = eqx.field(static=True)

    def __init__(
        self,
        primary_filter: ResolvedLESFilter,
        test_filter: ResolvedLESFilter,
        /,
        *,
        test_filter_input: LESFilterPairInput = "primary-resolved",
    ):
        if not isinstance(primary_filter, ResolvedLESFilter) or not isinstance(
            test_filter, ResolvedLESFilter
        ):
            raise TypeError(
                "primary_filter and test_filter must be ResolvedLESFilter values."
            )
        input_semantics = str(test_filter_input).strip()
        if input_semantics != "primary-resolved":
            raise ValueError(
                "The LES test filter must declare primary-resolved input semantics."
            )
        if primary_filter.filter_id == test_filter.filter_id:
            raise ValueError(
                "Primary and test LES filters must have distinct identities."
            )
        if (
            primary_filter.axis_names != test_filter.axis_names
            or primary_filter.topology != test_filter.topology
            or primary_filter.boundary_class != test_filter.boundary_class
        ):
            raise ValueError(
                "Primary and test LES filters must share axes, topology, and boundary class."
            )
        self.primary_filter = primary_filter
        self.test_filter = test_filter
        self.test_filter_input = input_semantics
        self.pair_id = canonical_fingerprint(
            {
                "kind": "les-filter-pair",
                "primary_filter": primary_filter.filter_id,
                "test_filter": test_filter.filter_id,
                "test_filter_input": input_semantics,
            }
        )


class FilterSpec(StrictModule, NonTrainableState):
    """Symbolic separable spatial filter, prepared explicitly for one grid shape."""

    kind: FilterKind = eqx.field(static=True)
    widths: tuple[int, ...] = eqx.field(static=True)
    sigma: tuple[float, ...] = eqx.field(static=True)
    cutoff_fraction: float = eqx.field(static=True)
    boundary: FilterBoundary = eqx.field(static=True)
    spec_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: FilterKind,
        /,
        *,
        widths: tuple[int, ...] = (),
        sigma: tuple[float, ...] = (),
        cutoff_fraction: float = 2.0 / 3.0,
        boundary: FilterBoundary = "periodic",
    ):
        kind_ = str(kind).strip()
        widths_ = tuple(int(value) for value in widths)
        sigma_ = tuple(float(value) for value in sigma)
        cutoff = float(cutoff_fraction)
        boundary_ = str(boundary).strip()
        if (
            kind_ not in ("identity", "box", "gaussian", "spectral_cutoff")
            or any(value <= 0 or value % 2 == 0 for value in widths_)
            or any(not np.isfinite(value) or value <= 0.0 for value in sigma_)
            or not np.isfinite(cutoff)
            or not 0.0 < cutoff <= 1.0
            or boundary_ not in ("periodic", "linear")
            or (kind_ == "box" and not widths_)
            or (kind_ == "gaussian" and (not widths_ or len(sigma_) != len(widths_)))
            or (kind_ in ("identity", "spectral_cutoff") and (widths_ or sigma_))
            or (kind_ == "spectral_cutoff" and boundary_ != "periodic")
        ):
            raise ValueError("Filter specification is invalid.")
        self.kind = kind_
        self.widths = widths_
        self.sigma = sigma_
        self.cutoff_fraction = cutoff
        self.boundary = boundary_
        self.spec_id = canonical_fingerprint(
            {
                "kind": "closure-filter-spec",
                "filter_kind": kind_,
                "widths": list(widths_),
                "sigma": list(sigma_),
                "cutoff_fraction": cutoff,
                "boundary": boundary_,
            }
        )

    @classmethod
    def identity(cls) -> FilterSpec:
        return cls("identity")

    @classmethod
    def box(
        cls, widths: tuple[int, ...], /, *, boundary: FilterBoundary = "periodic"
    ) -> FilterSpec:
        return cls("box", widths=widths, boundary=boundary)

    @classmethod
    def gaussian(
        cls,
        widths: tuple[int, ...],
        sigma: tuple[float, ...],
        /,
        *,
        boundary: FilterBoundary = "periodic",
    ) -> FilterSpec:
        return cls("gaussian", widths=widths, sigma=sigma, boundary=boundary)

    @classmethod
    def spectral_cutoff(cls, cutoff_fraction: float = 2.0 / 3.0) -> FilterSpec:
        return cls("spectral_cutoff", cutoff_fraction=cutoff_fraction)

    def prepare(self, spatial_shape: tuple[int, ...], /) -> PreparedFilter:
        return PreparedFilter(self, spatial_shape)


class PreparedFilter(StrictModule, NonTrainableState):
    """Shape-bound executable filter; component and batch axes may trail it."""

    spec: FilterSpec
    kernels: tuple[Array, ...]
    spectral_mask: Array
    spatial_shape: tuple[int, ...] = eqx.field(static=True)
    spatial_rank: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(self, spec: FilterSpec, spatial_shape: tuple[int, ...], /):
        if not isinstance(spec, FilterSpec):
            raise TypeError("spec must be a FilterSpec.")
        shape = tuple(int(value) for value in spatial_shape)
        if not shape or any(value < 2 for value in shape):
            raise ValueError(
                "Prepared filters require spatial dimensions of size at least two."
            )
        if spec.kind in ("box", "gaussian") and len(spec.widths) != len(shape):
            raise ValueError("One filter width is required per spatial dimension.")
        if spec.kind == "box":
            kernels = tuple(
                jnp.full((width,), 1.0 / width, dtype=jnp.float64)
                for width in spec.widths
            )
        elif spec.kind == "gaussian":
            kernel_values = []
            for width, sigma in zip(spec.widths, spec.sigma, strict=True):
                offsets = np.arange(-(width // 2), width // 2 + 1, dtype=np.float64)
                values = np.exp(-0.5 * (offsets / sigma) ** 2)
                values = values / np.sum(values)
                kernel_values.append(jnp.asarray(values))
            kernels = tuple(kernel_values)
        else:
            kernels = ()
        if spec.kind == "spectral_cutoff":
            mask = jnp.ones(shape, dtype=bool)
            for axis, size in enumerate(shape):
                frequencies = jnp.abs(jnp.fft.fftfreq(size) * size)
                maximum = max(1.0, float(size // 2))
                axis_mask = frequencies <= spec.cutoff_fraction * maximum
                axis_shape = [1] * len(shape)
                axis_shape[axis] = size
                mask = mask & axis_mask.reshape(tuple(axis_shape))
        else:
            mask = jnp.ones((1,) * len(shape), dtype=bool)
        self.spec = spec
        self.kernels = kernels
        self.spectral_mask = mask
        self.spatial_shape = shape
        self.spatial_rank = len(shape)
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-closure-filter",
                "spec": spec.spec_id,
                "spatial_shape": list(shape),
            }
        )

    def validate(self, values: ArrayLike, /, *, owner: str = "Filtered field") -> Array:
        array = jnp.asarray(values)
        if (
            array.ndim < self.spatial_rank
            or tuple(array.shape[: self.spatial_rank]) != self.spatial_shape
        ):
            raise ValueError(
                f"{owner} must begin with spatial shape {self.spatial_shape}; "
                f"got {array.shape}."
            )
        if not jnp.issubdtype(array.dtype, jnp.inexact):
            raise TypeError(f"{owner} must use an inexact dtype.")
        return array

    def apply(self, values: ArrayLike, /) -> Array:
        array = self.validate(values)
        if self.spec.kind == "identity":
            return array
        if self.spec.kind == "spectral_cutoff":
            transformed = jnp.fft.fftn(array, axes=tuple(range(self.spatial_rank)))
            mask = self.spectral_mask.reshape(
                self.spatial_shape + (1,) * (array.ndim - self.spatial_rank)
            )
            filtered = jnp.fft.ifftn(
                transformed * mask, axes=tuple(range(self.spatial_rank))
            )
            if not jnp.issubdtype(array.dtype, jnp.complexfloating):
                return filtered.real.astype(array.dtype)
            return filtered.astype(array.dtype)
        result = array
        for axis, kernel in enumerate(self.kernels):
            radius = int(kernel.size) // 2
            filtered = jnp.zeros_like(result)
            for kernel_index in range(int(kernel.size)):
                offset = kernel_index - radius
                shifted = _shift(result, axis, offset, self.spec.boundary)
                filtered = filtered + kernel[kernel_index].astype(result.dtype) * shifted
            result = filtered
        return result

    def __call__(self, values: ArrayLike, /) -> Array:
        return self.apply(values)


class ReynoldsFilter(StrictModule, NonTrainableState):
    prepared: PreparedFilter
    filter_id: str = eqx.field(static=True)

    def __init__(self, prepared: PreparedFilter, /):
        if not isinstance(prepared, PreparedFilter):
            raise TypeError("prepared must be a PreparedFilter.")
        self.prepared = prepared
        self.filter_id = canonical_fingerprint(
            {"kind": "reynolds-filter", "prepared": prepared.prepared_id}
        )

    def apply(self, values: ArrayLike, /) -> Array:
        return self.prepared.apply(values)

    def __call__(self, values: ArrayLike, /) -> Array:
        return self.apply(values)


class FavreFilter(StrictModule, NonTrainableState):
    prepared: PreparedFilter
    density_floor: float = eqx.field(static=True)
    filter_id: str = eqx.field(static=True)

    def __init__(self, prepared: PreparedFilter, /, *, density_floor: float = 0.0):
        if not isinstance(prepared, PreparedFilter):
            raise TypeError("prepared must be a PreparedFilter.")
        floor = float(density_floor)
        if not np.isfinite(floor) or floor < 0.0:
            raise ValueError("density_floor must be finite and nonnegative.")
        self.prepared = prepared
        self.density_floor = floor
        self.filter_id = canonical_fingerprint(
            {
                "kind": "favre-filter",
                "prepared": prepared.prepared_id,
                "density_floor": floor,
            }
        )

    def mean_density(self, density: ArrayLike, /) -> Array:
        rho = self.prepared.validate(density, owner="Favre density")
        rho = eqx.error_if(
            rho,
            jnp.any(~jnp.isfinite(rho)) | jnp.any(rho <= self.density_floor),
            "Favre filtering requires finite density strictly above density_floor.",
        )
        mean = self.prepared.apply(rho)
        return eqx.error_if(
            mean,
            jnp.any(~jnp.isfinite(mean)) | jnp.any(mean <= self.density_floor),
            "Favre-filtered density is not strictly positive.",
        )

    def apply(self, values: ArrayLike, density: ArrayLike, /) -> Array:
        field = self.prepared.validate(values, owner="Favre field")
        rho = self.prepared.validate(density, owner="Favre density")
        if (
            rho.shape != field.shape[: self.prepared.spatial_rank]
            and rho.shape != field.shape
        ):
            if rho.shape != field.shape[:-1]:
                raise ValueError(
                    "Favre density must match the field or its trailing components."
                )
        mean_density = self.mean_density(rho)
        while rho.ndim < field.ndim:
            rho = rho[..., None]
            mean_density = mean_density[..., None]
        numerator = self.prepared.apply(rho * field)
        return numerator / mean_density

    def __call__(self, values: ArrayLike, density: ArrayLike, /) -> Array:
        return self.apply(values, density)


class FilterCommutationReport(StrictModule, NonTrainableState):
    filtered_derivative: Array
    derivative_filtered: Array
    defect: Array
    defect_norm: Array
    reference_norm: Array
    axis: int = eqx.field(static=True)
    spacing: float = eqx.field(static=True)
    filter_id: str = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        filtered_derivative: ArrayLike,
        derivative_filtered: ArrayLike,
        axis: int,
        spacing: float,
        filter_id: str,
    ):
        left = jnp.asarray(filtered_derivative)
        right = jnp.asarray(derivative_filtered)
        if left.shape != right.shape:
            raise ValueError("Commutation operands must share shape.")
        defect = left - right
        self.filtered_derivative = left
        self.derivative_filtered = right
        self.defect = defect
        self.defect_norm = _norm(defect)
        self.reference_norm = jnp.maximum(_norm(left), _norm(right))
        self.axis = int(axis)
        self.spacing = float(spacing)
        self.filter_id = str(filter_id)
        self.report_id = canonical_fingerprint(
            {
                "kind": "filter-commutation-report",
                "filter": self.filter_id,
                "axis": self.axis,
                "spacing": self.spacing,
                "shape": list(left.shape),
            }
        )


class FilterRefinementReport(StrictModule, NonTrainableState):
    restrict_then_filter: Array
    filter_then_restrict: Array
    defect: Array
    defect_norm: Array
    fine_filter_id: str = eqx.field(static=True)
    coarse_filter_id: str = eqx.field(static=True)
    refinement_ratio: tuple[int, ...] = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        restrict_then_filter: ArrayLike,
        filter_then_restrict: ArrayLike,
        fine_filter_id: str,
        coarse_filter_id: str,
        refinement_ratio: tuple[int, ...],
    ):
        left = jnp.asarray(restrict_then_filter)
        right = jnp.asarray(filter_then_restrict)
        if left.shape != right.shape:
            raise ValueError("Refinement commutation operands must share shape.")
        self.restrict_then_filter = left
        self.filter_then_restrict = right
        self.defect = left - right
        self.defect_norm = _norm(self.defect)
        self.fine_filter_id = str(fine_filter_id)
        self.coarse_filter_id = str(coarse_filter_id)
        self.refinement_ratio = tuple(int(value) for value in refinement_ratio)
        self.report_id = canonical_fingerprint(
            {
                "kind": "filter-refinement-report",
                "fine_filter": self.fine_filter_id,
                "coarse_filter": self.coarse_filter_id,
                "ratio": list(self.refinement_ratio),
                "shape": list(left.shape),
            }
        )


def filter_commutation(
    prepared: PreparedFilter,
    values: ArrayLike,
    /,
    *,
    axis: int,
    spacing: float,
) -> FilterCommutationReport:
    if not isinstance(prepared, PreparedFilter):
        raise TypeError("prepared must be a PreparedFilter.")
    axis_ = int(axis)
    spacing_ = float(spacing)
    if (
        axis_ < 0
        or axis_ >= prepared.spatial_rank
        or not np.isfinite(spacing_)
        or spacing_ <= 0.0
    ):
        raise ValueError("Commutation derivative metadata is invalid.")
    array = prepared.validate(values)
    derivative = _central_difference(array, axis_, spacing_, prepared.spec.boundary)
    filtered_derivative = prepared.apply(derivative)
    derivative_filtered = _central_difference(
        prepared.apply(array), axis_, spacing_, prepared.spec.boundary
    )
    return FilterCommutationReport(
        filtered_derivative=filtered_derivative,
        derivative_filtered=derivative_filtered,
        axis=axis_,
        spacing=spacing_,
        filter_id=prepared.prepared_id,
    )


def filter_refinement_commutation(
    fine_filter: PreparedFilter,
    coarse_filter: PreparedFilter,
    fine_values: ArrayLike,
    refinement_ratio: tuple[int, ...],
    /,
) -> FilterRefinementReport:
    from ._alignment import conservative_restrict

    ratio = tuple(int(value) for value in refinement_ratio)
    restricted = conservative_restrict(fine_values, ratio)
    restrict_then_filter = coarse_filter.apply(restricted)
    filter_then_restrict = conservative_restrict(fine_filter.apply(fine_values), ratio)
    return FilterRefinementReport(
        restrict_then_filter=restrict_then_filter,
        filter_then_restrict=filter_then_restrict,
        fine_filter_id=fine_filter.prepared_id,
        coarse_filter_id=coarse_filter.prepared_id,
        refinement_ratio=ratio,
    )


def _shift(values: Array, axis: int, offset: int, boundary: FilterBoundary) -> Array:
    if boundary == "periodic":
        return jnp.roll(values, -offset, axis=axis)
    size = values.shape[axis]
    indices = jnp.arange(size) + offset
    clipped = jnp.clip(indices, 0, size - 1)
    selected = jnp.take(values, clipped, axis=axis)
    first = jnp.take(values, 0, axis=axis)
    second = jnp.take(values, 1, axis=axis)
    penultimate = jnp.take(values, size - 2, axis=axis)
    last = jnp.take(values, size - 1, axis=axis)
    reshape = [1] * values.ndim
    reshape[axis] = size
    low_steps = jnp.minimum(indices, 0).reshape(tuple(reshape))
    high_steps = jnp.maximum(indices - (size - 1), 0).reshape(tuple(reshape))
    low = jnp.expand_dims(first, axis) + low_steps * jnp.expand_dims(second - first, axis)
    high = jnp.expand_dims(last, axis) + high_steps * jnp.expand_dims(
        last - penultimate, axis
    )
    low_mask = (indices < 0).reshape(tuple(reshape))
    high_mask = (indices >= size).reshape(tuple(reshape))
    return jnp.where(low_mask, low, jnp.where(high_mask, high, selected))


def _central_difference(
    values: Array, axis: int, spacing: float, boundary: FilterBoundary
) -> Array:
    return (_shift(values, axis, 1, boundary) - _shift(values, axis, -1, boundary)) / (
        2.0 * spacing
    )


def _norm(values: Array) -> Array:
    return jnp.sqrt(jnp.sum(jnp.real(values * jnp.conj(values))))


__all__ = [
    "LESFilterPair",
    "LESFilterPairInput",
    "FavreFilter",
    "FilterBoundary",
    "FilterCommutationReport",
    "FilterKind",
    "FilterRefinementReport",
    "FilterSpec",
    "PreparedFilter",
    "ReynoldsFilter",
    "filter_commutation",
    "filter_refinement_commutation",
]
