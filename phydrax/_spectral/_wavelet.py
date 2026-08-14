#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from itertools import product
from typing import cast, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._jaxwavelets import BackendBoundary, dwt_axis, idwt_axis, load_filter_taps
from ._multiresolution import MultiresolutionCoefficients


WaveletBoundary: TypeAlias = Literal["periodization", "symmetric", "zero"]


def _backend_boundary(boundary: WaveletBoundary, /) -> BackendBoundary:
    if boundary == "zero":
        return "constant"
    return boundary


def _per_axis_strings(
    value: str | Sequence[str], count: int, name: str, /
) -> tuple[str, ...]:
    values = (
        (value,) * count if isinstance(value, str) else tuple(str(item) for item in value)
    )
    if len(values) != count:
        raise ValueError(f"{name} must provide one value per transformed axis.")
    if any(not item for item in values):
        raise ValueError(f"{name} values must be non-empty.")
    return values


def _resolve_axes(axes: tuple[int, ...], ndim: int, /) -> tuple[int, ...]:
    resolved = tuple(axis + ndim if axis < 0 else axis for axis in axes)
    if any(axis < 0 or axis >= ndim for axis in resolved):
        raise ValueError(f"Wavelet axes {axes} are invalid for an array of rank {ndim}.")
    if len(set(resolved)) != len(resolved):
        raise ValueError("Wavelet axes must be unique after normalization.")
    return resolved


def _crop_axes(values: Array, axes: tuple[int, ...], shape: tuple[int, ...], /) -> Array:
    if len(axes) != len(shape):
        raise ValueError("Reconstruction shape rank does not match transformed axes.")
    slices = [slice(None)] * values.ndim
    for axis, size in zip(axes, shape, strict=True):
        slices[axis] = slice(0, size)
    return values[tuple(slices)]


class WaveletFilterBank(StrictModule, NonTrainableState):
    """Immutable decomposition and reconstruction taps for one wavelet."""

    decomposition_low: Array
    decomposition_high: Array
    reconstruction_low: Array
    reconstruction_high: Array
    name: str = eqx.field(static=True)
    fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        decomposition_low: ArrayLike,
        decomposition_high: ArrayLike,
        reconstruction_low: ArrayLike,
        reconstruction_high: ArrayLike,
        /,
    ):
        identity = str(name).strip()
        taps = tuple(
            jnp.asarray(values)
            for values in (
                decomposition_low,
                decomposition_high,
                reconstruction_low,
                reconstruction_high,
            )
        )
        if not identity:
            raise ValueError("Wavelet filter-bank names must be non-empty.")
        if any(tap.ndim != 1 or int(tap.size) < 2 for tap in taps):
            raise ValueError(
                "Wavelet filters must be one-dimensional with at least two taps."
            )
        if len({int(tap.size) for tap in taps}) != 1:
            raise ValueError(
                "Wavelet decomposition and reconstruction filters must align."
            )
        if any(not bool(jnp.all(jnp.isfinite(tap))) for tap in taps):
            raise ValueError("Wavelet filters must be finite.")
        digest = array_tree_fingerprint(taps)["sha256"]
        self.decomposition_low = taps[0]
        self.decomposition_high = taps[1]
        self.reconstruction_low = taps[2]
        self.reconstruction_high = taps[3]
        self.name = identity
        self.fingerprint = canonical_fingerprint(
            {"kind": "wavelet-filter-bank-v1", "name": identity, "taps": digest}
        )

    @classmethod
    def from_name(cls, name: str, /) -> "WaveletFilterBank":
        """Resolve one standard wavelet through the isolated backend adapter."""
        taps = load_filter_taps(name)
        return cls(name, *taps)

    @property
    def taps(self) -> tuple[Array, Array, Array, Array]:
        """Backend-independent filter tuple in analysis/synthesis order."""
        return (
            self.decomposition_low,
            self.decomposition_high,
            self.reconstruction_low,
            self.reconstruction_high,
        )


class DiscreteWaveletTransform(StrictModule, NonTrainableState):
    """Shape-independent separable critically sampled wavelet transform."""

    filter_banks: tuple[WaveletFilterBank, ...]
    axes: tuple[int, ...] = eqx.field(static=True)
    boundaries: tuple[WaveletBoundary, ...] = eqx.field(static=True)
    levels: int = eqx.field(static=True)
    fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        axes: Sequence[int],
        /,
        *,
        levels: int,
        wavelet: str | Sequence[str] = "haar",
        boundary: WaveletBoundary | Sequence[WaveletBoundary] = "periodization",
    ):
        axes_value = tuple(int(axis) for axis in axes)
        level_count = int(levels)
        if not axes_value:
            raise ValueError("Discrete wavelet transforms require at least one axis.")
        if len(set(axes_value)) != len(axes_value):
            raise ValueError("Discrete wavelet transform axes must be unique.")
        if level_count <= 0:
            raise ValueError("Discrete wavelet transform levels must be positive.")
        wavelet_names = _per_axis_strings(wavelet, len(axes_value), "wavelet")
        boundary_names = _per_axis_strings(boundary, len(axes_value), "boundary")
        if any(
            name not in ("periodization", "symmetric", "zero") for name in boundary_names
        ):
            raise ValueError(
                "Wavelet boundaries must be 'periodization', 'symmetric', or 'zero'."
            )
        banks = tuple(WaveletFilterBank.from_name(name) for name in wavelet_names)
        boundaries = cast(tuple[WaveletBoundary, ...], boundary_names)
        self.filter_banks = banks
        self.axes = axes_value
        self.boundaries = boundaries
        self.levels = level_count
        self.fingerprint = canonical_fingerprint(
            {
                "kind": "discrete-wavelet-transform-v1",
                "axes": axes_value,
                "levels": level_count,
                "boundaries": boundaries,
                "filter_banks": [bank.fingerprint for bank in banks],
            }
        )

    @property
    def spatial_ndim(self) -> int:
        """Number of transformed axes."""
        return len(self.axes)

    @property
    def detail_labels(self) -> tuple[tuple[int, ...], ...]:
        """Lexicographic low/high labels excluding the all-low band."""
        all_low = (0,) * self.spatial_ndim
        return tuple(
            label
            for label in product((0, 1), repeat=self.spatial_ndim)
            if label != all_low
        )

    @property
    def detail_count(self) -> int:
        """Number of tensor-product detail bands per level."""
        return 2**self.spatial_ndim - 1

    def analysis(self, values: ArrayLike, /) -> MultiresolutionCoefficients:
        """Decompose an array into coarsest scaling and ordered detail bands."""
        approximation = jnp.asarray(values)
        axes = _resolve_axes(self.axes, approximation.ndim)
        detail_levels: list[tuple[Array, ...]] = []
        reconstruction_shapes: list[tuple[int, ...]] = []
        all_low = (0,) * self.spatial_ndim
        for _ in range(self.levels):
            shape = tuple(int(approximation.shape[axis]) for axis in axes)
            if any(size <= 1 for size in shape):
                raise ValueError(
                    "Too many wavelet levels for the transformed axis sizes."
                )
            reconstruction_shapes.append(shape)
            bands: list[tuple[tuple[int, ...], Array]] = [((), approximation)]
            for axis, bank, boundary in zip(
                axes, self.filter_banks, self.boundaries, strict=True
            ):
                transformed: list[tuple[tuple[int, ...], Array]] = []
                for label, band in bands:
                    low, high = dwt_axis(
                        band,
                        bank.taps,
                        _backend_boundary(boundary),
                        axis,
                    )
                    transformed.extend(((label + (0,), low), (label + (1,), high)))
                bands = transformed
            approximation = bands[0][1]
            detail_levels.append(tuple(band for label, band in bands if label != all_low))
        return MultiresolutionCoefficients(
            approximation,
            tuple(reversed(detail_levels)),
            reconstruction_shapes=tuple(reversed(reconstruction_shapes)),
            transform_fingerprint=self.fingerprint,
        )

    def synthesis(self, coefficients: MultiresolutionCoefficients, /) -> Array:
        """Reconstruct an array, cropping only axes transformed during analysis."""
        if not isinstance(coefficients, MultiresolutionCoefficients):
            raise TypeError("coefficients must be MultiresolutionCoefficients.")
        if coefficients.transform_fingerprint != self.fingerprint:
            raise ValueError("Wavelet coefficients belong to a different transform.")
        if coefficients.levels != self.levels:
            raise ValueError("Wavelet coefficient depth does not match this transform.")
        approximation = jnp.asarray(coefficients.scaling)
        axes = _resolve_axes(self.axes, approximation.ndim)
        expected_bands = 2**self.spatial_ndim
        for details, target_shape in zip(
            coefficients.details,
            coefficients.reconstruction_shapes,
            strict=True,
        ):
            if len(details) != self.detail_count:
                raise ValueError(
                    f"Each wavelet level requires {self.detail_count} detail bands."
                )
            reference_shape = tuple(int(details[0].shape[axis]) for axis in axes)
            approximation = _crop_axes(approximation, axes, reference_shape)
            bands = (approximation,) + details
            if len(bands) != expected_bands:
                raise ValueError("Wavelet coefficient topology is malformed.")
            for position in reversed(range(self.spatial_ndim)):
                axis = axes[position]
                bank = self.filter_banks[position]
                boundary = _backend_boundary(self.boundaries[position])
                bands = tuple(
                    idwt_axis(bands[index], bands[index + 1], bank.taps, boundary, axis)
                    for index in range(0, len(bands), 2)
                )
            approximation = _crop_axes(bands[0], axes, target_shape)
        return approximation

    def __call__(self, values: ArrayLike, /) -> MultiresolutionCoefficients:
        return self.analysis(values)


__all__ = [
    "DiscreteWaveletTransform",
    "WaveletBoundary",
    "WaveletFilterBank",
]
