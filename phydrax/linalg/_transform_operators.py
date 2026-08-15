#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.core as jax_core
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array, ArrayLike

from ._operators import _generic_adjoint, _id, AbstractLinearOperator
from ._properties import OperatorCapabilities, OperatorProperties
from ._spaces import _coordinate_dtype, _has_euclidean_pairing, ArraySpace


OrthogonalTransformKind = Literal["fft", "dct"]
SpectralProperty = Literal[
    "general",
    "self-adjoint",
    "positive-semidefinite",
    "positive-definite",
]


class TransformDiagonalLinearOperator(AbstractLinearOperator):
    """Operator ``T⁻¹ diag(spectrum) T`` for an orthonormal FFT or DCT."""

    spectrum: Array
    axes: tuple[int, ...] = eqx.field(static=True)
    transform: OrthogonalTransformKind = eqx.field(static=True)
    spectral_property: SpectralProperty = eqx.field(static=True)
    nonsingular: bool = eqx.field(static=True)
    source: ArraySpace
    target: ArraySpace

    def __init__(
        self,
        spectrum: ArrayLike,
        /,
        *,
        space: ArraySpace | None = None,
        transform: OrthogonalTransformKind = "fft",
        axes: tuple[int, ...] | None = None,
        spectral_property: SpectralProperty = "general",
        nonsingular: bool = False,
        operator_id: str | None = None,
    ):
        spectrum_ = jnp.asarray(spectrum)
        if spectrum_.ndim < 1 or not jnp.issubdtype(spectrum_.dtype, jnp.inexact):
            raise TypeError("spectrum must be a non-scalar inexact array.")
        if transform not in ("fft", "dct"):
            raise ValueError("transform must be 'fft' or 'dct'.")
        if spectral_property not in (
            "general",
            "self-adjoint",
            "positive-semidefinite",
            "positive-definite",
        ):
            raise ValueError("Unknown spectral_property.")
        space_ = (
            ArraySpace(spectrum_.shape, dtype=spectrum_.dtype) if space is None else space
        )
        if not isinstance(space_, ArraySpace) or space_.shape != spectrum_.shape:
            raise ValueError("space must be an ArraySpace matching spectrum.shape.")
        if _coordinate_dtype(space_) != np.dtype(spectrum_.dtype):
            raise TypeError("Spectrum dtype must match the space coordinate dtype.")
        if not _has_euclidean_pairing(space_):
            raise ValueError("Transform-diagonal operators require Euclidean pairing.")
        if transform == "fft" and not jnp.issubdtype(
            spectrum_.dtype, jnp.complexfloating
        ):
            raise TypeError("FFT-diagonal operators require complex coordinates.")
        if transform == "dct" and not jnp.issubdtype(spectrum_.dtype, jnp.floating):
            raise TypeError("DCT-diagonal operators require real coordinates.")
        axes_ = _normalize_axes(spectrum_.ndim, axes)
        self_adjoint = transform == "dct" or spectral_property != "general"
        positive_semidefinite = spectral_property in (
            "positive-semidefinite",
            "positive-definite",
        )
        positive_definite = spectral_property == "positive-definite"
        nonsingular_ = bool(nonsingular or positive_definite)
        invalid = ~jnp.isfinite(spectrum_)
        if self_adjoint:
            invalid = invalid | (jnp.imag(spectrum_) != 0)
        if positive_semidefinite:
            invalid = invalid | (jnp.real(spectrum_) < 0)
        if positive_definite:
            invalid = invalid | (jnp.real(spectrum_) <= 0)
        if nonsingular_:
            invalid = invalid | (spectrum_ == 0)
        invalid_any = jnp.any(invalid)
        if isinstance(invalid_any, jax_core.Tracer):
            spectrum_ = eqx.error_if(
                spectrum_,
                invalid_any,
                "Spectrum violates its declared transform-diagonal properties.",
            )
        elif bool(invalid_any):
            raise ValueError(
                "Spectrum violates its declared transform-diagonal properties."
            )
        rank = space_.size if nonsingular_ else None
        claims = {
            "self_adjoint": self_adjoint,
            "positive_definite": positive_definite,
            "positive_semidefinite": positive_semidefinite,
            "rank": rank,
        }
        self.spectrum = spectrum_
        self.axes = axes_
        self.transform = transform
        self.spectral_property = spectral_property
        self.nonsingular = nonsingular_
        self.source = space_
        self.target = space_
        self.properties = OperatorProperties(
            self_adjoint=self_adjoint,
            positive_definite=positive_definite,
            positive_semidefinite=positive_semidefinite,
            rank=rank,
            evidence={
                name: "verified"
                for name, value in claims.items()
                if value is not False and value is not None
            },
        )
        self.capabilities = OperatorCapabilities(
            transpose=True,
            adjoint=True,
            materialize=True,
        )
        self.batch_shape = ()
        self.operator_id = _id(
            operator_id,
            {
                "kind": "transform-diagonal",
                "space": space_.space_id,
                "transform": transform,
                "axes": list(axes_),
                "spectral_property": spectral_property,
                "nonsingular": nonsingular_,
            },
        )

    def to_transform_coordinates(self, value: ArrayLike, /) -> Array:
        """Apply the declared orthonormal forward transform."""
        array = self.source.validate(jnp.asarray(value))
        return _forward_transform(array, self.transform, self.axes)

    def from_transform_coordinates(self, value: ArrayLike, /) -> Array:
        """Apply the declared orthonormal inverse transform."""
        array = jnp.asarray(value)
        if array.shape != self.source.shape:
            raise ValueError("Transform coordinates must match the operator space shape.")
        return self.source.validate(_inverse_transform(array, self.transform, self.axes))

    def mv(self, vector: ArrayLike, /) -> Array:
        value = self.source.validate(jnp.asarray(vector))
        transformed = _forward_transform(value, self.transform, self.axes)
        return self.target.validate(
            _inverse_transform(
                self.spectrum * transformed,
                self.transform,
                self.axes,
            )
        )

    def transpose_mv(self, vector: ArrayLike, /) -> Array:
        value = self.target.validate(jnp.asarray(vector))
        if self.transform == "dct":
            return self.mv(value)
        return jnp.conj(self.adjoint_mv(jnp.conj(value)))

    def adjoint_mv(self, vector: ArrayLike, /) -> Array:
        value = self.target.validate(jnp.asarray(vector))
        if not _has_euclidean_pairing(self.source):
            return _generic_adjoint(self, value)
        transformed = _forward_transform(value, self.transform, self.axes)
        return self.source.validate(
            _inverse_transform(
                jnp.conj(self.spectrum) * transformed,
                self.transform,
                self.axes,
            )
        )

    def _materialize(self, /) -> Array:
        basis = jnp.eye(self.source.size, dtype=self.spectrum.dtype)

        def column(coordinates):
            value = coordinates.reshape(self.source.shape)
            return self.target.flatten(self.mv(value))

        return jax.vmap(column)(basis).T

    def _solve_flat_columns(self, rhs: Array, inverse_spectrum: Array, /) -> Array:
        """Apply a prepared inverse spectrum to canonical columns."""
        count = rhs.shape[1]
        values = rhs.reshape(self.source.shape + (count,))
        transformed = _forward_transform(values, self.transform, self.axes)
        scaled = inverse_spectrum.reshape(self.source.shape + (1,)) * transformed
        result = _inverse_transform(scaled, self.transform, self.axes)
        return result.reshape((self.source.size, count))


def _normalize_axes(ndim: int, axes: tuple[int, ...] | None, /) -> tuple[int, ...]:
    axes_ = tuple(range(ndim)) if axes is None else tuple(int(axis) for axis in axes)
    normalized = tuple(axis + ndim if axis < 0 else axis for axis in axes_)
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("axes must contain distinct transform axes.")
    if any(axis < 0 or axis >= ndim for axis in normalized):
        raise ValueError("A transform axis is out of range.")
    return normalized


def _forward_transform(
    value: Array,
    transform: OrthogonalTransformKind,
    axes: tuple[int, ...],
    /,
) -> Array:
    if transform == "fft":
        return jnp.fft.fftn(value, axes=axes, norm="ortho")
    result = value
    for axis in axes:
        result = jsp.fft.dct(result, type=2, axis=axis, norm="ortho")
    return result


def _inverse_transform(
    value: Array,
    transform: OrthogonalTransformKind,
    axes: tuple[int, ...],
    /,
) -> Array:
    if transform == "fft":
        return jnp.fft.ifftn(value, axes=axes, norm="ortho")
    result = value
    for axis in reversed(axes):
        result = jsp.fft.idct(result, type=2, axis=axis, norm="ortho")
    return result


__all__ = [
    "OrthogonalTransformKind",
    "SpectralProperty",
    "TransformDiagonalLinearOperator",
]
