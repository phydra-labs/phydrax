#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal

import jax
import jax.numpy as jnp
from jaxtyping import Array

import phydrax.ein as ein

from ...._doc import DOC_KEY0
from ...._strict import StrictModule
from ..._base import _AbstractBaseModel
from ..._keys import EvalKey


DifferentialTransform = Literal[
    "gradient",
    "curl_3d",
    "rotated_gradient_2d",
    "symmetric_gradient",
]
DerivativeBackend = Literal["autodiff", "central_difference"]


class DifferentialNormalization(StrictModule):
    """Affine scale factors needed to recover physical derivatives."""

    coordinate_scale: Array
    field_scale: Array

    def __init__(self, coordinate_scale: Array, field_scale: Array, /):
        coordinate = jnp.asarray(coordinate_scale, dtype=float).reshape((-1,))
        field = jnp.asarray(field_scale, dtype=float).reshape((-1,))
        if int(coordinate.size) == 0 or int(field.size) == 0:
            raise ValueError("Differential normalization scales must not be empty.")
        if bool(jnp.any(~jnp.isfinite(coordinate))) or bool(jnp.any(coordinate == 0.0)):
            raise ValueError("Coordinate scales must be finite and nonzero.")
        if bool(jnp.any(~jnp.isfinite(field))):
            raise ValueError("Field scales must be finite.")
        self.coordinate_scale = coordinate
        self.field_scale = field

    def physical_jacobian(self, jacobian: Array, /) -> Array:
        """Apply ``field_scale / coordinate_scale`` to a normalized Jacobian."""
        value = jnp.asarray(jacobian)
        expected = (int(self.field_scale.size), int(self.coordinate_scale.size))
        if value.shape[-2:] != expected:
            raise ValueError(
                f"Jacobian must end with field/coordinate shape {expected}; "
                f"got {value.shape}."
            )
        return (
            value
            * self.field_scale.reshape((1,) * (value.ndim - 2) + (-1, 1))
            / self.coordinate_scale.reshape((1,) * (value.ndim - 1) + (-1,))
        )


class LinearDifferentialTransform(StrictModule):
    """Validated linear map from a field Jacobian to derived output channels."""

    coefficients: Array

    def __init__(self, coefficients: Array, /):
        tensor = jnp.asarray(coefficients, dtype=float)
        if tensor.ndim != 3 or any(int(size) <= 0 for size in tensor.shape):
            raise ValueError(
                "Linear differential coefficients require shape "
                "(outputs, fields, coordinates)."
            )
        if bool(jnp.any(~jnp.isfinite(tensor))):
            raise ValueError("Linear differential coefficients must be finite.")
        self.coefficients = tensor

    @property
    def output_channels(self) -> int:
        return int(self.coefficients.shape[0])

    @property
    def field_channels(self) -> int:
        return int(self.coefficients.shape[1])

    @property
    def coordinate_dimension(self) -> int:
        return int(self.coefficients.shape[2])

    def __call__(self, jacobian: Array, /) -> Array:
        return ein.contract("ofc,...fc->...o", self.coefficients, jacobian)


def _field_channels(out_size: Any, /) -> int:
    if out_size == "scalar":
        return 1
    if isinstance(out_size, int):
        return int(out_size)
    count = 1
    for size in out_size:
        count *= int(size)
    return count


class DifferentialFieldDecoder(_AbstractBaseModel):
    """Derive physically structured fields from a coordinate-conditioned decoder."""

    decoder: Any
    transform: DifferentialTransform | LinearDifferentialTransform
    backend: DerivativeBackend
    normalization: DifferentialNormalization
    step: Array
    coord_dim: int
    field_channels: int
    in_size: int
    out_size: int | tuple[int, ...] | Literal["scalar"]

    def __init__(
        self,
        decoder: Any,
        /,
        *,
        transform: DifferentialTransform | LinearDifferentialTransform,
        coord_dim: int | None = None,
        backend: DerivativeBackend = "autodiff",
        step: float | Array = 1e-3,
        normalization: DifferentialNormalization | None = None,
    ):
        dimension = int(decoder.in_size if coord_dim is None else coord_dim)
        if dimension <= 0:
            raise ValueError("coord_dim must be positive.")
        channels = _field_channels(decoder.out_size)
        if isinstance(transform, LinearDifferentialTransform):
            if transform.coordinate_dimension != dimension:
                raise ValueError(
                    "Linear differential transform coordinate dimension is incompatible."
                )
            if transform.field_channels != channels:
                raise ValueError(
                    "Linear differential transform field channels are incompatible."
                )
            output_channels = transform.output_channels
            out_size: int | tuple[int, ...] | Literal["scalar"] = (
                "scalar" if output_channels == 1 else output_channels
            )
        elif transform == "gradient":
            if channels != 1:
                raise ValueError("gradient requires a scalar potential decoder.")
            out_size = "scalar" if dimension == 1 else dimension
        elif transform == "rotated_gradient_2d":
            if dimension != 2 or channels != 1:
                raise ValueError(
                    "rotated_gradient_2d requires a scalar decoder in two dimensions."
                )
            out_size = 2
        elif transform == "curl_3d":
            if dimension != 3 or channels != 3:
                raise ValueError("curl_3d requires a three-vector decoder in 3D.")
            out_size = 3
        elif transform == "symmetric_gradient":
            if channels != dimension:
                raise ValueError(
                    "symmetric_gradient requires one displacement component per coordinate."
                )
            out_size = (dimension, dimension)
        else:
            raise ValueError(f"Unknown differential transform {transform!r}.")
        step_ = jnp.asarray(step, dtype=float)
        if step_.ndim == 0:
            step_ = jnp.full((dimension,), step_)
        else:
            step_ = step_.reshape((-1,))
        if (
            step_.shape != (dimension,)
            or bool(jnp.any(step_ <= 0.0))
            or bool(jnp.any(~jnp.isfinite(step_)))
        ):
            raise ValueError(
                "Finite-difference step must be finite and positive per axis."
            )
        normalizer = (
            DifferentialNormalization(jnp.ones(dimension), jnp.ones(channels))
            if normalization is None
            else normalization
        )
        if normalizer.coordinate_scale.shape != (dimension,):
            raise ValueError("Differential normalization coordinate dimension differs.")
        if normalizer.field_scale.shape != (channels,):
            raise ValueError("Differential normalization field dimension differs.")
        self.decoder = decoder
        self.transform = transform
        self.backend = backend
        self.normalization = normalizer
        self.step = step_
        self.coord_dim = dimension
        self.field_channels = channels
        self.in_size = dimension
        self.out_size = out_size

    def _field(self, point: Array, key: EvalKey, /) -> Array:
        value = jnp.asarray(self.decoder(point, key=key))
        return value.reshape((self.field_channels,))

    def _jacobian(self, point: Array, key: EvalKey, /) -> Array:
        if self.backend == "autodiff":
            jacobian = jax.jacfwd(lambda coordinate: self._field(coordinate, key))(point)
        elif self.backend == "central_difference":
            basis = jnp.eye(self.coord_dim, dtype=point.dtype) * self.step[:, None]
            plus = jax.vmap(lambda offset: self._field(point + offset, key))(basis)
            minus = jax.vmap(lambda offset: self._field(point - offset, key))(basis)
            jacobian = jnp.swapaxes(
                (plus - minus) / (2.0 * self.step[:, None]),
                -1,
                -2,
            )
        else:
            raise ValueError(f"Unknown derivative backend {self.backend!r}.")
        return self.normalization.physical_jacobian(jacobian)

    def _transform(self, jacobian: Array, /) -> Array:
        if isinstance(self.transform, LinearDifferentialTransform):
            return self.transform(jacobian)
        if self.transform == "gradient":
            return jacobian[0]
        if self.transform == "rotated_gradient_2d":
            return jnp.stack((jacobian[0, 1], -jacobian[0, 0]))
        if self.transform == "curl_3d":
            return jnp.stack(
                (
                    jacobian[2, 1] - jacobian[1, 2],
                    jacobian[0, 2] - jacobian[2, 0],
                    jacobian[1, 0] - jacobian[0, 1],
                )
            )
        if self.transform == "symmetric_gradient":
            return 0.5 * (jacobian + jnp.swapaxes(jacobian, -1, -2))
        raise ValueError(f"Unknown differential transform {self.transform!r}.")

    def __call__(
        self,
        x: Array,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        coordinates = jnp.asarray(x)
        if coordinates.ndim == 0 and self.coord_dim == 1:
            coordinates = coordinates[None]
        if coordinates.ndim < 1 or int(coordinates.shape[-1]) != self.coord_dim:
            raise ValueError(
                f"DifferentialFieldDecoder expects trailing size {self.coord_dim}."
            )
        leading = coordinates.shape[:-1]
        flattened = coordinates.reshape((-1, self.coord_dim))
        transformed = jax.vmap(lambda point: self._transform(self._jacobian(point, key)))(
            flattened
        )
        channel_shape = (
            ()
            if self.out_size == "scalar"
            else tuple(self.out_size)
            if isinstance(self.out_size, tuple)
            else (int(self.out_size),)
        )
        return transformed.reshape(leading + channel_shape)


__all__ = [
    "DerivativeBackend",
    "DifferentialFieldDecoder",
    "DifferentialNormalization",
    "DifferentialTransform",
    "LinearDifferentialTransform",
]
