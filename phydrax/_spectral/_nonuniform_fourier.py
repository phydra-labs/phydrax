#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax import ein

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


NonuniformFourierType = Literal[1, 2]
NonuniformFourierRoute = Literal["direct", "chunked"]


def _static_int(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    return int(value)


def _complex_dtype(*arrays: Array) -> jnp.dtype:
    real_dtype = jnp.result_type(
        *(array.real.dtype for array in arrays),
        jnp.float32,
    )
    return jnp.dtype(jnp.complex64 if real_dtype.itemsize <= 4 else jnp.complex128)


class NonuniformFourierPlan(StrictModule):
    """Static exact nonuniform Fourier convention and resource policy."""

    mode_shape: tuple[int, ...] = eqx.field(static=True)
    transform_type: NonuniformFourierType = eqx.field(static=True)
    sign: int = eqx.field(static=True)
    centered: bool = eqx.field(static=True)
    route: NonuniformFourierRoute = eqx.field(static=True)
    chunk_size: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mode_shape: Sequence[int],
        transform_type: NonuniformFourierType,
        /,
        *,
        sign: int = 1,
        centered: bool = False,
        route: NonuniformFourierRoute = "direct",
        chunk_size: int = 256,
    ):
        supplied_shape = tuple(mode_shape)
        if not supplied_shape or len(supplied_shape) > 3:
            raise ValueError(
                "Nonuniform Fourier mode_shape must contain one to three positive sizes."
            )
        shape = tuple(
            _static_int(size, "Nonuniform Fourier mode size") for size in supplied_shape
        )
        if any(size <= 0 for size in shape):
            raise ValueError(
                "Nonuniform Fourier mode_shape must contain one to three positive sizes."
            )
        transform = _static_int(transform_type, "Nonuniform Fourier transform_type")
        if transform not in (1, 2):
            raise ValueError("Nonuniform Fourier transform_type must be one or two.")
        exponent_sign = _static_int(sign, "Nonuniform Fourier sign")
        if exponent_sign not in (-1, 1):
            raise ValueError("Nonuniform Fourier sign must be -1 or 1.")
        if not isinstance(centered, bool):
            raise TypeError("Nonuniform Fourier centered must be a bool.")
        if route not in ("direct", "chunked"):
            raise ValueError("Unknown nonuniform Fourier route.")
        chunk = _static_int(chunk_size, "Nonuniform Fourier chunk_size")
        if chunk < 1:
            raise ValueError("Nonuniform Fourier chunk_size must be positive.")
        self.mode_shape = shape
        self.transform_type = transform
        self.sign = exponent_sign
        self.centered = centered
        self.route = route
        self.chunk_size = chunk
        self.plan_id = canonical_fingerprint(
            {
                "kind": "nonuniform-fourier-plan",
                "mode_shape": shape,
                "type": transform,
                "sign": exponent_sign,
                "centered": centered,
                "route": route,
                "chunk_size": chunk,
            }
        )


class PreparedNonuniformFourier(StrictModule, NonTrainableState):
    """Prepared exact nonuniform Fourier mode arrays."""

    modes: tuple[Array, ...]
    plan: NonuniformFourierPlan = eqx.field(static=True)

    def __init__(
        self,
        plan: NonuniformFourierPlan,
        /,
        *,
        dtype: jnp.dtype = jnp.float32,
    ):
        if not isinstance(plan, NonuniformFourierPlan):
            raise TypeError("plan must be a NonuniformFourierPlan.")
        resolved_dtype = jnp.dtype(dtype)
        if not jnp.issubdtype(resolved_dtype, jnp.floating):
            raise TypeError("Nonuniform Fourier modes require a real floating dtype.")
        modes = []
        for size in plan.mode_shape:
            if plan.centered:
                values = jnp.arange(size, dtype=resolved_dtype) - size // 2
            else:
                values = jnp.fft.fftfreq(size).astype(resolved_dtype) * size
            modes.append(values)
        self.modes = tuple(modes)
        self.plan = plan

    def _type2_direct(self, points: Array, values: Array, /) -> Array:
        dtype = jnp.result_type(values.dtype, _complex_dtype(points, values, *self.modes))
        result = values.astype(dtype)
        imaginary_unit = jnp.asarray(1j, dtype=dtype)
        for position, (mode, coordinate) in enumerate(
            zip(self.modes, points.T, strict=True)
        ):
            angle = self.plan.sign * coordinate[:, None] * mode[None, :]
            phase = jnp.exp(imaginary_unit * angle)
            result = (
                ein.contract("mk,k...->m...", phase, result)
                if position == 0
                else ein.contract("mk,mk...->m...", phase, result)
            )
        return result

    def type2(self, coordinates: ArrayLike, coefficients: ArrayLike, /) -> Array:
        if self.plan.transform_type != 2:
            raise ValueError("Prepared plan is not Type 2.")
        points = jnp.asarray(coordinates)
        values = jnp.asarray(coefficients)
        if points.ndim != 2 or points.shape[1] != len(self.modes):
            raise ValueError(
                "Nonuniform Fourier coordinates must have shape (points, dimension)."
            )
        if jnp.issubdtype(points.dtype, jnp.complexfloating):
            raise TypeError("Nonuniform Fourier coordinates must be real-valued.")
        if values.shape[: len(self.modes)] != self.plan.mode_shape:
            raise ValueError("Nonuniform Fourier coefficients do not match mode_shape.")
        point_count = points.shape[0]
        if self.plan.route == "direct" or point_count == 0:
            return self._type2_direct(points, values)
        chunk_size = min(self.plan.chunk_size, point_count)
        chunk_count = (point_count + chunk_size - 1) // chunk_size
        padded_count = chunk_count * chunk_size
        if padded_count != point_count:
            points = jnp.concatenate(
                (
                    points,
                    jnp.zeros(
                        (padded_count - point_count, points.shape[1]),
                        dtype=points.dtype,
                    ),
                ),
                axis=0,
            )
        chunks = points.reshape((chunk_count, chunk_size, points.shape[1]))
        outputs = jax.lax.map(
            lambda chunk: self._type2_direct(chunk, values),
            chunks,
        )
        return outputs.reshape((padded_count, *outputs.shape[2:]))[:point_count]

    def _type1_direct(self, points: Array, values: Array, /) -> Array:
        dtype = jnp.result_type(values.dtype, _complex_dtype(points, values, *self.modes))
        result = values.astype(dtype)
        imaginary_unit = jnp.asarray(1j, dtype=dtype)
        for axis, mode in enumerate(self.modes):
            angle = self.plan.sign * points[:, axis, None] * mode[None, :]
            phase = jnp.exp(imaginary_unit * angle)
            result = ein.contract("m...,mk->m...k", result, phase)
        return ein.contract("m...->...", result)

    def type1(self, coordinates: ArrayLike, strengths: ArrayLike, /) -> Array:
        if self.plan.transform_type != 1:
            raise ValueError("Prepared plan is not Type 1.")
        points = jnp.asarray(coordinates)
        values = jnp.asarray(strengths)
        if points.ndim != 2 or points.shape[1] != len(self.modes):
            raise ValueError(
                "Nonuniform Fourier coordinates must have shape (points, dimension)."
            )
        if jnp.issubdtype(points.dtype, jnp.complexfloating):
            raise TypeError("Nonuniform Fourier coordinates must be real-valued.")
        if values.shape[0] != points.shape[0]:
            raise ValueError(
                "Nonuniform Fourier strengths need one leading value per point."
            )
        point_count = points.shape[0]
        if self.plan.route == "direct" or point_count == 0:
            return self._type1_direct(points, values)
        chunk_size = min(self.plan.chunk_size, point_count)
        chunk_count = (point_count + chunk_size - 1) // chunk_size
        padded_count = chunk_count * chunk_size
        if padded_count != point_count:
            padding = padded_count - point_count
            points = jnp.concatenate(
                (
                    points,
                    jnp.zeros((padding, points.shape[1]), dtype=points.dtype),
                ),
                axis=0,
            )
            values = jnp.concatenate(
                (
                    values,
                    jnp.zeros((padding, *values.shape[1:]), dtype=values.dtype),
                ),
                axis=0,
            )
        point_chunks = points.reshape((chunk_count, chunk_size, points.shape[1]))
        value_chunks = values.reshape((chunk_count, chunk_size, *values.shape[1:]))
        outputs = jax.lax.map(
            lambda data: self._type1_direct(data[0], data[1]),
            (point_chunks, value_chunks),
        )
        return jnp.sum(outputs, axis=0)


__all__ = [
    "NonuniformFourierPlan",
    "NonuniformFourierRoute",
    "NonuniformFourierType",
    "PreparedNonuniformFourier",
]
