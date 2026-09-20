#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
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
        shape = tuple(mode_shape)
        if not shape or len(shape) > 3 or any(size <= 0 for size in shape):
            raise ValueError(
                "Nonuniform Fourier mode_shape must contain one to three positive sizes."
            )
        if transform_type not in (1, 2):
            raise ValueError("Nonuniform Fourier transform_type must be one or two.")
        if sign not in (-1, 1):
            raise ValueError("Nonuniform Fourier sign must be -1 or 1.")
        if route not in ("direct", "chunked"):
            raise ValueError("Unknown nonuniform Fourier route.")
        chunk = int(chunk_size)
        if chunk < 1:
            raise ValueError("Nonuniform Fourier chunk_size must be positive.")
        self.mode_shape = shape
        self.transform_type = transform_type
        self.sign = sign
        self.centered = centered
        self.route = route
        self.chunk_size = chunk
        self.plan_id = canonical_fingerprint(
            {
                "kind": "nonuniform-fourier-plan",
                "mode_shape": shape,
                "type": transform_type,
                "sign": sign,
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
        modes = []
        for size in plan.mode_shape:
            if plan.centered:
                values = jnp.arange(size, dtype=dtype) - size // 2
            else:
                values = jnp.fft.fftfreq(size).astype(dtype) * size
            modes.append(values)
        self.modes = tuple(modes)
        self.plan = plan

    def _type2_direct(self, points: Array, values: Array, /) -> Array:
        result = values
        for position, (mode, coordinate) in enumerate(
            zip(self.modes, points.T, strict=True)
        ):
            phase = jnp.exp(
                1j * self.plan.sign * coordinate[:, None] * mode[None, :]
            ).astype(values.dtype)
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
        result = values
        for axis, mode in enumerate(self.modes):
            phase = jnp.exp(
                1j * self.plan.sign * points[:, axis, None] * mode[None, :]
            ).astype(values.dtype)
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
