#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax import ein

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


NUFFTType = Literal[1, 2]
NUFFTMethod = Literal["direct", "gridded"]


class NUFFTPlan(StrictModule):
    """Static finite Fourier transform convention and resource policy."""

    mode_shape: tuple[int, ...] = eqx.field(static=True)
    transform_type: NUFFTType = eqx.field(static=True)
    sign: int = eqx.field(static=True)
    centered: bool = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    method: NUFFTMethod = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mode_shape: Sequence[int],
        transform_type: NUFFTType,
        /,
        *,
        sign: int = 1,
        centered: bool = False,
        tolerance: float = 1e-6,
        method: NUFFTMethod = "direct",
    ):
        shape = tuple(int(size) for size in mode_shape)
        if not shape or len(shape) > 3 or any(size <= 0 for size in shape):
            raise ValueError("NUFFT mode_shape must contain one to three positive sizes.")
        if transform_type not in (1, 2):
            raise ValueError("NUFFT transform_type must be one or two.")
        if sign not in (-1, 1):
            raise ValueError("NUFFT sign must be -1 or 1.")
        tolerance_ = float(tolerance)
        if not 0.0 < tolerance_ < 1.0:
            raise ValueError("NUFFT tolerance must lie in (0, 1).")
        if method not in ("direct", "gridded"):
            raise ValueError("Unknown NUFFT method.")
        self.mode_shape = shape
        self.transform_type = transform_type
        self.sign = int(sign)
        self.centered = bool(centered)
        self.tolerance = tolerance_
        self.method = method
        self.plan_id = canonical_fingerprint(
            {
                "kind": "native-nufft",
                "mode_shape": shape,
                "type": transform_type,
                "sign": sign,
                "centered": centered,
                "tolerance": tolerance_,
                "method": method,
            }
        )


class PreparedNUFFT(StrictModule, NonTrainableState):
    """Prepared direct finite Fourier mode arrays."""

    modes: tuple[Array, ...]
    plan: NUFFTPlan = eqx.field(static=True)

    def __init__(self, plan: NUFFTPlan, /, *, dtype=float):
        if not isinstance(plan, NUFFTPlan):
            raise TypeError("plan must be a NUFFTPlan.")
        modes = []
        for size in plan.mode_shape:
            if plan.centered:
                values = jnp.arange(size, dtype=dtype) - size // 2
            else:
                values = jnp.fft.fftfreq(size).astype(dtype) * size
            modes.append(values)
        self.modes = tuple(modes)
        self.plan = plan

    def type2(self, coordinates: ArrayLike, coefficients: ArrayLike, /) -> Array:
        if self.plan.transform_type != 2:
            raise ValueError("Prepared plan is not Type 2.")
        points = jnp.asarray(coordinates)
        values = jnp.asarray(coefficients)
        if points.ndim != 2 or points.shape[1] != len(self.modes):
            raise ValueError("NUFFT coordinates must have shape (points, dimension).")
        if values.shape[: len(self.modes)] != self.plan.mode_shape:
            raise ValueError("NUFFT coefficients do not match mode_shape.")
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

    def type1(self, coordinates: ArrayLike, strengths: ArrayLike, /) -> Array:
        if self.plan.transform_type != 1:
            raise ValueError("Prepared plan is not Type 1.")
        points = jnp.asarray(coordinates)
        values = jnp.asarray(strengths)
        if points.ndim != 2 or points.shape[1] != len(self.modes):
            raise ValueError("NUFFT coordinates must have shape (points, dimension).")
        if values.shape[0] != points.shape[0]:
            raise ValueError("NUFFT strengths need one leading value per point.")
        result = values
        for axis, mode in enumerate(self.modes):
            phase = jnp.exp(
                1j * self.plan.sign * points[:, axis, None] * mode[None, :]
            ).astype(values.dtype)
            result = ein.contract("m...,mk->m...k", result, phase)
        return ein.contract("m...->...", result)


__all__ = ["NUFFTMethod", "NUFFTPlan", "NUFFTType", "PreparedNUFFT"]
