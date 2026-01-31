#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
import jax.random as jr
import opt_einsum as oe
from jaxtyping import Array, Key

from ...._doc import DOC_KEY0
from ...._strict import StrictModule
from ..._utils import _get_size
from ..core._base import _AbstractStructuredInputModel
from ..layers._linear import Linear


class SpectralConv1d(StrictModule):
    r"""Complex-valued spectral convolution on a 1D periodic grid.

    This layer takes an input with shape `(..., n, c_in)`, applies an FFT along the
    spatial axis, multiplies a fixed number of low-frequency modes by learned complex
    weights, and transforms back:

    $$
    \hat y_k = W_k\,\hat x_k \quad\text{for } k=0,\dots,m-1,\qquad
    y = \text{IFFT}(\hat y).
    $$
    """

    in_channels: int
    out_channels: int
    modes: int
    weight: Array

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        modes: int,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes = int(modes)
        if self.in_channels <= 0 or self.out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive.")
        if self.modes <= 0:
            raise ValueError("modes must be positive.")

        k_r, k_i = jr.split(key, 2)
        scale = 1.0 / float(self.in_channels * self.out_channels)
        w_r = scale * jr.normal(
            k_r, shape=(self.in_channels, self.out_channels, self.modes)
        )
        w_i = scale * jr.normal(
            k_i, shape=(self.in_channels, self.out_channels, self.modes)
        )
        self.weight = w_r + 1j * w_i

    def __call__(self, x: Array, /) -> Array:
        x_in = jnp.asarray(x)
        if x_in.ndim < 2:
            raise ValueError("SpectralConv1d expects input with shape (..., n, c).")
        if int(x_in.shape[-1]) != self.in_channels:
            raise ValueError(
                "SpectralConv1d channel mismatch: expected in_channels="
                f"{self.in_channels} but got {int(x_in.shape[-1])}."
            )

        n = int(x_in.shape[-2])
        x_ft = jnp.fft.rfft(x_in, axis=-2)
        n_ft = int(x_ft.shape[-2])
        m = min(int(self.modes), n_ft)

        out_ft = jnp.zeros((*x_ft.shape[:-1], self.out_channels), dtype=x_ft.dtype)
        w = self.weight[..., :m]
        out_low = oe.contract("...mi,iom->...mo", x_ft[..., :m, :], w)
        out_ft = out_ft.at[..., :m, :].set(out_low)
        return jnp.fft.irfft(out_ft, n=n, axis=-2)


class SpectralConv2d(StrictModule):
    r"""Complex-valued spectral convolution on a 2D periodic grid.

    Input shape is `(..., n_x, n_y, c_in)`. Uses an FFT over the two spatial axes and
    applies learned complex weights to a rectangle of low-frequency modes.
    """

    in_channels: int
    out_channels: int
    modes_x: int
    modes_y: int
    weight: Array

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        modes_x: int,
        modes_y: int,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes_x = int(modes_x)
        self.modes_y = int(modes_y)
        if self.in_channels <= 0 or self.out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive.")
        if self.modes_x <= 0 or self.modes_y <= 0:
            raise ValueError("modes_x and modes_y must be positive.")

        k_r, k_i = jr.split(key, 2)
        scale = 1.0 / float(self.in_channels * self.out_channels)
        w_r = scale * jr.normal(
            k_r,
            shape=(self.in_channels, self.out_channels, self.modes_x, self.modes_y),
        )
        w_i = scale * jr.normal(
            k_i,
            shape=(self.in_channels, self.out_channels, self.modes_x, self.modes_y),
        )
        self.weight = w_r + 1j * w_i

    def __call__(self, x: Array, /) -> Array:
        x_in = jnp.asarray(x)
        if x_in.ndim < 3:
            raise ValueError("SpectralConv2d expects input with shape (..., nx, ny, c).")
        if int(x_in.shape[-1]) != self.in_channels:
            raise ValueError(
                "SpectralConv2d channel mismatch: expected in_channels="
                f"{self.in_channels} but got {int(x_in.shape[-1])}."
            )

        nx = int(x_in.shape[-3])
        ny = int(x_in.shape[-2])
        x_ft = jnp.fft.rfftn(x_in, axes=(-3, -2))
        ny_ft = int(x_ft.shape[-2])

        mx = min(int(self.modes_x), int(x_ft.shape[-3]))
        my = min(int(self.modes_y), ny_ft)

        out_ft = jnp.zeros((*x_ft.shape[:-1], self.out_channels), dtype=x_ft.dtype)
        w = self.weight[..., :mx, :my]
        out_low = oe.contract("...xyi,ioxy->...xyo", x_ft[..., :mx, :my, :], w)
        out_ft = out_ft.at[..., :mx, :my, :].set(out_low)
        return jnp.fft.irfftn(out_ft, s=(nx, ny), axes=(-3, -2))


class FNO1d(_AbstractStructuredInputModel):
    r"""Minimal 1D Fourier Neural Operator for coord-separable grid evaluation.

    **Input convention (structured tuple):**

    - `data`: grid values with leading axis length `n` (shape `(n,)` or `(n, c_in)`),
    - `x_axis`: 1D coordinate axis of shape `(n,)` (must have `n>1`).

    The axis values are used for sanity checking and to enforce "grid mode" usage.
    """

    in_size: int | Literal["scalar"]
    out_size: int | Literal["scalar"]
    modes: int
    width: int

    lift: Linear
    spectral_layers: tuple[SpectralConv1d, ...]
    pointwise_layers: tuple[Linear, ...]
    proj: Linear

    def __init__(
        self,
        *,
        in_channels: int | Literal["scalar"] = "scalar",
        out_channels: int | Literal["scalar"] = "scalar",
        width: int = 32,
        depth: int = 4,
        modes: int = 16,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.in_size = in_channels
        self.out_size = out_channels
        self.width = int(width)
        self.modes = int(modes)

        if self.width <= 0:
            raise ValueError("width must be positive.")
        if int(depth) <= 0:
            raise ValueError("depth must be positive.")

        in_ch = _get_size(in_channels)
        out_ch = _get_size(out_channels)

        keys = jr.split(key, int(depth) * 2 + 3)
        self.lift = Linear(
            in_size=in_ch, out_size=self.width, activation=None, key=keys[0]
        )

        spectral = []
        pointwise = []
        for i in range(int(depth)):
            spectral.append(
                SpectralConv1d(
                    in_channels=self.width,
                    out_channels=self.width,
                    modes=self.modes,
                    key=keys[1 + 2 * i],
                )
            )
            pointwise.append(
                Linear(
                    in_size=self.width,
                    out_size=self.width,
                    activation=None,
                    key=keys[2 + 2 * i],
                )
            )
        self.spectral_layers = tuple(spectral)
        self.pointwise_layers = tuple(pointwise)

        self.proj = Linear(
            in_size=self.width, out_size=out_ch, activation=None, key=keys[-1]
        )

    def __call__(
        self,
        x: Array | tuple[Array, ...],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> Array:
        del key
        if not isinstance(x, tuple):
            raise ValueError("FNO1d requires a structured tuple input.")
        if len(x) != 2:
            raise ValueError("FNO1d expects exactly (data, x_axis) inputs.")

        data = jnp.asarray(x[0])
        x_axis = jnp.asarray(x[1])
        if x_axis.ndim != 1:
            raise ValueError("FNO1d expects a 1D coordinate axis array for x.")
        if int(x_axis.shape[0]) <= 1:
            raise ValueError(
                "FNO1d expects coord-separable grid evaluation; got a point-like x input."
            )

        if data.ndim == 0:
            raise ValueError("FNO1d expects data to have a leading spatial axis.")
        if int(data.shape[0]) != int(x_axis.shape[0]):
            raise ValueError(
                "FNO1d expects data leading axis to match x_axis length; got "
                f"{int(data.shape[0])} and {int(x_axis.shape[0])}."
            )

        n = int(x_axis.shape[0])
        xin = jnp.asarray(data).reshape((n, -1))
        if int(xin.shape[-1]) != _get_size(self.in_size):
            raise ValueError(
                "FNO1d input channel mismatch: expected "
                f"{_get_size(self.in_size)} but got {int(xin.shape[-1])}."
            )

        xw = self.lift(xin)
        xw = jnp.tanh(xw)
        for spec, pw in zip(self.spectral_layers, self.pointwise_layers, strict=True):
            xw = jnp.tanh(spec(xw) + pw(xw))
        y = self.proj(xw)
        if self.out_size == "scalar":
            return y[..., 0]
        return y


class FNO2d(_AbstractStructuredInputModel):
    r"""Minimal 2D Fourier Neural Operator for coord-separable grid evaluation.

    **Input convention (structured tuple):**

    - `data`: grid values with leading axes `(n_x, n_y)` (optionally with channels),
    - `x_axis`: 1D coordinate axis of shape `(n_x,)` (must have `n_x>1`),
    - `y_axis`: 1D coordinate axis of shape `(n_y,)` (must have `n_y>1`).

    The axis values are used for sanity checking and to enforce "grid mode" usage.
    """

    in_size: int | Literal["scalar"]
    out_size: int | Literal["scalar"]
    modes_x: int
    modes_y: int
    width: int

    lift: Linear
    spectral_layers: tuple[SpectralConv2d, ...]
    pointwise_layers: tuple[Linear, ...]
    proj: Linear

    def __init__(
        self,
        *,
        in_channels: int | Literal["scalar"] = "scalar",
        out_channels: int | Literal["scalar"] = "scalar",
        width: int = 32,
        depth: int = 4,
        modes: int = 12,
        modes_y: int | None = None,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.in_size = in_channels
        self.out_size = out_channels
        self.width = int(width)
        self.modes_x = int(modes)
        self.modes_y = int(modes if modes_y is None else modes_y)

        if self.width <= 0:
            raise ValueError("width must be positive.")
        if int(depth) <= 0:
            raise ValueError("depth must be positive.")

        in_ch = _get_size(in_channels)
        out_ch = _get_size(out_channels)

        keys = jr.split(key, int(depth) * 2 + 3)
        self.lift = Linear(
            in_size=in_ch, out_size=self.width, activation=None, key=keys[0]
        )

        spectral = []
        pointwise = []
        for i in range(int(depth)):
            spectral.append(
                SpectralConv2d(
                    in_channels=self.width,
                    out_channels=self.width,
                    modes_x=self.modes_x,
                    modes_y=self.modes_y,
                    key=keys[1 + 2 * i],
                )
            )
            pointwise.append(
                Linear(
                    in_size=self.width,
                    out_size=self.width,
                    activation=None,
                    key=keys[2 + 2 * i],
                )
            )
        self.spectral_layers = tuple(spectral)
        self.pointwise_layers = tuple(pointwise)

        self.proj = Linear(
            in_size=self.width, out_size=out_ch, activation=None, key=keys[-1]
        )

    def __call__(
        self,
        x: Array | tuple[Array, ...],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> Array:
        del key
        if not isinstance(x, tuple):
            raise ValueError("FNO2d requires a structured tuple input.")
        if len(x) != 3:
            raise ValueError("FNO2d expects exactly (data, x_axis, y_axis) inputs.")

        data = jnp.asarray(x[0])
        x_axis = jnp.asarray(x[1])
        y_axis = jnp.asarray(x[2])
        if x_axis.ndim != 1 or y_axis.ndim != 1:
            raise ValueError("FNO2d expects 1D coordinate axis arrays for x and y.")
        if int(x_axis.shape[0]) <= 1 or int(y_axis.shape[0]) <= 1:
            raise ValueError(
                "FNO2d expects coord-separable grid evaluation; got point-like coordinate input."
            )

        if data.ndim < 2:
            raise ValueError("FNO2d expects data to have leading spatial axes.")
        if int(data.shape[0]) != int(x_axis.shape[0]) or int(data.shape[1]) != int(
            y_axis.shape[0]
        ):
            raise ValueError(
                "FNO2d expects data leading axes to match (x_axis, y_axis) lengths; got "
                f"{tuple(int(s) for s in data.shape[:2])} and "
                f"({int(x_axis.shape[0])}, {int(y_axis.shape[0])})."
            )

        nx = int(x_axis.shape[0])
        ny = int(y_axis.shape[0])
        xin = jnp.asarray(data).reshape((nx, ny, -1))
        if int(xin.shape[-1]) != _get_size(self.in_size):
            raise ValueError(
                "FNO2d input channel mismatch: expected "
                f"{_get_size(self.in_size)} but got {int(xin.shape[-1])}."
            )

        xw = self.lift(xin)
        xw = jnp.tanh(xw)
        for spec, pw in zip(self.spectral_layers, self.pointwise_layers, strict=True):
            xw = jnp.tanh(spec(xw) + pw(xw))
        y = self.proj(xw)
        if self.out_size == "scalar":
            return y[..., 0]
        return y


__all__ = ["FNO1d", "FNO2d", "SpectralConv1d", "SpectralConv2d"]
