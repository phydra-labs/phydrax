#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp

from ..._interpolation import barycentric_differentiation_matrix


def _fd_first_derivative(
    y: jax.Array, /, *, dx: jax.Array, axis: int, periodic: bool
) -> jax.Array:
    dx_ = jnp.asarray(dx, dtype=float).reshape(())
    if periodic:
        return (jnp.roll(y, -1, axis=axis) - jnp.roll(y, 1, axis=axis)) / (2.0 * dx_)

    y0 = jnp.moveaxis(y, axis, 0)
    n = y0.shape[0]
    if n < 2:
        return jnp.zeros_like(y)
    out0 = jnp.zeros_like(y0)
    out0 = out0.at[1:-1].set((y0[2:] - y0[:-2]) / (2.0 * dx_))
    out0 = out0.at[0].set((y0[1] - y0[0]) / dx_)
    out0 = out0.at[-1].set((y0[-1] - y0[-2]) / dx_)
    return jnp.moveaxis(out0, 0, axis)


def _fd_nth_derivative(
    y: jax.Array, /, *, dx: jax.Array, axis: int, order: int, periodic: bool
) -> jax.Array:
    order_i = int(order)

    def _step(_: int, out_i: jax.Array) -> jax.Array:
        return _fd_first_derivative(out_i, dx=dx, axis=axis, periodic=periodic)

    return jax.lax.fori_loop(0, order_i, _step, y)


def _poly_nth_derivative(
    y: jax.Array, x: jax.Array, /, *, axis: int, order: int
) -> jax.Array:
    order_i = int(order)
    matrix = barycentric_differentiation_matrix(x)

    def _step(_: int, out_i: jax.Array) -> jax.Array:
        out0 = jnp.moveaxis(out_i, axis, 0)
        n = int(out0.shape[0])
        flat = out0.reshape((n, -1))
        differentiated = matrix @ flat
        return jnp.moveaxis(differentiated.reshape(out0.shape), 0, axis)

    return jax.lax.fori_loop(0, order_i, _step, y)


def _fourier_nth_derivative(
    y: jax.Array, x: jax.Array, /, *, axis: int, order: int
) -> jax.Array:
    x1 = jnp.asarray(x, dtype=float).reshape((-1,))
    n = int(x1.shape[0])
    if n < 2:
        return jnp.zeros_like(y)

    dx = x1[1] - x1[0]
    frequencies = 2.0 * jnp.pi * jnp.fft.fftfreq(n, d=dx)
    shape = [1] * y.ndim
    shape[int(axis)] = n
    frequencies = frequencies.reshape(tuple(shape))

    coefficients = jnp.fft.fft(y, axis=axis)
    multiplier = (1j * frequencies) ** int(order)
    derivative = jnp.fft.ifft(multiplier * coefficients, axis=axis)
    if not jnp.iscomplexobj(y):
        derivative = jnp.real(derivative)
    return derivative


def _cosine_nth_derivative(
    y: jax.Array, x: jax.Array, /, *, axis: int, order: int
) -> jax.Array:
    x1 = jnp.asarray(x, dtype=float).reshape((-1,))
    n = int(x1.shape[0])
    if n < 2:
        return jnp.zeros_like(y)

    dx = x1[1] - x1[0]
    extended_size = 2 * (n - 1)
    y0 = jnp.moveaxis(y, axis, 0)
    extended = jnp.concatenate([y0, y0[-2:0:-1]], axis=0)

    frequencies = 2.0 * jnp.pi * jnp.fft.fftfreq(extended_size, d=dx)
    shape = [extended_size] + [1] * (extended.ndim - 1)
    frequencies = frequencies.reshape(tuple(shape))

    coefficients = jnp.fft.fft(extended, axis=0)
    multiplier = (1j * frequencies) ** int(order)
    derivative = jnp.fft.ifft(multiplier * coefficients, axis=0)[:n]
    if not jnp.iscomplexobj(y):
        derivative = jnp.real(derivative)
    return jnp.moveaxis(derivative, 0, axis)


def _sine_nth_derivative(
    y: jax.Array, x: jax.Array, /, *, axis: int, order: int
) -> jax.Array:
    x1 = jnp.asarray(x, dtype=float).reshape((-1,))
    n = int(x1.shape[0])
    if n < 2:
        return jnp.zeros_like(y)

    dx = x1[1] - x1[0]
    extended_size = 2 * n
    y0 = jnp.moveaxis(y, axis, 0)
    extended = jnp.concatenate([y0, -y0[::-1]], axis=0)

    frequencies = 2.0 * jnp.pi * jnp.fft.fftfreq(extended_size, d=dx)
    shape = [extended_size] + [1] * (extended.ndim - 1)
    frequencies = frequencies.reshape(tuple(shape))

    coefficients = jnp.fft.fft(extended, axis=0)
    multiplier = (1j * frequencies) ** int(order)
    derivative = jnp.fft.ifft(multiplier * coefficients, axis=0)[:n]
    if not jnp.iscomplexobj(y):
        derivative = jnp.real(derivative)
    return jnp.moveaxis(derivative, 0, axis)


def _basis_nth_derivative(
    y: jax.Array,
    x: jax.Array,
    /,
    *,
    axis: int,
    order: int,
    basis: Literal["poly", "fourier", "sine", "cosine"],
) -> jax.Array:
    if int(order) == 0:
        return y
    if basis == "fourier":
        return _fourier_nth_derivative(y, x, axis=axis, order=order)
    if basis == "cosine":
        return _cosine_nth_derivative(y, x, axis=axis, order=order)
    if basis == "sine":
        return _sine_nth_derivative(y, x, axis=axis, order=order)
    return _poly_nth_derivative(y, x, axis=axis, order=order)


__all__ = [
    "_basis_nth_derivative",
    "_cosine_nth_derivative",
    "_fd_first_derivative",
    "_fd_nth_derivative",
    "_fourier_nth_derivative",
    "_poly_nth_derivative",
    "_sine_nth_derivative",
]
