#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.experimental.jet import jet


def _jet_terms_real(
    fun: Callable[[jax.Array], jax.Array],
    x: jax.Array,
    v: jax.Array,
    order: int,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    r"""Internal helper for Jet expansions on real-valued functions.

    Returns the primal output and the first `order` directional derivative terms along
    direction $v$, as computed by `jax.experimental.jet`.
    """
    order_i = int(order)
    if order_i < 1:
        raise ValueError("order must be >= 1.")

    x = jnp.asarray(x)
    v = jnp.asarray(v)
    zeros = jnp.zeros_like(x)
    series_in = (tuple([v] + [zeros] * (order_i - 1)),)
    primal_out, series_out = jet(fun, (x,), series_in)
    return primal_out, tuple(series_out)


def _jet_terms_real_multi(
    fun: Callable[..., jax.Array],
    primals: tuple[jax.Array, ...],
    tangents: tuple[jax.Array, ...],
    order: int,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    r"""Internal helper for Jet expansions with multiple primals/tangents (real-valued)."""
    order_i = int(order)
    if order_i < 1:
        raise ValueError("order must be >= 1.")

    primals = tuple(jnp.asarray(p) for p in primals)
    tangents = tuple(jnp.asarray(v) for v in tangents)

    series_in = tuple(
        tuple([v] + [jnp.zeros_like(p)] * (order_i - 1))
        for p, v in zip(primals, tangents, strict=True)
    )
    primal_out, series_out = jet(fun, primals, series_in)
    return primal_out, tuple(series_out)


def jet_terms(
    fun: Callable[[jax.Array], jax.Array],
    x: jax.Array,
    v: jax.Array,
    *,
    order: int,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    x = jnp.asarray(x)
    y0 = fun(x)
    if jnp.iscomplexobj(y0):
        r0, rs = _jet_terms_real(lambda z: jnp.real(fun(z)), x, v, int(order))
        i0, is_ = _jet_terms_real(lambda z: jnp.imag(fun(z)), x, v, int(order))
        return r0 + 1j * i0, tuple(r + 1j * i for r, i in zip(rs, is_, strict=True))
    return _jet_terms_real(fun, x, v, int(order))


def jet_terms_multi(
    fun: Callable[..., jax.Array],
    primals: tuple[jax.Array, ...],
    tangents: tuple[jax.Array, ...],
    *,
    order: int,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    r"""Compute Jet-series directional derivative terms for multi-argument functions."""
    primals = tuple(jnp.asarray(p) for p in primals)
    y0 = fun(*primals)
    if jnp.iscomplexobj(y0):
        r0, rs = _jet_terms_real_multi(
            lambda *args: jnp.real(fun(*args)), primals, tangents, int(order)
        )
        i0, is_ = _jet_terms_real_multi(
            lambda *args: jnp.imag(fun(*args)), primals, tangents, int(order)
        )
        return r0 + 1j * i0, tuple(r + 1j * i for r, i in zip(rs, is_, strict=True))
    return _jet_terms_real_multi(fun, primals, tangents, int(order))


def jet_dn(
    fun: Callable[[jax.Array], jax.Array],
    x: jax.Array,
    v: jax.Array,
    *,
    n: int,
) -> jax.Array:
    r"""Return the $n$-th Jet directional term along $v$.

    If `jet_terms(..., order=n)` returns `(y0, (d1, d2, ..., dn))`, then this returns `dn`.
    """
    _, terms = jet_terms(fun, x, v, order=int(n))
    return terms[int(n) - 1]


def jet_dn_multi(
    fun: Callable[..., jax.Array],
    primals: tuple[jax.Array, ...],
    tangents: tuple[jax.Array, ...],
    *,
    n: int,
) -> jax.Array:
    r"""Multi-argument version of `jet_dn`."""
    _, terms = jet_terms_multi(fun, primals, tangents, order=int(n))
    return terms[int(n) - 1]


def jet_d1_d2(
    fun: Callable[[jax.Array], jax.Array],
    x: jax.Array,
    v: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    r"""Convenience helper returning the first and second Jet directional terms."""
    _, (d1, d2) = jet_terms(fun, x, v, order=2)
    return d1, d2
