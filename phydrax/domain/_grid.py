#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
from typing import Literal

import jax
import jax.numpy as jnp
import orthax
from jaxtyping import Array

from .._strict import StrictModule


class AxisDiscretization(StrictModule):
    r"""Materialized 1D discretization data for a single coordinate axis.

    This bundles:

    - `nodes`: 1D coordinates \(x_j\) along an axis.
    - `quad_weights`: optional 1D quadrature weights \(w_j\) for approximating 1D integrals.
    - `basis`: a hint describing how the axis was constructed (`"fourier"`, `"sine"`,
      `"cosine"`, `"legendre"`, `"uniform"`),
    - `periodic`: whether the axis should be treated as periodic (useful for FFT/Fourier methods).

    When `quad_weights` are present, they approximate

    $$
    \int_a^b f(x)\,dx \approx \sum_j w_j f(x_j),
    $$
    """

    nodes: Array
    quad_weights: Array | None
    basis: Literal["uniform", "fourier", "sine", "cosine", "legendre"]
    periodic: bool

    def __init__(
        self,
        *,
        nodes: Array,
        quad_weights: Array | None,
        basis: Literal["uniform", "fourier", "sine", "cosine", "legendre"],
        periodic: bool,
    ):
        nodes_ = jnp.asarray(nodes, dtype=float).reshape((-1,))
        if nodes_.size == 0:
            raise ValueError("AxisDiscretization.nodes must be non-empty.")
        if quad_weights is not None:
            w = jnp.asarray(quad_weights, dtype=float).reshape((-1,))
            if w.shape != nodes_.shape:
                raise ValueError(
                    "AxisDiscretization.quad_weights must have the same shape as nodes."
                )
            self.quad_weights = w
        else:
            self.quad_weights = None
        self.nodes = nodes_
        self.basis = basis
        self.periodic = bool(periodic)


class AbstractAxisSpec(StrictModule):
    r"""Abstract base class for 1D grid/basis axis specifications.

    An `AxisSpec` is an instruction for how to discretize a 1D coordinate axis on
    an interval \([a,b]\). Calling `materialize(a, b)` produces an `AxisDiscretization`
    with nodes (and possibly quadrature weights) for that axis.
    """

    n: int

    def __init__(self, n: int):
        n_ = int(n)
        if n_ <= 0:
            raise ValueError("AxisSpec n must be positive.")
        self.n = n_

    @abc.abstractmethod
    def materialize(self, a: Array, b: Array, /) -> AxisDiscretization:
        raise NotImplementedError


class GridSpec(StrictModule):
    """A per-label grid spec: one axis spec per coordinate component.

    For a geometry variable with `var_dim=d`, use `GridSpec(axes=(spec0, ..., spec{d-1}))`
    to specify a different `AxisSpec` per coordinate axis.
    """

    axes: tuple[AbstractAxisSpec, ...]

    def __init__(self, axes: Sequence[AbstractAxisSpec]):
        axes_ = tuple(axes)
        if not axes_:
            raise ValueError("GridSpec.axes must be non-empty.")
        self.axes = axes_


class UniformAxisSpec(AbstractAxisSpec):
    r"""Uniform grid on \([a,b]\).

    Uses `jax.numpy.linspace(a, b, n, endpoint=...)`. Quadrature weights default to
    trapezoid weights when `endpoint=True` and uniform weights when the axis is treated
    as periodic (either `periodic=True` or `endpoint=False`).
    """

    endpoint: bool
    periodic: bool

    def __init__(self, n: int, *, endpoint: bool = True, periodic: bool = False):
        super().__init__(n)
        self.endpoint = bool(endpoint)
        self.periodic = bool(periodic)

    def materialize(self, a: Array, b: Array, /) -> AxisDiscretization:
        a_ = jnp.asarray(a, dtype=float).reshape(())
        b_ = jnp.asarray(b, dtype=float).reshape(())
        n = int(self.n)

        nodes = jnp.linspace(a_, b_, n, endpoint=bool(self.endpoint))

        if n == 1:
            w = jnp.asarray([b_ - a_], dtype=float)
        else:
            if self.periodic or not self.endpoint:
                dx = (b_ - a_) / float(n)
                w = jnp.full((n,), dx, dtype=float)
            else:
                dx = (b_ - a_) / float(n - 1)
                w = jnp.full((n,), dx, dtype=float)
                w = w.at[0].set(0.5 * dx)
                w = w.at[-1].set(0.5 * dx)

        return AxisDiscretization(
            nodes=nodes,
            quad_weights=w,
            basis="uniform",
            periodic=self.periodic or (not self.endpoint),
        )


class FourierAxisSpec(AbstractAxisSpec):
    r"""Uniform periodic grid for Fourier/FFT methods (endpoint excluded).

    Uses the nodes

    $$
    x_j = a + (b-a)\frac{j}{n},\quad j=0,\dots,n-1,
    $$

    with uniform weights \(w_j=(b-a)/n\). The resulting axis is marked `periodic=True`.
    """

    def materialize(self, a: Array, b: Array, /) -> AxisDiscretization:
        a_ = jnp.asarray(a, dtype=float).reshape(())
        b_ = jnp.asarray(b, dtype=float).reshape(())
        n = int(self.n)
        nodes = a_ + (b_ - a_) * (jnp.arange(n, dtype=float) / float(n))
        w = jnp.full((n,), (b_ - a_) / float(n), dtype=float)
        return AxisDiscretization(
            nodes=nodes,
            quad_weights=w,
            basis="fourier",
            periodic=True,
        )


class SineAxisSpec(AbstractAxisSpec):
    r"""Uniform interior grid (cell-centered) suitable for sine-like expansions.

    Uses the nodes

    $$
    x_j = a + (b-a)\frac{j+\tfrac12}{n},\quad j=0,\dots,n-1,
    $$

    with uniform weights \(w_j=(b-a)/n\). The resulting axis is non-periodic.
    """

    def materialize(self, a: Array, b: Array, /) -> AxisDiscretization:
        a_ = jnp.asarray(a, dtype=float).reshape(())
        b_ = jnp.asarray(b, dtype=float).reshape(())
        n = int(self.n)
        nodes = a_ + (b_ - a_) * ((jnp.arange(n, dtype=float) + 0.5) / float(n))
        w = jnp.full((n,), (b_ - a_) / float(n), dtype=float)
        return AxisDiscretization(
            nodes=nodes,
            quad_weights=w,
            basis="sine",
            periodic=False,
        )


class CosineAxisSpec(AbstractAxisSpec):
    r"""Uniform endpoint-including grid suitable for cosine-like expansions.

    Uses the nodes

    $$
    x_j = a + (b-a)\frac{j}{n-1},\quad j=0,\dots,n-1,
    $$

    with trapezoid weights \(w_0=w_{n-1}=\tfrac12\Delta x\), \(w_j=\Delta x\) otherwise.
    The resulting axis is non-periodic.
    """

    def materialize(self, a: Array, b: Array, /) -> AxisDiscretization:
        a_ = jnp.asarray(a, dtype=float).reshape(())
        b_ = jnp.asarray(b, dtype=float).reshape(())
        n = int(self.n)
        nodes = jnp.linspace(a_, b_, n, endpoint=True)

        if n == 1:
            w = jnp.asarray([b_ - a_], dtype=float)
        else:
            dx = (b_ - a_) / float(n - 1)
            w = jnp.full((n,), dx, dtype=float)
            w = w.at[0].set(0.5 * dx)
            w = w.at[-1].set(0.5 * dx)

        return AxisDiscretization(
            nodes=nodes,
            quad_weights=w,
            basis="cosine",
            periodic=False,
        )


class LegendreAxisSpec(AbstractAxisSpec):
    r"""Legendre Gauss/Radau/Lobatto nodes and weights (via orthax).

    orthax returns canonical nodes \(\xi_j\in[-1,1]\) and weights \(w_j\), which are
    mapped to \([a,b]\) via

    $$
    x_j=\tfrac{b-a}{2}\,\xi_j+\tfrac{a+b}{2},\qquad
    \tilde w_j=\tfrac{b-a}{2}\,w_j.
    $$
    """

    kind: Literal["gauss", "radau", "lobatto"]

    def __init__(self, n: int, *, kind: Literal["gauss", "radau", "lobatto"] = "gauss"):
        super().__init__(n)
        self.kind = kind

    def materialize(self, a: Array, b: Array, /) -> AxisDiscretization:
        a_ = jnp.asarray(a, dtype=float).reshape(())
        b_ = jnp.asarray(b, dtype=float).reshape(())
        n = int(self.n)

        rec = orthax.recurrence.Legendre(scale="standard")
        if self.kind == "gauss":
            x, w = orthax.orthgauss(n, rec)
        elif self.kind == "radau":
            x, w = orthax.orthgauss(n, rec, x0=-1.0)
        else:
            x, w = orthax.orthgauss(n, rec, x0=-1.0, x1=1.0)

        half = 0.5 * (b_ - a_)
        mid = 0.5 * (a_ + b_)
        nodes = half * x + mid
        weights = half * w
        return AxisDiscretization(
            nodes=nodes,
            quad_weights=weights,
            basis="legendre",
            periodic=False,
        )


def broadcasted_grid(coords: tuple[Array, ...], /) -> Array:
    """Broadcast 1D coordinate axes into a full Cartesian grid.

    If `coords=(x0, x1, ..., x{d-1})` with shapes `(n0,)`, `(n1,)`, ..., returns a
    grid array with shape `(n0, n1, ..., n{d-1}, d)`.
    """
    coords_ = tuple(jnp.asarray(c, dtype=float).reshape((-1,)) for c in coords)
    d = len(coords_)
    if d == 0:
        raise ValueError("coords must be non-empty.")

    reshaped = []
    for i, c in enumerate(coords_):
        shape = [1] * d
        shape[i] = int(c.shape[0])
        reshaped.append(jnp.reshape(c, tuple(shape)))

    if len(reshaped) == 1:
        return reshaped[0][..., None]
    reshaped = list(jnp.broadcast_arrays(*reshaped))
    return jnp.stack(reshaped, axis=-1)


def sdf_mask_from_adf(
    adf: Callable[[Array], Array],
    coords: tuple[Array, ...],
    /,
    *,
    inside_tol: float = 1e-6,
) -> Array:
    """Compute an interior mask on a coord-separable grid from a pointwise ADF."""
    grid = broadcasted_grid(coords)
    d = grid.shape[-1]
    pts = grid.reshape((-1, d))
    sdf = jax.vmap(adf)(pts)
    inside = jnp.asarray(sdf, dtype=float) < -float(inside_tol)
    return inside.reshape(grid.shape[:-1])


__all__ = [
    "AxisDiscretization",
    "AbstractAxisSpec",
    "GridSpec",
    "UniformAxisSpec",
    "FourierAxisSpec",
    "SineAxisSpec",
    "CosineAxisSpec",
    "LegendreAxisSpec",
    "broadcasted_grid",
    "sdf_mask_from_adf",
]
