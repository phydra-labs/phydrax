#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
import math
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
    basis: Literal["uniform", "fourier", "sine", "cosine", "legendre", "nested"]
    periodic: bool
    active: Array | None
    level: Array | None
    parent_interval: Array | None

    def __init__(
        self,
        *,
        nodes: Array,
        quad_weights: Array | None,
        basis: Literal["uniform", "fourier", "sine", "cosine", "legendre", "nested"],
        periodic: bool,
        active: Array | None = None,
        level: Array | None = None,
        parent_interval: Array | None = None,
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

        def normalize_metadata(name, value, dtype):
            if value is None:
                return None
            normalized = jnp.asarray(value, dtype=dtype).reshape((-1,))
            if normalized.shape != nodes_.shape:
                raise ValueError(
                    f"AxisDiscretization.{name} must have the same shape as nodes."
                )
            return normalized

        self.active = normalize_metadata("active", active, bool)
        self.level = normalize_metadata("level", level, jnp.int32)
        self.parent_interval = normalize_metadata(
            "parent_interval",
            parent_interval,
            jnp.int32,
        )

    def with_active(self, active: Array, /) -> "AxisDiscretization":
        """Return a nested discretization with updated active nodes and weights."""
        if self.basis != "nested" or self.level is None:
            raise ValueError("with_active requires a nested axis discretization.")
        active_ = jnp.asarray(active, dtype=bool).reshape(self.nodes.shape)
        if not bool(active_[0]) or not bool(active_[-1]):
            raise ValueError("Nested axis endpoints must remain active.")
        weights = _trapezoid_weights_from_active(self.nodes, active_)
        return AxisDiscretization(
            nodes=self.nodes,
            quad_weights=weights,
            basis=self.basis,
            periodic=self.periodic,
            active=active_,
            level=self.level,
            parent_interval=self.parent_interval,
        )


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

    For a geometry variable with `spatial_dim=d`, use `GridSpec(axes=(spec0, ..., spec{d-1}))`
    to specify a different `AxisSpec` per coordinate axis.
    """

    axes: tuple[AbstractAxisSpec, ...]
    cut_cell_order: int

    def __init__(
        self,
        axes: Sequence[AbstractAxisSpec],
        *,
        cut_cell_order: int = 0,
    ):
        axes_ = tuple(axes)
        if not axes_:
            raise ValueError("GridSpec.axes must be non-empty.")
        if int(cut_cell_order) < 0:
            raise ValueError("GridSpec.cut_cell_order must be non-negative.")
        self.axes = axes_
        self.cut_cell_order = int(cut_cell_order)


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


class NestedDyadicAxisSpec(AbstractAxisSpec):
    """Fixed-capacity nested dyadic nodes with an initially active coarse level."""

    initial_level: int

    def __init__(self, n: int, *, initial_level: int = 1):
        super().__init__(n)
        intervals = int(n) - 1
        if intervals <= 0 or intervals & (intervals - 1):
            raise ValueError("NestedDyadicAxisSpec requires n = 2**level + 1.")
        max_level = int(math.log2(intervals))
        if not 0 <= int(initial_level) <= max_level:
            raise ValueError(
                f"initial_level must lie in [0, {max_level}], got {initial_level}."
            )
        self.initial_level = int(initial_level)

    def materialize(self, a: Array, b: Array, /) -> AxisDiscretization:
        a_ = jnp.asarray(a, dtype=float).reshape(())
        b_ = jnp.asarray(b, dtype=float).reshape(())
        n = int(self.n)
        max_level = int(math.log2(n - 1))
        nodes = jnp.linspace(a_, b_, n, endpoint=True)
        levels: list[int] = []
        parents: list[int] = []
        for index in range(n):
            if index in (0, n - 1):
                levels.append(0)
                parents.append(-1)
                continue
            divisible = 0
            value = index
            while value % 2 == 0:
                divisible += 1
                value //= 2
            level = max_level - divisible
            levels.append(level)
            stride = 2 ** (max_level - level)
            parents.append(index - stride)
        level_arr = jnp.asarray(levels, dtype=jnp.int32)
        active = level_arr <= self.initial_level
        weights = _trapezoid_weights_from_active(nodes, active)
        return AxisDiscretization(
            nodes=nodes,
            quad_weights=weights,
            basis="nested",
            periodic=False,
            active=active,
            level=level_arr,
            parent_interval=jnp.asarray(parents, dtype=jnp.int32),
        )


def _trapezoid_weights_from_active(nodes: Array, active: Array) -> Array:
    nodes_ = jnp.asarray(nodes, dtype=float).reshape((-1,))
    active_ = jnp.asarray(active, dtype=bool).reshape((-1,))
    active_nodes = nodes_[active_]
    if active_nodes.size == 1:
        active_weights = jnp.ones((1,), dtype=nodes_.dtype)
    else:
        active_weights = jnp.empty_like(active_nodes)
        active_weights = active_weights.at[0].set(
            0.5 * (active_nodes[1] - active_nodes[0])
        )
        active_weights = active_weights.at[-1].set(
            0.5 * (active_nodes[-1] - active_nodes[-2])
        )
        if active_nodes.size > 2:
            active_weights = active_weights.at[1:-1].set(
                0.5 * (active_nodes[2:] - active_nodes[:-2])
            )
    return (
        jnp.zeros_like(nodes_)
        .at[jnp.where(active_, size=active_nodes.size)[0]]
        .set(active_weights)
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


def cut_cell_geometry_weight_from_adf(
    adf: Callable[[Array], Array],
    coords: tuple[Array, ...],
    bounds: Array,
    base_weights: tuple[Array, ...],
    center_mask: Array,
    target_measure: Array,
    /,
    *,
    order: int,
    inside_tol: float = 1e-6,
) -> Array:
    """Estimate geometry-aware tensor weights from deterministic subcell probes."""
    probe_order = int(order)
    if probe_order <= 0:
        raise ValueError("cut-cell probe order must be positive.")
    if len(coords) != len(base_weights):
        raise ValueError("coords and base_weights must have the same length.")
    bounds_ = jnp.asarray(bounds, dtype=float)
    if bounds_.shape != (2, len(coords)):
        raise ValueError("bounds must have shape (2, num_axes).")

    fractions = (jnp.arange(probe_order, dtype=float) + 0.5) / probe_order
    probe_axes: list[Array] = []
    cell_widths: list[Array] = []
    for axis_index, coordinate in enumerate(coords):
        values = jnp.asarray(coordinate, dtype=float).reshape((-1,))
        permutation = jnp.argsort(values)
        sorted_values = values[permutation]
        midpoints = 0.5 * (sorted_values[:-1] + sorted_values[1:])
        left_sorted = jnp.concatenate(
            (bounds_[0, axis_index : axis_index + 1], midpoints)
        )
        right_sorted = jnp.concatenate(
            (midpoints, bounds_[1, axis_index : axis_index + 1])
        )
        left = jnp.zeros_like(values).at[permutation].set(left_sorted)
        right = jnp.zeros_like(values).at[permutation].set(right_sorted)
        widths = jnp.maximum(right - left, 0.0)
        probes = left[:, None] + widths[:, None] * fractions[None, :]
        probe_axes.append(probes.reshape((-1,)))
        cell_widths.append(widths)

    probe_grid = broadcasted_grid(tuple(probe_axes))
    probe_points = probe_grid.reshape((-1, len(coords)))
    probe_inside = jax.vmap(adf)(probe_points) < -float(inside_tol)
    interleaved_shape = tuple(
        size
        for coordinate in coords
        for size in (int(jnp.asarray(coordinate).size), probe_order)
    )
    probe_inside = probe_inside.reshape(interleaved_shape)
    probe_axes_to_reduce = tuple(range(1, 2 * len(coords), 2))
    occupancy = jnp.mean(probe_inside.astype(float), axis=probe_axes_to_reduce)

    cell_measure = jnp.asarray(1.0, dtype=float)
    quadrature_measure = jnp.asarray(1.0, dtype=float)
    for axis_index, (width, base_weight) in enumerate(
        zip(cell_widths, base_weights, strict=True)
    ):
        shape = [1] * len(coords)
        shape[axis_index] = int(width.shape[0])
        cell_measure = cell_measure * width.reshape(tuple(shape))
        quadrature_measure = quadrature_measure * jnp.asarray(
            base_weight, dtype=float
        ).reshape(tuple(shape))

    mask = jnp.asarray(center_mask, dtype=bool)
    represented_measure = jnp.where(mask, occupancy * cell_measure, 0.0)
    estimate = jnp.sum(represented_measure)
    normalized_measure = represented_measure * (
        jnp.asarray(target_measure, dtype=float)
        / jnp.maximum(estimate, jnp.finfo(represented_measure.dtype).tiny)
    )
    return normalized_measure / jnp.maximum(
        quadrature_measure,
        jnp.finfo(normalized_measure.dtype).tiny,
    )


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
    "cut_cell_geometry_weight_from_adf",
]
