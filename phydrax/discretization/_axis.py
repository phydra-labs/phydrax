#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
import math
from collections.abc import Callable, Sequence
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._polynomial._orthogonal import legendre_rule_data
from .._strict import StrictModule
from ._axis_domain import AxisDomain
from ._core import (
    DiscretizationCapability,
    DiscretizationKey,
    DiscretizationRole,
)
from ._lifecycle import AbstractDiscretizationPlan


AxisPrimaryEntity = Literal["point", "interval"]
AxisBasis = Literal[
    "uniform",
    "fourier",
    "sine",
    "cosine",
    "chebyshev",
    "legendre",
    "nested",
    "rational_chebyshev_line",
    "rational_chebyshev_half_line",
]


class AxisDiscretization(StrictModule):
    """Materialized one-dimensional nodes, measure, support, and topology."""

    nodes: Array
    quad_weights: Array | None
    domain: AxisDomain
    active: Array | None
    level: Array | None
    parent_interval: Array | None
    basis: AxisBasis = eqx.field(static=True)
    primary_entity: AxisPrimaryEntity = eqx.field(static=True)
    lower_endpoint_included: bool = eqx.field(static=True)
    upper_endpoint_included: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        nodes: ArrayLike,
        quad_weights: ArrayLike | None,
        basis: AxisBasis,
        domain: AxisDomain,
        primary_entity: AxisPrimaryEntity = "point",
        lower_endpoint_included: bool,
        upper_endpoint_included: bool,
        active: ArrayLike | None = None,
        level: ArrayLike | None = None,
        parent_interval: ArrayLike | None = None,
    ):
        if not isinstance(domain, AxisDomain):
            raise TypeError("domain must be an AxisDomain.")
        if basis not in (
            "uniform",
            "fourier",
            "sine",
            "cosine",
            "chebyshev",
            "legendre",
            "nested",
            "rational_chebyshev_line",
            "rational_chebyshev_half_line",
        ):
            raise ValueError("Unknown axis basis.")
        nodes_ = jnp.asarray(nodes, dtype=float).reshape((-1,))
        if nodes_.size == 0:
            raise ValueError("AxisDiscretization.nodes must be non-empty.")
        nodes_ = eqx.error_if(
            nodes_,
            jnp.any(~jnp.isfinite(nodes_)),
            "AxisDiscretization nodes must be finite.",
        )
        weights = (
            None
            if quad_weights is None
            else jnp.asarray(quad_weights, dtype=float).reshape((-1,))
        )
        if weights is not None:
            if weights.shape != nodes_.shape:
                raise ValueError(
                    "AxisDiscretization.quad_weights must have the same shape as nodes."
                )
            weights = eqx.error_if(
                weights,
                jnp.any(~jnp.isfinite(weights)) | jnp.any(weights < 0.0),
                "AxisDiscretization quadrature weights must be finite and non-negative.",
            )
        if primary_entity not in ("point", "interval"):
            raise ValueError("primary_entity must be 'point' or 'interval'.")
        if primary_entity == "interval" and domain.finite_bounds is None:
            raise ValueError("Interval-primary axes require a finite domain.")
        lower = bool(lower_endpoint_included)
        upper = bool(upper_endpoint_included)
        if domain.periodic_axis and (lower or upper):
            raise ValueError("Periodic axes cannot include physical boundary endpoints.")
        if primary_entity == "interval" and (lower or upper):
            raise ValueError(
                "Interval-primary nodes are not physical boundary endpoints."
            )

        def normalize_metadata(name, value, dtype):
            if value is None:
                return None
            normalized = jnp.asarray(value, dtype=dtype).reshape((-1,))
            if normalized.shape != nodes_.shape:
                raise ValueError(
                    f"AxisDiscretization.{name} must have the same shape as nodes."
                )
            return normalized

        self.nodes = nodes_
        self.quad_weights = weights
        self.domain = domain
        self.active = normalize_metadata("active", active, bool)
        self.level = normalize_metadata("level", level, jnp.int32)
        self.parent_interval = normalize_metadata(
            "parent_interval",
            parent_interval,
            jnp.int32,
        )
        self.basis = basis
        self.primary_entity = primary_entity
        self.lower_endpoint_included = lower
        self.upper_endpoint_included = upper

    @property
    def periodic(self) -> bool:
        return self.domain.periodic_axis

    @property
    def bounds(self) -> Array | None:
        return self.domain.finite_bounds

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
            domain=self.domain,
            active=active_,
            primary_entity=self.primary_entity,
            lower_endpoint_included=self.lower_endpoint_included,
            upper_endpoint_included=self.upper_endpoint_included,
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


class TensorGridPlan(AbstractDiscretizationPlan):
    """Per-label tensor-grid construction plan."""

    axes: tuple[AbstractAxisSpec, ...]
    axis_names: tuple[str, ...] = eqx.field(static=True)
    cut_cell_order: int = eqx.field(static=True)
    key: DiscretizationKey
    capabilities: tuple[DiscretizationCapability, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        axes: Sequence[AbstractAxisSpec],
        *,
        axis_names: Sequence[str] | None = None,
        cut_cell_order: int = 0,
        key: DiscretizationKey | None = None,
        plan_id: str | None = None,
    ):
        axes_ = tuple(axes)
        if not axes_ or not all(isinstance(axis, AbstractAxisSpec) for axis in axes_):
            raise TypeError(
                "TensorGridPlan.axes must contain one or more AbstractAxisSpec values."
            )
        names = (
            tuple(f"axis{index}" for index in range(len(axes_)))
            if axis_names is None
            else tuple(str(name) for name in axis_names)
        )
        if (
            len(names) != len(axes_)
            or any(not name for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError(
                "axis_names must contain one unique non-empty name per axis."
            )
        order = int(cut_cell_order)
        if order < 0:
            raise ValueError("TensorGridPlan.cut_cell_order must be non-negative.")
        key_ = (
            DiscretizationKey(
                "tensor_grid",
                DiscretizationRole.PHYSICAL,
                domain_labels=names,
            )
            if key is None
            else key
        )
        if not isinstance(key_, DiscretizationKey):
            raise TypeError("key must be a DiscretizationKey.")
        capabilities = (DiscretizationCapability.RECONSTRUCTION,)
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "tensor-grid-plan",
                    "axes": [
                        {"type": type(axis).__name__, "specification": repr(axis)}
                        for axis in axes_
                    ],
                    "axis_names": list(names),
                    "cut_cell_order": order,
                    "key": key_.key_id,
                }
            )
            if plan_id is None
            else str(plan_id)
        )
        if not identifier:
            raise ValueError("plan_id must be non-empty.")
        self.axes = axes_
        self.axis_names = names
        self.cut_cell_order = order
        self.key = key_
        self.capabilities = capabilities
        self.plan_id = identifier

    def prepare(self, bounds: ArrayLike, /):
        """Materialize numerical support without selecting a calculus."""
        from ._tensor_support import PreparedTensorGrid

        return PreparedTensorGrid.from_plan(self, bounds)


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

        periodic = self.periodic or (not self.endpoint)
        return AxisDiscretization(
            nodes=nodes,
            quad_weights=w,
            basis="uniform",
            domain=(
                AxisDomain.periodic(a_, b_) if periodic else AxisDomain.interval(a_, b_)
            ),
            primary_entity="point",
            lower_endpoint_included=bool(self.endpoint) and not periodic,
            upper_endpoint_included=bool(self.endpoint) and n > 1 and not periodic,
        )


class UniformCellAxisSpec(AbstractAxisSpec):
    """Uniform cell-centered axis with explicit physical boundary points."""

    periodic: bool

    def __init__(self, n: int, *, periodic: bool = False):
        super().__init__(n)
        self.periodic = bool(periodic)

    def materialize(self, a: Array, b: Array, /) -> AxisDiscretization:
        a_ = jnp.asarray(a, dtype=float).reshape(())
        b_ = jnp.asarray(b, dtype=float).reshape(())
        count = int(self.n)
        width = (b_ - a_) / float(count)
        centers = a_ + (jnp.arange(count, dtype=float) + 0.5) * width
        return AxisDiscretization(
            nodes=centers,
            quad_weights=jnp.full((count,), width),
            basis="uniform",
            domain=(
                AxisDomain.periodic(a_, b_)
                if self.periodic
                else AxisDomain.interval(a_, b_)
            ),
            primary_entity="interval",
            lower_endpoint_included=False,
            upper_endpoint_included=False,
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
            domain=AxisDomain.interval(a_, b_),
            primary_entity="point",
            lower_endpoint_included=True,
            upper_endpoint_included=True,
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
            domain=AxisDomain.periodic(a_, b_),
            primary_entity="point",
            lower_endpoint_included=False,
            upper_endpoint_included=False,
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
            domain=AxisDomain.interval(a_, b_),
            primary_entity="interval",
            lower_endpoint_included=False,
            upper_endpoint_included=False,
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
            domain=AxisDomain.interval(a_, b_),
            primary_entity="point",
            lower_endpoint_included=True,
            upper_endpoint_included=n > 1,
        )


class LegendreAxisSpec(AbstractAxisSpec):
    r"""Legendre Gauss/Radau/Lobatto nodes and weights.

    Phydrax constructs canonical nodes \(\xi_j\in[-1,1]\) and weights \(w_j\), which
    are mapped to \([a,b]\) via

    $$
    x_j=\tfrac{b-a}{2}\,\xi_j+\tfrac{a+b}{2},\qquad
    \tilde w_j=\tfrac{b-a}{2}\,w_j.
    $$
    """

    kind: Literal["gauss", "radau", "lobatto"]

    def __init__(self, n: int, *, kind: Literal["gauss", "radau", "lobatto"] = "gauss"):
        super().__init__(n)
        if kind not in ("gauss", "radau", "lobatto"):
            raise ValueError("kind must be 'gauss', 'radau', or 'lobatto'.")
        if kind == "lobatto" and self.n < 2:
            raise ValueError("Legendre Lobatto axes require at least two nodes.")
        self.kind = kind

    def materialize(self, a: Array, b: Array, /) -> AxisDiscretization:
        a_ = jnp.asarray(a, dtype=float).reshape(())
        b_ = jnp.asarray(b, dtype=float).reshape(())
        n = int(self.n)

        rule = legendre_rule_data(n, self.kind, dtype=a_.dtype)
        x, w = rule.nodes, rule.weights

        half = 0.5 * (b_ - a_)
        mid = 0.5 * (a_ + b_)
        nodes = half * x + mid
        weights = half * w
        return AxisDiscretization(
            nodes=nodes,
            quad_weights=weights,
            basis="legendre",
            domain=AxisDomain.interval(a_, b_),
            primary_entity="point",
            lower_endpoint_included=self.kind in ("radau", "lobatto"),
            upper_endpoint_included=self.kind == "lobatto",
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
    "AbstractAxisSpec",
    "AxisBasis",
    "AxisDiscretization",
    "AxisPrimaryEntity",
    "CosineAxisSpec",
    "FourierAxisSpec",
    "LegendreAxisSpec",
    "NestedDyadicAxisSpec",
    "SineAxisSpec",
    "TensorGridPlan",
    "UniformAxisSpec",
    "UniformCellAxisSpec",
    "broadcasted_grid",
    "cut_cell_geometry_weight_from_adf",
    "sdf_mask_from_adf",
]
