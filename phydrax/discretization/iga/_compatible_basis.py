#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Prepared polynomial differential bases and tensor functional routes."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import prod
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._interpolation._bspline import bspline_jet_stencil
from ..._interpolation._bspline_grid import BSplineGrid
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...exterior._algebra import pullback
from ...exterior._basis import exterior_indices
from ...exterior._form_type import FormTwist, FormType
from ...linalg import (
    DenseLinearOperator,
    DenseLU,
    FailurePolicy,
    LinearSolvePolicy,
    LinearSystem,
    prepare,
    PreparedLinearSolve,
    RHSLayout,
    solve,
)


def basis(
    grid: BSplineGrid,
    points: Array,
    /,
    *,
    differential: bool = False,
    periodic: bool = False,
) -> Array:
    stencil = bspline_jet_stencil(
        grid.knots, points, degree=grid.degree, maximum_order=0, bounds="clip"
    )
    values = jnp.zeros((points.size, grid.coefficient_count), dtype=jnp.float64)
    values = values.at[jnp.arange(points.size)[:, None], stencil.indices].add(
        stencil.jets[:, 0, :]
    )
    if differential:
        p = grid.degree + 1
        widths = grid.knots[p:] - grid.knots[:-p]
        values = values * (p / widths[: grid.coefficient_count])[None, :]
    if periodic:
        values = values[:, :-1].at[:, 0].add(values[:, -1])
    return values


def tensor_basis(
    grids: Sequence[BSplineGrid],
    points: Array,
    axes: tuple[int, ...],
    periodic: tuple[bool, ...],
    /,
) -> Array:
    result = jnp.ones((points.shape[0], 1), dtype=jnp.float64)
    for axis, grid in enumerate(grids):
        factor = basis(
            grid,
            points[:, axis],
            differential=axis in axes,
            periodic=periodic[axis] and axis not in axes,
        )
        result = (result[:, :, None] * factor[:, None, :]).reshape((points.shape[0], -1))
    return result


def tensor_quadrature(grids: Sequence[BSplineGrid], order: int, /) -> tuple[Array, Array]:
    rules = tuple(grid.quadrature(order) for grid in grids)
    points = jnp.stack(
        jnp.meshgrid(*(rule[0] for rule in rules), indexing="ij"), axis=-1
    ).reshape((-1, len(grids)))
    weights = jnp.ones(tuple(rule[0].size for rule in rules), dtype=jnp.float64)
    for axis, (_, factor) in enumerate(rules):
        shape = tuple(factor.size if i == axis else 1 for i in range(len(grids)))
        weights = weights * factor.reshape(shape)
    return points, weights.reshape((-1,))


def component_functionals(
    base: Sequence[BSplineGrid],
    axes: tuple[int, ...],
    periodic: tuple[bool, ...],
    order: int,
    /,
    *,
    partitions: Sequence[Array] | None = None,
) -> tuple[Array, Array, tuple[int, ...]]:
    nodes: list[Array] = []
    weights: list[Array] = []
    if partitions is not None and len(partitions) != len(base):
        raise ValueError("Spline functional partitions must match the chart dimension.")
    axis_routes: list[np.ndarray] = []
    shapes: list[int] = []
    roots, root_weights = np.polynomial.legendre.leggauss(order)
    for axis, grid in enumerate(base):
        greville = np.asarray(grid.greville_abscissae)
        if axis in axes:
            points, factors, routes = [], [], []
            breaks = np.asarray(grid.breakpoints)
            if partitions is not None:
                breaks = np.unique(np.concatenate((breaks, np.asarray(partitions[axis]))))
            for index, (lower, upper) in enumerate(
                zip(greville[:-1], greville[1:], strict=True)
            ):
                partition = np.concatenate(
                    (
                        np.asarray([lower]),
                        breaks[(breaks > lower) & (breaks < upper)],
                        np.asarray([upper]),
                    )
                )
                for left, right in zip(partition[:-1], partition[1:], strict=True):
                    points.extend((left + right) / 2 + (right - left) * roots / 2)
                    factors.extend((right - left) * root_weights / 2)
                    routes.extend([index] * order)
            nodes.append(jnp.asarray(points, dtype=jnp.float64))
            weights.append(jnp.asarray(factors, dtype=jnp.float64))
            axis_routes.append(np.asarray(routes, dtype=np.int32))
            shapes.append(greville.size - 1)
        else:
            selected = greville[:-1] if periodic[axis] else greville
            nodes.append(jnp.asarray(selected))
            weights.append(jnp.ones((selected.size,), dtype=jnp.float64))
            axis_routes.append(np.arange(selected.size, dtype=np.int32))
            shapes.append(selected.size)
    points = jnp.stack(jnp.meshgrid(*nodes, indexing="ij"), axis=-1).reshape(
        (-1, len(base))
    )
    qshape = tuple(node.size for node in nodes)
    routes = np.zeros(qshape, dtype=np.int32)
    product_weights = jnp.ones(qshape, dtype=jnp.float64)
    for axis in range(len(base)):
        stride = prod(shapes[axis + 1 :])
        shape = tuple(qshape[axis] if i == axis else 1 for i in range(len(base)))
        routes += axis_routes[axis].reshape(shape) * stride
        product_weights = product_weights * weights[axis].reshape(shape)
    return (
        points,
        jnp.stack(
            (jnp.asarray(routes.reshape((-1,))), product_weights.reshape((-1,))), axis=0
        ),
        tuple(shapes),
    )


@final
class _PreparedSplineFunctional(StrictModule, NonTrainableState):
    """Tensor functional inversion through reusable native one-axis factors."""

    factors: tuple[PreparedLinearSolve, ...]
    coefficient_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        base: Sequence[BSplineGrid],
        grids: Sequence[BSplineGrid],
        axes: tuple[int, ...],
        periodic: tuple[bool, ...],
        identifier: str,
        order: int,
        /,
    ) -> None:
        factors, shape = [], []
        for axis, (original, grid) in enumerate(zip(base, grids, strict=True)):
            active_axes = (0,) if axis in axes else ()
            points, route, extent = component_functionals(
                (original,), active_axes, (periodic[axis],), order
            )
            values = basis(
                grid,
                points[:, 0],
                differential=axis in axes,
                periodic=periodic[axis] and axis not in axes,
            )

            def functional(column: Array) -> Array:
                return apply_functionals(column, route, extent[0])

            matrix = jax.vmap(functional, in_axes=1, out_axes=1)(values)
            operator = DenseLinearOperator(
                matrix, operator_id=f"{identifier}:axis:{axis}"
            )
            factors.append(
                prepare(
                    LinearSystem(operator, problem_id=f"{identifier}:axis:{axis}:system"),
                    LinearSolvePolicy(DenseLU(), failure=FailurePolicy("error")),
                )
            )
            shape.append(extent[0])
        self.factors, self.coefficient_shape = tuple(factors), tuple(shape)

    def solve(self, rhs: Array, /) -> Array:
        if rhs.shape != (prod(self.coefficient_shape),):
            raise ValueError(
                "Spline functional values must match the component tensor extent."
            )
        value = rhs.reshape(self.coefficient_shape)
        for axis, factor in enumerate(self.factors):
            moved = jnp.moveaxis(value, axis, 0)
            columns = moved.reshape((self.coefficient_shape[axis], -1))
            result = solve(
                factor, columns, rhs_layout=RHSLayout((columns.shape[1],))
            ).value
            value = jnp.moveaxis(result.reshape(moved.shape), 0, axis)
        return value.reshape((-1,))


def apply_functionals(values: Array, routes: Array, size: int, /) -> Array:
    return (
        jnp.zeros((size,), dtype=values.dtype)
        .at[routes[0].astype(jnp.int32)]
        .add(routes[1] * values)
    )


def pullback_values(
    form: Callable[[Array], Array],
    points: Array,
    axes: tuple[int, ...],
    geometry: Callable[[Array], Array] | None,
    twist: FormTwist,
    /,
) -> Array:
    physical_points = points if geometry is None else jax.vmap(geometry)(points)
    values = jax.vmap(form)(physical_points)
    if values.ndim != 2 or values.shape[1] != len(
        exterior_indices(physical_points.shape[1], len(axes))
    ):
        raise ValueError(
            "Spline interpolants require canonical form components with an explicit component axis."
        )
    if geometry is not None:
        jacobians = jax.vmap(jax.jacfwd(geometry))(points)
        values = pullback(
            values, FormType(physical_points.shape[1], len(axes), twist=twist), jacobians
        )
    component = exterior_indices(points.shape[1], len(axes)).index(axes)
    return values[:, component]
