# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Prepared field-view carriers with fixed cell ordering and coordinate scales."""

from __future__ import annotations

from collections.abc import Callable
from typing import overload

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from phydrax import discretization as D, domain


def field_view(
    reconstruction: D.PreparedFieldReconstruction,
    coefficients: ArrayLike,
) -> D.DiscreteFieldFunctionView:
    """Bind coefficients to the reconstruction's geometry under variable x."""
    support = domain.GeometryDomain(reconstruction.support_geometry, label="x")
    return D.DiscreteFieldFunctionView(
        reconstruction, coefficients, support, variable="x"
    )


def structured_fv(
    shape: tuple[int, ...] = (4, 2),
) -> D.FiniteVolumeDiscretization:
    """Prepare density/energy cell averages on the unit tensor box."""
    grid = D.TensorGridPlan(
        tuple(D.UniformCellAxisSpec(count) for count in shape),
        axis_names=tuple("xyz"[: len(shape)]),
    ).prepare(
        jnp.stack(
            (
                jnp.zeros(len(shape), dtype=jnp.float64),
                jnp.ones(len(shape), dtype=jnp.float64),
            )
        )
    )
    return D.FiniteVolumePlan(grid, component_names=("density", "energy")).prepare()


def triangulated_square(
    resolution: int = 4,
) -> D.UnstructuredFiniteVolumeDiscretization:
    """Prepare a row-major unit square with the lower-left/upper-right diagonal."""
    vertices = np.asarray(
        [
            (i / resolution, j / resolution)
            for j in range(resolution + 1)
            for i in range(resolution + 1)
        ],
        dtype=np.float64,
    )
    triangles: list[tuple[int, int, int]] = []
    for j in range(resolution):
        for i in range(resolution):
            corner = j * (resolution + 1) + i
            triangles.append((corner, corner + 1, corner + resolution + 2))
            triangles.append((corner, corner + resolution + 2, corner + resolution + 1))
    return D.UnstructuredFiniteVolumePlan(
        vertices, triangles=np.asarray(triangles, dtype=np.int32)
    ).prepare()


def cell_averages(
    discretization: D.UnstructuredFiniteVolumeDiscretization,
    function: Callable[[Array], Array],
) -> Array:
    """Integrate the fixture's scalar field on each retained cell rule."""
    values = function(discretization.cell_quadrature_points)
    averages = (
        jnp.sum(discretization.cell_quadrature_weights * values, axis=1)
        / discretization.cell_volumes
    )
    return averages[:, None]


@overload
def quadratic(points: Array) -> Array: ...


@overload
def quadratic(points: np.ndarray) -> np.ndarray: ...


def quadratic(points: Array | np.ndarray) -> Array | np.ndarray:
    """Evaluate the independent polynomial used for value/derivative comparisons."""
    x, y = points[..., 0], points[..., 1]
    return 1.0 + 2.0 * x - y + 0.5 * x * y + y**2


def finite_difference(
    shape: tuple[int, int] = (9, 7),
    periodic: tuple[bool, bool] = (False, False),
) -> D.PreparedFiniteDifferenceDiscretization:
    """Prepare a first-x-derivative grid on [0,1] × [0,2]."""
    grid = D.TensorGridPlan(
        tuple(
            D.UniformAxisSpec(count, periodic=flag, endpoint=not flag)
            for count, flag in zip(shape, periodic, strict=True)
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 2.0)), dtype=jnp.float64))
    request = D.DerivativeRequest(
        "dx",
        grid,
        "x",
        derivative_order=1,
        accuracy_order=2,
        boundary="periodic" if periodic[0] else "one_sided",
    )
    return D.FiniteDifferencePlan(grid, (request,), field_name="u").prepare()


def nodal(
    discretization: D.PreparedFiniteDifferenceDiscretization,
    function: Callable[[Array, Array], Array],
) -> Array:
    """Sample a two-coordinate field on the prepared primary layout."""
    x, y = discretization.grid.primary_entity_layout.coordinates_by_axis
    return function(x[:, None], y[None, :])


def point_cloud(points: np.ndarray) -> D.PreparedPointCloudDiscretization:
    """Prepare equal point weights with quadratic local stencil reproduction."""
    weights = np.full(points.shape[0], 1.0 / points.shape[0], dtype=np.float64)
    return D.PointCloudPlan(
        points, weights, stencil=D.LocalStencilPolicy(polynomial_degree=2)
    ).prepare()
