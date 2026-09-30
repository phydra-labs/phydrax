#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from scipy.interpolate import BSpline

from phydrax._strict import StrictModule
from phydrax.discretization.iga import (
    AssembledSplineDeRhamComplex,
    BSplineGrid,
    SplineDeRhamComplex,
)
from phydrax.linalg import (
    ArraySpace,
    DenseLinearOperator,
    MaterializationPolicy,
    materialize,
    OperatorProperties,
)


def _annulus(point: Array) -> Array:
    radius = 1.0 + point[0]
    angle = 2.0 * jnp.pi * point[1]
    return radius * jnp.stack((jnp.cos(angle), jnp.sin(angle)))


def _harmonic(point: Array) -> Array:
    return jnp.stack((-point[1], point[0])) / jnp.sum(point * point)


def _independent_gram(complex_: SplineDeRhamComplex, degree: int) -> np.ndarray:
    roots, gauss_weights = np.polynomial.legendre.leggauss(8)
    rules = []
    for grid in complex_.base_grids:
        breaks = np.asarray(grid.breakpoints)
        x = np.concatenate(
            tuple(
                (left + right) / 2 + (right - left) * roots / 2
                for left, right in zip(breaks[:-1], breaks[1:], strict=True)
            )
        )
        w = np.concatenate(
            tuple(
                (right - left) * gauss_weights / 2
                for left, right in zip(breaks[:-1], breaks[1:], strict=True)
            )
        )
        rules.append((x, w))
    points = np.stack(
        np.meshgrid(*(rule[0] for rule in rules), indexing="ij"), axis=-1
    ).reshape((-1, complex_.dimension))
    weight_mesh = np.meshgrid(*(rule[1] for rule in rules), indexing="ij")
    weights = np.prod(np.stack(weight_mesh), axis=0).reshape((-1,))
    space = complex_.spaces[degree]
    gram = np.zeros((space.dof_count, space.dof_count), dtype=np.float64)
    for offset, component in zip(space.component_offsets, space.components, strict=True):
        values = np.ones((points.shape[0], 1), dtype=np.float64)
        for axis, grid in enumerate(component.grids):
            factors = BSpline.design_matrix(
                points[:, axis], np.asarray(grid.knots), grid.degree
            ).toarray()
            if axis in component.component_axes:
                p = grid.degree + 1
                knots = np.asarray(grid.knots)
                factors *= (
                    p
                    / (
                        knots[p : p + grid.coefficient_count]
                        - knots[: grid.coefficient_count]
                    )
                )[None, :]
            values = (values[:, :, None] * factors[:, None, :]).reshape(
                (points.shape[0], -1)
            )
        block = values.T @ (weights[:, None] * values)
        gram[
            offset : offset + component.coefficient_count,
            offset : offset + component.coefficient_count,
        ] = block
    return gram


@pytest.mark.parametrize("degree", [0, 1, 2])
def test_kronecker_hodge_matches_independent_quadrature(degree: int) -> None:
    xgrid = BSplineGrid.open_uniform(2, 2)
    ygrid = BSplineGrid.open_uniform(3, 2)
    complex_ = SplineDeRhamComplex((xgrid, ygrid))
    matrix = _independent_gram(complex_, degree)
    values = jnp.linspace(-0.7, 1.2, complex_.dof_count(degree), dtype=jnp.float64)
    np.testing.assert_allclose(
        complex_.hodge_star(degree, values),
        matrix @ np.asarray(values),
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        complex_.inverse_hodge_star(degree, complex_.hodge_star(degree, values)),
        values,
        rtol=2e-9,
        atol=2e-9,
    )


@pytest.mark.parametrize("dimension", [2, 3])
def test_sparse_differential_matches_analytic_derivative(dimension: int) -> None:
    grid = BSplineGrid.open_uniform(2, 1)
    complex_ = SplineDeRhamComplex((grid,) * dimension)

    def scalar(point: Array) -> Array:
        return jnp.asarray([jnp.sum(point * point)], dtype=jnp.float64)

    def gradient(point: Array) -> Array:
        return 2.0 * point

    coefficients = complex_.interpolant(0, scalar)
    derivative = complex_.exterior_derivative(0, coefficients)
    points = jnp.full((2, dimension), 0.3, dtype=jnp.float64).at[1].set(0.7)
    np.testing.assert_allclose(
        complex_.reconstruction(1, derivative, points),
        jax.vmap(gradient)(points),
        rtol=2e-11,
        atol=2e-11,
    )
    np.testing.assert_allclose(
        complex_.exterior_derivative(1, derivative), 0.0, atol=2e-12
    )


def test_tensor_functional_interpolant_commutes_for_nonpolynomial_form() -> None:
    grid = BSplineGrid.open_uniform(3, 3)
    complex_ = SplineDeRhamComplex((grid, grid))

    def scalar(point: Array) -> Array:
        return jnp.asarray([jnp.exp(point[0]) * jnp.sin(point[1])], dtype=jnp.float64)

    def derivative(point: Array) -> Array:
        return jnp.exp(point[0]) * jnp.stack((jnp.sin(point[1]), jnp.cos(point[1])))

    np.testing.assert_allclose(
        complex_.exterior_derivative(0, complex_.interpolant(0, scalar)),
        complex_.interpolant(1, derivative),
        rtol=2e-10,
        atol=2e-10,
    )


@pytest.mark.parametrize(
    "axis,side", [(0, "lower"), (0, "upper"), (1, "lower"), (1, "upper")]
)
def test_oriented_trace_commutes_with_d(axis: int, side: str) -> None:
    from phydrax.discretization.iga import BoundarySide
    from phydrax.typing import parse

    grid = BSplineGrid.open_uniform(2, 2)
    complex_ = SplineDeRhamComplex((grid, grid))
    trace = complex_.trace(axis, parse(side, BoundarySide, "side"))
    values = jnp.linspace(0.1, 1.3, complex_.dof_count(0), dtype=jnp.float64)
    np.testing.assert_allclose(
        trace.target.differential(0).mv(trace.maps[0].mv(values)),
        trace.maps[1].mv(complex_.exterior_derivative(0, values)),
        atol=2e-12,
    )
    constant = trace.maps[0].mv(jnp.ones((complex_.dof_count(0),), dtype=jnp.float64))
    np.testing.assert_allclose(constant, 1.0, atol=2e-12)


def test_refinement_complex_map_commutes() -> None:
    coarse = BSplineGrid.open_uniform(2, 1)
    fine = BSplineGrid.open_uniform(2, 2)
    source = SplineDeRhamComplex((coarse, coarse))
    target = SplineDeRhamComplex((fine, fine))
    transfer = source.transfer(target)
    values = jnp.linspace(-0.3, 1.0, source.dof_count(0), dtype=jnp.float64)
    np.testing.assert_allclose(
        target.exterior_derivative(0, transfer.maps[0].mv(values)),
        transfer.maps[1].mv(source.exterior_derivative(0, values)),
        atol=2e-11,
    )
    points = jnp.asarray([[0.2, 0.4], [0.7, 0.8]], dtype=jnp.float64)
    np.testing.assert_allclose(
        target.reconstruction(0, transfer.maps[0].mv(values), points),
        source.reconstruction(0, values, points),
        atol=2e-11,
    )


def test_mapped_annulus_has_one_harmonic_one_form() -> None:
    radial = BSplineGrid.open_uniform(2, 2, interval=(0.0, 1.0))
    angular = BSplineGrid.open_uniform(1, 4, interval=(0.0, 1.0))
    complex_ = SplineDeRhamComplex(
        (radial, angular),
        periodic=(False, True),
        geometry=_annulus,
        geometry_id="analytic-annulus",
        quadrature_degree=18,
    )
    d0, d1 = (
        np.asarray(materialize(operator, MaterializationPolicy()))
        for operator in complex_.hilbert_complex().differentials
    )
    betti = complex_.dof_count(1) - np.linalg.matrix_rank(d0) - np.linalg.matrix_rank(d1)
    assert betti == 1
    harmonic = complex_.interpolant(1, _harmonic)
    np.testing.assert_allclose(complex_.hodge_laplacian(1, harmonic), 0.0, atol=2e-8)
    points = jnp.asarray([[0.2, 0.13], [0.8, 0.62]], dtype=jnp.float64)
    np.testing.assert_allclose(
        complex_.reconstruction(1, harmonic, points),
        jax.vmap(_harmonic)(jax.vmap(_annulus)(points)),
        atol=2e-9,
    )
    trace = complex_.trace(0, "upper")
    scalar = jnp.linspace(-0.4, 0.7, complex_.dof_count(0), dtype=jnp.float64)
    np.testing.assert_allclose(
        trace.target.differential(0).mv(trace.maps[0].mv(scalar)),
        trace.maps[1].mv(complex_.exterior_derivative(0, scalar)),
        atol=2e-12,
    )


def test_relative_hodge_uses_restricted_gram_inverse() -> None:
    grid = BSplineGrid.open_uniform(2, 3)
    complex_ = SplineDeRhamComplex((grid, grid))
    indices = np.asarray(complex_.active_indices(0, boundary="relative"))
    gram = _independent_gram(complex_, 0)
    rhs = jnp.linspace(-1.0, 0.5, indices.size, dtype=jnp.float64)
    space = complex_.hilbert_complex(boundary="relative").space(0)
    np.testing.assert_allclose(
        space.inverse_riesz(rhs),
        np.linalg.solve(gram[np.ix_(indices, indices)], np.asarray(rhs)),
        rtol=2e-8,
        atol=2e-8,
    )


def test_assembled_complex_refuses_nonzero_d_squared() -> None:
    space = ArraySpace((2,), dtype=jnp.float64, space_id="broken-assembly")
    gram = DenseLinearOperator(
        jnp.eye(2, dtype=jnp.float64),
        source=space,
        target=space,
        properties=OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "construction",
            },
        ),
        operator_id="broken-assembly:gram",
    )
    with pytest.raises(ValueError):
        AssembledSplineDeRhamComplex(
            2,
            (2, 2, 2),
            (np.eye(2), np.eye(2)),
            ("patch",),
            "broken",
            gram_operators=(gram,) * 3,
        )


@final
class _Stretch(StrictModule):
    scale: Array

    def __init__(self, scale: Array, /) -> None:
        self.scale = scale

    def __call__(self, point: Array, /) -> Array:
        return point.at[0].set(self.scale * point[0])


def test_prepared_metric_refresh_is_jittable_and_differentiable() -> None:
    grid = BSplineGrid.open_uniform(1, 2, interval=(0.0, 1.0))
    complex_ = SplineDeRhamComplex(
        (grid, grid),
        geometry=_Stretch(jnp.asarray(1.0, dtype=jnp.float64)),
        geometry_id="stretch-chart",
    )
    coefficients = jnp.ones((complex_.dof_count(0),), dtype=jnp.float64)

    def energy(scale: Array) -> Array:
        refreshed = complex_.refresh_geometry(_Stretch(scale))
        return coefficients @ refreshed.hodge_star(0, coefficients)

    value, derivative = jax.jit(jax.value_and_grad(energy))(
        jnp.asarray(1.7, dtype=jnp.float64)
    )
    np.testing.assert_allclose(value, 1.7, atol=2e-11)
    np.testing.assert_allclose(derivative, 1.0, atol=2e-10)


@pytest.mark.parametrize("side,sign", [("lower", -1), ("upper", 1)])
def test_annulus_boundary_trace_has_outward_period(side: str, sign: int) -> None:
    from phydrax.discretization.iga import BoundarySide
    from phydrax.typing import parse

    radial = BSplineGrid.open_uniform(1, 1, interval=(0.0, 1.0))
    angular = BSplineGrid.open_uniform(1, 4, interval=(0.0, 1.0))
    complex_ = SplineDeRhamComplex(
        (radial, angular),
        periodic=(False, True),
        geometry=_annulus,
        geometry_id="oriented-annulus",
        quadrature_degree=16,
    )
    trace = complex_.trace(0, parse(side, BoundarySide, "side"))
    values = complex_.interpolant(1, _harmonic)
    np.testing.assert_allclose(
        jnp.sum(trace.maps[1].mv(values)), sign * 2.0 * np.pi, atol=2e-10
    )
    scalar = jnp.linspace(-0.8, 1.3, complex_.dof_count(0), dtype=jnp.float64)
    np.testing.assert_allclose(
        trace.target.differential(0).mv(trace.maps[0].mv(scalar)),
        trace.maps[1].mv(complex_.exterior_derivative(0, scalar)),
        atol=2e-12,
    )


def test_mapped_quadrature_refuses_underresolved_gram() -> None:
    grid = BSplineGrid.open_uniform(2, 1, interval=(0.0, 1.0))
    with pytest.raises(ValueError, match="quadrature degree"):
        SplineDeRhamComplex(
            (grid, grid),
            geometry=_annulus,
            geometry_id="underresolved-annulus",
            quadrature_degree=1,
        )


def test_coarsening_projector_commutes_across_fine_knot_spans() -> None:
    coarse_grid = BSplineGrid.open_uniform(2, 1)
    fine_grid = BSplineGrid.open_uniform(2, 4)
    coarse = SplineDeRhamComplex((coarse_grid, coarse_grid))
    fine = SplineDeRhamComplex((fine_grid, fine_grid))
    inclusion = coarse.transfer(fine)
    projection = fine.transfer(coarse)
    fine_values = jnp.sin(jnp.arange(fine.dof_count(0), dtype=jnp.float64))
    np.testing.assert_allclose(
        coarse.exterior_derivative(0, projection.maps[0].mv(fine_values)),
        projection.maps[1].mv(fine.exterior_derivative(0, fine_values)),
        atol=2e-10,
    )
    coarse_values = jnp.linspace(-0.2, 0.8, coarse.dof_count(0), dtype=jnp.float64)
    np.testing.assert_allclose(
        projection.maps[0].mv(inclusion.maps[0].mv(coarse_values)),
        coarse_values,
        atol=2e-10,
    )


def test_mapped_complex_refuses_collapsed_boundary_chart() -> None:
    radial = BSplineGrid.open_uniform(1, 1, interval=(-1.0, 1.0))
    angular = BSplineGrid.open_uniform(1, 2, interval=(0.0, 1.0))
    with pytest.raises(eqx.EquinoxRuntimeError):
        SplineDeRhamComplex(
            (radial, angular),
            periodic=(False, True),
            geometry=_annulus,
            geometry_id="collapsed-inner-circle",
        )
