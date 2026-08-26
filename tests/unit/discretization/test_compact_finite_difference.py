from fractions import Fraction

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _grid(count, *, dimension=1):
    axes = tuple(
        phx.discretization.UniformAxisSpec(count, periodic=True, endpoint=False)
        for _ in range(dimension)
    )
    names = tuple("xyz"[:dimension])
    upper = jnp.ones((dimension,))
    return phx.discretization.TensorGridPlan(axes, axis_names=names).prepare(
        jnp.stack((jnp.zeros_like(upper), upper))
    )


def _derivative(count, derivative_order, accuracy_order):
    grid = _grid(count)
    request = phx.discretization.DerivativeRequest(
        f"d{derivative_order}",
        grid,
        "x",
        derivative_order=derivative_order,
        accuracy_order=accuracy_order,
    )
    return grid, phx.discretization.CompactDerivativePlan(grid, request).prepare()


@pytest.mark.parametrize("accuracy_order", (4, 6))
def test_periodic_compact_first_and_second_derivatives_converge(accuracy_order):
    errors = []
    for count in (16, 32):
        grid, first = _derivative(count, 1, accuracy_order)
        _, second = _derivative(count, 2, accuracy_order)
        x = grid.axes[0].nodes
        value = jnp.sin(2.0 * jnp.pi * x)
        first_error = jnp.max(
            jnp.abs(first.mv(value) - 2.0 * jnp.pi * jnp.cos(2.0 * jnp.pi * x))
        )
        second_error = jnp.max(
            jnp.abs(second.mv(value) + (2.0 * jnp.pi) ** 2 * value)
        )
        errors.append(jnp.maximum(first_error, second_error))
        assert first.report.passed
        assert second.report.passed
        assert first.report.dense_materialization_entries == 0
        assert second.report.dense_materialization_entries == 0
    assert errors[0] / errors[1] > 2.0 ** (accuracy_order - 1)


def test_compact_staggered_interpolation_and_derivative_preserve_locations():
    grid = _grid(32)
    point = grid.centered_location
    cell = grid.location((Fraction(1, 2),))
    interpolation = phx.discretization.CompactInterpolationPlan(
        grid,
        "x",
        point,
        cell,
        accuracy_order=6,
    ).prepare()
    request = phx.discretization.DerivativeRequest(
        "point-to-cell",
        grid,
        "x",
        derivative_order=1,
        accuracy_order=6,
        source_location=point,
        target_location=cell,
    )
    derivative = phx.discretization.CompactDerivativePlan(grid, request).prepare()
    source_x = grid.layout_at(point).coordinates_by_axis[0]
    target_x = grid.layout_at(cell).coordinates_by_axis[0]
    value = jnp.sin(2.0 * jnp.pi * source_x)

    np.testing.assert_allclose(
        interpolation.mv(value),
        jnp.sin(2.0 * jnp.pi * target_x),
        rtol=0.0,
        atol=2e-8,
    )
    np.testing.assert_allclose(
        derivative.mv(value),
        2.0 * jnp.pi * jnp.cos(2.0 * jnp.pi * target_x),
        rtol=0.0,
        atol=2e-6,
    )
    assert interpolation.source.shape == grid.layout_at(point).shape
    assert interpolation.target.shape == grid.layout_at(cell).shape


def test_compact_tensor_components_transpose_adjoint_and_grad():
    grid = _grid(24, dimension=2)
    request = phx.discretization.DerivativeRequest(
        "dy",
        grid,
        "y",
        derivative_order=1,
        accuracy_order=6,
    )
    operator = phx.discretization.CompactDerivativePlan(
        grid,
        request,
        component_shape=(2,),
    ).prepare()
    x, y = jnp.meshgrid(grid.axes[0].nodes, grid.axes[1].nodes, indexing="ij")
    value = jnp.stack((jnp.sin(2 * jnp.pi * y), jnp.cos(2 * jnp.pi * y)), axis=-1)
    covector = jnp.stack((jnp.cos(2 * jnp.pi * x), jnp.sin(2 * jnp.pi * x)), axis=-1)
    result = eqx.filter_jit(operator.mv)(value)
    coordinate_left = jnp.vdot(covector, result)
    coordinate_right = jnp.vdot(operator.transpose_mv(covector), value)
    pairing_left = operator.target.inner(covector, result)
    pairing_right = operator.source.inner(operator.adjoint_mv(covector), value)
    gradient = jax.grad(lambda scale: jnp.sum(operator.mv(scale * value) ** 2))(1.0)

    np.testing.assert_allclose(coordinate_left, coordinate_right, atol=2e-10)
    np.testing.assert_allclose(pairing_left, pairing_right, atol=2e-10)
    assert jnp.isfinite(gradient)
    assert result.shape == value.shape


def test_compact_rejects_unsupported_structure_and_dense_materialization():
    grid, operator = _derivative(16, 1, 4)
    with pytest.raises(ValueError, match="prohibit"):
        operator._materialize()
    bounded = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformAxisSpec(16),), axis_names=("x",)
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    request = phx.discretization.DerivativeRequest(
        "dx", bounded, "x", derivative_order=1, accuracy_order=4
    )
    with pytest.raises(ValueError, match="periodic"):
        phx.discretization.CompactDerivativePlan(bounded, request)
    biased = phx.discretization.DerivativeRequest(
        "biased",
        grid,
        "x",
        derivative_order=1,
        accuracy_order=4,
        bias="forward",
    )
    with pytest.raises(ValueError, match="centered"):
        phx.discretization.CompactDerivativePlan(grid, biased)
