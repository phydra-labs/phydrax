"""Smooth carrier laws independent of the coefficient implementation."""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax import metrix


pytestmark = pytest.mark.strict_jax


@eqx.filter_jit
def _compiled_value(function: Callable[[Array], Array], point: Array) -> Array:
    return function(point)


def _polynomial(point: Array) -> Array:
    x, y, z = point
    return jnp.stack((x * y * z, x * x * z, x * y * y))


def test_cartan_radial_polynomial_jit_and_ad() -> None:
    chart = metrix.CoordinateChart("radial_cartan", ("x", "y", "z"))
    form = metrix.DifferentialForm(_polynomial, chart=chart, degree=1, twist="twisted")
    lie = metrix.lie_derivative(lambda q: q, form)
    point = jnp.asarray([2.0, 3.0, 5.0], dtype=jnp.float64)
    # Euler homogeneity contributes degree three, and the one-form slot one.
    np.testing.assert_allclose(_compiled_value(lie, point), 4.0 * _polynomial(point))
    np.testing.assert_allclose(
        jax.jacrev(lie)(point), 4.0 * jax.jacrev(_polynomial)(point)
    )
    assert lie.twist == "twisted"
    derivative = metrix.exterior_derivative(form)
    twice = metrix.exterior_derivative(derivative)
    np.testing.assert_allclose(
        _compiled_value(twice, point), jnp.zeros((1,), dtype=jnp.float64), atol=1e-12
    )


def test_smooth_leibniz_and_scalar_laplacian() -> None:
    chart = metrix.CoordinateChart("leibniz", ("x", "y", "z"))
    metric = metrix.euclidean_metric(chart)
    scalar = metrix.DifferentialForm(
        lambda q: jnp.reshape(jnp.sum(q * q), (1,)), chart=chart, degree=0
    )
    form = metrix.DifferentialForm(_polynomial, chart=chart, degree=1, twist="twisted")
    point = jnp.asarray([2.0, 3.0, 5.0], dtype=jnp.float64)
    left = metrix.exterior_derivative(metrix.wedge(scalar, form))
    first = metrix.wedge(metrix.exterior_derivative(scalar), form)
    second = metrix.wedge(scalar, metrix.exterior_derivative(form))
    np.testing.assert_allclose(
        _compiled_value(left, point), first(point) + second(point), atol=1e-12
    )
    np.testing.assert_allclose(
        _compiled_value(metrix.hodge_laplacian(scalar, metric), point),
        jnp.asarray([-6.0], dtype=jnp.float64),
    )
    with pytest.raises(ValueError, match="zero"):
        metrix.codifferential(scalar, metric)


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
@pytest.mark.parametrize("lorentzian", [False, True])
def test_orientation_free_star_square_and_explicit_conversion(
    degree: int, lorentzian: bool
) -> None:
    chart = metrix.CoordinateChart("star_square", ("t", "x", "y"))
    metric = (
        metrix.minkowski_metric(chart) if lorentzian else metrix.euclidean_metric(chart)
    )
    from phydrax.exterior import FormType

    values = jnp.arange(1, FormType(3, degree).component_count + 1, dtype=jnp.float64)
    form = metrix.DifferentialForm(lambda q: values, chart=chart, degree=degree)
    point = jnp.asarray([0.2, 0.4, 0.6], dtype=jnp.float64)
    star = metrix.hodge_star(form, metric)
    assert star.twist == "twisted"
    second = metrix.hodge_star(star, metric)
    assert second.twist == "untwisted"
    expected = (-1) ** (degree * (3 - degree) + (1 if lorentzian else 0))
    np.testing.assert_allclose(_compiled_value(second, point), expected * values)
    oriented = metrix.to_untwisted(star, -1)
    np.testing.assert_allclose(oriented(point), -star(point))
    np.testing.assert_allclose(metrix.to_twisted(oriented, -1)(point), star(point))


def test_smooth_matrix_wedge_and_hodge_preserve_fiber_axes() -> None:
    chart = metrix.CoordinateChart("matrix_wedge", ("x", "y"))
    metric = metrix.euclidean_metric(chart)
    a = jnp.asarray([[0.0, 1.0], [0.0, 0.0]], dtype=jnp.float64)
    b = jnp.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.float64)
    zero = jnp.zeros((2, 2), dtype=jnp.float64)
    left = metrix.DifferentialForm(
        lambda q: jnp.stack((q[0] * a, zero)), chart=chart, degree=1, fiber_shape=(2, 2)
    )
    right = metrix.DifferentialForm(
        lambda q: jnp.stack((zero, q[1] * b)), chart=chart, degree=1, fiber_shape=(2, 2)
    )
    product = metrix.wedge(left, right, product="matrix")
    point = jnp.asarray([2.0, 3.0], dtype=jnp.float64)
    np.testing.assert_allclose(
        _compiled_value(product, point), (6.0 * (a @ b))[None, ...]
    )
    np.testing.assert_allclose(
        jax.jacfwd(product)(point)[0], jnp.stack((3.0 * (a @ b), 2.0 * (a @ b)), axis=-1)
    )
    double = metrix.hodge_star(metrix.hodge_star(left, metric), metric)
    np.testing.assert_allclose(
        _compiled_value(double, point[None, ...]), -left(point)[None, ...]
    )
    assert double.form_type == left.form_type


def test_g2_keeps_oriented_output_at_explicit_conversion_boundary() -> None:
    chart = metrix.CoordinateChart("g2_orientation", tuple(f"x{i}" for i in range(7)))
    algebra = metrix.algebra.OctonionAlgebraSpec()
    positive = metrix.OctonionG2Bridge(algebra, chart, orientation=1)
    negative = metrix.OctonionG2Bridge(algebra, chart, orientation=-1)
    point = jnp.zeros((7,), dtype=jnp.float64)
    ordinary = positive.associative_differential_form()
    twisted_dual = metrix.hodge_star(ordinary, positive.metric)
    assert twisted_dual.twist == "twisted"
    assert positive.coassociative_differential_form().twist == "untwisted"
    np.testing.assert_allclose(
        positive.coassociative_differential_form()(point), twisted_dual(point)
    )
    np.testing.assert_allclose(
        negative.associative_differential_form()(point), -ordinary(point)
    )
    np.testing.assert_allclose(
        negative.coassociative_differential_form()(point),
        positive.coassociative_differential_form()(point),
    )


def test_scalar_coefficients_require_component_axis_even_for_single_sample() -> None:
    chart = metrix.CoordinateChart("singleton", ("x", "y"))
    good = metrix.DifferentialForm(lambda q: q[:1], chart=chart, degree=0)
    point = jnp.asarray([[2.0, 3.0]], dtype=jnp.float64)
    np.testing.assert_array_equal(good(point), jnp.asarray([[2.0]], dtype=jnp.float64))
    bad = metrix.DifferentialForm(lambda q: q[0], chart=chart, degree=0)
    with pytest.raises(ValueError, match="shape"):
        bad(point)


def test_bigraded_dolbeault_derivatives_anticommute_on_nonholomorphic_polynomial() -> (
    None
):
    chart = metrix.CoordinateChart("dolbeault", ("x0", "x1", "y0", "y1"))
    convention = metrix.ComplexCoordinateConvention(chart)

    def coefficients(point: Array) -> Array:
        q = point.astype(jnp.complex128)
        z0, z1 = q[0] + 1j * q[2], q[1] + 1j * q[3]
        return jnp.reshape(z0 * jnp.conj(z1), (1,))

    form = metrix.BigradedForm(coefficients, convention=convention, bidegree=(0, 0))
    point = jnp.asarray([2.0, 3.0, 5.0, 7.0], dtype=jnp.float64)
    np.testing.assert_allclose(
        form(point[None, ...]), jnp.asarray([[41.0 + 1.0j]], dtype=jnp.complex128)
    )
    np.testing.assert_allclose(
        metrix.partial(form)(point), jnp.asarray([3.0 - 7.0j, 0.0], dtype=jnp.complex128)
    )
    np.testing.assert_allclose(
        metrix.partial_bar(form)(point),
        jnp.asarray([0.0, 2.0 + 5.0j], dtype=jnp.complex128),
    )
    expected = jnp.asarray([0.0, 1.0, 0.0, 0.0], dtype=jnp.complex128)
    np.testing.assert_allclose(
        _compiled_value(metrix.partial(metrix.partial_bar(form)), point), expected
    )
    np.testing.assert_allclose(metrix.partial_bar(metrix.partial(form))(point), -expected)


def test_bigraded_scalar_coefficients_require_an_explicit_component_axis() -> None:
    chart = metrix.CoordinateChart("bigraded_singleton", ("x", "y"))
    convention = metrix.ComplexCoordinateConvention(chart)
    form = metrix.BigradedForm(lambda q: q[0], convention=convention, bidegree=(0, 0))
    with pytest.raises(ValueError, match="shape"):
        form(jnp.asarray([[2.0, 3.0]], dtype=jnp.float64))
