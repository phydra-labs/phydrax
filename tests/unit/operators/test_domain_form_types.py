from collections.abc import Callable
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax import Array

from phydrax.domain import CallbackDerivativeRule, DomainFunction, HyperRectangle
from phydrax.metrix import CoordinateChart, RiemannianMetric
from phydrax.operators import (
    domain_codifferential,
    domain_exterior_derivative,
    domain_hodge_laplacian,
    domain_hodge_star,
    domain_interior_product,
    domain_lie_derivative,
    domain_pullback_form,
    domain_to_twisted,
    domain_to_untwisted,
    domain_wedge,
    DomainDifferentialForm,
)


@eqx.filter_jit
def _compiled_value(function: Callable[[Array], Array], point: Array) -> Array:
    return function(point)


def _plane() -> tuple[HyperRectangle, CoordinateChart, RiemannianMetric]:
    domain = HyperRectangle(jnp.array([-2.0, -2.0]), jnp.array([2.0, 2.0]), label="x")
    chart = CoordinateChart("domain_form_plane", ("x", "y"))
    return domain, chart, RiemannianMetric(lambda q: jnp.eye(2), chart=chart)


@pytest.mark.parametrize("mode", ["forward", "reverse"])
@pytest.mark.parametrize("backend", ["ad", "jet"])
def test_exterior_derivative_preserves_twist_and_explicit_singleton_batch(
    mode: Literal["forward", "reverse"],
    backend: Literal["ad", "jet"],
) -> None:
    domain, chart, _ = _plane()
    field = domain.Function("x")(lambda x: jnp.array([x[0] ** 2 * x[1]]))
    form = DomainDifferentialForm(field, chart=chart, degree=0, twist="twisted")
    derivative = domain_exterior_derivative(form, mode=mode, backend=backend)
    points = jnp.array([[0.3, -0.4]])
    values = eqx.filter_jit(lambda function, x: jax.vmap(function)(x))(
        derivative.coefficients.func, points
    )
    assert values.shape == (1, 2)
    assert derivative.twist == "twisted"
    assert jnp.allclose(values, jnp.array([[-0.24, 0.09]]))
    second = domain_exterior_derivative(derivative, mode=mode, backend=backend)
    assert jnp.allclose(jax.vmap(second.coefficients.func)(points), jnp.zeros((1, 1)))


def test_scalar_coefficients_without_component_axis_are_refused_even_for_one_sample() -> (
    None
):
    domain, chart, metric = _plane()
    field = domain.Function("x")(lambda x: x[0] ** 2)
    invalid = DomainDifferentialForm(field, chart=chart, degree=0)
    points = jnp.array([[0.3, -0.4]])
    with pytest.raises(ValueError):
        jax.vmap(domain_hodge_star(invalid, metric).coefficients.func)(points)
    with pytest.raises(ValueError):
        jax.vmap(domain_exterior_derivative(invalid).coefficients.func)(points)


def test_exterior_derivative_uses_registered_coefficient_rule() -> None:
    domain, chart, _ = _plane()

    def derive(**request: Any) -> DomainFunction:
        axis = request["axis"]
        # The declared derivative is analytic data, not the AD derivative of stop_gradient.
        return domain.Function("x")(
            lambda x: jnp.array([2.0 * x[0] if axis == 0 else 3.0])
        )

    field = domain.Function("x")(
        lambda x: jax.lax.stop_gradient(jnp.array([x[0] ** 2 + 3.0 * x[1]]))
    ).with_derivative_rule(CallbackDerivativeRule(derive))
    derivative = domain_exterior_derivative(
        DomainDifferentialForm(field, chart=chart, degree=0)
    )
    assert jnp.allclose(
        derivative.coefficients.func(jnp.array([0.4, 0.2])), jnp.array([0.8, 3.0])
    )


@pytest.mark.parametrize("mode", ["forward", "reverse"])
@pytest.mark.parametrize("backend", ["ad", "jet"])
def test_hodge_twist_round_trip_and_orientation_free_matrix_laplacian(
    mode: Literal["forward", "reverse"],
    backend: Literal["ad", "jet"],
) -> None:
    domain, chart, metric = _plane()
    matrix = jnp.array([[1.0, 2.0], [3.0, -1.0]])
    field = domain.Function("x")(lambda x: ((x[0] ** 2 + x[1] ** 2) * matrix)[None, ...])
    form = DomainDifferentialForm(field, chart=chart, degree=0, fiber_shape=(2, 2))
    star = domain_hodge_star(form, metric)
    point = jnp.array([0.2, -0.3])
    assert star.twist == "twisted"
    assert star.fiber_shape == (2, 2)
    assert jnp.allclose(
        domain_hodge_star(star, metric).coefficients.func(point), 0.13 * matrix[None, ...]
    )
    untwisted = domain_to_untwisted(star, -1)
    assert jnp.allclose(untwisted.coefficients.func(point), -0.13 * matrix[None, ...])
    assert jnp.allclose(
        domain_to_twisted(untwisted, -1).coefficients.func(point),
        star.coefficients.func(point),
    )
    laplacian = domain_hodge_laplacian(form, metric, mode=mode, backend=backend)
    assert laplacian.twist == "untwisted"
    assert jnp.allclose(
        _compiled_value(laplacian.coefficients.func, point), -4.0 * matrix[None, ...]
    )
    with pytest.raises(ValueError):
        domain_codifferential(form, metric)
    with pytest.raises(ValueError):
        domain_to_twisted(star, 1)


def test_matrix_wedge_interior_and_lie_derivative_follow_noncommuting_fibers() -> None:
    domain, chart, _ = _plane()
    a = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    b = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    left_field = domain.Function("x")(lambda x: jnp.stack([x[0] * a, jnp.zeros_like(a)]))
    right_field = domain.Function("x")(lambda x: jnp.stack([jnp.zeros_like(b), x[1] * b]))
    left = DomainDifferentialForm(left_field, chart=chart, degree=1, fiber_shape=(2, 2))
    right = DomainDifferentialForm(
        right_field, chart=chart, degree=1, twist="twisted", fiber_shape=(2, 2)
    )
    product = domain_wedge(left, right, product="matrix")
    reverse = domain_wedge(right, left, product="matrix")
    point = jnp.array([0.4, 0.7])
    assert product.twist == "twisted"
    assert jnp.allclose(product.coefficients.func(point), (0.28 * (a @ b))[None, ...])
    assert jnp.allclose(reverse.coefficients.func(point), (-0.28 * (b @ a))[None, ...])
    vector = domain.Function("x")(lambda x: jnp.array([1.0, 0.0]))
    contraction = domain_interior_product(vector, product)
    assert contraction.twist == "twisted"
    assert jnp.allclose(
        contraction.coefficients.func(point),
        jnp.stack([jnp.zeros_like(a), 0.28 * (a @ b)]),
    )
    lie = domain_lie_derivative(vector, product)
    assert jnp.allclose(lie.coefficients.func(point), (0.7 * (a @ b))[None, ...])


def test_reflection_pullback_distinguishes_twists_and_commutes_with_derivative() -> None:
    domain, chart, _ = _plane()
    source_chart = CoordinateChart("reflected_plane", ("u", "v"))
    mapping = domain.Function("x")(lambda x: jnp.array([-x[0], x[1]]))
    field = domain.Function("x")(lambda x: jnp.array([x[0], x[0] * x[1]]))
    form = DomainDifferentialForm(field, chart=chart, degree=1)
    pullback = domain_pullback_form(form, mapping, source_chart=source_chart)
    twisted = domain_pullback_form(
        domain_to_twisted(form, 1), mapping, source_chart=source_chart
    )
    point = jnp.array([0.4, 0.7])
    assert jnp.allclose(pullback.coefficients.func(point), jnp.array([0.4, -0.28]))
    assert jnp.allclose(
        twisted.coefficients.func(point), -pullback.coefficients.func(point)
    )
    before = domain_pullback_form(
        domain_exterior_derivative(form), mapping, source_chart=source_chart
    )
    after = domain_exterior_derivative(pullback)
    assert jnp.allclose(before.coefficients.func(point), jnp.array([-0.7]))
    assert jnp.allclose(after.coefficients.func(point), jnp.array([-0.7]))


def test_embedded_twisted_pullback_requires_coorientation() -> None:
    domain, chart, _ = _plane()
    field = domain.Function("x")(lambda x: jnp.array([1.0, 2.0]))
    form = DomainDifferentialForm(field, chart=chart, degree=1, twist="twisted")
    source = HyperRectangle(jnp.array([-1.0]), jnp.array([1.0]), label="s")
    source_chart = CoordinateChart("boundary_curve", ("s",))
    mapping = source.Function("s")(lambda s: jnp.array([s[0], 0.0]))
    with pytest.raises(ValueError, match="coorientation"):
        domain_pullback_form(form, mapping, source_chart=source_chart)
    trace = domain_pullback_form(
        form, mapping, source_chart=source_chart, coorientation=-1
    )
    assert trace.degree == 1
    assert trace.twist == "twisted"
    assert jnp.allclose(trace.coefficients.func(jnp.array([0.2])), jnp.array([-1.0]))


@pytest.mark.parametrize("backend", ["fd", "basis"])
def test_domain_derivative_forwards_grid_backends_with_explicit_components(
    backend: Literal["fd", "basis"],
) -> None:
    domain, chart, _ = _plane()
    field = domain.Function("x")(lambda x: jnp.array([2.0 * x[0] + 3.0 * x[1]]))
    form = DomainDifferentialForm(field, chart=chart, degree=0, twist="twisted")
    derivative = domain_exterior_derivative(form, backend=backend)
    coordinates = (jnp.linspace(-0.7, 0.7, 5), jnp.linspace(-0.4, 0.4, 4))
    values = derivative.coefficients.func(coordinates)
    assert values.shape == (5, 4, 2)
    assert derivative.twist == "twisted"
    assert jnp.allclose(values, jnp.broadcast_to(jnp.array([2.0, 3.0]), (5, 4, 2)))
