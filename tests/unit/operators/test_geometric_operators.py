#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

import phydrax as phx


def test_domain_signed_and_horizontal_operators_preserve_labeled_semantics():
    spacetime = phx.domain.HyperRectangle([-2.0] * 4, [2.0] * 4, label="x")

    @spacetime.Function("x")
    def field(x):
        return -(x[0] ** 2) + jnp.sum(x[1:] ** 2)

    chart = phx.metrix.CoordinateChart("spacetime", ("t", "x", "y", "z"))
    metric = phx.metrix.minkowski_metric(chart)
    point = jnp.array([0.1, 0.2, 0.3, 0.4])

    assert jnp.allclose(
        phx.operators.intrinsic_dalembertian(field, metric, var="x").func(point),
        8.0,
    )
    assert jnp.allclose(
        phx.operators.semi_riemannian_grad(field, metric, var="x").func(point),
        jnp.array([0.2, 0.4, 0.6, 0.8]),
    )

    horizontal = phx.metrix.HorizontalCometric(
        lambda q: jnp.array([[1.0, 0.0], [0.0, 1.0], [-0.5 * q[1], 0.5 * q[0]]]),
        phx.metrix.CoordinateChart("heisenberg", ("x", "y", "z")),
        2,
    )
    domain = phx.domain.HyperRectangle([-1.0] * 3, [1.0] * 3, label="q")

    @domain.Function("q")
    def radial(q):
        return q[0] ** 2 + q[1] ** 2

    horizontal_point = jnp.array([0.2, 0.3, -0.1])
    assert jnp.allclose(
        phx.operators.horizontal_grad(radial, horizontal, var="q").func(horizontal_point),
        jnp.array([0.4, 0.6, 0.0]),
    )
    assert jnp.allclose(
        phx.operators.sub_laplacian(radial, horizontal, var="q").func(horizontal_point),
        4.0,
    )


def test_domain_differential_forms_share_continuous_exterior_calculus():
    domain = phx.domain.HyperRectangle([-1.0, -1.0], [1.0, 1.0], label="x")
    chart = phx.metrix.CoordinateChart("plane", ("x", "y"))

    @domain.Function("x")
    def scalar(x):
        return x[0] ** 2 + x[1] ** 2

    form = phx.operators.domain_differential_form(
        scalar,
        chart=chart,
        degree=0,
        var="x",
    )
    differential = phx.operators.domain_exterior_derivative(form)

    assert differential.degree == 1
    assert jnp.allclose(
        differential.coefficients.func(jnp.array([0.2, 0.3])),
        jnp.array([0.4, 0.6]),
    )
    second_differential = phx.operators.domain_exterior_derivative(differential)
    metric = phx.metrix.RiemannianMetric(lambda q: jnp.eye(2), chart=chart)
    hodge = phx.operators.domain_hodge_star(form, metric)
    codifferential = phx.operators.domain_codifferential(form, metric)
    laplacian = phx.operators.domain_hodge_laplacian(form, metric)
    point = jnp.array([0.2, 0.3])

    assert jnp.allclose(second_differential.coefficients.func(point), jnp.array([0.0]))
    assert jnp.allclose(hodge.coefficients.func(point), jnp.array([0.13]))
    assert jnp.allclose(codifferential.coefficients.func(point), jnp.array([0.0]))
    assert jnp.allclose(laplacian.coefficients.func(point), jnp.array([-4.0]))
    mismatched_metric = phx.metrix.RiemannianMetric(
        lambda q: jnp.eye(2),
        chart=phx.metrix.CoordinateChart("other-plane", ("u", "v")),
    )
    with pytest.raises(ValueError, match="charts must match"):
        phx.operators.domain_codifferential(form, mismatched_metric)


def test_declared_poisson_structure_drives_domain_bracket_and_flow():
    q_space = phx.domain.HyperRectangle([-2.0], [2.0], label="q")
    p_space = phx.domain.HyperRectangle([-2.0], [2.0], label="p")
    phase = phx.domain.ProductDomain(q_space, p_space)

    @phase.Function("q", "p")
    def hamiltonian(q, p):
        return 0.5 * (q[0] ** 2 + p[0] ** 2)

    @phase.Function("q")
    def configuration(q):
        return q[0]

    @phase.Function("p")
    def momentum(p):
        return p[0]

    chart = phx.metrix.CoordinateChart("phase", ("q", "p"))
    symplectic = phx.metrix.canonical_symplectic_form(chart)
    point_q = jnp.array([2.0])
    point_p = jnp.array([3.0])

    assert jnp.allclose(
        phx.operators.poisson_bracket(
            configuration,
            momentum,
            symplectic,
            variables=("q", "p"),
        ).func(point_q, point_p),
        1.0,
    )
    assert jnp.allclose(
        phx.operators.hamiltonian_vector_field(
            hamiltonian,
            symplectic,
            variables=("q", "p"),
        ).func(point_q, point_p),
        jnp.array([3.0, -2.0]),
    )
