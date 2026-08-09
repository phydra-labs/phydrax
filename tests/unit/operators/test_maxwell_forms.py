#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _spacetime():
    domain = phx.domain.HyperRectangle([-2.0] * 4, [2.0] * 4, label="x")
    chart = phx.metrix.CoordinateChart("maxwell", ("t", "x", "y", "z"))
    return domain, chart, phx.metrix.minkowski_metric(chart)


def test_plane_wave_potential_satisfies_both_vacuum_maxwell_form_residuals():
    domain, chart, metric = _spacetime()

    @domain.Function("x")
    def potential_coefficients(x):
        return jnp.array([0.0, 0.0, jnp.sin(1.7 * (x[0] - x[1])), 0.0])

    potential = phx.operators.domain_differential_form(
        potential_coefficients,
        chart=chart,
        degree=1,
        var="x",
    )
    field_strength = phx.operators.domain_exterior_derivative(potential)
    residuals = phx.operators.domain_maxwell_residuals(field_strength, metric)
    points = jnp.array(
        [[0.2, -0.3, 0.1, 0.4], [0.7, 0.1, -0.2, 0.3]]
    )

    assert residuals.field_strength is field_strength
    assert residuals.homogeneous.degree == 3
    assert residuals.inhomogeneous.degree == 1
    assert jnp.allclose(
        jax.jit(jax.vmap(lambda q: residuals.homogeneous.coefficients.func(q)))(
            points
        ),
        0.0,
        atol=1e-10,
    )
    assert jnp.allclose(
        jax.jit(jax.vmap(lambda q: residuals.inhomogeneous.coefficients.func(q)))(
            points
        ),
        0.0,
        atol=1e-10,
    )


def test_electric_and_magnetic_form_sources_use_declared_maxwell_signs():
    domain, chart, metric = _spacetime()
    charge_density = 1.7

    @domain.Function("x")
    def electrostatic_potential(x):
        return jnp.array([0.5 * charge_density * x[1] ** 2, 0.0, 0.0, 0.0])

    @domain.Function("x")
    def electric_current_coefficients(x):
        del x
        return jnp.array([charge_density, 0.0, 0.0, 0.0])

    electric_field_strength = phx.operators.domain_exterior_derivative(
        phx.operators.domain_differential_form(
            electrostatic_potential,
            chart=chart,
            degree=1,
            var="x",
        )
    )
    electric_current = phx.operators.domain_differential_form(
        electric_current_coefficients,
        chart=chart,
        degree=1,
        var="x",
    )
    electric_residuals = phx.operators.domain_maxwell_residuals(
        electric_field_strength,
        metric,
        electric_current=electric_current,
    )

    @domain.Function("x")
    def magnetic_field_strength_coefficients(x):
        return jnp.array([0.0, 0.0, 0.0, x[3], 0.0, 0.0])

    @domain.Function("x")
    def magnetic_current_coefficients(x):
        del x
        return jnp.array([0.0, 0.0, 0.0, 1.0])

    magnetic_field_strength = phx.operators.domain_differential_form(
        magnetic_field_strength_coefficients,
        chart=chart,
        degree=2,
        var="x",
    )
    magnetic_current = phx.operators.domain_differential_form(
        magnetic_current_coefficients,
        chart=chart,
        degree=3,
        var="x",
    )
    magnetic_residuals = phx.operators.domain_maxwell_residuals(
        magnetic_field_strength,
        metric,
        magnetic_current=magnetic_current,
    )
    point = jnp.array([0.2, 0.3, -0.1, 0.4])

    assert jnp.allclose(
        electric_residuals.homogeneous.coefficients.func(point),
        0.0,
    )
    assert jnp.allclose(
        electric_residuals.inhomogeneous.coefficients.func(point),
        0.0,
    )
    assert jnp.allclose(
        magnetic_residuals.homogeneous.coefficients.func(point),
        0.0,
    )
    assert jnp.allclose(
        magnetic_residuals.inhomogeneous.coefficients.func(point),
        0.0,
    )


def test_maxwell_form_residuals_compose_into_standard_solver_conditions():
    domain, chart, metric = _spacetime()
    component = domain.component()

    @domain.Function("x")
    def field_strength_coefficients(x):
        phase = 1.3 * (x[0] - x[1])
        amplitude = 1.3 * jnp.cos(phase)
        return jnp.array([0.0, amplitude, 0.0, -amplitude, 0.0, 0.0])

    def maxwell(coefficients):
        field_strength = phx.operators.domain_differential_form(
            coefficients,
            chart=chart,
            degree=2,
            var="x",
        )
        return phx.operators.domain_maxwell_residuals(field_strength, metric)

    homogeneous = phx.conditions.Residual(
        "field_strength",
        component,
        lambda coefficients: maxwell(coefficients).homogeneous.coefficients,
    )
    inhomogeneous = phx.conditions.Residual(
        "field_strength",
        component,
        lambda coefficients: maxwell(coefficients).inhomogeneous.coefficients,
    )
    points = jnp.array(
        [[0.1, -0.2, 0.3, 0.4], [0.5, 0.2, -0.1, 0.3]]
    )
    realization = phx.integration.from_samples(
        phx.integration.mean_over(component),
        component.points(points),
    )
    source = phx.integration.fixed(realization)
    solver = phx.solver.FunctionalSolver(
        functions={"field_strength": field_strength_coefficients},
        terms=(
            phx.terms.ResidualPenalty(homogeneous, source),
            phx.terms.ResidualPenalty(inhomogeneous, source),
        ),
    )

    assert jnp.allclose(eqx.filter_jit(solver.loss)(), 0.0, atol=1e-10)


def test_maxwell_form_residuals_reject_wrong_degrees_and_metric_family():
    domain, chart, metric = _spacetime()

    @domain.Function("x")
    def covector_coefficients(x):
        return x

    covector = phx.operators.domain_differential_form(
        covector_coefficients,
        chart=chart,
        degree=1,
        var="x",
    )
    riemannian = phx.metrix.RiemannianMetric(lambda q: jnp.eye(4), chart=chart)

    with pytest.raises(ValueError, match="degree-2"):
        phx.operators.domain_maxwell_residuals(covector, metric)
    with pytest.raises(TypeError, match="LorentzianMetric"):
        phx.operators.domain_maxwell_residuals(
            phx.operators.domain_exterior_derivative(covector),
            riemannian,
        )
