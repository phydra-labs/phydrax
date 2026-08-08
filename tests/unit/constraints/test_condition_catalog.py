#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def _source(component):
    return phx.integration.per_step(
        phx.integration.over(component),
        phx.integration.MonteCarloPlan(16),
    )


def _loss(condition, source, functions):
    penalty = phx.terms.ResidualPenalty(condition, source)
    return penalty.loss(functions, key=jr.key(0))


def test_boundary_and_initial_conditions_are_treatment_independent():
    geometry = phx.domain.Interval1d(0.0, 1.0)
    boundary = geometry.component({"x": phx.domain.Boundary()})
    zero = geometry.Function()(0.0)

    conditions = (
        (phx.conditions.Dirichlet("u", boundary), _source(boundary)),
        (phx.conditions.Neumann("u", boundary), _source(boundary)),
        (
            phx.conditions.Robin(
                "u",
                boundary,
                dirichlet_coefficient=1.0,
                neumann_coefficient=1.0,
            ),
            _source(boundary),
        ),
    )

    for condition, source in conditions:
        assert jnp.allclose(_loss(condition, source, {"u": zero}), 0.0)

    time = phx.domain.ScalarInterval(0.0, 1.0, label="t")
    initial = time.component({"t": phx.domain.FixedStart()})
    initial_condition = phx.conditions.Initial("u", initial)
    initial_source = phx.integration.per_step(
        phx.integration.over(initial),
        phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(4)),
    )
    assert jnp.allclose(
        _loss(initial_condition, initial_source, {"u": time.Function()(0.0)}),
        0.0,
    )


def test_cfd_thermal_solid_and_electromagnetic_catalogs_preserve_formulas():
    geometry = phx.domain.Interval1d(0.0, 1.0)
    boundary = geometry.component({"x": phx.domain.Boundary()})
    source = _source(boundary)
    scalar_zero = geometry.Function()(0.0)
    vector_zero = geometry.Function()(jnp.zeros((1,)))

    conditions = (
        phx.conditions.cfd.NoPenetration("velocity", boundary),
        phx.conditions.cfd.ZeroNormalGradientVelocity("velocity", boundary),
        phx.conditions.thermal.HeatFlux(
            "temperature",
            boundary,
            conductivity=1.0,
        ),
        phx.conditions.thermal.Convection(
            "temperature",
            boundary,
            heat_transfer_coefficient=2.0,
            conductivity=1.0,
        ),
        phx.conditions.solids.Traction(
            "displacement",
            boundary,
            lambda_=1.0,
            mu=1.0,
        ),
        phx.conditions.electromagnetics.PEC("electric", boundary),
        phx.conditions.electromagnetics.PMC("magnetic", boundary),
    )
    functions = {
        "velocity": vector_zero,
        "temperature": scalar_zero,
        "displacement": vector_zero,
        "electric": vector_zero,
        "magnetic": vector_zero,
    }

    for condition in conditions:
        assert jnp.allclose(_loss(condition, source, functions), 0.0)


def test_stochastic_and_conservation_catalogs_use_generic_penalties():
    geometry = phx.domain.Interval1d(0.0, 1.0)
    interior = geometry.component()
    boundary = geometry.component({"x": phx.domain.Boundary()})
    density = geometry.Function()(0.0)
    drift = geometry.Function()(jnp.zeros((1,)))

    fokker_planck = phx.conditions.stochastic.FokkerPlanck(
        "density",
        interior,
        drift=drift,
        evolution_var=None,
        covariance=geometry.Function()(jnp.eye(1)),
    )
    assert jnp.allclose(
        _loss(fokker_planck, _source(interior), {"density": density}),
        0.0,
    )

    pressure = geometry.Function()(0.0)
    pressure_condition = phx.conditions.conservation.PressureIntegral(
        "pressure",
        boundary,
        0.0,
    )
    pressure_penalty = phx.terms.MomentPenalty(
        pressure_condition,
        _source(boundary),
    )
    assert jnp.allclose(
        pressure_penalty.loss({"pressure": pressure}, key=jr.key(1)),
        0.0,
    )
