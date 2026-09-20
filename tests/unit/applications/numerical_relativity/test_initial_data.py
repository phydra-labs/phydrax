#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

from phydrax.applications.numerical_relativity._initial_data import (
    adm_charge_diagnostics,
    adm_constraint_diagnostics,
    IsotropicSchwarzschildInitialData,
    KerrSchildInitialData,
    MinkowskiInitialData,
)
from phydrax.applications.numerical_relativity._puncture import (
    BowenYorkInitialData,
    BrillLindquistInitialData,
    Puncture,
    TwoPunctureHamiltonianPlan,
)
from phydrax.metrix._adm_exchange import ADMGridGeometry


def _pair(momentum=0.0):
    first = Puncture(
        0.45,
        (-0.6, 0.0, 0.0),
        linear_momentum=(0.0, momentum, 0.0),
        spin=(0.0, 0.0, 0.015),
    )
    second = Puncture(
        0.45,
        (0.6, 0.0, 0.0),
        linear_momentum=(0.0, -momentum, 0.0),
        spin=(0.0, 0.0, 0.015),
    )
    return first, second


def test_minkowski_fields_constraints_and_shared_adm_exchange_contract():
    points = jnp.asarray(((0.2, -0.3, 0.4), (1.0, 2.0, -1.0)))
    field = MinkowskiInitialData()
    data = field(points)

    assert jnp.allclose(data.lapse, 1.0)
    assert jnp.allclose(data.shift, 0.0)
    assert jnp.allclose(data.spatial_metric, jnp.eye(3))
    assert jnp.allclose(data.extrinsic_curvature, 0.0)
    assert bool(data.status.physically_valid)

    constraints = adm_constraint_diagnostics(field, points, tolerance=1.0e-12)
    assert bool(constraints.converged)
    assert jnp.allclose(constraints.hamiltonian, 0.0)
    assert jnp.allclose(constraints.momentum, 0.0)

    geometry = data.as_grid_geometry(
        chart_id="cartesian",
        convention_id="mostly-plus-adm",
        scale_id="geometric-units",
        topology_id="two-point-test",
    )
    assert isinstance(geometry, ADMGridGeometry)
    assert bool(geometry.all_active_valid)
    assert jnp.allclose(geometry.inverse_spatial_metric, jnp.eye(3))


def test_isotropic_schwarzschild_single_hole_limit_and_adm_mass():
    mass = 1.2
    field = IsotropicSchwarzschildInitialData(mass)
    point = jnp.asarray((4.0, 0.0, 0.0))
    data = field(point)
    conformal = 1.0 + mass / 8.0

    assert jnp.allclose(data.conformal_factor, conformal)
    assert jnp.allclose(data.spatial_metric, conformal**4 * jnp.eye(3))
    assert jnp.allclose(data.extrinsic_curvature, 0.0)
    assert bool(data.status.physically_valid)

    constraints = adm_constraint_diagnostics(field, point, tolerance=2.0e-5)
    assert bool(constraints.converged)

    radius = 100.0
    normals = jnp.asarray(
        (
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        )
    )
    weights = jnp.full((6,), 4.0 * jnp.pi * radius**2 / 6.0)
    charges = adm_charge_diagnostics(field, radius * normals, normals, weights)
    assert bool(charges.physically_valid)
    assert jnp.allclose(charges.mass, mass, rtol=2.0e-2)
    assert jnp.allclose(charges.linear_momentum, 0.0, atol=1.0e-10)


def test_nonspinning_kerr_schild_reduces_to_schwarzschild_kerr_schild():
    mass = 0.8
    radius = 3.0
    data = KerrSchildInitialData(mass)(jnp.asarray((radius, 0.0, 0.0)))
    factor = 2.0 * mass / radius

    assert jnp.allclose(data.lapse, 1.0 / jnp.sqrt(1.0 + factor))
    assert jnp.allclose(data.shift, jnp.asarray((factor / (1.0 + factor), 0.0, 0.0)))
    assert jnp.allclose(
        data.spatial_metric, jnp.diag(jnp.asarray((1.0 + factor, 1.0, 1.0)))
    )
    assert bool(data.status.physically_valid)
    assert jnp.all(jnp.isfinite(data.extrinsic_curvature))


def test_brill_lindquist_and_bowen_york_are_exchange_symmetric():
    pair = _pair(momentum=0.04)
    points = jnp.asarray(((0.0, 0.4, 0.2), (1.3, -0.5, 0.7)))

    brill = BrillLindquistInitialData(pair)(points)
    brill_exchanged = BrillLindquistInitialData(pair[::-1])(points)
    bowen = BowenYorkInitialData(pair)(points)
    bowen_exchanged = BowenYorkInitialData(pair[::-1])(points)

    assert jnp.allclose(brill.spatial_metric, brill_exchanged.spatial_metric)
    assert jnp.allclose(brill.extrinsic_curvature, 0.0)
    assert jnp.allclose(bowen.spatial_metric, bowen_exchanged.spatial_metric)
    assert jnp.allclose(bowen.extrinsic_curvature, bowen_exchanged.extrinsic_curvature)
    assert jnp.max(jnp.abs(bowen.extrinsic_curvature)) > 0.0
    assert bool(bowen.status.physically_valid)

    single = Puncture(1.0, (0.0, 0.0, 0.0))
    location = jnp.asarray((2.0, 0.0, 0.0))
    single_brill = BrillLindquistInitialData((single,))(location)
    schwarzschild = IsotropicSchwarzschildInitialData(1.0)(location)
    assert jnp.allclose(single_brill.conformal_factor, schwarzschild.conformal_factor)
    assert jnp.allclose(single_brill.spatial_metric, schwarzschild.spatial_metric)


def test_matrix_free_two_puncture_newton_krylov_reduces_residual_and_restarts():
    plan = TwoPunctureHamiltonianPlan(
        _pair(momentum=0.025),
        resolution=4,
        half_extent=3.0,
        nonlinear_tolerance=2.0e-7,
        linear_tolerance=1.0e-5,
        maximum_linear_steps=64,
        mass_tolerance=0.1,
    )
    result = plan.solve()

    assert plan.matrix_free
    assert plan.operator_storage_bytes < plan.degrees_of_freedom**2 * 8
    assert bool(result.status.finite)
    assert bool(result.status.converged)
    assert bool(result.status.physically_valid)
    assert float(result.final_residual_norm) < float(result.initial_residual_norm)
    assert jnp.all(result.conformal_factor > 0.0)
    assert result.content_id != result.restart.restart_id
    assert result.restart.plan_id == plan.plan_id
    assert bool(result.tuning.finite)
    assert jnp.allclose(result.tuning.achieved_linear_momentum, 0.0, atol=1.0e-12)

    restarted = plan.solve(result.restart)
    assert bool(restarted.status.converged)
    assert float(restarted.final_residual_norm) <= float(result.final_residual_norm) * 1.1
    assert jnp.allclose(restarted.correction, result.correction, rtol=1.0e-5, atol=1.0e-7)
