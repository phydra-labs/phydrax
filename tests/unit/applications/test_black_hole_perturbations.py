#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax


jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from phydrax.applications.compact_objects._perturbation import (
    PerturbationConvention,
    RadialBoundaryCondition,
    SeparatedMode,
)
from phydrax.applications.compact_objects._radial_perturbation import (
    evaluate_kerr_teukolsky_radial,
    evaluate_schwarzschild_radial,
    kerr_teukolsky_radial_coefficients,
    KerrTeukolskyRadialPlan,
    schwarzschild_regge_wheeler_potential,
    schwarzschild_zerilli_potential,
    SchwarzschildRadialPlan,
)
from phydrax.applications.compact_objects._spheroidal import (
    solve_spheroidal_angular,
    spheroidal_angular_matrix,
    spheroidal_angular_residual,
    SpheroidalAngularPlan,
)


def _mode(
    spin_weight=0,
    ell=2,
    m=1,
    *,
    sector="teukolsky",
    family="qnm",
    background_id="black-hole:test",
):
    return SeparatedMode(
        spin_weight,
        ell,
        m,
        sector=sector,
        family=family,
        background_id=background_id,
    )


def test_separated_mode_and_convention_are_content_addressed():
    convention = PerturbationConvention()
    first = _mode()
    repeated = _mode()
    scattering = _mode(family="scattering")
    boundary = RadialBoundaryCondition(convention_id=convention.convention_id)

    assert convention.fourier_phase == "exp(-i*omega*t+i*m*phi)"
    assert first.convention_id == convention.convention_id
    assert first.mode_id == repeated.mode_id
    assert first.mode_id != scattering.mode_id
    assert boundary.horizon == "ingoing"
    assert boundary.infinity == "outgoing"


def test_spheroidal_spherical_limit_and_projected_cosine_square():
    mode = _mode(spin_weight=-2, ell=2, m=2)
    plan = SpheroidalAngularPlan(mode, 5)
    matrix = spheroidal_angular_matrix(plan, 0.0)
    result = solve_spheroidal_angular(plan, 0.0)
    compiled = jax.jit(lambda c: solve_spheroidal_angular(plan, c))(jnp.asarray(0.1))

    np.testing.assert_allclose(matrix, jnp.diag(plan.spherical_separation))
    np.testing.assert_allclose(result.separation_constant, 4.0)
    np.testing.assert_allclose(result.coefficients[plan.target_index], 1.0)
    np.testing.assert_allclose(jnp.sum(jnp.abs(result.coefficients) ** 2), 1.0)
    np.testing.assert_allclose(
        spheroidal_angular_residual(plan, 0.0, result.separation_constant),
        0.0,
        atol=1.0e-6,
    )
    truncated_square = plan.cosine_matrix @ plan.cosine_matrix
    assert plan.cosine_squared_matrix[-1, -1] > truncated_square[-1, -1]
    assert bool(result.qualified)
    assert bool(compiled.finite)
    assert compiled.coefficients.shape == result.coefficients.shape


def test_schwarzschild_axial_and_polar_potentials_have_correct_structure():
    radius = jnp.asarray((2.0 + 1.0e-5, 10.0, 1.0e5))
    axial = schwarzschild_regge_wheeler_potential(
        radius,
        1.0,
        2,
        spin_weight=-2,
    )
    polar = schwarzschild_zerilli_potential(radius, 1.0, 2)

    np.testing.assert_allclose(axial[1], 0.8 * (6.0 / 100.0 - 6.0 / 1000.0))
    assert axial[0] > 0.0
    assert polar[0] > 0.0
    assert not np.isclose(float(axial[1]), float(polar[1]))
    np.testing.assert_allclose(radius[-1] ** 2 * axial[-1], 6.0, rtol=5.0e-5)
    np.testing.assert_allclose(radius[-1] ** 2 * polar[-1], 6.0, rtol=5.0e-5)


def test_kerr_teukolsky_scalar_and_spin_coefficients():
    scalar_mode = _mode(spin_weight=0, ell=2, m=1, sector="teukolsky")
    scalar_plan = KerrTeukolskyRadialPlan(
        scalar_mode,
        1.0,
        0.3,
        node_count=17,
        outer_radius=80.0,
    )
    omega = jnp.asarray(0.2 + 0.0j)
    angular = jnp.asarray(6.0 + 0.0j)
    scalar = kerr_teukolsky_radial_coefficients(
        scalar_plan,
        jnp.asarray(3.0),
        omega,
        angular,
    )
    delta = 3.0**2 - 2.0 * 3.0 + 0.3**2
    k = (3.0**2 + 0.3**2) * omega - 0.3
    separation_lambda = angular + (0.3 * omega) ** 2 - 2.0 * 0.3 * omega

    np.testing.assert_allclose(scalar.delta, delta)
    np.testing.assert_allclose(scalar.k, k)
    np.testing.assert_allclose(scalar.first_derivative, 4.0)
    np.testing.assert_allclose(scalar.separation_lambda, separation_lambda)
    np.testing.assert_allclose(
        scalar.zeroth_derivative,
        k**2 / delta - separation_lambda,
    )

    spin_mode = _mode(spin_weight=-2, ell=2, m=2, sector="teukolsky")
    spin_plan = KerrTeukolskyRadialPlan(
        spin_mode,
        1.0,
        0.3,
        node_count=17,
        outer_radius=80.0,
    )
    spin = kerr_teukolsky_radial_coefficients(
        spin_plan,
        jnp.asarray(3.0),
        omega,
        jnp.asarray(4.0 + 0.0j),
    )
    np.testing.assert_allclose(spin.first_derivative, -4.0)
    assert not np.isclose(float(jnp.imag(spin.zeroth_derivative)), 0.0)
    assert bool(spin.finite & spin.domain_valid)


def test_radial_results_have_fixed_shape_and_independent_evidence():
    schwarzschild_mode = _mode(
        spin_weight=-2,
        ell=2,
        m=2,
        sector="axial",
        background_id="schwarzschild:test",
    )
    schwarzschild = SchwarzschildRadialPlan(
        schwarzschild_mode,
        1.0,
        node_count=65,
        outer_radius=30.0,
        residual_tolerance=1.0e-5,
        matching_tolerance=1.0e-7,
        asymptotic_tolerance=5.0e-2,
        maximum_dimension=96,
        integration_substeps=32,
        infinity_asymptotic_order=12,
    )
    frequency = jnp.asarray(0.373671684418042 - 0.088962315688936j)
    schwarzschild_result = jax.jit(
        lambda omega: evaluate_schwarzschild_radial(
            schwarzschild,
            omega,
            jnp.asarray(4.0 + 0.0j),
        )
    )(frequency)

    assert schwarzschild_result.solution.shape == (65,)
    assert schwarzschild_result.residual_evidence.differential_residual.shape == (65,)
    assert schwarzschild_result.residual.shape == ()
    assert schwarzschild_result.residual_evidence.node_count == 65
    assert abs(schwarzschild_result.residual) <= 1.0e-7
    assert schwarzschild_result.residual_evidence.relative_residual <= 1.0e-5
    assert bool(schwarzschild_result.finite)
    assert bool(schwarzschild_result.converged)
    assert bool(schwarzschild_result.physically_valid)
    assert bool(schwarzschild_result.asymptotic.qualified)
    assert bool(schwarzschild_result.qualified)

    polar_mode = _mode(
        spin_weight=-2,
        ell=2,
        m=2,
        sector="polar",
        background_id="schwarzschild:test",
    )
    polar = SchwarzschildRadialPlan(
        polar_mode,
        1.0,
        node_count=65,
        outer_radius=30.0,
        residual_tolerance=1.0e-5,
        matching_tolerance=1.0e-7,
        asymptotic_tolerance=5.0e-2,
        maximum_dimension=96,
        integration_substeps=32,
        infinity_asymptotic_order=12,
    )
    polar_result = evaluate_schwarzschild_radial(
        polar,
        frequency,
        jnp.asarray(4.0 + 0.0j),
    )

    assert abs(polar_result.residual) <= 1.0e-7
    assert polar_result.residual_evidence.relative_residual <= 1.0e-5
    assert bool(polar_result.qualified)

    kerr_mode = _mode(
        spin_weight=0,
        ell=0,
        m=0,
        sector="scalar",
        background_id="kerr:test",
    )
    kerr = KerrTeukolskyRadialPlan(
        kerr_mode,
        1.0,
        0.2,
        node_count=17,
        outer_radius=80.0,
        asymptotic_tolerance=0.2,
    )
    kerr_result = jax.jit(
        lambda omega, angular: evaluate_kerr_teukolsky_radial(
            kerr,
            omega,
            angular,
        )
    )(jnp.asarray(0.4 - 0.05j), jnp.asarray(0.0 + 0.0j))

    assert kerr_result.solution.shape == (17,)
    assert kerr_result.coefficients.delta.shape == (17,)
    assert kerr_result.separation_lambda.shape == ()
    assert bool(kerr_result.finite)
    assert bool(kerr_result.physically_valid)
    assert kerr_result.plan_id == kerr.plan_id
