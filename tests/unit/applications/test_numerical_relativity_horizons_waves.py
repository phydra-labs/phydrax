#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications.numerical_relativity._horizon_tracking import (
    ApparentHorizonSearchPlan,
    ApparentHorizonSearchStatus,
    kerr_horizon_reference,
    quasilocal_horizon_geometry,
)
from phydrax.applications.numerical_relativity._mots import (
    MOTSSolvePlan,
    null_expansions,
    schwarzschild_isotropic_outgoing_expansion,
)
from phydrax.applications.numerical_relativity._surfaces import (
    schwarzschild_isotropic_spatial_metric,
    SphericalSurfacePlan,
)
from phydrax.applications.numerical_relativity._wave_extraction import (
    FiniteRadiusExtrapolationPlan,
    FixedFrequencyStrainPlan,
    Psi4ExtractionPlan,
    Psi4TetradConvention,
    SpinWeightedMultipolePlan,
    vacuum_weyl_curvature,
)


def test_schwarzschild_and_kerr_surface_geometry_matches_analytic_values():
    surface_plan = SphericalSurfacePlan(3)
    mass = 1.0
    surface = surface_plan.constant(mass / 2.0)
    coordinate_geometry = surface_plan.geometry(surface)
    spatial_metric = schwarzschild_isotropic_spatial_metric(
        coordinate_geometry.points, mass
    )
    geometry = surface_plan.geometry(surface, spatial_metric)
    zero_curvature = jnp.zeros(surface_plan.sample_shape + (3, 3))
    expansion = schwarzschild_isotropic_outgoing_expansion(geometry.radius, mass)
    null = null_expansions(
        spatial_metric,
        zero_curvature,
        geometry.outward_normal,
        expansion,
    )
    axial_vector = geometry.phi_tangent
    quantities = quasilocal_horizon_geometry(geometry, zero_curvature, axial_vector)

    np.testing.assert_allclose(geometry.area, 16.0 * np.pi * mass**2, rtol=1e-11)
    np.testing.assert_allclose(null.outgoing, 0.0, atol=1e-12)
    np.testing.assert_allclose(quantities.irreducible_mass, mass, rtol=1e-11)
    np.testing.assert_allclose(quantities.christodoulou_mass, mass, rtol=1e-11)
    np.testing.assert_allclose(quantities.dimensionless_spin, 0.0, atol=1e-12)
    assert bool(geometry.physically_valid)
    assert bool(null.qualified)
    assert bool(quantities.qualified)

    kerr = kerr_horizon_reference(2.0, 1.0)
    expected_radius = 2.0 + np.sqrt(3.0)
    np.testing.assert_allclose(kerr.horizon_radius, expected_radius)
    np.testing.assert_allclose(kerr.area, 16.0 * np.pi * expected_radius)
    np.testing.assert_allclose(kerr.christodoulou_mass, 2.0)
    np.testing.assert_allclose(kerr.dimensionless_spin, 0.5)
    assert bool(kerr.physically_valid)


def test_mots_is_not_promoted_without_complete_outermost_search_evidence():
    surface_plan = SphericalSurfacePlan(3)
    mots_plan = MOTSSolvePlan(surface_plan, residual_tolerance=1.0e-8, maximum_steps=20)

    def two_surface_expansion(surface):
        radius = surface_plan.radius(surface)
        return (radius - 1.0) * (radius - 2.0)

    solved = jax.jit(lambda seed: mots_plan.solve(seed, two_surface_expansion))(
        surface_plan.constant(1.8)
    )
    assert solved.surface.coefficients.shape == surface_plan.coefficient_shape
    assert solved.outgoing_expansion.shape == surface_plan.sample_shape
    assert bool(solved.converged)
    assert bool(solved.qualified)
    assert bool(solved.stability.stable)
    assert bool(solved.stability.derivative_valid)
    assert not bool(solved.derivative_valid)
    np.testing.assert_allclose(surface_plan.mean_radius(solved.surface), 2.0, atol=1e-7)

    search = ApparentHorizonSearchPlan(mots_plan, (0.8, 1.2, 2.2))
    incomplete = search.search(two_surface_expansion)
    assert not bool(incomplete.certified)
    assert int(incomplete.status) == int(ApparentHorizonSearchStatus.INCOMPLETE)

    multiple = search.search(two_surface_expansion, search_complete=True)
    assert int(multiple.search.found_count) == 2
    assert int(multiple.status) == int(ApparentHorizonSearchStatus.MULTIPLE_SURFACES)
    assert bool(multiple.certified)
    np.testing.assert_allclose(surface_plan.mean_radius(multiple.surface), 2.0, atol=1e-7)

    def no_physical_surface(surface):
        return surface_plan.radius(surface) + 1.0

    no_surface = search.search(
        no_physical_surface,
        excluded=jnp.ones((search.candidate_capacity,), dtype=bool),
        search_complete=True,
    )
    assert int(no_surface.search.found_count) == 0
    assert bool(no_surface.search.no_surface_certified)
    assert not bool(no_surface.certified)
    assert int(no_surface.status) == int(ApparentHorizonSearchStatus.NO_SURFACE)


def test_psi4_respects_explicit_sign_and_spin_frame_conventions():
    surface_plan = SphericalSurfacePlan(3)
    multipole_plan = SpinWeightedMultipolePlan(3)
    sample_shape = multipole_plan.transform.sample_shape
    metric = jnp.broadcast_to(jnp.eye(3), sample_shape + (3, 3))
    ricci = jnp.broadcast_to(
        jnp.diag(jnp.asarray((0.0, 1.0, -1.0))), sample_shape + (3, 3)
    )
    zero_curvature = jnp.zeros(sample_shape + (3, 3))
    zero_derivative = jnp.zeros(sample_shape + (3, 3, 3))
    weyl = vacuum_weyl_curvature(metric, ricci, zero_curvature, zero_derivative)
    canonical = Psi4ExtractionPlan(multipole_plan).extract(
        weyl,
        surface_plan.unit_radial,
        surface_plan.unit_theta,
        surface_plan.unit_phi,
    )
    sign_reversed = Psi4ExtractionPlan(
        multipole_plan, Psi4TetradConvention(psi4_sign=-1)
    ).extract(
        weyl,
        surface_plan.unit_radial,
        surface_plan.unit_theta,
        surface_plan.unit_phi,
    )
    quarter_turn = Psi4ExtractionPlan(multipole_plan).extract(
        weyl,
        surface_plan.unit_radial,
        surface_plan.unit_phi,
        -surface_plan.unit_theta,
    )

    np.testing.assert_allclose(sign_reversed.psi4, -canonical.psi4, atol=1e-12)
    np.testing.assert_allclose(quarter_turn.psi4, -canonical.psi4, atol=1e-12)
    assert bool(canonical.physically_valid)
    assert bool(canonical.multipoles.converged)
    assert bool(canonical.qualified)
    assert bool(sign_reversed.qualified)
    assert bool(quarter_turn.qualified)


def test_multipole_strain_and_finite_radius_evidence_converges_at_fixed_shapes():
    multipole_plan = SpinWeightedMultipolePlan(4)
    coefficients = (
        jnp.zeros(multipole_plan.transform.coefficient_shape, dtype=jnp.complex128)
        .at[2, 2 + multipole_plan.bandlimit - 1]
        .set(1.0 + 0.25j)
    )
    samples = multipole_plan.synthesize(coefficients)
    multipoles = jax.jit(lambda values: multipole_plan.analyze(values))(samples)
    assert multipoles.coefficients.shape == multipole_plan.transform.coefficient_shape
    assert bool(multipoles.converged)
    assert bool(multipoles.derivative_valid)
    np.testing.assert_allclose(multipoles.reconstruction, samples, atol=1e-11)

    count = 64
    interval = 0.1
    angular_frequency = 2.0 * np.pi * 8.0 / (count * interval)
    times = jnp.arange(count) * interval
    expected_strain = jnp.exp(1j * angular_frequency * times)
    psi4 = -(angular_frequency**2) * expected_strain
    strain_plan = FixedFrequencyStrainPlan(
        count,
        interval,
        angular_frequency / 2.0,
        reconstruction_tolerance=1.0e-10,
    )
    strain = jax.jit(lambda values: strain_plan.integrate(values))(psi4)
    assert strain.strain.shape == (count,)
    assert bool(strain.converged)
    assert bool(strain.derivative_valid)
    np.testing.assert_allclose(strain.strain, expected_strain, atol=1e-11)

    radii = jnp.asarray((50.0, 60.0, 75.0, 90.0))
    asymptotic = jnp.exp(0.15j * times)
    scaled = asymptotic[None, :] + 0.2 / radii[:, None] + 1.0e-3 / radii[:, None] ** 2
    finite_radius = scaled / radii[:, None]
    extrapolation_plan = FiniteRadiusExtrapolationPlan(
        radii, 2, convergence_tolerance=1.0e-4
    )
    extrapolated = jax.jit(lambda values: extrapolation_plan.extrapolate(values))(
        finite_radius
    )
    assert extrapolated.radial_coefficients.shape == (3, count)
    assert bool(extrapolated.fit_convergence.converged)
    assert bool(extrapolated.order_convergence.converged)
    assert bool(extrapolated.qualified)
    np.testing.assert_allclose(extrapolated.asymptotic_waveform, asymptotic, atol=1e-10)
