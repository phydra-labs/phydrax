import equinox as eqx
import jax
import numpy as np
import pytest

from phydrax._physical import RelativityScaleContract
from phydrax.applications.compact_objects._black_hole_thermodynamics import (
    evaluate_stationary_kerr_horizon,
    KerrInput,
)
from phydrax.applications.compact_objects._extended_thermodynamics import (
    EinsteinWaldEntropyPlan,
    KerrNewmanAdSThermodynamicsPlan,
    KerrNewmanThermodynamicsPlan,
    ReissnerNordstromCavityPlan,
)


jax.config.update("jax_enable_x64", True)


def test_kerr_newman_neutral_limit_matches_stationary_kerr_horizon():
    scale = RelativityScaleContract.si()
    mass = 2.0
    specific_spin = 0.35
    extended = eqx.filter_jit(
        KerrNewmanThermodynamicsPlan(
            mass,
            specific_spin,
            0.0,
            scale,
            ensemble="microcanonical",
        ).evaluate
    )()
    kerr = evaluate_stationary_kerr_horizon(KerrInput(mass, mass * specific_spin))

    np.testing.assert_allclose(extended.horizon_radii[1], kerr.outer_radius)
    np.testing.assert_allclose(extended.horizon_radii[0], kerr.inner_radius)
    np.testing.assert_allclose(extended.horizon_area, kerr.area)
    np.testing.assert_allclose(extended.surface_gravity, kerr.surface_gravity)
    np.testing.assert_allclose(extended.angular_velocity, kerr.angular_velocity)
    np.testing.assert_allclose(extended.electric_potential, 0.0)
    np.testing.assert_allclose(extended.smarr_residual, 0.0, atol=2.0e-15)
    assert extended.ensemble == "microcanonical"
    assert bool(extended.physically_valid)
    assert bool(extended.qualified)
    assert bool(extended.derivative_valid)


def test_kerr_newman_extremal_and_overextremal_branches_do_not_share_status():
    scale = RelativityScaleContract.si()
    extremal = KerrNewmanThermodynamicsPlan(1.0, 0.0, 1.0, scale).evaluate()
    overextremal = KerrNewmanThermodynamicsPlan(1.0, 0.0, 1.01, scale).evaluate()

    assert bool(extremal.finite)
    assert bool(extremal.converged)
    assert bool(extremal.physically_valid)
    assert bool(extremal.qualified)
    assert not bool(extremal.derivative_valid)
    assert int(extremal.branch_status) == 1

    assert not bool(overextremal.finite)
    assert not bool(overextremal.converged)
    assert not bool(overextremal.physically_valid)
    assert not bool(overextremal.qualified)
    assert not bool(overextremal.derivative_valid)
    assert int(overextremal.branch_status) == 2


def test_charged_flat_smarr_and_einstein_wald_entropy_are_explicit():
    scale = RelativityScaleContract.si()
    result = KerrNewmanThermodynamicsPlan(
        2.0,
        0.4,
        0.3,
        scale,
        ensemble="canonical-charge",
    ).evaluate()
    wald = EinsteinWaldEntropyPlan(scale).evaluate(result.horizon_area)

    np.testing.assert_allclose(result.smarr_residual, 0.0, atol=2.0e-15)
    np.testing.assert_allclose(wald.geometric_entropy, result.geometric_entropy)
    np.testing.assert_allclose(wald.entropy, result.entropy)
    assert wald.theory == "four-dimensional Einstein--Hilbert gravity"
    assert "Einstein-Hilbert" in wald.scope
    assert not wald.generalized_entropy_included
    assert not wald.island_prescription_included
    assert bool(wald.qualified)
    assert bool(result.qualified)


def test_ads_extended_variables_satisfy_enthalpy_and_smarr_relations():
    scale = RelativityScaleContract.si()
    result = eqx.filter_jit(
        KerrNewmanAdSThermodynamicsPlan(
            2.0,
            0.3,
            0.2,
            10.0,
            scale,
            ensemble="grand-canonical",
        ).evaluate
    )()

    np.testing.assert_allclose(result.enthalpy_constraint_residual, 0.0, atol=2.0e-14)
    np.testing.assert_allclose(result.temperature_constraint_residual, 0.0, atol=2.0e-14)
    np.testing.assert_allclose(result.smarr_residual, 0.0, atol=2.0e-14)
    assert float(result.pressure) > 0.0
    assert float(result.thermodynamic_volume) > 0.0
    assert result.ensemble == "grand-canonical"
    assert result.asymptotics == "asymptotically anti-de Sitter"
    assert bool(result.physically_valid)
    assert bool(result.qualified)
    assert bool(result.derivative_valid)


def test_ads_observables_recover_asymptotically_flat_kerr_newman_limit():
    scale = RelativityScaleContract.si()
    radius = 3.0
    spin = 0.4
    charge = 0.2
    ads_radius = 1.0e7
    asymptotic_rtol = 5.0 * (radius / ads_radius) ** 2
    geometric_mass = (radius**2 + spin**2 + charge**2) / (2.0 * radius)
    ads = KerrNewmanAdSThermodynamicsPlan(
        radius, spin, charge, ads_radius, scale
    ).evaluate()
    flat = KerrNewmanThermodynamicsPlan(geometric_mass, spin, charge, scale).evaluate()

    np.testing.assert_allclose(
        ads.geometric_mass_enthalpy, geometric_mass, rtol=asymptotic_rtol
    )
    np.testing.assert_allclose(ads.horizon_area, flat.horizon_area, rtol=asymptotic_rtol)
    np.testing.assert_allclose(
        ads.temperature,
        flat.surface_gravity / (2.0 * np.pi),
        rtol=asymptotic_rtol,
    )
    np.testing.assert_allclose(
        ads.angular_velocity, flat.angular_velocity, rtol=asymptotic_rtol
    )
    np.testing.assert_allclose(
        ads.electric_potential, flat.electric_potential, rtol=asymptotic_rtol
    )


def test_finite_cavity_exposes_tolman_first_law_and_heat_capacity_branch():
    scale = RelativityScaleContract.si()
    stable = ReissnerNordstromCavityPlan(
        2.0,
        0.0,
        2.5,
        scale,
        ensemble="canonical-charge",
    ).evaluate()
    unstable = ReissnerNordstromCavityPlan(
        2.0,
        0.0,
        4.0,
        scale,
        ensemble="canonical-charge",
    ).evaluate()

    np.testing.assert_allclose(stable.radial_first_law_residual, 0.0, atol=2.0e-15)
    np.testing.assert_allclose(stable.charge_first_law_residual, 0.0, atol=2.0e-15)
    np.testing.assert_allclose(
        stable.local_temperature,
        1.0 / (4.0 * np.pi * stable.horizon_radius * stable.wall_redshift),
    )
    assert bool(stable.stable_branch)
    assert int(stable.branch_status) == 0
    assert bool(stable.qualified)
    assert not bool(unstable.stable_branch)
    assert int(unstable.branch_status) == 1
    assert bool(unstable.qualified)
    assert stable.boundary_condition.startswith("finite spherical Dirichlet wall")

    with pytest.raises(ValueError, match="canonical-charge ensemble"):
        ReissnerNordstromCavityPlan(
            2.0,
            0.0,
            2.5,
            scale,
            ensemble="grand-canonical",
        )
