#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def _prepared():
    qcd = phx.applications.lattice_field
    convention = qcd.ChemicalChargeConvention(energy_unit=phx.units.GIGAELECTRONVOLT)
    domain = qcd.FiniteDensityDomain(
        (0.15, 0.25),
        ((-1.0, 1.0), (-0.5, 0.5), (-0.5, 0.5)),
        maximum_total_order=2,
    )
    indices = (
        qcd.GeneralizedSusceptibilityIndex(0, 0, 0),
        qcd.GeneralizedSusceptibilityIndex(2, 0, 0),
        qcd.GeneralizedSusceptibilityIndex(0, 2, 0),
        qcd.GeneralizedSusceptibilityIndex(0, 0, 2),
    )
    values = jnp.asarray(
        [[1.0, 0.2, 0.3, 0.4], [1.0, 0.2, 0.3, 0.4], [1.0, 0.2, 0.3, 0.4]]
    )
    estimate = qcd.SusceptibilityEstimate(
        jnp.asarray([0.15, 0.20, 0.25]),
        values,
        1.0e-8 * jnp.eye(values.size),
        indices=indices,
        convention=convention,
        domain=domain,
        source_kind=qcd.FiniteDensitySourceKind.CONTINUUM_EXTRAPOLATED,
        provenance_ids=("synthetic-continuum",),
    )
    return qcd.prepare_taylor_eos(estimate), convention, domain


def test_taylor_eos_fields_derive_from_one_pressure_potential():
    qcd = phx.applications.lattice_field
    prepared, _, _ = _prepared()
    temperature = 0.2
    mu_b = 0.04
    x = mu_b / temperature
    result = qcd.evaluate_taylor_eos(prepared, temperature, jnp.asarray([mu_b, 0.0, 0.0]))
    expected_pressure = 1.0 + 0.5 * 0.2 * x**2
    assert bool(result.successful)
    assert jnp.isclose(result.pressure_over_temperature4, expected_pressure)
    assert jnp.isclose(result.densities_over_temperature3[0], 0.2 * x)
    assert jnp.isclose(result.susceptibility_matrix[0, 0], 0.2)
    assert jnp.isclose(result.energy_over_temperature4, 3.0 * expected_pressure)
    assert jnp.abs(result.thermodynamic_identity_residual) < 1.0e-12


def test_heavy_ion_constraints_solve_declared_charge_ratio():
    qcd = phx.applications.lattice_field
    prepared, _, _ = _prepared()
    result = qcd.solve_heavy_ion_path(
        prepared,
        qcd.HeavyIonConstraintPlan(0.0, maximum_iterations=8),
        0.2,
        0.03,
        initial_charge_strangeness=jnp.asarray([0.01, -0.01]),
    )
    assert bool(result.converged)
    assert jnp.linalg.norm(result.residual) < 1.0e-10
    assert jnp.allclose(result.chemical_potentials[1:], 0.0, atol=1.0e-10)


def test_taylor_table_refuses_extrapolation_and_passes_stability_checks():
    qcd = phx.applications.lattice_field
    prepared, convention, domain = _prepared()
    table = qcd.build_taylor_eos_table(
        prepared,
        qcd.EOSGridPlan(
            jnp.asarray([0.16, 0.20, 0.24]),
            jnp.asarray([-0.04, 0.0, 0.04]),
            convention=convention,
            domain=domain,
        ),
    )
    qualification = qcd.qualify_eos_table(table)
    assert bool(qualification.passed)
    inside = qcd.evaluate_eos_table(table, 0.2, 0.0)
    outside = qcd.evaluate_eos_table(table, 0.3, 0.0)
    assert bool(inside.valid)
    assert not bool(outside.valid)
    assert jnp.isnan(outside.pressure_over_temperature4)
    metadata = qcd.eos_table_metadata(table, qualification)
    assert (
        metadata["source_kind"]
        == qcd.FiniteDensitySourceKind.CONTINUUM_EXTRAPOLATED.value
    )


def test_multi_charge_canonical_transform_reports_finite_support():
    qcd = phx.applications.lattice_field
    _, convention, _ = _prepared()
    plan = qcd.MultiChargeCanonicalPlan(
        convention,
        (5, 5, 5),
        (1, 1, 1),
        periodicities=(2.0 * jnp.pi,) * 3,
        volume=1.0,
    )
    sectors = jnp.zeros((5, 5, 5), dtype=jnp.complex128).at[2, 2, 2].set(1.0)
    imaginary_mu = jnp.fft.ifftn(jnp.fft.ifftshift(sectors) * sectors.size)
    result = qcd.canonical_sector_transform(plan, imaginary_mu)
    assert bool(result.qualified)
    assert result.reconstruction_residual < 1.0e-12
