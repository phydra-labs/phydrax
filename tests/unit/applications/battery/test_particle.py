#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications.battery._particle import BatteryParticlePlan


def _amounts(prepared, concentration, *, radius=2.0e-6, multiplicity=7.0):
    volumes = prepared.shell_measures(jnp.asarray(radius))
    return jnp.asarray(concentration) * multiplicity * volumes


def test_center_boundary_surface_reconstruction_and_electrode_scaling():
    prepared = BatteryParticlePlan(4, particle_id="test").prepare()
    radius = 2.0e-6
    multiplicity = 7.0
    support_volume = 3.0e-15
    diffusivity = 4.0e-14
    outward_flux = 2.0e-6
    concentration = 1.5e4
    result = prepared.evaluate(
        _amounts(prepared, concentration, radius=radius, multiplicity=multiplicity),
        particle_radius_m=radius,
        particle_multiplicity=multiplicity,
        support_volume_m3=support_volume,
        diffusivity_m2_s=diffusivity,
        outward_molar_flux_mol_m2_s=outward_flux,
    )

    surface_distance = radius * (1.0 - prepared.transport.mesh.reference_cells[-1])
    particle_area = 4.0 * np.pi * radius**2
    np.testing.assert_allclose(result.face_molar_flux_mol_m2_s[0], 0.0)
    np.testing.assert_allclose(result.face_molar_flux_mol_m2_s[-1], outward_flux)
    np.testing.assert_allclose(result.center_concentration_mol_m3, concentration)
    np.testing.assert_allclose(
        result.surface_concentration_mol_m3,
        concentration - outward_flux * surface_distance / diffusivity,
    )
    np.testing.assert_allclose(result.average_concentration_mol_m3, concentration)
    np.testing.assert_allclose(
        result.active_surface_area_m2, multiplicity * particle_area
    )
    np.testing.assert_allclose(
        result.specific_surface_area_m2_m3,
        multiplicity * particle_area / support_volume,
    )
    np.testing.assert_allclose(
        jnp.sum(result.amount_rate_mol_s),
        -multiplicity * particle_area * outward_flux,
    )
    np.testing.assert_allclose(
        result.outer_amount_rate_mol_s,
        multiplicity * particle_area * outward_flux,
    )
    np.testing.assert_allclose(result.conservation_residual_mol_s, 0.0, atol=1.0e-20)
    assert bool(result.domain_valid)


def test_quadratic_manufactured_solution_has_exact_spherical_diffusion_rate():
    shell_count = 8
    prepared = BatteryParticlePlan(shell_count, particle_id="manufactured").prepare()
    radius = 3.0e-6
    multiplicity = 11.0
    diffusivity = 2.0e-14
    base = 2.0e4
    curvature = 1.0e14
    metrics = prepared.transport.mesh.metrics(jnp.asarray((radius,)))
    centers = metrics.cell_coordinates[0]
    concentration = base + curvature * centers**2
    amounts = concentration * metrics.cell_measures[0] * multiplicity
    outward_flux = -2.0 * diffusivity * curvature * radius

    result = prepared.evaluate(
        amounts,
        particle_radius_m=radius,
        particle_multiplicity=multiplicity,
        support_volume_m3=1.0e-15,
        diffusivity_m2_s=diffusivity,
        outward_molar_flux_mol_m2_s=outward_flux,
    )

    expected_rate = (
        6.0 * diffusivity * curvature * metrics.cell_measures[0] * multiplicity
    )
    expected_face_flux = -2.0 * diffusivity * curvature * metrics.face_coordinates[0]
    np.testing.assert_allclose(result.face_molar_flux_mol_m2_s, expected_face_flux)
    np.testing.assert_allclose(result.amount_rate_mol_s, expected_rate, rtol=1.0e-6)
    np.testing.assert_allclose(result.conservation_residual_mol_s, 0.0, atol=1.0e-20)


def test_zero_boundary_flux_relaxes_concentration_without_changing_total_amount():
    prepared = BatteryParticlePlan(4, particle_id="relaxation").prepare()
    radius = 2.0e-6
    multiplicity = 5.0
    concentration = jnp.asarray((2.0e4, 1.6e4, 1.1e4, 7.0e3))
    result = prepared.evaluate(
        _amounts(prepared, concentration, radius=radius, multiplicity=multiplicity),
        particle_radius_m=radius,
        particle_multiplicity=multiplicity,
        support_volume_m3=1.0e-15,
        diffusivity_m2_s=1.0e-14,
        outward_molar_flux_mol_m2_s=0.0,
    )

    np.testing.assert_allclose(jnp.sum(result.amount_rate_mol_s), 0.0, atol=1.0e-20)
    assert float(jnp.sum(concentration * result.amount_rate_mol_s)) < 0.0
    assert bool(result.domain_valid)


def test_shell_diffusivity_uses_harmonic_face_values_and_surface_cell_property():
    prepared = BatteryParticlePlan(3, particle_id="variable-diffusivity").prepare()
    radius = 3.0e-6
    multiplicity = 2.0
    concentration = jnp.asarray((4.0e3, 2.0e3, 1.0e3))
    diffusivity = jnp.asarray((1.0e-14, 2.0e-14, 4.0e-14))
    outward_flux = 3.0e-7
    result = prepared.evaluate(
        _amounts(
            prepared,
            concentration,
            radius=radius,
            multiplicity=multiplicity,
        ),
        particle_radius_m=radius,
        particle_multiplicity=multiplicity,
        support_volume_m3=1.0e-15,
        diffusivity_m2_s=diffusivity,
        outward_molar_flux_mol_m2_s=outward_flux,
    )

    metrics = prepared.transport.mesh.metrics(jnp.asarray((radius,)))
    harmonic = (
        2.0 * diffusivity[:-1] * diffusivity[1:] / (diffusivity[:-1] + diffusivity[1:])
    )
    expected_interior_flux = (
        harmonic * (concentration[:-1] - concentration[1:]) / metrics.center_distances[0]
    )
    surface_distance = radius * (1.0 - prepared.transport.mesh.reference_cells[-1])
    np.testing.assert_allclose(
        result.face_molar_flux_mol_m2_s[1:-1],
        expected_interior_flux,
    )
    np.testing.assert_allclose(
        result.surface_concentration_mol_m3,
        concentration[-1] - outward_flux * surface_distance / diffusivity[-1],
    )
    assert bool(result.domain_valid)


def test_particle_transport_is_jittable_vmappable_and_differentiable():
    prepared = BatteryParticlePlan(3, particle_id="transforms").prepare()
    radius = 2.0e-6
    multiplicity = 4.0
    amounts = _amounts(
        prepared,
        jnp.asarray((1.0e4, 1.1e4, 1.2e4)),
        radius=radius,
        multiplicity=multiplicity,
    )

    def evaluate(state, flux):
        return prepared.evaluate(
            state,
            particle_radius_m=radius,
            particle_multiplicity=multiplicity,
            support_volume_m3=1.0e-15,
            diffusivity_m2_s=2.0e-14,
            outward_molar_flux_mol_m2_s=flux,
        )

    eager = evaluate(amounts, jnp.asarray(1.0e-7))
    compiled = jax.jit(evaluate)(amounts, jnp.asarray(1.0e-7))
    mapped = jax.vmap(evaluate)(jnp.stack((amounts, amounts)), jnp.asarray((0.0, 1.0e-7)))
    derivative = jax.grad(
        lambda flux: evaluate(amounts, flux).surface_concentration_mol_m3
    )(jnp.asarray(0.0))

    np.testing.assert_allclose(compiled.amount_rate_mol_s, eager.amount_rate_mol_s)
    assert mapped.amount_rate_mol_s.shape == (2, 3)
    np.testing.assert_allclose(
        derivative,
        -radius * (1.0 - prepared.transport.mesh.reference_cells[-1]) / 2.0e-14,
    )
    assert bool(jnp.all(mapped.domain_valid))


def test_particle_domain_rejects_negative_amount_and_nonpositive_diffusivity():
    prepared = BatteryParticlePlan(2, particle_id="domain").prepare()
    amounts = prepared.initial_amounts(1.0e4, 1.0e-6, 3.0)
    negative = amounts.at[0].set(-1.0)
    invalid_state = prepared.evaluate(
        negative,
        particle_radius_m=1.0e-6,
        particle_multiplicity=3.0,
        support_volume_m3=1.0e-15,
        diffusivity_m2_s=1.0e-14,
        outward_molar_flux_mol_m2_s=0.0,
    )
    invalid_property = prepared.evaluate(
        amounts,
        particle_radius_m=1.0e-6,
        particle_multiplicity=3.0,
        support_volume_m3=1.0e-15,
        diffusivity_m2_s=0.0,
        outward_molar_flux_mol_m2_s=0.0,
    )

    assert not bool(invalid_state.domain_valid)
    assert not bool(invalid_property.domain_valid)
