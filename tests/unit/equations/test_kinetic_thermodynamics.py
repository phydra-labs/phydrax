#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.discretization.lattice_boltzmann import (
    D2Q9,
    PreparedBinaryKineticThermodynamics,
)
from phydrax.equations import (
    BinaryPhaseThermodynamicClosure,
    BinaryThermodynamicParameters,
    DoubleWellFreeEnergy,
    ThermodynamicForceRepresentation,
)


def test_binary_closure_derives_energy_chemical_potential_and_symmetric_stress():
    closure = BinaryPhaseThermodynamicClosure()
    parameters = BinaryThermodynamicParameters(0.08, 0.12)
    phase = jnp.linspace(-0.8, 0.8, 12, dtype=jnp.float64).reshape((3, 4))
    gradient = jnp.broadcast_to(jnp.asarray((0.2, -0.1)), phase.shape + (2,))
    laplacian = jnp.full(phase.shape, 0.03)
    fields = closure.evaluate_local(phase, gradient, laplacian, parameters)
    bulk_derivative = jax.grad(
        lambda value: jnp.sum(parameters.bulk_scale * closure.free_energy.density(value))
    )(phase)

    np.testing.assert_allclose(
        fields.chemical_potential + parameters.gradient_coefficient * laplacian,
        bulk_derivative,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        fields.symmetric_stress,
        jnp.swapaxes(fields.symmetric_stress, -1, -2),
        atol=1e-14,
    )
    assert jnp.all(jnp.isfinite(fields.bulk_energy_density))
    assert jnp.all(jnp.isfinite(fields.gradient_energy_density))


def test_binary_closure_characteristics_include_bulk_potential_scale():
    closure = BinaryPhaseThermodynamicClosure(DoubleWellFreeEnergy(2.0))
    parameters = BinaryThermodynamicParameters(0.08, 0.12)
    effective_bulk = 0.16

    np.testing.assert_allclose(
        closure.characteristic_interface_width(parameters),
        np.sqrt(2.0 * 0.12 / effective_bulk),
        rtol=2e-14,
    )
    np.testing.assert_allclose(
        closure.planar_surface_tension(parameters),
        2.0 * np.sqrt(2.0 * effective_bulk * 0.12) / 3.0,
        rtol=2e-14,
    )


def test_periodic_kinetic_thermodynamic_forces_have_zero_net_internal_force():
    lattice = D2Q9()
    closure = BinaryPhaseThermodynamicClosure()
    parameters = BinaryThermodynamicParameters(0.08, 0.12)
    coordinate = jnp.arange(32, dtype=jnp.float64)
    x, y = jnp.meshgrid(coordinate, coordinate, indexing="ij")
    phase = 0.4 * jnp.sin(2.0 * jnp.pi * x / 32.0) * jnp.cos(2.0 * jnp.pi * y / 32.0)
    chemical_plan = PreparedBinaryKineticThermodynamics(
        closure,
        lattice,
        ThermodynamicForceRepresentation.CHEMICAL_POTENTIAL_GRADIENT,
    )
    stress_plan = PreparedBinaryKineticThermodynamics(
        closure,
        lattice,
        ThermodynamicForceRepresentation.STRESS_DIVERGENCE,
    )
    chemical = chemical_plan.evaluate(phase, parameters)
    stress = stress_plan.evaluate(phase, parameters)

    np.testing.assert_allclose(
        jnp.sum(chemical.chemical_force_density, axis=(0, 1)),
        0.0,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        jnp.sum(stress.stress_force_density, axis=(0, 1)),
        0.0,
        atol=2e-13,
    )
    np.testing.assert_array_equal(
        chemical.selected_force_density,
        chemical.chemical_force_density,
    )
    np.testing.assert_array_equal(
        stress.selected_force_density,
        stress.stress_force_density,
    )
    assert jnp.all(jnp.isfinite(chemical.force_representation_residual))
    assert jnp.max(jnp.abs(chemical.force_representation_residual)) < 4.0e-5
    assert chemical_plan.prepared_id != stress_plan.prepared_id
