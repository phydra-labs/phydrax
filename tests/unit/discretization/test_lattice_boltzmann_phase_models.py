#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe

from phydrax.discretization.lattice_boltzmann._colour_gradient import (
    recolour_populations,
)
from phydrax.discretization.lattice_boltzmann._free_energy import (
    phase_field_equilibrium,
    phase_population_moments,
)
from phydrax.discretization.lattice_boltzmann._interfacial import (
    continuum_surface_force,
    isotropic_gradient,
    natural_wetting_gradient,
    static_contact_angle_normal,
)
from phydrax.discretization.lattice_boltzmann._lattice import D2Q9
from phydrax.discretization.lattice_boltzmann._precision import (
    LatticeBoltzmannPrecisionPolicy,
)
from phydrax.discretization.lattice_boltzmann._thermodynamics import (
    PreparedBinaryKineticThermodynamics,
)
from phydrax.equations import (
    BinaryPhaseThermodynamicClosure,
    BinaryThermodynamicParameters,
    DoubleWellFreeEnergy,
    ThermodynamicForceRepresentation,
)


def test_recolouring_is_label_symmetric_and_conserves_required_moments():
    lattice = D2Q9()
    weights = jnp.asarray(lattice.weights)
    red = jnp.asarray([[0.7, 0.3], [0.4, 0.8]], dtype=jnp.float64)
    blue = 1.0 - red
    total = jnp.broadcast_to(weights, (*red.shape, lattice.population_count))
    normal = jnp.broadcast_to(jnp.asarray((0.6, 0.8), dtype=jnp.float64), (*red.shape, 2))

    split = recolour_populations(total, red, blue, normal, lattice, 0.7)
    swapped = recolour_populations(total, blue, red, -normal, lattice, 0.7)

    np.testing.assert_allclose(
        split.red_populations, swapped.blue_populations, atol=1e-14
    )
    np.testing.assert_allclose(
        split.blue_populations, swapped.red_populations, atol=1e-14
    )
    np.testing.assert_allclose(jnp.sum(split.red_populations, axis=-1), red, atol=1e-14)
    np.testing.assert_allclose(jnp.sum(split.blue_populations, axis=-1), blue, atol=1e-14)
    closure = split.red_populations + split.blue_populations
    np.testing.assert_allclose(closure, total, atol=1e-14)
    np.testing.assert_allclose(
        oe.contract("...q,qd->...d", closure, lattice.velocities),
        oe.contract("...q,qd->...d", total, lattice.velocities),
        atol=1e-14,
    )


def test_csf_circle_has_laplace_curvature_and_label_symmetry():
    lattice = D2Q9()
    count = 96
    coordinates = jnp.arange(count, dtype=jnp.float64) - 0.5 * count
    x, y = jnp.meshgrid(coordinates, coordinates, indexing="ij")
    radius = 24.0
    distance = jnp.sqrt(x**2 + y**2)
    colour = jnp.tanh((radius - distance) / 3.0)

    fields = continuum_surface_force(colour, lattice, 0.04)
    swapped = continuum_surface_force(-colour, lattice, 0.04)
    interface = jnp.abs(distance - radius) < 1.0

    np.testing.assert_allclose(
        jnp.mean(jnp.abs(fields.curvature[interface])), 1.0 / radius, rtol=0.2
    )
    np.testing.assert_allclose(fields.force_density, swapped.force_density, atol=2e-12)
    pressure_jump = 0.04 * jnp.mean(jnp.abs(fields.curvature[interface]))
    np.testing.assert_allclose(pressure_jump, 0.04 / radius, rtol=0.2)


def test_static_and_natural_wetting_primitives_enforce_boundary_contracts():
    interface = jnp.asarray([[[1.0, 0.0]]], dtype=jnp.float64)
    wall = jnp.asarray([[[0.0, 1.0]]], dtype=jnp.float64)
    mask = jnp.asarray([[True]])
    angle = jnp.asarray(jnp.pi / 3.0, dtype=jnp.float64)

    imposed = static_contact_angle_normal(interface, wall, angle, mask)
    supplement = static_contact_angle_normal(-interface, wall, jnp.pi - angle, mask)
    np.testing.assert_allclose(imposed, -supplement, atol=1e-14)
    np.testing.assert_allclose(
        oe.contract("...d,...d->...", imposed, wall), jnp.cos(angle), atol=1e-14
    )

    phase = jnp.asarray([[0.25]], dtype=jnp.float64)
    gradient = jnp.asarray([[[0.3, -0.2]]], dtype=jnp.float64)
    adjusted = natural_wetting_gradient(phase, gradient, wall, 0.04, 0.2, mask)
    normal_derivative = oe.contract("...d,...d->...", adjusted, wall)
    np.testing.assert_allclose(
        0.2 * normal_derivative + 0.04 * (phase**2 - 1.0), 0.0, atol=1e-14
    )


def test_free_energy_force_stress_and_phase_population_moments_are_explicit():
    lattice = D2Q9()
    precision = LatticeBoltzmannPrecisionPolicy()
    coordinates = jnp.arange(24, dtype=jnp.float64)
    x, y = jnp.meshgrid(coordinates, coordinates, indexing="ij")
    phase = 0.6 * jnp.sin(2.0 * jnp.pi * x / 24.0) * jnp.cos(2.0 * jnp.pi * y / 24.0)
    velocity = jnp.broadcast_to(
        jnp.asarray((0.02, -0.01), dtype=jnp.float64), (*phase.shape, 2)
    )
    closure = BinaryPhaseThermodynamicClosure()
    thermodynamics = PreparedBinaryKineticThermodynamics(
        closure,
        lattice,
        ThermodynamicForceRepresentation.CHEMICAL_POTENTIAL_GRADIENT,
    )
    parameters = BinaryThermodynamicParameters(0.08, 0.12)
    fields = thermodynamics.evaluate(phase, parameters)
    swapped = thermodynamics.evaluate(-phase, parameters)
    equilibrium = phase_field_equilibrium(
        phase, fields.chemical_potential, velocity, lattice, precision
    )
    moments = phase_population_moments(equilibrium, lattice, precision)

    np.testing.assert_allclose(moments.phase, phase, atol=2e-14)
    np.testing.assert_allclose(
        moments.phase_flux, phase[..., None] * velocity, atol=2e-14
    )
    np.testing.assert_allclose(
        fields.chemical_force_density, swapped.chemical_force_density, atol=2e-12
    )
    np.testing.assert_allclose(
        fields.symmetric_stress, swapped.symmetric_stress, atol=2e-12
    )
    np.testing.assert_allclose(
        fields.symmetric_stress,
        jnp.swapaxes(fields.symmetric_stress, -1, -2),
        atol=1e-14,
    )


def test_interfacial_and_free_energy_coefficients_are_differentiable():
    lattice = D2Q9()
    coordinate = jnp.arange(32, dtype=jnp.float64)
    x, _ = jnp.meshgrid(coordinate, coordinate, indexing="ij")
    phase = jnp.tanh((x - 16.0) / 3.0)

    surface_gradient = jax.grad(
        lambda sigma: jnp.sum(
            continuum_surface_force(phase, lattice, sigma).force_density ** 2
        )
    )(jnp.asarray(0.03, dtype=jnp.float64))
    closure = BinaryPhaseThermodynamicClosure()
    thermodynamics = PreparedBinaryKineticThermodynamics(
        closure,
        lattice,
        ThermodynamicForceRepresentation.CHEMICAL_POTENTIAL_GRADIENT,
    )
    bulk_gradient = jax.grad(
        lambda bulk: jnp.sum(
            thermodynamics.evaluate(
                phase, BinaryThermodynamicParameters(bulk, 0.1)
            ).bulk_energy_density
        )
    )(jnp.asarray(0.08, dtype=jnp.float64))

    assert jnp.isfinite(surface_gradient)
    assert jnp.isfinite(bulk_gradient)
    assert bulk_gradient > 0.0
    planar_gradient = isotropic_gradient(phase, lattice)
    assert jnp.all(jnp.isfinite(planar_gradient))


def test_lbm_and_application_double_well_share_one_constitutive_source():
    lattice = D2Q9()
    phase = jnp.linspace(-0.8, 0.8, 16, dtype=jnp.float64).reshape((4, 4))
    bulk = jnp.asarray(0.08)
    kappa = jnp.asarray(0.12)
    closure = BinaryPhaseThermodynamicClosure(DoubleWellFreeEnergy())
    thermodynamics = PreparedBinaryKineticThermodynamics(
        closure,
        lattice,
        ThermodynamicForceRepresentation.CHEMICAL_POTENTIAL_GRADIENT,
    )
    fields = thermodynamics.evaluate(phase, BinaryThermodynamicParameters(bulk, kappa))
    constitutive = DoubleWellFreeEnergy()

    np.testing.assert_allclose(
        fields.bulk_energy_density,
        bulk * constitutive.density(phase),
        atol=1e-14,
    )
    np.testing.assert_allclose(
        fields.chemical_potential + kappa * fields.laplacian,
        bulk * constitutive.derivative(phase),
        atol=1e-14,
    )
