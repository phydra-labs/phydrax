# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Analytic physical references, not source-wiring assertions.

The application is a declared orthogonal scalar 1D model. These tests do not
qualify atomistic parameters, a foundry process, scattering, or transient NEGF.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import brentq

from phydrax.applications.semiconductor.quantum import (
    bound_states,
    BoundStateOccupation,
    ChainHamiltonian,
    CoherentDevice,
    DensityGradient1D,
    EffectiveMass1D,
    integrate_coherent,
    QuantumPoisson1D,
    QuantumResources,
    scalar_embedding,
    SemiInfiniteLead,
    solve_coherent_poisson,
    solve_schrodinger,
    solve_schrodinger_poisson,
    TransverseModes,
)


Q = 1.602176634e-19
HBAR = 1.054571817e-34
H = 2 * np.pi * HBAR
ME = 9.1093837139e-31
EPS0 = 8.8541878128e-12
REFERENCE = "common declared vacuum alignment"


def _lead(*, onsite=2.0, hopping=-1.0, coupling=-1.0, mu=2.0, temperature=300.0):
    return SemiInfiniteLead(
        onsite * Q,
        hopping * Q,
        coupling * Q,
        mu * Q,
        temperature,
        energy_reference=REFERENCE,
    )


def _chain(n, *, barrier=0.0, left=None, right=None, modes=None):
    h = ChainHamiltonian(
        np.full(n, (2.0 + barrier) * Q),
        np.full(n - 1, -Q),
        np.full(n, 1e-27),
        energy_reference=REFERENCE,
    )
    return CoherentDevice(
        h,
        _lead() if left is None else left,
        _lead() if right is None else right,
        transverse=modes,
    )


def test_square_well_energies_normalization_and_mesh_convergence():
    errors = []
    for n in (39, 79):
        length = 10e-9
        dx = length / (n + 1)
        basis = EffectiveMass1D(
            dx * np.arange(1, n + 1),
            0.0,
            0.19 * ME,
            area=4e-16,
            energy_reference=REFERENCE,
        )
        result = solve_schrodinger(
            basis.hamiltonian(jnp.zeros(n)), 0.04 * Q, 100.0, count=3
        )
        expected = HBAR**2 * np.pi**2 * np.arange(1, 4) ** 2 / (2 * 0.19 * ME * length**2)
        errors.append(np.max(np.abs(np.asarray(result.energies) / expected - 1)))
        assert bool(result.successful)
        np.testing.assert_allclose(
            result.vectors.T @ result.vectors, np.eye(3), atol=2e-8
        )
        np.testing.assert_allclose(
            jnp.sum(result.electron_density * basis.base_hamiltonian.cell_volumes),
            jnp.sum(result.occupations),
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            result.charge_density, -Q * result.electron_density, rtol=1e-14
        )
    assert errors[1] < 0.27 * errors[0]
    assert errors[1] < 0.0013


def test_mass_step_matches_continuous_wavefunction_and_inverse_mass_flux():
    length, m1, m2 = 10e-9, 0.19 * ME, 0.38 * ME
    light = HBAR**2 * np.pi**2 / (2 * m1 * length**2) / Q
    heavy = HBAR**2 * np.pi**2 / (2 * m2 * length**2) / Q

    def matching(energy_ev):
        k1 = np.sqrt(2 * m1 * energy_ev * Q) / HBAR
        k2 = np.sqrt(2 * m2 * energy_ev * Q) / HBAR
        return (
            k1 * m2 * np.cos(k1 * length / 2) * np.sin(k2 * length / 2)
            + k2 * m1 * np.sin(k1 * length / 2) * np.cos(k2 * length / 2)
        ) / (m1 / length)

    expected = brentq(matching, heavy, light)
    n = 80
    x = np.arange(1, n + 1) * length / (n + 1)
    basis = EffectiveMass1D(
        x, 0.0, np.where(x < length / 2, m1, m2), area=1e-16, energy_reference=REFERENCE
    )
    result = solve_schrodinger(basis.base_hamiltonian, -0.1 * Q, 50.0, count=2)
    assert bool(result.successful)
    np.testing.assert_allclose(result.energies[0] / Q, expected, rtol=1e-3)


def test_triangular_well_matches_airy_ground_state():
    field, mass = 1e7, 0.19 * ME
    x = np.arange(1, 160) * 0.25e-9
    basis = EffectiveMass1D(
        x, Q * field * x, mass, area=1e-16, energy_reference=REFERENCE
    )
    result = solve_schrodinger(basis.base_hamiltonian, 0.0, 50.0, count=2)
    expected = (
        2.338107410459767 * (HBAR**2 / (2 * mass)) ** (1 / 3) * (Q * field) ** (2 / 3)
    )
    assert bool(result.successful)
    np.testing.assert_allclose(result.energies[0], expected, rtol=1e-3)


def test_occupation_guard_rejects_insufficient_subbands_and_gauge_is_invariant():
    basis = EffectiveMass1D(
        np.arange(1, 20) * 0.5e-9, 0.0, 0.19 * ME, area=2e-16, energy_reference=REFERENCE
    )
    h = basis.base_hamiltonian
    incomplete = solve_schrodinger(h, 0.5 * Q, 300.0, count=1)
    assert not bool(incomplete.successful)
    assert float(incomplete.omitted_particle_bound) > 1
    complete = solve_schrodinger(h, 0.02 * Q, 50.0, count=4)
    shifted = solve_schrodinger(h.shifted(0.73 * Q), 0.75 * Q, 50.0, count=4)
    assert bool(complete.successful & shifted.successful)
    np.testing.assert_allclose(
        shifted.electron_density, complete.electron_density, rtol=2e-7
    )
    np.testing.assert_allclose(shifted.energies - complete.energies, 0.73 * Q, rtol=1e-8)


def test_transverse_degeneracy_counts_particles_once():
    basis = EffectiveMass1D(
        np.arange(1, 12) * 1e-9, 0.0, 0.19 * ME, area=1e-14, energy_reference=REFERENCE
    )
    h = basis.base_hamiltonian
    one = solve_schrodinger(
        h, 0.03 * Q, 100.0, count=5, transverse=TransverseModes((0.0,), (1.0,))
    )
    two = solve_schrodinger(
        h, 0.03 * Q, 100.0, count=5, transverse=TransverseModes((0.0, 0.0), (1.0, 1.0))
    )
    np.testing.assert_allclose(two.electron_density, 2 * one.electron_density, rtol=1e-13)
    np.testing.assert_allclose(two.energy_density, 2 * one.energy_density, rtol=1e-13)


def test_density_gradient_matches_the_same_discrete_square_well_ground_mode():
    count, length, mass, area = 63, 10e-9, 0.19 * ME, 4e-16
    spacing = length / (count + 1)
    positions = spacing * np.arange(1, count + 1)
    density = 1e24 * jnp.sin(jnp.pi * positions / length) ** 2
    closure = DensityGradient1D(
        positions,
        mass,
        area=area,
        energy_reference=REFERENCE,
        provenance="Analytic discrete square-well density-gradient comparison",
    )
    result = closure.evaluate(density)
    expected = HBAR**2 / (mass * spacing**2) * (1 - np.cos(np.pi / (count + 1)))
    assert bool(result.successful)
    np.testing.assert_allclose(result.quantum_potential, expected, rtol=2e-9)
    np.testing.assert_allclose(
        result.total_energy,
        expected * jnp.sum(closure.cell_volumes * density),
        rtol=2e-9,
    )
    basis = EffectiveMass1D(positions, 0.0, mass, area=area, energy_reference=REFERENCE)
    confined = solve_schrodinger(basis.base_hamiltonian, -0.1 * Q, 50.0, count=1)
    np.testing.assert_allclose(confined.energies[0], expected, rtol=2e-9)


def test_semi_infinite_surface_satisfies_retarded_recursion_and_overlap_embedding():
    lead = _lead(onsite=0.0, hopping=-0.7, coupling=-0.2)
    energies = jnp.asarray([-2.0, -0.3, 0.0, 0.8, 2.0]) * Q
    eta = 1e-5 * Q
    g = lead.surface_green(energies, eta=eta)
    z = energies + 1j * eta
    np.testing.assert_allclose(
        (z - lead.onsite - lead.hopping**2 * g) * g, 1.0, rtol=2e-12, atol=2e-12
    )
    assert np.all(np.imag(g) < 0)
    assert np.all(np.asarray(lead.broadening(energies, eta=eta)) > 0)
    overlap = 0.12 + 0.03j
    coupling = (0.2 + 0.04j) * Q
    sigma = scalar_embedding(z, g, coupling, overlap)
    expected = (z * overlap - coupling) * g * (z * np.conj(overlap) - np.conj(coupling))
    np.testing.assert_allclose(sigma, expected, rtol=1e-14)


def test_transparent_channel_is_unit_transmission_with_local_current_conservation():
    device = _chain(9, left=_lead(mu=2.1), right=_lead(mu=1.9))
    for energy in (0.3 * Q, 2.0 * Q, 3.4 * Q):
        point = device.spectral(energy)
        assert bool(point.successful)
        np.testing.assert_allclose(point.transmission, 1.0, rtol=2e-11)
        np.testing.assert_allclose(
            jnp.sum(point.particle_current_kernel), 0.0, atol=2e-11
        )
        np.testing.assert_allclose(
            point.bond_particle_current_kernel,
            point.particle_current_kernel[0],
            atol=2e-11,
        )
    equilibrium = _chain(5).spectral(1.5 * Q)
    np.testing.assert_allclose(equilibrium.particle_current_kernel, 0.0, atol=2e-12)


def test_single_site_barrier_matches_discrete_scattering_and_thickness_suppresses_tunneling():
    energy, barrier = 0.7, 1.2
    point = _chain(1, barrier=barrier).spectral(energy * Q)
    cosine = (2 - energy) / 2
    expected = 4 * (1 - cosine * cosine) / (barrier**2 + 4 * (1 - cosine * cosine))
    np.testing.assert_allclose(point.transmission, expected, rtol=2e-12)
    thin = _chain(4, barrier=barrier).spectral(energy * Q)
    thick = _chain(8, barrier=barrier).spectral(energy * Q)
    assert bool(thin.successful & thick.successful)
    assert 0 < float(thick.transmission) < 0.05 * float(thin.transmission)


def test_resonant_level_matches_exact_energy_dependent_lead_formula_and_derivative():
    left = _lead(onsite=0.0, coupling=-0.2, mu=0.1)
    right = _lead(onsite=0.0, coupling=-0.2, mu=-0.1)
    h = ChainHamiltonian(
        jnp.asarray([0.0]), jnp.zeros(0), jnp.asarray([1e-27]), energy_reference=REFERENCE
    )
    device = CoherentDevice(h, left, right)
    np.testing.assert_allclose(device.spectral(0.0).transmission, 1.0, rtol=1e-12)
    energy = 0.12 * Q
    sigma_one = 0.04 * (energy - 1j * jnp.sqrt(4 * Q * Q - energy * energy)) / 2
    gamma = -2 * jnp.imag(sigma_one)
    expected = gamma**2 / jnp.abs(energy - 2 * sigma_one) ** 2
    np.testing.assert_allclose(device.spectral(energy).transmission, expected, rtol=1e-11)

    def transmission(level_ev):
        shifted = eqx.tree_at(
            lambda d: d.hamiltonian.diagonal, device, jnp.asarray([level_ev * Q])
        )
        return shifted.spectral(energy).transmission

    derivative = jax.grad(transmission)(0.04)
    finite_difference = (transmission(0.040001) - transmission(0.039999)) / 0.000002
    np.testing.assert_allclose(derivative, finite_difference, rtol=3e-6)


def test_actual_bound_pole_requires_preparation_and_includes_lead_tail_norm():
    left = _lead(onsite=0.0, coupling=-0.4, mu=4.0)
    h = ChainHamiltonian(
        jnp.asarray([3.0 * Q]),
        jnp.zeros(0),
        jnp.asarray([1e-27]),
        energy_reference=REFERENCE,
    )
    device = CoherentDevice(h, left, left)
    with pytest.raises(ValueError, match="occupation undetermined"):
        bound_states(device)
    preparation = BoundStateOccupation.equilibrium(left, left)
    states = bound_states(device, preparation)
    a, level = 0.16, 3.0
    expected = (
        level * (1 - a)
        - np.sqrt(level**2 * (1 - a) ** 2 - (1 - 2 * a) * (level**2 + 4 * a * a))
    ) / (1 - 2 * a)
    expected_weight = 1 / (1 - a * (1 - expected / np.sqrt(expected * expected - 4)))
    assert bool(states.successful)
    np.testing.assert_allclose(states.energies / Q, [expected], rtol=1e-8)
    np.testing.assert_allclose(states.device_weights, [expected_weight], rtol=1e-8)
    np.testing.assert_allclose(states.electron_counts, [2 * expected_weight], rtol=1e-8)
    assert float(states.device_weights[0]) < 1


def test_adaptive_open_channel_conductance_and_spectral_sum_rule():
    bias = 1e-3
    device = _chain(
        3,
        left=_lead(mu=2 + bias / 2, temperature=30.0),
        right=_lead(mu=2 - bias / 2, temperature=30.0),
    )
    result = integrate_coherent(device, tolerance=2e-7, spectral_tolerance=1e-4)
    assert bool(result.successful)
    np.testing.assert_allclose(
        result.terminal_currents, [-2 * Q * Q / H * bias, 2 * Q * Q / H * bias], rtol=3e-4
    )
    np.testing.assert_allclose(jnp.sum(result.terminal_currents), 0.0, atol=1e-15)
    assert float(result.evidence.spectral_sum_error) < 1e-4
    assert float(result.evidence.refinement_error) < 8e-7
    shifted = integrate_coherent(
        device.shifted(0.43 * Q), tolerance=2e-7, spectral_tolerance=1e-4
    )
    assert bool(shifted.successful)
    np.testing.assert_allclose(
        shifted.electron_density, result.electron_density, rtol=2e-6
    )
    np.testing.assert_allclose(
        shifted.terminal_currents, result.terminal_currents, rtol=2e-5
    )
    np.testing.assert_allclose(
        shifted.heat_currents, result.heat_currents, rtol=3e-4, atol=1e-15
    )


def test_bound_plus_continuum_closes_spectral_weight_and_equilibrium_charge():
    left = _lead(onsite=0.0, coupling=-0.4, mu=4.0)
    device = CoherentDevice(
        ChainHamiltonian([3.0 * Q], [], [1e-27], energy_reference=REFERENCE), left, left
    )
    result = integrate_coherent(
        device,
        bound_occupation=BoundStateOccupation.equilibrium(left, left),
        tolerance=1e-7,
        spectral_tolerance=1e-4,
    )
    assert bool(result.successful)
    np.testing.assert_allclose(
        result.electron_density * device.hamiltonian.cell_volumes, [2.0], rtol=1e-4
    )
    np.testing.assert_allclose(result.terminal_currents, 0.0, atol=1e-15)


def test_schrodinger_poisson_closes_charge_with_nonzero_quantum_feedback():
    basis = EffectiveMass1D(
        np.arange(1, 12) * 1e-9, 0.0, 0.19 * ME, area=1e-14, energy_reference=REFERENCE
    )
    poisson = QuantumPoisson1D(basis, 11.7 * EPS0, 0.0, [0.0, 0.0])
    result = solve_schrodinger_poisson(
        basis,
        poisson,
        0.03 * Q,
        100.0,
        count=5,
        potential_tolerance=2e-8,
        maximum_steps=60,
        damping=0.7,
    )
    assert bool(result.successful)
    assert np.min(np.asarray(result.potential)) < -1e-5
    assert np.all(np.asarray(result.potential) < 0)
    np.testing.assert_allclose(result.potential, result.potential[::-1], atol=2e-8)
    bulk = jnp.sum(result.quantum.charge_density * poisson.cell_volumes)
    np.testing.assert_allclose(jnp.sum(result.terminal_charges), -bulk, atol=2e-22)


def test_coherent_poisson_half_filled_uniform_channel_is_neutral():
    basis = EffectiveMass1D(
        np.arange(1, 6) * 1e-9, 0.0, 0.19 * ME, area=1e-14, energy_reference=REFERENCE
    )
    h = basis.base_hamiltonian
    hopping = float(h.off_diagonal[0])
    onsite = float(h.diagonal[0])
    lead = SemiInfiniteLead(
        onsite, hopping, hopping, onsite, 300.0, energy_reference=REFERENCE
    )
    device = CoherentDevice(h, lead, lead)
    poisson = QuantumPoisson1D(basis, 11.7 * EPS0, Q / h.cell_volumes, [0.0, 0.0])
    result = solve_coherent_poisson(
        device,
        poisson,
        bound_occupation=BoundStateOccupation.equilibrium(lead, lead),
        potential_tolerance=2e-7,
        maximum_steps=8,
        integration_options={"tolerance": 1e-7, "spectral_tolerance": 1e-4},
    )
    assert bool(result.successful)
    np.testing.assert_allclose(
        result.quantum.electron_density * h.cell_volumes, 1.0, rtol=1e-4
    )
    np.testing.assert_allclose(result.potential, 0.0, atol=2e-7)
    np.testing.assert_allclose(result.quantum.terminal_currents, 0.0, atol=1e-15)


def test_structural_admission_rejects_disconnected_or_nonuniform_models():
    with pytest.raises(ValueError, match="Disconnected"):
        ChainHamiltonian([0.0, 0.0], [0.0], [1e-27, 1e-27], energy_reference=REFERENCE)
    with pytest.raises(ValueError, match="uniform"):
        EffectiveMass1D(
            [0.0, 1e-9, 3e-9], 0.0, ME, area=1e-18, energy_reference=REFERENCE
        )
    with pytest.raises(ValueError, match="max_nodes"):
        ChainHamiltonian(
            [0.0, 0.0],
            [-Q],
            [1e-27, 1e-27],
            energy_reference=REFERENCE,
            resources=QuantumResources(max_nodes=1),
        )
