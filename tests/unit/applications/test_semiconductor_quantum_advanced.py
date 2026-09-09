# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Bounded model-level quantum regressions; no atomistic or foundry claim."""

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications import semiconductor as sc
from phydrax.applications.semiconductor import quantum as sq
from phydrax.meshing import MeshZone, MeshZoneRole


jax.config.update("jax_enable_x64", True)

Q = sc.ELEMENTARY_CHARGE_SI
HBAR = 1.054571817e-34
REFERENCE = "advanced quantum synthetic datum"


def _resources():
    return sq.QuantumResources(
        max_nodes=512,
        max_evaluations=20_000,
        max_intervals=256,
        workspace_bytes=512 * 1024 * 1024,
    )


def _lead(*, onsite=2.0, mu=2.0):
    return sq.SemiInfiniteLead(
        onsite * Q,
        -Q,
        -Q,
        mu * Q,
        300.0,
        energy_reference=REFERENCE,
    )


def _device(*, onsite=2.0, mu=2.0):
    hamiltonian = sq.ChainHamiltonian(
        jnp.asarray([onsite * Q]),
        jnp.zeros((0,)),
        jnp.asarray([1e-27]),
        energy_reference=REFERENCE,
        resources=_resources(),
    )
    return sq.CoherentDevice(
        hamiltonian,
        _lead(onsite=onsite, mu=mu),
        _lead(onsite=onsite, mu=mu),
        transverse=sq.TransverseModes((0.0,), (1.0,)),
    )


def test_optical_phonon_scba_closes_kms_particle_and_energy_ledgers():
    device = _device()
    bath = sq.OpticalPhononBath(
        0.5 * Q, 0.03 * Q, 300.0, bath_id="synthetic weak optical phonon"
    )
    grid = sq.PhononEnergyGrid(0.0, 4 * Q, points=16, phonon_energy=bath.energy)
    result = sq.solve_phonon_transport(
        device,
        bath,
        grid,
        tolerance=2e-6,
        observable_tolerance=0.2,
        maximum_steps=100,
        damping=0.4,
    )
    assert bool(result.successful)
    assert float(result.evidence.fixed_point_error) < 2e-6
    assert float(result.evidence.equilibrium_kms_error) < 1e-12
    assert float(result.evidence.collision_particle_error) < 1e-12
    assert float(result.evidence.terminal_energy_error) < 1e-12
    np.testing.assert_allclose(
        jnp.sum(result.energy_currents), jnp.sum(result.phonon_heat), atol=2e-18
    )
    np.testing.assert_allclose(jnp.sum(result.terminal_currents), 0.0, atol=2e-18)


def test_unitary_lead_dilation_is_causal_conservative_and_refinable():
    device = _device()
    times = jnp.asarray([0.0, 1e-17, 2e-17])
    pulse = sq.QuantumPulse(times, jnp.zeros((2, 1)), jnp.zeros((2, 2)))
    result = sq.solve_quantum_transient(
        device,
        sq.QuantumInitialState(preparation="equilibrium"),
        pulse,
        lead_sites=8,
        tolerance=0.1,
    )
    assert bool(result.successful)
    assert bool(result.evidence.recurrence_valid)
    assert float(result.evidence.kernel_error) < 1e-12
    assert float(result.evidence.number_error) < 1e-10
    assert float(result.evidence.energy_error) < 1e-10
    np.testing.assert_allclose(result.terminal_currents, 0.0, atol=3e-19)
    lags = jnp.asarray([-1e-17, 0.0, 1e-17])
    kernel = sq.lead_memory_kernel(device.left, lags, sites=16)
    np.testing.assert_array_equal(kernel[0], 0.0j)
    assert np.isfinite(np.asarray(kernel[1:])).all()


def test_coherent_noise_and_screened_ac_report_independent_physical_gates():
    device = _device()
    noise = sq.coherent_low_frequency_noise(device, tolerance=2e-5)
    assert bool(noise.successful)
    assert float(noise.fluctuation_dissipation_error) < 1e-12
    np.testing.assert_allclose(jnp.sum(noise.spectrum, axis=0), 0.0, atol=1e-35)
    capacitance = sq.QuantumCapacitance(jnp.asarray([[1e-18, 1e-18, 2e-18]]))
    response = sq.finite_frequency_quantum_response(
        device,
        capacitance,
        1e13,
        adiabatic_rate=7e13,
        lead_sites=96,
        tolerance=0.18,
    )
    assert bool(response.successful)
    assert float(response.evidence.gauge_error) < 1e-10
    assert float(response.evidence.kcl_error) < 1e-10
    assert float(response.evidence.ward_error) < 1e-10
    assert float(response.evidence.lead_refinement_error) < 0.18
    assert float(response.evidence.adiabatic_refinement_error) < 0.18
    assert float(response.evidence.recurrence_tail_bound) < 0.18


def test_stationary_hybrid_interface_matches_one_disjoint_reservoir():
    bands = sc.BandThermodynamics(
        "hybrid-classical-bands",
        conduction_band_edge=0.56 * Q,
        valence_band_edge=-0.56 * Q,
        conduction_density_of_states=2.8e25,
        valence_density_of_states=1.04e25,
        reference_temperature=300.0,
        temperature_range=(250.0, 400.0),
        energy_reference=REFERENCE,
        provenance="Synthetic aligned classical bands",
    )
    material = sc.SemiconductorMaterial(
        "hybrid-classical-material",
        permittivity=11.7 * sc.VACUUM_PERMITTIVITY_SI,
        thermodynamics=bands,
        electron_mobility=0.1,
        hole_mobility=0.04,
        provenance="Synthetic classical hybrid segment",
    )
    support = sc.TransportSupport.interval(np.linspace(0.0, 1e-7, 5), area=1e-14)
    zone = MeshZone("classical", MeshZoneRole.MATERIAL, support.node_scope())
    classical = sc.PreparedSemiconductorDevice(
        sc.DevicePlan(
            support,
            materials=(sc.MaterialBinding(zone, material),),
            donor_density=1e20,
            contacts=(
                sc.OhmicContact(
                    "external", support.boundary_patch("external", side="lower")
                ),
                sc.OhmicContact("join", support.boundary_patch("join", side="upper")),
            ),
        )
    )
    interface = sq.QuantumClassicalInterface(
        "join",
        "left",
        classical_region_id="classical-segment",
        quantum_region_id="quantum-segment",
        voltage_bounds=(-0.01, 0.01),
        current_tolerance=1e-10,
        heat_tolerance=1e-10,
        energy_reference=REFERENCE,
        provenance="Synthetic common-reservoir matching experiment",
        maximum_steps=8,
    )
    result = sq.solve_quantum_classical_interface(
        classical,
        _device(onsite=0.0, mu=0.0),
        interface,
        jnp.zeros((2,)),
        quantum_tolerance=2e-4,
        spectral_tolerance=5e-3,
        initial_panels=8,
        max_refinements=2,
    )
    assert bool(result.successful)
    assert bool(result.evidence.bracketed)
    assert len(result.attempts) == 3
    np.testing.assert_allclose(result.interface_voltage, 0.0, atol=1e-14)
    assert float(result.evidence.current_error) < 1e-10
    assert result.evidence.classical_region_id != result.evidence.quantum_region_id
