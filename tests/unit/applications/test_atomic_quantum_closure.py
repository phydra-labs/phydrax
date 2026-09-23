# Copyright © 2026 PHYDRA, Inc. All rights reserved.

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.atomic_quantum import (
    AtomicManifold,
    AtomicQuantumPlan,
    CoherentDrive,
    electric_dipole_allowed,
    hyperfine_dipole_coefficient,
    LeakageTransition,
    RadiativeTransition,
    rotate_spherical_vector,
    wigner_3j,
    wigner_6j,
)
from phydrax.discretization._temporal import TemporalMesh
from phydrax.solver._finite_cptp import integrate_finite_cptp
from phydrax.solver._lindblad import solve_lindblad
from phydrax.solver._quantum_jump import solve_quantum_jump_ensemble


jax.config.update("jax_enable_x64", True)


def _manifolds():
    ground = AtomicManifold(
        "ground",
        twice_electronic_j=0,
        twice_nuclear_i=0,
        twice_total_f=0,
        parity=1,
        angular_frequency=0.0,
    )
    excited = AtomicManifold(
        "excited",
        twice_electronic_j=2,
        twice_nuclear_i=0,
        twice_total_f=2,
        parity=-1,
        angular_frequency=3.0,
    )
    sink = AtomicManifold(
        "leakage-sink",
        twice_electronic_j=0,
        twice_nuclear_i=0,
        twice_total_f=0,
        parity=1,
        angular_frequency=-1.0,
    )
    return ground, excited, sink


def _density(state):
    return state[:, None] * jnp.conj(state[None, :])


def test_wigner_dipole_selection_and_radiative_branching_normalization():
    ground, excited, sink = _manifolds()
    np.testing.assert_allclose(wigner_3j(0, 2, 2, 0, 0, 0), -1.0 / np.sqrt(3.0))
    np.testing.assert_allclose(wigner_6j(0, 0, 0, 2, 2, 2), 1.0 / np.sqrt(3.0))
    assert electric_dipole_allowed(excited, ground)
    assert not electric_dipole_allowed(sink, ground)
    assert hyperfine_dipole_coefficient(sink, ground, 0, 0, 0) == 0.0

    gamma = 0.7
    leakage = 0.2
    prepared = AtomicQuantumPlan(
        (ground, excited, sink),
        radiative_transitions=(RadiativeTransition("excited", "ground", gamma),),
        leakage_transitions=(LeakageTransition("excited", "leakage-sink", leakage),),
        maximum_channels=8,
    ).prepare()
    active = np.asarray(prepared.active_channels)
    rates = np.asarray(prepared.rates)
    jumps = np.asarray(prepared.jumps)
    ground_row = prepared.basis_index("ground", 0)
    sink_row = prepared.basis_index("leakage-sink", 0)
    for excited_m in excited.magnetic_projections:
        source = prepared.basis_index("excited", excited_m)
        radiative_mask = active & (np.abs(jumps[:, ground_row, source]) > 0.0)
        leakage_mask = active & (np.abs(jumps[:, sink_row, source]) > 0.0)
        np.testing.assert_allclose(np.sum(rates[radiative_mask]), gamma, atol=1e-14)
        np.testing.assert_allclose(np.sum(rates[leakage_mask]), leakage, atol=1e-14)
    assert prepared.evidence.active_channel_count == 6
    assert prepared.evidence.channel_capacity == 8
    assert prepared.jumps.shape == (8, 5, 5)
    assert bool(prepared.evidence.valid)
    np.testing.assert_allclose(
        prepared.evidence.branching_normalization_residual, 0.0, atol=1e-14
    )


def test_forbidden_drive_is_rejected_during_preparation():
    ground, _, sink = _manifolds()
    forbidden = CoherentDrive("ground", "leakage-sink", 1.0, np.asarray((0.0, 1.0, 0.0)))
    with pytest.raises(ValueError, match="selection rules"):
        AtomicQuantumPlan((ground, sink), drives=(forbidden,)).prepare()


def test_coherent_drive_compilation_is_frame_covariant():
    ground, excited, _ = _manifolds()
    polarization = np.asarray((0.3 + 0.2j, 0.5, -0.1 + 0.4j))
    angles = np.asarray((0.31, -0.42, 0.17))
    transformed = np.asarray(rotate_spherical_vector(polarization, angles))
    framed = AtomicQuantumPlan(
        (ground, excited),
        drives=(
            CoherentDrive(
                "ground",
                "excited",
                0.8 - 0.2j,
                polarization,
                frame_euler_angles=angles,
            ),
        ),
        maximum_channels=4,
    ).prepare()
    quantization_frame = AtomicQuantumPlan(
        (ground, excited),
        drives=(CoherentDrive("ground", "excited", 0.8 - 0.2j, transformed),),
        maximum_channels=4,
    ).prepare()

    np.testing.assert_allclose(
        framed.hamiltonian, quantization_frame.hamiltonian, rtol=2e-13, atol=2e-13
    )
    np.testing.assert_allclose(
        framed.evidence.hamiltonian_hermiticity_residual, 0.0, atol=1e-14
    )


def test_compiled_lindblad_and_finite_channel_are_trace_preserving():
    ground, excited, sink = _manifolds()
    prepared = AtomicQuantumPlan(
        (ground, excited, sink),
        radiative_transitions=(RadiativeTransition("excited", "ground", 0.7),),
        leakage_transitions=(LeakageTransition("excited", "leakage-sink", 0.2),),
        maximum_channels=8,
    ).prepare()
    initial_state = prepared.basis_state("excited", 0)
    initial_density = _density(initial_state)
    problem = prepared.density_problem(initial_density)
    np.testing.assert_allclose(
        jnp.trace(problem.generator(initial_density)), 0.0, atol=1e-13
    )

    slicing = TemporalMesh.uniform(0.0, 0.2, 2, role="internal")
    finite = integrate_finite_cptp(prepared.finite_plan(slicing), initial_density)
    assert bool(finite.valid)
    np.testing.assert_allclose(finite.density_trace_residuals, 0.0, atol=2e-12)
    assert float(jnp.max(finite.trace_preservation_residuals)) < 2e-12
    assert float(prepared.evidence.trace_preservation_residual) < 1e-13
    nonphysical = integrate_finite_cptp(
        prepared.finite_plan(slicing),
        jnp.zeros_like(initial_density),
    )
    assert not bool(nonphysical.valid)
    assert jnp.all(nonphysical.density_trace_residuals == 1.0)


def test_density_and_quantum_trajectory_observables_agree():
    ground, excited, _ = _manifolds()
    prepared = AtomicQuantumPlan(
        (ground, excited),
        radiative_transitions=(RadiativeTransition("excited", "ground", 0.7),),
        maximum_channels=4,
    ).prepare()
    state = prepared.basis_state("excited", 0)
    density = _density(state)
    step_size = 0.005
    steps = 100
    dense = solve_lindblad(
        prepared.density_problem(density), step_size=step_size, steps=steps
    )
    trajectories = solve_quantum_jump_ensemble(
        prepared.jump_problem(state),
        jax.random.PRNGKey(173),
        step_size=step_size,
        steps=steps,
        trajectory_count=1024,
    )
    excited_projector = prepared.manifold_projector("excited")
    dense_observable = float(
        prepared.density_expectation(dense.states[-1], excited_projector)
    )
    trajectory_observable, standard_error = trajectories.observable(
        prepared.trajectory_operator(
            excited_projector, operator_id="excited-manifold-population"
        )
    )
    discrepancy = abs(float(trajectory_observable[-1]) - dense_observable)
    assert discrepancy < 5.0 * float(standard_error[-1]) + step_size
    assert int(jnp.sum(trajectories.jump_mask)) > 0
    assert bool(trajectories.valid)
