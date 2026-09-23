#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx
import phydrax.solver._gaussian_lindblad as gaussian_lindblad


def test_gaussian_bosonic_lindblad_reaches_thermal_state():
    problem = phx.solver.damped_thermal_oscillator(0.4, 1.0)
    stationary = problem.stationary_state()
    assert bool(stationary.valid)
    assert jnp.allclose(stationary.covariance, 1.5 * jnp.eye(2))
    solution = phx.solver.solve_gaussian_lindblad(problem, step_size=0.05, steps=10)
    assert bool(solution.valid)
    assert solution.covariances[-1, 0, 0] > solution.covariances[0, 0, 0]


def test_stationary_gaussian_rejects_failed_mean_solve(monkeypatch):
    problem = phx.solver.damped_thermal_oscillator(0.4, 1.0)
    solve_linear = gaussian_lindblad.solve_linear
    calls = 0

    def fail_mean(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = solve_linear(*args, **kwargs)
        if calls == 2:
            result = eqx.tree_at(
                lambda value: value.status,
                result,
                jnp.ones_like(result.status),
            )
        return result

    monkeypatch.setattr(gaussian_lindblad, "solve_linear", fail_mean)

    with pytest.raises(ValueError, match="certification failed"):
        problem.stationary_state()


def test_quantum_jump_ensemble_replays_and_decays():
    problem = phx.solver.amplitude_damping_trajectory_problem(
        0.5, jnp.asarray([0.0j, 1.0 + 0.0j])
    )
    first = phx.solver.solve_quantum_jump_ensemble(
        problem,
        jax.random.PRNGKey(2),
        step_size=0.02,
        steps=20,
        trajectory_count=32,
    )
    second = phx.solver.solve_quantum_jump_ensemble(
        problem,
        jax.random.PRNGKey(2),
        step_size=0.02,
        steps=20,
        trajectory_count=32,
    )
    assert bool(first.valid)
    assert jnp.array_equal(first.states, second.states)
    assert jnp.sum(first.jump_mask) > 0


def test_fock_ladder_cutoff_and_embedding_are_explicit():
    coarse = phx.operators.quantum.BosonicFockSpace((3,))
    fine = phx.operators.quantum.BosonicFockSpace((5,))
    state = jnp.asarray([0.0j, 0.0j, 1.0 + 0.0j])
    annihilated = coarse.annihilate(state, 0)
    assert jnp.allclose(annihilated, jnp.asarray([0.0j, jnp.sqrt(2.0), 0.0j]))
    evidence = coarse.cutoff_evidence(state)
    assert jnp.allclose(evidence.top_level_probability, 1.0)
    embedded = coarse.embed(state, fine)
    assert embedded.shape == (5,)
    assert jnp.allclose(embedded[:3], state)


def test_pseudomode_reduction_and_bath_expansion():
    expansion, mode, mapping = phx.operators.quantum.lorentzian_pseudomode(
        1.0, 0.5, 0.2, cutoff=3
    )
    assert bool(expansion.valid)
    assert mapping.coupling == 0.2
    initial = jnp.asarray([[0.01 + 0.0j, 0.0j], [0.0j, 0.99 + 0.0j]])
    problem = phx.solver.jaynes_cummings_pseudomode_problem(mode, initial)
    solution = phx.solver.solve_pseudomode(problem, step_size=0.02, steps=2)
    assert bool(solution.valid)
    assert solution.reduced_states.shape == (3, 2, 2)


def test_heom_topology_and_root_state():
    initial = jnp.asarray([[0.6 + 0.0j, 0.0j], [0.0j, 0.4 + 0.0j]])
    problem = phx.solver.thermal_drude_lorentz_qubit_heom(
        0.05, 1.0, 2.0, initial, depth=1
    )
    assert problem.hierarchy.auxiliary_count == 2
    solution = phx.solver.solve_heom(problem, step_size=0.01, steps=2)
    assert bool(solution.valid)
    assert solution.root_states.shape == (3, 2, 2)


def test_memory_kernel_and_dynamical_map_physicality():
    initial = jnp.asarray([[0.6 + 0.0j, 0.0j], [0.0j, 0.4 + 0.0j]])
    problem = phx.solver.exponential_memory_qubit_problem(0.05, 1.0, initial)
    solution = phx.solver.solve_memory_kernel(problem, step_size=0.01, steps=2)
    assert bool(solution.valid)
    identity = jnp.eye(4, dtype="complex128")
    report = phx.solver.DynamicalMapPhysicality(identity, 2)
    assert bool(report.valid)


def test_tensor_network_purification_and_gate_truncation():
    state = phx.tensor_network.product_mps(
        jnp.asarray([[1.0, 0.0], [1.0, 0.0]], dtype="complex128")
    )
    assert jnp.allclose(state.norm(), 1.0)
    swap = jnp.eye(4, dtype="complex128").reshape((2, 2, 2, 2))
    evolved, evidence = phx.tensor_network.apply_two_site_gate(
        state, 0, swap, maximum_bond_dimension=2
    )
    assert bool(evidence.valid)
    assert jnp.allclose(evolved.norm(), 1.0)
    purification = phx.tensor_network.LocallyPurifiedDensity(
        (jnp.asarray([[[[1.0]], [[0.0]]]], dtype="complex128"),)
    )
    assert jnp.allclose(jnp.trace(purification.to_dense_density(normalize=True)), 1.0)


def test_markov_process_tensor_contracts_identity_interventions():
    identity = jnp.eye(4, dtype="complex128")
    initial = jnp.asarray([[0.7 + 0.0j, 0.0j], [0.0j, 0.3 + 0.0j]])
    process = phx.tensor_network.markov_process_tensor((identity, identity), initial)
    result = process.contract()
    assert result.valid
    assert int(result.status) == int(
        phx.tensor_network.ProcessTensorContractionStatus.SUCCESS
    )
    assert jnp.allclose(result.final_state, initial)
    assert jnp.allclose(result.probability, 1.0)
    assert bool(process.physicality().valid)
