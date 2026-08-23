#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

import phydrax as phx


def test_event_driven_jump_finds_norm_threshold_events():
    problem = phx.solver.amplitude_damping_trajectory_problem(
        3.0, jnp.asarray([0.0j, 1.0 + 0.0j])
    )
    result = phx.solver.solve_event_driven_quantum_jump(
        problem,
        jax.random.PRNGKey(3),
        step_size=0.05,
        steps=20,
        maximum_events=8,
    )
    assert bool(result.valid)
    assert jnp.any(result.events.active)
    assert (
        jnp.max(jnp.where(result.events.active, result.events.root_residuals, 0.0)) < 1e-5
    )


def test_mps_canonicalization_and_tebd_identity():
    state = phx.tensor_network.product_mps(
        jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
    )
    canonical, evidence = phx.tensor_network.canonicalize_mps(state, center=1)
    assert bool(evidence.valid)
    hamiltonian = phx.tensor_network.NearestNeighborHamiltonian(
        (jnp.zeros((4, 4), dtype=complex),),
        (2, 2),
        hamiltonian_id="zero-two-site",
    )
    evolved, tebd = phx.tensor_network.tebd_step(
        canonical,
        hamiltonian,
        0.1,
        maximum_bond_dimension=2,
    )
    assert bool(tebd.valid)
    assert jnp.allclose(evolved.to_dense(), canonical.to_dense())


def test_mps_jump_and_locally_purified_channel():
    state = phx.tensor_network.product_mps(
        jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    )
    hamiltonian = phx.tensor_network.NearestNeighborHamiltonian(
        (jnp.zeros((4, 4), dtype=complex),),
        (2, 2),
        hamiltonian_id="zero",
    )
    lowering = jnp.asarray([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    problem = phx.solver.MPSQuantumJumpProblem(
        hamiltonian,
        (phx.solver.LocalMPSJump(0, lowering, jump_id="loss"),),
        state,
    )
    result = phx.solver.solve_mps_quantum_jump(
        problem,
        jax.random.PRNGKey(4),
        step_size=0.05,
        steps=2,
        maximum_bond_dimension=2,
    )
    assert bool(result.valid)

    purification = phx.tensor_network.LocallyPurifiedDensity(
        (jnp.asarray([[[[0.0]], [[1.0]]]], dtype=complex),)
    )
    gamma = 0.2
    kraus = jnp.asarray(
        [
            [[1.0, 0.0], [0.0, jnp.sqrt(1.0 - gamma)]],
            [[0.0, jnp.sqrt(gamma)], [0.0, 0.0]],
        ],
        dtype=complex,
    )
    channel = phx.solver.LocalKrausChannel(0, kraus, channel_id="amplitude-damping")
    purified = phx.solver.solve_purified_lindblad(
        phx.solver.PurifiedLindbladProblem(purification, (channel,)),
        steps=1,
        maximum_purification_dimension=2,
    )
    assert bool(purified.valid)


def test_heom_continuation_and_nonmarkovian_comparison():
    density = jnp.asarray([[0.6 + 0.0j, 0.0j], [0.0j, 0.4 + 0.0j]])
    expansion = phx.operators.quantum.drude_lorentz_matsubara(0.05, 1.0, 2.0, 1)
    problem = phx.solver.HEOMProblem(
        jnp.zeros((2, 2), dtype=complex),
        jnp.asarray([[1, 0], [0, -1]], dtype=complex),
        expansion,
        phx.solver.HEOMHierarchy(1, 1),
        density,
    )
    continuation = phx.solver.solve_heom_continuation(
        problem, (1, 2), step_size=0.005, steps=1, tolerance=1.0
    )
    assert continuation.stages[-1].valid
    comparison = phx.solver.lorentzian_qubit_comparison(
        density, cutoff=3, heom_depth=1, step_size=0.005, steps=1
    )
    assert bool(comparison.valid)


def test_map_level_nonmarkovian_physicality():
    identity = jnp.eye(4, dtype=complex)
    report = phx.operators.quantum.analyze_dynamical_map_series(
        jnp.stack((identity, identity)), 2
    )
    assert bool(report.valid)
    assert bool(report.cp_valid)
    assert bool(report.cp_divisible)


def test_adaptive_fock_continuation_and_fermionic_gaussian():
    initial_space = phx.operators.quantum.BosonicFockSpace((3,))
    initial = jnp.asarray([1.0 + 0.0j, 0.0j, 0.0j])

    def stage(space, state):
        return state, jnp.asarray([jnp.real(jnp.vdot(state, state))])

    continuation = phx.solver.solve_fock_continuation(
        initial_space,
        initial,
        stage,
        phx.solver.FockContinuationPolicy((5,), (1,), observable_tolerance=2.0),
    )
    assert bool(continuation.converged)

    fermionic = phx.solver.damped_fermionic_mode(0.4, 0.25)
    stationary = fermionic.stationary_state()
    assert bool(stationary.valid)
    result = phx.solver.solve_fermionic_gaussian(fermionic, step_size=0.05, steps=2)
    assert bool(result.valid)


def test_process_causality_and_neural_jump_projection():
    identity = jnp.eye(4, dtype=complex)
    density = jnp.asarray([[0.7 + 0.0j, 0.0j], [0.0j, 0.3 + 0.0j]])
    process = phx.tensor_network.markov_process_tensor((identity,), density)
    causality = phx.tensor_network.validate_process_comb_causality(process)
    assert bool(causality.valid)

    sigma_x = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    operator = phx.solver.StateVectorOperator.from_matrix(sigma_x, operator_id="x")
    problem = phx.solver.NeuralJumpProjectionProblem(
        lambda parameters: jnp.asarray(
            [jnp.cos(parameters[0]), jnp.sin(parameters[0])]
        ).astype(complex),
        jnp.asarray([0.1]),
        operator,
    )
    result = phx.solver.solve_neural_jump_projection(
        problem, learning_rate=0.1, iterations=20
    )
    assert bool(result.valid)
    assert result.infidelity_history[-1] < result.infidelity_history[0]
